"""Upcoming (future) matches: fixtures, current-season results, and prematch forecasts.

Forecasting a future match = append it to the historical frame as a row WITHOUT a result,
with forecast_time = now, and run the same walk-forward models. The driver releases only
results whose available_at <= now, so the look-ahead guarantees of the validated pipeline
carry over unchanged.

Freshness gate (never forecast from silently outdated ratings):
  * no results from the current OR previous season in the data  -> INSUFFICIENT_DATA
  * current season started > 21 days ago but none of its results are loaded -> STALE_DATA
    (probabilities shown, explicitly flagged; load current results to fix)
Every network call is read-only. Set SPORTSLAB_OFFLINE=1 to disable all network access.
"""
from __future__ import annotations

import io
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from sportslab.config import ARTIFACTS_DIR, GLOBAL_SEED
from sportslab.data.fetch import http_get, register_current
from sportslab.data.teams import canonical, match_known

UTC = timezone.utc
LONDON = ZoneInfo("Europe/London")
OK, STALE, INSUFFICIENT = "OK", "STALE_DATA", "INSUFFICIENT_DATA"
AVAILABLE, UNAVAILABLE = "AVAILABLE", "UNAVAILABLE"
FD_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
SEASON_START_MONTH = {"football": 8, "hockey": 10, "basketball": 10}
COMPETITION = {"football": "EPL", "hockey": "NHL", "basketball": "NBA", "tennis": "ATP"}
FD_BOOKS = {"PS": "pinnacle", "B365": "bet365", "Avg": "market_avg", "Max": "market_max"}


def offline() -> bool:
    return os.environ.get("SPORTSLAB_OFFLINE", "") not in ("", "0")


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("min")


# ------------------------------------------------------------------ seasons
def season_label(sport: str, ts) -> str:
    ts = pd.Timestamp(ts)
    if sport == "tennis":
        return str(ts.year)
    y = ts.year if ts.month >= SEASON_START_MONTH[sport] else ts.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def previous_season(sport: str, label: str) -> str:
    if sport == "tennis":
        return str(int(label) - 1)
    y = int(label[:4]) - 1
    return f"{y}-{(y + 1) % 100:02d}"


def season_start(sport: str, label: str) -> pd.Timestamp:
    if sport == "tennis":
        return pd.Timestamp(f"{label}-01-01", tz="UTC")
    return pd.Timestamp(f"{label[:4]}-{SEASON_START_MONTH[sport]:02d}-01", tz="UTC")


@dataclass
class SourceStatus:
    source: str
    status: str
    detail: str = ""
    retrieved_at: str | None = None
    url: str | None = None
    n: int = 0


@dataclass
class Fixture:
    sport: str
    home: str
    away: str
    kickoff: pd.Timestamp | None
    source: str
    raw_home: str = ""
    raw_away: str = ""
    name_match: str = ""
    odds: dict = field(default_factory=dict)        # odds_<book>_opening_<sel> -> price (OBSERVED)
    odds_retrieved_at: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        k = self.kickoff.strftime("%Y%m%d%H%M") if self.kickoff is not None else "nodate"
        return f"UP-{self.sport}-{k}-{self.home}-{self.away}".replace(" ", "")

    @property
    def label(self) -> str:
        when = self.kickoff.tz_convert("Europe/London" if self.sport == "football" else "UTC").strftime(
            "%a %d %b %H:%M") if self.kickoff is not None else "date ?"
        return f"{when}  {self.home} vs {self.away}"


# ------------------------------------------------------------------ current-season results
def refresh_football_current(now: pd.Timestamp | None = None) -> SourceStatus:
    """Download the current EPL season results+odds file from football-data.co.uk (same format as history)."""
    now = now or now_utc()
    season = season_label("football", now)
    code = f"{season[2:4]}{season[5:7]}"
    url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
    if offline():
        return SourceStatus("football-data.co.uk current season", UNAVAILABLE, "offline mode", url=url)
    data, err = http_get(url)
    if err:
        return SourceStatus("football-data.co.uk current season", UNAVAILABLE, err, url=url)
    try:
        df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
        assert {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"} <= set(df.columns)
    except Exception as e:  # noqa: BLE001
        return SourceStatus("football-data.co.uk current season", UNAVAILABLE, f"unexpected file format: {e}", url=url)
    rec = register_current(f"football_epl_{code}_current", sport="football", path=f"football/current/E0_{code}.csv",
                           data=data, url=url, original_provider="football-data.co.uk (E0.csv, current season)",
                           season=season, role="current-results")
    return SourceStatus("football-data.co.uk current season", AVAILABLE,
                        f"{len(df)} matches of {season}", rec["retrieved_at"], url, len(df))


def _final_snapshots(sport: str, day: str):
    from sportslab.live.sources import SportsSkillsLive
    r = SportsSkillsLive(sport).fetch(date=day)
    return r, [s for s in r.snapshots if s.state == "final" and s.home_score is not None]


def refresh_espn_results(sport: str, since: str, until: str | None = None, progress=None) -> SourceStatus:
    """NHL/NBA final results via the read-only sports-skills scoreboard, one call per date.
    Stored in the historical file formats so the standard ingest/validation code applies."""
    assert sport in ("hockey", "basketball")
    until = until or (now_utc() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    days = pd.date_range(since, until, freq="D")
    src = f"sports-skills/{'nhl' if sport == 'hockey' else 'nba'}.get_scoreboard (ESPN)"
    if offline():
        return SourceStatus(src, UNAVAILABLE, "offline mode")
    rows, errors = [], 0
    for i, d in enumerate(days):
        day = d.strftime("%Y-%m-%d")
        r, finals = _final_snapshots(sport, day)
        if r.status != AVAILABLE:
            errors += 1
            if errors >= 5 and not rows:
                return SourceStatus(src, UNAVAILABLE, f"{errors} failed calls, e.g. {r.error}")
        for s in finals:
            detail = (s.stats or {}).get("status_detail", "") or ""
            rows.append(dict(date=day, home=(s.stats or {}).get("home"), away=(s.stats or {}).get("away"),
                             hs=int(s.home_score), as_=int(s.away_score),
                             otso="SO" if "SO" in detail.upper() else ("OT" if "OT" in detail.upper() else "")))
        if progress:
            progress((i + 1) / len(days), f"{day}: {len(finals)} final games")
    if not rows:
        return SourceStatus(src, UNAVAILABLE, f"no final games returned for {since}..{until} ({errors} failed calls)")
    df = pd.DataFrame(rows).dropna(subset=["home", "away"]).drop_duplicates(["date", "home", "away"])
    tag = f"{since.replace('-', '')}_{until.replace('-', '')}"
    if sport == "hockey":
        out = pd.DataFrame({"Date": df.date, "Visitor": df.away, "G": df.as_, "Home": df.home, "G.1": df.hs,
                            "": df.otso})
        sid, path = "hockey_nhl_espn_recent", f"hockey/current/nhl_espn_{tag}.csv"
    else:
        dt = pd.to_datetime(df.date)
        end_year = np.where(dt.dt.month >= SEASON_START_MONTH["basketball"], dt.dt.year + 1, dt.dt.year)
        out = pd.DataFrame({"season": end_year, "date": df.date, "team1": df.home, "team2": df.away,
                            "prob1": np.nan, "prob1_outcome": (df.hs > df.as_).astype(int), "prob2": np.nan,
                            "prob2_outcome": (df.hs < df.as_).astype(int)})
        sid, path = "basketball_nba_espn_recent", f"basketball/current/nba_espn_{tag}.csv"
    rec = register_current(sid, sport=sport, path=path, data=out.to_csv(index=False).encode(),
                           url="ESPN public scoreboard via sports-skills (read-only)",
                           original_provider="ESPN (via machina-sports/sports-skills)",
                           season=f"{since}..{until}", role="current-results",
                           notes=f"{len(days)} daily scoreboard calls, {errors} failed")
    return SourceStatus(src, AVAILABLE, f"{len(out)} final games {since}..{until} ({errors} failed calls)",
                        rec["retrieved_at"], None, len(out))


def rebuild_database() -> pd.DataFrame:
    """Re-ingest all sources (historical + current) and refresh PIT caches. Returns quality checks."""
    from sportslab.data import pit
    from sportslab.data.ingest import ingest_all
    from sportslab.data.quality import run_checks
    from sportslab.data.store import export_parquet, read_table, replace_facts
    replace_facts(ingest_all())
    export_parquet()
    pit._load_cached.cache_clear()
    return run_checks({t: read_table(t) for t in ("events", "results", "odds")})


# ------------------------------------------------------------------ fixtures
def _known_names(sport: str) -> set[str]:
    from sportslab.data.pit import load_matches
    try:
        df = load_matches(sport)
    except Exception:  # noqa: BLE001
        return set()
    recent = df[df.event_time >= df.event_time.max() - pd.Timedelta(days=800)]
    return set(recent.home) | set(recent.away)


def _resolve_names(f: Fixture, known: set[str]) -> Fixture:
    h, a = canonical(f.sport, f.raw_home), canonical(f.sport, f.raw_away)
    if known:
        (h, mh), (a, ma) = match_known(h, known), match_known(a, known)
        f.name_match = f"{mh}/{ma}"
    f.home, f.away = h, a
    return f


def parse_fd_fixtures(data: bytes, now: pd.Timestamp, days: int, retrieved_at: str | None = None) -> list[Fixture]:
    df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
    df = df[df.get("Div", pd.Series(dtype=str)) == "E0"]
    out = []
    for _, r in df.iterrows():
        d = datetime.strptime(str(r["Date"]), "%d/%m/%Y" if len(str(r["Date"])) == 10 else "%d/%m/%y")
        hh, mm = (map(int, str(r["Time"]).split(":")) if isinstance(r.get("Time"), str) else (15, 0))
        ko = pd.Timestamp(datetime(d.year, d.month, d.day, hh, mm, tzinfo=LONDON)).tz_convert("UTC")
        if not (now < ko <= now + pd.Timedelta(days=days)):
            continue
        odds = {}
        for pre, book in FD_BOOKS.items():
            vals = [pd.to_numeric(r.get(f"{pre}{x}"), errors="coerce") for x in "HDA"]
            if all(pd.notna(v) and v > 1 for v in vals):
                odds.update({f"odds_{book}_opening_{x}": float(v) for x, v in zip("HDA", vals)})
        out.append(Fixture("football", r["HomeTeam"], r["AwayTeam"], ko, "football-data.co.uk/fixtures.csv",
                           raw_home=r["HomeTeam"], raw_away=r["AwayTeam"], odds=odds, odds_retrieved_at=retrieved_at))
    return out


def fixtures_from_snapshots(sport: str, snaps, now: pd.Timestamp, days: int, source: str) -> list[Fixture]:
    out = []
    for s in snaps:
        st = s.stats or {}
        if s.state != "scheduled" or not st.get("home") or not st.get("away"):
            continue
        if sport == "tennis" and st.get("type") not in (None, "singles"):
            continue
        ko = pd.to_datetime(st.get("start_time"), utc=True, errors="coerce") if st.get("start_time") else None
        ko = None if ko is None or pd.isna(ko) else ko
        if ko is not None and not (now < ko <= now + pd.Timedelta(days=days)):
            continue
        out.append(Fixture(sport, st["home"], st["away"], ko, source, raw_home=st["home"], raw_away=st["away"],
                           extra={k: st.get(k) for k in ("tournament", "round", "competition") if st.get(k)}))
    return out


def upcoming_fixtures(sport: str, days: int = 7, now: pd.Timestamp | None = None) -> tuple[list[Fixture], list[SourceStatus]]:
    now = now or now_utc()
    statuses: list[SourceStatus] = []
    fixtures: list[Fixture] = []
    if offline():
        return [], [SourceStatus("all", UNAVAILABLE, "offline mode (SPORTSLAB_OFFLINE=1)")]
    if sport == "football":
        data, err = http_get(FD_FIXTURES_URL)
        if data is not None:
            rec = register_current("football_fixtures", sport="football", path="football/current/fixtures.csv",
                                   data=data, url=FD_FIXTURES_URL, original_provider="football-data.co.uk fixtures",
                                   season=season_label("football", now), role="fixtures")
            try:
                fixtures = parse_fd_fixtures(data, now, days, rec["retrieved_at"])
                statuses.append(SourceStatus("football-data.co.uk/fixtures.csv", AVAILABLE,
                                             f"{len(fixtures)} EPL fixtures in the next {days} days (with odds)",
                                             rec["retrieved_at"], FD_FIXTURES_URL, len(fixtures)))
            except Exception as e:  # noqa: BLE001
                statuses.append(SourceStatus("football-data.co.uk/fixtures.csv", UNAVAILABLE, f"parse error: {e}",
                                             url=FD_FIXTURES_URL))
        else:
            statuses.append(SourceStatus("football-data.co.uk/fixtures.csv", UNAVAILABLE, err, url=FD_FIXTURES_URL))
    if not fixtures:
        from sportslab.live.sources import SportsSkillsLive
        feed = SportsSkillsLive(sport)
        snaps, errs = [], []
        for d in pd.date_range(now.normalize(), now.normalize() + pd.Timedelta(days=days), freq="D"):
            params = {"date": d.strftime("%Y-%m-%d")}
            if sport == "tennis":
                params["tour"] = "atp"
            r = feed.fetch(**params)
            if r.status == AVAILABLE:
                snaps += r.snapshots
            else:
                errs.append(r.error)
                if len(errs) >= 2 and not snaps:
                    break
        if sport == "football":
            snaps = [s for s in snaps if (s.stats or {}).get("competition") == "premier-league"]
        fx = fixtures_from_snapshots(sport, snaps, now, days, f"sports-skills/{sport} (ESPN)")
        statuses.append(SourceStatus(f"sports-skills/{sport} schedule (ESPN)", AVAILABLE if not errs or snaps else
                                     UNAVAILABLE, f"{len(fx)} scheduled matches" + (f"; errors: {errs[0]}" if errs else ""),
                                     now.isoformat(), None, len(fx)))
        fixtures = fx
    known = _known_names(sport) if sport != "tennis" else set()
    fixtures = [_resolve_names(f, known) for f in fixtures]
    uniq = {f.key: f for f in fixtures}
    return sorted(uniq.values(), key=lambda f: (f.kickoff is None, f.kickoff or now)), statuses


# ------------------------------------------------------------------ forecasting
@dataclass
class UpcomingForecast:
    fixture: dict
    status: str
    reasons: list
    probs: dict            # model -> {label: p}   (MODEL)
    ensemble: dict | None  # calibrated ensemble   (MODEL)
    market: dict | None    # de-vigged opening odds (OBSERVED prices -> derived)
    spread: float | None
    data_as_of: str | None
    teams: dict
    sources: list
    computed_at: str
    labels: tuple

    def to_dict(self):
        return asdict(self)


def _tuned(sport: str) -> dict:
    out = {}
    p = ARTIFACTS_DIR / "tuning.parquet"
    if p.exists():
        t = pd.read_parquet(p)
        for model, g in t[t.sport == sport].groupby("model"):
            best = g.sort_values("val_log_loss").iloc[0]
            out[model] = {k: float(best[k]) for k in g.columns
                          if k not in ("model", "sport", "val_log_loss") and pd.notna(best[k])}
    return out


def _temperature(sport: str, name: str = "ensemble:mean") -> float:
    p = ARTIFACTS_DIR / "validation_summary.json"
    if p.exists():
        return float(json.loads(p.read_text()).get("calibrators", {}).get(sport, {}).get(name, 1.0))
    return 1.0


def _models(sport: str, labels):
    from sportslab.models.goals import DixonColes, GammaPoisson
    from sportslab.models.ml import LogisticModel
    from sportslab.models.ratings import Elo, GaussianRating
    tuned = _tuned(sport)
    three = len(labels) == 3
    elo_kw = tuned.get("elo", {"k": 20.0 if three else 8.0, "home_adv": 65.0 if three else 50.0})
    ms = {"elo": Elo(labels, **elo_kw, mov=three, new_team="low" if sport == "football" else "mean", name="elo")}
    if sport in ("football", "hockey"):
        ms["dixon-coles"] = DixonColes(labels, **tuned.get("dixon-coles", {}), name="dixon-coles")
        ms["bayes:gamma-poisson"] = GammaPoisson(labels, n_samples=0, name="bayes:gamma-poisson")
    if not three:
        ms["bayes:gaussian-rating"] = GaussianRating(labels, **tuned.get("bayes:gaussian-rating", {}),
                                                     name="bayes:gaussian-rating")
    ms["logistic"] = LogisticModel(labels, sport, name="logistic")
    return ms


def freshness(sport: str, df: pd.DataFrame, now: pd.Timestamp) -> tuple[str, list[str], str | None]:
    hist = df.dropna(subset=["outcome"])
    hist = hist[hist.result_available_at <= now]
    if hist.empty:
        return INSUFFICIENT, ["no historical results available"], None
    last = hist.result_available_at.max()
    cur = season_label(sport, now)
    prev = previous_season(sport, cur)
    seasons = set(hist.season)
    data_as_of = last.isoformat()
    if cur not in seasons and prev not in seasons:
        return INSUFFICIENT, [f"latest results in the data are from season {hist.season.iloc[-1]} "
                              f"(last result {last:%Y-%m-%d}); ratings would be years out of date. "
                              f"Load {prev} / {cur} results first."], data_as_of
    if cur not in seasons and now - season_start(sport, cur) > pd.Timedelta(days=21):
        return STALE, [f"no {cur} results loaded (last result {last:%Y-%m-%d}): ratings ignore this season's "
                       f"matches so far. Use 'Refresh current-season results'."], data_as_of
    return OK, [], data_as_of


def forecast_fixtures(sport: str, fixtures: list[Fixture], now: pd.Timestamp | None = None,
                      df: pd.DataFrame | None = None) -> list[UpcomingForecast]:
    now = now or now_utc()
    if sport == "tennis":
        return forecast_tennis(fixtures, now)
    from sportslab.data.pit import load_matches, outcome_matrix  # noqa: F401
    from sportslab.ev.odds import devig
    from sportslab.models.base import LABELS, walk_forward
    from sportslab.models.calibration import TemperatureScaler
    from sportslab.research.validation import ENSEMBLE
    labels = LABELS[sport]
    df = load_matches(sport) if df is None else df
    status, reasons, data_as_of = freshness(sport, df, now)
    fx = [f for f in fixtures if f.kickoff is None or f.kickoff > now]
    if not fx:
        return []
    hist = df[(df.result_available_at <= now) | df.outcome.isna()]
    hist = hist[hist.event_time < now]          # nothing scheduled after 'now' from the history file
    team_info = {}
    for f in fx:
        for t in (f.home, f.away):
            if t not in team_info:
                g = hist[((hist.home == t) | (hist.away == t)) & hist.outcome.notna()]
                rec = g[g.event_time >= now - pd.Timedelta(days=400)]
                team_info[t] = dict(results_last_400d=int(len(rec)),
                                    last_match=None if g.empty else g.event_time.max().strftime("%Y-%m-%d"))
    if status == INSUFFICIENT:
        return [UpcomingForecast(_fx_dict(f), INSUFFICIENT, reasons, {}, None, None, None, data_as_of,
                                 {t: team_info[t] for t in (f.home, f.away)}, _sources(sport), now.isoformat(), labels)
                for f in fx]
    rows = []
    for f in fx:
        ko = f.kickoff if f.kickoff is not None else now + pd.Timedelta(hours=12)
        r = {c: np.nan for c in df.columns}
        r.update(event_id=f.key, sport=sport, competition=COMPETITION[sport], season=season_label(sport, ko),
                 event_time=ko, event_time_precision="minute" if f.kickoff is not None else "day",
                 forecast_time=now, home=f.home, away=f.away, neutral=0, outcome=np.nan,
                 result_available_at=pd.NaT)
        r.update({k: v for k, v in f.odds.items() if k in df.columns or k.startswith("odds_")})
        rows.append(r)
    frame = pd.concat([hist, pd.DataFrame(rows)], ignore_index=True)
    for c in ("event_time", "forecast_time", "result_available_at"):
        frame[c] = pd.to_datetime(frame[c], utc=True)
    probs: dict[str, pd.DataFrame] = {}
    for name, m in _models(sport, labels).items():
        p = walk_forward(frame, m, refit_from=now, record_from=now).preds.set_index("event_id")
        probs[name] = p[[f"p_{l}" for l in labels]]
    T = _temperature(sport)
    out = []
    for f in fx:
        per = {n: dict(zip(labels, map(float, P.loc[f.key].to_numpy()))) for n, P in probs.items()
               if f.key in P.index and np.isfinite(P.loc[f.key].to_numpy()).all()}
        comps = [np.array([per[c][l] for l in labels]) for c in ENSEMBLE[sport] if c in per]
        ens = spread = None
        if comps:
            mean = np.mean(comps, axis=0)
            ts = TemperatureScaler()
            ts.T = T
            ens = dict(zip(labels, map(float, ts.transform(mean[None, :])[0])))
            spread = float(np.std(comps, axis=0).max())
        market = None
        for book in ("pinnacle", "market_avg", "bet365"):
            o = [f.odds.get(f"odds_{book}_opening_{l}") for l in labels]
            if all(o):
                market = {"book": book, "odds": dict(zip(labels, o)),
                          "devig": dict(zip(labels, map(float, devig(o)))), "retrieved_at": f.odds_retrieved_at}
                break
        rs = list(reasons)
        for t in (f.home, f.away):
            if team_info[t]["results_last_400d"] < 10:
                rs.append(f"{t}: only {team_info[t]['results_last_400d']} results in the last 400 days -> rating "
                          f"largely imputed (MODEL assumption)")
        if "unknown" in f.name_match:
            rs.append(f"team name not matched to history ({f.name_match}); treated as new team")
        rs.append("lineups / injuries UNKNOWN: not used")
        out.append(UpcomingForecast(_fx_dict(f), status, rs, per, ens, market, spread, data_as_of,
                                    {t: team_info[t] for t in (f.home, f.away)}, _sources(sport),
                                    now.isoformat(), labels))
    return out


def _fx_dict(f: Fixture) -> dict:
    d = asdict(f)
    d["kickoff"] = None if f.kickoff is None else f.kickoff.isoformat()
    d["key"] = f.key
    return d


def _sources(sport: str) -> list[dict]:
    from sportslab.data.store import read_table
    try:
        s = read_table("sources", where="WHERE sport = ?", params=(sport,))
        return s[["source_id", "original_provider", "role", "retrieved_at", "sha256"]].to_dict("records")
    except Exception:  # noqa: BLE001
        return []


# ------------------------------------------------------------------ tennis (rank model)
_TENNIS_COEF: dict | None = None


def tennis_rank_model() -> dict:
    """Logistic P(p1 wins) = σ(a + b·log(rank_p2/rank_p1)), fitted on all historical ATP matches."""
    global _TENNIS_COEF
    if _TENNIS_COEF is None:
        from sklearn.linear_model import LogisticRegression

        from sportslab.data.pit import load_matches
        d = load_matches("tennis").dropna(subset=["pre_home_rank", "pre_away_rank", "outcome"])
        d = d[d.decided_by.astype(str) != "Walkover"]
        x = np.log(d.pre_away_rank / d.pre_home_rank).to_numpy()[:, None]
        lr = LogisticRegression(C=10.0).fit(x, (d.outcome == "H").astype(int))
        _TENNIS_COEF = dict(a=float(lr.intercept_[0]), b=float(lr.coef_[0, 0]), n=int(len(d)),
                            data_until=str(d.event_time.max().date()))
    return _TENNIS_COEF


def fetch_atp_rankings() -> tuple[dict[str, dict], SourceStatus]:
    src = "sports-skills/tennis.get_rankings (ESPN)"
    if offline():
        return {}, SourceStatus(src, UNAVAILABLE, "offline mode")
    try:
        from sports_skills import tennis
        data = tennis.get_rankings(tour="atp", limit=500)
    except Exception as e:  # noqa: BLE001
        return {}, SourceStatus(src, UNAVAILABLE, f"{type(e).__name__}: {e}"[:300])
    payload = data.get("data", data) if isinstance(data, dict) else {}
    if not isinstance(payload, dict) or data.get("status") is False or payload.get("error"):
        return {}, SourceStatus(src, UNAVAILABLE, str((data or {}).get("message"))[:300])
    ranks = {e["name"]: e for e in payload.get("rankings", []) if e.get("name") and e.get("rank")}
    return ranks, SourceStatus(src, AVAILABLE, f"{len(ranks)} ranked players", now_utc().isoformat(), None, len(ranks))


def forecast_tennis(fixtures: list[Fixture], now: pd.Timestamp, rankings: dict | None = None) -> list[UpcomingForecast]:
    labels = ("H", "A")
    coef = tennis_rank_model()
    if rankings is None:
        rankings, st = fetch_atp_rankings()
    else:
        st = SourceStatus("rankings (provided)", AVAILABLE, n=len(rankings))
    out = []
    for f in fixtures:
        if f.kickoff is not None and f.kickoff <= now:
            continue
        rh, ra = rankings.get(f.home), rankings.get(f.away)
        reasons = [f"rank-only model fitted on {coef['n']} ATP matches up to {coef['data_until']} "
                   "(validated: test log loss 0.622 vs Elo 0.619); form, surface, fatigue NOT used"]
        info = {f.home: rh, f.away: ra}
        if st.status != AVAILABLE or rh is None or ra is None:
            missing = [p for p, r in ((f.home, rh), (f.away, ra)) if r is None]
            reasons.append(f"current ATP ranking UNKNOWN for {missing or 'both'} ({st.detail})")
            out.append(UpcomingForecast(_fx_dict(f), INSUFFICIENT, reasons, {}, None, None, None, None, info,
                                        [asdict(st)], now.isoformat(), labels))
            continue
        z = coef["a"] + coef["b"] * math.log(ra["rank"] / rh["rank"])
        p = 1 / (1 + math.exp(-z))
        probs = {"rank-logit": {"H": p, "A": 1 - p}}
        out.append(UpcomingForecast(_fx_dict(f), OK, reasons, probs, probs["rank-logit"], None, None,
                                    st.retrieved_at, info, [asdict(st)], now.isoformat(), labels))
    return out


# ------------------------------------------------------------------ logging & resolution
def log_upcoming(fc: UpcomingForecast) -> str | None:
    if fc.status == INSUFFICIENT or not fc.ensemble:
        return None
    from sportslab.backtest.journal import log_forecast
    f = fc.fixture
    return log_forecast(mode="prematch-upcoming", sport=f["sport"], event_id=f["key"],
                        model="ensemble:mean+temp" if f["sport"] != "tennis" else "rank-logit",
                        as_of=fc.computed_at, probs=fc.ensemble, status=fc.status,
                        inputs={"fixture": f, "per_model": fc.probs, "market": fc.market, "spread": fc.spread,
                                "reasons": fc.reasons, "data_as_of": fc.data_as_of},
                        sources=fc.sources)


def resolve_logged(sport: str | None = None) -> list[dict]:
    """Attach real outcomes (from the DB) to logged upcoming forecasts once results exist."""
    from sportslab.backtest.journal import resolve_forecast
    from sportslab.data.store import connect, read_table
    with connect() as con:
        logs = pd.read_sql_query("SELECT forecast_id, sport, inputs_json FROM forecast_log WHERE mode='prematch-upcoming' "
                                 "AND resolved_outcome IS NULL", con)
    if sport:
        logs = logs[logs.sport == sport]
    done = []
    for r in logs.itertuples():
        fx = json.loads(r.inputs_json)["fixture"]
        if not fx.get("kickoff"):
            continue
        ko = pd.Timestamp(fx["kickoff"])
        ev = read_table("events", where="WHERE sport = ? AND home = ? AND away = ?", params=(r.sport, fx["home"], fx["away"]))
        if ev.empty:
            continue
        ev["dt"] = (pd.to_datetime(ev.event_time, utc=True) - ko).abs()
        ev = ev[ev.dt <= pd.Timedelta(hours=36)]
        if ev.empty:
            continue
        res = read_table("results", where="WHERE event_id = ?", params=(ev.sort_values("dt").event_id.iloc[0],))
        if res.empty or pd.isna(res.outcome.iloc[0]):
            continue
        done.append(resolve_forecast(r.forecast_id, res.outcome.iloc[0]))
    return done
