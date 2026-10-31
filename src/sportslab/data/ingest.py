"""Normalise raw files into the unified long-format tables.

Point-in-time conventions (documented in DATA_SOURCES.md):

football (EPL, kickoff time known, Europe/London):
  event_time     = kickoff (UTC)
  opening odds   observed_at ≈ latest Friday/Tuesday 16:00 local before kickoff-1h
                 (football-data.co.uk collection policy) – precision 'approx'
  forecast_time  = opening-odds observation instant (model and market see the same moment)
  closing odds   available_at = kickoff
  result+stats   available_at = kickoff + 2h30m
hockey / basketball (North America, date only):
  event_time = date 23:00 UTC (nominal, precision 'day'); forecast_time = date 12:00 UTC
  result available_at = date+1 10:00 UTC
tennis (worldwide, date only):
  event_time = date 12:00 UTC (nominal); forecast_time = date-1 12:00 UTC
  result available_at = date+1 12:00 UTC; rankings available at forecast_time
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from sportslab.config import OBSERVED, RAW_DIR, UNKNOWN
from sportslab.data.fetch import current_sources, load_current_manifest, load_manifest
from sportslab.data.sources import REGISTRY
from sportslab.data.teams import canonical_football, canonical_nba, canonical_nhl

UTC = ZoneInfo("UTC")
LONDON = ZoneInfo("Europe/London")


def iso(ts) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError("naive timestamp")
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(s))


def _retrieved(sid: str) -> str | None:
    return load_manifest().get(sid, {}).get("retrieved_at")


def _read(sid: str) -> pd.DataFrame:
    path = RAW_DIR / REGISTRY[sid].raw_relpath
    return pd.read_csv(path, encoding="latin-1")


# --------------------------------------------------------------------------- football
FD_BOOKS = {"PS": "pinnacle", "B365": "bet365", "Avg": "market_avg", "Max": "market_max"}


def football_odds_observed_at(kickoff_local: datetime) -> datetime:
    """football-data.co.uk: odds collected Friday afternoons (weekend games) and
    Tuesday afternoons (midweek games). Approximate as 16:00 local."""
    limit = kickoff_local - timedelta(hours=1)
    for back in range(0, 8):
        day = (kickoff_local - timedelta(days=back)).date()
        if day.weekday() in (1, 4):  # Tue, Fri
            cand = datetime(day.year, day.month, day.day, 16, 0, tzinfo=LONDON)
            if cand <= limit:
                return cand
    return kickoff_local - timedelta(days=2)


def ingest_football() -> dict[str, pd.DataFrame]:
    ev, res, st, od, cov = [], [], [], [], []
    stat_cols = {"HS": ("home", "shots"), "AS": ("away", "shots"), "HST": ("home", "shots_on_target"),
                 "AST": ("away", "shots_on_target"), "HC": ("home", "corners"), "AC": ("away", "corners"),
                 "HF": ("home", "fouls"), "AF": ("away", "fouls"), "HY": ("home", "yellow"),
                 "AY": ("away", "yellow"), "HR": ("home", "red"), "AR": ("away", "red")}
    files = [(sid, src.season, _read(sid), _retrieved(sid)) for sid, src in REGISTRY.items()
             if src.sport == "football" and src.role == "primary"]
    files += [(sid, m["season"], pd.read_csv(RAW_DIR / m["path"], encoding="latin-1"), m["retrieved_at"])
              for sid, m in current_sources("football").items()]
    for sid, season, df, retrieved in files:
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
        if "Div" in df:
            df = df[df.Div == "E0"]
        df["HomeTeam"] = df.HomeTeam.map(canonical_football)   # one spelling per club across sources
        df["AwayTeam"] = df.AwayTeam.map(canonical_football)
        has_time = "Time" in df.columns
        for _, r in df.iterrows():
            d = datetime.strptime(r["Date"], "%d/%m/%Y" if len(str(r["Date"])) == 10 else "%d/%m/%y")
            if has_time and isinstance(r.get("Time"), str):
                hh, mm = map(int, r["Time"].split(":"))
                kickoff = datetime(d.year, d.month, d.day, hh, mm, tzinfo=LONDON)
                prec = "minute"
            else:
                kickoff = datetime(d.year, d.month, d.day, 12, 0, tzinfo=LONDON)
                prec = "day"
            odds_obs = football_odds_observed_at(kickoff)
            eid = f"FB-EPL-{kickoff:%Y%m%d}-{slug(r['HomeTeam'])}-{slug(r['AwayTeam'])}"
            ev.append(dict(event_id=eid, sport="football", competition="EPL", season=season,
                           event_time=iso(kickoff), event_time_precision=prec, forecast_time=iso(odds_obs),
                           forecast_time_rule="opening-odds collection instant (Fri/Tue 16:00 local, approx)",
                           home=r["HomeTeam"], away=r["AwayTeam"], neutral=0,
                           extra=json.dumps({"referee": r.get("Referee")}) if isinstance(r.get("Referee"), str) else None,
                           source_id=sid, retrieved_at=retrieved))
            done = pd.notna(r.get("FTHG")) and pd.notna(r.get("FTAG"))
            avail = iso(kickoff + timedelta(hours=2, minutes=30) if prec == "minute" else
                        datetime(d.year, d.month, d.day, 23, 59, tzinfo=LONDON))
            if done:
                res.append(dict(event_id=eid, home_score=r["FTHG"], away_score=r["FTAG"], outcome=r["FTR"],
                                ht_home=r.get("HTHG"), ht_away=r.get("HTAG"),
                                regulation_draw=int(r["FTR"] == "D"), decided_by="REG", completed=1,
                                available_at=avail, source_id=sid, retrieved_at=retrieved))
                for c, (side, stat) in stat_cols.items():
                    if c in df.columns and pd.notna(r.get(c)):
                        st.append(dict(event_id=eid, side=side, stat=stat, value=float(r[c]), available_at=avail,
                                       source_id=sid, retrieved_at=retrieved))
            for pre, book in FD_BOOKS.items():
                for snap, suffix in (("opening", ""), ("closing", "C")):
                    cols = [f"{pre}{suffix}{x}" for x in "HDA"]
                    if not all(c in df.columns for c in cols):
                        continue
                    for sel, c in zip("HDA", cols):
                        v = r.get(c)
                        if pd.notna(v) and float(v) > 1.0:
                            od.append(dict(event_id=eid, bookmaker=book, market="1X2", selection=sel, price=float(v),
                                           snapshot=snap,
                                           observed_at=iso(odds_obs if snap == "opening" else kickoff),
                                           observed_at_precision="approx",
                                           available_at=iso(odds_obs if snap == "opening" else kickoff),
                                           source_id=sid, retrieved_at=retrieved))
        cov += [dict(sport="football", source_id=sid, field=f, status=s, note=n) for f, s, n in [
            ("result", OBSERVED, "FT and HT score"), ("team_stats", OBSERVED, "shots, SOT, corners, fouls, cards"),
            ("odds_opening", OBSERVED, "Pinnacle, Bet365, market avg/max; timestamp approx"),
            ("odds_closing", OBSERVED if "PSCH" in df.columns else UNKNOWN, "closing lines"),
            ("lineups", UNKNOWN, "not in source"), ("injuries", UNKNOWN, "no verifiable source available"),
            ("player_stats", UNKNOWN, "not in source"),
            ("kickoff_time", OBSERVED if has_time else UNKNOWN, "UK local time" if has_time else "date only")]]
    return dict(events=pd.DataFrame(ev), results=pd.DataFrame(res), team_match_stats=pd.DataFrame(st),
                odds=pd.DataFrame(od), coverage=pd.DataFrame(cov))


# --------------------------------------------------------------------------- hockey
NA_RULE = "date 12:00 UTC (before any North-American game; results of previous day known)"


def _na_times(date: pd.Timestamp):
    d = pd.Timestamp(date).tz_localize("UTC")
    return d + pd.Timedelta(hours=23), d + pd.Timedelta(hours=12), d + pd.Timedelta(days=1, hours=10)


def ingest_hockey() -> dict[str, pd.DataFrame]:
    ev, res, ext, cov = [], [], [], []
    files = [(sid, src.season, _read(sid), _retrieved(sid)) for sid, src in REGISTRY.items()
             if src.sport == "hockey" and src.role == "primary"]
    files += [(sid, m["season"], pd.read_csv(RAW_DIR / m["path"]), m["retrieved_at"])
              for sid, m in current_sources("hockey").items()]
    for sid, season, df, retrieved in files:
        df.columns = ["Date", "Visitor", "VG", "Home", "HG", "OTSO", *df.columns[6:]]
        df["Home"], df["Visitor"] = df.Home.map(canonical_nhl), df.Visitor.map(canonical_nhl)
        missing = int(df[["VG", "HG"]].isna().any(axis=1).sum())
        df = df.dropna(subset=["VG", "HG"])
        for _, r in df.iterrows():
            et, ft, av = _na_times(pd.Timestamp(r["Date"]))
            eid = f"HK-NHL-{et:%Y%m%d}-{slug(r['Home'])}-{slug(r['Visitor'])}"
            row_season = season if ".." not in str(season) else (
                f"{et.year if et.month >= 9 else et.year - 1}-{(et.year + (1 if et.month >= 9 else 0)) % 100:02d}")
            ev.append(dict(event_id=eid, sport="hockey", competition="NHL", season=row_season, event_time=iso(et),
                           event_time_precision="day", forecast_time=iso(ft), forecast_time_rule=NA_RULE,
                           home=r["Home"], away=r["Visitor"], neutral=0, source_id=sid, retrieved_at=retrieved))
            hg, vg = int(r["HG"]), int(r["VG"])
            otso = r["OTSO"] if isinstance(r["OTSO"], str) else "REG"
            res.append(dict(event_id=eid, home_score=hg, away_score=vg, outcome="H" if hg > vg else "A",
                            regulation_draw=int(otso in ("OT", "SO")), decided_by=otso, completed=1,
                            available_at=iso(av), source_id=sid, retrieved_at=retrieved))
        cov += [dict(sport="hockey", source_id=sid, field=f, status=s, note=n) for f, s, n in [
            ("result", OBSERVED, f"final score + OT/SO flag; {missing} rows without score dropped"),
            ("start_time", UNKNOWN, "date only"), ("odds", UNKNOWN, "no verified free odds source reachable"),
            ("team_stats", UNKNOWN, "not in source"), ("lineups", UNKNOWN, "not in source"),
            ("injuries", UNKNOWN, "no verifiable source")]]
    # FiveThirtyEight 2021-22 benchmark
    sid = "hockey_nhl_538_2122"
    df = _read(sid)
    retrieved = _retrieved(sid)
    for _, r in df.iterrows():
        et, ft, av = _na_times(pd.Timestamp(r["date"]))
        eid = f"HK-NHL538-{et:%Y%m%d}-{slug(r['team1'])}-{slug(r['team2'])}"
        ev.append(dict(event_id=eid, sport="hockey", competition="NHL-538", season="2021-22", event_time=iso(et),
                       event_time_precision="day", forecast_time=iso(ft), forecast_time_rule=NA_RULE,
                       home=r["team1"], away=r["team2"], neutral=0, source_id=sid, retrieved_at=retrieved))
        res.append(dict(event_id=eid, outcome="H" if r["prob1_outcome"] == 1 else "A", completed=1,
                        decided_by=UNKNOWN, available_at=iso(av), source_id=sid, retrieved_at=retrieved))
        for sel, p in (("H", r["prob1"]), ("A", r["prob2"])):
            ext.append(dict(event_id=eid, provider="fivethirtyeight", selection=sel, prob=float(p),
                            available_at=iso(ft), available_at_precision="approx (published pre-game)",
                            source_id=sid, retrieved_at=retrieved))
    cov.append(dict(sport="hockey", source_id=sid, field="team1_is_home", status=OBSERVED,
                    note="538 convention; scores not provided"))
    return dict(events=pd.DataFrame(ev), results=pd.DataFrame(res), external_forecasts=pd.DataFrame(ext),
                coverage=pd.DataFrame(cov))


def _basketball_rows(df: pd.DataFrame, sid: str, retrieved):
    ev, res, ext = [], [], []
    for _, r in df.iterrows():
        et, ft, av = _na_times(pd.Timestamp(r["date"]))
        eid = f"BB-NBA-{et:%Y%m%d}-{slug(r['team1'])}-{slug(r['team2'])}"
        season = f"{int(r['season']) - 1}-{str(int(r['season']))[2:]}"
        ev.append(dict(event_id=eid, sport="basketball", competition="NBA", season=season, event_time=iso(et),
                       event_time_precision="day", forecast_time=iso(ft), forecast_time_rule=NA_RULE,
                       home=r["team1"], away=r["team2"], neutral=0, source_id=sid, retrieved_at=retrieved))
        res.append(dict(event_id=eid, outcome="H" if r["prob1_outcome"] == 1 else "A", completed=1,
                        decided_by=UNKNOWN, available_at=iso(av), source_id=sid, retrieved_at=retrieved))
        for sel, p in (("H", r["prob1"]), ("A", r["prob2"])):
            if pd.isna(p):
                continue
            ext.append(dict(event_id=eid, provider="fivethirtyeight", selection=sel, prob=float(p),
                            available_at=iso(ft), available_at_precision="approx (published pre-game)",
                            source_id=sid, retrieved_at=retrieved))
    return ev, res, ext


# --------------------------------------------------------------------------- basketball
def ingest_basketball() -> dict[str, pd.DataFrame]:
    ev, res, ext, cov = [], [], [], []
    files = [("basketball_nba_538", _read("basketball_nba_538"), _retrieved("basketball_nba_538"))]
    files += [(sid, pd.read_csv(RAW_DIR / m["path"]), m["retrieved_at"]) for sid, m in current_sources("basketball").items()]
    for sid, df, retrieved in files:
        df["team1"], df["team2"] = df.team1.map(canonical_nba), df.team2.map(canonical_nba)
        ev_, res_, ext_ = _basketball_rows(df, sid, retrieved)
        ev += ev_; res += res_; ext += ext_
    sid = "basketball_nba_538"
    cov += [dict(sport="basketball", source_id=sid, field=f, status=s, note=n) for f, s, n in [
        ("result", OBSERVED, "winner only, no score"), ("team1_is_home", OBSERVED,
                                                         "verified on 2015-10-27 openers and 2022 Finals G6"),
        ("odds", UNKNOWN, "no verified free odds source reachable"),
        ("external_forecast", OBSERVED, "FiveThirtyEight pre-game probability"),
        ("bubble_2020_neutral_site", UNKNOWN, "2019-20 restart games not flagged neutral in source"),
        ("lineups", UNKNOWN, "not in source"), ("injuries", UNKNOWN, "not in source")]]
    return dict(events=pd.DataFrame(ev), results=pd.DataFrame(res), external_forecasts=pd.DataFrame(ext),
                coverage=pd.DataFrame(cov))


# --------------------------------------------------------------------------- tennis
TENNIS_RULE = "date-1 12:00 UTC (match local date unknown => conservative)"


def ingest_tennis() -> dict[str, pd.DataFrame]:
    ev, res, pp, od, cov = [], [], [], [], []
    for sid, src in REGISTRY.items():
        if src.sport != "tennis":
            continue
        df = _read(sid)
        retrieved = _retrieved(sid)
        df = df.dropna(subset=["Winner", "Loser", "Date"])
        for i, r in df.iterrows():
            d = pd.Timestamp(datetime.strptime(r["Date"], "%m/%d/%Y")).tz_localize("UTC")
            et, ft, av = d + pd.Timedelta(hours=12), d - pd.Timedelta(hours=12), d + pd.Timedelta(days=1, hours=12)
            names = sorted([r["Winner"], r["Loser"]])
            key = f"{r['Date']}|{r['Tournament']}|{r['Round']}|{names[0]}|{names[1]}"
            h = hashlib.sha1(key.encode()).hexdigest()
            winner_is_p1 = int(h, 16) % 2 == 0
            p1, p2 = (r["Winner"], r["Loser"]) if winner_is_p1 else (r["Loser"], r["Winner"])
            eid = f"TN-ATP-{d:%Y%m%d}-{h[:12]}"
            extra = {"tournament": r["Tournament"], "series": r["Series"], "surface": r["Surface"],
                     "court": r["Court"], "round": r["Round"], "best_of": int(r["Best of"]) if pd.notna(r["Best of"]) else None,
                     "location": r["Location"]}
            ev.append(dict(event_id=eid, sport="tennis", competition="ATP", season=src.season, event_time=iso(et),
                           event_time_precision="day", forecast_time=iso(ft), forecast_time_rule=TENNIS_RULE,
                           home=p1, away=p2, neutral=1, extra=json.dumps(extra), source_id=sid,
                           retrieved_at=retrieved))
            comment = str(r.get("Comment"))
            ws = [r.get(f"W{k}") for k in range(1, 6)]
            ls = [r.get(f"L{k}") for k in range(1, 6)]
            sets = [f"{int(a)}-{int(b)}" for a, b in zip(ws, ls) if pd.notna(a) and pd.notna(b)]
            w_sets, l_sets = r.get("Wsets"), r.get("Lsets")
            res.append(dict(event_id=eid, outcome="H" if winner_is_p1 else "A",
                            home_score=(w_sets if winner_is_p1 else l_sets),
                            away_score=(l_sets if winner_is_p1 else w_sets),
                            decided_by=comment, completed=int(comment == "Completed"),
                            detail=json.dumps({"winner_sets": sets}), available_at=iso(av),
                            source_id=sid, retrieved_at=retrieved))
            for side, pref in (("home", "W" if winner_is_p1 else "L"), ("away", "L" if winner_is_p1 else "W")):
                player = p1 if side == "home" else p2
                for stat, col in (("rank", f"{pref}Rank"), ("rank_points", f"{pref}Pts")):
                    v = pd.to_numeric(r.get(col), errors="coerce")
                    if pd.notna(v):
                        pp.append(dict(event_id=eid, side=side, player=player, stat=stat, value=float(v),
                                       available_at=iso(ft), source_id=sid, retrieved_at=retrieved))
            for pre, book in (("PS", "pinnacle"), ("B365", "bet365"), ("Avg", "market_avg"), ("Max", "market_max")):
                wv, lv = pd.to_numeric(r.get(f"{pre}W"), errors="coerce"), pd.to_numeric(r.get(f"{pre}L"), errors="coerce")
                if pd.notna(wv) and pd.notna(lv) and wv > 1 and lv > 1:
                    for sel, v in (("H", wv if winner_is_p1 else lv), ("A", lv if winner_is_p1 else wv)):
                        od.append(dict(event_id=eid, bookmaker=book, market="H2H", selection=sel, price=float(v),
                                       snapshot="single", observed_at=None, observed_at_precision="unknown",
                                       available_at=iso(et), source_id=sid, retrieved_at=retrieved))
        cov += [dict(sport="tennis", source_id=sid, field=f, status=s, note=n) for f, s, n in [
            ("result", OBSERVED, "winner, set scores, retirements/walkovers"),
            ("rankings", OBSERVED, "ATP rank & points at match time"),
            ("odds", OBSERVED, "single snapshot; observation time UNKNOWN -> treated as available at event_time "
                               "(used only for benchmark/backtest, never as a model feature)"),
            ("match_start_time", UNKNOWN, "date only"), ("point_by_point", UNKNOWN, "not in source"),
            ("injuries", UNKNOWN, "no verifiable source")]]
    return dict(events=pd.DataFrame(ev), results=pd.DataFrame(res), player_prematch=pd.DataFrame(pp),
                odds=pd.DataFrame(od), coverage=pd.DataFrame(cov))


def ingest_sources_table() -> pd.DataFrame:
    man = load_manifest()
    rows = []
    for sid, m in load_current_manifest().items():
        rows.append(dict(source_id=sid, sport=m["sport"], kind=m["role"], url=m["url"],
                         original_provider=m["original_provider"], host_repo="(direct, unpinned: changes over time)",
                         license_note="see DATA_SOURCES.md", season=m.get("season"), role=m["role"],
                         notes=m.get("notes", ""), retrieved_at=m["retrieved_at"], sha256=m["sha256"],
                         bytes=m["bytes"], status=m["status"]))
    for sid, s in REGISTRY.items():
        m = man.get(sid, {})
        rows.append(dict(source_id=sid, sport=s.sport, kind=s.kind, url=s.url, original_provider=s.original_provider,
                         host_repo=s.host_repo, license_note=s.license_note, season=s.season, role=s.role,
                         notes=s.notes, retrieved_at=m.get("retrieved_at"), sha256=m.get("sha256"),
                         bytes=m.get("bytes"), status=m.get("status", "NOT_FETCHED")))
    return pd.DataFrame(rows)


def ingest_all() -> dict[str, pd.DataFrame]:
    parts = [ingest_football(), ingest_hockey(), ingest_basketball(), ingest_tennis()]
    out: dict[str, pd.DataFrame] = {"sources": ingest_sources_table()}
    for p in parts:
        for k, v in p.items():
            out[k] = pd.concat([out[k], v], ignore_index=True) if k in out else v
    # de-duplicate events defensively (e.g. identical rows in a source)
    dup = out["events"].event_id.duplicated(keep="first")
    if dup.any():
        bad = set(out["events"].event_id[dup])
        out["events"] = out["events"][~dup]
        for t in ("results",):
            out[t] = out[t].drop_duplicates("event_id")
        out["coverage"] = pd.concat([out["coverage"], pd.DataFrame([dict(
            sport="*", source_id="*", field="duplicate_event_ids", status="WARNING",
            note=f"{len(bad)} duplicates dropped: {sorted(bad)[:5]}")])])
    for t in ("odds",):
        out[t] = out[t].drop_duplicates(["event_id", "bookmaker", "market", "selection", "snapshot"])
    for t in ("team_match_stats", "player_prematch"):
        out[t] = out[t].drop_duplicates(["event_id", "side", "stat"])
    out["external_forecasts"] = out["external_forecasts"].drop_duplicates(["event_id", "provider", "selection"])
    return {k: v.replace({np.nan: None}) for k, v in out.items()}
