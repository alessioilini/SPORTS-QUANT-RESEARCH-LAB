"""Upcoming matches: fixture parsing, freshness gate, PIT-correct forecasts, team names, resolution."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from sportslab import upcoming as U
from sportslab.data.teams import canonical_football, canonical_nba, canonical_nhl, match_known
from sportslab.live.sources import SportsSkillsLive, snapshot_from_tennis_match
from tests.conftest import synthetic_league

L3 = ("H", "D", "A")


def test_team_name_canonicalisation():
    assert canonical_football("Manchester City") == "Man City"
    assert canonical_football("Manchester Utd") == "Man United"
    assert canonical_football("Brighton & Hove Albion") == "Brighton"
    assert canonical_football("Nottingham Forest") == "Nott'm Forest"
    assert canonical_football("Arsenal") == "Arsenal"
    assert canonical_nba("Portland Trail Blazers") == "Trail Blazers"
    assert canonical_nba("Philadelphia 76ers") == "76ers" and canonical_nba("LA Clippers") == "Clippers"
    assert canonical_nhl("Montréal Canadiens") == "Montreal Canadiens"
    assert match_known("Tottenham", {"Tottenham", "Fulham"}) == ("Tottenham", "exact")
    assert match_known("Totenham", {"Tottenham", "Fulham"})[1].startswith("fuzzy")
    assert match_known("Real Madrid", {"Tottenham", "Fulham"})[1] == "unknown"   # never guesses


def test_season_labels():
    assert U.season_label("football", "2026-10-31") == "2026-27"
    assert U.season_label("football", "2026-05-01") == "2025-26"
    assert U.season_label("hockey", "2026-09-15") == "2025-26"
    assert U.season_label("tennis", "2026-10-31") == "2026"
    assert U.previous_season("basketball", "2026-27") == "2025-26"


def _fixtures_csv(rows):
    head = "Div,Date,Time,HomeTeam,AwayTeam,PSH,PSD,PSA,B365H,B365D,B365A\n"
    return (head + "\n".join(",".join(map(str, r)) for r in rows)).encode("latin-1")


def test_parse_football_data_fixtures():
    now = pd.Timestamp("2026-10-31 12:00", tz="UTC")
    data = _fixtures_csv([("E0", "01/11/2026", "15:00", "Manchester City", "Arsenal", 2.1, 3.6, 3.4, 2.0, 3.5, 3.5),
                          ("E0", "30/10/2026", "15:00", "Chelsea", "Fulham", 1.8, 3.8, 4.5, 1.8, 3.8, 4.5),  # past
                          ("SP1", "01/11/2026", "20:00", "Real Madrid", "Getafe", 1.3, 5, 9, 1.3, 5, 9),  # other league
                          ("E0", "20/11/2026", "15:00", "Everton", "Wolves", 2.5, 3.3, 2.9, 2.5, 3.3, 2.9)])  # > 7 days
    fx = U.parse_fd_fixtures(data, now, 7, "2026-10-31T12:00:00+00:00")
    assert len(fx) == 1
    f = fx[0]
    assert f.raw_home == "Manchester City" and f.kickoff == pd.Timestamp("2026-11-01 15:00", tz="UTC")  # GMT
    assert f.odds["odds_pinnacle_opening_H"] == 2.1 and f.odds_retrieved_at
    f = U._resolve_names(f, {"Man City", "Arsenal"})
    assert (f.home, f.away, f.name_match) == ("Man City", "Arsenal", "exact/exact")


def _league_now():
    df, _ = synthetic_league(n_teams=12, n_rounds=60, seed=21)
    df["season"] = np.where(df.event_time < pd.Timestamp("2021-06-01", tz="UTC"), "2020-21", "2021-22")
    return df


def test_freshness_gate():
    df = _league_now()
    last = df.result_available_at.max()
    assert U.freshness("football", df, last + pd.Timedelta(days=1))[0] == U.OK
    st, reasons, _ = U.freshness("football", df, pd.Timestamp("2022-10-31", tz="UTC"))   # season 2022-23 not loaded
    assert st == U.STALE and "2022-23" in reasons[0]
    st, reasons, _ = U.freshness("football", df, pd.Timestamp("2026-10-31", tz="UTC"))
    assert st == U.INSUFFICIENT and "years out of date" in reasons[0]


def test_forecast_future_fixture_is_point_in_time():
    df = _league_now()
    now = df.result_available_at.max() + pd.Timedelta(hours=1)
    fx = [U.Fixture("football", "T00", "T01", now + pd.Timedelta(days=2), "test",
                    odds={f"odds_pinnacle_opening_{l}": o for l, o in zip(L3, (2.0, 3.5, 4.0))}),
          U.Fixture("football", "T03", "T04", now - pd.Timedelta(days=1), "test")]          # already started
    res = U.forecast_fixtures("football", fx, now=now, df=df)
    assert len(res) == 1                                                                     # past one skipped
    r = res[0]
    assert r.status == U.OK and set(r.probs) >= {"elo", "dixon-coles", "bayes:gamma-poisson", "logistic"}
    for p in list(r.probs.values()) + [r.ensemble]:
        assert sum(p.values()) == pytest.approx(1.0)
    assert r.market["book"] == "pinnacle" and sum(r.market["devig"].values()) == pytest.approx(1.0)
    # look-ahead: adding (fake) results AFTER 'now' must not change the forecast
    future = df.iloc[-30:].copy()
    future["event_time"] += pd.Timedelta(days=30)
    future["forecast_time"] += pd.Timedelta(days=30)
    future["result_available_at"] += pd.Timedelta(days=30)
    future["event_id"] = future.event_id + "-F"
    future["outcome"] = "A"
    r2 = U.forecast_fixtures("football", fx, now=now, df=pd.concat([df, future], ignore_index=True))[0]
    for m in r.probs:
        assert r.probs[m] == pytest.approx(r2.probs[m]), m


def test_insufficient_data_returns_no_probabilities():
    df = _league_now()
    now = pd.Timestamp("2026-10-31", tz="UTC")
    r = U.forecast_fixtures("football", [U.Fixture("football", "T00", "T01", now + pd.Timedelta(days=1), "t")],
                            now=now, df=df)[0]
    assert r.status == U.INSUFFICIENT and r.probs == {} and r.ensemble is None


def test_tennis_rank_forecast_and_unknown_player(monkeypatch):
    monkeypatch.setattr(U, "_TENNIS_COEF", {"a": 0.0, "b": 0.45, "n": 1000, "data_until": "2024-06-01"})
    now = pd.Timestamp("2026-10-31", tz="UTC")
    ranks = {"Player A": {"rank": 2}, "Player B": {"rank": 40}}
    fx = [U.Fixture("tennis", "Player A", "Player B", now + pd.Timedelta(days=1), "t"),
          U.Fixture("tennis", "Player A", "Nobody", now + pd.Timedelta(days=1), "t")]
    r1, r2 = U.forecast_tennis(fx, now, rankings=ranks)
    assert r1.status == U.OK and r1.ensemble["H"] == pytest.approx(1 / (1 + np.exp(-0.45 * np.log(20))))
    assert r2.status == U.INSUFFICIENT and r2.probs == {} and any("UNKNOWN" in x for x in r2.reasons)


def test_tennis_scoreboard_structure_is_parsed():
    payload = {"tour": "ATP", "tournaments": [{"name": "Paris Masters", "draws": [{"name": "Men's Singles", "matches": [
        {"id": "1", "status": "in_progress", "date": "2026-10-31T13:00Z", "round": "QF",
         "competitors": [{"type": "singles", "name": "A", "serving": True, "set_scores": [{"games": 6}, {"games": 3}]},
                         {"type": "singles", "name": "B", "serving": False, "set_scores": [{"games": 4}, {"games": 3}]}]},
        {"id": "2", "status": "not_started", "date": "2026-11-01T13:00Z",
         "competitors": [{"type": "singles", "name": "C", "set_scores": []}, {"type": "singles", "name": "D", "set_scores": []}]},
    ]}]}]}
    r = SportsSkillsLive("tennis").parse("t", payload, datetime.now(timezone.utc))
    assert r.status == "AVAILABLE" and r.raw_count == 2
    live, sched = r.snapshots
    assert (live.sets_home, live.sets_away, live.games_home, live.games_away, live.home_serving) == (1, 0, 3, 3, True)
    assert live.points_home is None                     # point score not in feed -> UNKNOWN, not guessed
    fx = U.fixtures_from_snapshots("tennis", r.snapshots, pd.Timestamp("2026-10-31 14:00", tz="UTC"), 3, "t")
    assert [(f.raw_home, f.raw_away) for f in fx] == [("C", "D")]


def test_football_schedule_structure_is_parsed():
    ev = {"id": "9", "status": "not_started", "start_time": "2026-11-01T15:00Z",
          "competition": {"id": "premier-league"}, "scores": {"home": None, "away": None},
          "competitors": [{"team": {"name": "Manchester City"}, "qualifier": "home", "score": None},
                          {"team": {"name": "Arsenal"}, "qualifier": "away", "score": None}]}
    r = SportsSkillsLive("football").parse("t", {"events": [ev]}, datetime.now(timezone.utc))
    s = r.snapshots[0]
    assert s.state == "scheduled" and s.stats["home"] == "Manchester City" and s.stats["competition"] == "premier-league"
    fx = U.fixtures_from_snapshots("football", r.snapshots, pd.Timestamp("2026-10-31", tz="UTC"), 7, "t")
    assert len(fx) == 1 and fx[0].kickoff == pd.Timestamp("2026-11-01 15:00", tz="UTC")


def test_offline_mode_makes_no_calls(monkeypatch):
    monkeypatch.setenv("SPORTSLAB_OFFLINE", "1")
    fx, st = U.upcoming_fixtures("football", 7)
    assert fx == [] and st[0].status == U.UNAVAILABLE
    assert U.refresh_football_current().status == U.UNAVAILABLE
    assert U.refresh_espn_results("hockey", "2026-10-01", "2026-10-02").status == U.UNAVAILABLE


def test_espn_results_written_in_historical_format(monkeypatch, tmp_path):
    """Backfilled NHL results are stored as hockey-reference-style CSV with OT/SO flags."""
    from sportslab.data import fetch as F
    from sportslab.live.engine import LiveSnapshot
    monkeypatch.setattr(F, "RAW_DIR", tmp_path)
    monkeypatch.setattr(F, "CURRENT_MANIFEST_PATH", tmp_path / "CURRENT_MANIFEST.json")
    now = datetime.now(timezone.utc)

    class R:
        status = "AVAILABLE"
        error = None

    def fake(sport, day):
        s = LiveSnapshot("hockey", "e", "t", now, state="final", home_score=3, away_score=2,
                         stats={"home": "Boston Bruins", "away": "Toronto Maple Leafs", "status_detail": "Final/OT"})
        return R(), [s]
    monkeypatch.setattr(U, "_final_snapshots", fake)
    st = U.refresh_espn_results("hockey", "2026-10-10", "2026-10-11")
    assert st.status == U.AVAILABLE and st.n == 2
    man = F.load_current_manifest()["hockey_nhl_espn_recent"]
    df = pd.read_csv(tmp_path / man["path"])
    assert list(df.columns[:5]) == ["Date", "Visitor", "G", "Home", "G.1"] and set(df.iloc[:, 5]) == {"OT"}
