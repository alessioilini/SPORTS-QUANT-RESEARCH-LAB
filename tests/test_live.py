"""Live engine: stale feeds, insufficient data, probability sanity, source adapters."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from sportslab.live import tennis as tn
from sportslab.live.engine import DELAYED, INSUFFICIENT, OK, LiveEngine, LiveSnapshot, PrematchPrior
from sportslab.live.sources import (UNAVAILABLE, InferSports, SportsSkillsLive, _parse_clock, _parse_minute,
                                    parse_nhl_gamecenter, snapshot_from_scoreboard_event)

NOW = datetime(2026, 10, 31, 20, 0, tzinfo=timezone.utc)
FB_PRIOR = PrematchPrior("dc", NOW, lam=1.5, mu=1.1)


def fb(minute=30, h=0, a=0, lag=5, red_h=0, red_a=0, **kw):
    return LiveSnapshot("football", "E", "test", NOW, NOW - timedelta(seconds=lag) if lag is not None else None,
                        home_score=h, away_score=a, minute=minute, red_cards_home=red_h, red_cards_away=red_a, **kw)


E = LiveEngine()


def test_fresh_feed_ok_and_probs_sum_to_one():
    f = E.forecast(fb(), FB_PRIOR, NOW)
    assert f.status == OK and f.delay_seconds == pytest.approx(5)
    assert sum(f.probs.values()) == pytest.approx(1.0)


def test_delayed_feed_is_flagged_with_delay():
    f = E.forecast(fb(lag=150), FB_PRIOR, NOW)
    assert f.status == DELAYED and f.delay_seconds == pytest.approx(150)
    assert any("delayed by 150s" in r for r in f.reasons)
    assert f.probs is not None and f.data_as_of.startswith("2026-10-31T19:57:30")


def test_stale_feed_returns_insufficient_data():
    f = E.forecast(fb(lag=3600), FB_PRIOR, NOW)
    assert f.status == INSUFFICIENT and f.probs is None
    assert any("stale" in r for r in f.reasons)


def test_unknown_source_timestamp_is_not_reported_as_ok():
    f = E.forecast(fb(lag=None), FB_PRIOR, NOW)
    assert f.status == DELAYED and f.delay_seconds is None
    assert any("UNVERIFIED" in r for r in f.reasons)


def test_future_timestamp_rejected():
    f = E.forecast(fb(lag=-60), FB_PRIOR, NOW)
    assert f.status == INSUFFICIENT


@pytest.mark.parametrize("snap,prior", [
    (LiveSnapshot("football", "E", "t", NOW, NOW, home_score=1, away_score=None, minute=50), FB_PRIOR),
    (LiveSnapshot("football", "E", "t", NOW, NOW, home_score=1, away_score=0, minute=None), FB_PRIOR),
    (fb(), None),
    (fb(), PrematchPrior("x", NOW, lam=None, mu=1.0)),
    (LiveSnapshot("hockey", "E", "t", NOW, NOW, home_score=1, away_score=1, period=None, clock_remaining_s=100),
     PrematchPrior("x", NOW, lam=3, mu=3)),
    (LiveSnapshot("tennis", "E", "t", NOW, NOW, sets_home=1, sets_away=0), PrematchPrior("x", NOW, p_home=0.6)),
    (LiveSnapshot("cricket", "E", "t", NOW, NOW), None),
])
def test_insufficient_data_never_invents_probabilities(snap, prior):
    f = E.forecast(snap, prior, NOW)
    assert f.status == INSUFFICIENT and f.probs is None and f.reasons


def test_football_monotonicity_and_time():
    p0 = E.forecast(fb(minute=60, h=0, a=0), FB_PRIOR, NOW).probs
    p1 = E.forecast(fb(minute=60, h=1, a=0), FB_PRIOR, NOW).probs
    assert p1["H"] > p0["H"] and p1["A"] < p0["A"]
    late = E.forecast(fb(minute=89, h=1, a=0), FB_PRIOR, NOW).probs
    assert late["H"] > p1["H"] > 0.5
    red = E.forecast(fb(minute=60, h=0, a=0, red_h=1), FB_PRIOR, NOW).probs
    assert red["H"] < p0["H"]
    unk = E.forecast(fb(minute=60, red_h=None, red_a=None), FB_PRIOR, NOW)
    assert any("red-card data UNKNOWN" in r for r in unk.reasons)
    fin = E.forecast(LiveSnapshot("football", "E", "t", NOW, NOW, state="final", home_score=2, away_score=2), None, NOW)
    assert fin.probs == {"H": 0.0, "D": 1.0, "A": 0.0}


def test_football_kickoff_matches_prematch_poisson():
    from sportslab.models.goals import grid_to_probs, score_grid
    p = E.forecast(fb(minute=0), FB_PRIOR, NOW).probs
    exact = grid_to_probs(score_grid(1.5, 1.1), ("H", "D", "A"))
    assert np.allclose(list(p.values()), exact, atol=1e-6)


def test_hockey_power_play_helps_team_with_advantage():
    pr = PrematchPrior("dc", NOW, lam=3.0, mu=2.8)
    base = LiveSnapshot("hockey", "E", "t", NOW, NOW, home_score=1, away_score=1, period=3, clock_remaining_s=600,
                        home_skaters=5, away_skaters=5)
    pp = LiveSnapshot("hockey", "E", "t", NOW, NOW, home_score=1, away_score=1, period=3, clock_remaining_s=600,
                      home_skaters=5, away_skaters=4, pp_remaining_s=120)
    assert E.forecast(pp, pr, NOW).probs["H"] > E.forecast(base, pr, NOW).probs["H"]


def test_basketball_normal_margin():
    pr = PrematchPrior("elo", NOW, p_home=0.6)
    s = LiveSnapshot("basketball", "E", "t", NOW, NOW, home_score=0, away_score=0, period=1, clock_remaining_s=720)
    assert E.forecast(s, pr, NOW).probs["H"] == pytest.approx(0.6, abs=1e-6)
    s2 = LiveSnapshot("basketball", "E", "t", NOW, NOW, home_score=100, away_score=90, period=4, clock_remaining_s=30)
    assert E.forecast(s2, pr, NOW).probs["H"] > 0.99


def test_tennis_live_consistent_with_simulation():
    """Exact Markov chain vs brute-force point simulation from a mid-match state."""
    sh, sa = 0.66, 0.62
    exact = tn.p_match(sh, sa, 3, sh=1, sa=0, gh=3, ga=4, home_serving=True, ph=2, pa=3)
    rng = np.random.default_rng(0)

    def play_game(p_srv, a, b):
        while True:
            if rng.random() < p_srv: a += 1
            else: b += 1
            if a >= 4 and a - b >= 2: return True
            if b >= 4 and b - a >= 2: return False

    def play_tb(first_home):
        h = a = 0
        k = 0
        while True:
            home_srv = (((k + 1) // 2) % 2 == 0) == first_home
            win = rng.random() < (sh if home_srv else 1 - sa)
            h, a, k = h + win, a + (not win), k + 1
            if h >= 7 and h - a >= 2: return True
            if a >= 7 and a - h >= 2: return False

    def play_set(gh, ga, home_srv, ph=0, pa=0):
        first = True
        while True:
            if gh == 6 and ga == 6:
                return play_tb(home_srv)
            if first and (ph or pa):
                won_srv = play_game(sh if home_srv else sa, ph if home_srv else pa, pa if home_srv else ph)
            else:
                won_srv = play_game(sh if home_srv else sa, 0, 0)
            first = False
            home_won = won_srv if home_srv else not won_srv
            gh, ga = gh + home_won, ga + (not home_won)
            home_srv = not home_srv
            if (gh >= 6 and gh - ga >= 2) or gh == 7: return True
            if (ga >= 6 and ga - gh >= 2) or ga == 7: return False

    n, wins = 20000, 0
    for _ in range(n):
        s_h, s_a = 1, 0
        if play_set(3, 4, True, 2, 3): s_h += 1
        else: s_a += 1
        while s_h < 2 and s_a < 2:
            if play_set(0, 0, bool(rng.random() < 0.5)): s_h += 1
            else: s_a += 1
        wins += s_h == 2
    assert abs(wins / n - exact) < 4 * np.sqrt(exact * (1 - exact) / n)


def test_tennis_engine_end_to_end():
    pr = PrematchPrior("elo", NOW, p_home=0.7)
    s = LiveSnapshot("tennis", "E", "t", NOW, NOW, sets_home=0, sets_away=0, games_home=0, games_away=0,
                     points_home=0, points_away=0, home_serving=True, best_of=3)
    f = E.forecast(s, pr, NOW)
    assert f.status == OK and 0.69 < f.probs["H"] < 0.71    # serve asymmetry at 0-0 is small


def test_nhl_gamecenter_real_payload_parse():
    pbp = json.loads((Path(__file__).parent / "fixtures" / "nhl_gamecenter_2023030417_pbp_partial.json").read_text())
    s = parse_nhl_gamecenter(pbp, NOW, NOW)
    assert (s.home_score, s.away_score, s.period) == (1, 0, 1)      # FLA 1-0 EDM (goal at 4:27 of P1)
    assert s.clock_remaining_s == 14 * 60 + 48               # last recorded play: 05:12 of P1
    assert s.home_skaters == 5 and s.away_skaters == 5
    f = E.forecast(s, PrematchPrior("dc", NOW, lam=3.0, mu=2.9), NOW)
    assert f.status == OK and f.probs["H"] > 0.5


def test_scoreboard_parsers():
    assert _parse_clock("12:34 - 2nd") == (2, 754)
    assert _parse_clock("End of 3rd") == (3, 0)
    assert _parse_clock("Final") == (None, None)
    assert _parse_minute("67'") == 67 and _parse_minute("90+3'") == 93 and _parse_minute("HT") == 45
    ev = {"id": "1", "status": "in_progress", "status_detail": "5:00 - 3rd",
          "competitors": [{"home_away": "home", "score": 2}, {"home_away": "away", "score": 1}]}
    s = snapshot_from_scoreboard_event("hockey", ev, NOW)
    assert s.source_updated_at is None and s.period == 3 and s.clock_remaining_s == 300
    f = E.forecast(s, PrematchPrior("dc", NOW, lam=3, mu=3), NOW)
    assert f.status == DELAYED   # scoreboard carries no source timestamp -> never 'OK'


def test_live_sources_fail_closed():
    r = SportsSkillsLive("hockey").fetch()
    assert r.status in ("AVAILABLE", "UNAVAILABLE")
    if r.status == UNAVAILABLE:
        assert r.error and not r.snapshots
    assert InferSports().fetch().status == UNAVAILABLE
