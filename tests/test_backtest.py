"""Virtual-bankroll simulator: hand-checked P&L, chronology, staking rules, cross-check vs backtestbuddy."""
import numpy as np
import pandas as pd
import pytest

from sportslab.backtest.simulator import StrategyConfig, select_positions, simulate

L = ("H", "D", "A")
T0 = pd.Timestamp("2024-01-01", tz="UTC")


def frame(rows):
    """rows: (day, p_H,p_D,p_A, price_H,price_D,price_A, outcome)."""
    out = []
    for i, (d, ph, pd_, pa, oh, od, oa, y) in enumerate(rows):
        out.append(dict(event_id=f"E{i}", forecast_time=T0 + pd.Timedelta(days=d),
                        event_time=T0 + pd.Timedelta(days=d, hours=20),
                        result_available_at=T0 + pd.Timedelta(days=d, hours=22),
                        p_H=ph, p_D=pd_, p_A=pa, price_H=oh, price_D=od, price_A=oa,
                        close_H=oh, close_D=od, close_A=oa, outcome=y))
    return pd.DataFrame(out)


def test_selection_threshold_and_argmax_ev():
    f = frame([(0, 0.6, 0.2, 0.2, 2.0, 4.0, 4.0, "H"),     # EV_H=0.2, EV_D=-0.2
               (1, 0.5, 0.25, 0.25, 1.9, 3.8, 4.1, "A"),   # EV_H=-0.05, EV_A=0.025 < 0.03 -> skip
               (2, 0.3, 0.3, 0.4, 3.0, 3.4, 2.8, "A")])    # EV_A=0.12, EV_D=0.02 -> A
    pos = select_positions(f, L, StrategyConfig(min_edge=0.03))
    assert list(pos.event_id) == ["E0", "E2"] and list(pos.selection) == ["H", "A"]
    assert pos.ev_adj.tolist() == pytest.approx([0.2, 0.12])


def test_fixed_stake_pnl_by_hand_and_commission():
    f = frame([(0, 0.6, 0.2, 0.2, 2.0, 4.0, 4.0, "H"), (1, 0.6, 0.2, 0.2, 2.5, 4.0, 4.0, "A"),
               (2, 0.6, 0.2, 0.2, 2.2, 4.0, 4.0, "H")])
    pos = select_positions(f, L, StrategyConfig(min_edge=0.0))
    led, s = simulate(pos, StrategyConfig(stake=1.0, min_edge=0.0))
    assert s["virtual_pnl"] == pytest.approx(1.0 - 1.0 + 1.2)
    assert s["virtual_roi"] == pytest.approx(1.2 / 3)
    assert s["hit_rate"] == pytest.approx(2 / 3)
    led_c, s_c = simulate(pos, StrategyConfig(stake=1.0, min_edge=0.0, commission=0.05))
    assert s_c["virtual_pnl"] == pytest.approx(1.0 * 0.95 - 1.0 + 1.2 * 0.95)
    assert s_c["total_commission"] == pytest.approx(0.05 * 2.2)


def test_max_drawdown_and_stop():
    rows = [(d, 0.6, 0.2, 0.2, 2.0, 4.0, 4.0, "A") for d in range(30)]
    pos = select_positions(frame(rows), L, StrategyConfig(min_edge=0.0))
    led, s = simulate(pos, StrategyConfig(stake=5.0, initial_bankroll=100, min_edge=0.0))
    assert s["max_drawdown"] == pytest.approx(1.0)        # 20 losses of 5 -> 0
    assert s["n_positions"] == 20                           # cash exhausted, no negative bankroll
    led2, s2 = simulate(pos, StrategyConfig(stake=5.0, initial_bankroll=100, min_edge=0.0, max_drawdown_stop=0.25))
    assert s2["n_positions"] == 6 and s2["halted_by_drawdown_stop"]   # stops after DD > 25 %


def test_fraction_staking_uses_only_settled_bankroll():
    """Two events open before either settles: both stakes come from the same settled bankroll."""
    f = frame([(0, 0.6, 0.2, 0.2, 2.0, 4.0, 4.0, "H"), (0, 0.6, 0.2, 0.2, 2.0, 4.0, 4.0, "H"),
               (2, 0.6, 0.2, 0.2, 2.0, 4.0, 4.0, "H")])
    pos = select_positions(f, L, StrategyConfig(min_edge=0.0))
    led, _ = simulate(pos, StrategyConfig(staking="fraction", fraction=0.1, initial_bankroll=100, min_edge=0.0))
    stakes = led.sort_values("forecast_time").stake.tolist()
    assert stakes[0] == pytest.approx(10) and stakes[1] == pytest.approx(10)   # not 11 (no unsettled winnings)
    assert stakes[2] == pytest.approx(0.1 * 120)


def test_positions_settle_after_open_chronology():
    rng = np.random.default_rng(0)
    rows = [(d, *rng.dirichlet([3, 2, 2]), *rng.uniform(1.8, 5, 3), rng.choice(list("HDA"))) for d in range(200)]
    pos = select_positions(frame(rows), L, StrategyConfig(min_edge=0.0))
    led, s = simulate(pos, StrategyConfig(staking="fraction", fraction=0.02, min_edge=0.0))
    assert (led.settle_time > led.forecast_time).all()
    assert led.settle_time.is_monotonic_increasing
    assert s["virtual_roi_ci_lo"] <= s["virtual_roi"] <= s["virtual_roi_ci_hi"]


def test_uncertainty_haircut_reduces_positions():
    rng = np.random.default_rng(1)
    rows = [(d, *rng.dirichlet([3, 2, 2]), *rng.uniform(1.8, 5, 3), "H") for d in range(300)]
    f = frame(rows)
    f["sigma"] = 0.05
    n0 = len(select_positions(f, L, StrategyConfig(min_edge=0.03)))
    n1 = len(select_positions(f, L, StrategyConfig(min_edge=0.03, uncertainty_k=1.0)))
    assert n1 < n0


def test_clv_computed_from_closing_prices():
    f = frame([(0, 0.6, 0.2, 0.2, 2.2, 4.0, 4.0, "H")])
    f[["close_H", "close_D", "close_A"]] = [1.9, 3.9, 4.2]
    pos = select_positions(f, L, StrategyConfig(min_edge=0.0))
    _, s = simulate(pos, StrategyConfig(min_edge=0.0))
    fair = (1 / 1.9) / (1 / 1.9 + 1 / 3.9 + 1 / 4.2)
    assert s["clv_mean"] == pytest.approx(2.2 * fair - 1)
    assert s["clv_beat_close_share"] == 1.0


def test_crosscheck_against_backtestbuddy():
    import backtestbuddy as btb
    rng = np.random.default_rng(2)
    rows = [(d, *rng.dirichlet([3, 2, 2]), *np.round(rng.uniform(1.8, 5, 3), 2), rng.choice(list("HDA")))
            for d in range(150)]
    pos = select_positions(frame(rows), L, StrategyConfig(min_edge=0.0))
    _, s = simulate(pos, StrategyConfig(stake=1.0, initial_bankroll=1e9, min_edge=0.0))
    d = pos.copy()
    d["date"] = d.forecast_time.dt.tz_convert(None)
    d["y"] = d.outcome.map({"H": 0, "D": 1, "A": 2})
    d["pred"] = d.selection.map({"H": 0, "D": 1, "A": 2})
    bt = btb.PredictionBacktest(data=d.reset_index(drop=True), odds_columns=["price_H", "price_D", "price_A"],
                                outcome_column="y", date_column="date", prediction_column="pred",
                                initial_bankroll=1e9, strategy=btb.FixedStake(amount=1.0))
    bt.run()
    m = bt.calculate_metrics()
    assert m["Total Bets"] == s["n_positions"]
    assert m["Total Profit [$]"] == pytest.approx(s["virtual_pnl"], abs=1e-6)
