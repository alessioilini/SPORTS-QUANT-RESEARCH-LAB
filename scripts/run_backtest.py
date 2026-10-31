#!/usr/bin/env python3
"""Virtual-bankroll strategy research on historical prices (football EPL, tennis ATP).

Prices used:
  football: Pinnacle OPENING odds (known at forecast_time); Pinnacle CLOSING for CLV.
  tennis:   Pinnacle single pre-match snapshot (time unknown; no CLV possible).
In-sample = train+validation seasons (where models/thresholds were chosen);
out-of-sample = test seasons. Everything is VIRTUAL.
"""
import json
import warnings

import numpy as np
import pandas as pd

from sportslab.backtest.journal import append_ledger, new_run_id
from sportslab.backtest.simulator import StrategyConfig, select_positions, simulate
from sportslab.config import ARTIFACTS_DIR
from sportslab.data.pit import load_matches
from sportslab.models.base import LABELS

warnings.filterwarnings("ignore", category=RuntimeWarning)

PRICE = {"football": ("pinnacle", "opening", "closing"), "tennis": ("pinnacle", "single", None)}
MODELS = {"football": ["ensemble:mean+temp", "dixon-coles+temp", "logistic+temp", "elo+temp",
                       "market:pinnacle-opening"],
          "tennis": ["ensemble:mean+temp", "logistic+temp", "elo+temp"]}
BANK = 1000.0   # virtual units; 1u flat stake = 0.1% so flat strategies are never cash-constrained
STRATEGIES = [
    StrategyConfig("fixed-1u", staking="fixed", stake=1.0, min_edge=0.03, initial_bankroll=BANK),
    StrategyConfig("fraction-1pct", staking="fraction", fraction=0.01, min_edge=0.03, initial_bankroll=BANK),
    StrategyConfig("fraction-2pct+dd-stop-25pct", staking="fraction", fraction=0.02, min_edge=0.03,
                   max_drawdown_stop=0.25, initial_bankroll=BANK),
    StrategyConfig("fixed-1u+commission-2pct", staking="fixed", stake=1.0, min_edge=0.03, commission=0.02,
                   initial_bankroll=BANK),
    StrategyConfig("fixed-1u+uncertainty-k1", staking="fixed", stake=1.0, min_edge=0.03, uncertainty_k=1.0,
                   initial_bankroll=BANK),
]


def frame_for(sport: str, preds: pd.DataFrame, spread: pd.DataFrame) -> pd.DataFrame:
    labels = LABELS[sport]
    book, snap, close = PRICE[sport]
    df = load_matches(sport)
    keep = ["event_id", "result_available_at"] + [f"odds_{book}_{snap}_{l}" for l in labels]
    if close:
        keep += [f"odds_{book}_{close}_{l}" for l in labels]
    odds = df[keep].rename(columns={f"odds_{book}_{snap}_{l}": f"price_{l}" for l in labels} |
                           ({f"odds_{book}_{close}_{l}": f"close_{l}" for l in labels} if close else {}))
    out = preds.merge(odds, on="event_id").merge(spread[["event_id", "ensemble_spread"]], on="event_id", how="left")
    return out.rename(columns={"ensemble_spread": "sigma"})


def backtestbuddy_crosscheck(pos: pd.DataFrame, labels) -> dict:
    """Run OUR simulator and backtestbuddy on the same selections (flat 1u, unconstrained cash); compare P&L."""
    import backtestbuddy as btb
    _, ours_s = simulate(pos, StrategyConfig("xcheck", stake=1.0, initial_bankroll=1e9, min_edge=-1.0))
    d = pos.copy()
    d["date"] = d.forecast_time.dt.tz_convert(None)
    d["outcome_idx"] = d.outcome.map({l: i for i, l in enumerate(labels)})
    d["pred_idx"] = d.selection.map({l: i for i, l in enumerate(labels)})
    bt = btb.PredictionBacktest(data=d.reset_index(drop=True), odds_columns=[f"price_{l}" for l in labels],
                                outcome_column="outcome_idx", date_column="date", prediction_column="pred_idx",
                                initial_bankroll=1e9, strategy=btb.FixedStake(amount=1.0))
    bt.run()
    m = bt.calculate_metrics()
    ours = float(ours_s["virtual_pnl"])
    theirs = float(m["Total Profit [$]"])
    return dict(n=int(ours_s["n_positions"]), ours_pnl=ours, backtestbuddy_pnl=theirs, abs_diff=abs(ours - theirs),
                backtestbuddy_bets=int(m["Total Bets"]))


if __name__ == "__main__":
    preds = pd.read_parquet(ARTIFACTS_DIR / "predictions.parquet")
    spread = pd.read_parquet(ARTIFACTS_DIR / "spread.parquet")
    run_id = new_run_id()
    summaries, ledgers, checks = [], [], []
    for sport, models in MODELS.items():
        labels = LABELS[sport]
        sp = spread[spread.sport == sport]
        for model in models:
            p = preds[(preds.sport == sport) & (preds.model == model)]
            if p.empty:
                print(f"skip {sport}/{model}: no predictions")
                continue
            fr = frame_for(sport, p, sp)
            for period, splits in (("in-sample(train+val)", ["train", "validation"]), ("out-of-sample(test)", ["test"])):
                f = fr[fr.split.isin(splits)].dropna(subset=[f"p_{l}" for l in labels] + [f"price_{l}" for l in labels])
                for cfg in STRATEGIES:
                    pos = select_positions(f, labels, cfg)
                    led, summ = simulate(pos, cfg)
                    summaries.append(dict(sport=sport, model=model, period=period, n_events=len(f), **summ))
                    if cfg.name == "fixed-1u" and period.startswith("out") and len(led):
                        ledgers.append(led.assign(sport=sport, model=model, strategy=cfg.name, period=period))
                        append_ledger(led, run_id, f"{sport}:{model}")
                        checks.append(dict(sport=sport, model=model, **backtestbuddy_crosscheck(pos, labels)))
                    if cfg.name in ("fraction-1pct", "fraction-2pct+dd-stop-25pct") and len(led):
                        ledgers.append(led.assign(sport=sport, model=model, strategy=cfg.name, period=period))
    S = pd.DataFrame(summaries)
    S.to_parquet(ARTIFACTS_DIR / "backtest_summary.parquet", index=False)
    L = pd.concat(ledgers, ignore_index=True)
    keep = ["sport", "model", "strategy", "period", "event_id", "home", "away", "forecast_time", "settle_time",
            "selection", "model_prob", "price_taken", "closing_price", "closing_fair_prob", "won", "stake", "pnl",
            "commission", "bankroll_after", "drawdown", "ev_adj"]
    L[[c for c in keep if c in L]].to_parquet(ARTIFACTS_DIR / "backtest_ledgers.parquet", index=False)
    C = pd.DataFrame(checks)
    (ARTIFACTS_DIR / "backtest_crosscheck.json").write_text(json.dumps(
        dict(run_id=run_id, crosscheck=C.to_dict("records")), indent=2, default=str))
    cols = ["sport", "model", "period", "strategy", "n_positions", "virtual_roi", "virtual_roi_ci_lo",
            "virtual_roi_ci_hi", "virtual_pnl", "max_drawdown", "hit_rate", "clv_mean"]
    print(S[[c for c in cols if c in S]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nbacktestbuddy cross-check:\n", C.to_string(index=False))
    print(f"\nvirtual journal run_id: {run_id}")
