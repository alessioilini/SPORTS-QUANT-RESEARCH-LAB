#!/usr/bin/env python3
"""Run the full chronological model validation and write artifacts to reports/artifacts/."""
import json
import sys
import time
import warnings

import pandas as pd

from sportslab.config import ARTIFACTS_DIR
from sportslab.research import validation as V

warnings.filterwarnings("ignore", category=RuntimeWarning)


def log(msg):
    print(msg, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    sports = sys.argv[1:] or ["football", "hockey", "basketball", "tennis"]
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    acc = {k: [] for k in ("predictions", "metrics", "comparisons", "reliability", "tuning", "spread")}
    meta = {"timing": {}, "calibrators": {}}
    for s in sports:
        r = V.run_sport(s, log=log)
        for k in acc:
            acc[k].append(r[k])
        meta["timing"][s] = r["timing"]
        meta["calibrators"][s] = r["calibrators"]
    for k, frames in acc.items():
        df = pd.concat(frames, ignore_index=True)
        if k == "predictions":
            for c in ("forecast_time", "event_time"):
                df[c] = pd.to_datetime(df[c], utc=True)
        df.to_parquet(ARTIFACTS_DIR / f"{k}.parquet", index=False)
    if "football" in sports:
        fb = acc["predictions"][sports.index("football")]
        live, live_df, rates = V.football_live_validation(fb, log=log)
        live_df.to_parquet(ARTIFACTS_DIR / "live_ht_validation.parquet", index=False)
        meta["live_ht"] = live
        meta["montecarlo"] = V.montecarlo_checks(fb, rates, log=log)
        V.season_simulation(log=log).to_parquet(ARTIFACTS_DIR / "season_simulation.parquet", index=False)
    meta["basketball_live_sigma"] = V.estimate_basketball_sigma(log=log)
    meta["runtime_seconds"] = round(time.time() - t0, 1)
    (ARTIFACTS_DIR / "validation_summary.json").write_text(json.dumps(meta, indent=2, default=str))
    m = pd.concat(acc["metrics"])
    show = m[m.split == "test"].sort_values(["sport", "log_loss"])
    log("\nTEST (out-of-sample) metrics:")
    log(show[["sport", "model", "n", "log_loss", "log_loss_lo", "log_loss_hi", "brier", "accuracy", "ece"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    log(f"\nTotal runtime {time.time() - t0:.0f}s")
