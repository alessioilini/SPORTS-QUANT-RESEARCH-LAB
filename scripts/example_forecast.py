#!/usr/bin/env python3
"""Worked prematch forecast with full data provenance (point-in-time, out-of-sample match).

Usage: python scripts/example_forecast.py [EVENT_ID]
Default: the last EPL 2025-26 match with complete Pinnacle opening+closing prices.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

from sportslab import DISCLAIMER
from sportslab.backtest.journal import log_forecast, resolve_forecast
from sportslab.config import ARTIFACTS_DIR, GLOBAL_SEED
from sportslab.data.pit import history_view, load_matches
from sportslab.data.store import read_table
from sportslab.ev.odds import devig, explain
from sportslab.models import montecarlo as mc
from sportslab.models.base import walk_forward
from sportslab.models.goals import DixonColes, GammaPoisson, grid_to_probs, score_grid

warnings.filterwarnings("ignore")
L = ("H", "D", "A")

df = load_matches("football")
full = df[(df.season == "2025-26")].dropna(subset=[f"odds_pinnacle_{s}_{x}" for s in ("opening", "closing") for x in L])
eid = sys.argv[1] if len(sys.argv) > 1 else full.sort_values("event_time").event_id.iloc[-1]
row = df[df.event_id == eid].iloc[0]
t = row.forecast_time
hist = history_view(df, t)
assert (hist.result_available_at <= t).all()

best = json.loads((ARTIFACTS_DIR / "football_dc_best.json").read_text())
dc = DixonColes(L, **best)
dc.refit(hist, t)
lam, mu = dc.rates(row.home, row.away)
p_dc = grid_to_probs(score_grid(lam, mu, dc.rho), L)
sim = mc.simulate_match(lam, mu, dc.rho, n=20000, seed=GLOBAL_SEED)

gp = GammaPoisson(L, n_samples=200, seed=GLOBAL_SEED)
gp.observe(hist.sort_values(["result_available_at", "event_id"]))
p_gp = gp.predict(df[df.event_id == eid])[0]
sigma_gp = float(gp.last_sigma[0])

preds = pd.read_parquet(ARTIFACTS_DIR / "predictions.parquet")
pe = preds[(preds.event_id == eid)].set_index("model")[["p_H", "p_D", "p_A"]]
ens = pe.loc["ensemble:mean+temp"].to_numpy()
odds_open = [row[f"odds_pinnacle_opening_{x}"] for x in L]
odds_close = [row[f"odds_pinnacle_closing_{x}"] for x in L]
ex = explain(odds_open, ens, list(L))

srcs = read_table("sources")
used = srcs[(srcs.sport == "football") & (srcs.role == "primary")]
used = used[used.season <= row.season][["source_id", "original_provider", "host_repo", "retrieved_at", "sha256", "url"]]
odds_rows = read_table("odds", where="WHERE event_id = ? AND bookmaker = 'pinnacle'", params=(eid,))

out = {
    "disclaimer": DISCLAIMER,
    "event": {"event_id": eid, "match": f"{row.home} vs {row.away}", "competition": "EPL", "season": row.season,
              "kickoff_utc": str(row.event_time), "forecast_time_utc": str(t),
              "forecast_time_rule": "opening-odds collection instant (Fri/Tue 16:00 UK, approx)"},
    "information_set": {"results_used": int(len(hist)), "last_result_used_available_at": str(hist.result_available_at.max()),
                        "lookahead_check": "all results used have available_at <= forecast_time"},
    "models": {
        "dixon_coles": {"lambda_home": lam, "mu_away": mu, "rho": dc.rho, "home_adv_log": dc.gamma,
                        "hyperparameters(tuned on 2023-24 only)": best, "probs": dict(zip(L, map(float, p_dc)))},
        "monte_carlo(dixon-coles, n=20000, seed=%d)" % GLOBAL_SEED: {"H": sim.p_home, "D": sim.p_draw, "A": sim.p_away,
                                                                     "over_2.5": sim.p_over_2_5, "mc_se": sim.mc_se},
        "bayes_gamma_poisson": {"probs": dict(zip(L, map(float, p_gp))), "posterior_sd_of_P(H)": sigma_gp},
        "walk_forward_artifacts": {m: dict(zip(L, map(float, pe.loc[m]))) for m in
                                   ["elo", "logistic", "gradient-boosting", "ensemble:mean+temp"] if m in pe.index},
    },
    "market(OBSERVED, Pinnacle)": {"opening_odds": dict(zip(L, odds_open)), "opening_devig": dict(zip(L, map(float, devig(odds_open)))),
                                   "closing_odds(after forecast_time; benchmark only)": dict(zip(L, odds_close))},
    "ev_explanation(education)": ex.text,
    "data_sources": used.to_dict("records"),
    "odds_provenance": odds_rows[["snapshot", "selection", "price", "observed_at", "observed_at_precision",
                                  "source_id"]].to_dict("records"),
    "unknown_inputs": ["lineups", "injuries", "player availability", "weather"],
    "actual_result(OBSERVED, revealed after the fact)": {"score": f"{int(row.home_score)}-{int(row.away_score)}",
                                                          "outcome": row.outcome},
}
fid = log_forecast(mode="prematch", sport="football", event_id=eid, model="ensemble:mean+temp", as_of=str(t),
                   probs=dict(zip(L, map(float, ens))), status="OK",
                   inputs={"lambda": lam, "mu": mu, "rho": dc.rho}, sources=used.source_id.tolist())
out["forecast_log"] = resolve_forecast(fid, row.outcome)
(ARTIFACTS_DIR / "example_forecast.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))
