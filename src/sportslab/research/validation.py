"""End-to-end chronological validation of prematch models (+ live HT check, MC checks).

Protocol per sport (seasons are never shuffled):
  burn-in  -> ratings warm up, no evaluation
  train    -> supervised models learn; predictions recorded for in-sample comparison
  validation -> hyper-parameter grids and calibration fitted HERE only
  test     -> touched once with frozen choices; reported as out-of-sample
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportslab.config import ARTIFACTS_DIR, GLOBAL_SEED
from sportslab.data.pit import load_matches, outcome_matrix
from sportslab.ev import metrics as M
from sportslab.models.base import LABELS, walk_forward
from sportslab.models.baselines import BaseRate, ExternalForecast, MarketPre, Uniform, market_benchmark
from sportslab.models.calibration import TemperatureScaler
from sportslab.models.goals import DixonColes, GammaPoisson
from sportslab.models.ml import GBMModel, LogisticModel
from sportslab.models.ratings import Elo, GaussianRating, TennisElo


@dataclass
class SportCfg:
    sport: str
    burn_in: list
    train: list
    validation: list
    test: list
    ordinal: bool = False


CFG = {
    "football": SportCfg("football", ["2018-19"], ["2019-20", "2020-21", "2021-22", "2022-23"], ["2023-24"],
                         ["2024-25", "2025-26"], ordinal=True),
    "hockey": SportCfg("hockey", ["2010-11"], ["2011-12", "2012-13", "2013-14", "2014-15", "2015-16", "2016-17"],
                       ["2017-18"], ["2018-19", "2019-20"]),
    "basketball": SportCfg("basketball", ["2015-16"], ["2016-17", "2017-18", "2018-19"], ["2019-20"],
                           ["2020-21", "2021-22"]),
    "tennis": SportCfg("tennis", ["2019"], ["2020", "2021", "2022"], ["2023"], ["2024"]),
}


def split_of(cfg: SportCfg, season: str) -> str:
    for name in ("burn_in", "train", "validation", "test"):
        if season in getattr(cfg, name):
            return name
    return "unused"


def _pcols(labels):
    return [f"p_{l}" for l in labels]


def _score(preds: pd.DataFrame, df: pd.DataFrame, labels, splits=("validation",)) -> float:
    p = preds.merge(df[["event_id", "outcome", "split"]], on="event_id")
    p = p[p.split.isin(splits)]
    P = p[_pcols(labels)].to_numpy()
    ok = np.isfinite(P).all(axis=1)
    return float(M.log_loss(P[ok], outcome_matrix(p.outcome[ok], labels)).mean())


def tune(df, labels, factory, grid: dict, refit_from=None, log=print):
    """Grid search scored ONLY on the validation split (test rows removed)."""
    dev = df[df.split != "test"]
    rows = []
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(grid.keys(), combo))
        m = factory(**kw)
        r = walk_forward(dev, m, refit_from=refit_from)
        rows.append({**kw, "val_log_loss": _score(r.preds, dev, labels)})
    res = pd.DataFrame(rows).sort_values("val_log_loss")
    best = {k: res.iloc[0][k] for k in grid}
    best = {k: (int(v) if isinstance(grid[k][0], int) else float(v)) for k, v in best.items()}
    log(f"   tuned {factory.__name__ if hasattr(factory, '__name__') else 'model'}: {best} "
        f"(val LL {res.iloc[0].val_log_loss:.4f})")
    return best, res


def model_specs(sport: str, df: pd.DataFrame, labels, log=print):
    """Return {name: factory()} with hyper-parameters tuned on validation where applicable."""
    val_start = df[df.split == "validation"].forecast_time.min()
    tuning = []
    specs = {"baseline:uniform": lambda: Uniform(labels), "baseline:base-rate": lambda: BaseRate(labels)}
    if sport == "tennis":
        def te(k_scale, surface_weight):
            return TennisElo(labels, k_scale=k_scale, surface_weight=surface_weight)
        best, res = tune(df, labels, te, {"k_scale": [200.0, 250.0, 320.0], "surface_weight": [0.0, 0.3, 0.6]}, log=log)
        specs["elo"] = lambda b=best: TennisElo(labels, **b, name="elo(tennis,surface)")
        tuning.append(res.assign(model="elo"))
    else:
        three = len(labels) == 3
        def elo(k, home_adv):
            return Elo(labels, k=k, home_adv=home_adv, mov=three, new_team="low" if sport == "football" else "mean")
        grid = {"k": [12.0, 20.0, 30.0], "home_adv": [40.0, 65.0, 90.0]} if three else \
            {"k": [4.0, 8.0, 14.0], "home_adv": [25.0, 50.0, 80.0]}
        best, res = tune(df, labels, elo, grid, log=log)
        specs["elo"] = lambda b=best: Elo(labels, **b, mov=three, new_team="low" if sport == "football" else "mean",
                                          name="elo")
        tuning.append(res.assign(model="elo"))
    if sport in ("football", "hockey"):
        def dc(xi, ridge):
            return DixonColes(labels, xi=xi, ridge=ridge)
        best, res = tune(df, labels, dc, {"xi": [0.001, 0.002, 0.004], "ridge": [0.01, 0.3, 3.0]},
                         refit_from=val_start, log=log)
        if sport == "football":
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            (ARTIFACTS_DIR / "football_dc_best.json").write_text(json.dumps(best))
        specs["dixon-coles"] = lambda b=best: DixonColes(labels, **b, name="dixon-coles")
        specs["poisson"] = lambda b=best: DixonColes(labels, **b, dc=False, name="poisson")
        tuning.append(res.assign(model="dixon-coles"))
        specs["bayes:gamma-poisson"] = lambda: GammaPoisson(labels, n_samples=0, name="bayes:gamma-poisson")
    if len(labels) == 2:
        def gr(home_adv, tau):
            return GaussianRating(labels, home_adv=home_adv, tau=tau)
        best, res = tune(df, labels, gr, {"home_adv": [0.1, 0.25] if sport != "tennis" else [0.0],
                                          "tau": [0.005, 0.02, 0.05]}, log=log)
        specs["bayes:gaussian-rating"] = lambda b=best: GaussianRating(labels, **b, name="bayes:gaussian-rating")
        tuning.append(res.assign(model="bayes:gaussian-rating"))
    specs["logistic"] = lambda: LogisticModel(labels, sport, name="logistic")
    specs["gradient-boosting"] = lambda: GBMModel(labels, sport, name="gradient-boosting")
    if sport == "tennis":
        specs["baseline:rank-logit"] = lambda: LogisticModel(labels, sport, feature_subset=["log_rank_ratio"],
                                                             name="baseline:rank-logit")
    if sport == "football":
        specs["market:pinnacle-opening"] = lambda: MarketPre(labels)
    if sport == "basketball":
        specs["external:fivethirtyeight"] = lambda: ExternalForecast(labels)
    return specs, pd.concat(tuning, ignore_index=True)


ENSEMBLE = {"football": ["elo", "dixon-coles", "bayes:gamma-poisson", "logistic"],
            "hockey": ["elo", "dixon-coles", "bayes:gaussian-rating", "logistic"],
            "basketball": ["elo", "bayes:gaussian-rating", "logistic"],
            "tennis": ["elo", "bayes:gaussian-rating", "logistic"]}


def benchmarks_post(sport, df, labels) -> dict[str, np.ndarray]:
    """Benchmarks computed from information available AFTER forecast_time (reference only)."""
    out = {}
    if sport == "football":
        out["market:pinnacle-closing"] = market_benchmark(df, labels, "pinnacle", "closing")
    if sport == "tennis":
        out["market:pinnacle-single"] = market_benchmark(df, labels, "pinnacle", "single")
    return out


def run_sport(sport: str, log=print) -> dict:
    t0 = time.time()
    cfg = CFG[sport]
    labels = LABELS[sport]
    df = load_matches(sport)
    df["split"] = df.season.map(lambda s: split_of(cfg, s))
    train_start = df[df.split == "train"].forecast_time.min()
    log(f"[{sport}] {len(df)} matches; splits: " +
        ", ".join(f"{k}={v}" for k, v in df.split.value_counts().items()))
    specs, tuning = model_specs(sport, df, labels, log=log)

    preds = {}
    timing = {}
    for name, fac in specs.items():
        r = walk_forward(df, fac(), refit_from=train_start, record_from=train_start)
        preds[name] = r.preds.set_index("event_id")[_pcols(labels)]
        timing[name] = dict(seconds=round(r.seconds, 1), refits=r.n_refits)
        log(f"   {name:28} {r.seconds:6.1f}s refits={r.n_refits}")
    base = df.set_index("event_id")
    idx = base.index[base.split.isin(["train", "validation", "test"])]
    for name, P in benchmarks_post(sport, df, labels).items():
        preds[name] = pd.DataFrame(P, index=df.event_id, columns=_pcols(labels)).loc[idx]
    comps = [preds[c].reindex(idx) for c in ENSEMBLE[sport]]
    stack = np.stack([c.to_numpy() for c in comps])
    preds["ensemble:mean"] = pd.DataFrame(np.nanmean(stack, axis=0), index=idx, columns=_pcols(labels))
    spread = pd.Series(np.nanstd(stack[:, :, :], axis=0).max(axis=1), index=idx, name="ensemble_spread")

    # calibration on validation -> applied everywhere (only test is reported as OOS)
    calibrators = {}
    for name in list(preds):
        if name.startswith(("baseline:uniform", "market:", "external:")):
            continue
        P = preds[name].reindex(idx)
        v = base.loc[idx, "split"] == "validation"
        Pv, ok = P[v].to_numpy(), np.isfinite(P[v].to_numpy()).all(axis=1)
        if ok.sum() < 100:
            continue
        ts = TemperatureScaler().fit(Pv[ok], outcome_matrix(base.loc[idx][v].outcome[ok], labels))
        calibrators[name] = ts.T
        Pa = P.to_numpy()
        good = np.isfinite(Pa).all(axis=1)
        out = np.full_like(Pa, np.nan)
        out[good] = ts.transform(Pa[good])
        preds[f"{name}+temp"] = pd.DataFrame(out, index=idx, columns=_pcols(labels))

    # long prediction table
    long = []
    for name, P in preds.items():
        d = P.reindex(idx).reset_index().rename(columns={"index": "event_id"})
        d["model"] = name
        long.append(d)
    long = pd.concat(long, ignore_index=True).merge(
        df[["event_id", "season", "split", "forecast_time", "event_time", "home", "away", "outcome"]], on="event_id")
    long["sport"] = sport

    # metrics
    mrows, comp_rows, rel_rows = [], [], []
    for (name, split), g in long.groupby(["model", "split"]):
        P = g[_pcols(labels)].to_numpy()
        ok = np.isfinite(P).all(axis=1)
        if ok.sum() == 0:
            continue
        Y = outcome_matrix(g.outcome[ok], labels)
        s = M.summarize(P[ok], Y, ordinal=cfg.ordinal)
        ll = M.log_loss(P[ok], Y)
        _, lo, hi = M.bootstrap_ci(ll, seed=GLOBAL_SEED)
        mrows.append(dict(sport=sport, model=name, split=split, coverage=float(ok.mean()), **s,
                          log_loss_lo=lo, log_loss_hi=hi, temperature=calibrators.get(name)))
        if split == "test":
            for k, lab in enumerate(labels):
                for b in M.reliability_bins(P[ok][:, k], Y[:, k]):
                    rel_rows.append(dict(sport=sport, model=name, outcome=lab, **b))
    metrics = pd.DataFrame(mrows)
    test = long[long.split == "test"]
    refs = ["baseline:base-rate"] + [n for n in preds if n.startswith(("market:", "external:"))]
    for ref in refs:
        R = test[test.model == ref].set_index("event_id")[_pcols(labels)]
        for name in preds:
            if name == ref:
                continue
            A = test[test.model == name].set_index("event_id")[_pcols(labels)]
            j = A.index.intersection(R.index)
            a, r = A.loc[j].to_numpy(), R.loc[j].to_numpy()
            ok = np.isfinite(a).all(axis=1) & np.isfinite(r).all(axis=1)
            if ok.sum() < 50:
                continue
            Y = outcome_matrix(base.loc[j].outcome[ok], labels)
            d = M.paired_diff_ci(M.log_loss(a[ok], Y), M.log_loss(r[ok], Y), seed=GLOBAL_SEED)
            comp_rows.append(dict(sport=sport, model=name, reference=ref, n=int(ok.sum()), **d))
    log(f"[{sport}] done in {time.time() - t0:.0f}s")
    return dict(predictions=long, metrics=metrics, comparisons=pd.DataFrame(comp_rows),
                reliability=pd.DataFrame(rel_rows), tuning=tuning.assign(sport=sport),
                spread=spread.reset_index().rename(columns={"index": "event_id"}).assign(sport=sport),
                timing=timing, calibrators=calibrators)


# ------------------------------------------------------------------ live HT validation (football)
def football_live_validation(fb_preds: pd.DataFrame, log=print) -> dict:
    """Use REAL half-time scores: does the live engine improve on prematch probabilities at HT?

    second-half goal share and the Bayesian prior weight are estimated on TRAIN seasons only.
    """
    from scipy.optimize import minimize_scalar
    from scipy.stats import poisson as pois

    from sportslab.live.engine import LiveEngine

    df = load_matches("football")
    cfg = CFG["football"]
    df["split"] = df.season.map(lambda s: split_of(cfg, s))
    # recover λ, μ by re-running DC walk-forward with rate capture
    lam_rows = []
    m = DixonColes(LABELS["football"], name="dc-rates")
    best = json.loads((ARTIFACTS_DIR / "football_dc_best.json").read_text()) if \
        (ARTIFACTS_DIR / "football_dc_best.json").exists() else {}
    m.xi, m.ridge = best.get("xi", m.xi), best.get("ridge", m.ridge)
    orig_predict = m.predict

    def predict(fx):
        P = orig_predict(fx)
        for eid, (l, u) in zip(fx.event_id, m.last_rates):
            lam_rows.append((eid, l, u))
        return P
    m.predict = predict
    walk_forward(df, m, refit_from=df[df.split == "train"].forecast_time.min())
    rates = pd.DataFrame(lam_rows, columns=["event_id", "lam", "mu"]).dropna()
    d = df.merge(rates, on="event_id").dropna(subset=["ht_home", "ht_away"])
    tr = d[d.split == "train"]
    tot = tr.home_score.sum() + tr.away_score.sum()
    s2 = float(((tr.home_score - tr.ht_home).sum() + (tr.away_score - tr.ht_away).sum()) / tot)
    eng = LiveEngine(football={"second_half_share": s2})

    def nll(w):
        e = LiveEngine(football={"second_half_share": s2, "prior_weight": w})
        rem = e.football_remaining_share(45)
        out = 0.0
        for r in tr.itertuples():
            mh = (w + r.ht_home) / (w + r.lam * (1 - rem))
            ma = (w + r.ht_away) / (w + r.mu * (1 - rem))
            out -= pois.logpmf(r.home_score - r.ht_home, r.lam * rem * mh)
            out -= pois.logpmf(r.away_score - r.ht_away, r.mu * rem * ma)
        return out
    w = float(minimize_scalar(nll, bounds=(0.5, 200), method="bounded").x)
    eng = LiveEngine(football={"second_half_share": s2, "prior_weight": w})
    labels = LABELS["football"]
    rows = []
    for r in d.itertuples():
        p_live, _, _ = eng.football_probs(r.lam, r.mu, 45.0, r.ht_home, r.ht_away)
        rows.append((r.event_id, r.split, *p_live))
    live = pd.DataFrame(rows, columns=["event_id", "split", "p_H", "p_D", "p_A"]).merge(
        d[["event_id", "outcome", "lam", "mu", "ht_home", "ht_away"]], on="event_id")
    pre = fb_preds[fb_preds.model == "dixon-coles"].set_index("event_id")[_pcols(labels)]
    out = {"second_half_share": s2, "prior_weight": w, "by_split": {}}
    for split, g in live.groupby("split"):
        Y = outcome_matrix(g.outcome, labels)
        Pl = g[_pcols(labels)].to_numpy()
        Pp = pre.loc[g.event_id].to_numpy()
        ok = np.isfinite(Pp).all(axis=1)
        out["by_split"][split] = dict(
            n=int(ok.sum()), live_ht=M.summarize(Pl[ok], Y[ok], ordinal=True),
            prematch=M.summarize(Pp[ok], Y[ok], ordinal=True),
            diff_ll=M.paired_diff_ci(M.log_loss(Pl[ok], Y[ok]), M.log_loss(Pp[ok], Y[ok]), seed=GLOBAL_SEED))
    log(f"[live-HT] s2={s2:.3f} prior_weight={w:.1f} test: live LL "
        f"{out['by_split'].get('test', {}).get('live_ht', {}).get('log_loss')} vs prematch "
        f"{out['by_split'].get('test', {}).get('prematch', {}).get('log_loss')}")
    return out, live, rates


def estimate_basketball_sigma(log=print) -> dict:
    """σ of the final NBA margin around the FiveThirtyEight-Elo-implied mean (2005-2015, home rows)."""
    from scipy.optimize import minimize_scalar
    from scipy.stats import norm

    from sportslab.config import RAW_DIR
    e = pd.read_csv(RAW_DIR / "basketball" / "basketball_nba_elo_hist.csv",
                    usecols=["year_id", "_iscopy", "game_location", "pts", "opp_pts", "forecast"])
    e = e[(e.year_id >= 2005) & (e._iscopy == 0) & (e.game_location == "H")]
    margin = (e.pts - e.opp_pts).to_numpy(float)
    z = norm.ppf(np.clip(e.forecast.to_numpy(float), 1e-4, 1 - 1e-4))
    nll = lambda s: -norm.logpdf(margin, s * z, s).sum()
    s = float(minimize_scalar(nll, bounds=(5, 25), method="bounded").x)
    log(f"[live-NBA] sigma_full estimated {s:.2f} pts from {len(e)} games (2005-2015)")
    return {"sigma_full": s, "n_games": int(len(e))}


def montecarlo_checks(fb_preds: pd.DataFrame, rates: pd.DataFrame, log=print) -> dict:
    from sportslab.models import montecarlo as mc
    from sportslab.models.goals import grid_to_probs, score_grid
    sample = rates.sample(50, random_state=GLOBAL_SEED)
    errs = []
    for r in sample.itertuples():
        exact = grid_to_probs(score_grid(r.lam, r.mu), ("H", "D", "A"))
        sim = mc.simulate_match(r.lam, r.mu, n=20000, seed=GLOBAL_SEED)
        errs.append(np.abs(np.array([sim.p_home, sim.p_draw, sim.p_away]) - exact).max())
    a = mc.simulate_match(1.4, 1.1, n=10000, seed=7)
    b = mc.simulate_match(1.4, 1.1, n=10000, seed=7)
    c = mc.simulate_match(1.4, 1.1, n=10000, seed=8)
    out = dict(max_abs_err_vs_analytic=float(np.max(errs)), mean_abs_err=float(np.mean(errs)),
               n_matches=len(sample), sims_per_match=20000, same_seed_identical=a == b,
               different_seed_differs=a != c)
    log(f"[MC] max |MC - analytic| = {out['max_abs_err_vs_analytic']:.4f}; reproducible={out['same_seed_identical']}")
    return out


def season_simulation(as_of: str = "2026-01-01", n: int = 10000, log=print) -> pd.DataFrame:
    """Simulate the rest of EPL 2025-26 from `as_of` with DC fitted on data known then; compare to actual."""
    from sportslab.data.pit import history_view
    from sportslab.models import montecarlo as mc
    df = load_matches("football")
    t = pd.Timestamp(as_of, tz="UTC")
    season = df[df.season == "2025-26"]
    played = history_view(season, t)
    remaining = season[~season.event_id.isin(played.event_id)]
    table = {}
    for r in played.itertuples():
        table[r.home] = table.get(r.home, 0) + {"H": 3, "D": 1, "A": 0}[r.outcome]
        table[r.away] = table.get(r.away, 0) + {"H": 0, "D": 1, "A": 3}[r.outcome]
    best = json.loads((ARTIFACTS_DIR / "football_dc_best.json").read_text())
    m = DixonColes(LABELS["football"], **best)
    m.refit(history_view(df, t), t)
    rates = np.array([m.rates(h, a) for h, a in zip(remaining.home, remaining.away)])
    sim = mc.simulate_season(remaining, rates, m.rho, table, n=n, seed=GLOBAL_SEED)
    actual = {}
    for r in season.itertuples():
        actual[r.home] = actual.get(r.home, 0) + {"H": 3, "D": 1, "A": 0}[r.outcome]
        actual[r.away] = actual.get(r.away, 0) + {"H": 0, "D": 1, "A": 3}[r.outcome]
    sim["actual_points"] = sim.team.map(actual)
    sim["actual_position"] = sim.actual_points.rank(ascending=False, method="min")
    sim["points_at_as_of"] = sim.team.map(table).fillna(0)
    sim["within_90pct"] = (sim.actual_points >= sim.p05_points) & (sim.actual_points <= sim.p95_points)
    log(f"[MC season] as_of {as_of}: {len(played)} played, {len(remaining)} simulated x{n}; "
        f"actual points inside 90% interval for {int(sim.within_90pct.sum())}/{len(sim)} teams")
    return sim.assign(as_of=as_of, n_sims=n, seed=GLOBAL_SEED)
