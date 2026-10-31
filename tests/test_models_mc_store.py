"""Models on synthetic data with known truth, Monte Carlo reproducibility, storage & journal integrity."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from sportslab.models import montecarlo as mc
from sportslab.models.base import walk_forward
from sportslab.models.goals import DixonColes, GammaPoisson, grid_to_probs, score_grid
from sportslab.models.ratings import Elo, GaussianRating
from tests.conftest import synthetic_league

L3 = ("H", "D", "A")


def test_poisson_mle_matches_statsmodels_glm_and_truth():
    """Our analytic-gradient MLE must agree with an independent GLM fit (oracle) and track the truth."""
    import statsmodels.api as sm
    df, truth = synthetic_league(n_teams=10, n_rounds=120, seed=3)
    m = DixonColes(L3, xi=0.0, window_days=10_000, ridge=1e-6, dc=False)
    m.refit(df, df.result_available_at.max())
    n = 10
    rows = [dict(g=r.home_score, a=r.home, d=r.away, h=1) for r in df.itertuples()] + \
           [dict(g=r.away_score, a=r.away, d=r.home, h=0) for r in df.itertuples()]
    X = pd.DataFrame(rows)
    D = sm.add_constant(pd.get_dummies(X[["a", "d"]], drop_first=True).astype(float).assign(h=X.h))
    fit = sm.GLM(X.g, D, family=sm.families.Poisson()).fit()
    glm_att = np.r_[0, [fit.params[f"a_T{i:02d}"] for i in range(1, n)]]
    glm_def = np.r_[0, [-fit.params[f"d_T{i:02d}"] for i in range(1, n)]]
    att = np.array([m.att[f"T{i:02d}"] for i in range(n)])
    dfn = np.array([m.dfn[f"T{i:02d}"] for i in range(n)])
    # identical up to the identifiability shift
    np.testing.assert_allclose(att - att[0], glm_att, atol=2e-3)
    np.testing.assert_allclose(dfn - dfn[0], glm_def, atol=2e-3)
    assert m.gamma == pytest.approx(fit.params["h"], abs=2e-3)
    # and both track the generating truth (finite-sample noise: 120 games/team)
    assert np.corrcoef(att, truth["att"])[0, 1] > 0.9
    assert np.corrcoef(dfn, truth["dfn"])[0, 1] > 0.75
    assert m.gamma == pytest.approx(truth["home_adv"], abs=0.08)


def test_dixon_coles_rho_near_zero_without_dependence():
    df, _ = synthetic_league(n_teams=10, n_rounds=120, seed=3)
    m = DixonColes(L3, xi=0.0, window_days=10_000, ridge=0.001)
    m.refit(df, df.result_available_at.max())
    assert abs(m.rho) < 0.1


def test_models_beat_uniform_on_synthetic_league():
    from sportslab.ev.metrics import log_loss
    from sportslab.data.pit import outcome_matrix
    df, _ = synthetic_league(n_teams=12, n_rounds=80, seed=4)
    late = df.forecast_time > df.forecast_time.quantile(0.5)
    Y = outcome_matrix(df[late].outcome, L3)
    for m in (Elo(L3, k=20, home_adv=60, mov=True), DixonColes(L3), GammaPoisson(L3, n_samples=0)):
        p = walk_forward(df, m, refit_from=df.forecast_time.min()).preds.set_index("event_id").loc[df[late].event_id]
        assert log_loss(p[["p_H", "p_D", "p_A"]].to_numpy(), Y).mean() < np.log(3) - 0.02, m.name


def test_gaussian_rating_orders_teams_and_reports_uncertainty():
    df, truth = synthetic_league(n_teams=8, n_rounds=80, seed=6, sport="hockey")
    m = GaussianRating(("H", "A"), tau=0.0)
    m.observe(df)
    strength = truth["att"] + truth["dfn"]
    est = np.array([m.mu[f"T{i:02d}"] for i in range(8)])
    assert np.corrcoef(est, strength)[0, 1] > 0.7
    m.predict(df.head(3))
    assert (m.last_sigma > 0).all()


def test_montecarlo_reproducible_and_matches_analytic():
    a = mc.simulate_match(1.6, 1.0, rho=-0.05, n=50000, seed=11)
    b = mc.simulate_match(1.6, 1.0, rho=-0.05, n=50000, seed=11)
    c = mc.simulate_match(1.6, 1.0, rho=-0.05, n=50000, seed=12)
    assert a == b and a != c
    exact = grid_to_probs(score_grid(1.6, 1.0, -0.05), L3)
    assert abs(a.p_home - exact[0]) < 4 * a.mc_se
    assert abs(a.p_draw - exact[1]) < 0.01


def test_season_simulation_reproducible():
    fx = pd.DataFrame({"home": ["A", "B", "C"], "away": ["B", "C", "A"]})
    rates = np.array([[1.5, 1.0], [1.2, 1.2], [1.0, 1.4]])
    s1 = mc.simulate_season(fx, rates, 0.0, {"A": 10, "B": 9, "C": 3}, n=2000, seed=5)
    s2 = mc.simulate_season(fx, rates, 0.0, {"A": 10, "B": 9, "C": 3}, n=2000, seed=5)
    pd.testing.assert_frame_equal(s1, s2)
    assert s1.p_title.sum() == pytest.approx(1.0)


def test_store_schema_constraints(tmp_db):
    from sportslab.data.store import connect, init_db
    init_db(tmp_db)
    with pytest.raises(sqlite3.IntegrityError):
        with connect(tmp_db) as con:   # forecast_time after event_time is rejected
            con.execute("INSERT INTO events(event_id, sport, event_time, event_time_precision, forecast_time, home, away)"
                        " VALUES ('X','football','2024-01-01T00:00:00Z','day','2024-01-02T00:00:00Z','A','B')")


def test_virtual_journal_is_append_only(tmp_db):
    from sportslab.backtest.journal import append_ledger, read_journal
    from sportslab.data.store import connect
    led = pd.DataFrame([dict(event_id="E1", forecast_time="2024-01-01", selection="H", model_prob=0.5,
                             price_taken=2.1, closing_price=2.0, stake=1.0, won=1, pnl=1.1, commission=0.0,
                             bankroll_after=101.1)])
    assert append_ledger(led, "run-1", "m", db=tmp_db) == 1
    j = read_journal("run-1", db=tmp_db)
    assert len(j) == 1 and j.pnl_virtual.iloc[0] == pytest.approx(1.1)
    for sql in ("UPDATE virtual_journal SET pnl_virtual=999", "DELETE FROM virtual_journal"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with connect(tmp_db) as con:
                con.execute(sql)


def test_forecast_log_resolution_is_immutable(tmp_db):
    from sportslab.backtest.journal import log_forecast, resolve_forecast
    fid = log_forecast(mode="prematch", sport="football", event_id="E", model="m", as_of="t",
                       probs={"H": 0.5, "D": 0.3, "A": 0.2}, status="OK", inputs={}, sources=["s"], db=tmp_db)
    r = resolve_forecast(fid, "H", db=tmp_db)
    assert r["brier"] == pytest.approx(0.25 + 0.09 + 0.04)
    with pytest.raises(ValueError):
        resolve_forecast(fid, "A", db=tmp_db)


def test_poisson_matches_betting_backtester_reference_implementation():
    """Independent library (R1ch1k/betting-backtester) fits the same penalised Poisson objective."""
    from betting_backtester.dixon_coles import DixonColesModel, TrainingMatch
    df, _ = synthetic_league(n_teams=10, n_rounds=60, seed=8)
    anchor = df.result_available_at.max()
    ours = DixonColes(L3, xi=0.0, window_days=10_000, ridge=0.5, dc=False)
    ours.refit(df, anchor)
    ref = DixonColesModel(l2_reg=0.5, decay_rate=0.0)
    ref.fit([TrainingMatch(home_team=r.home, away_team=r.away, home_goals=int(r.home_score),
                           away_goals=int(r.away_score), settled_at=r.result_available_at.to_pydatetime())
             for r in df.itertuples()], anchor.to_pydatetime())
    for h, a in [("T00", "T01"), ("T05", "T09"), ("T03", "T07")]:
        i, j = ref._team_index[h], ref._team_index[a]
        lam_ref = np.exp(ref._attack[i] - ref._defence[j] + ref._home_advantage)
        mu_ref = np.exp(ref._attack[j] - ref._defence[i])
        lam, mu = ours.rates(h, a)
        assert lam == pytest.approx(lam_ref, rel=2e-3) and mu == pytest.approx(mu_ref, rel=2e-3)
        p = ref.predict(h, a)
        ours_p = grid_to_probs(score_grid(lam, mu), L3)
        assert np.allclose(ours_p, [p.home, p.draw, p.away], atol=5e-3)   # grid truncation 6 vs 10 goals
