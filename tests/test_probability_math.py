"""Correctness of probability / odds / scoring maths against hand-computed values."""
import math

import numpy as np
import pytest

from sportslab.ev import metrics as M
from sportslab.ev.odds import (devig, ev_sensitivity, expected_value, explain, implied_probabilities, margin,
                               overround)
from sportslab.live import tennis as tn
from sportslab.models.calibration import TemperatureScaler
from sportslab.models.goals import dc_tau, grid_to_probs, score_grid


def test_implied_and_overround():
    o = [2.0, 3.5, 4.0]
    q = implied_probabilities(o)
    assert np.allclose(q, [0.5, 1 / 3.5, 0.25])
    assert overround(o) == pytest.approx(0.5 + 1 / 3.5 + 0.25)
    assert margin([2.0, 2.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        implied_probabilities([1.0, 2.0])


@pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
def test_devig_sums_to_one_and_keeps_order(method):
    o = np.array([1.9, 3.6, 4.4])
    p = devig(o, method)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert (np.diff(p) < 0).all()  # favourite stays most likely


def test_devig_values():
    o = [1.952, 1.952]                     # symmetric 2.46% margin
    assert np.allclose(devig(o, "multiplicative"), [0.5, 0.5])
    assert np.allclose(devig([2.0, 2.0], "shin"), [0.5, 0.5])
    # power devig puts relatively more margin on the longshot than multiplicative
    o = [1.25, 4.5]
    assert devig(o, "power")[1] < devig(o, "multiplicative")[1]


def test_expected_value_and_explain():
    assert expected_value(0.5, 2.1) == pytest.approx(0.05)
    assert expected_value(0.4, 2.0) == pytest.approx(-0.2)
    e = explain([2.0, 3.4, 3.9], model_probs=[0.5, 0.27, 0.23], labels=["H", "D", "A"])
    assert "Overround" in e.text and "not evidence of an edge" in e.text
    assert len(e.ev) == 3
    s = ev_sensitivity(0.5, 2.1)
    assert s[0]["ev"] < s[-1]["ev"]


def test_scoring_rules_known_values():
    P = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    Y = np.array([[1, 0, 0], [0, 1, 0]], float)
    assert M.brier(P, Y) == pytest.approx([0.09 + 0.04 + 0.01, 0.04 + 0.49 + 0.25])
    assert M.log_loss(P, Y) == pytest.approx([-math.log(0.7), -math.log(0.3)])
    # RPS example: forecast (1,0,0) vs outcome (0,0,1) = ((1)^2 + (1)^2)/2 = 1
    assert M.rps(np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))[0] == pytest.approx(1.0)
    assert M.accuracy(P, Y).tolist() == [1.0, 0.0]


def test_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50000)
    y = (rng.uniform(0, 1, 50000) < p).astype(float)
    P, Y = np.column_stack([p, 1 - p]), np.column_stack([y, 1 - y])
    assert M.ece(P, Y) < 0.01


def test_bootstrap_ci_contains_mean_and_paired_diff_sign():
    rng = np.random.default_rng(1)
    v = rng.normal(1.0, 1.0, 2000)
    m, lo, hi = M.bootstrap_ci(v, seed=0)
    assert lo < m < hi and lo < 1.0 < hi
    d = M.paired_diff_ci(v - 0.5, v, seed=0)
    assert d["hi"] < 0 and d["p_better"] == 1.0


def test_score_grid_and_dc_tau():
    M_ = score_grid(1.5, 1.1)
    assert M_.sum() == pytest.approx(1.0)
    p = grid_to_probs(M_, ("H", "D", "A"))
    assert p.sum() == pytest.approx(1.0) and p[0] > p[2]
    # rho < 0 raises the 0-0 and 1-1 mass (Dixon-Coles low-score correction)
    assert score_grid(1.2, 1.0, rho=-0.1)[0, 0] > score_grid(1.2, 1.0)[0, 0]
    lam = np.array([1.0]); mu = np.array([1.0])
    assert dc_tau(np.array([1]), np.array([1]), lam, mu, 0.1)[0] == pytest.approx(0.9)
    # two-way (hockey): ties split proportionally to rates
    p2 = grid_to_probs(score_grid(3.0, 3.0), ("H", "A"), 3.0, 3.0)
    assert p2 == pytest.approx([0.5, 0.5])


def test_temperature_scaler():
    rng = np.random.default_rng(3)
    true = rng.dirichlet([2, 2, 2], 4000)
    Y = np.array([rng.multinomial(1, p) for p in true], float)
    over = TemperatureScaler._apply(true, 0.5)     # make over-confident
    ts = TemperatureScaler().fit(over, Y)
    assert 1.6 < ts.T < 2.5                       # recovers ~2
    assert M.log_loss(ts.transform(over), Y).mean() < M.log_loss(over, Y).mean()


def test_tennis_markov_closed_forms():
    assert tn.p_game(0.5) == pytest.approx(0.5)
    p = 0.6   # closed form for P(server holds)
    d = p * p / (p * p + (1 - p) ** 2)
    closed = p ** 4 * (1 + 4 * (1 - p) + 10 * (1 - p) ** 2) + 20 * p ** 3 * (1 - p) ** 3 * d
    assert tn.p_game(p) == pytest.approx(closed, rel=1e-12)
    assert tn.p_match(0.64, 0.64, 3) == pytest.approx(0.5)
    assert tn.p_match(0.66, 0.62, 5) > tn.p_match(0.66, 0.62, 3) > 0.5   # longer format favours stronger
    assert tn.p_match(0.6, 0.6, 3, sh=2) == 1.0
    sh, sa = tn.serve_probs_from_prematch(0.75, 3)
    assert tn.p_match(sh, sa, 3) == pytest.approx(0.75, abs=1e-9)
    assert tn.p_tiebreak(0.65, 0.65, 0, 0, True) == pytest.approx(0.5)


def test_missing_price_is_rejected_not_guessed():
    with pytest.raises(ValueError, match="UNKNOWN"):
        implied_probabilities([2.0, float("nan"), 3.0])
