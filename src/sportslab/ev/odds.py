"""Odds arithmetic for research and education.

Explains implied probability, bookmaker margin (overround), de-vigging and
expected value. Nothing here tells anyone to place a bet; EV is computed on
historical/virtual data to evaluate whether a *model* is better calibrated
than a market.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


def implied_probabilities(decimal_odds) -> np.ndarray:
    """Raw implied probability of each outcome: 1 / decimal odds (sums to > 1)."""
    o = np.asarray(decimal_odds, dtype=float)
    if not np.all(np.isfinite(o)):
        raise ValueError("odds missing (NaN) — price UNKNOWN, cannot compute implied probability")
    if np.any(o <= 1.0):
        raise ValueError("decimal odds must be > 1")
    return 1.0 / o


def overround(decimal_odds) -> float:
    """Booksum Σ1/o. 1.05 means a 5% margin built into the prices."""
    return float(implied_probabilities(decimal_odds).sum())


def margin(decimal_odds) -> float:
    return overround(decimal_odds) - 1.0


def devig(decimal_odds, method: str = "multiplicative") -> np.ndarray:
    """Remove the margin to obtain 'fair' probabilities summing to 1.

    multiplicative: p_i = q_i / Σq            (proportional; standard baseline)
    additive:       p_i = q_i - (Σq-1)/n      (equal margin per outcome)
    power:          p_i = q_i^k, k s.t. Σ=1   (more margin on longshots)
    shin:           Shin (1993) insider-trading model, solve for z
    """
    q = implied_probabilities(decimal_odds)
    s = q.sum()
    n = len(q)
    if method == "multiplicative":
        return q / s
    if method == "additive":
        p = q - (s - 1) / n
        if np.any(p <= 0):
            raise ValueError("additive devig produced non-positive probability")
        return p
    if method == "power":
        k = brentq(lambda k: (q ** k).sum() - 1.0, 1e-6, 50.0)
        return q ** k
    if method == "shin":
        def probs(z):
            return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / s) - z) / (2 * (1 - z))
        if abs(s - 1) < 1e-12:
            return q
        z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.999)
        return probs(z)
    raise ValueError(f"unknown method {method}")


def devig_frame(odds: np.ndarray, method: str = "multiplicative") -> np.ndarray:
    """Row-wise devig for an (n, k) array; rows with NaN stay NaN."""
    out = np.full(odds.shape, np.nan)
    for i, row in enumerate(odds):
        if np.all(np.isfinite(row)) and np.all(row > 1):
            out[i] = devig(row, method)
    return out


def expected_value(prob: float | np.ndarray, decimal_odds: float | np.ndarray) -> np.ndarray:
    """EV per 1 virtual unit staked: p*(o-1) - (1-p) = p*o - 1."""
    return np.asarray(prob) * np.asarray(decimal_odds) - 1.0


def break_even_probability(decimal_odds) -> np.ndarray:
    return 1.0 / np.asarray(decimal_odds, dtype=float)


@dataclass
class EVExplanation:
    odds: list[float]
    implied: list[float]
    overround: float
    fair_multiplicative: list[float]
    fair_shin: list[float]
    model_probs: list[float] | None
    ev: list[float] | None
    text: str


def explain(decimal_odds, model_probs=None, labels=None) -> EVExplanation:
    """Step-by-step explanation of the market maths (educational)."""
    o = np.asarray(decimal_odds, float)
    labels = labels or [f"outcome{i}" for i in range(len(o))]
    q = implied_probabilities(o)
    pm = devig(o, "multiplicative")
    ps = devig(o, "shin")
    lines = ["1) Implied probability q = 1/odds:"]
    lines += [f"   {lab}: 1/{oi:.3f} = {qi:.4f}" for lab, oi, qi in zip(labels, o, q)]
    lines.append(f"2) Overround Σq = {q.sum():.4f} -> bookmaker margin {100 * (q.sum() - 1):.2f}%")
    lines.append("3) Fair (de-vigged) probabilities:")
    lines += [f"   {lab}: multiplicative {a:.4f} | Shin {b:.4f}" for lab, a, b in zip(labels, pm, ps)]
    ev = None
    if model_probs is not None:
        p = np.asarray(model_probs, float)
        ev = expected_value(p, o)
        lines.append("4) Expected value per 1 virtual unit, EV = p*odds - 1 (p from a MODEL, uncertain):")
        lines += [f"   {lab}: {pi:.4f}*{oi:.3f}-1 = {e:+.4f}" for lab, pi, oi, e in zip(labels, p, o, ev)]
        lines.append("   A positive EV only means the model disagrees with the market; it is not evidence of "
                     "an edge unless the model is shown to be better calibrated out-of-sample.")
    return EVExplanation(list(o), list(q), float(q.sum()), list(pm), list(ps),
                         None if model_probs is None else list(map(float, model_probs)),
                         None if ev is None else list(map(float, ev)), "\n".join(lines))


def ev_sensitivity(prob: float, decimal_odds: float, prob_shifts=np.linspace(-0.05, 0.05, 11)) -> list[dict]:
    """How EV changes if the model probability is wrong by ±Δ (sensitivity analysis)."""
    return [{"delta_p": float(d), "prob": float(np.clip(prob + d, 0, 1)),
             "ev": float(expected_value(np.clip(prob + d, 0, 1), decimal_odds))} for d in prob_shifts]
