"""Probabilistic scoring rules, calibration diagnostics and bootstrap intervals."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _check(P, Y):
    P = np.asarray(P, float)
    Y = np.asarray(Y, float)
    if P.shape != Y.shape:
        raise ValueError(f"shape mismatch {P.shape} vs {Y.shape}")
    return P, Y


def brier(P, Y) -> np.ndarray:
    """Per-row multi-class Brier score Σ_k (p_k - y_k)^2 (0 = perfect, 2 = worst)."""
    P, Y = _check(P, Y)
    return ((P - Y) ** 2).sum(axis=1)


def log_loss(P, Y) -> np.ndarray:
    """Per-row negative log-likelihood of the realised outcome."""
    P, Y = _check(P, Y)
    return -np.log(np.clip((P * Y).sum(axis=1), EPS, 1.0))


def rps(P, Y) -> np.ndarray:
    """Ranked probability score for ordered outcomes (e.g. H/D/A)."""
    P, Y = _check(P, Y)
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return ((cp - cy) ** 2)[:, :-1].sum(axis=1) / (P.shape[1] - 1)


def accuracy(P, Y) -> np.ndarray:
    P, Y = _check(P, Y)
    return (P.argmax(axis=1) == Y.argmax(axis=1)).astype(float)


def reliability_bins(p, y, n_bins: int = 10) -> list[dict]:
    """Bin forecasts of one outcome; compare mean forecast to observed frequency."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            out.append(dict(bin=b, lo=edges[b], hi=edges[b + 1], n=int(m.sum()),
                            mean_pred=float(p[m].mean()), obs_freq=float(y[m].mean())))
    return out


def ece(P, Y, n_bins: int = 10) -> float:
    """Expected calibration error, averaged over classes (top-label agnostic)."""
    P, Y = _check(P, Y)
    total = 0.0
    for k in range(P.shape[1]):
        bins = reliability_bins(P[:, k], Y[:, k], n_bins)
        total += sum(b["n"] * abs(b["mean_pred"] - b["obs_freq"]) for b in bins) / len(P)
    return total / P.shape[1]


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of a statistic over i.i.d. rows."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.array([stat(v[i]) for i in idx]) if stat is not np.mean else v[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(lo), float(hi)


def paired_diff_ci(loss_a, loss_b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """CI of mean(loss_a - loss_b). Negative => A better. Also returns P(diff<0) bootstrap share."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return dict(mean_diff=float(d.mean()), lo=float(lo), hi=float(hi), p_better=float((boots < 0).mean()))


def summarize(P, Y, ordinal: bool = False) -> dict:
    P, Y = _check(P, Y)
    out = dict(n=len(P), log_loss=float(log_loss(P, Y).mean()), brier=float(brier(P, Y).mean()),
               accuracy=float(accuracy(P, Y).mean()), ece=float(ece(P, Y)))
    if ordinal:
        out["rps"] = float(rps(P, Y).mean())
    return out
