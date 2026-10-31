"""Reproducible Monte Carlo simulation (numpy PCG64 with explicit seeds)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportslab.models.goals import score_grid


def rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


@dataclass
class MatchSim:
    p_home: float
    p_draw: float
    p_away: float
    p_over_2_5: float
    mc_se: float
    n: int
    seed: int


def simulate_match(lam: float, mu: float, rho: float = 0.0, n: int = 20000, seed: int = 0) -> MatchSim:
    """Sample exact scorelines from the (Dixon-Coles) grid."""
    M = score_grid(lam, mu, rho)
    k = M.shape[0]
    idx = rng(seed).choice(k * k, size=n, p=M.ravel())
    h, a = np.divmod(idx, k)
    ph, pd_, pa = (h > a).mean(), (h == a).mean(), (h < a).mean()
    return MatchSim(float(ph), float(pd_), float(pa), float(((h + a) > 2.5).mean()),
                    float(np.sqrt(ph * (1 - ph) / n)), n, seed)


def simulate_season(fixtures: pd.DataFrame, rates: np.ndarray, rho: float, table: dict[str, int],
                    n: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Simulate remaining fixtures; return per-team points distribution & position probabilities.

    fixtures: home, away (remaining games); rates: (m, 2) λ, μ per fixture; table: current points.
    """
    teams = sorted(set(table) | set(fixtures.home) | set(fixtures.away))
    ix = {t: i for i, t in enumerate(teams)}
    g = rng(seed)
    pts = np.tile(np.array([table.get(t, 0) for t in teams], float), (n, 1))
    for (h, a), (lam, mu) in zip(fixtures[["home", "away"]].itertuples(index=False), rates):
        M = score_grid(lam, mu, rho)
        k = M.shape[0]
        s = g.choice(k * k, size=n, p=M.ravel())
        x, y = np.divmod(s, k)
        pts[:, ix[h]] += np.where(x > y, 3, np.where(x == y, 1, 0))
        pts[:, ix[a]] += np.where(y > x, 3, np.where(x == y, 1, 0))
    # tiny random tie-break so positions are well defined
    order = np.argsort(-(pts + g.random(pts.shape) * 1e-3), axis=1)
    pos = np.empty_like(order)
    rows = np.arange(n)[:, None]
    pos[rows, order] = np.arange(len(teams))[None, :] + 1
    out = pd.DataFrame({"team": teams, "exp_points": pts.mean(0), "p05_points": np.quantile(pts, 0.05, 0),
                        "p95_points": np.quantile(pts, 0.95, 0), "p_title": (pos == 1).mean(0),
                        "p_top4": (pos <= 4).mean(0), "p_relegation": (pos >= len(teams) - 2).mean(0),
                        "exp_position": pos.mean(0)})
    return out.sort_values("exp_points", ascending=False).reset_index(drop=True)
