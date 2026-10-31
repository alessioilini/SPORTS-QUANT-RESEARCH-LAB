"""Rating models: Elo (team sports), surface Elo (tennis), Bayesian Gaussian rating."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm

from sportslab.models.base import Model


def _mov_multiplier(gd: float) -> float:
    """World-Football-Elo style goal-difference multiplier."""
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


class Elo(Model):
    """Elo with home advantage, optional margin-of-victory, season regression.

    3-way (football): Davidson (1970) tie model, P(D) driven by ν estimated
    online from the observed draw rate:  ν = 2d / (1 - d).
    """

    def __init__(self, labels, *, k=20.0, home_adv=60.0, season_regress=0.25, mov=False,
                 new_team="mean", name=None):
        super().__init__(labels)
        self.k, self.home_adv, self.season_regress, self.mov = k, home_adv, season_regress, mov
        self.new_team = new_team
        self.name = name or f"elo(k={k:g},h={home_adv:g})"
        self.r: dict[str, float] = {}
        self.season = None
        self.draws, self.games = 2.5, 10.0   # Beta-ish prior: 25% draws

    # ----------------------------------------------------------------- helpers
    def _get(self, team: str) -> float:
        if team not in self.r:
            if self.new_team == "low" and len(self.r) >= 10:
                self.r[team] = float(np.quantile(list(self.r.values()), 0.15))
            else:
                self.r[team] = 1500.0
        return self.r[team]

    def _maybe_new_season(self, season):
        if season is None or season == self.season:
            return
        if self.season is not None and self.r:
            mean = np.mean(list(self.r.values()))
            for t in self.r:
                self.r[t] = mean + (1 - self.season_regress) * (self.r[t] - mean)
        self.season = season

    def _nu(self) -> float:
        d = self.draws / self.games
        return 2 * d / (1 - d)

    def probs(self, rh: float, ra: float, neutral: int = 0) -> np.ndarray:
        diff = rh - ra + (0 if neutral else self.home_adv)
        if len(self.labels) == 2:
            ph = 1.0 / (1.0 + 10 ** (-diff / 400))
            return np.array([ph, 1 - ph])
        ph, pa = 10 ** (diff / 800), 10 ** (-diff / 800)
        pd_ = self._nu()
        z = ph + pa + pd_
        return np.array([ph / z, pd_ / z, pa / z])

    # ----------------------------------------------------------------- Model API
    def observe(self, rows: pd.DataFrame) -> None:
        for r in rows.itertuples(index=False):
            self._maybe_new_season(r.season)
            rh, ra = self._get(r.home), self._get(r.away)
            p = self.probs(rh, ra, getattr(r, "neutral", 0) or 0)
            expected = p[0] + (0.5 * p[1] if len(p) == 3 else 0.0)
            actual = {"H": 1.0, "D": 0.5, "A": 0.0}[r.outcome]
            k = self.k
            if self.mov and pd.notna(getattr(r, "home_score", np.nan)):
                k *= _mov_multiplier(r.home_score - r.away_score)
            delta = k * (actual - expected)
            self.r[r.home], self.r[r.away] = rh + delta, ra - delta
            if len(self.labels) == 3:
                self.draws += r.outcome == "D"
                self.games += 1

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(fixtures), len(self.labels)))
        for i, r in enumerate(fixtures.itertuples(index=False)):
            self._maybe_new_season(r.season)
            out[i] = self.probs(self._get(r.home), self._get(r.away), r.neutral or 0)
        return out


class TennisElo(Model):
    """Player Elo with FiveThirtyEight-style decaying K and surface blending."""

    def __init__(self, labels=("H", "A"), *, k_scale=250.0, k_offset=5.0, k_shape=0.4,
                 surface_weight=0.5, name=None):
        super().__init__(labels)
        self.k_scale, self.k_offset, self.k_shape, self.sw = k_scale, k_offset, k_shape, surface_weight
        self.name = name or f"tennis-elo(sw={surface_weight:g})"
        self.r: dict[str, float] = defaultdict(lambda: 1500.0)
        self.rs: dict[tuple[str, str], float] = defaultdict(lambda: 1500.0)
        self.n: dict[str, int] = defaultdict(int)
        self.ns: dict[tuple[str, str], int] = defaultdict(int)

    def _k(self, n: int) -> float:
        return self.k_scale / (n + self.k_offset) ** self.k_shape

    def rating(self, player: str, surface: str | None) -> float:
        base = self.r[player]
        if surface is None or self.sw == 0:
            return base
        return (1 - self.sw) * base + self.sw * self.rs[(player, surface)]

    def p_home(self, home, away, surface) -> float:
        d = self.rating(home, surface) - self.rating(away, surface)
        return 1.0 / (1.0 + 10 ** (-d / 400))

    def observe(self, rows: pd.DataFrame) -> None:
        for r in rows.itertuples(index=False):
            if not r.completed and str(r.decided_by) == "Walkover":
                continue
            s = 1.0 if r.outcome == "H" else 0.0
            for table, cnt, kh, ka in ((self.r, self.n, r.home, r.away),
                                        (self.rs, self.ns, (r.home, r.surface), (r.away, r.surface))):
                e = 1.0 / (1.0 + 10 ** (-(table[kh] - table[ka]) / 400))
                table[kh] += self._k(cnt[kh]) * (s - e)
                table[ka] -= self._k(cnt[ka]) * (s - e)
                cnt[kh] += 1
                cnt[ka] += 1

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        p = np.array([self.p_home(r.home, r.away, r.surface) for r in fixtures.itertuples(index=False)])
        return np.column_stack([p, 1 - p])


class GaussianRating(Model):
    """Bayesian rating with Gaussian posterior per team/player (TrueSkill-style
    moment matching, binary outcomes). Exposes posterior uncertainty.

    P(home wins) = Φ((μ_h + h − μ_a) / c),   c² = 2β² + σ_h² + σ_a²
    Between games each σ² grows by τ²·days (skill drift).
    """

    def __init__(self, labels=("H", "A"), *, mu0=0.0, sigma0=1.0, beta=1.0, tau=0.01, home_adv=0.25,
                 season_inflate=0.3, name=None):
        super().__init__(labels)
        self.mu0, self.sigma0, self.beta, self.tau, self.h = mu0, sigma0, beta, tau, home_adv
        self.season_inflate = season_inflate
        self.name = name or f"bayes-gauss(h={home_adv:g},tau={tau:g})"
        self.mu: dict[str, float] = {}
        self.var: dict[str, float] = {}
        self.last: dict[str, pd.Timestamp] = {}
        self.season = None
        self.last_sigma = None

    def _state(self, team, when: pd.Timestamp):
        if team not in self.mu:
            self.mu[team], self.var[team] = self.mu0, self.sigma0 ** 2
        elif team in self.last and when is not None:
            days = max(0.0, (when - self.last[team]).total_seconds() / 86400)
            self.var[team] = min(self.var[team] + self.tau ** 2 * days, self.sigma0 ** 2 * 4)
        return self.mu[team], self.var[team]

    def _maybe_new_season(self, season):
        if season is not None and season != self.season:
            if self.season is not None:
                for t in self.var:
                    self.var[t] += self.season_inflate ** 2
            self.season = season

    def observe(self, rows: pd.DataFrame) -> None:
        for r in rows.itertuples(index=False):
            self._maybe_new_season(r.season)
            when = r.event_time
            mh, vh = self._state(r.home, when)
            ma, va = self._state(r.away, when)
            h = 0.0 if (getattr(r, "neutral", 0) or 0) else self.h
            c = math.sqrt(2 * self.beta ** 2 + vh + va)
            sign = 1.0 if r.outcome == "H" else -1.0
            t = sign * (mh + h - ma) / c
            v = norm.pdf(t) / max(norm.cdf(t), 1e-12)
            w = v * (v + t)
            self.mu[r.home] = mh + sign * vh / c * v
            self.mu[r.away] = ma - sign * va / c * v
            self.var[r.home] = vh * max(1 - vh / c ** 2 * w, 1e-4)
            self.var[r.away] = va * max(1 - va / c ** 2 * w, 1e-4)
            self.last[r.home] = self.last[r.away] = when

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        p = np.empty(len(fixtures))
        sig = np.empty(len(fixtures))
        for i, r in enumerate(fixtures.itertuples(index=False)):
            self._maybe_new_season(r.season)
            mh, vh = self._state(r.home, None)
            ma, va = self._state(r.away, None)
            h = 0.0 if (r.neutral or 0) else self.h
            c = math.sqrt(2 * self.beta ** 2 + vh + va)
            p[i] = norm.cdf((mh + h - ma) / c)
            # uncertainty of the probability itself: delta-method on the latent diff
            sd_diff = math.sqrt(vh + va)
            p_lo = norm.cdf((mh + h - ma - sd_diff) / c)
            p_hi = norm.cdf((mh + h - ma + sd_diff) / c)
            sig[i] = (p_hi - p_lo) / 2
        self.last_sigma = sig
        return np.column_stack([p, 1 - p])
