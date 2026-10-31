"""Goal-count models: Poisson / Dixon-Coles (batch MLE) and Gamma-Poisson (Bayesian online).

Hockey: models are fitted on *regulation* scores (final score minus the OT/SO
winning goal). 2-way win probability = P(H_reg) + P(tie_reg)·λ/(λ+μ)
(OT/SO winner ∝ scoring rates — a stated modelling assumption).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from sportslab.models.base import Model

MAX_GOALS = 10


def regulation_scores(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    hs, as_ = df.home_score.to_numpy(float).copy(), df.away_score.to_numpy(float).copy()
    if "decided_by" in df:
        extra = df.decided_by.isin(["OT", "SO"]).to_numpy()
        home_won = hs > as_
        hs[extra & home_won] -= 1
        as_[extra & ~home_won] -= 1
    return hs, as_


def dc_tau(x, y, lam, mu, rho):
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def score_grid(lam: float, mu: float, rho: float = 0.0, max_goals: int = MAX_GOALS) -> np.ndarray:
    g = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    if rho:
        M[0, 0] *= 1 - lam * mu * rho
        M[0, 1] *= 1 + lam * rho
        M[1, 0] *= 1 + mu * rho
        M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    return M / M.sum()


def grid_to_probs(M: np.ndarray, labels, lam=None, mu=None) -> np.ndarray:
    ph, pd_, pa = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
    if len(labels) == 3:
        return np.array([ph, pd_, pa])
    share = lam / (lam + mu)
    return np.array([ph + pd_ * share, pa + pd_ * (1 - share)])


class DixonColes(Model):
    needs_refit = True
    refit_every = pd.Timedelta(days=7)

    def __init__(self, labels, *, xi=0.0019, window_days=3 * 365, dc=True, ridge=0.01, name=None):
        super().__init__(labels)
        self.xi, self.window, self.dc, self.ridge = xi, window_days, dc, ridge
        self.name = name or ("dixon-coles" if dc else "poisson")
        self.teams: list[str] = []
        self.params = None
        self.att, self.dfn = {}, {}
        self.gamma, self.rho = 0.25, 0.0
        self.last_rates: np.ndarray | None = None

    def refit(self, history: pd.DataFrame, as_of: pd.Timestamp) -> None:
        h = history.dropna(subset=["home_score", "away_score"])
        h = h[h.event_time >= as_of - pd.Timedelta(days=self.window)]
        if len(h) < 100:
            return
        teams = sorted(set(h.home) | set(h.away))
        ix = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi, ai = h.home.map(ix).to_numpy(), h.away.map(ix).to_numpy()
        x, y = regulation_scores(h)
        days = (as_of - h.event_time).dt.total_seconds().to_numpy() / 86400
        w = np.exp(-self.xi * days)
        x0 = np.zeros(2 * n + 2)
        for t, i in ix.items():  # warm start
            x0[i], x0[n + i] = self.att.get(t, 0.0), self.dfn.get(t, 0.0)
        x0[2 * n], x0[2 * n + 1] = self.gamma, self.rho if self.dc else 0.0
        ridge = self.ridge

        def f(theta):
            a, d, g, rho = theta[:n], theta[n:2 * n], theta[2 * n], theta[2 * n + 1]
            el = a[hi] - d[ai] + g
            em = a[ai] - d[hi]
            lam, mu = np.exp(el), np.exp(em)
            tau = np.clip(dc_tau(x, y, lam, mu, rho), 1e-10, None)
            ll = np.log(tau) + x * el - lam + y * em - mu
            # gradients of log tau
            dl = np.zeros_like(lam); dm = np.zeros_like(lam); dr = np.zeros_like(lam)
            m00 = (x == 0) & (y == 0); m01 = (x == 0) & (y == 1); m10 = (x == 1) & (y == 0); m11 = (x == 1) & (y == 1)
            dl[m00] = -mu[m00] * rho / tau[m00]; dm[m00] = -lam[m00] * rho / tau[m00]; dr[m00] = -lam[m00] * mu[m00] / tau[m00]
            dl[m01] = rho / tau[m01]; dr[m01] = lam[m01] / tau[m01]
            dm[m10] = rho / tau[m10]; dr[m10] = mu[m10] / tau[m10]
            dr[m11] = -1 / tau[m11]
            g_el = w * ((x - lam) + lam * dl)
            g_em = w * ((y - mu) + mu * dm)
            grad = np.zeros_like(theta)
            np.add.at(grad, hi, g_el); np.add.at(grad, ai, g_em)            # attack
            np.add.at(grad, n + ai, -g_el); np.add.at(grad, n + hi, -g_em)  # defence
            grad[2 * n] = g_el.sum()
            grad[2 * n + 1] = (w * dr).sum() if self.dc else 0.0
            nll = -(w * ll).sum() + ridge * (a @ a + d @ d)
            grad = -grad
            grad[:n] += 2 * ridge * a
            grad[n:2 * n] += 2 * ridge * d
            return nll, grad

        bounds = [(None, None)] * (2 * n + 1) + [(-0.25, 0.25) if self.dc else (0.0, 0.0)]
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        th = res.x
        self.teams = teams
        self.att = {t: th[i] for t, i in ix.items()}
        self.dfn = {t: th[n + i] for t, i in ix.items()}
        self.gamma, self.rho = float(th[2 * n]), float(th[2 * n + 1])
        self.params = th
        self.converged = bool(res.success)

    def _team(self, t):
        if t in self.att:
            return self.att[t], self.dfn[t]
        # unseen (e.g. promoted) team: mean of the 3 weakest known teams
        strength = sorted(self.att, key=lambda k: self.att[k] + self.dfn[k])[:3]
        return (float(np.mean([self.att[k] for k in strength])), float(np.mean([self.dfn[k] for k in strength])))

    def rates(self, home, away, neutral=0):
        ah, dh = self._team(home)
        aa, da = self._team(away)
        g = 0.0 if neutral else self.gamma
        return float(np.exp(ah - da + g)), float(np.exp(aa - dh))

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        if self.params is None:
            self.last_rates = np.full((len(fixtures), 2), np.nan)
            return np.full((len(fixtures), len(self.labels)), np.nan)
        out, rates = [], []
        for r in fixtures.itertuples(index=False):
            lam, mu = self.rates(r.home, r.away, r.neutral or 0)
            out.append(grid_to_probs(score_grid(lam, mu, self.rho if self.dc else 0.0), self.labels, lam, mu))
            rates.append((lam, mu))
        self.last_rates = np.array(rates)
        return np.array(out)


class GammaPoisson(Model):
    """Online Bayesian attack/defence rates with conjugate Gamma updates.

    λ_home = base_h · A_home · D_away,   μ_away = base_a · A_away · D_home
    Each multiplier has a Gamma(shape, rate) posterior (prior mean 1). Given the
    other factors at their posterior means, a Poisson observation updates
    shape += goals, rate += exposure (coordinate-wise conjugate approximation).
    Exponential forgetting (δ per update) keeps the posterior adaptive.
    """

    def __init__(self, labels, *, prior_strength=8.0, forget=0.97, season_forget=0.6, n_samples=200,
                 seed=0, name=None):
        super().__init__(labels)
        self.k0, self.forget, self.season_forget = prior_strength, forget, season_forget
        self.n_samples, self.seed = n_samples, seed
        self.name = name or f"bayes-gamma-poisson(f={forget:g})"
        self.A: dict[str, list[float]] = {}
        self.D: dict[str, list[float]] = {}
        self.base_h, self.base_a, self.nb = 1.5, 1.2, 20.0
        self.season = None
        self.last_rates = None
        self.last_sigma = None

    def _g(self, store, t):
        if t not in store:
            store[t] = [self.k0, self.k0]
        return store[t]

    def _decay(self, s, f):
        s[0] = self.k0 + f * (s[0] - self.k0)
        s[1] = self.k0 + f * (s[1] - self.k0)

    def _maybe_new_season(self, season):
        if season != self.season:
            if self.season is not None:
                for s in list(self.A.values()) + list(self.D.values()):
                    self._decay(s, self.season_forget)
            self.season = season

    def observe(self, rows: pd.DataFrame) -> None:
        hs, as_ = regulation_scores(rows)
        for r, x, y in zip(rows.itertuples(index=False), hs, as_):
            self._maybe_new_season(r.season)
            Ah, Dh, Aa, Da = self._g(self.A, r.home), self._g(self.D, r.home), self._g(self.A, r.away), self._g(self.D, r.away)
            for s in (Ah, Dh, Aa, Da):
                self._decay(s, self.forget)
            mAh, mDh, mAa, mDa = Ah[0] / Ah[1], Dh[0] / Dh[1], Aa[0] / Aa[1], Da[0] / Da[1]
            Ah[0] += x; Ah[1] += self.base_h * mDa
            Da[0] += x; Da[1] += self.base_h * mAh
            Aa[0] += y; Aa[1] += self.base_a * mDh
            Dh[0] += y; Dh[1] += self.base_a * mAa
            self.nb += 1
            self.base_h += (x - self.base_h) / min(self.nb, 500)
            self.base_a += (y - self.base_a) / min(self.nb, 500)

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        out, rates, sig = [], [], []
        rng = np.random.default_rng(self.seed)
        for r in fixtures.itertuples(index=False):
            self._maybe_new_season(r.season)
            Ah, Dh, Aa, Da = self._g(self.A, r.home), self._g(self.D, r.home), self._g(self.A, r.away), self._g(self.D, r.away)
            lam = self.base_h * (Ah[0] / Ah[1]) * (Da[0] / Da[1])
            mu = self.base_a * (Aa[0] / Aa[1]) * (Dh[0] / Dh[1])
            p = grid_to_probs(score_grid(lam, mu), self.labels, lam, mu)
            if self.n_samples:
                k = self.n_samples
                ls = self.base_h * rng.gamma(Ah[0], 1 / Ah[1], k) * rng.gamma(Da[0], 1 / Da[1], k)
                ms = self.base_a * rng.gamma(Aa[0], 1 / Aa[1], k) * rng.gamma(Dh[0], 1 / Dh[1], k)
                ph = [grid_to_probs(score_grid(a, b, max_goals=8), self.labels, a, b)[0] for a, b in zip(ls[:50], ms[:50])]
                sig.append(float(np.std(ph)))
            out.append(p)
            rates.append((lam, mu))
        self.last_rates, self.last_sigma = np.array(rates), np.array(sig) if sig else None
        return np.array(out)
