"""Post-hoc probability calibration fitted on a validation window only."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from sportslab.ev.metrics import log_loss


class TemperatureScaler:
    """p_k ∝ p_k^(1/T). T>1 softens over-confident forecasts, T<1 sharpens."""

    def __init__(self):
        self.T = 1.0

    @staticmethod
    def _apply(P, T):
        L = np.log(np.clip(P, 1e-12, 1)) / T
        L -= L.max(axis=1, keepdims=True)
        E = np.exp(L)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, P, Y):
        ok = np.isfinite(P).all(axis=1)
        res = minimize_scalar(lambda T: log_loss(self._apply(P[ok], T), Y[ok]).mean(),
                              bounds=(0.3, 5.0), method="bounded")
        self.T = float(res.x)
        return self

    def transform(self, P):
        return self._apply(P, self.T)


class IsotonicOVR:
    """One-vs-rest isotonic regression, renormalised (needs more data than temperature)."""

    def fit(self, P, Y):
        ok = np.isfinite(P).all(axis=1)
        self.iso = [IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(P[ok, k], Y[ok, k])
                    for k in range(P.shape[1])]
        return self

    def transform(self, P):
        Q = np.column_stack([iso.predict(P[:, k]) for k, iso in enumerate(self.iso)])
        return Q / Q.sum(axis=1, keepdims=True)
