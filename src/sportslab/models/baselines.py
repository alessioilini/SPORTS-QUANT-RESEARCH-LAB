"""Baselines every model must beat, plus external/market benchmarks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportslab.ev.odds import devig_frame
from sportslab.models.base import Model


class Uniform(Model):
    name = "baseline:uniform"

    def predict(self, fixtures):
        return np.full((len(fixtures), len(self.labels)), 1 / len(self.labels))


class BaseRate(Model):
    """Historical outcome frequencies known at forecast time (Laplace-smoothed)."""
    name = "baseline:base-rate"

    def __init__(self, labels):
        super().__init__(labels)
        self.counts = np.ones(len(labels))

    def observe(self, rows):
        for lab in rows.outcome:
            self.counts[self.labels.index(lab)] += 1

    def predict(self, fixtures):
        return np.tile(self.counts / self.counts.sum(), (len(fixtures), 1))


class MarketPre(Model):
    """De-vigged market probabilities from odds known at forecast_time (opening lines)."""

    def __init__(self, labels, book="pinnacle", fallback="market_avg", method="multiplicative"):
        super().__init__(labels)
        self.book, self.fallback, self.method = book, fallback, method
        self.name = f"market:{book}-opening"

    def predict(self, fixtures):
        def arr(book):
            cols = [f"odds_{book}_opening_{lab}" for lab in self.labels]
            if not all(c in fixtures for c in cols):
                return np.full((len(fixtures), len(self.labels)), np.nan)
            return fixtures[cols].to_numpy(float, copy=True)
        o = arr(self.book)
        fb = arr(self.fallback)
        miss = ~np.isfinite(o).all(axis=1)
        o[miss] = fb[miss]
        return devig_frame(o, self.method)


class ExternalForecast(Model):
    """Third-party published probabilities (e.g. FiveThirtyEight) — OBSERVED, not ours."""

    def __init__(self, labels, provider="fivethirtyeight"):
        super().__init__(labels)
        self.provider = provider
        self.name = f"external:{provider}"

    def predict(self, fixtures):
        cols = [f"ext_{self.provider}_{lab}" for lab in self.labels]
        P = fixtures[cols].to_numpy(float)
        return P / P.sum(axis=1, keepdims=True)


def market_benchmark(df: pd.DataFrame, labels, book: str, snapshot: str, method="multiplicative") -> np.ndarray:
    """Benchmark from odds that may be observed AFTER forecast_time (closing / single snapshot).
    Used only as a reference point — never as a model input."""
    cols = [f"odds_{book}_{snapshot}_{lab}" for lab in labels]
    if not all(c in df for c in cols):
        return np.full((len(df), len(labels)), np.nan)
    return devig_frame(df[cols].to_numpy(float), method)
