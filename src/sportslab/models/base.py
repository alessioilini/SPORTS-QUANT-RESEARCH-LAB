"""Model interface + chronological walk-forward driver.

The driver is the only way models see data:
  * at each distinct forecast_time t it first feeds the model every result whose
    result_available_at <= t (in availability order) via `observe`,
  * optionally triggers `refit(history_view(df, t), t)` for batch models,
  * then calls `predict(fixture_view(block))` — fixtures carry no result columns.
Hence no model can use information from the future, by construction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportslab.data.pit import assert_no_lookahead, fixture_view, history_view

LABELS = {"football": ("H", "D", "A"), "hockey": ("H", "A"), "basketball": ("H", "A"), "tennis": ("H", "A")}


class Model:
    name = "model"
    needs_refit = False
    refit_every = pd.Timedelta(days=7)

    def __init__(self, labels: tuple[str, ...]):
        self.labels = labels

    def observe(self, rows: pd.DataFrame) -> None:  # online update with newly public results
        pass

    def refit(self, history: pd.DataFrame, as_of: pd.Timestamp) -> None:  # batch refit
        pass

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


@dataclass
class WalkForwardResult:
    model: str
    preds: pd.DataFrame       # event_id, forecast_time, p_<label>...
    seconds: float
    n_refits: int


def walk_forward(df: pd.DataFrame, model: Model, *, refit_from: pd.Timestamp | None = None,
                 record_from: pd.Timestamp | None = None) -> WalkForwardResult:
    t0 = time.time()
    df = df.sort_values(["forecast_time", "event_time", "event_id"]).reset_index(drop=True)
    results = df.dropna(subset=["outcome"]).sort_values(["result_available_at", "event_time", "event_id"])
    avail_ns = np.array([ts.value for ts in results.result_available_at], dtype=np.int64)
    ptr, n_refits, last_refit = 0, 0, None
    out = []
    for t, block in df.groupby("forecast_time", sort=True):
        hi = int(np.searchsorted(avail_ns, t.value, side="right"))
        if hi > ptr:
            newly = results.iloc[ptr:hi]
            assert_no_lookahead(newly, t)
            model.observe(newly)
            ptr = hi
        if model.needs_refit and refit_from is not None and t >= refit_from and (
                last_refit is None or t - last_refit >= model.refit_every):
            hist = history_view(df, t)
            assert_no_lookahead(hist, t)
            model.refit(hist, t)
            last_refit, n_refits = t, n_refits + 1
        P = model.predict(fixture_view(block))
        if record_from is None or t >= record_from:
            rec = pd.DataFrame(P, columns=[f"p_{lab}" for lab in model.labels])
            rec.insert(0, "event_id", block.event_id.to_numpy())
            rec.insert(1, "forecast_time", t)
            out.append(rec)
    preds = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return WalkForwardResult(model.name, preds, time.time() - t0, n_refits)
