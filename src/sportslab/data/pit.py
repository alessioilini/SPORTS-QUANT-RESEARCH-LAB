"""Point-in-time (PIT) access layer — the single gate against look-ahead bias.

Every column of the wide match frame belongs to one availability class:
  id    – identifiers / schedule, known at forecast_time
  pre   – facts with available_at <= forecast_time (opening odds, rankings, schedule)
  close – known at event_time (closing odds, single-snapshot odds of unknown time)
  post  – known only after the event (result, match stats) at result_available_at

`fixture_view(df)` exposes only id+pre columns -> what a model may use to predict.
`history_view(df, as_of)` exposes full rows of events whose result was known at as_of.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from sportslab.data.store import read_table

ID_COLS = ["event_id", "sport", "competition", "season", "event_time", "event_time_precision",
           "forecast_time", "home", "away", "neutral", "surface", "best_of", "round", "series", "tournament"]
POST_BASE = ["home_score", "away_score", "outcome", "ht_home", "ht_away", "regulation_draw", "decided_by",
             "completed", "result_available_at"]


class LookAheadError(RuntimeError):
    """Raised when data from after the decision time would be used."""


def column_class(col: str) -> str:
    if col in ID_COLS:
        return "id"
    if col.startswith("pre_") or col.startswith("ext_"):
        return "pre"
    if col.startswith("odds_") and ("_opening_" in col):
        return "pre"
    if col.startswith("odds_"):
        return "close"
    return "post"


def _to_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


@lru_cache(maxsize=8)
def _load_cached(sport: str, competition: str | None, db: str | None) -> pd.DataFrame:
    kw = {"path": db} if db else {}
    ev = read_table("events", where="WHERE sport = ?", params=(sport,), **kw)
    if competition:
        ev = ev[ev.competition == competition]
    ids = set(ev.event_id)
    res = read_table("results", **kw)
    res = res[res.event_id.isin(ids)].rename(columns={"available_at": "result_available_at"})
    df = ev.merge(res.drop(columns=["source_id", "retrieved_at", "value_kind", "detail"]), on="event_id", how="left")
    if "extra" in df and df.extra.notna().any():
        ex = df.extra.apply(lambda s: json.loads(s) if isinstance(s, str) else {})
        for k in ("surface", "best_of", "round", "series", "tournament"):
            df[k] = ex.apply(lambda d: d.get(k))
    df = df.drop(columns=["extra", "source_id", "retrieved_at", "value_kind", "forecast_time_rule"])

    st = read_table("team_match_stats", **kw)
    st = st[st.event_id.isin(ids)]
    if len(st):
        w = st.pivot_table(index="event_id", columns=["side", "stat"], values="value")
        w.columns = [f"{side}_{stat}" for side, stat in w.columns]
        df = df.merge(w, left_on="event_id", right_index=True, how="left")
    pp = read_table("player_prematch", **kw)
    pp = pp[pp.event_id.isin(ids)]
    if len(pp):
        pp = pp.merge(ev[["event_id", "forecast_time"]], on="event_id")
        pp = pp[_to_ts(pp.available_at) <= _to_ts(pp.forecast_time)]          # PIT guard
        w = pp.pivot_table(index="event_id", columns=["side", "stat"], values="value")
        w.columns = [f"pre_{side}_{stat}" for side, stat in w.columns]
        df = df.merge(w, left_on="event_id", right_index=True, how="left")
    od = read_table("odds", **kw)
    od = od[od.event_id.isin(ids)]
    if len(od):
        od = od.merge(ev[["event_id", "forecast_time"]], on="event_id")
        bad = (od.snapshot == "opening") & (_to_ts(od.available_at) > _to_ts(od.forecast_time))
        od = od[~bad]                                                          # PIT guard
        w = od.pivot_table(index="event_id", columns=["bookmaker", "snapshot", "selection"], values="price")
        w.columns = [f"odds_{b}_{s}_{sel}" for b, s, sel in w.columns]
        df = df.merge(w, left_on="event_id", right_index=True, how="left")
    ext = read_table("external_forecasts", **kw)
    ext = ext[ext.event_id.isin(ids)]
    if len(ext):
        w = ext.pivot_table(index="event_id", columns=["provider", "selection"], values="prob")
        w.columns = [f"ext_{p}_{s}" for p, s in w.columns]
        df = df.merge(w, left_on="event_id", right_index=True, how="left")
    for c in ("event_time", "forecast_time", "result_available_at"):
        df[c] = _to_ts(df[c])
    df = df.sort_values(["forecast_time", "event_time", "event_id"]).reset_index(drop=True)
    return df


def load_matches(sport: str, competition: str | None = None, db: str | None = None) -> pd.DataFrame:
    """Wide, chronologically sorted match frame (copy; safe to mutate)."""
    if competition is None:
        competition = {"football": "EPL", "hockey": "NHL", "basketball": "NBA", "tennis": "ATP"}.get(sport)
    return _load_cached(sport, competition, db).copy()


def fixture_view(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if column_class(c) in ("id", "pre")]
    return df[keep].copy()


def history_view(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rows whose results were public at `as_of` (all columns; past closing odds included)."""
    return df[df.result_available_at <= as_of].copy()


def assert_no_lookahead(used: pd.DataFrame, as_of: pd.Timestamp) -> None:
    if "result_available_at" in used and (used.result_available_at > as_of).any():
        n = int((used.result_available_at > as_of).sum())
        raise LookAheadError(f"{n} rows with results not yet available at {as_of}")


def outcome_matrix(outcomes: pd.Series, labels: tuple[str, ...]) -> np.ndarray:
    return np.stack([(outcomes == lab).to_numpy(float) for lab in labels], axis=1)
