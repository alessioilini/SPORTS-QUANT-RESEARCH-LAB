"""Append-only virtual journal + forecast log in SQLite.

virtual_journal rows cannot be UPDATEd or DELETEd (enforced by SQLite triggers).
forecast_log stores each forecast with inputs and sources at creation time; only
its resolution fields are filled later (resolve_forecast refuses to touch
anything else and refuses to re-resolve).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sportslab.config import DB_PATH
from sportslab.ev import metrics as M
from sportslab.data.store import connect, init_db


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_ledger(led: pd.DataFrame, run_id: str, model: str, db=DB_PATH) -> int:
    init_db(db)
    rows = [(run_id, _now(), r.event_id, str(r.forecast_time), model, r.selection, float(r.model_prob),
             float(r.price_taken), None if pd.isna(getattr(r, "closing_price", np.nan)) else float(r.closing_price),
             float(r.stake), int(r.won), float(r.pnl), float(r.commission), float(r.bankroll_after),
             "VIRTUAL research simulation") for r in led.itertuples()]
    with connect(db) as con:
        con.executemany(
            "INSERT INTO virtual_journal(run_id, created_at, event_id, forecast_time, model, selection, model_prob,"
            " price_taken, closing_price, stake_virtual, outcome_won, pnl_virtual, commission_virtual,"
            " bankroll_after_virtual, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def read_journal(run_id: str | None = None, db=DB_PATH) -> pd.DataFrame:
    with connect(db) as con:
        q = "SELECT * FROM virtual_journal" + (" WHERE run_id = ?" if run_id else "") + " ORDER BY entry_id"
        return pd.read_sql_query(q, con, params=(run_id,) if run_id else ())


def log_forecast(*, mode: str, sport: str, event_id: str, model: str, as_of: str, probs: dict | None,
                 status: str, inputs: dict, sources: list, db=DB_PATH) -> str:
    init_db(db)
    fid = str(uuid.uuid4())
    with connect(db) as con:
        con.execute("INSERT INTO forecast_log(forecast_id, created_at, mode, sport, event_id, model, as_of,"
                    " probs_json, status, inputs_json, sources_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (fid, _now(), mode, sport, event_id, model, as_of, json.dumps(probs), status,
                     json.dumps(inputs, default=str), json.dumps(sources, default=str)))
    return fid


def resolve_forecast(forecast_id: str, outcome: str, db=DB_PATH) -> dict:
    with connect(db) as con:
        row = con.execute("SELECT probs_json, resolved_outcome FROM forecast_log WHERE forecast_id=?",
                          (forecast_id,)).fetchone()
        if row is None:
            raise KeyError(forecast_id)
        if row[1] is not None:
            raise ValueError("forecast already resolved; history is immutable")
        probs = json.loads(row[0]) if row[0] else None
        brier = logloss = None
        if probs:
            labels = list(probs)
            P = np.array([[probs[k] for k in labels]])
            Y = np.array([[1.0 if k == outcome else 0.0 for k in labels]])
            brier, logloss = float(M.brier(P, Y)[0]), float(M.log_loss(P, Y)[0])
        con.execute("UPDATE forecast_log SET resolved_outcome=?, resolved_at=?, brier=?, logloss=? "
                    "WHERE forecast_id=? AND resolved_outcome IS NULL", (outcome, _now(), brier, logloss, forecast_id))
    return dict(forecast_id=forecast_id, outcome=outcome, brier=brier, log_loss=logloss)


def new_run_id(prefix="bt") -> str:
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
