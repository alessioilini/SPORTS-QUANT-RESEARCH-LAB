#!/usr/bin/env python3
"""Shared helpers for the Superforecasting command-line tools."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any

LOG_PATH_ENV = "FORECAST_LOG"
LOG_PATH_FALLBACK = os.path.join("~", ".superforecasting", "forecasts.jsonl")


def default_log_path() -> str:
    """Return the ledger path to use when the caller gives no explicit ``--log``.

    Resolution order: the ``FORECAST_LOG`` environment variable when set to a
    non-blank value, otherwise ``~/.superforecasting/forecasts.jsonl``.  An
    explicit ``--log`` always wins over both.

    The ledger deliberately lives *outside* the skill directory so the toolkit
    never writes into its own checkout: it keeps working when the skill is
    installed read-only or symlinked into a harness's skills directory, and one
    env var points every harness on the machine at the same record.

    Resolved at call time, not import time, so setting the variable after import
    takes effect.
    """
    configured = os.environ.get(LOG_PATH_ENV, "")
    if configured.strip():
        return os.path.expanduser(configured)
    return os.path.expanduser(LOG_PATH_FALLBACK)


def validate_probability(p: float, name: str = "probability") -> float:
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p}")
    return p


def validate_open_probability(p: float, name: str = "probability") -> float:
    if not math.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {p}")
    return p


def validate_outcome(o: int, name: str = "outcome") -> int:
    if o not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {o}")
    return o


def validate_positive(x: float, name: str) -> float:
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be > 0, got {x}")
    return x


def validate_non_negative(x: float, name: str) -> float:
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be >= 0, got {x}")
    return x


def brier_score(prob: float, outcome: int) -> float:
    validate_probability(prob, "prob")
    validate_outcome(outcome)
    return (prob - outcome) ** 2


def ranked_probability_score(dist: list[float], outcome_index: int) -> float:
    """Score an ordered categorical forecast using the un-normalized RPS.

    This is the sum of squared CDF differences through the penultimate bin.
    For two bins it is exactly this project's ``(p - outcome) ** 2`` Brier
    convention.
    """
    if len(dist) < 2:
        raise ValueError("dist must contain at least two probabilities")
    if not 0 <= outcome_index < len(dist):
        raise ValueError(f"outcome_index must be in [0, {len(dist) - 1}], got {outcome_index}")
    probabilities = [validate_probability(float(p), "dist probability") for p in dist]
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("dist probabilities must sum to 1 within 1e-9")
    if len(probabilities) == 2:
        return brier_score(probabilities[1], outcome_index)
    cumulative = 0.0
    score = 0.0
    for index, probability in enumerate(probabilities[:-1]):
        cumulative += probability
        observed_cumulative = 1.0 if outcome_index <= index else 0.0
        score += (cumulative - observed_cumulative) ** 2
    return score


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if len(raw) == 10:
        dt = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def age_days(stamp: str, as_of: datetime) -> float:
    dt = parse_datetime(stamp)
    if dt is None:
        return 0.0
    return max(0.0, (as_of - dt).total_seconds() / 86400.0)


def time_weighted_brier_score(snapshot: dict[str, Any]) -> float:
    """Return a resolved snapshot's duration-weighted trajectory Brier score.

    Each probability in ``prob_history`` is a right-continuous step held until
    the next history entry; the final step ends at ``resolved``. Unusable
    trajectories deliberately degrade to the standard latest-probability Brier.
    """
    prob = validate_probability(float(snapshot["prob"]), "prob")
    outcome = validate_outcome(int(snapshot["outcome"]))
    standard = brier_score(prob, outcome)
    history = snapshot.get("prob_history")
    if not isinstance(history, list) or len(history) < 2:
        return standard
    try:
        resolved = parse_datetime(snapshot.get("resolved"))
        if resolved is None:
            return standard
        weighted_error = 0.0
        total_duration = 0.0
        for index, entry in enumerate(history):
            segment_prob = validate_probability(float(entry["prob"]), "prob_history prob")
            start = entry.get("at")
            end_value = history[index + 1].get("at") if index + 1 < len(history) else snapshot.get("resolved")
            end = parse_datetime(end_value)
            if not start or end is None:
                return standard
            duration = age_days(str(start), end)
            weighted_error += duration * brier_score(segment_prob, outcome)
            total_duration += duration
    except (KeyError, TypeError, ValueError):
        return standard
    return standard if total_duration == 0 else weighted_error / total_duration


def append_jsonl(path: str, rec: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def fold_forecast_log(path: str) -> list[dict[str, Any]]:
    """Return latest forecast snapshots from legacy rows plus v2 event rows.

    An ``update`` event revises an open forecast's ``prob`` in place; the full
    trajectory is preserved in a ``prob_history`` list on the snapshot (original
    first, each entry ``{prob, at, note}``). ``prob_history`` is attached only
    when at least one update event exists, so logs without updates fold exactly
    as before. Resolution/scoring use the latest (most recently updated) prob.
    """
    forecasts: dict[int, dict[str, Any]] = {}
    histories: dict[int, list[dict[str, Any]]] = {}
    order: list[int] = []

    for rec in read_jsonl(path):
        event = rec.get("event", "forecast")
        if event == "supersede":
            fid = int(rec["id"])
            if fid not in forecasts:
                continue
            row = forecasts[fid]
            if row.get("status") in {"resolved", "superseded"}:
                continue
            row["status"] = "superseded"
            row["superseded"] = rec.get("superseded")
            row["reason"] = rec.get("reason", "")
            if rec.get("replaced_by") is not None:
                row["replaced_by"] = int(rec["replaced_by"])
            continue

        if event == "resolve":
            fid = int(rec["id"])
            if fid not in forecasts:
                continue
            row = forecasts[fid]
            if row.get("status") in {"resolved", "superseded"}:
                continue
            if "dist" in row:
                outcome_bin = rec.get("outcome_bin")
                if outcome_bin not in row["bins"]:
                    continue
                outcome_index = row["bins"].index(outcome_bin)
                row["outcome_bin"] = outcome_bin
                row["rps"] = rec.get("rps", ranked_probability_score(row["dist"], outcome_index))
            else:
                outcome = validate_outcome(int(rec["outcome"]))
                row["outcome"] = outcome
                row["brier"] = round(brier_score(float(row["prob"]), outcome), 4)
            row["status"] = "resolved"
            row["resolved"] = rec.get("resolved")
            continue

        if event == "update":
            fid = int(rec["id"])
            if fid not in forecasts:
                continue
            row = forecasts[fid]
            if row.get("status") in {"resolved", "superseded"}:
                continue
            new_prob = validate_probability(float(rec["prob"]), "prob")
            row["prob"] = new_prob
            hist = histories[fid]
            hist.append({"prob": new_prob, "at": rec.get("updated"), "note": rec.get("note")})
            row["prob_history"] = hist
            continue

        if "id" not in rec or ("prob" not in rec and "dist" not in rec):
            continue
        fid = int(rec["id"])
        row = dict(rec)
        row["event"] = "forecast"
        row["id"] = fid
        if "dist" in row:
            row["bins"] = list(row.get("bins", []))
            row["dist"] = [validate_probability(float(p), "dist probability") for p in row["dist"]]
            if len(row["bins"]) < 2 or len(row["bins"]) != len(row["dist"]):
                continue
            if not math.isclose(sum(row["dist"]), 1.0, rel_tol=0.0, abs_tol=1e-9):
                continue
            row["outcome"] = None
            row.setdefault("status", "open")
            histories[fid] = []
        else:
            row["prob"] = validate_probability(float(row["prob"]), "prob")
            if row.get("outcome") is not None:
                row["outcome"] = validate_outcome(int(row["outcome"]))
                row["status"] = "resolved"
                row["brier"] = round(brier_score(float(row["prob"]), int(row["outcome"])), 4)
            else:
                row["outcome"] = None
                row.setdefault("status", "open")
            histories[fid] = [{"prob": row["prob"], "at": rec.get("created"), "note": None}]
        if fid not in forecasts:
            order.append(fid)
        forecasts[fid] = row

    return [forecasts[fid] for fid in order]
