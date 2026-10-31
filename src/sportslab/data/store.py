"""Unified storage layer: SQLite (source of truth) + Parquet mirrors.

Every fact row carries provenance columns:
  source_id     – which registered source produced it
  retrieved_at  – when the raw file was obtained (UTC ISO)
  available_at  – earliest UTC instant the fact could have been known in reality
  value_kind    – OBSERVED | MODEL | UNKNOWN
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from sportslab.config import DB_PATH, PROCESSED_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources(
  source_id TEXT PRIMARY KEY, sport TEXT, kind TEXT, url TEXT, original_provider TEXT,
  host_repo TEXT, license_note TEXT, season TEXT, role TEXT, notes TEXT,
  retrieved_at TEXT, sha256 TEXT, bytes INTEGER, status TEXT);

CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY, sport TEXT NOT NULL, competition TEXT, season TEXT,
  event_time TEXT NOT NULL, event_time_precision TEXT CHECK(event_time_precision IN ('minute','day')),
  forecast_time TEXT NOT NULL, forecast_time_rule TEXT,
  home TEXT NOT NULL, away TEXT NOT NULL, neutral INTEGER DEFAULT 0, extra TEXT,
  source_id TEXT, retrieved_at TEXT, value_kind TEXT DEFAULT 'OBSERVED',
  CHECK(forecast_time <= event_time));

CREATE TABLE IF NOT EXISTS results(
  event_id TEXT PRIMARY KEY REFERENCES events(event_id), home_score REAL, away_score REAL,
  outcome TEXT, ht_home REAL, ht_away REAL, regulation_draw INTEGER, decided_by TEXT,
  completed INTEGER, detail TEXT,
  available_at TEXT NOT NULL, source_id TEXT, retrieved_at TEXT, value_kind TEXT DEFAULT 'OBSERVED');

CREATE TABLE IF NOT EXISTS team_match_stats(
  event_id TEXT, side TEXT CHECK(side IN ('home','away')), stat TEXT, value REAL,
  available_at TEXT, source_id TEXT, retrieved_at TEXT, value_kind TEXT DEFAULT 'OBSERVED',
  PRIMARY KEY(event_id, side, stat));

CREATE TABLE IF NOT EXISTS player_prematch(
  event_id TEXT, side TEXT, player TEXT, stat TEXT, value REAL,
  available_at TEXT, source_id TEXT, retrieved_at TEXT, value_kind TEXT DEFAULT 'OBSERVED',
  PRIMARY KEY(event_id, side, stat));

CREATE TABLE IF NOT EXISTS lineups(
  event_id TEXT, side TEXT, player TEXT, role TEXT, status TEXT,
  available_at TEXT, source_id TEXT, retrieved_at TEXT, value_kind TEXT);

CREATE TABLE IF NOT EXISTS injuries(
  event_id TEXT, team TEXT, player TEXT, status TEXT, verified_by TEXT,
  available_at TEXT, source_id TEXT, retrieved_at TEXT, value_kind TEXT);

CREATE TABLE IF NOT EXISTS odds(
  event_id TEXT, bookmaker TEXT, market TEXT, selection TEXT, price REAL,
  snapshot TEXT CHECK(snapshot IN ('opening','closing','single')),
  observed_at TEXT, observed_at_precision TEXT, available_at TEXT,
  source_id TEXT, retrieved_at TEXT, value_kind TEXT DEFAULT 'OBSERVED',
  PRIMARY KEY(event_id, bookmaker, market, selection, snapshot));

CREATE TABLE IF NOT EXISTS external_forecasts(
  event_id TEXT, provider TEXT, selection TEXT, prob REAL, available_at TEXT,
  available_at_precision TEXT, source_id TEXT, retrieved_at TEXT, value_kind TEXT DEFAULT 'OBSERVED',
  PRIMARY KEY(event_id, provider, selection));

CREATE TABLE IF NOT EXISTS coverage(
  sport TEXT, source_id TEXT, field TEXT, status TEXT, note TEXT);

CREATE TABLE IF NOT EXISTS model_forecasts(
  run_id TEXT, model TEXT, event_id TEXT, forecast_time TEXT, selection TEXT, prob REAL,
  split TEXT, created_at TEXT, value_kind TEXT DEFAULT 'MODEL');

CREATE TABLE IF NOT EXISTS forecast_log(
  forecast_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, mode TEXT, sport TEXT, event_id TEXT,
  model TEXT, as_of TEXT, probs_json TEXT, status TEXT, inputs_json TEXT, sources_json TEXT,
  resolved_outcome TEXT, resolved_at TEXT, brier REAL, logloss REAL);

CREATE TABLE IF NOT EXISTS virtual_journal(
  entry_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, created_at TEXT NOT NULL,
  event_id TEXT, forecast_time TEXT, model TEXT, selection TEXT, model_prob REAL,
  price_taken REAL, closing_price REAL, stake_virtual REAL, outcome_won INTEGER,
  pnl_virtual REAL, commission_virtual REAL, bankroll_after_virtual REAL, note TEXT);

CREATE TRIGGER IF NOT EXISTS journal_no_update BEFORE UPDATE ON virtual_journal
BEGIN SELECT RAISE(ABORT, 'virtual_journal is append-only'); END;
CREATE TRIGGER IF NOT EXISTS journal_no_delete BEFORE DELETE ON virtual_journal
BEGIN SELECT RAISE(ABORT, 'virtual_journal is append-only'); END;

CREATE TABLE IF NOT EXISTS live_snapshots(
  snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT, sport TEXT, event_id TEXT, source TEXT,
  fetched_at TEXT, source_updated_at TEXT, payload_json TEXT, status TEXT);

CREATE INDEX IF NOT EXISTS ix_events_sport_time ON events(sport, forecast_time);
CREATE INDEX IF NOT EXISTS ix_results_avail ON results(available_at);
CREATE INDEX IF NOT EXISTS ix_odds_event ON odds(event_id);
"""

FACT_TABLES = ["sources", "events", "results", "team_match_stats", "player_prematch", "lineups",
               "injuries", "odds", "external_forecasts", "coverage"]


@contextmanager
def connect(path: Path | str = DB_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(path: Path | str = DB_PATH) -> None:
    with connect(path) as con:
        con.executescript(SCHEMA)


def replace_facts(tables: dict[str, pd.DataFrame], path: Path | str = DB_PATH, sports: list[str] | None = None) -> None:
    """Replace fact rows (all, or only for the given sports) with new frames."""
    init_db(path)
    with connect(path) as con:
        cols = {t: [r[1] for r in con.execute(f"PRAGMA table_info({t})")] for t in tables}
        if sports is None:
            for t in reversed(FACT_TABLES):   # children before parents (FK results->events)
                con.execute(f"DELETE FROM {t}")
        for t, df in tables.items():
            if df is None or df.empty:
                continue
            extra = set(df.columns) - set(cols[t])
            if extra:
                raise ValueError(f"{t}: unknown columns {sorted(extra)}")
            df.to_sql(t, con, if_exists="append", index=False)


def read_table(table: str, path: Path | str = DB_PATH, where: str = "", params: tuple = ()) -> pd.DataFrame:
    with connect(path) as con:
        return pd.read_sql_query(f"SELECT * FROM {table} {where}", con, params=params)


def export_parquet(path: Path | str = DB_PATH, out_dir: Path = PROCESSED_DIR) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    with connect(path) as con:
        for t in FACT_TABLES:
            df = pd.read_sql_query(f"SELECT * FROM {t}", con)
            df.to_parquet(out_dir / f"{t}.parquet", index=False)
            counts[t] = len(df)
    return counts
