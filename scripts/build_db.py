#!/usr/bin/env python3
"""Build the unified SQLite DB + Parquet mirrors from data/raw, then run data-quality checks."""
import sys
import time

from sportslab.config import ARTIFACTS_DIR, DB_PATH
from sportslab.data.ingest import ingest_all
from sportslab.data.quality import run_checks
from sportslab.data.store import export_parquet, read_table, replace_facts

if __name__ == "__main__":
    t0 = time.time()
    tables = ingest_all()
    replace_facts(tables)
    counts = export_parquet()
    print(f"DB: {DB_PATH}")
    for k, v in counts.items():
        print(f"  {k:20} {v:>8} rows")
    q = run_checks({t: read_table(t) for t in ('events', 'results', 'odds')})
    q['value'] = q['value'].astype(str)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    q.to_parquet(ARTIFACTS_DIR / "data_quality.parquet", index=False)
    print(q.to_string(index=False, max_colwidth=70))
    print(f"\n{(q.status == 'PASS').sum()}/{len(q)} checks PASS  ({time.time() - t0:.1f}s)")
    sys.exit(0 if (q.status == "PASS").all() else 1)
