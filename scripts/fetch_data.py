#!/usr/bin/env python3
"""Fetch all registered historical sources (pinned URLs, SHA-256 verified)."""
import sys

from sportslab.data.fetch import fetch_all

if __name__ == "__main__":
    force = "--force" in sys.argv
    m = fetch_all(force=force)
    bad = 0
    for sid, r in m.items():
        print(f"{r['status']:18} {sid:32} {r.get('bytes', '-'):>9}  {r.get('error', '')}")
        bad += r["status"] not in ("downloaded", "cached")
    print(f"\n{len(m) - bad}/{len(m)} sources OK")
    sys.exit(1 if bad else 0)
