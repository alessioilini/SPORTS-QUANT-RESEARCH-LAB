#!/usr/bin/env python3
"""Reference-class / base-rate helper — the 'outside view' anchor.

Commandment #3: before diving into case specifics, ask "how often do things of THIS
sort happen in situations of THIS sort?"  Define a reference class, count how many
cases resolved YES out of the total, and that frequency is your starting probability.

This tool turns counts into a point estimate plus an honest interval, with smoothing
so small samples don't collapse to 0% or 100%.

    p_hat      = k / n                         (raw frequency)
    Laplace    = (k + 1) / (n + 2)             (rule of succession — good default prior)
    Jeffreys   = (k + 0.5) / (n + 1)
    Wilson 95% CI                              (reliable even for small n / extreme p)

Examples
--------
  # 3 of the last 11 incumbent parties in this situation lost -> base rate
  python3 baserate.py --hits 3 --total 11

  # just print the reference-class checklist to guide the outside view
  python3 baserate.py --checklist
"""
from __future__ import annotations
import argparse
import math
import sys

CHECKLIST = """\
REFERENCE-CLASS CHECKLIST (build the outside view first)
  1. State the event in resolvable terms (what exactly counts as YES?).
  2. Pick a reference class: a set of past cases 'of this sort'. Prefer:
       - similar mechanism, not just similar surface topic
       - large enough to count, narrow enough to be relevant
  3. Count: how many of those cases resolved YES?  k out of n.
  4. That frequency k/n is your ANCHOR probability. Write it down BEFORE inside view.
  5. Only then adjust for case-specifics (inside view) — and adjust modestly.
  6. If several reference classes are defensible, compute each and reconcile the range.
"""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hits", "-k", type=int, help="number of reference cases that resolved YES")
    ap.add_argument("--total", "-n", type=int, help="total number of reference cases")
    ap.add_argument("--checklist", action="store_true", help="print the reference-class checklist and exit")
    args = ap.parse_args(argv)

    if args.checklist or args.hits is None or args.total is None:
        print(CHECKLIST)
        if args.hits is None or args.total is None:
            return 0

    k, n = args.hits, args.total
    if n <= 0 or k < 0 or k > n:
        ap.error("require 0 <= hits <= total and total > 0")

    lo, hi = wilson(k, n)
    print(f"reference class    : {k} YES out of {n}")
    print(f"raw frequency      : {k/n:.4f}")
    print(f"Laplace (default)  : {(k+1)/(n+2):.4f}   <- recommended anchor for small n")
    print(f"Jeffreys           : {(k+0.5)/(n+1):.4f}")
    print(f"Wilson 95% CI      : [{lo:.4f}, {hi:.4f}]")
    if n < 10:
        print("note: n < 10 — treat the anchor as soft; widen your uncertainty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
