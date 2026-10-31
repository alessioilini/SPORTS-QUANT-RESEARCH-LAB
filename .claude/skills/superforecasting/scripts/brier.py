#!/usr/bin/env python3
"""Brier scoring + calibration/resolution decomposition — how you get better over time.

Perpetual beta: superforecasters improve because they SCORE themselves and study the
gap.  The Brier score is the mean squared error of probabilistic forecasts:

    Brier = mean( (p - outcome)^2 )      lower is better; 0 = perfect, 0.25 = coin-flip guess

Murphy's decomposition (with binning) splits it into three interpretable parts:

    Brier = Reliability  -  Resolution  +  Uncertainty
            (calibration)   (discrimination)  (base rate difficulty)

  * Reliability  (want LOW):  do your 70%s happen 70% of the time?
  * Resolution   (want HIGH): do you push away from the base rate toward 0/1 correctly?
  * Uncertainty  (fixed):     ō(1-ō), the irreducible difficulty of the event mix.

Usage
-----
  # score one forecast
  python3 brier.py --prob 0.7 --outcome 1

  # score your whole ledger ($FORECAST_LOG, else ~/.superforecasting/forecasts.jsonl)
  python3 brier.py

  # or point at any JSONL log with prob & outcome fields
  python3 brier.py --log /path/to/forecasts.jsonl

  # additionally score belief trajectories by days each probability was live
  python3 brier.py --time-weighted

  # compare against a baseline probability
  python3 brier.py --baseline-prob 0.5
  python3 brier.py --baseline-field crowd_prob

  # inline list "prob:outcome"
  python3 brier.py --pairs 0.7:1 0.3:0 0.9:1 0.6:0 0.8:1
"""
from __future__ import annotations
import argparse
import sys

from forecast_common import (
    default_log_path,
    fold_forecast_log,
    time_weighted_brier_score,
    validate_outcome,
    validate_probability,
)


def brier(pairs: list[tuple[float, int]]) -> float:
    for p, o in pairs:
        validate_probability(p, "prob")
        validate_outcome(o)
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def decompose(pairs: list[tuple[float, int]], nbins: int = 10):
    if nbins <= 0:
        raise ValueError("bins must be > 0")
    n = len(pairs)
    obar = sum(o for _, o in pairs) / n
    uncertainty = obar * (1 - obar)
    # bin by forecast probability
    bins: dict[int, list[tuple[float, int]]] = {}
    for p, o in pairs:
        b = min(nbins - 1, int(p * nbins))
        bins.setdefault(b, []).append((p, o))
    reliability = 0.0
    resolution = 0.0
    table = []
    for b in sorted(bins):
        grp = bins[b]
        nk = len(grp)
        pbar = sum(p for p, _ in grp) / nk
        okbar = sum(o for _, o in grp) / nk
        reliability += nk * (pbar - okbar) ** 2
        resolution += nk * (okbar - obar) ** 2
        table.append((b / nbins, (b + 1) / nbins, nk, pbar, okbar))
    reliability /= n
    resolution /= n
    return reliability, resolution, uncertainty, obar, table


def load_rows(args) -> list[dict]:
    rows: list[dict] = []
    if args.prob is not None:
        if args.outcome is None:
            raise SystemExit("--prob requires --outcome")
        rows.append({"prob": validate_probability(args.prob, "prob"),
                     "outcome": validate_outcome(int(args.outcome))})
    for tok in args.pairs or []:
        try:
            p, o = tok.split(":")
            rows.append({"prob": validate_probability(float(p), "prob"),
                         "outcome": validate_outcome(int(o))})
        except ValueError as exc:
            raise ValueError(f"invalid --pairs token {tok!r}: {exc}") from exc
    if args.log:
        for rec in fold_forecast_log(args.log):
            if (rec.get("status") != "resolved" or rec.get("outcome") is None
                    or rec.get("prob") is None):
                continue  # unresolved
            row = dict(rec)
            row["prob"] = validate_probability(float(row["prob"]), "prob")
            row["outcome"] = validate_outcome(int(row["outcome"]))
            rows.append(row)
    return rows


def load_distribution_rows(log_path: str | None) -> list[dict]:
    """Return resolved ordered-bin snapshots, separate from binary Brier rows."""
    if not log_path:
        return []
    return [rec for rec in fold_forecast_log(log_path)
            if rec.get("status") == "resolved"
            and rec.get("dist") is not None and rec.get("rps") is not None]


def baseline_pairs(rows: list[dict], args) -> tuple[str, list[tuple[float, int]]]:
    if args.baseline_prob is not None:
        p = validate_probability(args.baseline_prob, "baseline_prob")
        return f"constant {p:.3f}", [(p, int(r["outcome"])) for r in rows]
    if args.baseline_field:
        pairs = []
        for r in rows:
            if r.get(args.baseline_field) is None:
                continue
            pairs.append((validate_probability(float(r[args.baseline_field]), args.baseline_field),
                          int(r["outcome"])))
        return args.baseline_field, pairs
    return "", []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prob", type=float, help="single forecast probability")
    ap.add_argument("--outcome", type=int, help="single outcome 0 or 1")
    ap.add_argument("--pairs", nargs="*", help="list of prob:outcome tokens")
    ap.add_argument("--log", help="JSONL file with prob & outcome fields (default when no "
                                  "--prob/--pairs given: $FORECAST_LOG, else "
                                  "~/.superforecasting/forecasts.jsonl)")
    ap.add_argument("--time-weighted", action="store_true",
                    help="with --log, add duration-weighted trajectory Brier: each probability is "
                         "held until its next update (or resolution); unusable trajectories use "
                         "the standard latest-probability Brier")
    ap.add_argument("--bins", type=int, default=10, help="calibration bins (default 10)")
    ap.add_argument("--baseline-prob", type=float,
                    help="constant baseline probability for Brier Skill Score")
    ap.add_argument("--baseline-field",
                    help="record field containing per-forecast baseline probabilities")
    args = ap.parse_args(argv)
    # Only fall back to the default ledger when no other data source was given:
    # `args.log` doubles as a "read the ledger" flag below, so defaulting it
    # unconditionally would silently fold the whole ledger into a --prob/--pairs run.
    defaulted_log = args.log is None and args.prob is None and not args.pairs
    if defaulted_log:
        args.log = default_log_path()

    try:
        if args.bins <= 0:
            raise ValueError("--bins must be > 0")
        if args.baseline_prob is not None and args.baseline_field:
            raise ValueError("use only one of --baseline-prob or --baseline-field")
        if args.time_weighted and (not args.log or args.prob is not None or args.pairs):
            raise ValueError("--time-weighted requires --log and cannot be combined with --prob or --pairs")
        rows = load_rows(args)
        distribution_rows = load_distribution_rows(args.log)
    except ValueError as exc:
        ap.error(str(exc))

    pairs = [(float(r["prob"]), int(r["outcome"])) for r in rows]
    if not pairs and not distribution_rows:
        if defaulted_log:
            ap.error(f"no resolved forecasts in {args.log} — resolve one with "
                     "'forecastlog.py resolve', or score inline with --prob/--pairs")
        ap.error("no data — use --prob/--outcome, --pairs, or --log")

    if pairs:
        bs = brier(pairs)
        print(f"forecasts scored : {len(pairs)}")
        print(f"Brier score      : {bs:.4f}   (0=perfect, 0.25=uninformed coin flip)")
        if args.time_weighted:
            twb = sum(time_weighted_brier_score(row) for row in rows) / len(rows)
            print(f"Time-weighted Brier: {twb:.4f}   (duration-weighted trajectory; lower is better)")

    label, ref_pairs = baseline_pairs(rows, args)
    if ref_pairs:
        ref_bs = brier(ref_pairs)
        bss = None if ref_bs == 0 else 1 - (bs / ref_bs)
        print(f"baseline          : {label}")
        print(f"baseline Brier    : {ref_bs:.4f}")
        print(f"Brier Skill Score : {'n/a' if bss is None else f'{bss:.4f}'}")
        if len(ref_pairs) != len(rows):
            print(f"baseline note     : used {len(ref_pairs)} rows with field {args.baseline_field!r}")

    if len(pairs) >= 2:
        try:
            rel, res, unc, obar, table = decompose(pairs, args.bins)
        except ValueError as exc:
            ap.error(str(exc))
        print(f"base rate (ō)    : {obar:.4f}")
        print(f"  reliability    : {rel:.4f}  (calibration error — lower better)")
        print(f"  resolution     : {res:.4f}  (discrimination — higher better)")
        print(f"  uncertainty    : {unc:.4f}  (difficulty — fixed by outcomes)")
        print(f"  check rel-res+unc = {rel - res + unc:.4f}")
        if unc > 0:
            print(f"  BSS vs base rate: {1 - (bs / unc):.4f}")
        print("\ncalibration table:")
        print("  bin           n   mean_p  observed")
        for lo, hi, nk, pbar, okbar in table:
            flag = "  <-- overconfident" if pbar - okbar > 0.15 else (
                   "  <-- underconfident" if okbar - pbar > 0.15 else "")
            print(f"  {lo:.1f}-{hi:.1f}  {nk:4d}   {pbar:5.2f}    {okbar:5.2f}{flag}")
    if distribution_rows:
        mean_rps = sum(float(row["rps"]) for row in distribution_rows) / len(distribution_rows)
        print(f"ordered distributions scored : {len(distribution_rows)}")
        print(f"Mean RPS                    : {mean_rps:.4f}   (lower is better)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
