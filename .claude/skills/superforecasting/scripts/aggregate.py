#!/usr/bin/env python3
"""Aggregate several probability forecasts into one — the 'dragonfly eye' / crowd step.

This is the engine for the ENSEMBLE workflow: run the same question through
several agents (Claude, Codex, Aider, humans...), collect each one's
probability, then pool them here.

Default method is the GEOMETRIC MEAN OF ODDS, which the forecasting literature
recommends as the best simple pool (it satisfies "external Bayesianity" — pooling
then updating equals updating then pooling):

    odds_i   = p_i / (1 - p_i)
    pooled   = geomean(odds_i)  = (prod odds_i) ** (1/n)     # or weighted
    p_pooled = pooled / (1 + pooled)

Optional EXTREMIZING sharpens the pool by scaling the pooled log-odds:

    logit(p_ext) = a * logit(p_pooled),   a >= 1

Extremizing helps when forecasters are diverse and each sees independent signal;
it hurts when they are correlated or few.  Keep a in ~[1.0, 2.5]; default 1.0 (off).

Examples
--------
  python3 aggregate.py 0.6 0.7 0.55 0.8
  python3 aggregate.py 0.6 0.7 0.55 0.8 --extremize 2.0
  python3 aggregate.py --p 0.3 --w 2 --p 0.6 --w 1        # weighted
  python3 aggregate.py --p 0.6 --forecaster alice --p 0.7 --forecaster bob --track-log "$FORECAST_LOG"
  python3 aggregate.py --p 0.6 --date 2026-07-01 --p 0.7 --date 2026-07-12 --half-life-days 14
  echo "0.6 0.7 0.55" | python3 aggregate.py -            # read from stdin
"""
from __future__ import annotations
import argparse
import math
import sys
from datetime import datetime, timezone
from statistics import median

from forecast_common import (
    age_days,
    fold_forecast_log,
    parse_datetime,
    validate_positive,
    validate_probability,
)


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def geomean_of_odds(ps: list[float], weights: list[float] | None = None) -> float:
    """Pool probabilities by the weighted geometric mean of their odds.

    Endpoint contract: probabilities must be in the closed interval [0, 1] — the
    same interval `validate_probability` accepts everywhere else in the toolkit —
    and 0.0 / 1.0 are clipped to `_clip`'s epsilon so their odds stay finite. The
    returned probability is therefore always strictly between 0 and 1, and no
    input raises `ZeroDivisionError`.
    """
    if weights is None:
        weights = [1.0] * len(ps)
    if len(weights) != len(ps):
        raise ValueError("weights must match probabilities")
    W = sum(weights)
    if W <= 0:
        raise ValueError("sum of weights must be > 0")
    for w in weights:
        validate_positive(w, "weight")
    for i, p in enumerate(ps, 1):
        validate_probability(p, f"geomean_of_odds probability {i}")
    # weighted geometric mean of odds == inverse-logit of weighted mean of logits
    s = sum(w * logit(p) for p, w in zip(_clip(ps), weights))
    return inv_logit(s / W)


def extremize(p: float, a: float) -> float:
    """Sharpen a pooled probability by scaling its log-odds by `a` (a >= 1).

    Same endpoint contract as `geomean_of_odds`: `p` must be in [0, 1] and the
    endpoints are clipped, so extremizing 0.0 or 1.0 returns a value strictly
    inside (0, 1) instead of raising `ZeroDivisionError`.
    """
    if a < 1.0:
        raise ValueError(f"extremize: exponent a must be >= 1, got {a}")
    validate_probability(p, "extremize probability")
    return inv_logit(a * logit(_clip_one(p)))


def _clip_one(p: float, eps: float = 1e-6) -> float:
    return min(1 - eps, max(eps, p))


def _clip(ps: list[float], eps: float = 1e-6) -> list[float]:
    return [_clip_one(p, eps) for p in ps]


def track_record_weights(
    log_path: str,
    forecasters: list[str | None],
    prior_n: float = 5.0,
    prior_brier: float = 0.25,
) -> tuple[list[float], dict[str, tuple[int, float]]]:
    """Return smoothed Brier-derived weights and diagnostics by forecaster.

    Weight 1.0 means "as good as the prior". Better historical Brier scores get
    weights above 1; worse scores get weights below 1. The prior prevents a
    single lucky forecast from dominating the pool.
    """
    validate_positive(prior_n, "prior_n")
    validate_positive(prior_brier, "prior_brier")
    rows = fold_forecast_log(log_path)
    by_forecaster: dict[str, list[float]] = {}
    for row in rows:
        name = row.get("forecaster")
        if not name or row.get("outcome") is None:
            continue
        by_forecaster.setdefault(str(name), []).append((float(row["prob"]) - int(row["outcome"])) ** 2)

    weights: list[float] = []
    stats: dict[str, tuple[int, float]] = {}
    for name in forecasters:
        if not name:
            weights.append(1.0)
            continue
        scores = by_forecaster.get(name, [])
        smoothed = (sum(scores) + prior_brier * prior_n) / (len(scores) + prior_n)
        weights.append(prior_brier / smoothed)
        stats[name] = (len(scores), smoothed)
    return weights, stats


def time_decay_weights(dates: list[str | None], as_of: datetime, half_life_days: float) -> list[float]:
    validate_positive(half_life_days, "half_life_days")
    out: list[float] = []
    for stamp in dates:
        if not stamp:
            out.append(1.0)
        else:
            out.append(0.5 ** (age_days(stamp, as_of) / half_life_days))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probs", nargs="*", help="probabilities (0-1); or use '-' to read stdin")
    ap.add_argument("--p", type=float, action="append", default=[],
                    help="a probability (repeatable); pair with --w for weights")
    ap.add_argument("--w", type=float, action="append", default=[],
                    help="weight for the preceding --p (default 1)")
    ap.add_argument("--extremize", type=float, default=1.0,
                    help="extremizing exponent a>=1 (default 1.0 = none)")
    ap.add_argument("--forecaster", action="append", default=[],
                    help="forecaster id for the corresponding --p; enables --track-log weighting")
    ap.add_argument("--track-log", help="forecastlog JSONL path used to derive track-record weights")
    ap.add_argument("--prior-n", type=float, default=5.0,
                    help="pseudo-count for track-record smoothing (default 5)")
    ap.add_argument("--prior-brier", type=float, default=0.25,
                    help="prior Brier score for track-record smoothing (default 0.25)")
    ap.add_argument("--date", action="append", default=[],
                    help="forecast date for the corresponding --p (YYYY-MM-DD or ISO)")
    ap.add_argument("--as-of", default=None,
                    help="as-of date for time decay (default now, UTC)")
    ap.add_argument("--half-life-days", type=float, default=None,
                    help="apply exponential time decay with this half-life")
    args = ap.parse_args(argv)

    ps: list[float] = []
    ws: list[float] = []
    forecasters: list[str | None] = []
    dates: list[str | None] = []

    if not args.p and (args.w or args.forecaster or args.date):
        ap.error("--w, --forecaster, and --date require probabilities provided with --p")
    if args.track_log and not args.forecaster:
        ap.error("--track-log requires --forecaster values paired with --p")
    if args.half_life_days is not None and not args.date:
        ap.error("--half-life-days requires --date values paired with --p")

    if args.p:
        if len(args.w) > len(args.p):
            ap.error("--w cannot be provided more times than --p")
        if args.forecaster and len(args.forecaster) != len(args.p):
            ap.error("--forecaster must be provided exactly once for each --p")
        if args.date and len(args.date) != len(args.p):
            ap.error("--date must be provided exactly once for each --p")
        ps.extend(args.p)
        w = list(args.w) + [1.0] * (len(args.p) - len(args.w))
        ws.extend(w[:len(args.p)])
        forecasters.extend((args.forecaster or [None] * len(args.p))[:len(args.p)])
        dates.extend((args.date or [None] * len(args.p))[:len(args.p)])

    tokens = list(args.probs)
    if tokens == ["-"] or (not tokens and not ps and not sys.stdin.isatty()):
        tokens = sys.stdin.read().split()
    for t in tokens:
        if t == "-":
            continue
        ps.append(float(t))
        ws.append(1.0)
        forecasters.append(None)
        dates.append(None)

    if not ps:
        ap.error("provide probabilities as positional args, --p, or via stdin")
    try:
        for i, x in enumerate(ps, 1):
            validate_probability(x, f"probability {i}")
        for i, w in enumerate(ws, 1):
            validate_positive(w, f"weight {i}")
        if args.extremize < 1.0:
            raise ValueError("--extremize must be >= 1")
        if args.half_life_days is not None:
            decay_as_of = parse_datetime(args.as_of) or datetime.now(timezone.utc)
            decay_weights = time_decay_weights(dates, decay_as_of, args.half_life_days)
            ws = [w * d for w, d in zip(ws, decay_weights)]
        if args.track_log:
            track_weights, track_stats = track_record_weights(
                args.track_log, forecasters, args.prior_n, args.prior_brier
            )
            ws = [w * t for w, t in zip(ws, track_weights)]
        else:
            track_stats = {}
    except ValueError as exc:
        ap.error(str(exc))

    try:
        pooled = geomean_of_odds(ps, ws)
        result = extremize(pooled, args.extremize) if args.extremize != 1.0 else pooled
    except ValueError as exc:
        ap.error(str(exc))

    print(f"n forecasts        : {len(ps)}")
    print(f"inputs             : {', '.join(f'{p:.3f}' for p in ps)}")
    if any(w != 1.0 for w in ws):
        print(f"weights            : {', '.join(f'{w:g}' for w in ws)}")
        print(f"weighted mean      : {sum(p * w for p, w in zip(ps, ws)) / sum(ws):.4f}")
    if any(forecasters):
        print(f"forecasters        : {', '.join(f or '-' for f in forecasters)}")
    if track_stats:
        detail = ", ".join(f"{name}: n={n}, smoothed_brier={score:.3f}"
                           for name, (n, score) in sorted(track_stats.items()))
        print(f"track record       : {detail}")
    if args.half_life_days is not None:
        print(f"time decay         : half_life_days={args.half_life_days:g}")
    print(f"simple mean        : {sum(ps)/len(ps):.4f}")
    print(f"median             : {median(ps):.4f}")
    print(f"geo-mean of odds   : {pooled:.4f}   <- recommended pool")
    if args.extremize != 1.0:
        print(f"extremized (a={args.extremize:g})  : {result:.4f}")
    spread = max(ps) - min(ps)
    print(f"disagreement (max-min): {spread:.3f}"
          + ("   [wide — inspect why before trusting the pool]" if spread > 0.4 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
