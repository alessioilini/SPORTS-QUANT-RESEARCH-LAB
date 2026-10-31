#!/usr/bin/env python3
"""Bayesian belief updating in odds form.

The superforecaster's core move: start from a base rate (prior), then update
in small steps as evidence arrives.  Working in ODDS makes updates multiplicative
and order-independent:

    posterior_odds = prior_odds * LR_1 * LR_2 * ... * LR_n

where each likelihood ratio  LR = P(evidence | hypothesis) / P(evidence | not hypothesis).
LR > 1 pushes the probability up, LR < 1 pushes it down, LR = 1 is uninformative.

Examples
--------
  # Prior 30%, two pieces of evidence (one supporting 3x, one weakly against 0.7x)
  python3 bayes.py --prior 0.30 --lr 3.0 --lr 0.7

  # Give evidence as conditional probabilities instead of a raw LR
  python3 bayes.py --prior 0.10 --pe-h 0.80 --pe-nh 0.20

  # Show the running update after each step
  python3 bayes.py --prior 0.50 --lr 2 --lr 2 --lr 0.5 --steps

This is language-agnostic in spirit: any agent/harness can shell out to it, or
import `update()` directly.
"""
from __future__ import annotations
import argparse
import sys

from forecast_common import validate_open_probability, validate_positive


def p_to_odds(p: float) -> float:
    validate_open_probability(p)
    return p / (1.0 - p)


def odds_to_p(o: float) -> float:
    return o / (1.0 + o)


def update(prior: float, lrs: list[float]) -> float:
    """Return posterior probability from a prior and a list of likelihood ratios."""
    o = p_to_odds(prior)
    for lr in lrs:
        validate_positive(lr, "likelihood ratio")
        o *= lr
    return odds_to_p(o)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", type=float, required=True,
                    help="prior probability (0-1), typically your base rate")
    ap.add_argument("--lr", type=float, action="append", default=[],
                    help="a likelihood ratio; repeat for multiple pieces of evidence")
    ap.add_argument("--pe-h", type=float, action="append", default=[],
                    help="P(evidence|hypothesis); pair with --pe-nh")
    ap.add_argument("--pe-nh", type=float, action="append", default=[],
                    help="P(evidence|NOT hypothesis); pair with --pe-h")
    ap.add_argument("--steps", action="store_true",
                    help="print the running probability after each update")
    args = ap.parse_args(argv)

    lrs = list(args.lr)
    if len(args.pe_h) != len(args.pe_nh):
        ap.error("--pe-h and --pe-nh must be given the same number of times")
    for ph, pnh in zip(args.pe_h, args.pe_nh):
        if not (0 < ph <= 1 and 0 < pnh <= 1):
            ap.error("--pe-h / --pe-nh must be in (0, 1]")
        lrs.append(ph / pnh)

    if not lrs:
        ap.error("give at least one --lr or one --pe-h/--pe-nh pair")

    try:
        validate_open_probability(args.prior, "prior")
        for lr in lrs:
            validate_positive(lr, "likelihood ratio")
    except ValueError as exc:
        ap.error(str(exc))

    if args.steps:
        o = p_to_odds(args.prior)
        print(f"prior           p={args.prior:6.3f}  odds={o:8.3f}")
        for i, lr in enumerate(lrs, 1):
            o *= lr
            print(f"after evidence {i}  LR={lr:6.3f}  ->  p={odds_to_p(o):6.3f}  odds={o:8.3f}")
        print(f"\nPOSTERIOR       p={odds_to_p(o):6.3f}")
    else:
        post = update(args.prior, lrs)
        combined_lr = 1.0
        for lr in lrs:
            combined_lr *= lr
        print(f"prior       : {args.prior:.4f}")
        print(f"combined LR : {combined_lr:.4f}")
        print(f"posterior   : {post:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
