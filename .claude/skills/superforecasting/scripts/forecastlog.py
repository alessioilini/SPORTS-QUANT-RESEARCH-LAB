#!/usr/bin/env python3
"""Forecast log — record every forecast, resolve it later, score yourself.

A JSONL ledger (one forecast per line) is the backbone of deliberate practice.
Log the probability AND the rationale when you make the call; resolve it when the
outcome is known; then run brier.py over the file to measure calibration.

Commands
--------
  add      register a new open forecast
  update   revise the probability of an open forecast (append-only; keeps a trajectory)
  supersede retire an invalid/withdrawn forecast without scoring it
  resolve  set the outcome (0/1) on an existing forecast
  list     show forecasts (optionally filtered by status or due date)
  score    quick Brier over all resolved forecasts (delegates to brier logic)

Examples
--------
  python3 forecastlog.py add \\
      --question "Will X ship v2 before 2026-12-31?" \\
      --prob 0.35 --resolve-by 2026-12-31 \\
      --rationale "base rate of on-time major releases ~0.3; team behind schedule" \\
      --tags product,deadlines --forecaster alice --baseline-prob 0.30

  python3 forecastlog.py update --id 3 --prob 0.45 --note "earnings beat"
  python3 forecastlog.py supersede --id 3 --reason "question withdrawn" --replaced-by 4
  python3 forecastlog.py list --status open
  python3 forecastlog.py list --overdue
  python3 forecastlog.py list --due 14 --as-of 2026-08-01
  python3 forecastlog.py resolve --id 3 --outcome 0
  python3 forecastlog.py score

Log path resolution (highest priority first):
  1. --log <path>
  2. $FORECAST_LOG
  3. ~/.superforecasting/forecasts.jsonl

The ledger lives outside the skill directory, so the skill never writes into its
own checkout and one $FORECAST_LOG points every harness at the same record.
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime, timezone

from forecast_common import (
    append_jsonl,
    brier_score,
    default_log_path,
    fold_forecast_log,
    parse_datetime,
    ranked_probability_score,
    read_jsonl,
    validate_outcome,
    validate_probability,
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _all_ids(path: str) -> list[int]:
    ids = []
    for row in read_jsonl(path):
        if row.get("id") is not None:
            ids.append(int(row["id"]))
    return ids


def _validated_optional_prob(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return validate_probability(float(value), name)


def _parse_distribution(bins_text: str, dist_text: str) -> tuple[list[str], list[float]]:
    if bins_text is None:
        raise ValueError("--bins is required with --dist")
    bins = [item.strip() for item in bins_text.split(",")]
    if len(bins) < 2 or any(not item for item in bins) or len(set(bins)) != len(bins):
        raise ValueError("--bins must contain at least two distinct, non-empty labels")
    try:
        dist = [validate_probability(float(item.strip()), "dist probability")
                for item in dist_text.split(",")]
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if len(dist) != len(bins):
        raise ValueError("--dist must have the same number of probabilities as --bins")
    # A practical CLI tolerance: permit ordinary decimal rounding, reject incoherent vectors.
    if abs(sum(dist) - 1.0) > 1e-9:
        raise ValueError("--dist probabilities must sum to 1 within 1e-9")
    return bins, dist


def _outcome_bin(bins: list[str], value: str) -> str:
    if value in bins:
        return value
    try:
        index = int(value)
    except ValueError as exc:
        raise ValueError(f"--outcome-bin must be a declared bin label or index (got {value!r})") from exc
    if not 0 <= index < len(bins):
        raise ValueError(f"--outcome-bin index must be in [0, {len(bins) - 1}]")
    return bins[index]


def cmd_add(args) -> int:
    try:
        if args.dist is None:
            prob = validate_probability(args.prob, "prob")
            bins = None
            dist = None
        else:
            prob = None
            bins, dist = _parse_distribution(args.bins, args.dist)
        baseline_prob = _validated_optional_prob(args.baseline_prob, "baseline_prob")
        crowd_prob = _validated_optional_prob(args.crowd_prob, "crowd_prob")
        ensemble_prob = _validated_optional_prob(args.ensemble_prob, "ensemble_prob")
        base_rate_prob = _validated_optional_prob(args.base_rate_prob, "base_rate_prob")
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    new_id = (max(_all_ids(args.log), default=0) + 1)
    rec = {
        "event": "forecast",
        "schema_version": 2,
        "id": new_id,
        "question": args.question,
        "created": _now(),
        "resolve_by": args.resolve_by,
        "rationale": args.rationale or "",
        "forecaster": args.forecaster,
        "baseline_prob": baseline_prob,
        "crowd_prob": crowd_prob,
        "ensemble_prob": ensemble_prob,
        "base_rate_prob": base_rate_prob,
        "tags": [t for t in (args.tags or "").split(",") if t],
        "status": "open",
        "outcome": None,
        "resolved": None,
    }
    if dist is None:
        rec["prob"] = prob
    else:
        rec["question_type"] = "ordered_distribution"
        rec["bins"] = bins
        rec["dist"] = dist
    append_jsonl(args.log, rec)
    detail = f"p={prob}" if dist is None else f"distribution over {len(bins)} bins"
    print(f"added forecast #{new_id}: {detail} — {args.question}")
    return 0


def cmd_resolve(args) -> int:
    rows = fold_forecast_log(args.log)
    for r in rows:
        if r["id"] == args.id:
            if r["status"] == "resolved":
                print(f"forecast #{args.id} is already resolved", file=sys.stderr)
                return 1
            if r["status"] == "superseded":
                print(f"cannot resolve superseded forecast (#{args.id}): {r['reason']}",
                      file=sys.stderr)
                return 1
            if "dist" in r:
                if args.outcome_bin is None:
                    raise ValueError("distribution forecasts require --outcome-bin")
                try:
                    outcome_bin = _outcome_bin(r["bins"], args.outcome_bin)
                except ValueError as exc:
                    raise ValueError(str(exc)) from exc
                rps = ranked_probability_score(r["dist"], r["bins"].index(outcome_bin))
                event = {
                    "event": "resolve", "schema_version": 2, "id": args.id,
                    "outcome_bin": outcome_bin, "resolved": _now(), "rps": rps,
                }
                append_jsonl(args.log, event)
                print(f"resolved #{args.id}: outcome_bin={outcome_bin}  rps={rps:.4f}")
                return 0
            if args.outcome is None:
                raise ValueError("binary forecasts require --outcome")
            try:
                outcome = validate_outcome(int(args.outcome))
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            brier = round(brier_score(float(r["prob"]), outcome), 4)
            event = {
                "event": "resolve",
                "schema_version": 2,
                "id": args.id,
                "outcome": outcome,
                "resolved": _now(),
                "prob": r["prob"],
                "brier": brier,
            }
            append_jsonl(args.log, event)
            print(f"resolved #{args.id}: outcome={outcome}  brier={brier}")
            return 0
    print(f"no forecast with id {args.id}", file=sys.stderr)
    return 1


def cmd_update(args) -> int:
    try:
        new_prob = validate_probability(args.prob, "prob")
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    for r in fold_forecast_log(args.log):
        if r["id"] == args.id:
            if r["status"] == "resolved":
                print(f"cannot update a resolved forecast (#{args.id})", file=sys.stderr)
                return 1
            if r["status"] == "superseded":
                print(f"cannot update superseded forecast (#{args.id}): {r['reason']}",
                      file=sys.stderr)
                return 1
            if "dist" in r:
                print(f"distribution updates are not supported (#{args.id})", file=sys.stderr)
                return 1
            old_prob = r["prob"]
            event = {
                "event": "update",
                "schema_version": 2,
                "id": args.id,
                "prob": new_prob,
                "updated": _now(),
                "note": args.note,
                "rationale": args.rationale,
            }
            append_jsonl(args.log, event)
            print(f"updated #{args.id}: p={old_prob} -> {new_prob}")
            return 0
    print(f"no forecast with id {args.id}", file=sys.stderr)
    return 1


def cmd_supersede(args) -> int:
    for row in fold_forecast_log(args.log):
        if row["id"] != args.id:
            continue
        if row["status"] == "resolved":
            print(f"cannot supersede resolved forecast (#{args.id})", file=sys.stderr)
            return 1
        if row["status"] == "superseded":
            print(f"forecast #{args.id} is already superseded: {row['reason']}", file=sys.stderr)
            return 1
        event = {
            "event": "supersede",
            "schema_version": 2,
            "id": args.id,
            "status": "superseded",
            "reason": args.reason,
            "superseded": _now(),
        }
        if args.replaced_by is not None:
            event["replaced_by"] = args.replaced_by
        append_jsonl(args.log, event)
        replacement = f"; replaced by #{args.replaced_by}" if args.replaced_by is not None else ""
        print(f"superseded #{args.id}: {args.reason}{replacement}")
        return 0
    print(f"no forecast with id {args.id}", file=sys.stderr)
    return 1


def cmd_list(args) -> int:
    rows = fold_forecast_log(args.log)
    try:
        reference = parse_datetime(args.as_of) if args.as_of else None
    except ValueError as exc:
        raise ValueError(f"invalid --as-of date: {args.as_of}") from exc
    if args.overdue or args.due is not None:
        if args.due is not None and args.due < 0:
            raise ValueError("--due must be >= 0")
        reference = reference or datetime.now(timezone.utc)
        reference_day = reference.date()
        rows = [r for r in rows if r["status"] == "open" and r.get("resolve_by")]
        if args.overdue:
            rows = [r for r in rows if parse_datetime(r["resolve_by"]).date() < reference_day]
        else:
            rows = [r for r in rows
                    if 0 <= (parse_datetime(r["resolve_by"]).date() - reference_day).days <= args.due]
        rows.sort(key=lambda r: parse_datetime(r["resolve_by"]))
        if not rows:
            if args.overdue:
                print("(no overdue forecasts)")
            else:
                print(f"(none due within {args.due} days)")
            return 0
    elif args.status:
        rows = [r for r in rows if r["status"] == args.status]
    if not rows:
        print("(no forecasts)")
        return 0
    for r in rows:
        if "dist" in r:
            oc = "?" if r.get("outcome_bin") is None else r["outcome_bin"]
            tags = (" [" + ",".join(r["tags"]) + "]") if r["tags"] else ""
            print(f"#{r['id']:>3}  dist={len(r['bins'])} bins  {r['status']:8}  out={oc}  "
                  f"by {r['resolve_by'] or '-'}{tags}\n       {r['question']}")
            continue
        oc = "?" if r["outcome"] is None else r["outcome"]
        tags = (" [" + ",".join(r["tags"]) + "]") if r["tags"] else ""
        hist = r.get("prob_history")
        upd = ""
        if hist and len(hist) > 1:
            upd = f"  (updated x{len(hist) - 1}: {hist[0]['prob']:.2f}->{r['prob']:.2f})"
        print(f"#{r['id']:>3}  p={r['prob']:.2f}  {r['status']:8}  out={oc}  "
              f"by {r['resolve_by'] or '-'}{tags}{upd}\n       {r['question']}")
    return 0


def cmd_score(args) -> int:
    resolved = fold_forecast_log(args.log)
    rows = [r for r in resolved if r.get("status") == "resolved"
            and r.get("outcome") is not None and r.get("prob") is not None]
    distributions = [r for r in resolved if r.get("status") == "resolved"
                     and r.get("rps") is not None and r.get("dist") is not None]
    if not rows and not distributions:
        print("no resolved forecasts yet")
        return 0
    if rows:
        bs = sum(brier_score(float(r["prob"]), int(r["outcome"])) for r in rows) / len(rows)
        print(f"resolved forecasts : {len(rows)}")
        print(f"Brier score        : {bs:.4f}")
    if distributions:
        mean_rps = sum(float(r["rps"]) for r in distributions) / len(distributions)
        print(f"resolved distributions : {len(distributions)}")
        print(f"Mean RPS               : {mean_rps:.4f}")
    print("run brier.py --log <path> for the full calibration decomposition.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=None,
                    help="path to JSONL log (default: $FORECAST_LOG, "
                         "else ~/.superforecasting/forecasts.jsonl)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="register a new forecast")
    a.add_argument("--question", required=True)
    add_kind = a.add_mutually_exclusive_group(required=True)
    add_kind.add_argument("--prob", type=float)
    add_kind.add_argument("--dist", help="comma-separated ordered-bin probabilities summing to 1")
    a.add_argument("--bins", help="comma-separated ordered labels; required with --dist")
    a.add_argument("--resolve-by", default=None)
    a.add_argument("--rationale", default=None)
    a.add_argument("--tags", default=None, help="comma-separated")
    a.add_argument("--forecaster", default=None, help="forecaster/model/person id")
    a.add_argument("--baseline-prob", type=float, default=None,
                   help="generic baseline probability for later skill scoring")
    a.add_argument("--crowd-prob", type=float, default=None,
                   help="crowd or local-median baseline probability")
    a.add_argument("--ensemble-prob", type=float, default=None,
                   help="ensemble probability available at forecast time")
    a.add_argument("--base-rate-prob", type=float, default=None,
                   help="outside-view base-rate probability")
    a.set_defaults(func=cmd_add)

    u = sub.add_parser("update", help="revise the probability of an open forecast")
    u.add_argument("--id", type=int, required=True)
    u.add_argument("--prob", type=float, required=True, help="revised probability in [0,1]")
    u.add_argument("--note", default=None, help="what triggered the update")
    u.add_argument("--rationale", default=None, help="optional updated rationale")
    u.set_defaults(func=cmd_update)

    r = sub.add_parser("resolve", help="resolve an existing forecast")
    r.add_argument("--id", type=int, required=True)
    resolve_kind = r.add_mutually_exclusive_group(required=True)
    resolve_kind.add_argument("--outcome", type=int, choices=[0, 1])
    resolve_kind.add_argument("--outcome-bin", help="declared ordered-bin label or zero-based index")
    r.set_defaults(func=cmd_resolve)

    x = sub.add_parser("supersede", help="retire an invalid or withdrawn forecast without scoring")
    x.add_argument("--id", type=int, required=True)
    x.add_argument("--reason", required=True, help="why the forecast cannot be resolved normally")
    x.add_argument("--replaced-by", type=int, default=None,
                   help="optional id of the replacement forecast")
    x.set_defaults(func=cmd_supersede)

    l = sub.add_parser("list", help="list forecasts")
    l.add_argument("--status", choices=["open", "resolved", "superseded"], default=None)
    due_filter = l.add_mutually_exclusive_group()
    due_filter.add_argument("--overdue", action="store_true", help="show open forecasts past resolve_by")
    due_filter.add_argument("--due", type=int, metavar="N", help="show open forecasts due within N days")
    l.add_argument("--as-of", default=None, metavar="DATE", help="UTC reference date for --due/--overdue")
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("score", help="quick Brier over resolved forecasts")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args(argv)
    if args.log is None:
        args.log = default_log_path()
    try:
        return args.func(args)
    except ValueError as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
