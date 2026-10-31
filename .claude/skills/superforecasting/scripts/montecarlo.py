#!/usr/bin/env python3
"""Monte Carlo simulation for Fermi-ized forecasts.

Once you have decomposed a question into sub-quantities (the Fermi move), you rarely
know each one exactly — you know a plausible RANGE.  This tool samples each uncertain
input from a distribution, combines them with your formula thousands of times, and
reports the resulting distribution of the answer.  It turns "I roughly think..." into
a calibrated probability with honest error bars.

Spec (JSON, via --spec file.json or --json '...'):

    {
      "trials": 100000,
      "vars": {
        "market_size":  {"dist": "lognormal", "mean": 1e6, "sd_factor": 1.5},
        "our_share":    {"dist": "triangular", "lo": 0.01, "mode": 0.05, "hi": 0.15},
        "price":        {"dist": "normal", "mean": 20, "sd": 4},
        "launches":     {"dist": "bernoulli", "p": 0.7}
      },
      "expr": "market_size * our_share * price * launches",
      "threshold": 500000,
      "direction": ">"
    }

Distributions
    normal      {mean, sd}
    uniform     {lo, hi}
    triangular  {lo, mode, hi}          # great default for expert ranges
    lognormal   {mean, sd_factor}       # sd_factor = multiplicative 1-sigma (e.g. 2 = "could be 2x off")
                {mu, sigma}             # or give log-space params directly
    beta        {a, b}                  # for probabilities/proportions
    bernoulli   {p}                     # 0/1 event
    point       {value}                 # a fixed constant

Output: mean, sd, and p5/p25/p50/p75/p95; plus P(expr direction threshold) if given.

The formula is evaluated in a restricted namespace (only your variables + basic math).
"""
from __future__ import annotations
import argparse
import ast
import json
import math
import random
import statistics
import sys

from forecast_common import validate_non_negative, validate_positive, validate_probability

_SAFE_FUNCS = {k: getattr(math, k) for k in
               ("sqrt", "log", "log10", "exp", "sin", "cos", "tan", "floor", "ceil", "fabs", "pi", "e")}
_SAFE_FUNCS.update({"min": min, "max": max, "abs": abs})

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
)


def compile_expr(expr: str, var_names: set[str]):
    tree = ast.parse(expr, mode="eval")
    allowed_names = set(_SAFE_FUNCS) | var_names
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed expression element: {node.__class__.__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"unknown name in expression: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
                raise ValueError("only whitelisted math functions may be called")
    return compile(tree, "<expr>", "eval")


def sample(spec: dict) -> float:
    d = spec["dist"]
    if d == "normal":
        validate_non_negative(float(spec["sd"]), "normal sd")
        return random.gauss(spec["mean"], spec["sd"])
    if d == "uniform":
        if spec["lo"] > spec["hi"]:
            raise ValueError("uniform requires lo <= hi")
        return random.uniform(spec["lo"], spec["hi"])
    if d == "triangular":
        if not spec["lo"] <= spec["mode"] <= spec["hi"]:
            raise ValueError("triangular requires lo <= mode <= hi")
        return random.triangular(spec["lo"], spec["hi"], spec["mode"])
    if d == "lognormal":
        if "mu" in spec:
            validate_non_negative(float(spec["sigma"]), "lognormal sigma")
            return random.lognormvariate(spec["mu"], spec["sigma"])
        # parameterise by median (`mean`) and a multiplicative 1-sigma factor
        validate_positive(float(spec["mean"]), "lognormal mean")
        validate_positive(float(spec["sd_factor"]), "lognormal sd_factor")
        mu = math.log(spec["mean"])
        sigma = math.log(spec["sd_factor"])
        return random.lognormvariate(mu, sigma)
    if d == "beta":
        validate_positive(float(spec["a"]), "beta a")
        validate_positive(float(spec["b"]), "beta b")
        return random.betavariate(spec["a"], spec["b"])
    if d == "bernoulli":
        validate_probability(float(spec["p"]), "bernoulli p")
        return 1.0 if random.random() < spec["p"] else 0.0
    if d == "point":
        return float(spec["value"])
    raise ValueError(f"unknown dist: {d}")


def run(model: dict) -> dict:
    trials = int(model.get("trials", 50000))
    if trials <= 0:
        raise ValueError("trials must be > 0")
    varsp = model["vars"]
    if not isinstance(varsp, dict) or not varsp:
        raise ValueError("vars must be a non-empty object")
    expr = model["expr"]
    code = compile_expr(expr, set(varsp))
    results = []
    for _ in range(trials):
        env = {name: sample(s) for name, s in varsp.items()}
        env.update(_SAFE_FUNCS)
        results.append(float(eval(code, {"__builtins__": {}}, env)))
    results.sort()

    def pct(q):
        i = min(len(results) - 1, max(0, int(round(q * (len(results) - 1)))))
        return results[i]

    out = {
        "trials": trials,
        "mean": statistics.fmean(results),
        "sd": statistics.pstdev(results),
        "p5": pct(0.05), "p25": pct(0.25), "p50": pct(0.50),
        "p75": pct(0.75), "p95": pct(0.95),
        "min": results[0], "max": results[-1],
    }
    if "threshold" in model:
        thr = model["threshold"]
        direction = model.get("direction", ">")
        if direction == ">":
            hit = sum(1 for r in results if r > thr)
        elif direction == ">=":
            hit = sum(1 for r in results if r >= thr)
        elif direction == "<":
            hit = sum(1 for r in results if r < thr)
        elif direction == "<=":
            hit = sum(1 for r in results if r <= thr)
        else:
            raise ValueError("direction must be one of > >= < <=")
        out["prob"] = hit / len(results)
        out["condition"] = f"expr {direction} {thr}"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec", help="path to a JSON spec file")
    g.add_argument("--json", help="inline JSON spec string")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    args = ap.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    raw = open(args.spec).read() if args.spec else args.json
    try:
        model = json.loads(raw)
        out = run(model)
    except (OSError, json.JSONDecodeError, KeyError, SyntaxError, ValueError) as exc:
        ap.error(str(exc))

    print(f"trials : {out['trials']}")
    print(f"mean   : {out['mean']:.4g}   sd: {out['sd']:.4g}")
    print("percentiles:")
    for k in ("p5", "p25", "p50", "p75", "p95"):
        print(f"   {k:>4}: {out[k]:.4g}")
    print(f"range  : [{out['min']:.4g}, {out['max']:.4g}]")
    if "prob" in out:
        print(f"\nP({out['condition']}) = {out['prob']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
