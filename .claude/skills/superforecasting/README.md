# Superforecasting — a cross-harness forecasting playbook + toolkit

[![tests](https://github.com/sloik/superforecasting/actions/workflows/tests.yml/badge.svg)](https://github.com/sloik/superforecasting/actions/workflows/tests.yml)

> Ask any agent "how likely is X?" and get a calibrated, tool-verified, Brier-scored answer —
> from one Markdown playbook and six dependency-free Python scripts.

An "automated" way to answer forecasting and consequential-decision questions the way
Philip Tetlock's **superforecasters** do. It is a **skill** (a Markdown playbook + a set of
stdlib-only Python tools) designed to run identically across Claude, Codex, Aider,
or any agent that can read Markdown and shell out to `python3`.

Built for a Good Judgment-style project: discussion-driven reasoning, tool-verified claims,
real statistics, an **ensemble** mode (run the same question through several agents and pool
the answers), and a **calibration log** so you get measurably better over time.

## What's inside

```
Superforecasting/
├── SKILL.md                     ← the playbook (start here; every harness loads this)
├── references/
│   ├── ten-commandments.md      Tetlock's 10 commandments, expanded
│   ├── workflow.md              per-question checklist
│   ├── aggregation-and-scoring.md   the math: geo-mean of odds, extremizing, Brier
│   ├── rationale-quality.md     checklist for judging the written rationale
│   ├── cognitive-biases.md      bias watchlist + debiasing moves
│   └── question-typing.md       triage: clocklike vs cloudlike, question shapes
├── templates/
│   ├── question-packet.md       portable packet to run across agents
│   ├── forecast-record.md       single-forecast writeup
│   └── ensemble-compile.md      combine multiple agents' answers
├── scripts/                     stdlib-only Python 3, run with --help
│   ├── baserate.py              outside view: reference class → base rate + interval
│   ├── bayes.py                 inside view: Bayesian odds update
│   ├── montecarlo.py            Fermi-ized simulation of range-valued quantities
│   ├── aggregate.py             pool many forecasts (geo-mean of odds + extremizing)
│   ├── brier.py                 Brier score + calibration/resolution decomposition
│   └── forecastlog.py           record → resolve → score forecasts (JSONL ledger)
└── tests/                       stdlib unittest suite
```

Your calibration ledger is **not** stored in this folder — see
[Where your forecasts are stored](#where-your-forecasts-are-stored).

## The workflow in one line
**Triage → Fermi-ize → base rate → adjust (Bayes) → many eyes (ensemble) → number → update-triggers → log & score.**

See `SKILL.md` for the full pipeline and `references/workflow.md` for the paste-in checklist.

## Install

There is nothing to build and nothing to install — clone it and it works.

```bash
git clone https://github.com/sloik/superforecasting.git
cd superforecasting
python3 -m unittest discover -s tests     # optional sanity check; stdlib only
```

**Claude (Claude Code / Cowork)** — symlink the checkout into your skills directory so it is
auto-discovered:

```bash
mkdir -p ~/.claude/skills
ln -s "$PWD" ~/.claude/skills/superforecasting
```

**Any other agent (Codex, Aider, …)** — point it at `SKILL.md` and let it call the scripts;
they are plain CLIs with `--help`:

```bash
python3 scripts/baserate.py --hits 3 --total 11
```

## Quick start

```bash
cd scripts

# Outside view: 3 of 11 comparable cases resolved YES
python3 baserate.py --hits 3 --total 11

# Inside view: update that anchor with two pieces of evidence
python3 bayes.py --prior 0.31 --lr 2.0 --lr 0.8

# Fermi-ized numeric question via Monte Carlo
python3 montecarlo.py --json '{"trials":50000,"vars":{"x":{"dist":"triangular","lo":10,"mode":20,"hi":40}},"expr":"x","threshold":25,"direction":">"}'

# Ensemble: pool four agents' probabilities
python3 aggregate.py 0.6 0.7 0.55 0.8

# Ensemble with track-record and recency weighting
python3 aggregate.py --p 0.6 --forecaster alice --p 0.7 --forecaster bob --track-log "$FORECAST_LOG"
python3 aggregate.py --p 0.6 --date 2026-07-01 --p 0.7 --date 2026-07-12 --half-life-days 14

# Log a forecast, update it when a trigger fires, then (later) resolve and score it
python3 forecastlog.py add --question "Will X ship by 2026-12-31?" --prob 0.35 --resolve-by 2026-12-31 --tags product --baseline-prob 0.30
python3 forecastlog.py update --id 1 --prob 0.45 --note "positive earnings surprise"
# If the question is withdrawn/invalid, retire it without changing its probability or scoring it
python3 forecastlog.py supersede --id 1 --reason "question withdrawn" --replaced-by 3
python3 forecastlog.py list --overdue
python3 forecastlog.py list --due 14
python3 forecastlog.py resolve --id 1 --outcome 0
python3 brier.py --baseline-field baseline_prob
# Optional: also score how long each belief was held before resolution
python3 brier.py --time-weighted

# Log an ordered numeric/range distribution; it is RPS-scored separately from binary Brier
python3 forecastlog.py add --question "2027 units?" --bins "under-10,10-20,over-20" --dist "0.2,0.5,0.3" --resolve-by 2027-12-31
python3 forecastlog.py resolve --id 2 --outcome-bin 10-20
```

Requirements: **Python 3.8+ only** (standard library — no pip installs).

## Where your forecasts are stored

The ledger lives **outside** this folder, so the skill never writes into its own checkout —
it keeps working when installed read-only or symlinked, and one setting points every harness
on the machine at the same record. Resolution order, highest priority first:

1. `--log <path>` on any command
2. `$FORECAST_LOG`
3. `~/.superforecasting/forecasts.jsonl` (created on first write)

```bash
# recommended: pick a durable location once, in your shell profile
export FORECAST_LOG="$HOME/forecasts/calibration.jsonl"
```

`aggregate.py --track-log` is the exception: it always needs an explicit path, because it is an
optional weighting *input* rather than the ledger the toolkit reads and writes by default.

## Using it in each harness

The playbook is plain Markdown and the tools are plain CLIs, so portability is just
"point the agent at `SKILL.md` and let it run `scripts/*.py`".

- **Claude (Cowork / Claude Code):** the folder is already a valid skill — `SKILL.md` has the
  `name`/`description` frontmatter. Symlink or copy it into your skills directory
  (e.g. `~/.claude/skills/superforecasting`) to have it auto-discovered, or just open the
  project and reference `SKILL.md`.
- **Codex / Aider / other agents:** these read a project context file. `AGENTS.md`
  (included) points them at `SKILL.md` and the tools. Add the repo to the agent's context, or
  paste `SKILL.md` into the system prompt. The scripts run the same everywhere.
- **Ensemble across all of them:** fill `templates/question-packet.md`, give the identical
  packet to each agent independently, collect their `FINAL PROBABILITY` lines, and pool with
  `scripts/aggregate.py`. Use `templates/ensemble-compile.md` to record and reconcile.

## Design choices
- **Stdlib-only Python** so it runs anywhere with zero setup; the language is an implementation
  detail — any harness can call the CLIs.
- **Outside-view-first, tool-verified, logged-and-scored** are enforced as discipline rules in
  `SKILL.md`, because those are the habits the research shows actually move accuracy.
- **Geometric mean of odds** as the default pool (external Bayesianity), with extremizing
  available but off by default (it hurts on correlated/small panels).
- **Append-only forecast ledger**: new forecasts, belief updates, resolutions, and supersessions
  are JSONL events; readers fold them into current snapshots for listing/scoring. Updates preserve
  the full `prob_history` trajectory and resolve on the latest probability. Supersessions preserve
  that trajectory but are excluded from Brier/RPS scoring; do not move probability for bookkeeping.
- **Ordered numeric distributions**: distributions over declared bins use Ranked Probability
  Score (RPS), which rewards near misses, and are reported separately from binary Brier.
- **Rationale quality matters**: the final prose is checked for markers such as statistical
  reasoning, sourcing, clashing forces, updating logic, and probability/rationale alignment.

## Sources
Tetlock & Gardner, *Superforecasting* (2015); Good Judgment Project / IARPA ACE; forecast
aggregation literature (geo-mean of odds, extremizing); Brier (1950) & Murphy (1973)
decomposition. Full links in the `references/` files.

`references/ten-commandments.md` is a **paraphrased summary with commentary**, not the book's
text. Read the book for the original.

## Contributing / local setup
If you clone this repo to work on it, run `tools/install-hooks.sh` **before your first commit** —
git does not clone `.git/hooks`, so a fresh checkout has no commit guard until you install it.

## License
MIT — see [LICENSE](LICENSE).
