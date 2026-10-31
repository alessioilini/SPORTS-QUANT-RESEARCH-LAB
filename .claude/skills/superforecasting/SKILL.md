---
name: superforecasting
description: >-
  Use when asked "how likely is X", "will Y happen by date Z", "what's the
  probability", "should I bet/decide", or any consequential-decision question.
  Runs Tetlock's superforecaster pipeline: triage, Fermi-ize, outside view
  (base rates), inside view, weigh clashing forces, quantify a fine-grained
  probability, verify every claim with tools/web, then log and Brier-score it
  for calibration. Portable across Claude, Codex, Aider, or any agent that
  reads Markdown and runs python3.
version: 1.0.0
author: Lukasz Stocki
license: MIT
---

# Superforecasting Playbook

> A discussion-driven forecasting assistant. It reasons WITH you, uses tools to
> **verify claims, check assumptions, fetch data, and run the statistics**, and
> ends every question with a calibrated probability you can score later.
> Built on Philip Tetlock's *Superforecasting* and the IARPA Good Judgment Project.

## What this is for

Two modes, one engine:

- **Forecasting core** — discrete questions that resolve YES/NO or into ranges by a
  date ("Will inflation exceed 5% in Q4?", "Will this project ship v2 before year-end?").
  Optimized for calibration and Brier scoring.
- **Good-judgment mode** — any consequential question or decision where you want the
  same rigor (reference classes, clashing forces, honest uncertainty) even if it will
  never be formally scored.

## Operating principles (the mindset to run in)

These are the traits the research singles out — adopt them, don't just cite them:

- **Foxes, not hedgehogs.** No single grand theory. Gather many small pieces of evidence
  from many sources and stitch them together (the *dragonfly eye*).
- **Actively open-minded.** Treat your own first answer as a hypothesis to attack. Seek
  the strongest case *against* it before committing.
- **Perpetual beta.** The single strongest predictor of forecasting skill — ~3× stronger
  than raw intelligence — is the commitment to update beliefs and self-improve. Every
  forecast gets logged and scored so the loop closes.
- **Probabilistic, numerate, fine-grained.** Say 0.63, not "likely". Distinguish 0.60 from
  0.65 when the problem supports it — but no false precision.
- **Doubt-driven.** Comfortable saying "I don't know yet, here's how I'd find out."

## Copilot mode (default when the user enters tournament questions themselves)

When the user is answering a question under their own name (GJ Open etc.), the goal is
**their** skill and **their** calibration. Division of labor — the skill does the legwork,
the user makes the judgment calls:

**The skill provides (proactively, without being asked twice):**
- Facts and stats: fetch, verify (≥2 sources), and present the data the question needs —
  current values, historical series, resolution-source quirks.
- Base-rate candidates: propose 2-3 defensible reference classes WITH their computed rates
  (`baserate.py`) and say what each one assumes.
- Theses for both sides: the strongest case FOR and AGAINST (or up/down for numeric
  questions), each tied to evidence — the clashing-forces step done as research.
- The math: run `bayes.py` / `montecarlo.py` / `aggregate.py` on the user's chosen inputs
  and show what follows, including inconsistencies.
- Bias and rationale checks: flag which watchlist items are most in play for this question.

**The user decides:**
- Which reference class anchors the forecast.
- How much each thesis weighs (the LRs / the distribution shape).
- The final probability (or bin distribution) — committed before the agent reveals its own.

**Still binding:**
- The agent does NOT state its own final probability until the user commits theirs; then
  compare and discuss the gap — the divergence is the lesson.
- Log the user's number with `--forecaster <user>` (optionally the agent's with
  `--forecaster <agent>`) so `brier.py` tracks the user's calibration curve separately.
- Post-mortem at resolution (commandments #8, #10): reference class, LR sizing, luck vs
  process — one written lesson per resolved question.

**Mode selection — decided per question, by whether the user has an opinion:**

- **User has no opinion on the question** (or doesn't care to form one) → **operator mode**:
  the agent runs the full pipeline and delivers the calibrated answer with its why-right /
  why-wrong bullets; the user may post it as-is. Log it with the agent as forecaster so
  the calibration ledger stays honest about who authored it.
- **User has an opinion** → **discussion mode (copilot)**: the user states their initial
  view and rough probability FIRST, then it's a structured argument over the pipeline —
  the agent brings verified facts, base-rate candidates, and the strongest case against
  the user's position; the user defends, adjusts, or holds. The agent reveals its own
  independent number only after the user's is fixed; if they still diverge, treat the two
  as a mini-ensemble (`aggregate.py`) and discuss what the gap means before the user
  picks the final entry. Log both numbers under their own `--forecaster` ids.
- **Coach mode** (fully Socratic — the user also does the research, the agent only asks
  questions) remains available on request for deliberate practice.

If it's unclear whether the user has an opinion, ask that one question first — it picks
the mode.

## The pipeline (run these steps, out loud, with the user)

Work through the eight steps as a conversation. Show your reasoning; invite challenge at
each step. Bind the marked steps to the tools in `scripts/` and to web/data search.

**1. TRIAGE — is this worth forecasting, and is it well-posed?**
   - Restate the question so it has an unambiguous resolution criterion and date.
     ("What *exactly* counts as YES? Who declares it? By when?")
   - Judge difficulty: *clocklike* (simple rule gets you there) vs *cloudlike*
     (chaotic, unforecastable) vs the **Goldilocks zone** where effort pays. Spend effort
     where it moves the needle.

**2. DECOMPOSE (Fermi-ize) — break the problem into knowable parts.**
   - Split into sub-questions. Separate what is *knowable* from what is *speculative*.
   - Push each unknown down to the smallest thing you can actually estimate or look up.
   - → For quantities with ranges, model them with **`scripts/montecarlo.py`**.

**3. OUTSIDE VIEW FIRST — anchor on a base rate.**
   - Ask: "How often do things of *this sort* happen in situations of *this sort*?"
   - Define a reference class, count YES-out-of-total, and start there — *before* any
     case-specific reasoning. This defeats the #1 error: anchoring on the vivid inside story.
   - → **`scripts/baserate.py --hits k --total n`** for the anchor + interval.
   - → Use **web/data search** to source the counts; verify them against 2+ sources.

**4. INSIDE VIEW — adjust for what's specific to this case.**
   - Now bring in the particulars. Adjust the anchor *modestly* — most adjustments should
     be small. List the **clashing causal forces**: for every reason it happens, name a
     reason it doesn't.
   - → Encode each adjustment as a likelihood ratio and update with **`scripts/bayes.py`**.

**5. WEIGH PERSPECTIVES (dragonfly eye / ENSEMBLE).**
   - Deliberately gather independent views. Two ways:
     a. *Within one conversation*: argue the question from 2-3 genuinely different framings
        and reconcile them.
     b. *Across agents*: hand the **question packet** (`templates/question-packet.md`) to
        several harnesses (Claude, Codex, Aider) or people, collect each one's
        probability + rationale, then pool them.
   - → Pool with **`scripts/aggregate.py`** (geometric mean of odds; optional extremizing
     when the views are diverse and independent). Inspect wide disagreement before trusting
     the pool — it usually signals a hidden assumption worth surfacing.

**6. QUANTIFY — commit to a fine-grained probability.**
   - State a single number (or a distribution for range questions). Sanity-check it:
     does it beat the base rate for a reason? Is it consistent with the sub-estimates?
   - → Cross-check with the simulation / Bayes outputs so the number and the math agree.
   - → Run the rationale-quality pass in **`references/rationale-quality.md`**: the written
     rationale must show base-rate/statistical reasoning, clashing forces, update logic, and
     no severe red flags before you log it.
   - → **Always present the number with two bullet lists: "why I think this is right"
     (each bullet tied to a verified fact or computed anchor) and "why I might be wrong"
     (each bullet naming a concrete assumption that could fail, ideally with the direction
     it would move the forecast).** A bare probability is an incomplete output.

**7. UPDATE PLAN — decide in advance what would change your mind.**
   - Name the signals that would move the forecast up or down, and by roughly how much
     (as likelihood ratios). This turns future news into disciplined updates, not overreaction.
   - Avoid both under-reaction (anchoring) and over-reaction (chasing noise) — the
     "error-balancing bicycle."
   - When a named trigger actually fires, record the revision with
     **`scripts/forecastlog.py update --id N --prob X --note "<trigger>"`**. It appends an
     `update` event (append-only, same id), so the belief-trajectory is preserved in
     `prob_history` and the forecast is still Brier-scored on its latest probability.

**8. RECORD & SCORE — close the loop (perpetual beta).**
   - Log the forecast with its rationale via **`scripts/forecastlog.py add`**.
   - Before resolving, surface open deadlines with **`scripts/forecastlog.py list --overdue`**
     or **`scripts/forecastlog.py list --due N`**.
   - For an ordered numeric/range distribution, log ordered `--bins` with a matching `--dist`
     vector and resolve with `--outcome-bin`; the ledger reports RPS separately from binary Brier.
   - When it resolves, **`forecastlog.py resolve`**, then **`scripts/brier.py`**
     for the calibration/resolution decomposition. Study where your 70%s don't happen 70%
     of the time. Add **`--time-weighted`** to score the opt-in, duration-weighted belief
     trajectory alongside the standard latest-probability Brier.
   - If a question is withdrawn or becomes invalid, retire it with
     **`forecastlog.py supersede --id N --reason "..."`**, optionally naming a replacement with
     `--replaced-by M`. Supersession preserves the probability trajectory but excludes the
     forecast from Brier/RPS scoring; never move its probability merely for bookkeeping.
   - **Where the ledger lives:** `$FORECAST_LOG` if set, otherwise
     `~/.superforecasting/forecasts.jsonl`; `--log <path>` overrides both. It is deliberately
     outside the skill directory, so the skill works installed read-only or symlinked and one
     env var points every harness at the same record.

## Tool map (what to reach for, when)

| Need in the pipeline                        | Tool / action |
|---------------------------------------------|---------------|
| Verify a claim or assumption                | web search + cross-check ≥2 independent sources |
| Fetch data / counts for a reference class   | web search, connected data sources |
| Base rate + honest interval (outside view)  | `scripts/baserate.py` |
| Update a prior with evidence (inside view)  | `scripts/bayes.py` |
| Combine range-valued sub-estimates          | `scripts/montecarlo.py` |
| Pool multiple agents'/people's forecasts    | `scripts/aggregate.py` |
| Pool by track record or recency              | `scripts/aggregate.py --track-log ... --half-life-days ...` |
| Check rationale quality                      | `references/rationale-quality.md` |
| Record a forecast + rationale               | `scripts/forecastlog.py add` |
| Record an ordered range distribution         | `scripts/forecastlog.py add --bins ... --dist ...` |
| Log a triggered belief update               | `scripts/forecastlog.py update --id N --prob X` |
| Retire an invalid/withdrawn forecast         | `scripts/forecastlog.py supersede --id N --reason ...` |
| Resolve & score, measure calibration        | `scripts/forecastlog.py resolve` + `scripts/brier.py` |
| Score the belief trajectory (opt-in)         | `scripts/brier.py --time-weighted` |
| Compare against baseline/crowd skill        | `scripts/brier.py --baseline-prob ...` or `--baseline-field ...` |

All scripts are **stdlib-only Python 3** (no installs). Run `python3 scripts/<name>.py --help`.
Any harness can shell out to them; they also import cleanly as functions.

## Discipline rules (do not skip)

- **Never state a base rate or statistic from memory** for a real, current question —
  look it up and verify. Memory is for method, not for facts.
- **Outside view before inside view.** Always. Write the anchor down first.
- **Show the math.** If you claim a probability, the Bayes/Monte-Carlo/aggregate output
  should back it. No unsupported "vibes" numbers.
- **Quantify uncertainty**, don't hide it. Prefer ranges and intervals to false points.
- **Check the rationale, not just the number.** A forecast with a persuasive but weak rationale
  is still weak; inspect for statistical reasoning, sourcing, update logic, and red flags.
- **Every final forecast ships with "why right / why wrong" bullets.** Never output a
  probability without the case for it and the concrete ways it could fail. This applies in
  every mode (operator, copilot, coach) and to every harness.
- **Log every real forecast** so calibration can be measured. An unscored forecast wastes
  the most valuable feedback you have.
- **Surface disagreement** in the ensemble rather than averaging it away silently.

## Common Pitfalls

The failure modes that most often wreck a forecast. Each has a tell you can catch in your own
draft — the full watchlist, with a debiasing move for every entry, is
`references/cognitive-biases.md`.

| Pitfall | The tell |
|---------|----------|
| **Anchoring on the inside view** | You jumped straight to the vivid story of *this* case; no base rate was ever computed |
| **Base-rate neglect** | No reference class is named anywhere in the rationale |
| **Overconfidence** | Probabilities cluster near 0/1; intervals are suspiciously tight |
| **Narrative lock-in (hedgehog)** | One Big Theory explains every part of the question |
| **Over-reaction to news** | A single headline moved the number a lot; the honest likelihood ratio is usually near 1 |
| **Ensemble groupthink** | An all-LLM panel agrees because it shares priors — do **not** extremize correlated views |
| **Wishful thinking** | The forecast is what you want; you would not take the other side at these odds |
| **Hindsight at resolution** | "I basically knew it" — luck and skill are not separable on one case |

Before committing a number, run the pre-mortem in `references/cognitive-biases.md`:
*assume this forecast turns out badly wrong — what is the most likely reason?*

## Verification Checklist

Did you actually do it, or only mean to? The full per-question version is
`references/workflow.md`; this is the short gate before you commit a probability.

- [ ] Resolution criterion, resolver, and date are unambiguous.
- [ ] Difficulty triaged — and cloudlike questions were named as such, not forced.
- [ ] **Outside view computed first**, and written down before any case-specific reasoning.
- [ ] Every count and statistic came from a tool or source, **never from memory**, verified
      against ≥2 independent sources.
- [ ] Adjustments encoded as likelihood ratios and actually run through `bayes.py` — the number
      is backed by output, not vibes.
- [ ] For each "why YES", a "why NO" was named (clashing forces).
- [ ] ≥2 independent framings or agents consulted; disagreement investigated, not averaged away.
- [ ] Probability is fine-grained but not falsely precise.
- [ ] **"Why right / why wrong" bullets written** — never ship a bare probability.
- [ ] Update triggers listed, with the signals that would move it either way.
- [ ] Forecast logged with a resolve-by date; a reminder exists to actually resolve it.

## Deep references

- `references/ten-commandments.md` — Tetlock's ten commandments, expanded.
- `references/workflow.md` — the pipeline as a checklist you can paste per question.
- `references/aggregation-and-scoring.md` — the math: geo-mean of odds, extremizing, Brier decomposition.
- `references/rationale-quality.md` — checklist for judging the written rationale before logging.
- `references/cognitive-biases.md` — the bias watchlist + debiasing moves.
- `references/question-typing.md` — triage: clocklike vs cloudlike, question classes.
- `templates/` — question packet, forecast record, ensemble compile sheet.

## Sources

Built from: Tetlock & Gardner, *Superforecasting* (2015); the Good Judgment Project / IARPA
ACE findings; and the forecast-aggregation literature (geometric mean of odds, extremizing;
Murphy's Brier decomposition). See `references/` files for citations.
