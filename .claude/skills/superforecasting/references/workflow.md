# Per-Question Workflow Checklist

Paste this into the conversation and work top to bottom. Tick each box.

> **Copilot mode (default for the user's own tournament entries):** the agent supplies
> facts, stats, base-rate candidates, and both-sides theses; the USER picks the reference
> class, sizes the adjustments, and commits the final number before the agent reveals its
> own. Log with `--forecaster <user>`. See SKILL.md § Copilot mode.

## 0. Frame
- [ ] Question restated with an **unambiguous resolution criterion** and **resolution date**.
- [ ] Who/what officially decides YES vs NO is identified.
- [ ] Units / thresholds fixed (e.g. ">5.0% CPI YoY", not "high inflation").

## 1. Triage
- [ ] Difficulty judged: clocklike / **Goldilocks** / cloudlike.
- [ ] Effort level chosen to match. (Cloudlike → say so, give the base rate, stop.)

## 2. Decompose (Fermi)
- [ ] Broken into sub-questions.
- [ ] Knowable vs speculative parts separated.
- [ ] Range-valued parts → drafted a `montecarlo.py` spec.

## 3. Outside view (anchor)
- [ ] Reference class defined (mechanism-based, not just topic-based).
- [ ] Counts sourced from web/data and **verified against ≥2 sources**.
- [ ] `baserate.py --hits k --total n` run → anchor + interval written down.
- [ ] Alternative reference class considered if one is defensible.

## 4. Inside view (adjust)
- [ ] Case-specific factors listed.
- [ ] **Clashing forces**: for each "why YES", a "why NO".
- [ ] Adjustments encoded as likelihood ratios → `bayes.py` update run.
- [ ] Adjustment magnitude sanity-checked (most should be modest).

## 5. Weigh perspectives (ensemble)
- [ ] ≥2 independent framings argued, or question packet sent to ≥2 agents/people.
- [ ] Forecasts pooled with `aggregate.py`.
- [ ] Wide disagreement (>0.4 spread) investigated, not averaged away.

## 6. Quantify
- [ ] Single fine-grained probability (or distribution) committed.
- [ ] Number reconciled with the Bayes / Monte-Carlo / pool outputs.
- [ ] No false precision; no unsupported "vibes" number.
- [ ] Rationale-quality pass complete: positive markers present, no severe red flags unresolved.
- [ ] **"Why right / why wrong" bullets written** — the case for the number (fact-tied) and
      the concrete assumptions that could fail (with direction). No bare probabilities.

## 7. Update plan
- [ ] Signals that would raise the forecast listed (with rough LRs).
- [ ] Signals that would lower it listed.
- [ ] Review cadence set (when to revisit before resolution).
- [ ] When a trigger fires: `forecastlog.py update --id N --prob X --note "<trigger>"` to log
      the revision (append-only; preserves `prob_history`, scores on the latest prob).

## 8. Record & score
- [ ] `forecastlog.py add` with probability + rationale + resolve-by date.
- [ ] Calendar reminder / note to resolve on the date.
- [ ] On resolution: `forecastlog.py resolve` + `brier.py` with baseline/crowd comparison + one lesson captured.

---

### One-line mental model
> **Triage → Fermi-ize → base rate → adjust → many eyes → number → update-triggers → log & score.**
