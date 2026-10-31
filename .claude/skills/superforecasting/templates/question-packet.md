# Question Packet — <short title>

> Portable, harness-agnostic. Fill this out once, then hand the SAME packet to several
> agents (Claude, Codex, Aider) or people. Collect each filled-in copy and pool
> the probabilities with `scripts/aggregate.py`. Independent inputs are the point — don't
> let responders see each other's answers first.

## 1. Question (exact wording)
<one sentence, resolvable>

## 2. Resolution criteria
- **YES means:** <precise, observable condition>
- **Judged by:** <source / authority / dataset>
- **Resolution date:** <YYYY-MM-DD>
- **Edge cases:** <cancellation, ambiguity, partial outcomes>

## 3. Background / known facts (shared context, cite sources)
- <fact — source>
- <fact — source>

## 4. What each responder should return
Please work the pipeline (triage → Fermi-ize → base rate → adjust → weigh → number) and return:

    RESPONDER:        <agent/model/person id>
    BASE RATE:        <p, with the reference class you used>
    KEY DRIVERS UP:   <factors + rough LRs>
    KEY DRIVERS DOWN: <factors + rough LRs>
    FINAL PROBABILITY: <single number 0-1>   (for range Qs: p5 / p50 / p95)
    CONFIDENCE:       <low/med/high> and why
    ONE-LINE RATIONALE: <...>
    WHY RIGHT:        <1-3 bullets, fact-tied>
    WHY WRONG:        <1-3 bullets, failable assumptions + direction>
    RATIONALE QUALITY: <positive markers + any unresolved red flags>
    WHAT WOULD CHANGE MY MIND: <signals + direction>

## 5. Collected responses (paste each responder's block here)
```
RESPONDER: ...
FINAL PROBABILITY: ...
...
```

## 6. Pooling
    python3 scripts/aggregate.py <p1> <p2> <p3> ...        # geometric mean of odds
    # add --extremize 1.5..2.5 ONLY if responders are many, diverse, and independent
    # NOTE: an all-LLM panel is correlated by default (shared training data) — do not extremize it

Pooled probability: __________   (investigate any spread > 0.4 before trusting it)
