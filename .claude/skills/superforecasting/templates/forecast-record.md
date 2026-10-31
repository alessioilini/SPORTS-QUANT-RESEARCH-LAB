# Forecast Record — <question title>

**Date made:** <YYYY-MM-DD>  **Resolves:** <YYYY-MM-DD>  **Tags:** <...>

## Question & resolution criteria
<exact wording; what counts as YES; who judges>

## 1. Triage
Difficulty: clocklike / **Goldilocks** / cloudlike — <why>

## 2. Decomposition (Fermi)
- Sub-question A: <knowable? estimate/source>
- Sub-question B: <...>

## 3. Outside view (anchor)
- Reference class: <...>
- Counts: <k> of <n>  →  base rate `baserate.py`: <p, interval>
- Sources (verified ≥2): <links>

## 4. Inside view (adjustments)
| Factor | Direction | LR | Source / reasoning |
|--------|-----------|----|--------------------|
|        | up/down   |    |                    |

Bayes result (`bayes.py`): prior <..> → posterior <..>

## 5. Perspectives / ensemble
- Framing 1: <p>
- Framing 2 / other agents: <p, p, ...>
- Pooled (`aggregate.py`): <p>   spread: <..>

## 6. Final probability
**<0.NN>**  (range Qs: p5 <..> / p50 <..> / p95 <..>)

### Range/distribution variant
For an ordered numeric/range question, declare bins and probabilities instead of one scalar:

| Ordered bin | Probability |
|-------------|-------------|
| <lower range> | <0.NN> |
| <middle range> | <0.NN> |
| <upper range> | <0.NN> |

Bins must be distinct, ordered, and sum to 1. Resolve to one declared bin; the ledger records
Ranked Probability Score (RPS), which gives partial credit to nearby bins.

## 6.4 Why right / why wrong (mandatory with every number)
- Why I think this is right: <bullets, each tied to a verified fact or computed anchor>
- Why I might be wrong: <bullets, each a concrete failable assumption + direction it moves the forecast>

## 6.5 Rationale quality
- Positive markers present: <statistical reasoning / sourcing / clashing forces / update logic / timing / ...>
- Red flags checked: <gut-based / extreme confidence / confirmation bias / simplification / mismatch / ...>
- Rationale/probability alignment: <why the prose supports the final number>

## 7. Update triggers
- Would raise if: <signal> (~LR <..>)
- Would lower if: <signal> (~LR <..>)
- Revisit on: <date/event>

## 8. Resolution (fill in later)
- Outcome: <0/1>  on <date>
- Brier: <(p-o)^2>
- Lesson (perpetual beta): <what I'd do differently>

---
Logged: `python3 scripts/forecastlog.py add --question "..." --prob 0.NN --resolve-by YYYY-MM-DD --rationale "..." --tags ... --baseline-prob 0.NN`

Distribution logged: `python3 scripts/forecastlog.py add --question "..." --bins "low,mid,high" --dist "0.2,0.5,0.3" --resolve-by YYYY-MM-DD --rationale "..." --tags ...`; resolve with `python3 scripts/forecastlog.py resolve --id N --outcome-bin mid`
