# Question Typing & Triage

Match the method to the question. Getting this right is Commandment #1.

## Difficulty spectrum
- **Clocklike** — governed by stable, well-understood mechanisms. A simple rule or model
  already gets you close (e.g. "will the sun rise", tomorrow's tide, a bond's coupon).
  → Give the answer, note the mechanism, don't over-invest.
- **Goldilocks zone** — enough structure that reasoning helps, enough uncertainty that it's
  not trivial. **This is where the full pipeline pays off.** Most tournament questions live here.
- **Cloudlike** — dominated by chaos / reflexivity / one-off contingency far out in time
  (e.g. "who wins the 2044 election"). Effort barely beats chance.
  → State the base rate, flag it as low-resolution, resist false precision, move on.

## Question shapes and how to handle each

| Shape | Example | Handling |
|-------|---------|----------|
| **Binary** | "Will X happen by <date>?" | Base rate → Bayes update → single p. Log & Brier-score. |
| **Multinomial** | "Which of A/B/C/D?" | Probabilities summing to 1; base-rate each; keep coherent. Unordered outcomes are not yet ledger-scored. |
| **Threshold / numeric** | "Will metric > T?" | Fermi-ize the metric, `montecarlo.py`, read off P(>T). |
| **Range / distribution** | "How many units in 2027?" | Use ordered bins when logging: `forecastlog.py add --bins "low,mid,high" --dist "0.2,0.5,0.3"`; resolve with `--outcome-bin`. Scored with RPS. |

For a numeric/range forecast, declare the bins in increasing order before forecasting. The
ledger preserves that order and uses it when scoring, so an adjacent-bin miss is less serious
than a distant one. This is deliberately distinct from an unordered multinomial question,
which needs a different scoring rule and is not yet supported by the ledger.
| **Conditional** | "If A, then how likely B?" | Forecast B within the A-world; state the condition explicitly. |
| **Decision (good-judgment mode)** | "Should I do X?" | Forecast the outcomes under each option; combine with your values/utility. Keep the forecast separate from the value judgment. |

## Resolution-criteria hygiene (do this before anything else)
A question you can't score is a question you can't learn from. Pin down:
1. **What counts as YES** — exact, observable, agreed in advance.
2. **The judge/source** — who or what data declares the outcome.
3. **The date** — when it resolves; what happens if it's ambiguous/unresolved.
4. **Edge cases** — partial outcomes, cancellations, redefinitions.

Ambiguity here is the most common reason forecasts "feel unfair" at resolution. Fix it up front.
