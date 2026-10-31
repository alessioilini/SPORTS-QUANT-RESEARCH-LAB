# Rationale Quality Checklist

Use this just before finalizing a forecast. It is a lightweight, human-readable version of
the newer Explanation Quality Marker idea: the written rationale itself should show markers
that tend to correlate with better forecasting.

## Positive markers

Score each 0 / 1 / 2: absent, partial, clear.

| Marker | What to look for |
|--------|------------------|
| **Resolution criteria** | The rationale uses the exact YES/NO threshold, source, and date. |
| **Statistical reasoning** | It names a reference class, base rate, trend, poll/model, or comparison data. |
| **Fact-based sourcing** | Key claims are verifiable and sourced, not asserted from memory. |
| **Causal reasoning** | It identifies mechanisms that would make the event happen or not happen. |
| **Clashing forces** | It gives serious arguments on both the YES and NO sides. |
| **Timing** | It accounts for how the probability changes as the resolution date approaches. |
| **Updating logic** | It explains what changed from the prior/base rate and why. |
| **Forecast alignment** | The final probability matches the rationale's strength; no 80% rationale for a 55% forecast. |
| **Open-mindedness** | It names what would change the forecast and treats the conclusion as revisable. |
| **Concrete uncertainty** | It distinguishes unknowns, ranges, and confidence instead of hiding them. |

## Red flags

If any are present, revise before logging.

| Red flag | Fix |
|----------|-----|
| **Gut-based** | Replace with a base rate, model, or explicit causal driver. |
| **Extreme confidence** | Pull away from 0/1 unless there is overwhelming evidence and strong resolution clarity. |
| **Confirmation bias** | Add the strongest counterargument and assign it a likelihood ratio. |
| **Simplification bias** | Decompose into drivers instead of forcing one explanation. |
| **Speculative terms** | Convert "could/might/maybe" into probabilities or evidence gaps. |
| **Inevitable/impossible framing** | Replace categorical language with odds and update triggers. |
| **Rationale/probability mismatch** | Adjust the number or rewrite the rationale until they agree. |
| **Unsupported authority** | Treat expert claims as evidence to weigh, not a substitute for reasoning. |

## Pass rule

Before recording a real forecast:

1. At least three positive markers should be clear.
2. No severe red flag should remain unresolved.
3. The rationale should be short enough to inspect but specific enough that another forecaster
   can challenge the base rate, drivers, and update triggers.

This is not a mechanical score to optimize. It is a guardrail against persuasive-but-weak prose.
