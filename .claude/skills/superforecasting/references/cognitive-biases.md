# Bias Watchlist + Debiasing Moves

Run this as a pre-mortem before committing a probability. For each, the tell and the fix.

| Bias | Tell in a forecast | Debiasing move |
|------|--------------------|----------------|
| **Anchoring on the inside view** | You jumped straight to the vivid story of *this* case | Compute the base rate FIRST (`baserate.py`); only then adjust |
| **Base-rate neglect** | No reference class was ever named | Force step 3 of the pipeline; name k out of n |
| **Availability** | Recent/dramatic events dominate the estimate | Ask "what's the long-run frequency?", not "what comes to mind?" |
| **Confirmation** | You only gathered evidence for your view | Actively list the strongest case AGAINST; assign it an LR |
| **Overconfidence** | Probabilities cluster near 0/1; ranges too tight | Widen intervals; check calibration table for over-confident bins |
| **Scope insensitivity** | Same number regardless of magnitude/time horizon | Re-derive with the horizon explicit; longer horizon → toward base rate |
| **Wishful thinking / motivated reasoning** | The forecast is what you *want* | Ask "would I bet on this at these odds against a sharp opponent?" |
| **Narrative/hedgehog lock-in** | One Big Theory explains everything | Fox it: list multiple clashing causal forces |
| **Over-reaction to news** | Big swing on a single headline | Convert the news into an LR; usually the honest LR is close to 1 |
| **Under-reaction / anchoring to prior forecast** | Ignoring genuine new signal to stay consistent | Pre-commit update triggers (step 7) so real signal moves you |
| **Conjunction fallacy** | A detailed scenario feels *more* likely than its parts | Remember P(A∧B) ≤ P(A); decompose and multiply |
| **Hindsight (at resolution)** | "I basically knew it" post-hoc | Score honestly with `brier.py`; luck ≠ skill on a single case |
| **Ensemble groupthink** | All agents agree because they share priors/training | Prefer genuinely independent sources; don't extremize correlated views |

## Pre-mortem prompt (say this before finalizing)
> "Assume this forecast turns out badly wrong. What's the most likely reason? Which of the
> biases above is most in play here?" — then adjust the probability and/or widen the interval.
