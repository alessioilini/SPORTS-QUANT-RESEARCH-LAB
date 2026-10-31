# Ensemble Compile Sheet — <question title>

Use after collecting independent responses to one question packet. Goal: a single pooled
forecast **plus** an understanding of *why* the panel disagrees.

## Responses
| Responder (agent/person) | Base rate used | Final p | Confidence | Rationale quality | One-line rationale |
|--------------------------|----------------|---------|------------|-------------------|--------------------|
| claude                   |                |         |            |                   |                    |
| codex                    |                |         |            |                   |                    |
| agent-3                  |                |         |            |                   |                    |
| aider                    |                |         |            |                   |                    |
| (human)                  |                |         |            |                   |                    |

## Pool
    python3 scripts/aggregate.py <p1> <p2> <p3> <p4> ...
    python3 scripts/aggregate.py --p <p1> --forecaster <id1> --p <p2> --forecaster <id2> --track-log "$FORECAST_LOG"

- Simple mean: <..>
- **Geo-mean of odds (use this): <..>**
- Extremized (only if diverse+independent, a=<..>): <..>
- Disagreement (max−min): <..>

## Disagreement analysis (the valuable part)
- Where do the rationales diverge? <different reference class? different key driver? different reading of resolution criteria?>
- Which rationales are weak despite confident probabilities? <rationale-quality red flags>
- Is any responder using information the others lack? <if so, that's signal, not noise>
- Are responders correlated (shared training/priors)? <if yes → do NOT extremize>

## Decision
- **Pooled forecast to record:** <p>
- Adjustments after disagreement analysis: <if any, and why>
- Log it: `forecastlog.py add ...`
