# Aggregation & Scoring — the math the assistant runs

## 1. Combining forecasts (the ensemble / crowd step)

You have several probabilities p₁…pₙ for the same YES/NO question (from different agents,
models, framings, or people). How to pool them into one?

### Recommended: geometric mean of ODDS
Convert each to odds oᵢ = pᵢ / (1 − pᵢ), take the geometric mean, convert back:

    o_pool  = (o₁ · o₂ · … · oₙ)^(1/n)
    p_pool  = o_pool / (1 + o_pool)

Equivalently, it is the inverse-logit of the mean of the log-odds (logits). This pool
satisfies **external Bayesianity** — pooling-then-updating equals updating-then-pooling —
and empirically beats the plain average of probabilities on real, small, diverse panels.
Use weights when some forecasters are more trustworthy (track record).

Why not the simple average of probabilities? It is dominated by the crowd's least-confident
members and can be logically inconsistent under updating.

### Optional: extremizing
Diverse independent forecasters each see part of the signal; the pool is systematically
*under-confident*. Sharpen it by scaling the pooled log-odds by a factor a ≥ 1:

    logit(p_ext) = a · logit(p_pool)

- a = 1 → no extremizing.
- a ≈ 1.5–2.5 → typical helpful range **when** forecasters are many, diverse, and independent.
- Extremizing **hurts** when forecasters are few or highly correlated (they'd be echoing each
  other, so pushing to the extreme just amplifies a shared error). Default to a = 1 and only
  extremize deliberately.
- **All-LLM panels are correlated by default.** Claude, Codex, etc. share large parts
  of their training data and priors — treat their agreement as partially one vote, and do NOT
  extremize an ensemble made only of LLM agents. Extremizing earns its keep with genuinely
  independent inputs (different people, different data sources, different methods).

Tool: `scripts/aggregate.py 0.6 0.7 0.55 0.8 [--extremize 2.0] [--p .. --w ..]`

### Track-record weights
When forecasters have enough scored history, weight them by smoothed historical Brier:

    python3 scripts/aggregate.py \
      --p 0.60 --forecaster alice \
      --p 0.70 --forecaster bob \
      --track-log "$FORECAST_LOG"

The tool uses a conservative prior (`prior_brier=0.25`, `prior_n=5`) so one lucky forecast
does not dominate. Missing track records default to weight 1. Treat these weights as a nudge,
not a substitute for inspecting rationales.

### Time decay
Older forecasts can become stale as resolution approaches. Apply exponential decay when you
are pooling forecasts made at different times:

    python3 scripts/aggregate.py \
      --p 0.60 --date 2026-07-01 \
      --p 0.70 --date 2026-07-12 \
      --half-life-days 14

Use time decay when the information environment changes quickly. Do not use it to erase a
still-valid older rationale just because it is older.

## 2. Bayesian updating (the inside-view step)

Work in odds so updates multiply and are order-independent:

    posterior_odds = prior_odds · LR₁ · LR₂ · …
    LRᵢ = P(evidenceᵢ | YES) / P(evidenceᵢ | NO)

LR > 1 supports YES, LR < 1 supports NO, LR = 1 is uninformative. Small steps, often.
Tool: `scripts/bayes.py --prior 0.3 --lr 3.0 --lr 0.7` (or `--pe-h/--pe-nh`).

## 3. Scoring yourself — the Brier score

    Brier = mean( (p − outcome)² )        outcome ∈ {0,1}

- 0.00 = perfect. 0.25 = the score of always guessing 0.5. Lower is better.
- (For 2-outcome events using the {0,1} convention above; note some texts use a ×2 convention
  that puts the "coin flip" at 0.50 — this project uses the (p−o)² convention, coin flip 0.25.)

### Murphy decomposition (with K probability bins)
    Brier = Reliability − Resolution + Uncertainty

    Reliability = (1/N) Σ_k n_k (p̄_k − ō_k)²     ← calibration error (want LOW)
    Resolution  = (1/N) Σ_k n_k (ō_k − ō)²        ← discrimination (want HIGH)
    Uncertainty = ō (1 − ō)                        ← difficulty of the event mix (fixed)

where within bin k there are n_k forecasts with mean forecast p̄_k and observed frequency ō_k,
and ō is the overall base rate.

- **Reliability / calibration**: do your 70% forecasts come true ~70% of the time?
- **Resolution / discrimination**: do you dare to move away from the base rate, correctly?
- You improve by lowering reliability error and raising resolution. Uncertainty you can't change.

Tool: `scripts/brier.py` prints the score, the decomposition, and a
calibration table flagging over/under-confident bins.

### Opt-in time-weighted (trajectory) Brier

When a forecast has belief updates, inspect how long each probability was live as well as its
latest value:

    python3 scripts/brier.py --time-weighted

For each resolved forecast, `prob_history` is treated as a right-continuous step function: a
probability holds from its timestamp to the next update, and the final probability holds to the
resolution timestamp. Its trajectory score is the duration-weighted mean of segment errors:

    TWB = Σ days_liveᵢ · (pᵢ − outcome)² / Σ days_liveᵢ

The aggregate is the mean of those per-forecast scores, so each forecast has equal weight. A
zero or out-of-order interval contributes zero duration. If a trajectory has no updates, lacks a
usable timestamp, or has zero total duration, it degrades to that forecast's standard latest-
probability Brier. This is additive and opt-in: default Brier output, including the `(p−o)²`
convention and its 0.25 coin-flip score, remains unchanged.

### Ordered-bin distributions: Ranked Probability Score (RPS)

For a numeric or range question discretized into ordered bins, log a coherent distribution:

    python3 scripts/forecastlog.py add --question "2027 units?" \
      --bins "under-10,10-20,over-20" --dist "0.2,0.5,0.3" --resolve-by 2027-12-31
    python3 scripts/forecastlog.py resolve --id 7 --outcome-bin 10-20

The distribution has K ordered bins and probabilities p₁…p_K that sum to 1. If the realized
bin is m, let P_k and O_k be the forecast and realized cumulative probabilities through k.
The stored, un-normalized score is:

    RPS = Σ_(k=1)^(K−1) (P_k − O_k)²

Lower is better; zero is a point mass on the realized bin. RPS is strictly proper for ordered
categorical outcomes (Epstein 1969; Murphy 1969/1970) and is the discrete analog of CRPS. It
credits near misses, which is appropriate for a numeric scale. With two bins `[NO, YES]`, it
reduces exactly to this project's `(p−o)²` Brier convention. Log score and multi-category Brier
treat every wrong category equally, so they are reserved for a future unordered-multinomial path.

### Brier Skill Score (baseline comparison)
Raw Brier is hard to interpret without a baseline. Compare against a constant prior, base rate,
crowd probability, or local median:

    python3 scripts/brier.py --baseline-prob 0.5
    python3 scripts/brier.py --baseline-field crowd_prob

    BSS = 1 - (Brier / baseline_Brier)

Positive BSS means you beat the baseline; negative means the baseline was better. The tool also
prints skill against the observed base-rate baseline when there are at least two resolved
forecasts and outcome uncertainty is non-zero.

## Sources
- Geometric mean of odds / external Bayesianity — EA Forum, "When pooling forecasts, use the
  geometric mean of odds"; Satopää et al. on aggregation.
- Extremizing — Baron, Mellers, Tetlock et al.; "Principled extremizing of aggregated forecasts."
- Brier score & decomposition — Brier (1950); Murphy (1973) decomposition.
