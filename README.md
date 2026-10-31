# Sports Quant Research Lab

A research system for **probabilistic sports forecasting** (prematch and live) in football, hockey,
basketball and tennis. It is built to measure forecast accuracy honestly, compare statistical models with
baselines and markets, explain expected-value maths, and run **reproducible historical simulations with a
virtual bankroll**.

> ⚠️ Research and education only. There are no real bets, payments, bookmaker-account automation or
> real-stake sizing anywhere in this code. Every bankroll figure is a virtual unit from a historical
> simulation.

## Quick start

```bash
bash scripts/setup_env.sh
```

```bash
. .venv/bin/activate && python scripts/fetch_data.py && python scripts/build_db.py
```

```bash
python scripts/run_validation.py && python scripts/run_backtest.py && python scripts/example_forecast.py && python scripts/make_reports.py
```

```bash
pytest
```

```bash
streamlit run dashboard/app.py
```

The dashboard opens at <http://localhost:8501>. Its eight sections are: Prematch Analytics, Live Match
Analytics, Probability Models, Forecast History, Backtesting, Virtual Bankroll, Model Performance and
Data Quality. The validated artifacts are committed under `reports/artifacts/`, so after `fetch_data` +
`build_db` (about 1 minute) the dashboard works without re-running the 20-minute validation.

## Layout

```
src/sportslab/
  data/      sources.py (pinned registry) · fetch.py (SHA-256) · ingest.py · store.py (SQLite+Parquet)
             pit.py (point-in-time gate) · quality.py
  models/    base.py (walk-forward driver) · ratings.py (Elo, tennis surface Elo, Bayesian Gaussian rating)
             goals.py (Poisson, Dixon-Coles, Bayesian Gamma-Poisson) · ml.py (logistic, gradient boosting)
             features.py (online PIT features) · baselines.py · calibration.py · montecarlo.py
  live/      engine.py (OK / DELAYED / INSUFFICIENT_DATA) · tennis.py (exact Markov chain) · sources.py
  ev/        odds.py (implied prob, overround, 4 devig methods, EV, sensitivity) · metrics.py (Brier, LL, RPS, ECE, CIs)
  backtest/  simulator.py (virtual bankroll) · journal.py (append-only SQLite journal, forecast log)
  research/  validation.py (chronological protocol)
dashboard/app.py · scripts/ · tests/ · .claude/skills/ (37 installed skills)
```

## Principles enforced in code
- **No look-ahead.** Every fact has `available_at`. The walk-forward driver reveals results only after
  that time, and fixtures carry no result or closing-odds columns. Poisoning tests (scramble the future,
  check the past is unchanged) cover every model.
- **Observed vs model vs unknown.** Each row is labelled `OBSERVED`, `MODEL` or `UNKNOWN`. Missing
  lineups and injuries stay empty; they are never filled with invented values.
- **Live honesty.** Feed latency is measured and reported. A stale or incomplete feed returns
  `INSUFFICIENT_DATA` with no probabilities.
- **Chronological validation.** Burn-in → train → validation (tuning and calibration only) → test (touched once).

## Headline results (test seasons, log loss, lower is better)

| sport (test n) | uniform | best lab model | strongest benchmark |
|---|---|---|---|
| Football EPL 2024-26 (760) | 1.0986 | ensemble 1.0086 · Gamma-Poisson 1.0087 | Pinnacle closing 0.9739 · opening 0.9931 |
| Hockey NHL 2018-20 (2,353) | 0.6931 | logistic 0.6788 | — (no odds available) |
| Basketball NBA 2020-22 (2,494) | 0.6931 | ensemble+temp 0.6454 · Elo 0.6462 | FiveThirtyEight 0.6291 |
| Tennis ATP 2024 (1,813) | 0.6931 | logistic 0.6098 | Pinnacle (single snapshot) 0.5751 |

The models beat naive baselines in every sport, with paired-bootstrap CIs excluding zero. They do **not**
beat sharp market prices or FiveThirtyEight. In the virtual strategy research, no strategy shows a
positive out-of-sample ROI whose CI excludes zero. Football CLV is negative (about −3% to −4%). Tennis
flat-stake ROI is about −11% to −15% (CI below 0). On selected positions the realised hit rate (0.347) is
far below the model's expected rate (0.446), which is adverse selection. The Live model, tested on real
EPL half-time scores, improves log loss from 1.046 to 0.866 on the test seasons. Full details are in
`MODEL_VALIDATION.md`.

## Documents
- `INSTALL_REPORT.md`: what was installed, how it was verified, and what was excluded and why.
- `DATA_SOURCES.md`: sources, mirrors, checksums, point-in-time rules, known gaps, unavailable sources.
- `MODEL_VALIDATION.md`: auto-generated metrics, CIs, comparisons, live/MC checks and backtests.
- `CLAUDE.md`: project rules for Claude Code sessions.
