# INSTALL_REPORT — Sports Quant Research Lab

Install date: 2026-10-31 · Host: Claude Code cloud container · Branch: `claude/exciting-hawking-b1kelv`.
Each line below was checked with a command or a test. Anything that could not be checked is marked as
such.

## 0. Summary

| component | status | verified by |
|---|---|---|
| Environment (OS, Python, Git, Node, npm, Claude Code) | ✅ present | version commands (§1) |
| `~/AI-Lab/sports-research` | ✅ symlink → repository (persists via git) | `ls -la ~/AI-Lab` |
| Python venv + SQLite + Parquet | ✅ | `.venv`, `data/sportslab.sqlite`, `data/processed/*.parquet` |
| Claude Code Skills | ✅ **37 installed** from 4 repos, pinned SHAs, 0 conflicts | `skills.lock.json`, `tests/test_data_and_skills.py` |
| backtestbuddy 0.3.1 | ✅ installed; upstream suite **447/447 passed**; used as an independent P&L cross-check | `tests/test_backtest.py`, `backtest_crosscheck.json` |
| betting-backtester 0.1.0 | ✅ installed; upstream suite **764/764 passed**; used as an independent Poisson-MLE cross-check | `tests/test_models_mc_store.py` |
| sports-skills 0.40.1 (CLI / Python) | ✅ installed (no extras) · ⚠️ live endpoints **blocked** (HTTP 403 via egress proxy) | `sports-skills nhl get_scoreboard` |
| superforecasting scripts | ✅ upstream suite 256 passed / 3 failed (failures are in its git pre-commit hook under `tools/`, which is not installed: GNU vs BSD `mktemp`) | upstream pytest |
| InferSports MCP | ❌ **not connected**: endpoint unverified, host blocked | `config/mcp/infersports.DISABLED.json` |
| PuckAPI MCP (paid) | ⛔ deliberately not connected | `.mcp.json` absent (asserted in tests) |
| Historical data | ✅ 28/28 sources, SHA-256 pinned | `scripts/fetch_data.py` |
| Data quality | ✅ 30 PASS, 2 WARN (odds coverage gaps), 0 FAIL | `scripts/build_db.py` |
| Models + validation | ✅ 4 sports, walk-forward, test set held out | `MODEL_VALIDATION.md` |
| Live engine | ✅ tested (synthetic + real NHL payload + real EPL half-time data) · live feeds unavailable here | `tests/test_live.py` |
| Virtual backtests + journal | ✅ append-only journal (SQLite triggers) | `tests/test_backtest.py`, `tests/test_models_mc_store.py` |
| Streamlit dashboard | ✅ 8 sections × 4 sports render without exceptions; server health check OK | `tests/test_dashboard.py` |
| Test suite | ✅ **105 passed**, 0 failed, 0 skipped | `pytest` |

## 1. Environment check (stage 1)

| tool | version |
|---|---|
| OS | Ubuntu 24.04.5 LTS (Linux 6.18 x86_64), 4 vCPU, 15 GB RAM |
| Python | 3.13.18 (venv `.venv`) |
| Git | 2.43.0 |
| Node.js / npm | v22.23.4 / 10.9.9 |
| Claude Code | 2.1.316 |
| SQLite | 3.53.1 |
| Key packages | numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, statsmodels 0.15.0, pyarrow 26.0.0, streamlit 1.66.1, plotly 7.1.0 (`requirements.txt`) |

`~/AI-Lab/sports-research` is a **symlink to this repository**. The container is ephemeral, so the git
repository is the only place where work persists. Setup is reproducible with `bash scripts/setup_env.sh`
(it creates the venv, installs pinned requirements and vendor libraries, installs skills, and creates
the symlink).

## 2. Third-party audit, done before installing anything

| check | finding / action |
|---|---|
| Install scripts | No repo needs `curl \| sh`. **Nothing from the upstream repos was executed during install.** `superforecasting/tools/install-hooks.sh` and the sports-skills `packaging/` and `release` scripts were not run. |
| Code patterns | grep for `curl\|sh`, `base64 -d`, `eval(`, `os.system`, `subprocess`. The only hits were benign: sports-skills `_premium.py` / `cli.py` (subprocess for an optional premium installer, not used), `superforecasting/scripts/montecarlo.py` (`eval` with `__builtins__` removed after AST compilation; kept). |
| Money / trading | The sports-skills catalog marks `polymarket-trading` as `financial_execution` + `money_movement=true` and `machina` as `premium_mcp`. Both are excluded, along with every prediction-market/exchange module. The project `.claude/settings.json` **denies** `sports-skills polymarket-trading/kalshi/novig/prophetx/storefront` and `mcp__puckapi__*`. |
| MCP servers | The PuckAPI `.mcp.json` (paid, bearer token) was **not** registered. No MCP server is configured in this project. |
| Licenses | All 6 repos are MIT. Copies are in `third_party/licenses/`. |
| Existing skills | `.claude/skills` did not exist before. The installer compares a directory hash and **refuses to overwrite** a modified skill (reported as `CONFLICT`; `--force` only after review). Verified by modifying a skill on purpose: the result was `CONFLICT metadata`. |

## 3. Skills actually installed (stage 2) — 37

| repository | commit | # | skills (actual `name:` from SKILL.md) |
|---|---|---|---|
| `machina-sports/sports-skills` | `940f346c97` | 7 | `football-data`, `nba-data`, `nhl-data`, `tennis-data`, `espn-api`, `metadata`, `sports-news` |
| `WalrusQuant/sports-analytic-skills` | `0f90d2463b` | 19 | `eda-sports`, `feature-rules`, `leakage-audit`, `predictive-modeling`, `baseline-models`, `validation-design`, `calibration-check`, `simulation-sports`, `results-reporting`, `ratings-strength-models`, `statistical-modeling`, `time-series-sports`, `model-card`, `experiment-log`, `model-interpretation`, `sports-modeling-doctrine`, `data-sources`, `anti-slop-analytics`, `sports-visualization` |
| `PuckAPI/claude-sports-analytics` | `d542e7d793` | 10 | `hockey-analytics`, `xg-model-building`, `elo-engineering`, `model-building`, `probability-calibration`, `visualization`, `backtesting`, `walk-forward-validation`, `feature-engineering`, `goalie-analysis` |
| `sloik/superforecasting` | `850824ed62` | 1 | `superforecasting` |

Note on the request: the WalrusQuant skills for leakage audit, walk-forward validation and simulation are
named `leakage-audit`, `validation-design` and `simulation-sports`. "Walk-forward" is covered by
`predictive-modeling/references/walk_forward.md` and `validation-design`.

The PuckAPI skills name PuckAPI as their default data tool. Since PuckAPI is not connected, `CLAUDE.md`
tells Claude to use the lab's local data instead.

**Excluded (with reasons, also in `skills.lock.json`)**: `polymarket-trading` (real trading),
`machina` (premium/secrets), `kalshi`, `polymarket`, `novig`, `prophetx`, `gemini`, `markets`
(prediction markets/exchanges, not sports statistics), `betting` (includes Kelly stake sizing;
devig/EV re-implemented for education in `sportslab.ev`), `world-cup` (catalog risk=high),
out-of-scope sports modules. PuckAPI: `puckapi-tool`, `dispatch`, `data-pipeline`, `nl-to-query`,
`game-lookup`, `odds-explorer`, `daily-card`, `game-preview` (paid API) and `bet-tracker`,
`edge-detection`, `prop-modeling` (real-betting workflows). WalrusQuant: `nflreadpy`, `pybaseball`,
`sportsdataverse-py` (other sports), `environment-setup` (runs pip), `sports-ds-bridge`.

Re-install / verify: `python scripts/install_skills.py`. Idempotent: the second run shows 37 `unchanged`
and 0 conflicts.

## 4. Python libraries (stage 2, items 5–6)

| library | install | upstream tests in this venv | role in the lab |
|---|---|---|---|
| backtestbuddy 0.3.1 | `requirements-vendor.txt` (git+SHA `0cb9dd9`) | 447 passed | independent recomputation of flat-stake P&L: identical to `sportslab.backtest` (max diff 7e-15) on 7 real model/sport runs plus 1 synthetic test |
| betting-backtester 0.1.0 | git+SHA `a941b24` | 764 passed | its Poisson MLE matches `sportslab` Poisson rates to within 0.2% (test) |
| sports-skills 0.40.1 | git+SHA `940f346`, no extras (no trading clients) | — | read-only live adapter `SportsSkillsLive`; currently UNAVAILABLE (403) |
| sports-ds 0.12.0 | `--no-deps` (skill scripts use only the stdlib) | — | backs the WalrusQuant skill scripts |

## 5. InferSports MCP (stage 2, item 7) — NOT connected

Web search found only third-party directory listings, which describe InferSports as a keyless,
read-only MCP with live scores and Pinnacle-devigged odds. Official documentation and the endpoint could
not be verified, and `infersports.com` / `api.infersports.com` are refused by this environment's egress
proxy (`connect_rejected`, HTTP 403). To avoid connecting an unverified server, the lab ships a
**disabled template** (`config/mcp/infersports.DISABLED.json`) with enable steps and a read-only
policy, plus a `sportslab.live.sources.InferSports` adapter that always reports `UNAVAILABLE` with
this reason.
Sources: [mcp.so listing](https://mcp.so/zh/servers/infersports), [PolicyLayer listing](https://policylayer.com/policies/dev-infersports-infersports).

## 6. Unavailable sources (summary)

ESPN / NBA CDN / NHL api-web (live), InferSports, football-data.co.uk and tennis-data.co.uk direct,
JeffSackmann tennis_atp, HuggingFace/Kaggle: all blocked or unavailable here. PuckAPI: excluded (paid).
Details are in `DATA_SOURCES.md` §5. To allow these hosts, change the environment's network policy
(Claude Code on the web → environment settings → network access).

## 7. Tests (stage 9)

`pytest` → **105 passed in ~40 s** (`OMP_NUM_THREADS=1` is set in `tests/conftest.py` to avoid OpenMP
oversubscription).

| area | file | what it proves |
|---|---|---|
| Data acquisition | `test_data_and_skills.py` | URLs pinned to commits; SHA-256 mismatch rejected; HTTP 403, proxy errors and git-LFS pointers → `UNAVAILABLE` (never crash, never fake data) |
| Stale live data | `test_live.py` | fresh → `OK`; lag > warn → `DELAYED` + seconds; lag > max → `INSUFFICIENT_DATA`; unknown timestamp is never `OK`; clock-skew rejected; missing fields/prior → `INSUFFICIENT_DATA` with `probs=None` |
| Probability maths | `test_probability_math.py` | implied/overround/4 devig methods, EV, Brier/log loss/RPS by hand, ECE, bootstrap; Dixon-Coles τ; tennis hold probability vs closed form; temperature scaling recovers T≈2 |
| Monte Carlo | `test_models_mc_store.py` | same seed → identical; different seed → different; MC within 4 SE of the analytic value; season sim reproducible |
| No look-ahead | `test_lookahead.py` | spy model never sees `available_at > t`; fixture view has no results or closing odds; **poisoning test**: scramble all future results → identical past predictions for Elo, Dixon-Coles, Gamma-Poisson, logistic, GBM, Gaussian rating; PIT guard drops "opening" odds stamped after the forecast time |
| Chronological backtest | `test_backtest.py` | hand-computed P&L, commission, drawdown stop, fraction stakes use only *settled* bankroll, settle after open, CLV formula, identical to backtestbuddy |
| Virtual journal | `test_models_mc_store.py` | `UPDATE`/`DELETE` on `virtual_journal` aborted by trigger; forecast resolution is immutable |
| Models | `test_models_mc_store.py` | Poisson MLE == statsmodels GLM (2e-3) == betting-backtester; synthetic truth recovered; models beat uniform |
| Dashboard | `test_dashboard.py` | all 8 sections × 4 sports via `streamlit.testing.AppTest` with no exceptions; `streamlit run` health endpoint returns `ok` |
| Historical | `test_data_and_skills.py` | DB counts, 0 FAIL quality checks, models beat uniform on test, backtest cross-check & journal present |

## 8. Problems found and fixed during installation (for transparency)

1. The skill installer's hash included ignored folders, which caused a false conflict on re-run. Fixed: the hash uses the same ignore set.
2. FK-ordered deletes when rebuilding the DB. Fixed (children first).
3. Tennis 2022 ATP Finals had no `Best of`. Stored as UNKNOWN instead of being guessed.
4. GBM crashed when a feature was entirely missing. It now drops never-observed features at fit time.
5. OpenMP oversubscription made the tests 50× slower while the validation ran. Tests are now pinned to 1 thread.
6. The first backtest used a 100-unit bankroll, so flat strategies hit cash limits. It now uses a 1,000-unit bankroll, and the cross-check runs the real simulator.
7. Pinnacle odds are missing for the second half of 2025-26. This is detected (WARN), documented, and handled: the example forecast picks a match with complete prices, and NaN odds are rejected by `implied_probabilities`.
8. The dashboard log-loss bar chart hid the differences between models. It was replaced by a point + CI plot (checked with a Playwright screenshot).
9. **Team-name mismatch in the 2025-26 football file** (found while building upcoming-match support): 7 clubs were
   spelled differently, which reset their ratings in the 2025-26 test season. Fixed with canonical names. A new
   quality check catches this, and validation, backtests and reports were **re-run** afterwards.
10. Live adapter: tennis scoreboards (`tournaments → draws → matches`) and the football daily schedule
    (`qualifier`, no `get_scoreboard`) were parsed with the wrong shape, giving "AVAILABLE, 0 events". Fixed and tested.
    InferSports is now reported as `NOT_CONNECTED` (by design) instead of a misleading "HTTP 403" message.
11. macOS setup: Python ≥ 3.12 is selected automatically, and stale 3.9 venvs are rebuilt. The skill installer
    retries, shows git errors, and verifies skills offline when GitHub is unreachable.
