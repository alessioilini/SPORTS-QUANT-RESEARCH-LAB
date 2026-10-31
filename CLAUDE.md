# Sports Quant Research Lab — project rules for Claude Code

Research-only lab: probabilistic forecasting, calibration, and *virtual*-bankroll simulations.

Hard rules
- No real bets, payments, bookmaker-account automation, browser control of betting sites, or
  real-stake sizing advice. All money figures are virtual units in historical simulations.
- Never use information from the future: every historical forecast must go through
  `sportslab.data.pit` / `sportslab.models.walkforward` (results are visible only after `available_at`).
- Label every value as OBSERVED (fetched from a source), MODEL (computed here) or UNKNOWN.
- Live: if a feed is late, say so (status DELAYED + delay seconds); if inputs are missing,
  return INSUFFICIENT_DATA — never invent numbers.
- Do not claim a component works until a test/command has verified it.

Installed skills live in `.claude/skills` (pinned in `skills.lock.json`; installer refuses to
overwrite modified skills). PuckAPI paid MCP and InferSports MCP are NOT connected
(see `config/mcp/`). The PuckAPI skills default to PuckAPI as data tool — in this lab use the
local data in `data/` instead.

Commands
- setup: `bash scripts/setup_env.sh`
- data: `python scripts/fetch_data.py && python scripts/build_db.py`
- validation: `python scripts/run_validation.py`
- tests: `pytest`
- dashboard: `streamlit run dashboard/app.py`
