# AGENTS.md — Superforecasting skill

This file orients non-Claude agents (Codex, Aider, etc.).

**Read `SKILL.md` first.** It is the full playbook: the operating mindset, the 8-step
forecasting pipeline, the discipline rules, and the tool map.

When the user asks a forecasting or consequential-decision question:
1. Run the pipeline in `SKILL.md` (triage → Fermi-ize → base rate → adjust → weigh → number
   → update-triggers → log & score), discussing your reasoning out loud.
2. Use the tools in `scripts/` instead of doing the math in your head:
   - `baserate.py`  — reference-class base rate + interval (outside view)
   - `bayes.py`     — Bayesian odds update (inside view)
   - `montecarlo.py`— simulate Fermi-ized range quantities
   - `aggregate.py` — pool multiple forecasts (ensemble)
   - `brier.py`     — score calibration
   - `forecastlog.py` — record/resolve/score forecasts
   All are stdlib-only Python 3: `python3 scripts/<name>.py --help`.
3. **Verify every real-world fact/base-rate with web or data sources** — do not state figures
   from memory. Before logging, run the rationale-quality checklist in
   `references/rationale-quality.md`. Log the final forecast with `forecastlog.py`.

Deeper material is in `references/`; reusable templates (including the cross-agent question
packet) are in `templates/`.
