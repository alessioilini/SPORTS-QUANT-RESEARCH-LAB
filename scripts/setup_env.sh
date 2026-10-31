#!/usr/bin/env bash
# Reproducible environment setup. Safe to re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "== tool check"
uname -sr; python3 --version; git --version
command -v node >/dev/null && node --version || echo "node: not found (optional)"
command -v npm  >/dev/null && npm --version  || echo "npm: not found (optional)"
command -v claude >/dev/null && claude --version || echo "claude: not found (optional)"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r requirements-vendor.txt
pip install -q -e .
python scripts/install_skills.py
mkdir -p "$HOME/AI-Lab"
if [ ! -e "$HOME/AI-Lab/sports-research" ]; then ln -s "$ROOT" "$HOME/AI-Lab/sports-research"; fi
echo "== done. Next: python scripts/fetch_data.py && python scripts/build_db.py"
