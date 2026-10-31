#!/usr/bin/env bash
# Reproducible environment setup (Linux / macOS). Safe to re-run.
# Requires Python >= 3.12 (numpy 2.5, pandas 3, betting-backtester). Override with: PYTHON=/path/to/python3.13 bash scripts/setup_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

version_ok() {  # $1 = python executable; succeeds if >= 3.12
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null
}

PY=""
for cand in "${PYTHON:-}" python3.13 python3.12 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
            /usr/local/bin/python3.13 /usr/local/bin/python3.12 python3; do
  [ -n "$cand" ] || continue
  if command -v "$cand" >/dev/null 2>&1 && version_ok "$cand"; then PY="$(command -v "$cand")"; break; fi
done

echo "== tool check"
uname -sr
git --version
command -v node >/dev/null && node --version || echo "node: not found (optional)"
command -v npm  >/dev/null && npm --version  || echo "npm: not found (optional)"
command -v claude >/dev/null && claude --version || echo "claude: not found (optional)"

if [ -z "$PY" ]; then
  echo
  echo "ERROR: Python >= 3.12 not found (found: $(python3 --version 2>&1 || echo none))."
  echo "Install one of:"
  echo "  macOS (Homebrew):  brew install python@3.13"
  echo "  macOS/Windows:     https://www.python.org/downloads/  (3.13.x installer)"
  echo "  any OS (uv):       curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv.sh && less /tmp/uv.sh && sh /tmp/uv.sh && uv python install 3.13"
  echo "then re-run:  bash scripts/setup_env.sh   (or PYTHON=/path/to/python3.13 bash scripts/setup_env.sh)"
  exit 1
fi
echo "python: $PY ($("$PY" --version))"

# Recreate a venv that was built with an unsupported interpreter (e.g. macOS system Python 3.9).
if [ -d .venv ] && ! version_ok .venv/bin/python; then
  echo "== existing .venv uses $(.venv/bin/python --version 2>&1 || echo 'a broken interpreter'); recreating"
  rm -rf .venv
fi
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r requirements-vendor.txt
pip install -q -e .
python scripts/install_skills.py || echo "WARNING: skill sync reported problems (see above); the lab itself is installed."
mkdir -p "$HOME/AI-Lab"
if [ ! -e "$HOME/AI-Lab/sports-research" ]; then ln -s "$ROOT" "$HOME/AI-Lab/sports-research"; fi
echo "== done. Next: . .venv/bin/activate && python scripts/fetch_data.py && python scripts/build_db.py"
