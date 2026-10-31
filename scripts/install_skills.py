#!/usr/bin/env python3
"""Install a vetted subset of third-party Claude Code Skills into .claude/skills.

Design rules (see INSTALL_REPORT.md):
* every upstream repo is pinned to an exact commit SHA;
* only skills listed in SELECTION are copied; everything else is recorded as
  excluded together with the reason;
* an existing skill directory is NEVER overwritten silently: if it differs
  from the upstream copy the installer stops for that skill and reports a
  conflict (use --force to replace after reviewing the diff);
* nothing from the upstream repos is executed (no install scripts, no hooks,
  no MCP servers are registered).

Usage:
    python scripts/install_skills.py [--cache DIR] [--force] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / ".claude" / "skills"
LOCK_FILE = ROOT / "skills.lock.json"
LICENSE_DIR = ROOT / "third_party" / "licenses"

REPOS = {
    "machina-sports/sports-skills": "940f346c978f1abd39077f1ebddd27866613ef22",
    "WalrusQuant/sports-analytic-skills": "0f90d2463b7d4c793821cce71fc82d06fcb06a3c",
    "PuckAPI/claude-sports-analytics": "d542e7d7935d6be4514f1117057281c344d8683f",
    "sloik/superforecasting": "850824ed6287ec5a6ede8866adc4e0109d333ef6",
}

# (repo, path inside repo) for every skill we install.
SELECTION: dict[str, list[str]] = {
    "machina-sports/sports-skills": [
        # read-only sports statistics modules (catalog mode=read_only, money_movement=false)
        "skills/football-data", "skills/nba-data", "skills/nhl-data", "skills/tennis-data",
        "skills/espn-api", "skills/metadata", "skills/sports-news",
    ],
    "WalrusQuant/sports-analytic-skills": [
        "skills/eda-sports", "skills/feature-rules", "skills/leakage-audit",
        "skills/predictive-modeling", "skills/baseline-models", "skills/validation-design",
        "skills/calibration-check", "skills/simulation-sports", "skills/results-reporting",
        "skills/ratings-strength-models", "skills/statistical-modeling",
        "skills/time-series-sports", "skills/model-card", "skills/experiment-log",
        "skills/model-interpretation", "skills/sports-modeling-doctrine",
        "skills/data-sources", "skills/anti-slop-analytics", "skills/sports-visualization",
    ],
    "PuckAPI/claude-sports-analytics": [
        "skills/hockey-analytics", "skills/xg-model-building", "skills/elo-engineering",
        "skills/model-building", "skills/probability-calibration", "skills/visualization",
        "skills/backtesting", "skills/walk-forward-validation", "skills/feature-engineering",
        "skills/goalie-analysis",
    ],
    "sloik/superforecasting": ["."],
}

EXCLUDED: dict[str, dict[str, str]] = {
    "machina-sports/sports-skills": {
        "polymarket-trading": "catalog mode=financial_execution, money_movement=true (real trading)",
        "machina": "catalog mode=premium_mcp, requires secrets / metered billing",
        "kalshi": "prediction-market module, not sports statistics",
        "polymarket": "prediction-market module, not sports statistics",
        "novig": "betting-exchange module, not sports statistics",
        "prophetx": "betting-exchange module, not sports statistics",
        "gemini": "prediction-market module, not sports statistics",
        "markets": "prediction-market aggregator, not sports statistics",
        "betting": "includes Kelly stake sizing; devig/EV re-implemented in sportslab.ev instead",
        "world-cup": "catalog risk=high; not needed for the 4 target sports",
        "*other sports*": "nfl/mlb/cfb/cbb/wnba/gleague/golf/cricket/volleyball/esports/fastf1/xctf: out of scope (4 target sports)",
        "sports-reporter": "article generation, not data",
    },
    "WalrusQuant/sports-analytic-skills": {
        "nflreadpy": "NFL-specific", "pybaseball": "MLB-specific",
        "sportsdataverse-py": "US-college/NFL loaders, out of scope",
        "environment-setup": "runs pip installs; environment handled by scripts/setup_env.sh",
        "sports-ds-bridge": "bridge to an external toolkit not used here",
    },
    "PuckAPI/claude-sports-analytics": {
        "puckapi-tool": "paid PuckAPI MCP (credits) — not connected by request",
        "dispatch": "routes to paid PuckAPI tools", "data-pipeline": "paid PuckAPI ingestion",
        "nl-to-query": "paid PuckAPI query layer", "game-lookup": "paid PuckAPI lookup",
        "odds-explorer": "paid PuckAPI odds", "daily-card": "daily real-betting card",
        "game-preview": "depends on paid PuckAPI data",
        "bet-tracker": "real bet tracking — out of scope (virtual journal only)",
        "edge-detection": "today's-slate real betting edges — out of scope",
        "prop-modeling": "player-prop betting via paid API — out of scope",
        "*others*": "team-analysis, player-scouting, playoff-simulation, totals-modeling, "
                    "war-gar-decomposition, ai-hockey-workflow: not requested",
        ".mcp.json": "PuckAPI MCP server NOT registered (paid API)",
    },
}


def sh(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


IGNORED = (".git", ".github", "tests", "tools", "__pycache__")


def dir_hash(path: Path) -> str:
    h = hashlib.sha256()
    files = (p for p in path.rglob("*")
             if p.is_file() and not set(p.relative_to(path).parts) & set(IGNORED))
    for f in sorted(files):
        h.update(str(f.relative_to(path)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def skill_name(skill_md: Path) -> str:
    for line in skill_md.read_text(encoding="utf-8").splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    raise ValueError(f"no name: in {skill_md}")


def ensure_clone(cache: Path, repo: str, sha: str) -> Path:
    dest = cache / repo.replace("/", "__")
    if not dest.exists():
        sh("git", "clone", "--quiet", f"https://github.com/{repo}.git", str(dest))
    head = sh("git", "rev-parse", "HEAD", cwd=dest)
    if head != sha:
        sh("git", "fetch", "--quiet", "origin", sha, cwd=dest)
        sh("git", "checkout", "--quiet", sha, cwd=dest)
    assert sh("git", "rev-parse", "HEAD", cwd=dest) == sha, f"{repo} not at pinned {sha}"
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / ".cache" / "skill-repos"))
    ap.add_argument("--force", action="store_true", help="replace conflicting skills")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    LICENSE_DIR.mkdir(parents=True, exist_ok=True)

    lock = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "installed": [], "conflicts": [], "excluded": EXCLUDED, "repos": REPOS}
    seen_names: dict[str, str] = {}
    for repo, sha in REPOS.items():
        src_repo = ensure_clone(cache, repo, sha)
        lic = src_repo / "LICENSE"
        if lic.exists() and not a.dry_run:
            shutil.copyfile(lic, LICENSE_DIR / f"{repo.replace('/', '__')}.LICENSE")
        for rel in SELECTION[repo]:
            src = (src_repo / rel).resolve()
            name = skill_name(src / "SKILL.md")
            if name in seen_names:
                raise SystemExit(f"skill name collision: {name} ({seen_names[name]} vs {repo})")
            seen_names[name] = repo
            dst = SKILLS_DIR / name
            src_hash = dir_hash(src)
            status = "installed"
            if dst.exists():
                if dir_hash(dst) == src_hash:
                    status = "unchanged"
                elif not a.force:
                    lock["conflicts"].append({"name": name, "repo": repo, "reason":
                                              "existing skill differs; not overwritten (use --force after review)"})
                    print(f"CONFLICT {name}: existing directory differs, skipped")
                    continue
                else:
                    status = "replaced"
            if status != "unchanged" and not a.dry_run:
                if dst.exists():
                    shutil.rmtree(dst)
                ignore = shutil.ignore_patterns(*IGNORED)
                shutil.copytree(src, dst, ignore=ignore)
            lock["installed"].append({"name": name, "repo": repo, "commit": sha, "path": rel,
                                      "sha256": dir_hash(dst) if dst.exists() else src_hash,
                                      "status": status})
            print(f"{status:9} {name:28} <- {repo}@{sha[:7]}:{rel}")
    if not a.dry_run:
        LOCK_FILE.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"\n{len(lock['installed'])} skills, {len(lock['conflicts'])} conflicts -> {LOCK_FILE.name}")
    return 1 if lock["conflicts"] else 0


if __name__ == "__main__":
    sys.exit(main())
