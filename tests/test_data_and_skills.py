"""Data acquisition (pinned + checksummed), installed skills integrity, historical pipeline outputs."""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from sportslab.config import ROOT
from sportslab.data import fetch as F
from sportslab.data.sources import REGISTRY


def test_every_source_url_is_pinned_to_a_commit():
    for sid, s in REGISTRY.items():
        assert re.search(r"/[0-9a-f]{40}/", s.url), f"{sid} not pinned: {s.url}"
        assert s.original_provider and s.license_note


class FakeResp:
    def __init__(self, code, content):
        self.status_code, self.content = code, content


def _src(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "RAW_DIR", tmp_path)
    return next(iter(REGISTRY.values()))


def test_fetch_success_and_checksum(tmp_path, monkeypatch):
    src = _src(tmp_path, monkeypatch)
    monkeypatch.setattr(F.requests, "get", lambda url, timeout: FakeResp(200, b"a,b\n1,2\n"))
    rec = F.fetch_one(src, {}, force=True)
    assert rec["status"] == "downloaded" and rec["sha256"] == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    bad = F.fetch_one(src, {src.source_id: "0" * 64}, force=True)
    assert bad["status"] == "CHECKSUM_MISMATCH" and not (tmp_path / src.raw_relpath).exists()


def test_fetch_unavailable_is_reported_not_raised(tmp_path, monkeypatch):
    src = _src(tmp_path, monkeypatch)
    monkeypatch.setattr(F.requests, "get", lambda url, timeout: FakeResp(403, b"Forbidden"))
    assert F.fetch_one(src, {}, force=True)["status"] == "UNAVAILABLE"

    def boom(url, timeout):
        raise F.requests.ConnectionError("Tunnel connection failed: 403")
    monkeypatch.setattr(F.requests, "get", boom)
    rec = F.fetch_one(src, {}, force=True)
    assert rec["status"] == "UNAVAILABLE" and "403" in rec["error"]
    monkeypatch.setattr(F.requests, "get", lambda url, timeout: FakeResp(200, b"version https://git-lfs.github.com/spec/v1"))
    assert F.fetch_one(src, {}, force=True)["status"] == "UNAVAILABLE"   # LFS pointer is not data


def test_installed_skills_match_lock_and_exclusions():
    lock = json.loads((ROOT / "skills.lock.json").read_text())
    skills_dir = ROOT / ".claude" / "skills"
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("inst", ROOT / "scripts" / "install_skills.py")
    inst = module_from_spec(spec)
    spec.loader.exec_module(inst)
    assert not lock["conflicts"]
    for s in lock["installed"]:
        d = skills_dir / s["name"]
        md = (d / "SKILL.md").read_text(encoding="utf-8")
        assert md.startswith("---") and f"name: {s['name']}" in md and "description:" in md
        assert inst.dir_hash(d) == s["sha256"], f"{s['name']} modified since install"
    names = {s["name"] for s in lock["installed"]}
    for must in ("football-data", "nba-data", "nhl-data", "tennis-data", "leakage-audit", "validation-design",
                 "calibration-check", "simulation-sports", "xg-model-building", "elo-engineering",
                 "hockey-analytics", "superforecasting"):
        assert must in names
    for banned in ("polymarket-trading", "machina", "puckapi-tool", "bet-tracker", "edge-detection", "kalshi"):
        assert banned not in names and not (skills_dir / banned).exists()
    assert not (ROOT / ".mcp.json").exists()       # no MCP server auto-connected


def test_no_real_betting_surface_in_lab_code():
    """The lab package must not contain order-placement / payment / browser-automation code."""
    src = "\n".join(p.read_text() for p in (ROOT / "src" / "sportslab").rglob("*.py"))
    for pat in (r"place_bet", r"place_order", r"\bselenium\b", r"playwright", r"private_key", r"payment"):
        assert not re.search(pat, src, re.I), pat


@pytest.mark.historical
def test_historical_database_and_quality():
    from sportslab.data.store import read_table
    q = pd.read_parquet(ROOT / "reports" / "artifacts" / "data_quality.parquet")
    assert not (q.status == "FAIL").any(), q[q.status == "FAIL"]
    assert set(q.status) <= {"PASS", "WARN"}
    ev = read_table("events")
    assert ev.groupby("sport").size().to_dict() == {"basketball": 8886, "football": 3040, "hockey": 13125,
                                                   "tennis": 13514}
    src = read_table("sources")
    assert (src.status.isin(["downloaded", "cached"])).all()
    assert src.sha256.str.len().eq(64).all()


@pytest.mark.historical
def test_historical_models_validated_out_of_sample():
    m = pd.read_parquet(ROOT / "reports" / "artifacts" / "metrics.parquet")
    t = m[m.split == "test"].set_index(["sport", "model"])
    for sport in ("football", "hockey", "basketball", "tennis"):
        uni = t.loc[(sport, "baseline:uniform"), "log_loss"]
        for model in ("elo", "logistic", "ensemble:mean+temp"):
            assert t.loc[(sport, model), "log_loss"] < uni, (sport, model)
    # sharp market closing line is a strong benchmark: our models should not "beat" it by much (sanity)
    assert t.loc[("football", "market:pinnacle-closing"), "log_loss"] < t.loc[("football", "elo"), "log_loss"]


@pytest.mark.historical
def test_historical_backtest_and_journal():
    s = pd.read_parquet(ROOT / "reports" / "artifacts" / "backtest_summary.parquet")
    assert {"in-sample(train+val)", "out-of-sample(test)"} <= set(s.period)
    cc = json.loads((ROOT / "reports" / "artifacts" / "backtest_crosscheck.json").read_text())
    for c in cc["crosscheck"]:
        assert c["abs_diff"] < 1e-6 and c["n"] == c["backtestbuddy_bets"]
    from sportslab.backtest.journal import read_journal
    j = read_journal(cc["run_id"])
    assert len(j) > 0 and (j.note == "VIRTUAL research simulation").all()
