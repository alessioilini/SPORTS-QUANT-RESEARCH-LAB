"""Download raw historical files with provenance + SHA-256 verification."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import requests

from sportslab.config import CONFIG_DIR, RAW_DIR
from sportslab.data.sources import REGISTRY, Source

LOCK_PATH = CONFIG_DIR / "data_sources.lock.json"
MANIFEST_PATH = RAW_DIR / "MANIFEST.json"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_lock() -> dict[str, str]:
    return json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() else {}


def load_manifest() -> dict[str, dict]:
    return json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}


def fetch_one(src: Source, lock: dict[str, str], *, timeout: int = 60, force: bool = False) -> dict:
    dest = RAW_DIR / src.raw_relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    rec = {**asdict(src), "path": str(dest.relative_to(RAW_DIR))}
    if dest.exists() and not force:
        data = dest.read_bytes()
        rec.update(status="cached", http_status=None)
    else:
        try:
            r = requests.get(src.url, timeout=timeout)
        except requests.RequestException as e:   # network policy / DNS / timeout
            rec.update(status="UNAVAILABLE", error=f"{type(e).__name__}: {e}"[:300])
            return rec
        rec["http_status"] = r.status_code
        if r.status_code != 200 or r.content.startswith(b"version https://git-lfs"):
            rec.update(status="UNAVAILABLE", error=f"HTTP {r.status_code}")
            return rec
        data = r.content
        dest.write_bytes(data)
        rec["status"] = "downloaded"
        rec["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    digest = sha256_bytes(data)
    rec.update(sha256=digest, bytes=len(data))
    pinned = lock.get(src.source_id)
    if pinned and pinned != digest:
        rec.update(status="CHECKSUM_MISMATCH", expected_sha256=pinned)
        dest.unlink(missing_ok=True)
    return rec


def fetch_all(source_ids: list[str] | None = None, *, pin_new: bool = True, force: bool = False) -> dict[str, dict]:
    lock = load_lock()
    manifest = load_manifest()
    ids = source_ids or list(REGISTRY)
    for sid in ids:
        rec = fetch_one(REGISTRY[sid], lock, force=force)
        if rec["status"] == "cached" and sid in manifest:
            rec["retrieved_at"] = manifest[sid].get("retrieved_at")
        manifest[sid] = rec
        if pin_new and rec.get("sha256") and sid not in lock and rec["status"] in ("downloaded", "cached"):
            lock[sid] = rec["sha256"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(dict(sorted(lock.items())), indent=2) + "\n")
    return manifest


# ------------------------------------------------------------------ "current" (unpinned) sources
# Files that keep changing (current-season results, upcoming fixtures, recent ESPN results).
# They cannot be SHA-pinned in advance; every download is recorded with its sha256 + retrieved_at.
CURRENT_MANIFEST_PATH = RAW_DIR / "CURRENT_MANIFEST.json"


def load_current_manifest() -> dict[str, dict]:
    return json.loads(CURRENT_MANIFEST_PATH.read_text()) if CURRENT_MANIFEST_PATH.exists() else {}


def current_sources(sport: str, role: str = "current-results") -> dict[str, dict]:
    return {sid: m for sid, m in load_current_manifest().items()
            if m.get("sport") == sport and m.get("role") == role and (RAW_DIR / m["path"]).exists()}


def register_current(sid: str, *, sport: str, path: str, data: bytes, url: str, original_provider: str,
                     season: str | None, role: str, notes: str = "") -> dict:
    """Write bytes to data/raw/<path> and record provenance in CURRENT_MANIFEST.json."""
    dest = RAW_DIR / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    rec = dict(sport=sport, path=path, url=url, original_provider=original_provider, season=season, role=role,
               retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               sha256=sha256_bytes(data), bytes=len(data), status="downloaded", notes=notes)
    man = load_current_manifest()
    man[sid] = rec
    CURRENT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_MANIFEST_PATH.write_text(json.dumps(man, indent=2) + "\n")
    return rec


def http_get(url: str, timeout: int = 30) -> tuple[bytes | None, str | None]:
    """GET returning (content, error). Never raises; 'error' explains why data is UNAVAILABLE."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "sports-quant-research-lab/0.1 (research)"})
    except requests.RequestException as e:
        return None, f"{type(e).__name__}: {e}"[:300]
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    return r.content, None
