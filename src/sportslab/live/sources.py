"""Read-only live data adapters.

* SportsSkillsLive — wraps the installed `sports_skills` package (ESPN/NBA CDN
  public endpoints). Only scoreboard reads are used.
* parse_nhl_gamecenter — converts an NHL api-web gamecenter play-by-play
  payload into a LiveSnapshot (score, period, clock, manpower, PP time).
* InferSports — NOT connected (endpoint unverified, host blocked by network
  policy); always reports UNAVAILABLE with the reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sportslab.live.engine import LiveSnapshot

AVAILABLE, UNAVAILABLE = "AVAILABLE", "UNAVAILABLE"


@dataclass
class SourceResult:
    source: str
    status: str
    fetched_at: datetime
    snapshots: list = field(default_factory=list)
    error: str | None = None
    raw_count: int = 0


def _now():
    return datetime.now(timezone.utc)


_SPORT_MODULE = {"football": "football", "hockey": "nhl", "basketball": "nba", "tennis": "tennis"}


def _parse_clock(detail: str):
    """'12:34 - 2nd', '5:21 - 3rd Quarter', 'End of 2nd' -> (period, seconds_remaining)."""
    m = re.search(r"(\d{1,2}):(\d{2})\s*-\s*(\d)(?:st|nd|rd|th)", detail or "")
    if m:
        return int(m.group(3)), int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r"End of (\d)(?:st|nd|rd|th)", detail or "")
    if m:
        return int(m.group(1)), 0
    return None, None


def _parse_minute(detail: str):
    m = re.search(r"(\d{1,3})(?:\+(\d{1,2}))?'", detail or "")
    if m:
        return float(m.group(1)) + float(m.group(2) or 0)
    if (detail or "").upper() in ("HT", "HALFTIME"):
        return 45.0
    return None


def snapshot_from_scoreboard_event(sport: str, ev: dict, fetched_at: datetime) -> LiveSnapshot:
    """Normalised sports-skills scoreboard event -> LiveSnapshot. Unknown fields stay None."""
    comps = {c.get("home_away"): c for c in ev.get("competitors", [])}
    home, away = comps.get("home", {}), comps.get("away", {})
    status = ev.get("status", "")
    state = "final" if status in ("closed", "final", "STATUS_FINAL") else (
        "scheduled" if status in ("not_started", "scheduled", "STATUS_SCHEDULED") else "in_progress")
    snap = LiveSnapshot(sport=sport, event_id=f"espn:{ev.get('id')}", source=f"sports-skills/{_SPORT_MODULE[sport]}",
                        fetched_at=fetched_at, source_updated_at=None, state=state,
                        home_score=home.get("score"), away_score=away.get("score"))
    detail = ev.get("status_detail", "")
    if sport in ("hockey", "basketball"):
        snap.period, snap.clock_remaining_s = _parse_clock(detail)
    elif sport == "football":
        snap.minute = _parse_minute(detail)
    return snap


class SportsSkillsLive:
    """Read-only scoreboard reader. Network failures -> UNAVAILABLE (never fabricated)."""

    def __init__(self, sport: str):
        self.sport = sport

    def fetch(self, **params) -> SourceResult:
        fetched = _now()
        src = f"sports-skills/{_SPORT_MODULE[self.sport]}.get_scoreboard"
        try:
            import importlib
            mod = importlib.import_module(f"sports_skills.{_SPORT_MODULE[self.sport]}")
            data = mod.get_scoreboard(**params)
        except Exception as e:  # noqa: BLE001 — any import/network error means unavailable
            return SourceResult(src, UNAVAILABLE, fetched, error=f"{type(e).__name__}: {e}"[:300])
        if not isinstance(data, dict) or data.get("status") is False or data.get("error"):
            msg = (data or {}).get("message") if isinstance(data, dict) else str(data)
            return SourceResult(src, UNAVAILABLE, fetched, error=str(msg)[:300])
        payload = data.get("data", data)
        events = payload.get("events", []) if isinstance(payload, dict) else []
        snaps = [snapshot_from_scoreboard_event(self.sport, e, fetched) for e in events]
        return SourceResult(src, AVAILABLE, fetched, snaps, raw_count=len(events))


class InferSports:
    """Placeholder for the InferSports read-only MCP. Not connected — see config/mcp/."""

    def fetch(self, **_) -> SourceResult:
        return SourceResult("infersports-mcp", UNAVAILABLE, _now(),
                            error="not connected: official endpoint unverified; infersports.com blocked by "
                                  "egress policy (HTTP 403). See config/mcp/infersports.DISABLED.json")


def parse_nhl_gamecenter(pbp: dict, fetched_at: datetime, source_updated_at: datetime | None = None) -> LiveSnapshot:
    """NHL api-web play-by-play -> LiveSnapshot at the last recorded play.

    situationCode digits: [away goalie][away skaters][home skaters][home goalie].
    """
    home_id, away_id = pbp["homeTeam"]["id"], pbp["awayTeam"]["id"]
    plays = pbp.get("plays", [])
    hs = sum(1 for p in plays if p["typeDescKey"] == "goal" and p["details"].get("eventOwnerTeamId") == home_id)
    as_ = sum(1 for p in plays if p["typeDescKey"] == "goal" and p["details"].get("eventOwnerTeamId") == away_id)
    last = plays[-1] if plays else None
    period = last["periodDescriptor"]["number"] if last else None
    rem = None
    if last and last.get("timeRemaining"):
        mm, ss = last["timeRemaining"].split(":")
        rem = int(mm) * 60 + int(ss)
    sc = last.get("situationCode") if last else None
    home_sk = away_sk = None
    if sc and len(sc) == 4:
        away_sk, home_sk = int(sc[1]), int(sc[2])
    # penalty time remaining for the most recent active minor (if manpower differs)
    pp_rem = None
    if last and home_sk is not None and home_sk != away_sk:
        def abs_s(p):
            m, s = p["timeInPeriod"].split(":")
            return (p["periodDescriptor"]["number"] - 1) * 1200 + int(m) * 60 + int(s)
        now_s = abs_s(last)
        active = [abs_s(p) + 60 * p["details"].get("duration", 2) - now_s for p in plays if p["typeDescKey"] == "penalty"]
        active = [a for a in active if a > 0]
        pp_rem = max(active) if active else None
    state = "final" if pbp.get("gameState") in ("OFF", "FINAL") and len(plays) and \
        plays[-1]["typeDescKey"] == "game-end" else "in_progress"
    return LiveSnapshot(sport="hockey", event_id=f"nhl:{pbp['id']}", source="nhl-api-web/gamecenter",
                        fetched_at=fetched_at, source_updated_at=source_updated_at, state=state,
                        home_score=hs, away_score=as_, period=period, clock_remaining_s=rem,
                        home_skaters=home_sk, away_skaters=away_sk, pp_remaining_s=pp_rem)
