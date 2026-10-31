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

AVAILABLE, UNAVAILABLE, NOT_CONNECTED = "AVAILABLE", "UNAVAILABLE", "NOT_CONNECTED"


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


def _state(status: str) -> str:
    status = (status or "").lower()
    if status in ("closed", "final", "status_final", "ended", "complete", "completed"):
        return "final"
    if status in ("not_started", "scheduled", "status_scheduled", "pre", ""):
        return "scheduled"
    return "in_progress"


def _side(c: dict) -> str | None:
    return c.get("home_away") or c.get("homeAway") or c.get("qualifier")


def snapshot_from_scoreboard_event(sport: str, ev: dict, fetched_at: datetime) -> LiveSnapshot:
    """Normalised sports-skills event (nhl/nba scoreboard, football daily schedule) -> LiveSnapshot.
    Fields the payload does not carry stay None (the engine then reports INSUFFICIENT_DATA)."""
    comps = {_side(c): c for c in ev.get("competitors", [])}
    home, away = comps.get("home", {}), comps.get("away", {})
    scores = ev.get("scores") or {}
    snap = LiveSnapshot(sport=sport, event_id=f"espn:{ev.get('id')}", source=f"sports-skills/{_SPORT_MODULE[sport]}",
                        fetched_at=fetched_at, source_updated_at=None, state=_state(ev.get("status", "")),
                        home_score=home.get("score", scores.get("home")),
                        away_score=away.get("score", scores.get("away")))
    detail = ev.get("status_detail", "")
    if sport in ("hockey", "basketball"):
        snap.period, snap.clock_remaining_s = _parse_clock(detail)
    elif sport == "football":
        snap.minute = _parse_minute(detail)     # daily schedule carries no clock -> stays None
    snap.stats = {"home": (home.get("team") or {}).get("name") or home.get("name"),
                  "away": (away.get("team") or {}).get("name") or away.get("name"), "status_detail": detail,
                  "start_time": ev.get("start_time"), "competition": (ev.get("competition") or {}).get("id")}
    return snap


def _set_won(a: int, b: int) -> bool:
    return (a >= 6 and a - b >= 2) or a == 7


def snapshot_from_tennis_match(m: dict, fetched_at: datetime, tournament: str = "") -> LiveSnapshot:
    """sports-skills tennis match (competitors[].set_scores, serving) -> LiveSnapshot.
    The feed has no point score inside the current game -> points stay None (UNKNOWN)."""
    c = m.get("competitors", [])
    p1, p2 = (c + [{}, {}])[:2]
    s1 = [x.get("games", 0) for x in p1.get("set_scores", [])]
    s2 = [x.get("games", 0) for x in p2.get("set_scores", [])]
    state = _state(m.get("status", ""))
    n = min(len(s1), len(s2))
    done = n if state == "final" else max(n - 1, 0)          # last listed set is in progress
    sets_h = sum(1 for a, b in zip(s1[:done], s2[:done]) if _set_won(a, b))
    sets_a = sum(1 for a, b in zip(s1[:done], s2[:done]) if _set_won(b, a))
    in_prog = state == "in_progress" and n > done
    serving = True if p1.get("serving") else (False if p2.get("serving") else None)
    return LiveSnapshot(sport="tennis", event_id=f"espn:{m.get('id')}", source="sports-skills/tennis",
                        fetched_at=fetched_at, source_updated_at=None, state=state,
                        sets_home=sets_h, sets_away=sets_a,
                        games_home=s1[-1] if in_prog else (0 if state == "in_progress" else None),
                        games_away=s2[-1] if in_prog else (0 if state == "in_progress" else None),
                        points_home=None, points_away=None, home_serving=serving, best_of=None,
                        stats={"home": p1.get("name"), "away": p2.get("name"), "tournament": tournament,
                               "round": m.get("round"), "status_detail": m.get("status_detail"),
                               "start_time": m.get("date"), "type": p1.get("type")})


# module, function, how to extract events
_FEEDS = {"football": ("football", "get_daily_schedule"), "hockey": ("nhl", "get_scoreboard"),
          "basketball": ("nba", "get_scoreboard"), "tennis": ("tennis", "get_scoreboard")}


class SportsSkillsLive:
    """Read-only reader of the installed sports-skills package. Failures -> UNAVAILABLE (never fabricated)."""

    def __init__(self, sport: str):
        self.sport = sport

    def fetch(self, **params) -> SourceResult:
        fetched = _now()
        module, fn = _FEEDS[self.sport]
        src = f"sports-skills/{module}.{fn}"
        try:
            import importlib
            data = getattr(importlib.import_module(f"sports_skills.{module}"), fn)(**params)
        except Exception as e:  # noqa: BLE001 — any import/network error means unavailable
            return SourceResult(src, UNAVAILABLE, fetched, error=f"{type(e).__name__}: {e}"[:300])
        return self.parse(src, data, fetched)

    def parse(self, src: str, data, fetched: datetime) -> SourceResult:
        if not isinstance(data, dict) or data.get("status") is False or data.get("error"):
            msg = (data or {}).get("message") if isinstance(data, dict) else str(data)
            return SourceResult(src, UNAVAILABLE, fetched, error=str(msg)[:300])
        payload = data.get("data", data)
        if not isinstance(payload, dict):
            return SourceResult(src, UNAVAILABLE, fetched, error="unexpected payload shape")
        if self.sport == "tennis":
            snaps = [snapshot_from_tennis_match(m, fetched, t.get("name", ""))
                     for t in payload.get("tournaments", []) for d in t.get("draws", [])
                     for m in d.get("matches", [])]
        else:
            snaps = [snapshot_from_scoreboard_event(self.sport, e, fetched) for e in payload.get("events", [])]
        return SourceResult(src, AVAILABLE, fetched, snaps, raw_count=len(snaps))


class InferSports:
    """Placeholder for the InferSports read-only MCP. Deliberately not connected — see config/mcp/."""

    def fetch(self, **_) -> SourceResult:
        return SourceResult("infersports-mcp", NOT_CONNECTED, _now(),
                            error="not connected by design: the official endpoint could not be verified. "
                                  "To enable, follow config/mcp/infersports.DISABLED.json")


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
