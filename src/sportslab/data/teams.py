"""Canonical team / player names.

Different sources spell the same club differently (the 2025-26 football mirror file
uses "Manchester City" where every other season uses "Man City"; ESPN uses
"Brighton & Hove Albion"). Every name is mapped to ONE canonical spelling, the
football-data.co.uk convention for football, hockey-reference full names for the NHL,
and FiveThirtyEight nicknames for the NBA, so ratings never silently reset.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

FOOTBALL_ALIASES = {
    # football-data.co.uk canonical names on the right
    "manchester city": "Man City", "man city": "Man City",
    "manchester united": "Man United", "manchester utd": "Man United", "man utd": "Man United",
    "man united": "Man United",
    "newcastle united": "Newcastle", "newcastle utd": "Newcastle",
    "nottingham forest": "Nott'm Forest", "nott'm forest": "Nott'm Forest", "nottm forest": "Nott'm Forest",
    "tottenham hotspur": "Tottenham", "spurs": "Tottenham",
    "west ham united": "West Ham", "west ham": "West Ham",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "wolverhampton wanderers": "Wolves", "wolverhampton": "Wolves",
    "brighton & hove albion": "Brighton", "brighton and hove albion": "Brighton", "brighton hove albion": "Brighton",
    "afc bournemouth": "Bournemouth",
    "ipswich town": "Ipswich", "luton town": "Luton",
    "sheffield united": "Sheffield United", "sheffield utd": "Sheffield United",
    "west bromwich albion": "West Brom", "west brom": "West Brom",
    "norwich city": "Norwich", "cardiff city": "Cardiff", "huddersfield town": "Huddersfield",
    "swansea city": "Swansea", "stoke city": "Stoke", "hull city": "Hull",
}

NBA_MULTIWORD = ("Trail Blazers",)


def _key(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.strip().lower().replace(".", ""))


def canonical_football(name: str) -> str:
    return FOOTBALL_ALIASES.get(_key(name), str(name).strip())


def canonical_nba(name: str) -> str:
    """'Portland Trail Blazers' -> 'Trail Blazers', 'LA Clippers' -> 'Clippers' (538 nicknames)."""
    name = str(name).strip()
    for mw in NBA_MULTIWORD:
        if name.endswith(mw):
            return mw
    return name.split()[-1] if name else name


def canonical_nhl(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().strip()
    return {"Utah Hockey Club": "Utah Mammoth"}.get(s, s)


def canonical(sport: str, name: str) -> str:
    return {"football": canonical_football, "basketball": canonical_nba, "hockey": canonical_nhl}.get(
        sport, lambda x: str(x).strip())(name)


def match_known(name: str, known: set[str], cutoff: float = 0.85) -> tuple[str, str]:
    """Return (name_to_use, how): exact | fuzzy:<score> | unknown. Never guesses below the cutoff."""
    if name in known:
        return name, "exact"
    best = difflib.get_close_matches(name, list(known), n=1, cutoff=cutoff)
    if best:
        return best[0], f"fuzzy:{difflib.SequenceMatcher(None, name, best[0]).ratio():.2f}"
    return name, "unknown"
