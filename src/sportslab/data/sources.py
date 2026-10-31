"""Registry of historical data sources.

Every file is fetched from an immutable URL (git commit SHA) so the dataset is
reproducible. Mirrors are labelled with the original publisher. SHA-256 of
each downloaded file is pinned in config/data_sources.lock.json after the first
verified download and checked on every later fetch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

RAW = "https://raw.githubusercontent.com"

MIRRORS = {
    "fd_mirror": ("Mentat-Uran/ScoutFootball_for_World_Cup", "4be5e5d70460442182fbf4cab107d4c4497ecd86"),
    "fd_crosscheck": ("datasets/football-datasets", "436a51f15258e7d0c6042c231227202b70a5a0ae"),
    "fte_checking": ("fivethirtyeight/checking-our-work-data", "f6d5b2e1d6da2889345d381c41431f9a4ee208dd"),
    "fte_data": ("fivethirtyeight/data", "4c1ff5e3aef1816ae04af63218015066e186c147"),
    "hockeyref_mirror": ("Kalman-filter/main", "9bde84b0aec9f00d4104cce2b69670808f2c2a43"),
    "tennisdata_mirror": ("DanielSzakacs/atp_data", "9e35f3e0f3a4318b3e6c543ac464509eeb70ef41"),
}


@dataclass(frozen=True)
class Source:
    source_id: str
    sport: str
    kind: str
    url: str
    original_provider: str
    host_repo: str
    license_note: str
    season: str | None = None
    role: str = "primary"          # primary | crosscheck | parameter-estimation | benchmark
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def raw_relpath(self) -> str:
        return f"{self.sport}/{self.source_id}.csv"


def _url(mirror: str, path: str) -> tuple[str, str]:
    repo, sha = MIRRORS[mirror]
    return f"{RAW}/{repo}/{sha}/{path}", f"{repo}@{sha[:10]}"


def _season_label(code: str) -> str:   # "1819" -> "2018-19"
    return f"20{code[:2]}-{code[2:]}"


def build_registry() -> dict[str, Source]:
    reg: dict[str, Source] = {}
    for code in ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]:
        url, host = _url("fd_mirror", f"data/raw/football_data/{code}/E0.csv")
        sid = f"football_epl_{code}"
        reg[sid] = Source(
            sid, "football", "results+match_stats+odds(opening,closing)", url,
            "football-data.co.uk (E0.csv, English Premier League)", host,
            "football-data.co.uk: free for research use; mirror redistributes unmodified CSV",
            season=_season_label(code),
            notes="Opening odds collected Fri/Tue afternoons (provider policy); closing odds = last pre-kickoff.",
        )
    url, host = _url("fd_crosscheck", "datasets/premier-league/season-2324.csv")
    reg["football_epl_2324_crosscheck"] = Source(
        "football_epl_2324_crosscheck", "football", "results (cross-check)", url,
        "football-data.co.uk via datasets.io", host, "ODC-PDDL (datasets.io packaging)",
        season="2023-24", role="crosscheck",
        notes="Independent mirror used only to verify the primary mirror's results.")
    for y0 in range(2010, 2020):
        season = f"{y0}-{str(y0 + 1)[2:]}"
        url, host = _url("hockeyref_mirror", f"DATA/NHL/NHL-{y0}-{y0 + 1}.csv")
        sid = f"hockey_nhl_{y0}{str(y0 + 1)[2:]}"
        reg[sid] = Source(
            sid, "hockey", "results (final score, OT/SO flag)", url,
            "hockey-reference.com (season game logs)", host,
            "hockey-reference data; research use, not redistributed by this repo",
            season=season, notes="Date only (no start time). Score includes OT/SO winner.")
    url, host = _url("fte_checking", "nhl_games.csv")
    reg["hockey_nhl_538_2122"] = Source(
        "hockey_nhl_538_2122", "hockey", "external forecasts + outcomes", url,
        "FiveThirtyEight 'Checking Our Work'", host, "CC BY 4.0 (FiveThirtyEight)",
        season="2021-22", role="benchmark",
        notes="FiveThirtyEight's published game win probabilities (external model, OBSERVED).")
    url, host = _url("fte_checking", "nba_games.csv")
    reg["basketball_nba_538"] = Source(
        "basketball_nba_538", "basketball", "outcomes + external forecasts", url,
        "FiveThirtyEight 'Checking Our Work'", host, "CC BY 4.0 (FiveThirtyEight)",
        season="2015-16..2021-22",
        notes="No scores; team1/team2 + 538 pre-game win probability + outcome.")
    url, host = _url("fte_data", "nba-elo/nbaallelo.csv")
    reg["basketball_nba_elo_hist"] = Source(
        "basketball_nba_elo_hist", "basketball", "historical scores (1946-2015)", url,
        "FiveThirtyEight nba-elo", host, "CC BY 4.0 (FiveThirtyEight)",
        role="parameter-estimation",
        notes="Used only to estimate the live final-margin dispersion (2005-2015 seasons).")
    for y in range(2019, 2025):
        url, host = _url("tennisdata_mirror", f"{y}.csv")
        sid = f"tennis_atp_{y}"
        reg[sid] = Source(
            sid, "tennis", "results + set scores + rankings + odds", url,
            "tennis-data.co.uk (ATP)", host,
            "tennis-data.co.uk: free for research use; mirror redistributes CSV",
            season=str(y), notes="Match date only. Odds: single pre-match snapshot (timestamp unknown).")
    return reg


REGISTRY = build_registry()
