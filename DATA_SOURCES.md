# DATA_SOURCES — Sports Quant Research Lab

All historical data is fetched by `python scripts/fetch_data.py` from **immutable URLs** (git commit
SHA in the path). On the first verified download, each file's SHA-256 is pinned in
`config/data_sources.lock.json`. Later fetches that don't match are rejected (`CHECKSUM_MISMATCH`)
and the file is deleted. Raw files are **not** committed (`data/` is git-ignored): rebuild with
`fetch_data.py` followed by `build_db.py`.

Status on 2026-10-31: **28/28 historical sources downloaded and verified**; data-quality checks: 30 PASS, 2 WARN (odds coverage), 0 FAIL. Every live source is
unavailable in this environment (see the last section).

## 1. Registered historical sources

| source_id | sport | role | original publisher | mirror (repo@commit) | season | bytes | sha256 |
|---|---|---|---|---|---|---|---|
| `football_epl_1819` | football | primary | football-data.co.uk (E0.csv, English Premier League) | `Mentat-Uran/ScoutFootball_for_World_Cup@4be5e5d704` | 2018-19 | 95,763 | `4a44573970f0…` |
| `football_epl_1920` | football | primary | football-data.co.uk | same | 2019-20 | 175,094 | `92bf96bebc4f…` |
| `football_epl_2021` | football | primary | football-data.co.uk | same | 2020-21 | 175,434 | `5fe0773d57db…` |
| `football_epl_2122` | football | primary | football-data.co.uk | same | 2021-22 | 174,982 | `95d1dd4afacf…` |
| `football_epl_2223` | football | primary | football-data.co.uk | same | 2022-23 | 175,799 | `e51361323bcd…` |
| `football_epl_2324` | football | primary | football-data.co.uk | same | 2023-24 | 171,815 | `0b669038e94b…` |
| `football_epl_2425` | football | primary | football-data.co.uk | same | 2024-25 | 196,729 | `316173b11634…` |
| `football_epl_2526` | football | primary | football-data.co.uk | same | 2025-26 | 218,368 | `bfa8585d7a6a…` |
| `football_epl_2324_crosscheck` | football | crosscheck | football-data.co.uk via datasets.io | `datasets/football-datasets@436a51f152` | 2023-24 | 30,383 | `f64a0999d037…` |
| `hockey_nhl_201011` … `hockey_nhl_201920` (10 files) | hockey | primary | hockey-reference.com season game logs | `Kalman-filter/main@9bde84b0ae` | 2010-11 … 2019-20 | 45–89 KB each | see lock file |
| `hockey_nhl_538_2122` | hockey | benchmark | FiveThirtyEight “Checking Our Work” | `fivethirtyeight/checking-our-work-data@f6d5b2e1d6` | 2021-22 | 105,551 | `f9b46b204820…` |
| `basketball_nba_538` | basketball | primary | FiveThirtyEight “Checking Our Work” | `fivethirtyeight/checking-our-work-data@f6d5b2e1d6` | 2015-16 … 2021-22 | 655,771 | `fb99d5a4870a…` |
| `basketball_nba_elo_hist` | basketball | parameter-estimation | FiveThirtyEight nba-elo | `fivethirtyeight/data@4c1ff5e3ae` | 1946 … 2015 | 18,173,265 | `d46ed3540ee8…` |
| `tennis_atp_2019` … `tennis_atp_2024` (6 files) | tennis | primary | tennis-data.co.uk (ATP) | `DanielSzakacs/atp_data@9e35f3e0f3` | 2019 … 2024 | 226–481 KB each | see lock file |

For full SHA-256 values, see `config/data_sources.lock.json`. Per-file retrieval times and HTTP
status are in `data/raw/MANIFEST.json` and in the `sources` table of `data/sportslab.sqlite`.

### Mirror verification
The primary sources come from third-party GitHub mirrors, because the original hosts
(football-data.co.uk, tennis-data.co.uk) are blocked by this environment's egress proxy. The
football mirror was checked against an independent mirror for 2023-24: FT goals and shots on target
matched on **380/380** matches (`crosscheck_epl_2324_vs_independent_mirror`). The NBA `team1 = home`
convention was checked on known fixtures: the 2015-10-27 openers (Warriors, Hawks and Bulls at home)
and 2022 Finals Game 6 (Celtics at home).

## 2. What each sport contains

| field | football (EPL) | hockey (NHL) | basketball (NBA) | tennis (ATP) |
|---|---|---|---|---|
| final result | OBSERVED (FT + HT score) | OBSERVED (score, OT/SO flag) | OBSERVED (winner only) | OBSERVED (winner, sets, retirements) |
| kickoff time | OBSERVED (UK local; 2018-19 date only) | UNKNOWN (date only) | UNKNOWN (date only) | UNKNOWN (date only) |
| team match stats | OBSERVED (shots, SOT, corners, fouls, cards) | UNKNOWN | UNKNOWN | UNKNOWN |
| player pre-match stats | UNKNOWN | UNKNOWN | UNKNOWN | OBSERVED (ATP rank, points) |
| lineups | UNKNOWN | UNKNOWN | UNKNOWN | n/a |
| injuries | UNKNOWN (no verifiable source) | UNKNOWN | UNKNOWN | UNKNOWN |
| market odds | OBSERVED: Pinnacle, Bet365, market avg/max — opening **and** closing | UNKNOWN (no free source reachable) | UNKNOWN | OBSERVED: Pinnacle, Bet365, avg/max — single snapshot, time UNKNOWN |
| external forecasts | — | OBSERVED FiveThirtyEight (2021-22 only) | OBSERVED FiveThirtyEight | — |

### Known gaps (found by the data-quality checks)
* **Football 2025-26:** Pinnacle opening and closing odds are missing in the source file for the 170 matches
  from 2026-01-17 to the end of the season (`odds_coverage_by_book_season` = WARN, 55% coverage). Bet365 and
  market average/max prices are complete. Because of this, the Pinnacle-based backtest and the
  `market:pinnacle-closing` benchmark cover 590 of the 760 test matches. `market:pinnacle-opening` falls back to
  the market-average opening price for those 170 matches.
* **Tennis 2024:** 1,813 matches. The mirror stops before the second-half hard-court swing.
* **Tennis 2022 ATP Finals:** `Best of` is missing for 15 matches and is stored as UNKNOWN (`None`), not
  guessed.
* **NHL 2013-14:** one game without a score is dropped (documented in the mirror's readme).

The `lineups` and `injuries` tables exist but are **empty on purpose**: no verifiable source was
reachable, so nothing was filled in. The `coverage` table records each UNKNOWN with the reason.

## 3. Provenance model (stage 3)

Every fact row in SQLite carries:

| column | meaning |
|---|---|
| `source_id` | registered source that produced the row |
| `retrieved_at` | UTC time the raw file was downloaded |
| `available_at` | earliest UTC instant the fact was public in reality |
| `value_kind` | `OBSERVED` (from a source) · `MODEL` (computed here) · `UNKNOWN` |

Tables: `sources, events, results, team_match_stats, player_prematch, lineups, injuries, odds,
external_forecasts, coverage, model_forecasts, forecast_log, virtual_journal, live_snapshots`.
The SQLite DB is mirrored to Parquet in `data/processed/*.parquet`.

## 4. Point-in-time (no look-ahead) conventions

| sport | event_time | forecast_time (decision moment) | result `available_at` |
|---|---|---|---|
| football | kickoff (UTC) | opening-odds collection instant: latest Fri/Tue 16:00 UK before kickoff−1h (football-data.co.uk policy; *approx*) | kickoff + 2h30 |
| hockey / basketball | date 23:00 UTC (nominal) | date 12:00 UTC (before any NA game) | date+1 10:00 UTC |
| tennis | date 12:00 UTC (nominal) | **date−1** 12:00 UTC (match local time unknown → conservative) | date+1 12:00 UTC |

Column availability classes (`sportslab.data.pit`):
`id` and `pre` columns (opening odds, rankings, published external forecasts) are visible to models.
`close` (closing odds, odds snapshots of unknown time) and `post` (results, match stats) are
**never** passed to a model for the match being predicted. The walk-forward driver releases results
only after `available_at`. Tests check this, including "poisoning" tests where every future result
is scrambled and past predictions must not change.

## 5. Sources that are unavailable or deliberately not used

| source | status | reason |
|---|---|---|
| ESPN site API (sports-skills football/nba/nhl/tennis) | **UNAVAILABLE** | egress proxy: `Tunnel connection failed: 403` (verified 2026-10-31) |
| NBA CDN live scoreboard | **UNAVAILABLE** | egress proxy 403 |
| NHL api-web | **UNAVAILABLE** | egress proxy (connection refused) |
| InferSports MCP | **NOT CONNECTED** | official endpoint could not be verified (only third-party listings found); `infersports.com` and `api.infersports.com` blocked (403). Template: `config/mcp/infersports.DISABLED.json` |
| football-data.co.uk, tennis-data.co.uk (direct) | UNAVAILABLE | proxy-blocked → pinned GitHub mirrors used instead |
| JeffSackmann/tennis_atp | UNAVAILABLE | repository returns 403/404 |
| PuckAPI | NOT CONNECTED | paid API, excluded by request |
| HuggingFace, Kaggle, FiveThirtyEight project pages | UNAVAILABLE | proxy-blocked |

Live functionality was tested with synthetic snapshots and one **real** recorded NHL gamecenter
payload (`tests/fixtures/nhl_gamecenter_2023030417_pbp_partial.json`, 2024 Stanley Cup Final G7,
first 30 plays, from the MIT-licensed sports-skills test fixtures).

## 6. Licensing
- football-data.co.uk and tennis-data.co.uk: free for research. This repo does not redistribute
  their files; it downloads them from mirrors.
- FiveThirtyEight data: CC BY 4.0.
- hockey-reference data: used for research only and not redistributed.
- Skill and library code: MIT. Licenses are in `third_party/licenses/`.
