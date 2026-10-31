"""Online, point-in-time feature engines.

State is updated ONLY through `observe(rows)` (rows whose results are public),
and features for fixtures are computed from that state — so every feature is
a pre-match quantity by construction. Features are snapshotted at prediction
time for later supervised training.
"""
from __future__ import annotations

import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from sportslab.models.ratings import Elo, TennisElo


class TeamFeatureEngine:
    """Football / hockey / basketball team features."""

    def __init__(self, sport: str, labels, window: int = 8):
        self.sport, self.window = sport, window
        three = len(labels) == 3
        self.elo = Elo(labels, k=20 if three else 8, home_adv=60 if three else 50, mov=three)
        self.hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.last_played: dict[str, pd.Timestamp] = {}
        self.season_games: dict[tuple, int] = defaultdict(int)

    def observe(self, rows: pd.DataFrame) -> None:
        self.elo.observe(rows)
        for r in rows.itertuples(index=False):
            hs, as_ = getattr(r, "home_score", np.nan), getattr(r, "away_score", np.nan)
            hsot, asot = getattr(r, "home_shots_on_target", np.nan), getattr(r, "away_shots_on_target", np.nan)
            win_h = 1.0 if r.outcome == "H" else (0.5 if r.outcome == "D" else 0.0)
            pts_h = {"H": 3, "D": 1, "A": 0}[r.outcome]
            pts_a = {"H": 0, "D": 1, "A": 3}[r.outcome]
            self.hist[r.home].append((hs, as_, hsot, asot, win_h, pts_h))
            self.hist[r.away].append((as_, hs, asot, hsot, 1 - win_h, pts_a))
            for t in (r.home, r.away):
                self.last_played[t] = max(self.last_played.get(t, r.event_time), r.event_time)
                self.season_games[(t, r.season)] += 1

    def _form(self, team):
        h = self.hist.get(team)
        if not h:
            return [np.nan] * 6
        a = np.array(h, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns (e.g. no scores in source)
            return list(np.nanmean(a, axis=0))

    def features(self, fx: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for r in fx.itertuples(index=False):
            self.elo._maybe_new_season(r.season)
            eh, ea = self.elo._get(r.home), self.elo._get(r.away)
            fh, fa = self._form(r.home), self._form(r.away)
            rest = []
            for t in (r.home, r.away):
                lp = self.last_played.get(t)
                rest.append(min((r.event_time - lp).total_seconds() / 86400, 14) if lp is not None else 14.0)
            d = dict(elo_diff=eh - ea + (0 if r.neutral else self.elo.home_adv),
                     form_gf_h=fh[0], form_ga_h=fh[1], form_gf_a=fa[0], form_ga_a=fa[1],
                     form_win_h=fh[4], form_win_a=fa[4],
                     rest_h=rest[0], rest_a=rest[1], b2b_h=float(rest[0] <= 1.5), b2b_a=float(rest[1] <= 1.5),
                     games_h=self.season_games[(r.home, r.season)], games_a=self.season_games[(r.away, r.season)])
            if self.sport == "football":
                d.update(form_sotf_h=fh[2], form_sota_h=fh[3], form_sotf_a=fa[2], form_sota_a=fa[3],
                         form_pts_h=fh[5], form_pts_a=fa[5])
            if self.sport == "basketball":
                for k in ("form_gf_h", "form_ga_h", "form_gf_a", "form_ga_a"):
                    d.pop(k)   # no scores in source
            rows.append(d)
        return pd.DataFrame(rows, index=fx.index)


class TennisFeatureEngine:
    def __init__(self, labels=("H", "A")):
        self.elo = TennisElo(labels, surface_weight=0.0)   # observe() updates overall AND surface tables
        self.dates: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

    def observe(self, rows: pd.DataFrame) -> None:
        self.elo.observe(rows)   # updates both overall and surface tables
        for r in rows.itertuples(index=False):
            for p in (r.home, r.away):
                self.dates[p].append(r.event_time)

    def _recent(self, p, t, days):
        return sum(1 for d in self.dates.get(p, ()) if (t - d).days <= days)

    def features(self, fx: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for r in fx.itertuples(index=False):
            rh, ra = getattr(r, "pre_home_rank", np.nan), getattr(r, "pre_away_rank", np.nan)
            ph, pa = getattr(r, "pre_home_rank_points", np.nan), getattr(r, "pre_away_rank_points", np.nan)
            rows.append(dict(
                elo_diff=self.elo.rating(r.home, None) - self.elo.rating(r.away, None),
                surf_elo_diff=self.elo.rs[(r.home, r.surface)] - self.elo.rs[(r.away, r.surface)],
                log_rank_ratio=np.log(ra / rh) if (rh and ra and rh > 0 and ra > 0) else np.nan,
                log_pts_ratio=np.log(ph / pa) if (ph and pa and ph > 0 and pa > 0) else np.nan,
                n_h=self.elo.n[r.home], n_a=self.elo.n[r.away],
                recent14_h=self._recent(r.home, r.event_time, 14), recent14_a=self._recent(r.away, r.event_time, 14),
                bo5=float(r.best_of == 5) if pd.notna(r.best_of) else np.nan,
                clay=float(r.surface == "Clay"), grass=float(r.surface == "Grass")))
        return pd.DataFrame(rows, index=fx.index)


def make_engine(sport: str, labels):
    return TennisFeatureEngine(labels) if sport == "tennis" else TeamFeatureEngine(sport, labels)
