"""Live (in-play) probability engine.

Rules enforced here:
  * probabilities are updated ONLY from fields present in the snapshot;
  * feed latency is measured (now - source_updated_at) and reported:
      delay > warn  -> status DELAYED (probabilities describe the state at data_as_of)
      delay > max   -> status INSUFFICIENT_DATA (too stale to be useful)
      unknown source timestamp -> DELAYED with delay_seconds=None ("freshness unverified");
  * missing required inputs -> INSUFFICIENT_DATA, probs=None (never invented).

Parameters marked ASSUMPTION are literature-informed defaults that this lab has
NOT estimated from its own data (no minute-level event data is available here).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy.stats import norm, poisson

from sportslab.live import tennis as tn

OK, DELAYED, INSUFFICIENT = "OK", "DELAYED", "INSUFFICIENT_DATA"

# feed staleness thresholds in seconds (warn, max)
STALENESS = {"football": (60, 300), "hockey": (45, 240), "basketball": (30, 180), "tennis": (45, 240)}

# ---- parameters -------------------------------------------------------------------------------
FOOTBALL = dict(
    second_half_share=0.55,    # ESTIMATED by scripts/run_validation.py from EPL HT/FT goals (overwritten)
    prior_weight=6.0,          # ESTIMATED on train seasons from HT->FT goals (overwritten)
    red_card_own=0.67,         # ASSUMPTION: scoring-rate multiplier per own dismissal
    red_card_opp=1.25,         # ASSUMPTION: opponent rate multiplier per dismissal
    match_minutes=90.0,
)
HOCKEY = dict(pp_mult=2.8, sh_mult=0.35,  # ASSUMPTION: power-play / short-handed rate multipliers
              regulation_minutes=60.0)
BASKETBALL = dict(sigma_full=12.5,        # ESTIMATED from FiveThirtyEight nba-elo 2005-15 (overwritten)
                  minutes=48.0)


@dataclass
class LiveSnapshot:
    sport: str
    event_id: str
    source: str
    fetched_at: datetime
    source_updated_at: datetime | None = None
    state: str = "in_progress"           # scheduled | in_progress | final
    home_score: float | None = None
    away_score: float | None = None
    minute: float | None = None          # football elapsed minutes
    red_cards_home: int | None = None
    red_cards_away: int | None = None
    period: int | None = None            # hockey / basketball
    clock_remaining_s: float | None = None
    home_skaters: int | None = None
    away_skaters: int | None = None
    pp_remaining_s: float | None = None
    sets_home: int | None = None         # tennis
    sets_away: int | None = None
    games_home: int | None = None
    games_away: int | None = None
    points_home: int | None = None
    points_away: int | None = None
    home_serving: bool | None = None
    in_tiebreak: bool = False
    best_of: int | None = None
    lineups: dict | None = None
    substitutions: list | None = None
    stats: dict | None = None


@dataclass
class PrematchPrior:
    model: str
    as_of: datetime
    lam: float | None = None      # expected home goals (football/hockey, regulation)
    mu: float | None = None
    p_home: float | None = None   # basketball / tennis
    sources: list = field(default_factory=list)


@dataclass
class LiveForecast:
    status: str
    sport: str
    event_id: str
    labels: tuple
    probs: dict | None
    computed_at: str
    data_as_of: str | None
    delay_seconds: float | None
    reasons: list
    inputs_used: dict
    model: str
    prior_model: str | None
    value_kind: str = "MODEL"

    def to_dict(self):
        return asdict(self)


REQUIRED = {
    "football": ["home_score", "away_score", "minute"],
    "hockey": ["home_score", "away_score", "period", "clock_remaining_s"],
    "basketball": ["home_score", "away_score", "period", "clock_remaining_s"],
    "tennis": ["sets_home", "sets_away", "games_home", "games_away", "points_home", "points_away",
               "home_serving", "best_of"],
}
PRIOR_REQUIRED = {"football": ["lam", "mu"], "hockey": ["lam", "mu"], "basketball": ["p_home"], "tennis": ["p_home"]}


def _iso(dt):
    return None if dt is None else dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class LiveEngine:
    def __init__(self, staleness: dict | None = None, football: dict | None = None,
                 hockey: dict | None = None, basketball: dict | None = None):
        self.staleness = {**STALENESS, **(staleness or {})}
        self.fb = {**FOOTBALL, **(football or {})}
        self.hk = {**HOCKEY, **(hockey or {})}
        self.bb = {**BASKETBALL, **(basketball or {})}

    # --------------------------------------------------------------------- public
    def forecast(self, snap: LiveSnapshot, prior: PrematchPrior | None, now: datetime | None = None) -> LiveForecast:
        now = now or datetime.now(timezone.utc)
        labels = ("H", "D", "A") if snap.sport == "football" else ("H", "A")
        reasons: list[str] = []

        def out(status, probs=None, delay=None, inputs=None, model="-"):
            return LiveForecast(status, snap.sport, snap.event_id, labels, probs, _iso(now),
                                _iso(snap.source_updated_at), delay, reasons, inputs or {}, model,
                                prior.model if prior else None)

        if snap.sport not in REQUIRED:
            reasons.append(f"unsupported sport {snap.sport}")
            return out(INSUFFICIENT)
        # 1) freshness
        delay = None
        status = OK
        warn, mx = self.staleness[snap.sport]
        if snap.source_updated_at is None:
            status = DELAYED
            reasons.append("source provides no update timestamp: freshness UNVERIFIED")
        else:
            delay = (now - snap.source_updated_at).total_seconds()
            if delay < -5:
                reasons.append(f"source timestamp {-delay:.0f}s in the future (clock skew) — rejected")
                return out(INSUFFICIENT, delay=delay)
            if delay > mx:
                reasons.append(f"feed stale: last update {delay:.0f}s ago > max {mx}s")
                return out(INSUFFICIENT, delay=delay)
            if delay > warn:
                status = DELAYED
                reasons.append(f"feed delayed by {delay:.0f}s (> {warn}s); probabilities reflect state at "
                               f"{_iso(snap.source_updated_at)}")
        # 2) completeness
        missing = [f for f in REQUIRED[snap.sport] if getattr(snap, f) is None]
        if snap.state == "final":
            missing = [f for f in missing if f in ("home_score", "away_score")] if snap.sport != "tennis" else \
                [f for f in ("sets_home", "sets_away") if getattr(snap, f) is None]
        if missing:
            reasons.append(f"missing live fields: {missing}")
            return out(INSUFFICIENT, delay=delay)
        if snap.state != "final":
            if prior is None:
                reasons.append("no pre-match prior available")
                return out(INSUFFICIENT, delay=delay)
            pm = [f for f in PRIOR_REQUIRED[snap.sport] if getattr(prior, f) is None]
            if pm:
                reasons.append(f"pre-match prior lacks {pm}")
                return out(INSUFFICIENT, delay=delay)
        # 3) compute
        fn = getattr(self, f"_{snap.sport}")
        probs, inputs, model = fn(snap, prior, reasons)
        if probs is None:
            return out(INSUFFICIENT, delay=delay, inputs=inputs, model=model)
        probs = {k: float(v) for k, v in zip(labels, probs)}
        return out(status, probs, delay, inputs, model)

    # --------------------------------------------------------------------- football
    def football_remaining_share(self, minute: float) -> float:
        s2 = self.fb["second_half_share"]
        t = min(max(minute, 0.0), 90.0)
        if t < 45:
            return ((45 - t) * (1 - s2) + 45 * s2) / 45
        return (90 - t) * s2 / 45

    def football_probs(self, lam, mu, minute, hs, as_, red_h=0, red_a=0):
        rem = self.football_remaining_share(minute)
        elapsed = 1 - rem
        w = self.fb["prior_weight"]
        # Gamma-Poisson update of each team's rate multiplier from goals so far
        mult_h = (w + hs) / (w + lam * elapsed)
        mult_a = (w + as_) / (w + mu * elapsed)
        rh = lam * rem * mult_h * self.fb["red_card_own"] ** red_h * self.fb["red_card_opp"] ** red_a
        ra = mu * rem * mult_a * self.fb["red_card_own"] ** red_a * self.fb["red_card_opp"] ** red_h
        g = np.arange(16)
        M = np.outer(poisson.pmf(g, rh), poisson.pmf(g, ra))
        fh, fa = hs + g[:, None], as_ + g[None, :]
        return np.array([M[fh > fa].sum(), M[fh == fa].sum(), M[fh < fa].sum()]) / M.sum(), rh, ra

    def _football(self, s, prior, reasons):
        if s.state == "final":
            d = np.sign(s.home_score - s.away_score)
            return np.array([d > 0, d == 0, d < 0], float), {"final": True}, "final-score"
        red_h, red_a = s.red_cards_home, s.red_cards_away
        if red_h is None or red_a is None:
            reasons.append("red-card data UNKNOWN in feed -> dismissals not modelled (uncertainty understated)")
        p, rh, ra = self.football_probs(prior.lam, prior.mu, s.minute, s.home_score, s.away_score,
                                        red_h or 0, red_a or 0)
        if s.minute > 90:
            reasons.append("stoppage time: remaining scoring share treated as ~0")
        inputs = dict(minute=s.minute, score=f"{s.home_score:g}-{s.away_score:g}", red_h=red_h, red_a=red_a,
                      prematch_lam=prior.lam, prematch_mu=prior.mu, remaining_lam=rh, remaining_mu=ra)
        return p, inputs, "live-poisson(bayes rate update)"

    # --------------------------------------------------------------------- hockey
    def _hockey(self, s, prior, reasons):
        if s.state == "final":
            return np.array([s.home_score > s.away_score, s.home_score < s.away_score], float), {"final": True}, "final-score"
        regm = self.hk["regulation_minutes"]
        if s.period > 3:
            reasons.append("overtime: winner modelled as next-goal race ∝ scoring rates")
            share = prior.lam / (prior.lam + prior.mu)
            if s.home_score != s.away_score:
                return np.array([s.home_score > s.away_score, s.home_score < s.away_score], float), {}, "ot-final"
            return np.array([share, 1 - share]), dict(period=s.period), "live-poisson-ot"
        rem_min = (3 - s.period) * 20 + s.clock_remaining_s / 60
        rem_min = max(rem_min, 0.0)
        lam_m, mu_m = prior.lam / regm, prior.mu / regm
        pp_min = 0.0
        rh, ra = lam_m * rem_min, mu_m * rem_min
        if s.home_skaters is not None and s.away_skaters is not None and s.home_skaters != s.away_skaters:
            if s.pp_remaining_s is None:
                reasons.append("manpower differential present but penalty time remaining UNKNOWN -> ignored")
            else:
                pp_min = min(s.pp_remaining_s / 60, rem_min)
                adv_home = s.home_skaters > s.away_skaters
                mh, ma = (self.hk["pp_mult"], self.hk["sh_mult"]) if adv_home else (self.hk["sh_mult"], self.hk["pp_mult"])
                rh += lam_m * pp_min * (mh - 1)
                ra += mu_m * pp_min * (ma - 1)
        elif s.home_skaters is None:
            reasons.append("manpower (penalties) UNKNOWN in feed -> even strength assumed for remaining time")
        g = np.arange(16)
        M = np.outer(poisson.pmf(g, rh), poisson.pmf(g, ra))
        fh, fa = s.home_score + g[:, None], s.away_score + g[None, :]
        ph, pd_, pa = M[fh > fa].sum(), M[fh == fa].sum(), M[fh < fa].sum()
        share = prior.lam / (prior.lam + prior.mu)
        p = np.array([ph + pd_ * share, pa + pd_ * (1 - share)])
        inputs = dict(period=s.period, clock_remaining_s=s.clock_remaining_s, score=f"{s.home_score:g}-{s.away_score:g}",
                      home_skaters=s.home_skaters, away_skaters=s.away_skaters, pp_minutes_used=pp_min,
                      remaining_lam=rh, remaining_mu=ra)
        return p / p.sum(), inputs, "live-poisson(manpower-adjusted)"

    # --------------------------------------------------------------------- basketball
    def _basketball(self, s, prior, reasons):
        if s.state == "final":
            return np.array([s.home_score > s.away_score, s.home_score < s.away_score], float), {"final": True}, "final-score"
        tot = self.bb["minutes"]
        elapsed = min((s.period - 1) * 12 + (12 - s.clock_remaining_s / 60), tot) if s.period <= 4 else tot
        if s.period > 4:
            reasons.append("overtime: treated as 5 extra minutes")
            rem = (s.clock_remaining_s / 60) / tot
        else:
            rem = (tot - elapsed) / tot
        sig = self.bb["sigma_full"]
        mu_full = sig * norm.ppf(min(max(prior.p_home, 1e-4), 1 - 1e-4))
        mean = (s.home_score - s.away_score) + mu_full * rem
        sd = sig * math.sqrt(max(rem, 1e-9))
        ph = float(norm.cdf(mean / sd)) if rem > 0 else float(mean > 0) + 0.5 * float(mean == 0)
        inputs = dict(period=s.period, clock_remaining_s=s.clock_remaining_s, margin=s.home_score - s.away_score,
                      remaining_share=rem, prematch_margin=mu_full, sigma_full=sig)
        return np.array([ph, 1 - ph]), inputs, "live-normal-margin"

    # --------------------------------------------------------------------- tennis
    def _tennis(self, s, prior, reasons):
        if s.state == "final":
            return np.array([s.sets_home > s.sets_away, s.sets_home < s.sets_away], float), {"final": True}, "final-score"
        sh, sa = tn.serve_probs_from_prematch(prior.p_home, s.best_of)
        p = tn.p_match(sh, sa, s.best_of, s.sets_home, s.sets_away, s.games_home, s.games_away,
                       bool(s.home_serving), s.points_home, s.points_away, s.in_tiebreak)
        reasons.append("assumes iid points on serve; serve strengths inverted from pre-match probability")
        inputs = dict(score=f"sets {s.sets_home}-{s.sets_away} games {s.games_home}-{s.games_away} "
                            f"points {s.points_home}-{s.points_away}", home_serving=s.home_serving,
                      serve_win_home=sh, serve_win_away=sa)
        return np.array([p, 1 - p]), inputs, "live-markov(point-level)"
