"""Data-quality checks. Each check returns a row: (check, sport, status, value, detail)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportslab.config import RAW_DIR
from sportslab.data.sources import REGISTRY


def _row(check, sport, ok, value, detail=""):
    return dict(check=check, sport=sport, status="PASS" if ok else "FAIL", value=value, detail=detail)


def run_checks(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ev, res, odds = t["events"], t["results"], t["odds"]
    rows = []
    for sport, e in ev.groupby("sport"):
        r = res[res.event_id.isin(e.event_id)]
        rows.append(_row("events_have_result", sport, len(r) == len(e), f"{len(r)}/{len(e)}"))
        rows.append(_row("forecast_time<=event_time", sport,
                         bool((pd.to_datetime(e.forecast_time) <= pd.to_datetime(e.event_time)).all()), len(e)))
        m = r.merge(e[["event_id", "event_time"]], on="event_id")
        rows.append(_row("result_available_after_event", sport,
                         bool((pd.to_datetime(m.available_at) > pd.to_datetime(m.event_time)).all()), len(m)))
        rows.append(_row("unique_event_ids", sport, not e.event_id.duplicated().any(), e.event_id.nunique()))
        if sport in ("football", "hockey"):
            sc = r.dropna(subset=["home_score"])
            ok = bool(((sc.home_score >= 0) & (sc.away_score >= 0)).all())
            rows.append(_row("scores_non_negative", sport, ok, len(sc)))
            want = np.where(sc.home_score > sc.away_score, "H", np.where(sc.home_score < sc.away_score, "A", "D"))
            rows.append(_row("outcome_consistent_with_score", sport, bool((want == sc.outcome).all()),
                             int((want != sc.outcome).sum()), "mismatches"))
        o = odds[odds.event_id.isin(e.event_id)]
        if len(o):
            rows.append(_row("odds_price>1", sport, bool((o.price > 1).all()), len(o)))
            g = o.groupby(["event_id", "bookmaker", "snapshot"]).price.apply(lambda p: (1 / p).sum())
            g = g[g.index.get_level_values("bookmaker").isin(["pinnacle", "bet365"])]
            frac = float(((g > 0.98) & (g < 1.25)).mean())
            rows.append(_row("overround_in_[0.98,1.25]", sport, frac > 0.99, round(frac, 4),
                             f"median overround {g.median():.4f}"))
            pre = o[o.snapshot == "opening"].merge(e[["event_id", "forecast_time"]], on="event_id")
            if len(pre):
                ok = bool((pd.to_datetime(pre.available_at) <= pd.to_datetime(pre.forecast_time)).all())
                rows.append(_row("opening_odds_known_at_forecast_time", sport, ok, len(pre)))
        per_season = e.groupby("season").size()
        rows.append(_row("events_per_season", sport, True, int(per_season.min()),
                         "; ".join(f"{k}:{v}" for k, v in per_season.items())))
    rows.extend(crosscheck_football())
    return pd.DataFrame(rows)


def crosscheck_football() -> list[dict]:
    """Compare the primary football mirror against an independent mirror (2023-24)."""
    p = RAW_DIR / REGISTRY["football_epl_2324"].raw_relpath
    c = RAW_DIR / REGISTRY["football_epl_2324_crosscheck"].raw_relpath
    if not (p.exists() and c.exists()):
        return [_row("crosscheck_epl_2324", "football", False, None, "files missing")]
    a = pd.read_csv(p, encoding="latin-1")
    b = pd.read_csv(c)
    a["d"] = pd.to_datetime(a.Date, dayfirst=True).dt.date
    b["d"] = pd.to_datetime(b.Date).dt.date
    m = a.merge(b, on=["d", "HomeTeam", "AwayTeam"], suffixes=("", "_x"))
    same = (m.FTHG == m.FTHG_x) & (m.FTAG == m.FTAG_x) & (m.HST == m.HST_x)
    ok = len(m) == len(a) == len(b) and bool(same.all())
    return [_row("crosscheck_epl_2324_vs_independent_mirror", "football", ok, f"{int(same.sum())}/{len(a)}",
                 "FT goals + shots on target identical")]
