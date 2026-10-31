"""Historical strategy research with a VIRTUAL bankroll.

No real money, no bookmaker connectivity, no stake advice. The simulator asks:
"if positions had been opened (virtually) whenever a model's probability
disagreed with the historical price by more than a threshold, what would the
virtual P&L distribution have been?" — a falsification test of model edge.

Timeline (event-driven, no look-ahead):
  open   at forecast_time  — stake from SETTLED virtual bankroll known then,
                              capped by free (unreserved) virtual cash
  settle at result_available_at
Drawdown stop: once settled equity falls more than `max_drawdown_stop` below its
running peak, no further positions are opened.
"""
from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sportslab.ev import metrics as M
from sportslab.ev.odds import devig_frame


@dataclass
class StrategyConfig:
    name: str = "fixed-1u"
    staking: str = "fixed"              # fixed | fraction
    stake: float = 1.0                  # virtual units per position (fixed)
    fraction: float = 0.01              # share of settled virtual bankroll (fraction)
    initial_bankroll: float = 100.0
    min_edge: float = 0.03              # required EV on the (haircut) probability
    uncertainty_k: float = 0.0          # p_adj = p - k * sigma  (model-uncertainty haircut)
    max_drawdown_stop: float | None = None
    commission: float = 0.0             # share of net winnings (exchange-style)
    min_odds: float = 1.01
    max_odds: float = 8.0


def select_positions(df: pd.DataFrame, labels, cfg: StrategyConfig) -> pd.DataFrame:
    """At most one virtual position per event: the outcome with the largest adjusted EV."""
    P = df[[f"p_{l}" for l in labels]].to_numpy(float)
    O = df[[f"price_{l}" for l in labels]].to_numpy(float)
    sigma = df["sigma"].to_numpy(float) if "sigma" in df else np.zeros(len(df))
    sigma = np.nan_to_num(sigma)
    padj = P - cfg.uncertainty_k * sigma[:, None]
    ev = padj * O - 1
    valid = np.isfinite(ev) & (O >= cfg.min_odds) & (O <= cfg.max_odds)
    ev = np.where(valid, ev, -np.inf)
    k = ev.argmax(axis=1)
    best = ev[np.arange(len(df)), k]
    keep = best >= cfg.min_edge
    sel = df.loc[keep].copy()
    kk = k[keep]
    sel["selection"] = [labels[i] for i in kk]
    sel["model_prob"] = P[keep, kk]
    sel["adj_prob"] = padj[keep, kk]
    sel["price_taken"] = O[keep, kk]
    sel["ev_adj"] = best[keep]
    if all(f"close_{l}" in df for l in labels):
        C = df[[f"close_{l}" for l in labels]].to_numpy(float)[keep]
        fair = devig_frame(C)
        sel["closing_price"] = C[np.arange(len(kk)), kk]
        sel["closing_fair_prob"] = fair[np.arange(len(kk)), kk]
    sel["won"] = (sel.outcome == sel.selection).astype(int)
    return sel


def simulate(positions: pd.DataFrame, cfg: StrategyConfig) -> tuple[pd.DataFrame, dict]:
    """Event-driven virtual bankroll simulation over selected positions."""
    pos = positions.sort_values(["forecast_time", "event_time", "event_id"]).reset_index(drop=True)
    settled = cfg.initial_bankroll
    reserved = 0.0
    peak = settled
    halted = False
    pending: list = []   # (settle_time_ns, idx)
    rows = []
    stakes = np.zeros(len(pos))

    def settle_until(t_ns):
        nonlocal settled, reserved, peak, halted
        while pending and pending[0][0] <= t_ns:
            _, i = heapq.heappop(pending)
            r = pos.iloc[i]
            st = stakes[i]
            gross = st * (r.price_taken - 1) if r.won else -st
            comm = cfg.commission * gross if gross > 0 else 0.0
            pnl = gross - comm
            settled += pnl
            reserved -= st
            peak = max(peak, settled)
            dd = 1 - settled / peak if peak > 0 else 1.0
            if cfg.max_drawdown_stop is not None and dd > cfg.max_drawdown_stop:
                halted = True
            rows.append(dict(i=i, settle_time=r.result_available_at, stake=st, pnl=pnl, commission=comm,
                             bankroll_after=settled, drawdown=dd))

    for i, r in pos.iterrows():
        settle_until(r.forecast_time.value)
        if halted:
            continue
        free = settled - reserved
        st = cfg.stake if cfg.staking == "fixed" else cfg.fraction * settled
        st = min(st, max(free, 0.0))
        if st <= 1e-9:
            continue
        stakes[i] = st
        reserved += st
        heapq.heappush(pending, (r.result_available_at.value, i))
    settle_until(np.iinfo(np.int64).max)
    led = pd.DataFrame(rows).sort_values("settle_time").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["i", "settle_time", "stake", "pnl", "commission", "bankroll_after", "drawdown"])
    led = led.merge(pos.reset_index(names="i"), on="i", how="left")
    return led, summarize(led, cfg)


def summarize(led: pd.DataFrame, cfg: StrategyConfig, n_boot: int = 2000, seed: int = 0) -> dict:
    n = len(led)
    out = dict(strategy=cfg.name, **{f"cfg_{k}": v for k, v in asdict(cfg).items() if k != "name"},
               n_positions=n)
    if n == 0:
        return {**out, "virtual_pnl": 0.0, "virtual_roi": np.nan}
    staked, pnl = led.stake.sum(), led.pnl.sum()
    eq = np.concatenate([[cfg.initial_bankroll], led.bankroll_after.to_numpy()])
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max(1 - eq / peak))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    st, pl = led.stake.to_numpy(), led.pnl.to_numpy()
    roi_b = pl[idx].sum(1) / st[idx].sum(1)
    pnl_b = pl[idx].sum(1)
    P = np.column_stack([led.model_prob, 1 - led.model_prob])
    Y = np.column_stack([led.won, 1 - led.won]).astype(float)
    out.update(
        virtual_staked=float(staked), virtual_pnl=float(pnl), virtual_roi=float(pnl / staked),
        virtual_roi_ci_lo=float(np.quantile(roi_b, 0.025)), virtual_roi_ci_hi=float(np.quantile(roi_b, 0.975)),
        virtual_pnl_ci_lo=float(np.quantile(pnl_b, 0.025)), virtual_pnl_ci_hi=float(np.quantile(pnl_b, 0.975)),
        p_roi_positive=float((roi_b > 0).mean()),
        final_virtual_bankroll=float(eq[-1]), max_drawdown=mdd,
        hit_rate=float(led.won.mean()), mean_price=float(led.price_taken.mean()),
        expected_hit_rate=float(led.model_prob.mean()),
        brier_selected=float(M.brier(P, Y).mean() / 2), log_loss_selected=float(M.log_loss(P, Y).mean()),
        total_commission=float(led.commission.sum()),
        halted_by_drawdown_stop=bool(cfg.max_drawdown_stop is not None and len(led) and
                                     (led.drawdown > cfg.max_drawdown_stop).any()))
    if "closing_fair_prob" in led and led.closing_fair_prob.notna().any():
        c = led.dropna(subset=["closing_fair_prob"])
        clv = c.price_taken * c.closing_fair_prob - 1     # EV of taken price under closing fair prob
        out.update(clv_mean=float(clv.mean()), clv_beat_close_share=float((c.price_taken > c.closing_price).mean()),
                   clv_ci_lo=float(np.quantile(clv.to_numpy()[rng.integers(0, len(c), (n_boot, len(c)))].mean(1), 0.025)),
                   clv_ci_hi=float(np.quantile(clv.to_numpy()[rng.integers(0, len(c), (n_boot, len(c)))].mean(1), 0.975)))
    return out
