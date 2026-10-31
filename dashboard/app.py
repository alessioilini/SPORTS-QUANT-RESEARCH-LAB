"""Sports Quant Research Lab — local research dashboard (Streamlit).

Run:  streamlit run dashboard/app.py
Everything shown is historical research output; bankroll figures are VIRTUAL.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportslab import DISCLAIMER  # noqa: E402
from sportslab.config import ARTIFACTS_DIR, DB_PATH, ROOT as LAB_ROOT  # noqa: E402
from sportslab.ev.odds import explain  # noqa: E402

st.set_page_config(page_title="Sports Quant Research Lab", page_icon="📊", layout="wide")

SECTIONS = ["1. Prematch Analytics", "2. Live Match Analytics", "3. Probability Models", "4. Forecast History",
            "5. Backtesting", "6. Virtual Bankroll", "7. Model Performance", "8. Data Quality"]
SPORTS = ["football", "hockey", "basketball", "tennis"]
LABELS = {"football": ["H", "D", "A"], "hockey": ["H", "A"], "basketball": ["H", "A"], "tennis": ["H", "A"]}


@st.cache_data(show_spinner=False)
def art(name: str) -> pd.DataFrame | None:
    p = ARTIFACTS_DIR / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def art_json(name: str) -> dict | None:
    p = ARTIFACTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def db(sql: str, params: tuple = ()) -> pd.DataFrame:
    import sqlite3
    if not Path(DB_PATH).exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(sql, con, params=params)


def need(df, what="artifacts"):
    if df is None or (hasattr(df, "empty") and df.empty):
        st.warning(f"No {what} yet. Run: `python scripts/fetch_data.py && python scripts/build_db.py && "
                   "python scripts/run_validation.py && python scripts/run_backtest.py`")
        st.stop()


def pcols(sport):
    return [f"p_{l}" for l in LABELS[sport]]


# ----------------------------------------------------------------------------- layout
st.sidebar.title("📊 Sports Quant Research Lab")
section = st.sidebar.radio("Section", SECTIONS, key="section")
sport = st.sidebar.selectbox("Sport", SPORTS, key="sport")
st.sidebar.caption(DISCLAIMER)
st.caption("⚠️ " + DISCLAIMER)

# ============================================================================= 1. PREMATCH
if section == SECTIONS[0]:
    st.header("Prematch Analytics")
    preds = art("predictions")
    need(preds, "predictions")
    p = preds[(preds.sport == sport) & (preds.split == "test")]
    ex = art_json("example_forecast.json")
    if ex and sport == "football":
        with st.expander("Worked example forecast with data sources (scripts/example_forecast.py)", expanded=True):
            st.json(ex, expanded=False)
    teams = sorted(set(p.home) | set(p.away))
    c1, c2 = st.columns(2)
    team = c1.selectbox("Team / player", teams, key="pm_team")
    ev = p[(p.home == team) | (p.away == team)].drop_duplicates("event_id").sort_values("event_time")
    labels_ev = {r.event_id: f"{r.event_time:%Y-%m-%d} {r.home} vs {r.away}" for r in ev.itertuples()}
    eid = c2.selectbox("Out-of-sample match", list(labels_ev), format_func=labels_ev.get, key="pm_event")
    rows = p[p.event_id == eid]
    show = rows[["model"] + pcols(sport)].dropna().sort_values("model")
    st.subheader(f"{labels_ev[eid]} — outcome: {rows.outcome.iloc[0]}")
    long = show.melt(id_vars="model", var_name="outcome", value_name="probability")
    fig = px.bar(long, x="probability", y="model", color="outcome", orientation="h", barmode="stack", height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(show.style.format({c: "{:.3f}" for c in pcols(sport)}), use_container_width=True)
    # odds & EV explanation (educational)
    odds = db("SELECT bookmaker, snapshot, selection, price, observed_at, observed_at_precision, source_id "
              "FROM odds WHERE event_id = ?", (eid,))
    if not odds.empty:
        st.markdown("**Historical market prices (OBSERVED)**")
        st.dataframe(odds, use_container_width=True)
        pin = odds[(odds.bookmaker == "pinnacle") & (odds.snapshot.isin(["opening", "single"]))]
        pin = pin.set_index("selection").reindex(LABELS[sport])
        model_name = st.selectbox("Model for the EV explanation", show.model.tolist(),
                                  index=max(0, show.model.tolist().index("ensemble:mean+temp"))
                                  if "ensemble:mean+temp" in show.model.tolist() else 0)
        mp = show.set_index("model").loc[model_name, pcols(sport)].to_numpy(float)
        if pin.price.notna().all():
            st.markdown("**Market maths explained (education only — not a recommendation)**")
            st.code(explain(pin.price.to_numpy(), mp, LABELS[sport]).text)
    st.markdown("**Provenance**")
    st.dataframe(db("SELECT e.event_id, e.event_time, e.event_time_precision, e.forecast_time, e.forecast_time_rule, "
                    "e.source_id, e.retrieved_at, r.available_at AS result_available_at FROM events e "
                    "LEFT JOIN results r USING(event_id) WHERE e.event_id = ?", (eid,)), use_container_width=True)

# ============================================================================= 2. LIVE
elif section == SECTIONS[1]:
    from sportslab.live.engine import LiveEngine, LiveSnapshot, PrematchPrior
    from sportslab.live.sources import InferSports, SportsSkillsLive
    st.header("Live Match Analytics")
    summ = art_json("validation_summary.json") or {}
    live_cfg = summ.get("live_ht", {})
    eng = LiveEngine(football={k: live_cfg[k] for k in ("second_half_share", "prior_weight") if k in live_cfg},
                     basketball=summ.get("basketball_live_sigma", {}) and
                     {"sigma_full": summ["basketball_live_sigma"]["sigma_full"]})
    with st.expander("Live data sources status", expanded=False):
        if st.button("Probe live sources now (read-only)"):
            for src in (SportsSkillsLive(sport), InferSports()):
                r = src.fetch()
                st.write({"source": r.source, "status": r.status, "events": r.raw_count, "error": r.error,
                          "fetched_at": r.fetched_at.isoformat()})
        st.caption("In this environment ESPN/NBA CDN and InferSports are blocked by the network policy: the engine "
                   "below is driven by manual / replayed snapshots.")
    now = datetime.now(timezone.utc)
    c = st.columns(3)
    delay = c[0].slider("Feed delay (seconds since source update)", 0, 900, 10)
    unknown_ts = c[1].checkbox("Source timestamp unknown", False)
    st.subheader("Snapshot")
    if sport == "football":
        a = st.columns(6)
        lam = a[0].number_input("Prematch λ home", 0.1, 5.0, 1.55)
        mu = a[1].number_input("Prematch μ away", 0.1, 5.0, 1.15)
        minute = a[2].slider("Minute", 0, 95, 55)
        hs = a[3].number_input("Home goals", 0, 10, 1)
        as_ = a[4].number_input("Away goals", 0, 10, 0)
        reds = a[5].selectbox("Red cards (home, away)", ["0,0", "1,0", "0,1", "unknown"])
        rh, ra = (None, None) if reds == "unknown" else map(int, reds.split(","))
        snap = LiveSnapshot("football", "manual", "manual", now, None if unknown_ts else now - timedelta(seconds=delay),
                            home_score=hs, away_score=as_, minute=minute, red_cards_home=rh, red_cards_away=ra)
        prior = PrematchPrior("user", now, lam=lam, mu=mu)
    elif sport == "hockey":
        a = st.columns(7)
        lam, mu = a[0].number_input("λ home (reg.)", 0.5, 6.0, 3.0), a[1].number_input("μ away", 0.5, 6.0, 2.8)
        period = a[2].selectbox("Period", [1, 2, 3, 4], index=2)
        clock = a[3].slider("Seconds left in period", 0, 1200, 600)
        hs, as_ = a[4].number_input("Home", 0, 15, 2), a[5].number_input("Away", 0, 15, 2)
        man = a[6].selectbox("Manpower", ["5v5", "home PP 5v4", "away PP 4v5", "unknown"])
        sk = {"5v5": (5, 5), "home PP 5v4": (5, 4), "away PP 4v5": (4, 5), "unknown": (None, None)}[man]
        snap = LiveSnapshot("hockey", "manual", "manual", now, None if unknown_ts else now - timedelta(seconds=delay),
                            home_score=hs, away_score=as_, period=period, clock_remaining_s=clock,
                            home_skaters=sk[0], away_skaters=sk[1], pp_remaining_s=120 if sk[0] != sk[1] else None)
        prior = PrematchPrior("user", now, lam=lam, mu=mu)
    elif sport == "basketball":
        a = st.columns(5)
        ph = a[0].slider("Prematch P(home)", 0.05, 0.95, 0.6)
        period = a[1].selectbox("Quarter", [1, 2, 3, 4, 5], index=2)
        clock = a[2].slider("Seconds left in quarter", 0, 720, 300)
        hs, as_ = a[3].number_input("Home pts", 0, 200, 70), a[4].number_input("Away pts", 0, 200, 66)
        snap = LiveSnapshot("basketball", "manual", "manual", now, None if unknown_ts else now - timedelta(seconds=delay),
                            home_score=hs, away_score=as_, period=period, clock_remaining_s=clock)
        prior = PrematchPrior("user", now, p_home=ph)
    else:
        a = st.columns(8)
        ph = a[0].slider("Prematch P(player1)", 0.05, 0.95, 0.65)
        bo = a[1].selectbox("Best of", [3, 5])
        sh, sa = a[2].number_input("Sets P1", 0, 2, 0), a[3].number_input("Sets P2", 0, 2, 1)
        gh, ga = a[4].number_input("Games P1", 0, 7, 3), a[5].number_input("Games P2", 0, 7, 3)
        pts = a[6].selectbox("Points (P1-P2)", ["0-0", "1-0", "0-1", "2-2", "3-2", "2-3", "3-3"])
        srv = a[7].selectbox("Server", ["P1", "P2"])
        p1, p2 = map(int, pts.split("-"))
        snap = LiveSnapshot("tennis", "manual", "manual", now, None if unknown_ts else now - timedelta(seconds=delay),
                            sets_home=sh, sets_away=sa, games_home=gh, games_away=ga, points_home=p1, points_away=p2,
                            home_serving=srv == "P1", best_of=bo)
        prior = PrematchPrior("user", now, p_home=ph)
    f = eng.forecast(snap, prior, now)
    color = {"OK": "green", "DELAYED": "orange", "INSUFFICIENT_DATA": "red"}[f.status]
    st.markdown(f"### Status: :{color}[{f.status}]  — delay: {f.delay_seconds if f.delay_seconds is not None else 'unknown'} s")
    if f.probs:
        st.plotly_chart(px.bar(x=list(f.probs), y=list(f.probs.values()), labels={"x": "outcome", "y": "probability"},
                               range_y=[0, 1], height=300), use_container_width=True)
    st.write({"model": f.model, "reasons": f.reasons, "inputs_used": f.inputs_used})
    if sport == "football":
        st.subheader("Probability path: pre-match → half-time → full-time (real EPL test matches)")
        lv = art("live_ht_validation")
        preds = art("predictions")
        if lv is not None and preds is not None:
            lv = lv[lv.split == "test"]
            pick = st.selectbox("Match", lv.event_id.tolist()[:400])
            r = lv[lv.event_id == pick].iloc[0]
            pre = preds[(preds.model == "dixon-coles") & (preds.event_id == pick)].iloc[0]
            path = pd.DataFrame({"stage": ["pre-match", f"HT {int(r.ht_home)}-{int(r.ht_away)}", f"FT ({r.outcome})"],
                                 "H": [pre.p_H, r.p_H, float(r.outcome == "H")],
                                 "D": [pre.p_D, r.p_D, float(r.outcome == "D")],
                                 "A": [pre.p_A, r.p_A, float(r.outcome == "A")]})
            st.plotly_chart(px.line(path.melt(id_vars="stage"), x="stage", y="value", color="variable", markers=True,
                                    labels={"value": "probability"}, height=320), use_container_width=True)
            hv = live_cfg.get("by_split", {}).get("test", {})
            if hv:
                st.caption(f"Test seasons: live-HT log loss {hv['live_ht']['log_loss']:.4f} vs pre-match "
                           f"{hv['prematch']['log_loss']:.4f} (n={hv['n']}); estimated 2nd-half goal share "
                           f"{live_cfg['second_half_share']:.3f}, Bayesian prior weight {live_cfg['prior_weight']:.1f}")

# ============================================================================= 3. MODELS
elif section == SECTIONS[2]:
    st.header("Probability Models — chronological out-of-sample comparison")
    m = art("metrics")
    need(m, "metrics")
    m = m[m.sport == sport]
    split = st.radio("Split", ["test", "validation", "train"], horizontal=True)
    mm = m[m.split == split].sort_values("log_loss")
    bench = mm.model.str.startswith(("market:", "external:", "baseline:"))
    fig = go.Figure(go.Scatter(x=mm.log_loss, y=mm.model, mode="markers",
                               marker=dict(size=10, symbol=np.where(bench, "diamond-open", "circle")),
                               error_x=dict(type="data", symmetric=False, array=mm.log_loss_hi - mm.log_loss,
                                            arrayminus=mm.log_loss - mm.log_loss_lo)))
    fig.update_layout(height=34 * len(mm) + 120, xaxis_title="log loss (lower = better), 95% bootstrap CI; "
                      "◇ = baseline / benchmark", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    cols = ["model", "n", "coverage", "log_loss", "log_loss_lo", "log_loss_hi", "brier", "accuracy", "ece"] + \
           (["rps"] if "rps" in mm and mm.rps.notna().any() else []) + ["temperature"]
    st.dataframe(mm[cols], use_container_width=True)
    comp = art("comparisons")
    if comp is not None:
        st.subheader("Paired bootstrap: Δ log loss vs reference (test). Negative = model better.")
        c = comp[comp.sport == sport]
        ref = st.selectbox("Reference", sorted(c.reference.unique()))
        cc = c[c.reference == ref].sort_values("mean_diff")
        fig = go.Figure(go.Scatter(x=cc.mean_diff, y=cc.model, mode="markers",
                                   error_x=dict(type="data", symmetric=False, array=cc.hi - cc.mean_diff,
                                                arrayminus=cc.mean_diff - cc.lo)))
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(height=30 * len(cc) + 120, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cc, use_container_width=True)
    tun = art("tuning")
    if tun is not None:
        st.subheader("Hyper-parameter grids (scored on validation only)")
        st.dataframe(tun[tun.sport == sport].dropna(axis=1, how="all"), use_container_width=True)

# ============================================================================= 4. HISTORY
elif section == SECTIONS[3]:
    st.header("Forecast History")
    preds = art("predictions")
    need(preds)
    p = preds[preds.sport == sport]
    model = st.selectbox("Model", sorted(p.model.unique()),
                         index=sorted(p.model.unique()).index("ensemble:mean+temp") if "ensemble:mean+temp" in set(p.model) else 0)
    q = p[p.model == model].dropna(subset=pcols(sport)).copy()
    lab = LABELS[sport]
    P = q[pcols(sport)].to_numpy()
    Y = np.stack([(q.outcome == l).to_numpy(float) for l in lab], axis=1)
    q["log_loss"] = -np.log(np.clip((P * Y).sum(1), 1e-12, 1))
    q["p_outcome"] = (P * Y).sum(1)
    st.subheader("Rolling log loss (100 matches)")
    q = q.sort_values("event_time")
    q["rolling_ll"] = q.log_loss.rolling(100, min_periods=30).mean()
    st.plotly_chart(px.line(q, x="event_time", y="rolling_ll", color="split", height=320), use_container_width=True)
    st.subheader("Biggest misses (lowest probability on the realised outcome)")
    miss = q.nsmallest(25, "p_outcome")[["event_time", "season", "split", "home", "away", "outcome", "p_outcome"] + pcols(sport)]
    st.plotly_chart(px.histogram(q, x="p_outcome", color="split", nbins=40, barmode="overlay",
                                 title="Probability assigned to what actually happened", height=300),
                    use_container_width=True)
    st.dataframe(miss, use_container_width=True)
    st.subheader("Logged forecasts (forecast_log)")
    st.dataframe(db("SELECT created_at, mode, sport, event_id, model, status, probs_json, resolved_outcome, brier, "
                    "logloss FROM forecast_log ORDER BY created_at DESC LIMIT 200"), use_container_width=True)

# ============================================================================= 5. BACKTESTING
elif section == SECTIONS[4]:
    st.header("Backtesting — virtual historical simulation")
    st.info("Positions are VIRTUAL. Prices: football = Pinnacle opening (known at forecast time), CLV vs Pinnacle "
            "closing; tennis = Pinnacle single snapshot (time unknown, no CLV).")
    S = art("backtest_summary")
    need(S, "backtest results")
    s = S[S.sport == sport]
    if s.empty:
        st.warning("No historical prices for this sport → no backtest (see Data Quality).")
        st.stop()
    strat = st.selectbox("Strategy", sorted(s.strategy.unique()))
    v = s[s.strategy == strat]
    fig = go.Figure()
    for period, g in v.groupby("period"):
        fig.add_trace(go.Bar(name=period, x=g.model, y=g.virtual_roi,
                             error_y=dict(type="data", symmetric=False, array=g.virtual_roi_ci_hi - g.virtual_roi,
                                          arrayminus=g.virtual_roi - g.virtual_roi_ci_lo)))
    fig.update_layout(barmode="group", yaxis_tickformat=".1%", yaxis_title="virtual ROI (95% bootstrap CI)",
                      height=420)
    st.plotly_chart(fig, use_container_width=True)
    cols = ["model", "period", "n_events", "n_positions", "virtual_staked", "virtual_pnl", "virtual_roi",
            "virtual_roi_ci_lo", "virtual_roi_ci_hi", "p_roi_positive", "max_drawdown", "hit_rate",
            "expected_hit_rate", "brier_selected", "log_loss_selected", "clv_mean", "clv_beat_close_share",
            "total_commission", "final_virtual_bankroll"]
    st.dataframe(v[[c for c in cols if c in v]], use_container_width=True)
    st.subheader("In-sample vs out-of-sample gap (virtual ROI)")
    piv = v.pivot_table(index="model", columns="period", values="virtual_roi")
    if piv.shape[1] == 2:
        piv["OOS - IS"] = piv.iloc[:, 1] - piv.iloc[:, 0]
    st.dataframe(piv.style.format("{:.2%}"), use_container_width=True)
    cc = art_json("backtest_crosscheck.json")
    if cc:
        st.caption("Cross-check vs backtestbuddy (flat 1u, same selections)")
        st.dataframe(pd.DataFrame(cc["crosscheck"]), use_container_width=True)

# ============================================================================= 6. BANKROLL
elif section == SECTIONS[5]:
    st.header("Virtual Bankroll")
    L = art("backtest_ledgers")
    need(L, "ledgers")
    l = L[L.sport == sport]
    if l.empty:
        st.warning("No virtual ledger for this sport.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", sorted(l.model.unique()))
    strats = sorted(l[l.model == model].strategy.unique())          # only combinations that exist
    strat = c2.selectbox("Strategy", strats, index=strats.index("fixed-1u") if "fixed-1u" in strats else 0)
    periods = sorted(l[(l.model == model) & (l.strategy == strat)].period.unique(), reverse=True)  # OOS first
    period = c3.selectbox("Period", periods)
    g = l[(l.model == model) & (l.strategy == strat) & (l.period == period)].sort_values("settle_time")
    if g.empty:
        st.warning("No positions for this combination.")
        st.stop()
    st.plotly_chart(px.line(g, x="settle_time", y="bankroll_after", title="Virtual bankroll (units)", height=330),
                    use_container_width=True)
    fig = px.area(g, x="settle_time", y=-g.drawdown, title="Virtual drawdown from peak", height=260)
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="drawdown")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(g.drop(columns=["sport"]), use_container_width=True)
    st.subheader("Virtual journal (append-only, SQLite)")
    st.dataframe(db("SELECT * FROM virtual_journal ORDER BY entry_id DESC LIMIT 300"), use_container_width=True)

# ============================================================================= 7. PERFORMANCE
elif section == SECTIONS[6]:
    st.header("Model Performance — calibration & stability")
    R = art("reliability")
    need(R)
    r = R[R.sport == sport]
    models = sorted(r.model.unique())
    default = [x for x in ("ensemble:mean+temp", "elo", "logistic", "market:pinnacle-closing",
                           "external:fivethirtyeight", "market:pinnacle-single") if x in models]
    pick = st.multiselect("Models", models, default=default[:4])
    outcome = st.selectbox("Outcome", LABELS[sport])
    rr = r[(r.model.isin(pick)) & (r.outcome == outcome)]
    fig = px.line(rr, x="mean_pred", y="obs_freq", color="model", markers=True, height=420,
                  hover_data=["n"], title="Reliability diagram (test seasons)")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    preds = art("predictions")
    p = preds[(preds.sport == sport) & (preds.model.isin(pick))].dropna(subset=pcols(sport))
    P = p[pcols(sport)].to_numpy()
    Y = np.stack([(p.outcome == l).to_numpy(float) for l in LABELS[sport]], axis=1)
    p = p.assign(brier=((P - Y) ** 2).sum(1), log_loss=-np.log(np.clip((P * Y).sum(1), 1e-12, 1)))
    by = p.groupby(["model", "season"], as_index=False)[["brier", "log_loss"]].mean()
    st.plotly_chart(px.line(by, x="season", y="log_loss", color="model", markers=True,
                            title="Log loss by season (train → validation → test)", height=360), use_container_width=True)
    summ = art_json("validation_summary.json") or {}
    if sport == "football":
        st.subheader("Monte Carlo checks")
        st.json(summ.get("montecarlo", {}))
        sim = art("season_simulation")
        if sim is not None:
            st.subheader(f"Season simulation EPL 2025-26 from {sim.as_of.iloc[0]} (n={sim.n_sims.iloc[0]}, seed={sim.seed.iloc[0]})")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sim.team, y=sim.exp_points, mode="markers", name="expected",
                                     error_y=dict(type="data", symmetric=False, array=sim.p95_points - sim.exp_points,
                                                  arrayminus=sim.exp_points - sim.p05_points)))
            fig.add_trace(go.Scatter(x=sim.team, y=sim.actual_points, mode="markers", name="actual",
                                     marker=dict(symbol="x", size=10)))
            fig.update_layout(height=420, yaxis_title="final points")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sim, use_container_width=True)

# ============================================================================= 8. DATA QUALITY
elif section == SECTIONS[7]:
    st.header("Data Quality & provenance")
    q = art("data_quality")
    if q is not None:
        st.subheader("Automated checks")
        st.dataframe(q, use_container_width=True)
        st.metric("checks passing", f"{(q.status == 'PASS').sum()}/{len(q)}")
    st.subheader("Coverage: OBSERVED vs UNKNOWN fields")
    cov = db("SELECT sport, field, status, note, COUNT(*) AS n_sources FROM coverage GROUP BY sport, field, status, note")
    st.dataframe(cov[cov.sport.isin([sport, "*"])] if not cov.empty else cov, use_container_width=True)
    st.subheader("Registered sources (pinned URLs, SHA-256)")
    st.dataframe(db("SELECT source_id, sport, role, original_provider, host_repo, season, status, retrieved_at, sha256, "
                    "bytes, url FROM sources ORDER BY sport, source_id"), use_container_width=True)
    st.subheader("Table sizes")
    st.dataframe(pd.DataFrame([dict(table=t, rows=int(db(f"SELECT COUNT(*) AS n FROM {t}").n.iloc[0]))
                               for t in ("events", "results", "team_match_stats", "player_prematch", "odds",
                                         "external_forecasts", "lineups", "injuries", "virtual_journal", "forecast_log")]))
    st.subheader("Unavailable sources in this environment")
    st.markdown("- ESPN / NBA CDN / NHL api-web live feeds: blocked by egress proxy (HTTP 403)\n"
                "- InferSports MCP: not connected (endpoint unverified; host blocked)\n"
                "- football-data.co.uk / tennis-data.co.uk direct: blocked — pinned GitHub mirrors used\n"
                "- JeffSackmann tennis_atp: repository unavailable (403/404)\n"
                "- PuckAPI: paid, intentionally not connected")
    lock = json.loads((LAB_ROOT / "skills.lock.json").read_text()) if (LAB_ROOT / "skills.lock.json").exists() else None
    if lock:
        st.subheader("Installed Claude Code skills (skills.lock.json)")
        st.dataframe(pd.DataFrame(lock["installed"]), use_container_width=True)
