"""Dashboard tab: choose an upcoming (future) match and get a prematch forecast."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from sportslab import upcoming as U
from sportslab.config import DB_PATH
from sportslab.ev.odds import explain

STATUS_COLOR = {U.OK: "green", U.STALE: "orange", U.INSUFFICIENT: "red"}


def _db_version() -> float:
    try:
        return DB_PATH.stat().st_mtime
    except OSError:
        return 0.0


def _refresh_block(sport: str):
    now = U.now_utc()
    if sport == "football":
        if st.button("Refresh current-season results (football-data.co.uk)", key="refresh_fb"):
            with st.spinner("Downloading current season and rebuilding the database…"):
                s = U.refresh_football_current(now)
                st.write(s.__dict__)
                if s.status == U.AVAILABLE:
                    q = U.rebuild_database()
                    st.success(f"Database rebuilt: {(q.status == 'FAIL').sum()} FAIL checks")
                    st.cache_data.clear()
                    st.session_state.pop(f"fc_{sport}", None)
    elif sport in ("hockey", "basketball"):
        cur = U.season_label(sport, now)
        default_since = U.season_start(sport, U.previous_season(sport, cur)).date()
        c1, c2 = st.columns([1, 2])
        since = c1.date_input("Load results since", value=default_since, key=f"since_{sport}")
        c2.caption("One read-only ESPN scoreboard call per day via sports-skills (≈1–3 min per season). "
                   "Needed because the historical files end in 2020 (NHL) / 2022 (NBA).")
        if c2.button("Load recent results via ESPN", key=f"refresh_{sport}"):
            bar = st.progress(0.0, text="starting…")
            s = U.refresh_espn_results(sport, since.strftime("%Y-%m-%d"),
                                       progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
            st.write(s.__dict__)
            if s.status == U.AVAILABLE:
                with st.spinner("Rebuilding the database…"):
                    q = U.rebuild_database()
                st.success(f"Database rebuilt: {(q.status == 'FAIL').sum()} FAIL checks")
                st.cache_data.clear()
                st.session_state.pop(f"fc_{sport}", None)
    else:
        st.caption("Tennis uses current ATP rankings (fetched at forecast time) with a rank model fitted on 2019-2024.")


def _freshness_banner(sport: str):
    if sport == "tennis":
        return
    from sportslab.data.pit import load_matches
    status, reasons, as_of = U.freshness(sport, load_matches(sport), U.now_utc())
    st.markdown(f"**Data freshness:** :{STATUS_COLOR[status]}[{status}] — results up to "
                f"{(as_of or '?')[:10]}" + (f"  \n{reasons[0]}" if reasons else ""))


def _manual_fixture(sport: str) -> U.Fixture | None:
    with st.expander("…or enter a future match manually"):
        if sport == "tennis":
            c1, c2 = st.columns(2)
            h = c1.text_input("Player 1 (as in ATP rankings, e.g. 'Jannik Sinner')", key="man_h")
            a = c2.text_input("Player 2", key="man_a")
        else:
            known = sorted(U._known_names(sport))
            if not known:
                st.info("No team history available.")
                return None
            c1, c2 = st.columns(2)
            h = c1.selectbox("Home", known, key=f"man_h_{sport}")
            a = c2.selectbox("Away", known, index=min(1, len(known) - 1), key=f"man_a_{sport}")
        c3, c4 = st.columns(2)
        d = c3.date_input("Date", value=date.today() + timedelta(days=1), key=f"man_d_{sport}")
        t = c4.time_input("Kick-off (UTC)", value=time(15, 0), key=f"man_t_{sport}")
        if st.button("Add this match", key=f"man_add_{sport}") and h and a and h != a:
            ko = pd.Timestamp(datetime.combine(d, t), tz="UTC")
            return U.Fixture(sport, h, a, ko, "manual entry", raw_home=h, raw_away=a, name_match="manual")
    return None


def render(sport: str):
    st.subheader("Upcoming matches — choose a future match")
    st.caption("Forecasts are MODEL estimates computed now from results published before now. "
               "Lineups and injuries are not available to the models.")
    _freshness_banner(sport)
    with st.expander("Update data before forecasting", expanded=False):
        _refresh_block(sport)
    c1, c2 = st.columns([2, 1])
    days = c1.slider("Days ahead", 1, 14, 7, key=f"days_{sport}")
    if c2.button("Load upcoming fixtures", key=f"load_{sport}", type="primary"):
        with st.spinner("Fetching fixtures (read-only)…"):
            fx, statuses = U.upcoming_fixtures(sport, days)
        st.session_state[f"fx_{sport}"] = fx
        st.session_state[f"fxstat_{sport}"] = statuses
        st.session_state.pop(f"fc_{sport}", None)
    man = _manual_fixture(sport)
    if man is not None:
        st.session_state[f"fx_{sport}"] = [man] + [f for f in st.session_state.get(f"fx_{sport}", []) if f.key != man.key]
        st.session_state.pop(f"fc_{sport}", None)
    statuses = st.session_state.get(f"fxstat_{sport}")
    if statuses:
        st.dataframe(pd.DataFrame([s.__dict__ for s in statuses]), use_container_width=True, hide_index=True)
    fixtures: list[U.Fixture] = st.session_state.get(f"fx_{sport}", [])
    if not fixtures:
        st.info("Press **Load upcoming fixtures** (or add a match manually).")
        return
    cache = st.session_state.get(f"fc_{sport}")
    if cache is None or cache["db"] != _db_version() or set(cache["keys"]) != {f.key for f in fixtures}:
        with st.spinner(f"Computing forecasts for {len(fixtures)} match(es)… (walk-forward over all history)"):
            res = U.forecast_fixtures(sport, fixtures)
        cache = {"db": _db_version(), "keys": [f.key for f in fixtures], "res": {r.fixture["key"]: r for r in res}}
        st.session_state[f"fc_{sport}"] = cache
    res = cache["res"]
    rows = []
    for f in fixtures:
        r = res.get(f.key)
        row = {"kickoff": f.label, "status": r.status if r else "PAST/SKIPPED", "source": f.source}
        if r and r.ensemble:
            row.update({f"P({k})": round(v, 3) for k, v in r.ensemble.items()})
        if r and r.market:
            row.update({f"market {k}": round(v, 3) for k, v in r.market["devig"].items()})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    labels = {f.key: f.label for f in fixtures if f.key in res}
    if not labels:
        return
    key = st.selectbox("Match", list(labels), format_func=labels.get, key=f"pick_{sport}")
    r = res[key]
    st.markdown(f"### {labels[key]} — :{STATUS_COLOR[r.status]}[{r.status}]")
    for reason in r.reasons:
        st.markdown(f"- {reason}")
    if not r.probs:
        return
    long = [dict(model=m, outcome=k, probability=v) for m, p in r.probs.items() for k, v in p.items()]
    if r.ensemble:
        long += [dict(model="ENSEMBLE (+temp)", outcome=k, probability=v) for k, v in r.ensemble.items()]
    if r.market:
        long += [dict(model=f"market {r.market['book']} (de-vigged)", outcome=k, probability=v)
                 for k, v in r.market["devig"].items()]
    st.plotly_chart(px.bar(pd.DataFrame(long), x="probability", y="model", color="outcome", orientation="h",
                           barmode="stack", height=60 * (len(r.probs) + 3), range_x=[0, 1]),
                    use_container_width=True)
    c1, c2 = st.columns(2)
    c1.markdown("**Teams / players (OBSERVED history)**")
    c1.json(r.teams, expanded=False)
    c2.markdown(f"**Data as of:** {r.data_as_of}  \n**Model disagreement (spread):** "
                f"{'—' if r.spread is None else f'{r.spread:.3f}'}  \n**Computed at:** {r.computed_at}")
    if r.market and r.ensemble:
        st.markdown("**Market maths (education only — not a recommendation)**")
        st.code(explain(list(r.market["odds"].values()), list(r.ensemble.values()), list(r.labels)).text)
    with st.expander("Data sources"):
        st.dataframe(pd.DataFrame(r.sources), use_container_width=True, hide_index=True)
    if st.button("Save this forecast to the forecast log", key=f"log_{key}"):
        fid = U.log_upcoming(r)
        st.success(f"Logged: {fid}. It will be scored automatically once the result is in the database."
                   if fid else "Not logged (insufficient data).")
