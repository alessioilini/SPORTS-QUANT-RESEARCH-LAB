"""No look-ahead bias: models only ever see results that were public at decision time."""
import numpy as np
import pandas as pd
import pytest

from sportslab.data.pit import LookAheadError, assert_no_lookahead, column_class, fixture_view, history_view
from sportslab.models.base import Model, walk_forward
from sportslab.models.goals import DixonColes, GammaPoisson
from sportslab.models.ml import GBMModel, LogisticModel
from sportslab.models.ratings import Elo, GaussianRating

L3 = ("H", "D", "A")


class Spy(Model):
    """Records what it was shown and when."""
    name = "spy"
    needs_refit = True
    refit_every = pd.Timedelta(days=1)

    def __init__(self, labels):
        super().__init__(labels)
        self.seen_at = []        # (t_last_predict, max available_at observed)
        self.fixture_cols = set()
        self.max_avail = pd.Timestamp.min.tz_localize("UTC")
        self.refits = []

    def observe(self, rows):
        self.max_avail = max(self.max_avail, rows.result_available_at.max())
        self.observed_ids = getattr(self, "observed_ids", set()) | set(rows.event_id)

    def refit(self, history, as_of):
        self.refits.append((as_of, history.result_available_at.max()))

    def predict(self, fixtures):
        t = fixtures.forecast_time.iloc[0]
        self.seen_at.append((t, self.max_avail))
        self.fixture_cols |= set(fixtures.columns)
        assert not (set(fixtures.event_id) & getattr(self, "observed_ids", set())), "fixture result already observed"
        return np.full((len(fixtures), len(self.labels)), 1 / 3)


def test_column_classes():
    assert column_class("odds_pinnacle_opening_H") == "pre"
    assert column_class("odds_pinnacle_closing_H") == "close"
    assert column_class("home_score") == "post" and column_class("outcome") == "post"
    assert column_class("home_shots_on_target") == "post"
    assert column_class("pre_home_rank") == "pre" and column_class("forecast_time") == "id"


def test_fixture_view_strips_results_and_closing(league):
    df, _ = league
    fx = fixture_view(df)
    for c in ("home_score", "away_score", "outcome", "result_available_at", "odds_pinnacle_closing_H", "true_lam"):
        assert c not in fx.columns
    assert "odds_pinnacle_opening_H" in fx.columns


def test_walk_forward_feeds_only_past_results(league):
    df, _ = league
    spy = Spy(L3)
    walk_forward(df, spy, refit_from=df.forecast_time.min())
    assert spy.seen_at, "model never predicted"
    for t, max_avail in spy.seen_at:
        assert max_avail <= t
    assert spy.refits
    for as_of, max_hist in spy.refits:
        assert pd.isna(max_hist) or max_hist <= as_of   # NaT = no results public yet
    assert not ({"outcome", "home_score", "result_available_at"} & spy.fixture_cols)


def test_history_view_and_guard(league):
    df, _ = league
    t = df.forecast_time.iloc[len(df) // 2]
    h = history_view(df, t)
    assert (h.result_available_at <= t).all()
    with pytest.raises(LookAheadError):
        assert_no_lookahead(df, t)


@pytest.mark.parametrize("factory", [
    lambda: Elo(L3, k=20, home_adv=60, mov=True),
    lambda: DixonColes(L3),
    lambda: GammaPoisson(L3, n_samples=0),
    lambda: LogisticModel(L3, "football"),
    lambda: GBMModel(L3, "football"),
], ids=["elo", "dixon-coles", "gamma-poisson", "logistic", "gbm"])
def test_poisoning_future_does_not_change_past_predictions(league, factory):
    """Scramble every result after a cut-off: predictions made before the cut-off must be identical."""
    df, _ = league
    cut = df.forecast_time.quantile(0.6)
    poisoned = df.copy()
    fut = poisoned.result_available_at > cut
    rng = np.random.default_rng(99)
    poisoned.loc[fut, "home_score"] = rng.integers(0, 6, fut.sum()).astype(float)
    poisoned.loc[fut, "away_score"] = rng.integers(0, 6, fut.sum()).astype(float)
    poisoned.loc[fut, "outcome"] = np.where(poisoned.loc[fut, "home_score"] > poisoned.loc[fut, "away_score"], "H",
                                            np.where(poisoned.loc[fut, "home_score"] < poisoned.loc[fut, "away_score"],
                                                     "A", "D"))
    m_kw = dict(refit_from=df.forecast_time.min())

    def make():
        m = factory()
        if hasattr(m, "min_train"):
            m.min_train = 60          # instance-level: small synthetic league
        return m
    a = walk_forward(df, make(), **m_kw).preds
    b = walk_forward(poisoned, make(), **m_kw).preds
    early = a.forecast_time <= cut
    assert early.sum() > 50
    np.testing.assert_allclose(a[early].iloc[:, 2:].to_numpy(), b[early].iloc[:, 2:].to_numpy(), equal_nan=True)
    # sanity: the poisoning is visible later (otherwise the test would be vacuous)
    late = ~early
    assert not np.allclose(np.nan_to_num(a[late].iloc[:, 2:].to_numpy()), np.nan_to_num(b[late].iloc[:, 2:].to_numpy()))


def test_gaussian_rating_binary_no_lookahead():
    from tests.conftest import synthetic_league
    df, _ = synthetic_league(sport="hockey", seed=5)
    cut = df.forecast_time.quantile(0.5)
    p2 = df.copy()
    fut = p2.result_available_at > cut
    p2.loc[fut, "outcome"] = np.where(p2.loc[fut, "outcome"] == "H", "A", "H")
    a = walk_forward(df, GaussianRating(("H", "A"))).preds
    b = walk_forward(p2, GaussianRating(("H", "A"))).preds
    early = a.forecast_time <= cut
    np.testing.assert_allclose(a[early].p_H, b[early].p_H)


def test_pit_guard_drops_opening_odds_published_after_forecast(tmp_db):
    """An 'opening' price stamped after forecast_time must not reach the fixture view."""
    from sportslab.data import pit
    from sportslab.data.store import replace_facts
    ev = pd.DataFrame([dict(event_id="E1", sport="football", competition="EPL", season="X",
                            event_time="2024-01-02T15:00:00Z", event_time_precision="minute",
                            forecast_time="2024-01-01T15:00:00Z", home="A", away="B", neutral=0)])
    res = pd.DataFrame([dict(event_id="E1", home_score=1, away_score=0, outcome="H",
                             available_at="2024-01-02T17:30:00Z")])
    odds = pd.DataFrame([dict(event_id="E1", bookmaker="pinnacle", market="1X2", selection=s, price=2.5,
                              snapshot="opening", observed_at=ts, observed_at_precision="exact", available_at=ts)
                         for s, ts in (("H", "2024-01-01T10:00:00Z"), ("D", "2024-01-01T20:00:00Z"),
                                       ("A", "2024-01-01T10:00:00Z"))])
    replace_facts({"events": ev, "results": res, "odds": odds}, path=tmp_db)
    pit._load_cached.cache_clear()
    df = pit.load_matches("football", db=str(tmp_db))
    assert df.odds_pinnacle_opening_H.iloc[0] == 2.5
    assert "odds_pinnacle_opening_D" not in df or pd.isna(df.odds_pinnacle_opening_D.iloc[0])
