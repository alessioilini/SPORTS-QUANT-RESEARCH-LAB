"""Supervised models on point-in-time features: logistic regression & gradient boosting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sportslab.config import GLOBAL_SEED
from sportslab.models.base import Model
from sportslab.models.features import make_engine


class FeatureModel(Model):
    needs_refit = True
    refit_every = pd.Timedelta(days=28)
    min_train = 300

    def __init__(self, labels, sport: str, feature_subset: list[str] | None = None, name=None):
        super().__init__(labels)
        self.sport = sport
        self.engine = make_engine(sport, labels)
        self.store: dict[str, pd.Series] = {}
        self.subset = feature_subset
        self.est = None
        self.name = name or self.name
        self.n_train = 0

    def make_estimator(self):
        raise NotImplementedError

    def observe(self, rows: pd.DataFrame) -> None:
        self.engine.observe(rows)

    def _X(self, feats: pd.DataFrame) -> pd.DataFrame:
        return feats[self.subset] if self.subset else feats

    def refit(self, history: pd.DataFrame, as_of: pd.Timestamp) -> None:
        h = history[history.event_id.isin(self.store.keys())]
        if "completed" in h:
            h = h[h.decided_by.astype(str) != "Walkover"]
        if len(h) < self.min_train or h.outcome.nunique() < len(self.labels):
            return
        X = self._X(pd.DataFrame([self.store[e] for e in h.event_id]))
        cols = [c for c in X.columns if X[c].notna().any()]   # drop features never observed so far
        est = self.make_estimator()
        est.fit(X[cols], h.outcome.to_numpy())
        self.est, self.n_train, self.cols = est, len(h), cols

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        feats = self.engine.features(fixtures)
        for eid, (_, row) in zip(fixtures.event_id, feats.iterrows()):
            self.store[eid] = row
        if self.est is None:
            return np.full((len(fixtures), len(self.labels)), np.nan)
        P = self.est.predict_proba(self._X(feats)[self.cols])
        cls = list(self.est.classes_)
        return np.column_stack([P[:, cls.index(lab)] for lab in self.labels])


class LogisticModel(FeatureModel):
    name = "logistic"

    def __init__(self, labels, sport, C: float = 0.3, **kw):
        self.C = C
        super().__init__(labels, sport, **kw)

    def make_estimator(self):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=self.C, max_iter=2000))


class GBMModel(FeatureModel):
    name = "gradient-boosting"

    def make_estimator(self):
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=250,
                                              l2_regularization=1.0, min_samples_leaf=40,
                                              random_state=GLOBAL_SEED)
