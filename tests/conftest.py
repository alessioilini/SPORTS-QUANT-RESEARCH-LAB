import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # avoid OpenMP oversubscription (HistGradientBoosting)

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sportslab.config import DB_PATH, ROOT


def synthetic_league(n_teams=12, n_rounds=40, seed=1, sport="football", start="2020-08-01",
                     with_odds=True, margin=0.05):
    """Synthetic double round-robin-ish league with KNOWN attack/defence strengths."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams)
    dfn = rng.normal(0, 0.3, n_teams)
    home_adv = 0.25
    rows = []
    t0 = pd.Timestamp(start, tz="UTC")
    for r in range(n_rounds):
        perm = rng.permutation(n_teams)
        ko = t0 + pd.Timedelta(days=7 * r) + pd.Timedelta(hours=15)
        for j in range(0, n_teams, 2):
            h, a = perm[j], perm[j + 1]
            lam, mu = np.exp(att[h] - dfn[a] + home_adv), np.exp(att[a] - dfn[h])
            x, y = rng.poisson(lam), rng.poisson(mu)
            outcome = "H" if x > y else ("A" if x < y else "D")
            if sport != "football" and outcome == "D":
                outcome = "H" if rng.random() < lam / (lam + mu) else "A"
            row = dict(event_id=f"SYN-{r:03d}-{teams[h]}-{teams[a]}", sport=sport, competition="SYN",
                       season=f"S{r // 20}", event_time=ko, event_time_precision="minute",
                       forecast_time=ko - pd.Timedelta(days=1), home=teams[h], away=teams[a], neutral=0,
                       home_score=float(x), away_score=float(y), outcome=outcome,
                       decided_by="REG" if outcome != "D" or sport == "football" else "OT",
                       completed=1, result_available_at=ko + pd.Timedelta(hours=2),
                       true_lam=lam, true_mu=mu)
            if with_odds:
                from sportslab.models.goals import grid_to_probs, score_grid
                p = grid_to_probs(score_grid(lam, mu), ("H", "D", "A"))
                for lab, pi in zip("HDA", p):
                    row[f"odds_pinnacle_opening_{lab}"] = round(1 / (pi * (1 + margin)), 2)
                    row[f"odds_pinnacle_closing_{lab}"] = round(1 / (pi * (1 + margin)), 2)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, dict(teams=teams, att=att, dfn=dfn, home_adv=home_adv)


@pytest.fixture
def league():
    return synthetic_league()


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test.sqlite"


def has_historical() -> bool:
    return Path(DB_PATH).exists() and (ROOT / "reports" / "artifacts" / "predictions.parquet").exists()


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="historical data/artifacts not built (run scripts/fetch_data.py, build_db.py, "
                                   "run_validation.py, run_backtest.py)")
    if not has_historical():
        for it in items:
            if "historical" in it.keywords:
                it.add_marker(skip)
