"""Stage 5: Counterfactuals (V1 -- CausalML meta-learners, per spec; V2 GANs
deferred).

Binarizes the configured treatment at `treatment_threshold` and fits a
T-learner to estimate each employee's individual treatment effect (CATE),
then reports the population mean. Falls back to a hand-rolled two-model
T-learner (same math: separate treated/control regressors, CATE = mu1(x) -
mu0(x)) if `causalml` isn't importable -- it has Cython extensions that can
fail to build on Windows without a C++ toolchain; the fallback keeps this
stage working either way with an identical call signature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from rootcause.models.schemas import CounterfactualResult


def _causalml_t_learner(X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    from causalml.inference.meta import BaseTRegressor

    learner = BaseTRegressor(learner=LinearRegression())
    cate = learner.fit_predict(X=X, treatment=w, y=y)
    return np.asarray(cate).flatten()


def _fallback_t_learner(X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu0 = LinearRegression().fit(X[w == 0], y[w == 0])
    mu1 = LinearRegression().fit(X[w == 1], y[w == 1])
    return mu1.predict(X) - mu0.predict(X)


def estimate_counterfactuals(
    feature_df: pd.DataFrame, domain_config: dict
) -> list[CounterfactualResult]:
    cfg = domain_config["counterfactuals"]
    treatment_col = cfg["treatment"]
    outcome_col = cfg["outcome"]
    threshold = cfg.get("treatment_threshold", 0.0)
    meta_learner = cfg.get("meta_learner", "t_learner")

    w = (feature_df[treatment_col] >= threshold).astype(int).to_numpy()
    covariates = [c for c in feature_df.columns if c not in (treatment_col, outcome_col)]
    X = feature_df[covariates].to_numpy(dtype=float)
    y = feature_df[outcome_col].to_numpy(dtype=float)

    if w.sum() == 0 or w.sum() == len(w):
        raise ValueError(
            f"Treatment '{treatment_col}' has no variation at threshold={threshold}; "
            "cannot estimate a treatment effect"
        )

    try:
        cate = _causalml_t_learner(X, w, y)
        used = "causalml.BaseTRegressor"
    except ImportError:
        cate = _fallback_t_learner(X, w, y)
        used = "sklearn-fallback T-learner"

    mean_cate = float(np.mean(cate))
    return [
        CounterfactualResult(
            treatment=treatment_col,
            outcome=outcome_col,
            meta_learner=f"{meta_learner} ({used})",
            mean_cate=mean_cate,
            description=(
                f"If employees below the {treatment_col} threshold were shifted "
                f"to match those above it, mean predicted {outcome_col} would "
                f"change by {mean_cate:+.4f}"
            ),
        )
    ]
