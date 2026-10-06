"""Stage 5: Counterfactuals (V1 -- CausalML meta-learners, per spec; V2 GANs
deferred).

Binarizes the configured treatment at `treatment_threshold` and fits a
meta-learner to estimate each unit's individual treatment effect (CATE),
then reports the population mean. `counterfactuals.meta_learner` picks the
learner (a value outside the list is an error, not silently a T-learner):

  t_learner   separate outcome models for treated and control (default)
  s_learner   one outcome model with the treatment as a feature; with a linear base
              model the effect is a single constant, so it shows no heterogeneity
  x_learner   T-learner, then models of the imputed effects blended by propensity
  r_learner   residual-on-residual (Robinson) with cross-fitting
  dr_learner  doubly robust: outcome model plus inverse-propensity correction

`base_learner` is the outcome model inside them (`linear`, the default, or `gbm`,
gradient boosting). X, R and DR need a propensity score P(treated | covariates):
`propensity: estimated` (default) fits an elastic-net model; `constant` uses the
treated share, which is right only when treatment was randomized (an RCT). The score
is clipped to [`propensity_clip`, 1 - `propensity_clip`] (default 0.01) before use,
and the share of units whose raw score fell outside [0.05, 0.95] is reported: where
the treatment is nearly a function of the covariates there is little overlap, the
inverse-propensity weights of the DR-learner get large, and X/R/DR estimates lean on
extrapolation of the outcome models rather than on comparable treated and control units.

The T-learner falls back to a hand-rolled two-model version (same math: separate
treated/control regressors, CATE = mu1(x) - mu0(x)) if `causalml` isn't
importable -- it has Cython extensions that can fail to build on Windows without
a C++ toolchain. The other learners need `causalml` and say so if it is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from causal_engine.models.schemas import CounterfactualResult, SubgroupEffect
from causal_engine.pipeline.preprocessing import feature_columns_of
from causal_engine.utils.vocabulary import domain_vocabulary


LEARNERS = {
    "t_learner": "BaseTRegressor",
    "s_learner": "BaseSRegressor",
    "x_learner": "BaseXRegressor",
    "r_learner": "BaseRRegressor",
    "dr_learner": "BaseDRRegressor",
}
NEEDS_PROPENSITY = {"x_learner", "r_learner", "dr_learner"}
BASE_LEARNERS = ("linear", "gbm")
PROPENSITY_MODES = ("estimated", "constant")


def _base_model(name: str):
    if name == "linear":
        return LinearRegression()
    return GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)


def _check_choice(key: str, value: str, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"counterfactuals.{key}={value!r} is not supported; use one of {sorted(allowed)}")


EXTREME_PROPENSITY = 0.05  # a raw score below this or above 1 minus it counts as "no overlap"
DEFAULT_CLIP = 0.01


@dataclass(frozen=True)
class CateFit:
    cate: np.ndarray
    label: str  # what ran, for the result's `meta_learner` field
    extreme_propensity_share: Optional[float]  # None when the learner uses no propensity score


def _propensity(X: np.ndarray, w: np.ndarray, mode: str) -> np.ndarray:
    if mode == "constant":
        return np.full(len(w), float(np.mean(w)))
    from causalml.propensity import ElasticNetPropensityModel

    return np.asarray(ElasticNetPropensityModel().fit_predict(X, w), dtype=float)


def _causalml_learner(
    X: np.ndarray, w: np.ndarray, y: np.ndarray, meta_learner: str, base_learner: str, propensity: str, clip: float
) -> tuple[np.ndarray, Optional[float]]:
    from causalml.inference import meta

    learner_cls = getattr(meta, LEARNERS[meta_learner])
    kwargs, extreme = {}, None
    if meta_learner in NEEDS_PROPENSITY:
        raw = _propensity(X, w, propensity)
        extreme = float(np.mean((raw < EXTREME_PROPENSITY) | (raw > 1 - EXTREME_PROPENSITY)))
        kwargs["p"] = np.clip(raw, clip, 1 - clip)
    learner = learner_cls(learner=_base_model(base_learner))
    cate = learner.fit_predict(X=X, treatment=w, y=y, **kwargs)
    return np.asarray(cate).flatten(), extreme


def estimate_cate(
    X: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    meta_learner: str = "t_learner",
    base_learner: str = "linear",
    propensity: str = "estimated",
    clip: float = DEFAULT_CLIP,
) -> CateFit:
    """Per-unit effect estimates from the chosen meta-learner, and a label saying what ran."""
    _check_choice("meta_learner", meta_learner, LEARNERS)
    _check_choice("base_learner", base_learner, BASE_LEARNERS)
    _check_choice("propensity", propensity, PROPENSITY_MODES)
    if not 0.0 < clip < 0.5:
        raise ValueError(f"counterfactuals.propensity_clip={clip!r} must be between 0 and 0.5")

    label_extras = []
    if base_learner != "linear":
        label_extras.append(f"{base_learner} base")
    if meta_learner in NEEDS_PROPENSITY and propensity != "estimated":
        label_extras.append(f"{propensity} propensity")
    try:
        cate, extreme = _causalml_learner(X, w, y, meta_learner, base_learner, propensity, clip)
        used = ", ".join([f"causalml.{LEARNERS[meta_learner]}", *label_extras])
    except ImportError:
        if meta_learner != "t_learner" or base_learner != "linear":
            raise ImportError(
                f"counterfactuals.meta_learner={meta_learner!r} with base_learner={base_learner!r} needs "
                "the `causalml` package; only the linear T-learner has a scikit-learn fallback"
            ) from None
        cate, extreme, used = _fallback_t_learner(X, w, y), None, "sklearn-fallback T-learner"
    return CateFit(cate=cate, label=used, extreme_propensity_share=extreme)


def _fallback_t_learner(X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu0 = LinearRegression().fit(X[w == 0], y[w == 0])
    mu1 = LinearRegression().fit(X[w == 1], y[w == 1])
    return mu1.predict(X) - mu0.predict(X)


def binary_subgroup_levels(frame: pd.DataFrame, column: str) -> list[tuple[str, np.ndarray]]:
    """(level label, boolean mask) for each level of a 0/1 column present in `frame`.

    Subgroups are limited to binary columns on purpose: a continuous one would need a
    cut point, and how it is cut would change the answer."""
    if column not in frame.columns:
        raise ValueError(f"counterfactuals.subgroups: column {column!r} is not in the data")
    values = frame[column]
    if not set(values.dropna().unique()) <= {0, 1}:
        raise ValueError(
            f"counterfactuals.subgroups: column {column!r} is not binary (0/1); "
            "subgroups are only supported for binary columns"
        )
    return [(str(level), (values == level).to_numpy()) for level in (0, 1) if (values == level).any()]


def subgroup_effects(feature_df: pd.DataFrame, cate: np.ndarray, columns: list[str]) -> list[SubgroupEffect]:
    """Mean per-unit effect within each level of each listed binary column."""
    effects = []
    for column in columns:
        for level, mask in binary_subgroup_levels(feature_df, column):
            effects.append(
                SubgroupEffect(variable=column, level=level, n=int(mask.sum()), mean_cate=float(np.mean(cate[mask])))
            )
    return effects


def _describe(domain_config: dict, vocab, treatment_col: str, mean_cate: float) -> str:
    """What the mean effect means: it is averaged over ALL units (each one's modelled effect of being
    at or above the threshold rather than below it), not over only the units that are below it."""
    if domain_config.get("explanation", {}).get("observational"):
        return (
            f"Across all {vocab.entities}, being at or above the {treatment_col} threshold rather than below it goes with "
            f"a difference of {mean_cate:+.4f} in mean predicted {vocab.outcome} (an association in observational data)"
        )
    return (
        f"Across all {vocab.entities}, being at or above the {treatment_col} threshold rather than below it "
        f"is estimated to change mean predicted {vocab.outcome} by {mean_cate:+.4f}"
    )


def estimate_counterfactuals(
    feature_df: pd.DataFrame, domain_config: dict
) -> list[CounterfactualResult]:
    cfg = domain_config["counterfactuals"]
    treatment_col = cfg["treatment"]
    outcome_col = cfg["outcome"]
    threshold = cfg.get("treatment_threshold", 0.0)
    meta_learner = cfg.get("meta_learner", "t_learner")
    base_learner = cfg.get("base_learner", "linear")
    propensity = cfg.get("propensity", "estimated")
    clip = cfg.get("propensity_clip", DEFAULT_CLIP)

    w = (feature_df[treatment_col] >= threshold).astype(int).to_numpy()
    vocab = domain_vocabulary(domain_config)
    # Every encoded feature except the treatment itself. The entity id and
    # outcome are excluded: an id is not a covariate, it just lets the learner
    # memorise rows.
    covariates = [c for c in feature_columns_of(feature_df, domain_config) if c != treatment_col]
    X = feature_df[covariates].to_numpy(dtype=float)
    y = feature_df[outcome_col].to_numpy(dtype=float)

    if w.sum() == 0 or w.sum() == len(w):
        raise ValueError(
            f"Treatment '{treatment_col}' has no variation at threshold={threshold}; "
            "cannot estimate a treatment effect"
        )

    fit = estimate_cate(X, w, y, meta_learner, base_learner, propensity, clip)
    cate, used = fit.cate, fit.label

    mean_cate = float(np.mean(cate))
    subgroups = subgroup_effects(feature_df, cate, list(cfg.get("subgroups", [])))
    return [
        CounterfactualResult(
            treatment=treatment_col,
            outcome=outcome_col,
            meta_learner=f"{meta_learner} ({used})",
            mean_cate=mean_cate,
            description=_describe(domain_config, vocab, treatment_col, mean_cate),
            cate_std=float(np.std(cate)),
            subgroups=subgroups,
            extreme_propensity_share=fit.extreme_propensity_share,
        )
    ]
