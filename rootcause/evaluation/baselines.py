"""Naive baselines: what an analyst without the causal pipeline would do.

The obvious objection to RootCause is "isn't this just feature importance?". These
baselines answer it by running the same question -- how much does each lever move
the outcome, and which intervention should we fund -- through the two shortcuts
people actually take, on the same data and against the same simulated truth:

  regression_all_columns    regress the outcome on EVERY observed column and read
                            the treatment's coefficient. Adjusts for everything,
                            including mediators, so it recovers the *direct*
                            effect. Pay reaches attrition only through satisfaction,
                            so once satisfaction is held fixed pay's coefficient
                            is ~0 and the lever looks useless.
  regression_treatment_only regress the outcome on the treatment alone (no
                            adjustment). Right when nothing confounds the
                            treatment; wrong when something does, and blind to it.
  shap_importance           fit a gradient-boosted classifier, take mean |SHAP| per
                            feature, and treat importance as causal weight: rank
                            the candidate interventions by the importance of the
                            variable each one moves, per unit cost. SHAP explains
                            the model's predictions, not what happens when you
                            intervene; it has no sign and no notion of mediation.

The two regressions feed Stage 6's own `rank_interventions`, so the ONLY difference
from the pipeline is the effect estimate. Everything is scored with the same
metrics as the pipeline (metrics.py), so the rows are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from rootcause.evaluation import metrics
from rootcause.models.schemas import EffectEstimate, InterventionRecommendation
from rootcause.pipeline import interventions
from rootcause.pipeline.explanation import _shap_summary
from rootcause.pipeline.preprocessing import feature_columns_of

REGRESSION_ALL = "regression_all_columns"
REGRESSION_TREATMENT_ONLY = "regression_treatment_only"
SHAP_IMPORTANCE = "shap_importance"

METHOD_LABELS = {
    "pipeline": "RootCause pipeline",
    REGRESSION_ALL: "Regression, all columns",
    REGRESSION_TREATMENT_ONLY: "Regression, treatment only",
    SHAP_IMPORTANCE: "SHAP importance ÷ cost",
}


@dataclass
class BaselineRun:
    """One baseline scored on one dataset draw."""

    name: str
    effect_rows: list[metrics.EffectRow] = field(default_factory=list)  # empty for SHAP (no ATE)
    ranking: Optional[metrics.RankingScore] = None
    values: dict[str, Optional[float]] = field(default_factory=dict)
    importances: Optional[dict[str, float]] = None  # SHAP only: feature -> mean |SHAP|
    top_feature: Optional[str] = None  # SHAP only


def _ols_coefficient(y: np.ndarray, X: np.ndarray) -> float:
    """Coefficient of X's first column in an OLS fit of y on [1, X] (a linear
    probability model for a binary outcome -- the same model Stage 4 fits)."""
    design = np.column_stack([np.ones(len(y)), X])
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def regression_effects(
    features: pd.DataFrame, domain_config: dict, adjust_for_all: bool
) -> list[EffectEstimate]:
    """One ATE per configured treatment from a plain OLS coefficient."""
    outcome = domain_config["effect_estimation"]["outcome"]
    columns = feature_columns_of(features, domain_config)
    y = features[outcome].to_numpy(dtype=float)
    name = REGRESSION_ALL if adjust_for_all else REGRESSION_TREATMENT_ONLY

    estimates = []
    for treatment_cfg in domain_config["effect_estimation"]["treatments"]:
        treatment = treatment_cfg["name"]
        others = [c for c in columns if c != treatment] if adjust_for_all else []
        X = features[[treatment, *others]].to_numpy(dtype=float)
        estimates.append(
            EffectEstimate(
                treatment=treatment,
                outcome=outcome,
                ate=_ols_coefficient(y, X),
                estimator=f"naive.{name}",
            )
        )
    return estimates


def shap_recommendations(
    features: pd.DataFrame, domain_config: dict
) -> tuple[list[InterventionRecommendation], dict[str, float]]:
    """Rank the candidate interventions by (mean |SHAP| of the variable each one
    targets) / cost. `expected_shift` is ignored: importance has no direction, so
    it cannot say whether the configured shift helps or hurts."""
    importances = _shap_summary(features, domain_config)
    scored = []
    for cand in domain_config["interventions"]["candidates"]:
        importance = importances.get(cand["target_variable"])
        if importance is None or not cand["cost"]:
            continue
        scored.append(
            {
                "id": cand["id"],
                "target_variable": cand["target_variable"],
                "expected_effect": importance,
                "cost": cand["cost"],
                "roi": importance / cand["cost"],
            }
        )
    scored.sort(key=lambda r: r["roi"], reverse=True)
    recs = [InterventionRecommendation(**row, rank=i + 1) for i, row in enumerate(scored)]
    return recs, importances


def _score_ranking_into(run: BaselineRun, recs, true_roi: Mapping[str, float]) -> None:
    run.ranking = metrics.score_ranking(recs, true_roi)
    if run.ranking is not None:
        run.values["top1_correct"] = float(run.ranking.top1_correct)
        run.values["rank_tau"] = run.ranking.kendall_tau
        run.values["roi_regret"] = run.ranking.roi_regret


def score_baselines(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    domain_config: dict,
    true_effects: Mapping[str, float],
    true_roi: Mapping[str, float],
) -> dict[str, BaselineRun]:
    """Run every baseline on one draw and score it against the SCM's truth."""
    out: dict[str, BaselineRun] = {}

    for name, adjust_for_all in ((REGRESSION_ALL, True), (REGRESSION_TREATMENT_ONLY, False)):
        effects = regression_effects(features, domain_config, adjust_for_all)
        run = BaselineRun(name)
        run.effect_rows = metrics.score_effects(effects, true_effects)
        if run.effect_rows:
            run.values["ate_mae"] = float(np.mean([r.abs_error for r in run.effect_rows]))
            run.values["ate_sign_agreement"] = float(np.mean([r.sign_agrees for r in run.effect_rows]))
        recs = interventions.rank_interventions(raw, effects, domain_config)
        _score_ranking_into(run, recs, true_roi)
        out[name] = run

    recs, importances = shap_recommendations(features, domain_config)
    shap_run = BaselineRun(SHAP_IMPORTANCE, importances=importances)
    shap_run.top_feature = max(importances, key=importances.get) if importances else None
    levers = {t["name"] for t in domain_config["effect_estimation"]["treatments"]}
    shap_run.values["top_feature_is_lever"] = float(shap_run.top_feature in levers)
    _score_ranking_into(shap_run, recs, true_roi)
    out[SHAP_IMPORTANCE] = shap_run
    return out
