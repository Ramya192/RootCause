"""Stage 6: Interventions.

Ranks the domain config's candidate actions by ROI (estimated reduction in
the outcome per dollar spent, derived from Stage 4's effect estimates) and
runs a fairlearn four-fifths-rule disparate-impact check on the outcome
across the configured sensitive attribute. If the underlying population
already shows disparate outcomes by group, every recommendation is flagged
rather than silently ranked as if the population were homogeneous -- a
targeted per-recommendation fairness audit would need per-individual
predictions of what each action would do, which this pipeline does not make, so
the four-fifths check is a population-level caution flag, not a per-recommendation
guarantee.

It also reports equalized odds, which needs per-record predictions: a logistic
regression of the outcome on the domain's other columns (never the sensitive
attribute itself), scored out of fold so no record is predicted by a model that saw
it. The largest gap between groups in true-positive or false-positive rate is
compared with `interventions.equalized_odds_threshold` (default 0.1). This
describes how a predictive model would treat the groups, not how any intervention
would, so it is shown beside the four-fifths flag and does not change the ranking.

An action is only recommended when its estimated effect lowers the outcome AND that estimate passed its
placebo check; one whose check failed stays in the list (ranked by ROI, so the evaluation of the ranking is
unchanged) but is flagged not recommended, since its estimated reduction cannot be told apart from zero.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, equalized_odds_difference, selection_rate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_engine.models.schemas import EffectEstimate, InterventionRecommendation

logger = logging.getLogger(__name__)

DEFAULT_EQUALIZED_ODDS_THRESHOLD = 0.1
_EQUALIZED_ODDS_MAX_ROWS = 20_000  # a logistic regression on more rows than this adds time, not accuracy
_EQUALIZED_ODDS_MAX_LEVELS = 30  # a non-numeric column with more distinct values than this is dropped, not one-hot encoded
_EQUALIZED_ODDS_FOLDS = 5
_EQUALIZED_ODDS_MIN_PER_CELL = 5  # fewer positives or negatives than this in a group and a rate is noise


def _fairness_ratio(raw_data: pd.DataFrame, outcome_col: str, sensitive_attr: str) -> float:
    frame = MetricFrame(
        metrics=selection_rate,
        y_true=raw_data[outcome_col],
        y_pred=raw_data[outcome_col],
        sensitive_features=raw_data[sensitive_attr],
    )
    return float(frame.ratio(method="between_groups"))


def _equalized_odds(
    raw_data: pd.DataFrame, outcome_col: str, sensitive_attr: str, id_col: str | None, seed: int = 0
) -> float | None:
    """Largest group gap in true-positive or false-positive rate of an out-of-fold outcome model, or None.

    Predicted positives are the top-scoring fraction equal to the outcome's base rate, so a rare outcome
    (fraud at 6%) is not predicted all-negative at a 0.5 cut-off, which would make every rate zero.
    """
    y = raw_data[outcome_col].astype(int)
    groups = raw_data[sensitive_attr]
    if y.nunique() < 2 or groups.nunique() < 2:
        return None
    for _, g_y in y.groupby(groups):
        if (g_y == 1).sum() < _EQUALIZED_ODDS_MIN_PER_CELL or (g_y == 0).sum() < _EQUALIZED_ODDS_MIN_PER_CELL:
            return None

    drop = [c for c in (outcome_col, sensitive_attr, id_col) if c and c in raw_data.columns]
    X = raw_data.drop(columns=drop).select_dtypes(exclude=["datetime", "timedelta"])
    # a text column with one value per record (an id, a free-text field) is not a feature, and one-hot encoding it
    # would create a column per record
    wide = [c for c in X.select_dtypes(exclude="number").columns if X[c].nunique() > _EQUALIZED_ODDS_MAX_LEVELS]
    X = pd.get_dummies(X.drop(columns=wide), dtype=float)
    X = X.loc[:, X.nunique() > 1].fillna(X.median(numeric_only=True))
    if X.shape[1] == 0:
        return None

    idx = np.arange(len(raw_data))
    if len(idx) > _EQUALIZED_ODDS_MAX_ROWS:
        idx = np.sort(np.random.default_rng(seed).choice(idx, _EQUALIZED_ODDS_MAX_ROWS, replace=False))
    X, y, groups = X.iloc[idx], y.iloc[idx], groups.iloc[idx]

    folds = StratifiedKFold(n_splits=_EQUALIZED_ODDS_FOLDS, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    score = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
    n_positive = max(1, int(round(float(y.mean()) * len(y))))
    predicted = np.zeros(len(y), dtype=int)
    predicted[np.argsort(-score)[:n_positive]] = 1
    return float(equalized_odds_difference(y, predicted, sensitive_features=groups))


def rank_interventions(
    raw_data: pd.DataFrame,
    effect_estimates: list[EffectEstimate],
    domain_config: dict,
) -> list[InterventionRecommendation]:
    cfg = domain_config["interventions"]
    sensitive_attr = cfg["sensitive_attribute"]
    fairness_threshold = cfg["fairness_threshold"]
    outcome_col = domain_config["effect_estimation"]["outcome"]

    fairness_ratio = _fairness_ratio(raw_data, outcome_col, sensitive_attr)
    population_fair = fairness_ratio >= fairness_threshold

    eo_difference = _equalized_odds(
        raw_data, outcome_col, sensitive_attr, domain_config.get("ingestion", {}).get("id_column")
    )
    eo_threshold = cfg.get("equalized_odds_threshold", DEFAULT_EQUALIZED_ODDS_THRESHOLD)

    effects_by_treatment = {e.treatment: e for e in effect_estimates}

    scored = []
    for candidate in cfg["candidates"]:
        target = candidate["target_variable"]
        effect = effects_by_treatment.get(target)
        if effect is None:
            # skip rather than guess, but say so: a target_variable that matches no treatment is a config slip
            logger.warning("intervention %r targets %r, which has no effect estimate; skipped", candidate["id"], target)
            continue

        delta_outcome = effect.ate * candidate["expected_shift"]
        benefit = -delta_outcome  # positive = expected reduction in outcome
        cost = candidate["cost"]
        if not cost > 0:
            raise ValueError(f"intervention {candidate['id']!r}: cost must be positive to compute an ROI, got {cost!r}")
        roi = benefit / cost

        # a failed placebo check comes first: the sign of an effect that cannot be told apart from noise
        # is not a finding, so "raises the outcome" would be an invented claim
        if effect.refutation_passed is False:
            reason = "effect_not_distinguishable_from_noise"
        elif benefit < 0:
            reason = "raises_outcome"
        elif benefit == 0:
            reason = "no_expected_reduction"
        else:
            reason = None

        scored.append(
            {
                "id": candidate["id"],
                "target_variable": target,
                "expected_effect": benefit,
                "cost": cost,
                "roi": roi,
                "recommended": reason is None,
                "not_recommended_reason": reason,
            }
        )

    scored.sort(key=lambda r: r["roi"], reverse=True)

    return [
        InterventionRecommendation(
            **row,
            fairness_ratio=fairness_ratio,
            fairness_pass=population_fair,
            equalized_odds_difference=eo_difference,
            equalized_odds_pass=None if eo_difference is None else eo_difference <= eo_threshold,
            rank=i + 1,
        )
        for i, row in enumerate(scored)
    ]
