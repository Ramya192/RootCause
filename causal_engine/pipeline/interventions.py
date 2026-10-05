"""Stage 6: Interventions.

Ranks the domain config's candidate actions by ROI (estimated reduction in
the outcome per dollar spent, derived from Stage 4's effect estimates) and
runs a fairlearn four-fifths-rule disparate-impact check on the outcome
across the configured sensitive attribute. If the underlying population
already shows disparate outcomes by group, every recommendation is flagged
rather than silently ranked as if the population were homogeneous -- a
targeted per-recommendation fairness audit would need per-individual
predictions we don't have in this vertical slice, so this is a
population-level caution flag, not a per-recommendation guarantee.
"""

from __future__ import annotations

import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate

from causal_engine.models.schemas import EffectEstimate, InterventionRecommendation


def _fairness_ratio(raw_data: pd.DataFrame, outcome_col: str, sensitive_attr: str) -> float:
    frame = MetricFrame(
        metrics=selection_rate,
        y_true=raw_data[outcome_col],
        y_pred=raw_data[outcome_col],
        sensitive_features=raw_data[sensitive_attr],
    )
    return float(frame.ratio(method="between_groups"))


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

    effects_by_treatment = {e.treatment: e for e in effect_estimates}

    scored = []
    for candidate in cfg["candidates"]:
        target = candidate["target_variable"]
        effect = effects_by_treatment.get(target)
        if effect is None:
            continue  # no effect estimate for this target -- skip, don't guess

        delta_outcome = effect.ate * candidate["expected_shift"]
        benefit = -delta_outcome  # positive = expected reduction in outcome
        cost = candidate["cost"]
        roi = benefit / cost if cost else 0.0

        scored.append(
            {
                "id": candidate["id"],
                "target_variable": target,
                "expected_effect": benefit,
                "cost": cost,
                "roi": roi,
            }
        )

    scored.sort(key=lambda r: r["roi"], reverse=True)

    return [
        InterventionRecommendation(
            **row,
            fairness_ratio=fairness_ratio,
            fairness_pass=population_fair,
            rank=i + 1,
        )
        for i, row in enumerate(scored)
    ]
