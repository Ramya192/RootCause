"""Stage 4: Effect Estimation.

Uses DoWhy to identify and estimate each configured treatment->outcome
effect. Confounders come straight from the domain config (declared per
treatment) rather than being re-derived from the Stage 3 graph -- DoWhy's
backdoor identification only needs the common-causes set, and taking it
from config keeps this stage decoupled from Stage 3's occasionally-partial
graph recovery (PC on a small sample won't always find every true edge).
"""

from __future__ import annotations

import pandas as pd
from dowhy import CausalModel

from rootcause.models.schemas import EffectEstimate


def estimate_effects(feature_df: pd.DataFrame, domain_config: dict) -> list[EffectEstimate]:
    cfg = domain_config["effect_estimation"]
    outcome = cfg["outcome"]
    estimator = cfg.get("estimator", "backdoor.linear_regression")
    refuter = cfg.get("refutation")

    results: list[EffectEstimate] = []
    for treatment_cfg in cfg["treatments"]:
        treatment = treatment_cfg["name"]
        confounders = treatment_cfg.get("confounders", [])

        model = CausalModel(
            data=feature_df,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders,
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(identified_estimand, method_name=estimator)

        refutation_passed = None
        if refuter:
            refutation = model.refute_estimate(
                identified_estimand, estimate, method_name=refuter
            )
            refutation_passed = abs(refutation.new_effect) < abs(estimate.value)

        results.append(
            EffectEstimate(
                treatment=treatment,
                outcome=outcome,
                ate=float(estimate.value),
                estimator=estimator,
                refuter=refuter,
                refutation_passed=refutation_passed,
            )
        )
    return results
