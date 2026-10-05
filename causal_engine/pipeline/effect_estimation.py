"""Stage 4: Effect Estimation.

Uses DoWhy to identify and estimate each configured treatment->outcome
effect. Confounders come straight from the domain config (declared per
treatment) rather than being re-derived from the Stage 3 graph -- DoWhy's
backdoor identification only needs the common-causes set, and taking it
from config keeps this stage decoupled from Stage 3's occasionally-partial
graph recovery (PC on a small sample won't always find every true edge).

Refutation is a permutation placebo test (`refutation: permutation_placebo`):
shuffle the treatment column, re-run the same estimator, repeat, and compare
the real estimate to that placebo distribution. `refutation_p_value` is the
two-sided share of placebo effects at least as large as the real one, and
`refutation_passed` means p < alpha: the estimate is distinguishable from what
a treatment with no link to the outcome produces. A failure therefore reads
"no evidence of an effect beyond noise", which is the right answer for a null
result. It does NOT test for unmeasured confounding -- no placebo can. That is what
`sensitivity.py` is for (`effect_estimation.sensitivity`, on by default): it reports how
strong an unmeasured confounder would have to be to explain each estimate away.

DoWhy's own `placebo_treatment_refuter` is deliberately not used: with the
linear-regression estimator it returned a placebo effect of exactly 0.0 on
every run, so a "|placebo| < |estimate|" rule could never fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dowhy import CausalModel

from causal_engine.models.schemas import EffectEstimate
from causal_engine.pipeline import modalities, sensitivity

PERMUTATION_PLACEBO = "permutation_placebo"
DEFAULT_SIMULATIONS = 100
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 0
COPY_CORRELATION = 0.99  # an adjustment column this correlated with the treatment is a proxy of it, not a confounder


def permutation_placebo_p_value(
    data: pd.DataFrame,
    identified_estimand,
    estimate,
    treatment: str,
    simulations: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> float:
    """Two-sided permutation p-value of `estimate` against a shuffled-treatment null.

    Each simulation permutes `treatment` (severing its link to the outcome and
    to the confounders, keeping its marginal distribution) and re-fits the
    original estimator. Uses (1 + #{|null| >= |estimate|}) / (1 + simulations),
    so the smallest possible value is 1 / (1 + simulations), never 0.
    """
    estimator = estimate.estimator
    rng = np.random.default_rng(seed)
    values = data[treatment].to_numpy()

    null_effects = np.empty(simulations)
    for i in range(simulations):
        permuted = data.copy()
        permuted[treatment] = rng.permutation(values)
        placebo = estimator.get_new_estimator_object(identified_estimand)
        placebo.fit(
            permuted,
            effect_modifier_names=estimator._effect_modifier_names,
            **(placebo._fit_params if hasattr(placebo, "_fit_params") else {}),
        )
        null_effects[i] = placebo.estimate_effect(
            permuted,
            control_value=estimate.control_value,
            treatment_value=estimate.treatment_value,
            target_units=estimator._target_units,
        ).value

    as_extreme = int(np.sum(np.abs(null_effects) >= abs(estimate.value)))
    return (1 + as_extreme) / (1 + simulations)


def estimate_effects(feature_df: pd.DataFrame, domain_config: dict) -> list[EffectEstimate]:
    cfg = domain_config["effect_estimation"]
    outcome = cfg["outcome"]
    estimator = cfg.get("estimator", "backdoor.linear_regression")
    refuter = cfg.get("refutation")
    if refuter is not None and refuter != PERMUTATION_PLACEBO:
        raise ValueError(
            f"effect_estimation.refutation={refuter!r} is not supported; "
            f"use {PERMUTATION_PLACEBO!r} (or omit it to skip refutation)"
        )
    simulations = cfg.get("refutation_simulations", DEFAULT_SIMULATIONS)
    alpha = cfg.get("refutation_alpha", DEFAULT_ALPHA)
    seed = cfg.get("refutation_seed", DEFAULT_SEED)
    # With `adjust_for_attachments`, PDF/image feature columns (Stage 1 attachments) join every
    # treatment's adjustment set; they are absent from real-only runs, where this adds nothing.
    extra_adjustment = (
        [c for c in modalities.attachment_columns(domain_config) if c in feature_df.columns]
        if cfg.get("adjust_for_attachments", False)
        else []
    )
    run_sensitivity = cfg.get("sensitivity", True)  # cheap (two OLS fits per treatment)
    sens_alpha = cfg.get("sensitivity_alpha", DEFAULT_ALPHA)

    results: list[EffectEstimate] = []
    for treatment_cfg in cfg["treatments"]:
        treatment = treatment_cfg["name"]
        confounders = [*treatment_cfg.get("confounders", []), *extra_adjustment]
        # A declared level that no row has (Freddie Mac 2010-11 have no `tpo_unspecified` loans) is
        # all zeros: it adjusts for nothing and makes the design matrix rank-deficient.
        confounders = [c for c in confounders if feature_df[c].nunique(dropna=False) > 1]
        for column in extra_adjustment:
            if column != treatment and abs(feature_df[treatment].corr(feature_df[column])) > COPY_CORRELATION:
                raise ValueError(
                    f"attachment feature {column!r} is almost a copy of treatment {treatment!r} "
                    f"(|correlation| > {COPY_CORRELATION}); adjusting for it would remove the effect being estimated. "
                    "Do not render or extract the treatment into attachments."
                )

        model = CausalModel(
            data=feature_df,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders,
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(identified_estimand, method_name=estimator)

        p_value = refutation_passed = None
        if refuter:
            p_value = permutation_placebo_p_value(
                feature_df, identified_estimand, estimate, treatment, simulations, seed
            )
            refutation_passed = p_value < alpha

        sens = None
        if run_sensitivity and estimator in sensitivity.LINEAR_ESTIMATORS:
            sens = sensitivity.linear_sensitivity(feature_df, treatment, outcome, list(confounders), sens_alpha)

        results.append(
            EffectEstimate(
                treatment=treatment,
                outcome=outcome,
                ate=float(estimate.value),
                estimator=estimator,
                refuter=refuter,
                refutation_passed=refutation_passed,
                refutation_p_value=p_value,
                sensitivity=sens,
            )
        )
    return results
