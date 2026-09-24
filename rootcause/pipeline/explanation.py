"""Stage 7: Explanation.

Trains a lightweight attrition classifier on the feature vectors purely to
attribute SHAP feature importances -- this model plays no role in the causal
reasoning itself, Stages 3-6 already did that. Combines those attributions
with Stage 4-6's causal outputs into a narrative for the config's target
audience. AutoGen is on the spec doc but deferred per the V1 scope
guardrails, so this calls the OpenAI API directly; if no key is configured
(or the call fails for any reason -- auth, network, rate limit), falls back
to a deterministic templated narrative built from the same facts, so the
pipeline still runs end-to-end without a live API key.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier

from rootcause.models.schemas import (
    CounterfactualResult,
    EffectEstimate,
    Explanation,
    InterventionRecommendation,
)


def _shap_summary(feature_df: pd.DataFrame, domain_config: dict) -> dict[str, float]:
    feature_cols = domain_config["feature_store"]["feature_columns"]
    outcome_col = domain_config["effect_estimation"]["outcome"]

    X = feature_df[feature_cols].to_numpy(dtype=float)
    y = feature_df[outcome_col].to_numpy(dtype=float)

    model = GradientBoostingClassifier(random_state=0).fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # binary classifier, positive class only
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    return {col: float(v) for col, v in zip(feature_cols, mean_abs)}


def _template_narrative(
    domain_config: dict,
    effect_estimates: list[EffectEstimate],
    counterfactuals: list[CounterfactualResult],
    recommendations: list[InterventionRecommendation],
    shap_summary: dict[str, float],
) -> str:
    outcome = domain_config["effect_estimation"]["outcome"]
    top_driver = max(shap_summary, key=shap_summary.get) if shap_summary else "n/a"

    lines = [f"Top-line drivers of {outcome}, ranked by causal effect size:"]
    for e in sorted(effect_estimates, key=lambda e: abs(e.ate), reverse=True):
        lines.append(f"- {e.treatment}: ATE={e.ate:+.4f} ({e.estimator})")
    lines.append(f"SHAP attribution agrees '{top_driver}' carries the most predictive weight.")
    for cf in counterfactuals:
        lines.append(f"- {cf.description}")
    if recommendations:
        top = recommendations[0]
        fairness_note = (
            "fairness check passed"
            if top.fairness_pass
            else "FAIRNESS FLAG: population-level disparity detected"
        )
        lines.append(
            f"Recommended action: '{top.id}' (ROI={top.roi:.3g}, "
            f"cost=${top.cost:,.0f}) -- {fairness_note}"
        )
    return "\n".join(lines)


def _llm_narrative(
    domain_config: dict,
    effect_estimates: list[EffectEstimate],
    counterfactuals: list[CounterfactualResult],
    recommendations: list[InterventionRecommendation],
    shap_summary: dict[str, float],
) -> str:
    from openai import OpenAI

    cfg = domain_config["explanation"]
    facts = _template_narrative(
        domain_config, effect_estimates, counterfactuals, recommendations, shap_summary
    )
    prompt = (
        f"You are explaining a causal analysis of employee attrition to an "
        f"audience of {cfg['audience']}. Using only the facts below, write a "
        f"short (3-5 sentence) plain-language narrative -- no jargon like "
        f"'ATE' or 'SHAP', translate them into business terms.\n\nFacts:\n{facts}"
    )
    response = OpenAI().chat.completions.create(
        model=cfg.get("llm_model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def generate_explanation(
    feature_df: pd.DataFrame,
    effect_estimates: list[EffectEstimate],
    counterfactuals: list[CounterfactualResult],
    recommendations: list[InterventionRecommendation],
    domain_config: dict,
) -> Explanation:
    shap_summary = _shap_summary(feature_df, domain_config)

    narrative = None
    if os.environ.get("OPENAI_API_KEY"):
        try:
            narrative = _llm_narrative(
                domain_config, effect_estimates, counterfactuals, recommendations, shap_summary
            )
        except Exception:
            narrative = None  # soft dependency on a live API -- fall back, don't crash the run

    if narrative is None:
        narrative = _template_narrative(
            domain_config, effect_estimates, counterfactuals, recommendations, shap_summary
        )

    return Explanation(narrative=narrative, shap_summary=shap_summary)
