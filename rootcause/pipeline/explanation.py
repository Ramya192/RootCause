"""Stage 7: Explanation.

Trains a lightweight outcome classifier on the feature vectors purely to
attribute SHAP feature importances -- this model plays no role in the causal
reasoning itself, Stages 3-6 already did that. Combines those attributions
with Stage 4-6's causal outputs into a narrative for the config's target
audience, trying three writers in order and keeping the first whose text passes
the grounding check in narrative.py (every number comes from the facts, required
caveats present):

  1. autogen: analyst -> skeptic -> writer chain (optional `autogen` dependency)
  2. llm: one OpenAI call over the same facts
  3. template: deterministic text from the facts; always available, needs no key

Tiers 1 and 2 need OPENAI_API_KEY; any failure (auth, network, rate limit, missing
package, a rejected draft) moves on to the next tier instead of crashing the run.
`Explanation.narrative_log` records what happened to each tier.
Config: `explanation.tiers` (default all three, in that order; 'template' is always the
last resort) and an optional `explanation.autogen` dict (max_tokens, temperature,
timeout_seconds, total_timeout_seconds).
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
    NarrativeAttempt,
)
from rootcause.pipeline import narrative
from rootcause.pipeline.preprocessing import feature_columns_of
from rootcause.utils.vocabulary import domain_vocabulary


def _shap_summary(feature_df: pd.DataFrame, domain_config: dict) -> dict[str, float]:
    feature_cols = feature_columns_of(feature_df, domain_config)
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
    outcome = domain_vocabulary(domain_config).outcome
    top_driver = max(shap_summary, key=shap_summary.get) if shap_summary else "n/a"

    if domain_config.get("explanation", {}).get("observational"):
        lines = [
            f"Associations with {outcome} in observational data (not proven causes), ranked by estimated "
            f"effect size:"
        ]
    else:
        lines = [f"Top-line drivers of {outcome}, ranked by causal effect size:"]
    for e in sorted(effect_estimates, key=lambda e: abs(e.ate), reverse=True):
        line = f"- {e.treatment}: ATE={e.ate:+.4f} ({e.estimator})"
        if e.refutation_passed is False:
            p = f" (permutation p={e.refutation_p_value:.3f})" if e.refutation_p_value is not None else ""
            line += f" -- placebo check FAILED{p}: no evidence of an effect beyond noise"
        lines.append(line)
    lines.append(f"SHAP attribution agrees '{top_driver}' carries the most predictive weight.")
    for cf in counterfactuals:
        lines.append(f"- {cf.description}")
    if recommendations:
        top = recommendations[0]
        fairness_note = (
            "fairness check passed"
            if top.fairness_pass
            else "FAIRNESS FLAG: population-level disparity detected"
            + (f" (group ratio {top.fairness_ratio:.2f})" if top.fairness_ratio is not None else "")
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
    requirements: str = "",
) -> str:
    from openai import OpenAI

    cfg = domain_config["explanation"]
    facts = _template_narrative(
        domain_config, effect_estimates, counterfactuals, recommendations, shap_summary
    )
    vocab = domain_vocabulary(domain_config)
    prompt = (
        f"You are explaining a causal analysis of {vocab.outcome} "
        f"among {vocab.entities} to an "
        f"audience of {cfg['audience']}. Using only the facts below, write a "
        f"short (3-5 sentence) plain-language narrative -- no jargon like "
        f"'ATE', 'SHAP', 'placebo', 'p-value' or 'statistically significant', translate them into business "
        f"terms. Use only numbers that "
        f"appear in the facts, and keep any note that a check failed or that fairness "
        f"was flagged.\n\nFacts:\n{facts}"
        + (f"\n\nYour narrative MUST:\n{requirements}" if requirements else "")
    )
    response = OpenAI().chat.completions.create(
        model=cfg.get("llm_model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def _tier_order(domain_config: dict) -> list[str]:
    tiers = list(domain_config.get("explanation", {}).get("tiers") or narrative.NARRATIVE_TIERS)
    unknown = [t for t in tiers if t not in narrative.NARRATIVE_TIERS]
    if unknown:
        raise ValueError(
            f"unknown explanation tier(s) {unknown}; expected a subset of "
            f"{list(narrative.NARRATIVE_TIERS)}"
        )
    return [t for t in tiers if t != "template"] + ["template"]


def _roi_below_cost(recommendations: list[InterventionRecommendation]) -> bool:
    return bool(recommendations) and recommendations[0].roi < 1.0


def _grounding(
    text, facts, effect_estimates, recommendations, plain_language=False, observational=False
) -> narrative.GroundingReport:
    return narrative.check_grounding(
        text,
        facts,
        plain_language=plain_language,
        observational=observational,
        roi_below_cost=_roi_below_cost(recommendations),
        failed_refutation=any(e.refutation_passed is False for e in effect_estimates),
        fairness_flagged=bool(recommendations) and not recommendations[0].fairness_pass,
    )


def generate_explanation(
    feature_df: pd.DataFrame,
    effect_estimates: list[EffectEstimate],
    counterfactuals: list[CounterfactualResult],
    recommendations: list[InterventionRecommendation],
    domain_config: dict,
) -> Explanation:
    shap_summary = _shap_summary(feature_df, domain_config)
    args = (domain_config, effect_estimates, counterfactuals, recommendations, shap_summary)
    facts = _template_narrative(*args)
    cfg = domain_config.get("explanation", {})
    vocab = domain_vocabulary(domain_config)
    has_key = bool(os.environ.get("OPENAI_API_KEY"))

    observational = bool(cfg.get("observational"))
    requirements = narrative.required_points(
        [e.treatment for e in effect_estimates if e.refutation_passed is False],
        bool(recommendations) and not recommendations[0].fairness_pass,
        observational,
        _roi_below_cost(recommendations),
    )
    writers = {
        "autogen": lambda: narrative.autogen_narrative(
            facts + (f"\n\nThe narrative MUST:\n{requirements}" if requirements else ""),
            model=cfg.get("llm_model", "gpt-4o-mini"),
            audience=cfg["audience"],
            outcome=vocab.outcome,
            entities=vocab.entities,
            cfg=cfg.get("autogen"),
        ),
        "llm": lambda: _llm_narrative(*args, requirements=requirements),
    }

    log: list[NarrativeAttempt] = []
    for tier in _tier_order(domain_config):
        if tier == "template":
            text = facts
            log.append(NarrativeAttempt(tier=tier, outcome="used"))
            return Explanation(
                narrative=text, shap_summary=shap_summary, narrative_tier=tier, narrative_log=log
            )
        if not has_key:
            log.append(NarrativeAttempt(tier=tier, outcome="skipped", detail="no OPENAI_API_KEY"))
            continue
        if tier == "autogen" and not narrative.autogen_available():
            log.append(
                NarrativeAttempt(tier=tier, outcome="skipped", detail="autogen is not installed")
            )
            continue
        try:
            text = writers[tier]()
        except Exception as exc:  # soft dependency on a live API -- move on, don't crash the run
            log.append(
                NarrativeAttempt(tier=tier, outcome="error", detail=f"{type(exc).__name__}: {exc}"[:300])
            )
            continue
        report = _grounding(
            text, facts, effect_estimates, recommendations, plain_language=True, observational=observational
        )
        if report.passed:
            log.append(NarrativeAttempt(tier=tier, outcome="used"))
            return Explanation(
                narrative=text, shap_summary=shap_summary, narrative_tier=tier, narrative_log=log
            )
        log.append(NarrativeAttempt(tier=tier, outcome="rejected", detail=report.detail()))
