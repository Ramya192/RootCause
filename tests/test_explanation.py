"""Stage 7: Explanation (rootcause/pipeline/explanation.py).

Both tests force the deterministic template path (no real OpenAI call) so
the suite stays free, fast, and hermetic -- `test_llm_failure_falls_back_to_template`
still exercises the soft-fallback `except Exception` branch by making the
OpenAI client construction itself raise, without touching the network.
"""

from __future__ import annotations

import openai

from rootcause.pipeline import counterfactuals, explanation, interventions


def _run_explanation(feature_df, effect_estimates, raw_df, domain_config):
    cf_results = counterfactuals.estimate_counterfactuals(feature_df, domain_config)
    recs = interventions.rank_interventions(raw_df, effect_estimates, domain_config)
    return explanation.generate_explanation(
        feature_df, effect_estimates, cf_results, recs, domain_config
    )


def test_template_narrative_when_no_api_key(
    monkeypatch, feature_df, effect_estimates, raw_df, domain_config
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = _run_explanation(feature_df, effect_estimates, raw_df, domain_config)
    assert result.narrative.startswith("Top-line drivers of attrition")
    feature_cols = domain_config["feature_store"]["feature_columns"]
    assert set(result.shap_summary.keys()) == set(feature_cols)
    assert all(v >= 0 for v in result.shap_summary.values())  # mean |SHAP|


def test_llm_failure_falls_back_to_template(
    monkeypatch, feature_df, effect_estimates, raw_df, domain_config
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-does-not-hit-network")

    class _RaisingClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated OpenAI client failure")

    monkeypatch.setattr(openai, "OpenAI", _RaisingClient)

    result = _run_explanation(feature_df, effect_estimates, raw_df, domain_config)
    assert result.narrative.startswith("Top-line drivers of attrition")
