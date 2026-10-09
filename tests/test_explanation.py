"""Stage 7: Explanation (causal_engine/pipeline/explanation.py).

Both tests force the deterministic template path (no real OpenAI call) so
the suite stays free, fast, and hermetic -- `test_llm_failure_falls_back_to_template`
still exercises the soft-fallback `except Exception` branch by making the
OpenAI client construction itself raise, without touching the network.
"""

from __future__ import annotations

import openai
import pytest

from causal_engine.pipeline import counterfactuals, explanation, interventions, narrative


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
    assert result.narrative_tier == "template"
    assert [(a.tier, a.outcome) for a in result.narrative_log] == [
        ("autogen", "skipped"), ("llm", "skipped"), ("template", "used")
    ]


def test_llm_failure_falls_back_to_template(
    monkeypatch, feature_df, effect_estimates, raw_df, domain_config
):
    pytest.importorskip("autogen_agentchat")  # tier 1 only errors (rather than "skipped") if AutoGen is installed
    pytest.importorskip("autogen_ext.models.openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-does-not-hit-network")

    class _RaisingClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated OpenAI client failure")

    monkeypatch.setattr(openai, "OpenAI", _RaisingClient)
    # tier 1 builds its own client; make that fail too so nothing reaches the network
    monkeypatch.setattr(narrative, "_make_autogen_client", lambda *a, **k: _RaisingClient())

    result = _run_explanation(feature_df, effect_estimates, raw_df, domain_config)
    assert result.narrative.startswith("Top-line drivers of attrition")
    assert result.narrative_tier == "template"
    assert [a.outcome for a in result.narrative_log] == ["error", "error", "used"]


def test_narrative_says_so_when_no_candidate_action_would_help(domain_config):
    from causal_engine.models.schemas import InterventionRecommendation

    rec = InterventionRecommendation(
        id="x", target_variable="compensation", expected_effect=-0.01, cost=100.0, roi=-1e-4,
        recommended=False, fairness_ratio=0.99, rank=1,
    )
    text = explanation._template_narrative(domain_config, [], [], [rec], {"compensation": 1.0})
    assert "No candidate action is expected to reduce attrition" in text
    assert "Recommended action" not in text
    assert explanation._has_recommended_action([rec]) is False  # there is no recommended action to judge
