"""End-to-end stage-by-stage run (ingestion -> ... -> explanation), calling
each causal_engine/pipeline/*.py function directly rather than through the
CrewAI/LLM orchestration in causal_engine/agents/crew.py (that path costs real
money/minutes per an LLM manager -- see tests/test_crew.py for a free
wiring-only check of it) and asserts the whole thing assembles into a valid
PipelineResult.
"""

from __future__ import annotations

from causal_engine.models.schemas import PipelineResult
from causal_engine.pipeline import counterfactuals, interventions


def test_full_pipeline_assembles_valid_result(
    monkeypatch, raw_df, feature_df, causal_graph, effect_estimates, domain_config
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # keep this test free/deterministic

    cf_results = counterfactuals.estimate_counterfactuals(feature_df, domain_config)
    recs = interventions.rank_interventions(raw_df, effect_estimates, domain_config)

    from causal_engine.pipeline import explanation

    explanation_result = explanation.generate_explanation(
        feature_df, effect_estimates, cf_results, recs, domain_config
    )

    result = PipelineResult(
        domain_id="employee_attrition",
        causal_graph=causal_graph,
        effect_estimates=effect_estimates,
        counterfactuals=cf_results,
        recommendations=recs,
        explanation=explanation_result,
    )

    assert result.domain_id == "employee_attrition"
    assert len(result.causal_graph.edges) == 6
    assert len(result.effect_estimates) == 3
    assert len(result.counterfactuals) == 1
    # Regression guard for the treatments/candidates naming-mismatch bug
    # documented in memory: this must be 3, not 0.
    assert len(result.recommendations) == 3
    assert result.explanation.narrative
