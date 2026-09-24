"""Pydantic contracts threaded between stages (rootcause/models/schemas.py)."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from rootcause.models.schemas import (
    CausalGraph,
    CounterfactualResult,
    EffectEstimate,
    Explanation,
    InterventionRecommendation,
    PipelineContext,
    PipelineResult,
)


def test_pipeline_context_allows_dataframe():
    ctx = PipelineContext(
        domain_id="employee_attrition",
        config={"a": 1},
        raw_data=pd.DataFrame({"x": [1, 2]}),
    )
    assert isinstance(ctx.raw_data, pd.DataFrame)
    assert ctx.feature_vectors is None


def test_causal_graph_requires_edge_tuples():
    graph = CausalGraph(nodes=["a", "b"], edges=[("a", "b")], algorithm="pc")
    assert graph.edges == [("a", "b")]


def test_effect_estimate_optional_refutation_fields_default_none():
    estimate = EffectEstimate(treatment="x", outcome="y", ate=-0.5, estimator="backdoor.linear_regression")
    assert estimate.refuter is None
    assert estimate.refutation_passed is None


def test_intervention_recommendation_requires_rank():
    with pytest.raises(ValidationError):
        InterventionRecommendation(
            id="a", target_variable="x", expected_effect=1.0, cost=100.0, roi=0.01
        )  # missing `rank`


def test_explanation_shap_summary_defaults_empty_dict():
    explanation = Explanation(narrative="n/a")
    assert explanation.shap_summary == {}


def test_pipeline_result_assembles_all_stage_outputs():
    result = PipelineResult(
        domain_id="employee_attrition",
        causal_graph=CausalGraph(nodes=["a"], edges=[], algorithm="pc"),
        effect_estimates=[EffectEstimate(treatment="a", outcome="b", ate=1.0, estimator="x")],
        counterfactuals=[
            CounterfactualResult(
                treatment="a", outcome="b", meta_learner="t_learner", mean_cate=0.1, description="d"
            )
        ],
        recommendations=[
            InterventionRecommendation(
                id="i1", target_variable="a", expected_effect=1.0, cost=10.0, roi=0.1, rank=1
            )
        ],
        explanation=Explanation(narrative="n"),
    )
    assert result.domain_id == "employee_attrition"
    assert len(result.effect_estimates) == 1
    assert len(result.recommendations) == 1
