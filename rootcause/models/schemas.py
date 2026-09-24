"""Shared Pydantic contracts passed between pipeline stages 1-7, so agents
exchange typed objects instead of loose dicts."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class PipelineContext(BaseModel):
    """Threaded through all 7 stages: raw + feature data plus the resolved
    domain config each stage reads its own section from."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    domain_id: str
    config: dict[str, Any]
    raw_data: pd.DataFrame
    feature_vectors: Optional[pd.DataFrame] = None


class CausalGraph(BaseModel):
    """Output of Stage 3."""

    nodes: list[str]
    edges: list[tuple[str, str]]
    algorithm: str


class EffectEstimate(BaseModel):
    """One treatment->outcome estimate, output of Stage 4."""

    treatment: str
    outcome: str
    ate: float
    estimator: str
    refuter: Optional[str] = None
    refutation_passed: Optional[bool] = None


class CounterfactualResult(BaseModel):
    """Output of Stage 5."""

    treatment: str
    outcome: str
    meta_learner: str
    mean_cate: float
    description: str


class InterventionRecommendation(BaseModel):
    """One ranked candidate action, output of Stage 6."""

    id: str
    target_variable: str
    expected_effect: float
    cost: float
    roi: float
    fairness_ratio: Optional[float] = None
    fairness_pass: bool = True
    rank: int


class Explanation(BaseModel):
    """Output of Stage 7."""

    narrative: str
    shap_summary: dict[str, float] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Final artifact returned by the crew / API for one analysis run."""

    domain_id: str
    causal_graph: CausalGraph
    effect_estimates: list[EffectEstimate]
    counterfactuals: list[CounterfactualResult]
    recommendations: list[InterventionRecommendation]
    explanation: Explanation
