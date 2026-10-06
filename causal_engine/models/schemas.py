"""Shared Pydantic contracts for the stage outputs and the final PipelineResult, so the stages,
the crew and the API exchange typed objects instead of loose dicts."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CausalGraph(BaseModel):
    """Output of Stage 3."""

    nodes: list[str]
    edges: list[tuple[str, str]]
    algorithm: str
    # Things worth knowing about the structure, e.g. that the edges contain a cycle
    warnings: list[str] = Field(default_factory=list)


class SensitivityResult(BaseModel):
    """How strong an unmeasured confounder must be to explain an estimate away
    (Cinelli & Hazlett 2020; see pipeline/sensitivity.py). Shares are partial R^2:
    the fraction of RESIDUAL variance a confounder explains."""

    estimate: float
    alpha: float
    partial_r2_treatment_outcome: float
    # Confounder strength (in BOTH treatment and outcome) that would reduce the estimate to
    # zero, and that would make it statistically insignificant at `alpha`.
    robustness_value: float
    robustness_value_alpha: float
    # Worst-case benchmark: the observed covariate that would bias the estimate most if an
    # unobserved confounder were as strong as it. None if there are no covariates.
    benchmark_covariate: Optional[str] = None
    benchmark_r2_treatment: Optional[float] = None
    benchmark_r2_outcome: Optional[float] = None
    benchmark_bias: Optional[float] = None
    benchmark_adjusted_estimate: Optional[float] = None
    robust_to_benchmark: Optional[bool] = None


class EffectEstimate(BaseModel):
    """One treatment->outcome estimate, output of Stage 4."""

    treatment: str
    outcome: str
    ate: float
    estimator: str
    # Sampling interval for the ATE (linear-regression estimator only; None otherwise). It does not
    # cover unmeasured confounding or a wrong model.
    std_error: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    refuter: Optional[str] = None
    refutation_passed: Optional[bool] = None
    # Permutation p-value behind `refutation_passed` (see pipeline/effect_estimation.py)
    refutation_p_value: Optional[float] = None
    # Omitted-variable-bias sensitivity (linear-regression estimator only; None otherwise)
    sensitivity: Optional[SensitivityResult] = None


class SubgroupEffect(BaseModel):
    """Mean estimated effect within one level of a subgroup variable (Stage 5)."""

    variable: str
    level: str  # "0" / "1" for a binary variable
    n: int
    mean_cate: float

    @property
    def key(self) -> str:
        return f"{self.variable}={self.level}"


class CounterfactualResult(BaseModel):
    """Output of Stage 5."""

    treatment: str
    outcome: str
    meta_learner: str
    mean_cate: float
    description: str
    # Spread of the per-unit effects, and mean effects within the subgroups the domain
    # config lists under `counterfactuals.subgroups` (empty if it lists none).
    cate_std: Optional[float] = None
    subgroups: list[SubgroupEffect] = Field(default_factory=list)
    # X/R/DR-learners only: share of units whose propensity score was below 0.05 or above 0.95
    # (treatment nearly determined by the covariates: no overlap, so the estimate is unreliable)
    extreme_propensity_share: Optional[float] = None


class InterventionRecommendation(BaseModel):
    """One ranked candidate action, output of Stage 6."""

    id: str
    target_variable: str
    expected_effect: float
    cost: float
    roi: float
    # False when the estimated effect would RAISE the outcome (negative benefit): the action is
    # listed so the reader can see it, but it is not something to do.
    recommended: bool = True
    fairness_ratio: Optional[float] = None
    fairness_pass: bool = True
    rank: int


class NarrativeAttempt(BaseModel):
    """What happened to one narrative tier in Stage 7 (see pipeline/narrative.py)."""

    tier: str  # autogen | llm | template
    outcome: str  # used | rejected | skipped | error
    detail: str = ""


class Explanation(BaseModel):
    """Output of Stage 7."""

    narrative: str
    shap_summary: dict[str, float] = Field(default_factory=dict)
    # Which tier wrote `narrative`, and why each tier tried before it was passed over
    narrative_tier: str = "template"
    narrative_log: list[NarrativeAttempt] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Final artifact returned by the crew / API for one analysis run."""

    domain_id: str
    causal_graph: CausalGraph
    effect_estimates: list[EffectEstimate]
    counterfactuals: list[CounterfactualResult]
    recommendations: list[InterventionRecommendation]
    explanation: Explanation
