"""Runs the 7 stages either directly or through the CrewAI crew, returning
the same PipelineResult either way.

`run_direct` calls each stage's pure function in order -- the exact code path
the pytest integration test exercises -- so it's fast and only costs an LLM
call if OPENAI_API_KEY is set (Stage 7's narrative, which falls back to a
template otherwise). `run_crew` is the CDIA spec's hierarchical CrewAI
orchestration: several minutes and many paid LLM calls per run (see
rootcause/agents/crew.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rootcause.models.schemas import (
    CausalGraph,
    CounterfactualResult,
    EffectEstimate,
    InterventionRecommendation,
    PipelineResult,
)
from rootcause.pipeline import (
    causal_discovery,
    counterfactuals,
    effect_estimation,
    explanation,
    feature_store,
    ingestion,
    interventions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def available_datasets(domain_config: dict) -> list[str]:
    return sorted(domain_config["ingestion"]["datasets"])


def resolve_dataset(domain_config: dict, requested: str | None = None) -> str:
    """The dataset kind to run: `requested`, else the domain's default."""
    ingestion_cfg = domain_config["ingestion"]
    name = requested or ingestion_cfg["default_dataset"]
    if name not in ingestion_cfg["datasets"]:
        raise ValueError(
            f"dataset {name!r} is not configured for this domain; available: "
            f"{available_datasets(domain_config)}"
        )
    return name


def resolve_data_path(domain_config: dict, dataset: str | None = None) -> Path:
    """The chosen dataset's file, resolved against the repo root."""
    name = resolve_dataset(domain_config, dataset)
    return REPO_ROOT / domain_config["ingestion"]["datasets"][name]


def with_dataset(domain_config: dict, dataset: str) -> dict:
    """Copy of the config tagged with the dataset being run, so Stage 2 keeps
    each (domain, dataset) pair's Feast state apart."""
    return {**domain_config, "run": {"dataset": dataset}}


@dataclass
class AnalysisOutputs:
    """Everything Stages 1-6 produce -- all of it except the Stage 7 narrative."""

    raw: pd.DataFrame
    features: pd.DataFrame
    graph: CausalGraph
    effects: list[EffectEstimate]
    counterfactuals: list[CounterfactualResult]
    recommendations: list[InterventionRecommendation]


def run_analysis_stages(domain_config: dict, data_path: str | Path) -> AnalysisOutputs:
    """Stages 1-6. Split out from `run_direct` so the evaluation harness can score
    them without paying for Stage 7 (a SHAP fit, plus an LLM call if a key is set)."""
    raw = ingestion.ingest(data_path, domain_config)
    features = feature_store.build_feature_vectors(raw, domain_config)
    graph = causal_discovery.discover_graph(features, domain_config)
    effects = effect_estimation.estimate_effects(features, domain_config)
    cf_results = counterfactuals.estimate_counterfactuals(features, domain_config)
    recs = interventions.rank_interventions(raw, effects, domain_config)
    return AnalysisOutputs(raw, features, graph, effects, cf_results, recs)


def run_direct(domain_id: str, domain_config: dict, data_path: str | Path) -> PipelineResult:
    out = run_analysis_stages(domain_config, data_path)
    explained = explanation.generate_explanation(
        out.features, out.effects, out.counterfactuals, out.recommendations, domain_config
    )
    return PipelineResult(
        domain_id=domain_id,
        causal_graph=out.graph,
        effect_estimates=out.effects,
        counterfactuals=out.counterfactuals,
        recommendations=out.recommendations,
        explanation=explained,
    )


def run_crew(domain_id: str, domain_config: dict, data_path: str | Path) -> PipelineResult:
    # Imported lazily: crewai is slow to import and only this mode needs it.
    from rootcause.agents.crew import run_pipeline

    return run_pipeline(domain_id, domain_config, str(data_path))
