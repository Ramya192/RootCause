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

from pathlib import Path

from rootcause.models.schemas import PipelineResult
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


def resolve_data_path(domain_config: dict) -> Path:
    """The domain's configured dataset, resolved against the repo root."""
    return REPO_ROOT / domain_config["ingestion"]["data_path"]


def run_direct(domain_id: str, domain_config: dict, data_path: str | Path) -> PipelineResult:
    raw = ingestion.ingest(data_path, domain_config)
    features = feature_store.build_feature_vectors(raw, domain_config)
    graph = causal_discovery.discover_graph(features, domain_config)
    effects = effect_estimation.estimate_effects(features, domain_config)
    cf_results = counterfactuals.estimate_counterfactuals(features, domain_config)
    recs = interventions.rank_interventions(raw, effects, domain_config)
    explained = explanation.generate_explanation(
        features, effects, cf_results, recs, domain_config
    )
    return PipelineResult(
        domain_id=domain_id,
        causal_graph=graph,
        effect_estimates=effects,
        counterfactuals=cf_results,
        recommendations=recs,
        explanation=explained,
    )


def run_crew(domain_id: str, domain_config: dict, data_path: str | Path) -> PipelineResult:
    # Imported lazily: crewai is slow to import and only this mode needs it.
    from rootcause.agents.crew import run_pipeline

    return run_pipeline(domain_id, domain_config, str(data_path))
