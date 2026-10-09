"""CrewAI orchestration wiring the 7 pipeline stages behind 7 specialized
agents in a hierarchical crew: a manager LLM delegates each stage's task to
the agent that owns it, per the CDIA spec's target architecture.

Each stage's real work stays in the already-verified, pure functions under
causal_engine/pipeline/ -- these tools are thin wrappers that read/write a shared
PipelineRun object rather than round-tripping DataFrames or Pydantic models
through the LLM's JSON tool-call protocol, which only handles small
serializable payloads well. A tool's return value is a short human-readable
summary for the manager's context; the real typed outputs land on
PipelineRun and are assembled into the final PipelineResult after kickoff.

Tasks intentionally don't pin a `task.agent` -- in CrewAI's hierarchical
process the manager_agent is what actually executes each task, delegating to
the best-fit specialized agent via its role/goal/backstory, so pinning would
just bypass the delegation this architecture is meant to exercise. Explicit
`context=[previous_task]` still forces stage order regardless of how the
manager reasons about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

from causal_engine.models.schemas import (
    CausalGraph,
    CounterfactualResult,
    EffectEstimate,
    Explanation,
    InterventionRecommendation,
    PipelineResult,
)
from causal_engine.pipeline import (
    causal_discovery,
    counterfactuals,
    effect_estimation,
    explanation,
    feature_store,
    ingestion,
    interventions,
)


def _interval(e: EffectEstimate) -> str:
    return "" if e.ci_low is None or e.ci_high is None else f" (95% interval {e.ci_low:+.4f} to {e.ci_high:+.4f})"


def _noise_check(e: EffectEstimate) -> str:
    if e.refutation_passed is None:
        return "noise check not run"
    p = "" if e.refutation_p_value is None else f", p={e.refutation_p_value:.2f}"
    return "noise check passed" + p if e.refutation_passed else "noise check FAILED" + p + " (no evidence of an effect beyond noise)"


def summarize_effects(estimates: list[EffectEstimate]) -> str:
    """What Stage 4 found, including the checks that did not pass, so an agent has nothing to fill in."""
    return "; ".join(f"{e.treatment}->{e.outcome}: ATE={e.ate:+.4f}{_interval(e)}, {_noise_check(e)}" for e in estimates)


def summarize_counterfactuals(results: list[CounterfactualResult]) -> str:
    """Each result is about ONE treatment; the wording says so, since an agent once read it as a combined effect."""
    return "; ".join(f"[{cf.treatment} only, {cf.meta_learner}] {cf.description}" for cf in results)


def summarize_recommendations(recs: list[InterventionRecommendation]) -> str:
    def one(r: InterventionRecommendation) -> str:
        why = {
            "effect_not_distinguishable_from_noise": "its effect failed the noise check",
            "no_expected_reduction": "no expected reduction",
        }.get(r.not_recommended_reason or "", "would raise the outcome")
        verdict = "recommended" if r.recommended else f"NOT recommended ({why})"
        eo = "" if r.equalized_odds_pass is None else f", equalized_odds_pass={r.equalized_odds_pass}"
        return f"#{r.rank} {r.id} ({verdict}, roi={r.roi:.3g}, fairness_pass={r.fairness_pass}{eo})"

    return "; ".join(one(r) for r in recs)


@dataclass
class PipelineRun:
    """Shared state one crew kickoff writes into and reads from."""

    domain_config: dict
    data_path: str
    raw_data: Optional[pd.DataFrame] = None
    feature_df: Optional[pd.DataFrame] = None
    causal_graph: Optional[CausalGraph] = None
    effect_estimates: list[EffectEstimate] = field(default_factory=list)
    counterfactual_results: list[CounterfactualResult] = field(default_factory=list)
    recommendations: list[InterventionRecommendation] = field(default_factory=list)
    explanation_result: Optional[Explanation] = None
    completed: set[str] = field(default_factory=set)  # stages whose tool ran to completion


def _build_tools(run: PipelineRun) -> dict[str, object]:
    @tool("ingest_raw_data")
    def ingest_tool() -> str:
        """Load and validate the domain's raw CSV data. Always the first stage to run."""
        run.raw_data = ingestion.ingest(run.data_path, run.domain_config)
        run.completed.add("ingestion")
        return f"Ingested {len(run.raw_data)} rows, columns={list(run.raw_data.columns)}"

    @tool("build_feature_vectors")
    def feature_store_tool() -> str:
        """Materialize causal feature vectors via the Feast feature store.
        Requires the ingestion tool to have already run."""
        if run.raw_data is None:
            return "ERROR: no ingested data yet -- run the ingestion tool first"
        run.feature_df = feature_store.build_feature_vectors(run.raw_data, run.domain_config)
        run.completed.add("feature_store")
        return (
            f"Built feature vectors for {len(run.feature_df)} entities, "
            f"columns={list(run.feature_df.columns)}"
        )

    @tool("discover_causal_graph")
    def causal_discovery_tool() -> str:
        """Learn the causal DAG structure with PC plus domain priors.
        Requires the feature store tool to have already run."""
        if run.feature_df is None:
            return "ERROR: no feature vectors yet -- run the feature store tool first"
        run.causal_graph = causal_discovery.discover_graph(run.feature_df, run.domain_config)
        run.completed.add("causal_discovery")
        return f"Discovered graph edges: {run.causal_graph.edges}"

    @tool("estimate_causal_effects")
    def effect_estimation_tool() -> str:
        """Estimate the ATE for each configured treatment->outcome pair via
        DoWhy, with refutation checks. Requires the feature store tool to
        have already run."""
        if run.feature_df is None:
            return "ERROR: no feature vectors yet -- run the feature store tool first"
        run.effect_estimates = effect_estimation.estimate_effects(run.feature_df, run.domain_config)
        run.completed.add("effect_estimation")
        return summarize_effects(run.effect_estimates)

    @tool("simulate_counterfactuals")
    def counterfactuals_tool() -> str:
        """Estimate the configured what-if treatment shift's effect via a
        CausalML meta-learner. Requires the feature store tool to have
        already run."""
        if run.feature_df is None:
            return "ERROR: no feature vectors yet -- run the feature store tool first"
        run.counterfactual_results = counterfactuals.estimate_counterfactuals(
            run.feature_df, run.domain_config
        )
        run.completed.add("counterfactuals")
        return summarize_counterfactuals(run.counterfactual_results)

    @tool("rank_interventions")
    def interventions_tool() -> str:
        """Rank candidate interventions by ROI with a fairness check.
        Requires the ingestion and effect estimation tools to have already
        run."""
        if run.raw_data is None or not run.effect_estimates:
            return "ERROR: need ingested data and effect estimates first"
        run.recommendations = interventions.rank_interventions(
            run.raw_data, run.effect_estimates, run.domain_config
        )
        run.completed.add("interventions")
        return summarize_recommendations(run.recommendations)

    @tool("generate_explanation")
    def explanation_tool() -> str:
        """Generate the final SHAP attribution plus narrative explanation.
        Requires the feature store and effect estimation tools to have
        already run; counterfactuals and interventions are used if present."""
        if run.feature_df is None or not run.effect_estimates:
            return "ERROR: run the feature store and effect estimation tools first"
        run.explanation_result = explanation.generate_explanation(
            run.feature_df,
            run.effect_estimates,
            run.counterfactual_results,
            run.recommendations,
            run.domain_config,
        )
        run.completed.add("explanation")
        return run.explanation_result.narrative

    return {
        "ingestion": ingest_tool,
        "feature_store": feature_store_tool,
        "causal_discovery": causal_discovery_tool,
        "effect_estimation": effect_estimation_tool,
        "counterfactuals": counterfactuals_tool,
        "interventions": interventions_tool,
        "explanation": explanation_tool,
    }


_AGENT_SPECS = {
    "ingestion": dict(
        role="Data Ingestion Specialist",
        goal="Load and validate the domain's raw data for downstream causal analysis.",
        backstory="An ETL engineer who ensures every row is clean before any causal claim is made from it.",
    ),
    "feature_store": dict(
        role="Feature Store Engineer",
        goal="Serve consistent, point-in-time-correct causal feature vectors.",
        backstory=(
            "Owns the Feast feature store; guarantees every downstream stage sees "
            "the same features an online model would."
        ),
    ),
    "causal_discovery": dict(
        role="Causal Discovery Scientist",
        goal="Recover the true causal DAG among the domain's variables, respecting known domain priors.",
        backstory=(
            "A causal inference researcher who never lets structure-learning noise "
            "override a domain expert's known constraints."
        ),
    ),
    "effect_estimation": dict(
        role="Effect Estimation Analyst",
        goal="Quantify the causal effect of each actionable lever on the outcome, with refutation checks.",
        backstory=(
            "A DoWhy-fluent econometrician who distrusts any effect estimate that "
            "hasn't survived a placebo refutation."
        ),
    ),
    "counterfactuals": dict(
        role="Counterfactual Simulation Specialist",
        goal="Estimate what would happen to the outcome under a hypothetical treatment shift.",
        backstory=(
            "Runs CausalML meta-learners to answer 'what if' questions no "
            "observational average can answer alone."
        ),
    ),
    "interventions": dict(
        role="Intervention Strategist",
        goal="Rank candidate interventions by ROI while flagging any fairness concern.",
        backstory="Balances a finance team's ROI math against a fairness audit before recommending any action.",
    ),
    "explanation": dict(
        role="Explanation Writer",
        goal="Translate the full causal analysis into a narrative the target audience can act on.",
        backstory="A data storyteller who turns ATEs and SHAP values into a paragraph a business partner can actually use.",
    ),
}

# Data-order dependency between stages; each stage's task gets `context=` the
# previous one so the manager sees its output regardless of delegation order.
_STAGE_ORDER = [
    "ingestion",
    "feature_store",
    "causal_discovery",
    "effect_estimation",
    "counterfactuals",
    "interventions",
    "explanation",
]

_STAGE_TOOL_NAMES = {
    "ingestion": "ingest_raw_data",
    "feature_store": "build_feature_vectors",
    "causal_discovery": "discover_causal_graph",
    "effect_estimation": "estimate_causal_effects",
    "counterfactuals": "simulate_counterfactuals",
    "interventions": "rank_interventions",
    "explanation": "generate_explanation",
}


# The agents are LLMs, and left to summarise a tool's output they add claims the tool never made (a placebo check
# that was never reported, a combined effect from a one-treatment result). The pipeline result is built from what the
# tools wrote, not from this text, but the crew's printed answer should not mislead either.
_VERBATIM = (
    "When the tool returns, reply with its output exactly as given and nothing else: no interpretation, no "
    "causal wording, no checks or numbers the tool did not report, no advice."
)


def _mandate(stage: str, goal: str) -> str:
    tool_name = _STAGE_TOOL_NAMES[stage]
    role = _AGENT_SPECS[stage]["role"]
    return (
        f"{goal} This task belongs ONLY to the coworker with role '{role}' -- "
        f"if delegating, delegate to exactly that coworker and no other, even "
        f"if another coworker's role name sounds related. That coworker MUST "
        f"call the `{tool_name}` tool exactly once to do this -- it is the "
        f"only source of real output for this stage. Do not answer from "
        f"reasoning or general knowledge instead of calling it, and do not "
        f"call it more than once. {_VERBATIM}"
    )


_TASK_DESCRIPTIONS = {
    "ingestion": (
        _mandate("ingestion", "Ingest the domain's raw data."),
        "The ingestion tool's output, verbatim.",
    ),
    "feature_store": (
        _mandate("feature_store", "Build causal feature vectors from the ingested data."),
        "The tool's output, verbatim.",
    ),
    "causal_discovery": (
        _mandate("causal_discovery", "Discover the causal graph from the feature vectors."),
        "The tool's output, verbatim.",
    ),
    "effect_estimation": (
        _mandate(
            "effect_estimation",
            "Estimate the causal effect of each configured treatment on the outcome.",
        ),
        "The tool's output, verbatim.",
    ),
    "counterfactuals": (
        _mandate("counterfactuals", "Simulate the domain's configured counterfactual scenario."),
        "The tool's output, verbatim.",
    ),
    "interventions": (
        _mandate("interventions", "Rank the candidate interventions using the effect estimates."),
        "The tool's output, verbatim.",
    ),
    "explanation": (
        _mandate(
            "explanation",
            "Generate the final narrative explanation using every prior stage's output.",
        ),
        "The tool's output, verbatim.",
    ),
}


def build_crew(domain_config: dict, data_path: str, manager_llm: str) -> tuple[Crew, PipelineRun]:
    run = PipelineRun(domain_config=domain_config, data_path=data_path)
    tools = _build_tools(run)

    # temperature 0 for the agents and the manager: nothing here benefits from creative wording
    agents = {
        stage: Agent(tools=[tools[stage]], allow_delegation=False, llm=LLM(model=manager_llm, temperature=0), **spec)
        for stage, spec in _AGENT_SPECS.items()
    }

    tasks: list[Task] = []
    for stage in _STAGE_ORDER:
        description, expected_output = _TASK_DESCRIPTIONS[stage]
        tasks.append(
            Task(
                description=description,
                expected_output=expected_output,
                context=[tasks[-1]] if tasks else [],
            )
        )

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.hierarchical,
        manager_llm=LLM(model=manager_llm, temperature=0),
        verbose=True,
    )
    return crew, run


def missing_stages(run: PipelineRun) -> list[str]:
    """Stages, in order, whose tool never completed (the manager LLM skipped or mis-delegated them)."""
    return [stage for stage in _STAGE_ORDER if stage not in run.completed]


def run_pipeline(domain_id: str, domain_config: dict, data_path: str) -> PipelineResult:
    manager_llm = domain_config.get("explanation", {}).get("llm_model", "gpt-4o-mini")
    crew, run = build_crew(domain_config, data_path, manager_llm)
    crew.kickoff()

    missing = missing_stages(run)
    if missing:
        raise RuntimeError(
            f"Crew finished without running every stage (missing: {', '.join(missing)}) -- the manager LLM "
            "skipped or mis-delegated a task. Check crew.kickoff()'s output above."
        )

    return PipelineResult(
        domain_id=domain_id,
        causal_graph=run.causal_graph,
        effect_estimates=run.effect_estimates,
        counterfactuals=run.counterfactual_results,
        recommendations=run.recommendations,
        explanation=run.explanation_result,
    )
