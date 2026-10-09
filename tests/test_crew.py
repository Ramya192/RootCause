"""CrewAI wiring (causal_engine/agents/crew.py) -- structure only, no kickoff().

build_crew() just constructs Agent/Task/Crew objects and makes no network
call; crew.kickoff() is what runs the real (paid, multi-minute) manager-LLM
orchestration, per memory's cost/time note, so it's deliberately not
exercised here. This test would have caught, at import/construction time,
a regression to either of the two real delegation bugs documented in
memory (a task's mandate no longer naming its exact tool/coworker), though
not the delegation behavior itself -- only kickoff() proves that.
"""

from __future__ import annotations

import pytest
from crewai import Crew, Process

from causal_engine.agents import crew as crew_module
from causal_engine.agents.crew import (
    _AGENT_SPECS,
    _STAGE_ORDER,
    _STAGE_TOOL_NAMES,
    PipelineRun,
    _build_tools,
    build_crew,
    missing_stages,
)
from causal_engine.models.schemas import PipelineResult

# CrewAI builds an OpenAI client while constructing the crew, so it needs *a* key. This one is fake and no
# request is ever sent (kickoff() is never called for real), so the suite stays free and never reads .env.
FAKE_KEY = "sk-test-fake-does-not-hit-network"


@pytest.fixture(autouse=True)
def _fake_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)


@pytest.fixture(scope="module")
def built_crew(domain_config, data_path):
    mp = pytest.MonkeyPatch()
    mp.setenv("OPENAI_API_KEY", FAKE_KEY)
    try:
        yield build_crew(domain_config, str(data_path), manager_llm="gpt-4o-mini")
    finally:
        mp.undo()


def test_build_crew_wires_all_seven_stages_in_order(built_crew, domain_config):
    crew, run = built_crew

    assert len(crew.agents) == 7
    assert len(crew.tasks) == 7
    assert crew.process == Process.hierarchical
    assert run.domain_config is domain_config
    assert run.raw_data is None  # nothing has run yet -- kickoff() populates this


def test_every_task_mandates_its_exact_tool_and_coworker(built_crew):
    crew, _ = built_crew
    for stage, task in zip(_STAGE_ORDER, crew.tasks):
        tool_name = _STAGE_TOOL_NAMES[stage]
        role = _AGENT_SPECS[stage]["role"]
        assert tool_name in task.description
        assert role in task.description


def test_tasks_are_chained_in_stage_order(built_crew):
    crew, _ = built_crew
    for prev_task, task in zip(crew.tasks, crew.tasks[1:]):
        assert task.context == [prev_task]


def test_a_crew_run_that_skipped_stages_is_reported_by_name(domain_config, data_path):
    _, run = build_crew(domain_config, str(data_path), manager_llm="gpt-4o-mini")
    assert missing_stages(run) == list(_STAGE_ORDER)  # nothing has run
    run.completed.update({"ingestion", "feature_store", "causal_discovery", "explanation"})
    assert missing_stages(run) == ["effect_estimation", "counterfactuals", "interventions"]
    run.completed.update(_STAGE_ORDER)
    assert missing_stages(run) == []


def _call(tool):
    return tool.func()


def test_a_stage_tool_called_out_of_order_refuses_instead_of_running(domain_config, data_path):
    run = PipelineRun(domain_config=domain_config, data_path=str(data_path))
    tools = _build_tools(run)

    for stage in ("feature_store", "causal_discovery", "effect_estimation", "counterfactuals"):
        assert _call(tools[stage]).startswith("ERROR"), stage
    assert _call(tools["interventions"]).startswith("ERROR")
    assert _call(tools["explanation"]).startswith("ERROR")
    assert run.completed == set()  # a refused call must not count as a completed stage
    assert run.raw_data is None and run.feature_df is None


def _stub_kickoff_that_runs_every_stage_tool(monkeypatch):
    """A manager that delegates perfectly: calls each agent's tool once, in stage order, with no LLM."""

    def kickoff(self, *args, **kwargs):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # Stage 7 then uses the free template tier
        for agent in self.agents:
            for stage_tool in agent.tools:
                _call(stage_tool)
        return "done"

    monkeypatch.setattr(Crew, "kickoff", kickoff)


def test_run_pipeline_assembles_the_result_from_what_the_stage_tools_wrote(domain_config, data_path, monkeypatch):
    _stub_kickoff_that_runs_every_stage_tool(monkeypatch)

    result = crew_module.run_pipeline("employee_attrition", domain_config, str(data_path))

    assert isinstance(result, PipelineResult)
    assert result.domain_id == "employee_attrition"
    assert result.causal_graph.edges
    assert len(result.effect_estimates) == len(domain_config["effect_estimation"]["treatments"])
    assert result.counterfactuals and result.recommendations
    assert result.explanation.narrative_tier == "template"


def test_run_pipeline_fails_loudly_when_the_manager_skips_every_stage(domain_config, data_path, monkeypatch):
    monkeypatch.setattr(Crew, "kickoff", lambda self, *a, **k: "the manager answered from general knowledge")

    with pytest.raises(RuntimeError) as err:
        crew_module.run_pipeline("employee_attrition", domain_config, str(data_path))

    for stage in _STAGE_ORDER:
        assert stage in str(err.value)


# --- keeping the agents from adding claims the tools never made ---


def test_every_task_tells_the_agent_to_return_the_tool_output_verbatim(built_crew):
    crew, _ = built_crew
    for task in crew.tasks:
        assert "exactly as given and nothing else" in task.description
        assert "verbatim" in task.expected_output


def test_the_agents_and_the_manager_run_at_temperature_zero(built_crew):
    crew, _ = built_crew
    assert crew.manager_llm.temperature == 0
    assert all(agent.llm.temperature == 0 for agent in crew.agents)


def _estimate(**kw):
    from causal_engine.models.schemas import EffectEstimate

    return EffectEstimate(treatment="pay", outcome="quit", ate=-0.1286, estimator="fake", **kw)


def test_the_effect_summary_reports_the_checks_so_an_agent_has_nothing_to_invent():
    passed = crew_module.summarize_effects([_estimate(ci_low=-0.15, ci_high=-0.10, refutation_passed=True, refutation_p_value=0.0)])
    failed = crew_module.summarize_effects([_estimate(refutation_passed=False, refutation_p_value=0.89)])
    unrun = crew_module.summarize_effects([_estimate()])

    assert "ATE=-0.1286" in passed and "95% interval -0.1500 to -0.1000" in passed and "noise check passed, p=0.00" in passed
    assert "noise check FAILED, p=0.89 (no evidence of an effect beyond noise)" in failed
    assert "noise check not run" in unrun


def test_the_counterfactual_summary_names_the_single_treatment_it_is_about():
    from causal_engine.models.schemas import CounterfactualResult

    cf = CounterfactualResult(
        treatment="job_satisfaction", outcome="attrition", meta_learner="t", mean_cate=-0.18, description="shifts attrition by -0.1787"
    )

    text = crew_module.summarize_counterfactuals([cf])

    assert text.startswith("[job_satisfaction only, t]") and "-0.1787" in text


def test_the_recommendation_summary_says_when_an_action_is_not_recommended():
    from causal_engine.models.schemas import InterventionRecommendation

    good = InterventionRecommendation(id="train", target_variable="x", expected_effect=0.1, cost=1.0, roi=0.1, rank=1, equalized_odds_pass=True)
    bad = InterventionRecommendation(id="cut", target_variable="y", expected_effect=-0.1, cost=1.0, roi=-0.1, rank=2, recommended=False)

    text = crew_module.summarize_recommendations([good, bad])

    assert "#1 train (recommended" in text and "equalized_odds_pass=True" in text
    assert "#2 cut (NOT recommended (would raise the outcome)" in text


def test_the_recommendation_summary_says_why_an_unsupported_action_is_not_recommended():
    from causal_engine.models.schemas import InterventionRecommendation

    rec = InterventionRecommendation(
        id="agents", target_variable="x", expected_effect=0.01, cost=1.0, roi=0.01, rank=2, recommended=False,
        not_recommended_reason="effect_not_distinguishable_from_noise",
    )

    assert "NOT recommended (its effect failed the noise check)" in crew_module.summarize_recommendations([rec])
