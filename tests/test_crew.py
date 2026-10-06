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
from crewai import Process

from causal_engine.agents.crew import _AGENT_SPECS, _STAGE_ORDER, _STAGE_TOOL_NAMES, build_crew, missing_stages


@pytest.fixture(scope="module")
def built_crew(domain_config, data_path):
    return build_crew(domain_config, str(data_path), manager_llm="gpt-4o-mini")


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
