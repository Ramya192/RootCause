"""One real CrewAI run, to show the manager LLM delegates every stage to the right agent.

The test suite never calls `crew.kickoff()` (it makes many paid LLM calls), so it can only check the
crew's wiring. This script is the manual check of the part the tests cannot reach: that the manager
actually hands each of the seven tasks to its agent, and that every stage tool ran.

COSTS REAL MONEY and takes several minutes. It reads OPENAI_API_KEY from the environment or .env and
refuses to start without --yes. Use the smallest domain (the default) for the first run.

    PYTHONPATH=. .venv/Scripts/python.exe scripts/reports/run_crew_once.py --yes
    PYTHONPATH=. .venv/Scripts/python.exe scripts/reports/run_crew_once.py --yes --domain german_credit

Prints the stages that ran, the narrative tier, the token usage CrewAI reports and the wall time, and
writes crew_run.json (no row-level data) next to the other evaluation outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from causal_engine.agents import crew as crew_module
from causal_engine.models.schemas import PipelineResult
from causal_engine.pipeline import runner
from causal_engine.utils.config_loader import ConfigLoader

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--domain", default="employee_attrition")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--yes", action="store_true", help="confirm that this makes paid OpenAI calls")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "evaluation" / "crew_run.json")
    args = parser.parse_args()

    load_dotenv()
    if not args.yes:
        print("This runs the real CrewAI manager and makes many paid OpenAI calls (several minutes). Add --yes to proceed.")
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (environment or .env); the manager agent is an LLM.")
        return 2

    domain = ConfigLoader().get_domain(args.domain)
    dataset = runner.resolve_dataset(domain.extra, args.dataset)
    data_path = runner.resolve_data_path(domain.extra, dataset)
    config = runner.with_dataset(domain.extra, dataset)
    manager_llm = config.get("explanation", {}).get("llm_model", "gpt-4o-mini")

    crew, run = crew_module.build_crew(config, str(data_path), manager_llm)
    print(f"Running the crew on {args.domain}/{dataset} with manager {manager_llm} ...")
    started = time.monotonic()
    crew.kickoff()
    seconds = time.monotonic() - started

    missing = crew_module.missing_stages(run)
    usage = getattr(crew, "usage_metrics", None)
    summary = {
        "domain": args.domain,
        "dataset": dataset,
        "manager_llm": manager_llm,
        "seconds": round(seconds, 1),
        "stages_run": [s for s in crew_module._STAGE_ORDER if s not in missing],
        "stages_missing": missing,
        "token_usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
    }
    if not missing:
        result = PipelineResult(
            domain_id=args.domain,
            causal_graph=run.causal_graph,
            effect_estimates=run.effect_estimates,
            counterfactuals=run.counterfactual_results,
            recommendations=run.recommendations,
            explanation=run.explanation_result,
        )
        summary["narrative_tier"] = result.explanation.narrative_tier
        summary["edges"] = len(result.causal_graph.edges)
        summary["recommendations"] = len(result.recommendations)

    print(json.dumps(summary, indent=2, default=str))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.out}", file=sys.stderr)
    if missing:
        print(f"FAILED: the manager skipped or mis-delegated: {', '.join(missing)}", file=sys.stderr)
        return 1
    print("OK: every stage tool ran, in a real manager-delegated run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
