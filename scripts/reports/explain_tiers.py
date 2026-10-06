"""Run Stage 7 for one domain and show which narrative tier won and why.

Needs OPENAI_API_KEY (read from the environment or .env) to exercise the AutoGen and
single-call tiers; without one both are skipped and the template is used. Costs a few
cents at most: the AutoGen chain is exactly three model turns.

    PYTHONPATH=. .venv/Scripts/python.exe scripts/reports/explain_tiers.py --domain german_credit
    PYTHONPATH=. .venv/Scripts/python.exe scripts/reports/explain_tiers.py --domain carclaims --repeat 5
    PYTHONPATH=. .venv/Scripts/python.exe scripts/reports/explain_tiers.py --domain carclaims --repeat 10 --compare

`--compare` runs each LLM tier ON ITS OWN (the other is switched off, the template stays as the
last resort) and counts how often its text passed the grounding check, so the tiers can be
compared instead of only seeing the one that happened to win.
"""

from __future__ import annotations

import argparse
from collections import Counter

from dotenv import load_dotenv

from causal_engine.pipeline import explanation, runner
from causal_engine.utils.config_loader import ConfigLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--domain", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--repeat", type=int, default=1, help="Stage 7 runs (Stages 1-6 run once)")
    parser.add_argument("--compare", action="store_true", help="run each LLM tier alone, --repeat times each")
    args = parser.parse_args()

    load_dotenv()
    cfg = ConfigLoader().get_domain(args.domain).extra
    dataset = runner.resolve_dataset(cfg, args.dataset)
    cfg = runner.with_dataset(cfg, dataset)
    out = runner.run_analysis_stages(cfg, runner.resolve_data_path(cfg, dataset))

    def run_stage7(config):
        return explanation.generate_explanation(
            out.features, out.effects, out.counterfactuals, out.recommendations, config
        )

    if args.compare:
        for tier in ("autogen", "llm"):
            solo = {**cfg, "explanation": {**cfg["explanation"], "tiers": [tier]}}
            outcomes: Counter = Counter()
            reasons: Counter = Counter()
            for _ in range(args.repeat):
                attempt = run_stage7(solo).narrative_log[0]
                outcomes[attempt.outcome] += 1
                if attempt.outcome != "used":
                    reasons[attempt.detail.split(":")[0] if attempt.outcome == "rejected" else attempt.outcome] += 1
            print(f"{args.domain}/{dataset} {tier:8s} alone: {dict(outcomes)} of {args.repeat}; "
                  f"rejection kinds {dict(reasons)}")
        return

    used = Counter()
    for i in range(args.repeat):
        result = run_stage7(cfg)
        used[result.narrative_tier] += 1
        print(f"\n--- run {i + 1}: tier used = {result.narrative_tier}")
        for a in result.narrative_log:
            print(f"    {a.tier:9s} {a.outcome:9s} {a.detail}")
        print(result.narrative)
    print(f"\ntier counts over {args.repeat} run(s): {dict(used)}")


if __name__ == "__main__":
    main()
