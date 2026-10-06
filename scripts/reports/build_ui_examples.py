"""Run the pipeline on a few (domain, dataset) pairs and save each result as a JSON sample for the web UI.

    python scripts/reports/build_ui_examples.py
    python scripts/reports/build_ui_examples.py --only german_credit/real

The UI (causal_engine/api/ui.html) loads these so it works instantly with no server round trip. Each
file holds the PipelineResult plus the domain facts the page needs to explain it (what the outcome
and levers are, whether the data are observational) and, where a structural causal model of the
domain exists, the TRUE graph and effects so the page can show recovery against the truth.

Samples are single runs of the direct pipeline and contain only aggregate outputs (graph, effect
estimates, rankings, narrative), never loan-level or person-level rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from causal_engine.evaluation import harness  # noqa: E402
from causal_engine.evaluation.scms import SCM_REGISTRY  # noqa: E402
from causal_engine.pipeline import runner  # noqa: E402
from causal_engine.utils.config_loader import ConfigLoader  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "ui_examples"

# (domain, dataset, label shown in the UI)
SAMPLES = [
    ("employee_attrition", "synthetic", "Known-answer simulation"),
    ("german_credit", "real", "Real credit data"),
    ("freddie_mac", "real_2019", "Mortgages 2019, 90+ day outcome"),
    ("freddie_mac", "real_2019_relief_adjusted", "Mortgages 2019, forbearance excluded"),
    ("carclaims", "real", "Real insurance claims"),
]


def _meta(domain, dataset: str, label: str, path: Path) -> dict:
    cfg = domain.extra
    meta = {
        "domain_id": domain.id,
        "domain_name": domain.name,
        "description": " ".join(str(domain.description).split()),
        "dataset": dataset,
        "label": label,
        "n_rows": int(len(pd.read_csv(path, usecols=[cfg["ingestion"]["id_column"]]))),
        "outcome": cfg["effect_estimation"]["outcome"],
        "treatments": [t["name"] for t in cfg["effect_estimation"]["treatments"]],
        "sensitive_attribute": cfg["interventions"].get("sensitive_attribute"),
        "observational": bool(cfg.get("explanation", {}).get("observational", False)),
        "entity_noun_plural": cfg["domain"].get("entity_noun_plural", "records"),
        "built": date.today().isoformat(),
    }
    factory = SCM_REGISTRY.get((domain.id, dataset))
    if factory:  # the data come from a known structural causal model, so the answer is known
        truth = harness.compute_truth(factory(), cfg)
        meta["truth"] = {
            "edges": [list(e) for e in (truth.edges or [])],
            "effects": truth.effects,
            "fairness_ratio": truth.fairness_ratio,
        }
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", help="domain/dataset to rebuild (repeatable)")
    args = parser.parse_args()

    loader = ConfigLoader()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for domain_id, dataset, label in SAMPLES:
        key = f"{domain_id}/{dataset}"
        stem = f"{domain_id}__{dataset}"
        target = OUT_DIR / f"{stem}.json"
        if args.only and key not in args.only:
            if target.exists():
                index.append(json.loads(target.read_text(encoding="utf-8"))["meta"])
            continue
        domain = loader.get_domain(domain_id)
        path = runner.resolve_data_path(domain.extra, dataset)
        if not path.is_file():
            print(f"skip {key}: {path} not found")
            continue
        print(f"running {key} ...", flush=True)
        result = runner.run_direct(domain_id, runner.with_dataset(domain.extra, dataset), str(path))
        meta = _meta(domain, dataset, label, path)
        target.write_text(
            json.dumps({"meta": meta, "result": json.loads(result.model_dump_json())}, indent=1),
            encoding="utf-8",
        )
        index.append(meta)
        print(f"  wrote {target.relative_to(REPO_ROOT)} (narrative tier: {result.explanation.narrative_tier})")
    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
