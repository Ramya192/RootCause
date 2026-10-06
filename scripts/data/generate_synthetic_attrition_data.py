"""Generate the synthetic employee-attrition dataset with a KNOWN causal DAG,
so causal discovery (stage 3) and effect estimation (stage 4) have ground
truth to be checked against instead of unverifiable real-world data.

The data-generating process itself lives in causal_engine/evaluation/scms.py
(`attrition_scm`), which is also what the evaluation harness intervenes on to
compute true effects -- one definition, so the CSV and the ground truth can't
drift apart. See that module for the structural equations.

Ground-truth edges (parent -> child):
    compensation      -> job_satisfaction
    manager_quality   -> job_satisfaction
    manager_quality   -> burnout
    workload          -> burnout
    job_satisfaction  -> attrition
    burnout           -> attrition
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # so `python scripts/<group>/<script>.py` finds the causal_engine package

from causal_engine.evaluation.scms import N_ROWS, SEED, attrition_scm  # noqa: E402

GROUND_TRUTH_EFFECTS = {
    # direct structural coefficients (NOT total effects -- for those see
    # `python -m causal_engine.evaluation`, which simulates do() on the SCM)
    "job_satisfaction->attrition": -1.2,
    "burnout->attrition": 1.0,
}

VARIABLES = [
    "employee_id",
    "compensation",
    "manager_quality",
    "workload",
    "job_satisfaction",
    "burnout",
    "gender",
    "attrition",
]


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    df = attrition_scm().sample(n_rows, seed)
    df.insert(0, "employee_id", range(1, n_rows + 1))
    return df[VARIABLES]


def main() -> None:
    out_dir = REPO_ROOT / "data" / "employee_attrition"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate()
    csv_path = out_dir / "attrition.csv"
    df.to_csv(csv_path, index=False)

    ground_truth = {
        "edges": attrition_scm().edges(),
        "effects": GROUND_TRUTH_EFFECTS,
        "seed": SEED,
        "n_rows": N_ROWS,
    }
    gt_path = out_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    print(f"Wrote {len(df)} rows to {csv_path}")
    print(f"Wrote ground truth to {gt_path}")


if __name__ == "__main__":
    main()
