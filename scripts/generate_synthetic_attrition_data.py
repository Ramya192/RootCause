"""Generate a semi-synthetic employee-attrition dataset with a KNOWN causal
DAG, so causal discovery (stage 3) and effect estimation (stage 4) have
ground truth to be checked against instead of unverifiable real-world data.

Structural equations (each variable is a noisy function of its parents):

    compensation        ~ N(0, 1)                                   (root)
    manager_quality      ~ N(0, 1)                                   (root)
    workload              ~ N(0, 1)                                   (root)
    job_satisfaction     = 0.6*compensation + 0.5*manager_quality + noise
    burnout                = 0.7*workload - 0.2*manager_quality + noise
    attrition (prob)     = sigmoid(-1.2*job_satisfaction + 1.0*burnout + noise)

Ground-truth edges (parent -> child):
    compensation      -> job_satisfaction
    manager_quality    -> job_satisfaction
    manager_quality    -> burnout
    workload            -> burnout
    job_satisfaction   -> attrition
    burnout              -> attrition
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 2000

GROUND_TRUTH_EDGES = [
    ("compensation", "job_satisfaction"),
    ("manager_quality", "job_satisfaction"),
    ("manager_quality", "burnout"),
    ("workload", "burnout"),
    ("job_satisfaction", "attrition"),
    ("burnout", "attrition"),
]

GROUND_TRUTH_EFFECTS = {
    # approximate true coefficients, for sanity-checking stage 4 ATE estimates
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    compensation = rng.normal(0, 1, n_rows)
    manager_quality = rng.normal(0, 1, n_rows)
    workload = rng.normal(0, 1, n_rows)

    # sensitive attribute for the fairness check in stage 6 -- deliberately
    # NOT wired into any structural equation below, so a fair pipeline
    # should find no causal effect of gender on attrition.
    gender = rng.choice(["A", "B"], size=n_rows)

    job_satisfaction = (
        0.6 * compensation + 0.5 * manager_quality + rng.normal(0, 0.5, n_rows)
    )
    burnout = 0.7 * workload - 0.2 * manager_quality + rng.normal(0, 0.5, n_rows)

    attrition_logit = (
        -1.2 * job_satisfaction + 1.0 * burnout + rng.normal(0, 0.3, n_rows)
    )
    attrition_prob = sigmoid(attrition_logit)
    attrition = (rng.uniform(0, 1, n_rows) < attrition_prob).astype(int)

    df = pd.DataFrame(
        {
            "employee_id": np.arange(1, n_rows + 1),
            "compensation": compensation,
            "manager_quality": manager_quality,
            "workload": workload,
            "job_satisfaction": job_satisfaction,
            "burnout": burnout,
            "gender": gender,
            "attrition": attrition,
        }
    )
    return df[VARIABLES]


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "data" / "employee_attrition"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate()
    csv_path = out_dir / "attrition.csv"
    df.to_csv(csv_path, index=False)

    ground_truth = {
        "edges": GROUND_TRUTH_EDGES,
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
