"""Generate data/illinois_wellness/wellness_synthetic.csv: a fully synthetic mirror of the
Illinois Workplace Wellness schema with a PLANTED treatment effect and a known graph.

The data-generating process lives in rootcause/evaluation/scms.py (`illinois_wellness_scm`),
which is also what the evaluation harness intervenes on to compute the true effect, so the
CSV and its ground truth cannot drift apart (tests/test_evaluation_scm.py checks this).

    python scripts/generate_synthetic_wellness_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `python scripts/...` finds the rootcause package

from rootcause.evaluation.scms import (  # noqa: E402
    WELLNESS_N_ROWS,
    WELLNESS_SEED,
    illinois_wellness_scm,
)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from prepare_illinois_wellness import COLUMNS  # noqa: E402  (same columns, same order as the real file)


def generate(n_rows: int = WELLNESS_N_ROWS, seed: int = WELLNESS_SEED) -> pd.DataFrame:
    df = illinois_wellness_scm().sample(n_rows, seed)
    df.insert(0, "employee_id", range(1, n_rows + 1))
    return df[COLUMNS]


def main() -> None:
    path = REPO_ROOT / "data" / "illinois_wellness" / "wellness_synthetic.csv"
    df = generate()
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
