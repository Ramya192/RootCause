"""Generate data/german_credit/credit_semi_synthetic.csv: real German Credit applicants with a
SIMULATED default outcome whose lever effects are planted.

Same columns as the real data/german_credit/credit.csv. The data-generating process lives in
causal_engine/evaluation/scms.py (`german_credit_semi_synthetic_scm`), which is also what the
evaluation harness intervenes on to compute the true effects, so the CSV and its ground truth
cannot drift apart (tests/test_evaluation_credit.py checks this). Needs credit.csv, which
scripts/data/prepare_german_credit.py builds from the raw UCI file.

    python scripts/data/generate_semi_synthetic_credit_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # so `python scripts/<group>/<script>.py` finds the causal_engine package

from causal_engine.evaluation.scms import (  # noqa: E402
    CREDIT_ID_COLUMN,
    CREDIT_N_ROWS,
    CREDIT_SEED,
    german_credit_semi_synthetic_scm,
)

sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))
from prepare_german_credit import COLUMNS  # noqa: E402  (same columns, same order as the real file)


def generate(n_rows: int = CREDIT_N_ROWS, seed: int = CREDIT_SEED) -> pd.DataFrame:
    df = german_credit_semi_synthetic_scm().sample(n_rows, seed)
    df.insert(0, CREDIT_ID_COLUMN, range(1, n_rows + 1))
    return df[COLUMNS]


def main() -> None:
    path = REPO_ROOT / "data" / "german_credit" / "credit_semi_synthetic.csv"
    df = generate()
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows ({df['default'].mean():.1%} default) to {path}")


if __name__ == "__main__":
    main()
