"""Generate data/freddie_mac/loans_2007_semi_synthetic.csv: the real 2007 Freddie Mac loans with a
SIMULATED default outcome whose lever effects are planted.

Same columns as data/freddie_mac/loans_2007.csv except `loan_sequence` (a real Freddie Mac
identifier, not reproduced). The data-generating process lives in causal_engine/evaluation/scms.py
(`freddie_mac_semi_synthetic_scm`), which is also what the evaluation harness intervenes on to
compute the true effects, so the CSV and its ground truth cannot drift apart. Needs
loans_2007.csv, which scripts/prepare_freddie_mac.py builds from the registered Freddie Mac
download; both files are gitignored (see data/freddie_mac/README.md).

    python scripts/generate_semi_synthetic_freddie_mac_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `python scripts/...` finds the causal_engine package

from causal_engine.evaluation.scms import (  # noqa: E402
    FREDDIE_ID_COLUMN,
    FREDDIE_N_ROWS,
    FREDDIE_SEED,
    freddie_mac_semi_synthetic_scm,
)


def generate(n_rows: int = FREDDIE_N_ROWS, seed: int = FREDDIE_SEED) -> pd.DataFrame:
    df = freddie_mac_semi_synthetic_scm().sample(n_rows, seed)
    df.insert(0, FREDDIE_ID_COLUMN, range(1, n_rows + 1))
    return df


def main() -> None:
    path = REPO_ROOT / "data" / "freddie_mac" / "loans_2007_semi_synthetic.csv"
    df = generate()
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows ({df['default'].mean():.1%} default) to {path}")


if __name__ == "__main__":
    main()
