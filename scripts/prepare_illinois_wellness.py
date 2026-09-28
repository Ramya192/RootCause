"""Build data/illinois_wellness/wellness.csv from the public Illinois Workplace Wellness data.

Source: Jones, Molitor & Reif (2019), "What do Workplace Wellness Programs do?
Evidence from the Illinois Workplace Wellness Study", QJE. CC0 public-use data,
https://github.com/reifjulian/illinois-wellness-data (file data/csv/firm_admin.csv,
kept next to the output as firm_admin.csv). Its terms forbid using it to investigate
specific research subjects.

The study randomized 4,834 enrolled employees (3,300 treated, 1,534 control) to a
workplace wellness program. This script keeps ONLY what a causal analysis of "does
the program change termination?" may use:

  treat                    randomized assignment (the treatment)
  terminated_0119          not employed as of January 2019 (the outcome; fully observed)
  male, age50, age37_49,   pre-treatment covariates (age50/age37_49 are dummies; the
  white, sickleave_0815_0716,  omitted group is under 37)
  gym_0815_0716, prod_index_yr0

and drops everything measured AFTER assignment (gym_0816_*, hra_c_yr1, prod_index_yr1/2,
sick leave after Aug 2016, terminated_0717, promotions): adjusting for those would bias
the estimate. The salary/title columns are entirely censored in the public file. The
six public files cannot be linked to each other, so only firm_admin is used.

    python scripts/prepare_illinois_wellness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "illinois_wellness"

TREATMENT = "treat"
OUTCOME = "terminated_0119"
COVARIATES = [
    "male",
    "age50",
    "age37_49",
    "white",
    "sickleave_0815_0716",
    "gym_0815_0716",
    "prod_index_yr0",
]
COLUMNS = ["employee_id", TREATMENT, *COVARIATES, OUTCOME]


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[[TREATMENT, *COVARIATES, OUTCOME]].copy()
    df.insert(0, "employee_id", range(1, len(df) + 1))  # the file has no id; row order is the id
    return df[COLUMNS]


def main() -> None:
    source = DATA_DIR / "firm_admin.csv"
    if not source.exists():
        sys.exit(
            f"{source} not found. Download data/csv/firm_admin.csv from "
            "https://github.com/reifjulian/illinois-wellness-data and save it there."
        )
    out = prepare(pd.read_csv(source))
    path = DATA_DIR / "wellness.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} rows ({int(out[TREATMENT].sum())} treated) to {path}")


if __name__ == "__main__":
    main()
