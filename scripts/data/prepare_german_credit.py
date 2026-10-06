"""Build data/german_credit/credit.csv from the raw Statlog German Credit file.

Source: Hofmann, H. (1994), Statlog (German Credit Data), UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data  (kept next to the
raw/ as german.data, with the codebook german.doc). 1,000 loan applicants at a German
bank, 700 repaid ("good") and 300 did not ("bad"); observational, and the outcome is only
known for loans that were granted.

The raw file is space-separated with no header and coded categories (A11, A34, ...). This
script decodes them with german.doc into readable levels, and makes these choices, all of
them visible in the output:

  * credit_amount is rescaled to thousands of DM (credit_amount_kdm) so that "one unit" is a
    meaningful change rather than one Deutschmark.
  * personal status and sex is ONE coded attribute; only `female` (code A92) is recoverable
    from it, and the file has no single women. It is not a clean sex variable.
  * purpose: the four rarest categories (domestic appliances, repairs, retraining, others;
    55 applicants together) are grouped as `other`.
  * young = age < 25, the usual cut for the age-fairness question on this dataset. It is
    the sensitive attribute Stage 6 checks and is derived from `age`.
  * default = 1 for "bad" (the outcome; the raw class is 1 good / 2 bad).

    python scripts/data/prepare_german_credit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "german_credit"

RAW_COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_plans", "housing", "n_existing_credits",
    "job", "n_dependents", "telephone", "foreign_worker", "raw_class",
]

DECODE = {
    "checking_status": {"A11": "lt_0", "A12": "0_to_200", "A13": "ge_200", "A14": "none"},
    "credit_history": {
        "A30": "no_credits_or_all_paid", "A31": "all_paid_this_bank", "A32": "existing_paid",
        "A33": "past_delay", "A34": "critical_or_other_credits",
    },
    "purpose": {
        "A40": "car_new", "A41": "car_used", "A42": "furniture", "A43": "radio_tv",
        "A44": "other", "A45": "other", "A46": "education", "A48": "other", "A49": "business",
        "A410": "other",
    },
    "savings": {"A61": "lt_100", "A62": "100_to_500", "A63": "500_to_1000", "A64": "ge_1000", "A65": "unknown"},
    "employment_since": {"A71": "unemployed", "A72": "lt_1y", "A73": "1_to_4y", "A74": "4_to_7y", "A75": "ge_7y"},
    "other_debtors": {"A101": "none", "A102": "co_applicant", "A103": "guarantor"},
    "property": {"A121": "real_estate", "A122": "savings_or_insurance", "A123": "car_or_other", "A124": "unknown"},
    "other_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "free"},
    "job": {"A171": "unskilled_nonresident", "A172": "unskilled_resident", "A173": "skilled", "A174": "management"},
}

OUTCOME = "default"
COLUMNS = [
    "applicant_id", "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount_kdm", "savings", "employment_since", "installment_rate", "female",
    "other_debtors", "residence_since", "property", "age", "young", "other_plans", "housing",
    "n_existing_credits", "job", "n_dependents", "telephone", "foreign_worker", OUTCOME,
]


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError(f"expected columns {RAW_COLUMNS}, got {list(raw.columns)}")
    df = pd.DataFrame(index=raw.index)
    for column, mapping in DECODE.items():
        unknown = sorted(set(raw[column]) - set(mapping))
        if unknown:
            raise ValueError(f"{column}: codes {unknown} are not in the codebook")
        df[column] = raw[column].map(mapping)

    for column in ("duration_months", "installment_rate", "residence_since", "age", "n_existing_credits", "n_dependents"):
        df[column] = raw[column].astype(int)
    df["credit_amount_kdm"] = (raw["credit_amount"] / 1000.0).round(3)
    df["female"] = (raw["personal_status"] == "A92").astype(int)
    df["young"] = (raw["age"] < 25).astype(int)
    df["telephone"] = (raw["telephone"] == "A192").astype(int)
    df["foreign_worker"] = (raw["foreign_worker"] == "A201").astype(int)
    df[OUTCOME] = (raw["raw_class"] == 2).astype(int)
    df.insert(0, "applicant_id", range(1, len(df) + 1))  # the file has no id; row order is the id
    return df[COLUMNS]


def main() -> None:
    source = DATA_DIR / "raw" / "german.data"
    if not source.exists():
        sys.exit(
            f"{source} not found. Download the zip from "
            "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip "
            "and put german.data (and german.doc) in that raw/ folder."
        )
    raw = pd.read_csv(source, sep=" ", header=None, names=RAW_COLUMNS)
    out = prepare(raw)
    path = DATA_DIR / "credit.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} rows ({int(out[OUTCOME].sum())} defaults, {out[OUTCOME].mean():.1%}) to {path}")


if __name__ == "__main__":
    main()
