"""Stage 1: Data Ingestion.

For the employee_attrition vertical slice this is CSV-only (per the spec's
own guardrail: image/audio ingestion is only needed for the insurance_claims
domain). Parses the file, checks the id/outcome columns the domain config
declares actually exist, and coerces the outcome to numeric 0/1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ingest(data_path: str | Path, domain_config: dict) -> pd.DataFrame:
    ingestion_cfg = domain_config["ingestion"]
    if ingestion_cfg["file_type"] != "csv":
        raise NotImplementedError(
            f"file_type={ingestion_cfg['file_type']!r} not supported in this "
            "vertical slice; only 'csv' is implemented"
        )

    df = pd.read_csv(data_path)

    id_col = ingestion_cfg["id_column"]
    outcome_col = ingestion_cfg["outcome_column"]
    missing = [c for c in (id_col, outcome_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{data_path}: missing required column(s) {missing}")

    df[outcome_col] = pd.to_numeric(df[outcome_col]).astype(int)
    return df
