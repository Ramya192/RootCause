"""Stage 1: Data Ingestion.

Reads the domain's CSV, checks the id/outcome columns the domain config
declares actually exist, and coerces the outcome to numeric 0/1 (anything else is an error).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def ingest(data_path: str | Path, domain_config: dict) -> pd.DataFrame:
    ingestion_cfg = domain_config["ingestion"]
    if ingestion_cfg["file_type"] != "csv":
        raise NotImplementedError(
            f"file_type={ingestion_cfg['file_type']!r} is not supported; only 'csv' is implemented"
        )

    df = pd.read_csv(data_path)

    id_col = ingestion_cfg["id_column"]
    outcome_col = ingestion_cfg["outcome_column"]
    missing = [c for c in (id_col, outcome_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{data_path}: missing required column(s) {missing}")

    if df[id_col].isna().any():
        raise ValueError(f"{data_path}: id column {id_col!r} has missing values")

    # A row with no observed outcome can't inform any causal estimate. Dropping
    # is only unbiased if the gaps are unrelated to treatment -- callers should
    # check df.attrs['rows_dropped_missing_outcome'] before trusting an estimate.
    outcome = pd.to_numeric(df[outcome_col])
    n_missing = int(outcome.isna().sum())
    if n_missing:
        logger.warning("dropping %d/%d rows with missing outcome %r", n_missing, len(df), outcome_col)
        df, outcome = df[outcome.notna()].reset_index(drop=True), outcome.dropna().reset_index(drop=True)
    not_binary = sorted(float(v) for v in set(outcome.unique()) - {0, 1})
    if not_binary:
        # Casting 0.5 or 2 to int would silently turn them into 0 or 2; every stage assumes a 0/1 outcome.
        raise ValueError(f"{data_path}: outcome column {outcome_col!r} must be 0/1, found {not_binary[:5]}")
    df[outcome_col] = outcome.astype(int)
    df.attrs["rows_dropped_missing_outcome"] = n_missing
    return df
