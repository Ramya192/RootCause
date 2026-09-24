"""Stage 1: Ingestion (rootcause/pipeline/ingestion.py)."""

from __future__ import annotations

import copy

import pytest

from rootcause.pipeline import ingestion


def test_ingest_real_csv_shape_and_dtype(raw_df):
    assert len(raw_df) == 2000
    assert "employee_id" in raw_df.columns
    assert "attrition" in raw_df.columns
    assert set(raw_df["attrition"].unique()).issubset({0, 1})
    assert raw_df["attrition"].dtype.kind == "i"


def test_ingest_missing_required_column_raises(domain_config, data_path):
    bad_config = copy.deepcopy(domain_config)
    bad_config["ingestion"]["id_column"] = "not_a_real_column"
    with pytest.raises(ValueError, match="missing required column"):
        ingestion.ingest(data_path, bad_config)


def test_ingest_unsupported_file_type_raises(domain_config, data_path):
    bad_config = copy.deepcopy(domain_config)
    bad_config["ingestion"]["file_type"] = "pdf"
    with pytest.raises(NotImplementedError, match="not supported"):
        ingestion.ingest(data_path, bad_config)
