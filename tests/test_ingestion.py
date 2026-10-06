"""Stage 1: Ingestion (causal_engine/pipeline/ingestion.py)."""

from __future__ import annotations

import copy

import pytest

from causal_engine.pipeline import ingestion


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


def test_ingest_rejects_a_non_binary_outcome(domain_config, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("employee_id,attrition\n1,0\n2,1\n3,2\n4,0.5\n", encoding="utf-8")
    config = copy.deepcopy(domain_config)
    config["ingestion"].update(id_column="employee_id", outcome_column="attrition")
    with pytest.raises(ValueError, match=r"must be 0/1, found \[0.5, 2.0\]"):
        ingestion.ingest(csv, config)
