"""Stage 2 preprocessing (rootcause/pipeline/preprocessing.py): categorical
encoding + missing-value handling, plus the missing-outcome rule in Stage 1
and a real Feast round-trip on a messy frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rootcause.pipeline import feature_store, ingestion, preprocessing


def make_config(**fs_overrides) -> dict:
    fs = {"entity": "person", "entity_id_column": "pid", "feature_columns": ["age"]}
    fs.update(fs_overrides)
    return {
        "domain": {"id": "test_preprocessing_domain"},
        "ingestion": {"file_type": "csv", "id_column": "pid", "outcome_column": "y"},
        "feature_store": fs,
    }


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pid": [1, 2, 3, 4],
            "age": [30.0, 40.0, np.nan, 50.0],
            "checking": ["A11", "A13", "A12", "A14"],
            "foreign": ["A201", "A202", "A201", "A201"],
            "housing": ["own", "rent", None, "own"],
            "y": [0, 1, 0, 1],
        }
    )


CATS = [
    {"name": "checking", "encoding": "ordinal", "order": ["A11", "A12", "A13", "A14"]},
    {"name": "foreign", "encoding": "binary", "positive": "A201"},
    {"name": "housing", "encoding": "onehot"},
]


def test_default_fails_loudly_on_missing_numeric():
    with pytest.raises(ValueError, match=r"missing values in feature column.*'age': 1"):
        preprocessing.preprocess(make_df(), make_config())


def test_median_impute_with_indicator():
    cfg = make_config(missing={"numeric": "median", "add_indicator": True})
    res = preprocessing.preprocess(make_df(), cfg)
    assert res.feature_columns == ["age", "age__missing"]
    assert res.df["age"].tolist() == [30.0, 40.0, 40.0, 50.0]  # median of 30/40/50
    assert res.df["age__missing"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert res.report["missing_counts"] == {"age": 1}


def test_mean_impute_no_indicator():
    res = preprocessing.preprocess(make_df(), make_config(missing={"numeric": "mean"}))
    assert res.feature_columns == ["age"]
    assert res.df.loc[2, "age"] == pytest.approx(40.0)


def test_drop_removes_rows_and_skips_indicator():
    cfg = make_config(missing={"numeric": "drop", "add_indicator": True})
    res = preprocessing.preprocess(make_df(), cfg)
    assert res.df["pid"].tolist() == [1, 2, 4]
    assert res.feature_columns == ["age"]
    assert res.report["rows_dropped_missing_features"] == 1
    assert list(res.df.index) == [0, 1, 2]


def test_ordinal_binary_onehot_encoding():
    cfg = make_config(
        feature_columns=[], categorical_columns=CATS, missing={"numeric": "median"}
    )
    res = preprocessing.preprocess(make_df(), cfg)
    assert res.df["checking"].tolist() == [0.0, 2.0, 1.0, 3.0]
    assert res.df["foreign"].tolist() == [1.0, 0.0, 1.0, 1.0]
    # levels: own (reference, dropped), rent, missing -- missing became its own level
    assert res.feature_columns == ["checking", "foreign", "housing__rent", "housing__missing"]
    assert res.df["housing__rent"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert res.df["housing__missing"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert res.report["encoded_from"]["housing"] == ["housing__rent", "housing__missing"]


def test_onehot_drop_first_false_keeps_every_level():
    cats = [{"name": "housing", "encoding": "onehot", "drop_first": False}]
    cfg = make_config(feature_columns=[], categorical_columns=cats)
    df = make_df().dropna(subset=["housing"])
    res = preprocessing.preprocess(df, cfg)
    assert res.feature_columns == ["housing__own", "housing__rent"]


def test_unknown_ordinal_value_raises():
    cats = [{"name": "checking", "encoding": "ordinal", "order": ["A11", "A12"]}]
    cfg = make_config(feature_columns=[], categorical_columns=cats)
    with pytest.raises(ValueError, match="not in order"):
        preprocessing.preprocess(make_df(), cfg)


def test_text_in_numeric_column_points_to_categorical_columns():
    cfg = make_config(feature_columns=["checking"])
    with pytest.raises(ValueError, match="categorical_columns"):
        preprocessing.preprocess(make_df(), cfg)


def test_bad_strategy_and_bad_encoding_rejected():
    with pytest.raises(ValueError, match="missing.numeric"):
        preprocessing.preprocess(make_df(), make_config(missing={"numeric": "magic"}))
    bad = [{"name": "housing", "encoding": "hash"}]
    with pytest.raises(ValueError, match="encoding"):
        preprocessing.preprocess(make_df(), make_config(categorical_columns=bad))


def test_ingestion_drops_rows_with_missing_outcome(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"pid": [1, 2, 3, 4], "y": [0, np.nan, 1, np.nan]}).to_csv(path, index=False)
    out = ingestion.ingest(path, make_config())
    assert out["pid"].tolist() == [1, 3]
    assert out["y"].tolist() == [0, 1]
    assert out["y"].dtype.kind == "i"
    assert out.attrs["rows_dropped_missing_outcome"] == 2


def test_ingestion_rejects_missing_id(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"pid": [1, np.nan], "y": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="id column"):
        ingestion.ingest(path, make_config())


def test_feast_round_trip_with_categoricals_and_missing(isolated_feast_root):
    cfg = make_config(
        categorical_columns=CATS, missing={"numeric": "median", "add_indicator": True}
    )
    out = feature_store.build_feature_vectors(make_df(), cfg)
    expected = ["age", "age__missing", "checking", "foreign", "housing__rent", "housing__missing"]
    assert set(out.columns) == {"pid", "y", *expected}
    assert out[expected].notna().all().all()
    assert out.attrs["preprocessing_report"]["missing_counts"] == {"age": 1}

    # different levels next run -> schema is rebuilt from the data, not stale
    out2 = feature_store.build_feature_vectors(make_df().assign(housing="own"), cfg)
    assert "housing__rent" not in out2.columns
