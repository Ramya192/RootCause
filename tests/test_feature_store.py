"""Stage 2: Feature Store (rootcause/pipeline/feature_store.py).

Exercises the real Feast round-trip (parquet offline source -> materialize
-> online read) via the session-scoped `feature_df` fixture -- slower than a
pure-function unit test, but this stage's whole job is that plumbing, so a
mock would test nothing real.
"""

from __future__ import annotations


def test_feature_vectors_cover_every_entity(raw_df, feature_df, domain_config):
    entity_id_col = domain_config["feature_store"]["entity_id_column"]
    assert len(feature_df) == len(raw_df)
    assert set(feature_df[entity_id_col]) == set(raw_df[entity_id_col])


def test_feature_vectors_contain_configured_columns(feature_df, domain_config):
    feature_cols = domain_config["feature_store"]["feature_columns"]
    outcome_col = domain_config["ingestion"]["outcome_column"]
    for col in feature_cols + [outcome_col]:
        assert col in feature_df.columns
        assert feature_df[col].notna().all()
