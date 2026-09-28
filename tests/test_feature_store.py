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


def test_feature_vector_columns_follow_the_config_order(feature_df, domain_config):
    """Feast returns columns in a per-process order; downstream models are sensitive to it."""
    expected = [
        domain_config["feature_store"]["entity_id_column"],
        *domain_config["feature_store"]["feature_columns"],
        domain_config["ingestion"]["outcome_column"],
    ]
    assert list(feature_df.columns) == expected


def test_feature_vectors_contain_configured_columns(feature_df, domain_config):
    feature_cols = domain_config["feature_store"]["feature_columns"]
    outcome_col = domain_config["ingestion"]["outcome_column"]
    for col in feature_cols + [outcome_col]:
        assert col in feature_df.columns
        assert feature_df[col].notna().all()


def test_second_domain_gets_isolated_feast_state(
    raw_df, feature_df, domain_config, isolated_feast_root
):
    """A different domain (own id, entity, columns) must not collide with
    employee_attrition's parquet/registry/online store, nor disturb its data."""
    import copy

    import pandas as pd

    from rootcause.feature_repo import definitions
    from rootcause.pipeline import feature_store

    other = copy.deepcopy(domain_config)
    other["domain"]["id"] = "test_second_domain"
    other["feature_store"].update(
        entity="worker", entity_id_column="worker_id", feature_columns=["a", "b"]
    )
    other["ingestion"]["outcome_column"] = "left"
    other_df = pd.DataFrame(
        {"worker_id": [1, 2, 3, 4], "a": [0.1, 0.2, 0.3, 0.4], "b": [1.0, 2.0, 3.0, 4.0],
         "left": [0, 1, 0, 1]}
    )

    other_dir = definitions.domain_data_dir(other)
    out = feature_store.build_feature_vectors(other_df, other)
    assert list(out["worker_id"]) == [1, 2, 3, 4]
    assert set(out.columns) == {"worker_id", "a", "b", "left"}
    assert other_dir != definitions.domain_data_dir(domain_config)
    assert (other_dir / "registry.db").exists()

    # the original domain still serves its own features after the other ran
    again = feature_store.build_feature_vectors(raw_df, domain_config)
    assert len(again) == len(feature_df)


def test_same_domain_different_datasets_do_not_share_feast_state(
    domain_config, isolated_feast_root
):
    """The collision item 4 exists to prevent: real and synthetic data of ONE
    domain (same schema, same entity ids) must each keep their own features."""
    import copy

    import pandas as pd

    from rootcause.feature_repo import definitions
    from rootcause.pipeline import feature_store

    cfg = copy.deepcopy(domain_config)
    cfg["feature_store"].update(feature_columns=["a"])
    cfg["ingestion"]["outcome_column"] = "left"
    cfg["feature_store"]["entity_id_column"] = "worker_id"
    cfg["feature_store"]["entity"] = "worker"

    def frame(values):
        return pd.DataFrame({"worker_id": [1, 2, 3], "a": values, "left": [0, 1, 0]})

    real_cfg = {**cfg, "run": {"dataset": "real"}}
    synth_cfg = {**cfg, "run": {"dataset": "synthetic"}}

    feature_store.build_feature_vectors(frame([1.0, 2.0, 3.0]), real_cfg)
    feature_store.build_feature_vectors(frame([100.0, 200.0, 300.0]), synth_cfg)

    assert definitions.domain_data_dir(real_cfg) != definitions.domain_data_dir(synth_cfg)
    # re-reading the real dataset's store must still give the REAL values
    again = feature_store.build_feature_vectors(frame([1.0, 2.0, 3.0]), real_cfg)
    assert again.sort_values("worker_id")["a"].tolist() == [1.0, 2.0, 3.0]
    stored = pd.read_parquet(definitions.domain_data_dir(synth_cfg) / "features.parquet")
    assert stored.sort_values("worker_id")["a"].tolist() == [100.0, 200.0, 300.0]


def test_online_read_is_chunked_and_keeps_row_order(domain_config, isolated_feast_root, monkeypatch):
    """SQLite caps the variables per query (32,766), so a ~33,000-entity file failed with
    'too many SQL variables' until the online read was split into chunks. Chunking must not
    reorder or lose rows."""
    import copy

    import pandas as pd

    from rootcause.pipeline import feature_store

    monkeypatch.setattr(feature_store, "ONLINE_READ_CHUNK", 3)
    cfg = copy.deepcopy(domain_config)
    cfg["feature_store"].update(feature_columns=["a"], entity="worker", entity_id_column="worker_id")
    cfg["ingestion"]["outcome_column"] = "left"
    ids = [9, 2, 7, 4, 5, 1, 8]  # seven rows over three chunks, deliberately not sorted
    df = pd.DataFrame({"worker_id": ids, "a": [i * 10.0 for i in ids], "left": [i % 2 for i in ids]})

    out = feature_store.build_feature_vectors(df, cfg)

    assert out["worker_id"].tolist() == ids
    assert out["a"].tolist() == [i * 10.0 for i in ids]
    assert out["left"].tolist() == [i % 2 for i in ids]
