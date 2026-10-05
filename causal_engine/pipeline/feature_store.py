"""Stage 2: Feature Store.

Encodes categoricals and handles missing values (preprocessing.py), then
writes the ingested rows to the domain's own Feast offline (parquet) source,
applies the entity/feature-view definitions built from the domain config,
materializes into that domain's local SQLite online store, then retrieves
each entity's causal feature vector back out via Feast's online-serving
path -- the same path a production deployment would use, just backed by
local files instead of a managed offline/online store.

Each (domain, dataset) pair gets its own directory under
causal_engine/feature_repo/data/, so different domains -- or the real and
synthetic data of one domain -- never collide. Two runs of the SAME pair
still share that directory and must not overlap (the API's job queue
enforces this).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from feast import FeatureStore

from causal_engine.feature_repo import definitions
from causal_engine.pipeline import preprocessing


ONLINE_READ_CHUNK = 10_000  # entities per online-store read (tests shrink it)


def build_feature_vectors(df: pd.DataFrame, domain_config: dict) -> pd.DataFrame:
    entity_id_col = domain_config["feature_store"]["entity_id_column"]
    outcome_col = domain_config["ingestion"]["outcome_column"]

    prepared = preprocessing.preprocess(df, domain_config)
    feature_cols = prepared.feature_columns

    repo = definitions.build_domain_repo(domain_config, feature_cols)
    repo.parquet_path.parent.mkdir(parents=True, exist_ok=True)
    materialize_df = prepared.df.copy()
    materialize_df["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    materialize_df.to_parquet(repo.parquet_path, index=False)

    store = FeatureStore(config=repo.repo_config)
    store.apply([repo.entity, repo.feature_view])
    store.materialize_incremental(end_date=datetime.now(timezone.utc))

    entity_ids = [int(v) for v in prepared.df[entity_id_col].tolist()]
    view_name = definitions.feature_view_name(domain_config)
    feature_refs = [f"{view_name}:{col}" for col in feature_cols + [outcome_col]]
    # One online read per chunk: SQLite caps the variables in a single query (32,766), and Feast
    # binds one per entity, so a 33,000-loan file failed with "too many SQL variables".
    parts = [
        store.get_online_features(
            features=feature_refs,
            entity_rows=[{entity_id_col: v} for v in entity_ids[start:start + ONLINE_READ_CHUNK]],
        ).to_df()
        for start in range(0, len(entity_ids), ONLINE_READ_CHUNK)
    ]
    online_resp = pd.concat(parts, ignore_index=True)
    # Feast returns columns in an order that varies from process to process (it is
    # set-derived, so it follows PYTHONHASHSEED). Column order is not harmless: a
    # gradient-boosted model draws its per-split feature permutation by position, so
    # SHAP importances changed between runs. Pin it to the config's order.
    features = online_resp[[entity_id_col, *feature_cols, outcome_col]]
    features.attrs["preprocessing_report"] = prepared.report

    return features
