"""Stage 2: Feature Store.

Writes the ingested rows to the Feast feature repo's offline (parquet)
source, applies the entity/feature-view definitions, materializes into the
local SQLite online store, then retrieves each employee's causal feature
vector back out via Feast's online-serving path -- the same path a
production deployment would use, just backed by local files instead of a
managed offline/online store.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from feast import FeatureStore

from rootcause.feature_repo import definitions

REPO_PATH = definitions.DATA_PATH.parent.parent


def build_feature_vectors(df: pd.DataFrame, domain_config: dict) -> pd.DataFrame:
    fs_cfg = domain_config["feature_store"]
    entity_id_col = fs_cfg["entity_id_column"]
    feature_cols = fs_cfg["feature_columns"]
    outcome_col = domain_config["ingestion"]["outcome_column"]

    definitions.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    materialize_df = df[[entity_id_col] + feature_cols + [outcome_col]].copy()
    materialize_df["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    materialize_df.to_parquet(definitions.DATA_PATH, index=False)

    store = FeatureStore(repo_path=str(REPO_PATH))
    store.apply([definitions.employee, definitions.employee_causal_features])
    store.materialize_incremental(end_date=datetime.now(timezone.utc))

    entity_rows = [{entity_id_col: int(v)} for v in df[entity_id_col].tolist()]
    feature_refs = [
        f"employee_causal_features:{col}" for col in feature_cols + [outcome_col]
    ]
    online_resp = store.get_online_features(
        features=feature_refs, entity_rows=entity_rows
    ).to_df()

    return online_resp
