"""Feast feature repo definitions, built per domain from its config (Stage 2).

Nothing here names a specific domain: the entity, feature view and column
schema all come from the domain config's `feature_store` block plus its
`ingestion.outcome_column`, and every artifact (offline parquet, registry,
online store) lives under `data/<domain_id>/<dataset>/` so two domains never share or
overwrite each other's Feast state. This file is data-source plumbing, not a
place to redefine what a domain's variables are.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.repo_config import RepoConfig
from feast.types import Float32, Int64

REPO_PATH = Path(__file__).resolve().parent
DATA_ROOT = REPO_PATH / "data"


@dataclass(frozen=True)
class DomainFeatureRepo:
    """Everything Stage 2 needs to write, apply and read one domain's features."""

    entity: Entity
    feature_view: FeatureView
    repo_config: RepoConfig
    parquet_path: Path


def domain_data_dir(domain_config: dict) -> Path:
    """Feast state is per (domain, dataset): a real and a synthetic run of the
    same domain share a schema but must not share parquet/registry/online store."""
    dataset = domain_config.get("run", {}).get("dataset") or "default"
    return DATA_ROOT / domain_config["domain"]["id"] / dataset


def build_domain_repo(domain_config: dict, feature_columns: list[str]) -> DomainFeatureRepo:
    """`feature_columns` are the encoded columns (see pipeline/preprocessing.py),
    which can differ from the config's raw list -- one-hot levels come from the data."""
    domain_id = domain_config["domain"]["id"]
    fs_cfg = domain_config["feature_store"]
    outcome_col = domain_config["ingestion"]["outcome_column"]
    data_dir = domain_data_dir(domain_config)
    parquet_path = data_dir / "features.parquet"

    entity = Entity(
        name=fs_cfg["entity"],
        join_keys=[fs_cfg["entity_id_column"]],
        value_type=ValueType.INT64,
    )
    source = FileSource(
        name=f"{domain_id}_source",
        path=str(parquet_path),
        timestamp_field="event_timestamp",
    )
    feature_view = FeatureView(
        name=feature_view_name(domain_config),
        entities=[entity],
        ttl=timedelta(days=3650),
        schema=[Field(name=col, dtype=Float32) for col in feature_columns]
        + [Field(name=outcome_col, dtype=Int64)],
        source=source,
        online=True,
    )
    repo_config = RepoConfig(
        project=f"rootcause_{domain_id}",
        provider="local",
        registry=str(data_dir / "registry.db"),
        online_store={"type": "sqlite", "path": str(data_dir / "online_store.db")},
        entity_key_serialization_version=3,
        repo_path=REPO_PATH,
    )
    return DomainFeatureRepo(entity, feature_view, repo_config, parquet_path)


def feature_view_name(domain_config: dict) -> str:
    return f"{domain_config['feature_store']['entity']}_causal_features"
