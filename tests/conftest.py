"""Shared fixtures for the RootCause test suite.

Everything here runs against real code and real data (the committed
employee_attrition synthetic dataset), not mocks -- matching how this
pipeline was verified during Phase 1 development (see memory: ad-hoc
per-stage checks against data/employee_attrition/ground_truth.json).

Expensive fixtures (Feast materialization, DoWhy estimation) are
session-scoped so each stage's real work happens once per test run and
downstream stages' tests reuse the result, rather than re-deriving it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rootcause.pipeline import (
    causal_discovery,
    effect_estimation,
    feature_store,
    ingestion,
)
from rootcause.utils.config_loader import ConfigLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "employee_attrition" / "attrition.csv"
GROUND_TRUTH_PATH = REPO_ROOT / "data" / "employee_attrition" / "ground_truth.json"


@pytest.fixture
def isolated_feast_root(tmp_path, monkeypatch) -> Path:
    """Point Feast's per-domain state at a temp dir. Tests that invent a
    throwaway domain use this instead of deleting files afterwards: Feast keeps
    its SQLite online store open until GC, so rmtree fails on Windows."""
    from rootcause.feature_repo import definitions

    monkeypatch.setattr(definitions, "DATA_ROOT", tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def data_path() -> Path:
    return DATA_PATH


@pytest.fixture(scope="session")
def config_loader() -> ConfigLoader:
    return ConfigLoader()


@pytest.fixture(scope="session")
def domain_config(config_loader: ConfigLoader) -> dict:
    """The employee_attrition domain's `extra` dict -- the same dict every
    rootcause/pipeline/*.py stage function takes as `domain_config`."""
    return config_loader.get_domain("employee_attrition").extra


@pytest.fixture(scope="session")
def ground_truth() -> dict:
    return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def raw_df(domain_config: dict):
    return ingestion.ingest(DATA_PATH, domain_config)


@pytest.fixture(scope="session")
def feature_df(raw_df, domain_config: dict):
    """Real Feast round-trip (offline parquet write -> materialize -> online
    read), run once per test session -- this is the actual Stage 2 code
    path, not a stand-in for it."""
    return feature_store.build_feature_vectors(raw_df, domain_config)


@pytest.fixture(scope="session")
def causal_graph(feature_df, domain_config: dict):
    return causal_discovery.discover_graph(feature_df, domain_config)


@pytest.fixture(scope="session")
def effect_estimates(feature_df, domain_config: dict):
    return effect_estimation.estimate_effects(feature_df, domain_config)
