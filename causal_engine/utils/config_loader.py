# causal_engine/utils/config_loader.py
# Ported from Prism's core/config_loader.py (C:\Users\priya\Documents\personal-projects\prism)
# per the CDIA spec's own note: "Reuse Prism's ConfigLoader to avoid rebuilding
# configuration infrastructure." Same envelope-plus-extra pattern: the shared
# "domain" section is validated by Pydantic, everything CDIA-stage-specific
# (causal_discovery, effect_estimation, counterfactuals, interventions,
# explanation) is kept as a resolved raw dict on DomainConfig.extra, because
# each domain genuinely needs different treatment/outcome/confounder shapes.
#
# Dropped relative to Prism's version: classification_hints/capabilities
# (Prism-RAG/document-chat concepts with no CDIA equivalent) and AgentConfig
# (CDIA's 7 pipeline-stage agents are fixed and defined in causal_engine/agents/,
# not a per-domain plugin list). Kept: the optional `pipeline` pointer, for a
# future domain that needs to override a stage with domain-specific logic.

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _resolve_env(value):
    """Recursively replace ${VAR:default} in strings/dicts/lists with the
    environment value, falling back to the given default (or "" if none
    and the env var is unset). Mirrors os.getenv(VAR, default)."""
    if isinstance(value, str):
        def _sub(match: re.Match) -> str:
            var_name, default = match.group(1), match.group(2)
            return os.getenv(var_name, default if default is not None else "")
        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


class PipelineConfig(BaseModel):
    module: str
    class_name: str = Field(alias="class")

    model_config = {"populate_by_name": True}


class DomainConfig(BaseModel):
    id: str
    name: str
    description: str = ""
    status: str = "coming_soon"  # "working" | "coming_soon"
    input_type: str = "tabular"

    # Wording for generated text; defaults resolved in utils/vocabulary.py
    entity_noun: Optional[str] = None
    entity_noun_plural: Optional[str] = None
    outcome_label: Optional[str] = None

    # Optional domain-specific stage override, unused by employee_attrition
    # in Phase 1 -- present for future domains (e.g. insurance_claims) that
    # need a non-default stage implementation.
    pipeline: Optional[PipelineConfig] = None

    # Everything else in the YAML (causal_discovery, effect_estimation,
    # counterfactuals, interventions, explanation, ...) -- domain-specific,
    # resolved (env placeholders substituted) but otherwise untouched. Each
    # causal_engine/pipeline/*.py stage reads the keys it needs from here.
    extra: dict = Field(default_factory=dict)

    @property
    def is_runnable(self) -> bool:
        return self.status == "working"


_KNOWN_TOP_KEYS = {"domain", "pipeline"}

# Data-source kinds a domain's ingestion.datasets may name. real = observed data,
# synthetic = fully simulated (known graph), semi_synthetic = real covariates with
# simulated treatment/outcome (known effect), multimodal = real rows plus SYNTHETIC PDF/image
# attachments (see ingestion.attachments). They share one schema and config.
DATASET_KINDS = ("real", "synthetic", "semi_synthetic", "multimodal")
# `real_<slug>` (e.g. real_2007) names one more observed slice of the same schema, for a domain whose
# real data come as separate populations that must not be pooled (Freddie Mac origination years).
REAL_SLICE = re.compile(r"real_[a-z0-9]+(_[a-z0-9]+)*")


def _validate_datasets(path: Path, ingestion: dict) -> None:
    datasets = ingestion.get("datasets")
    if not datasets or not isinstance(datasets, dict):
        raise ValueError(f"{path}: ingestion.datasets must map dataset kind -> file path")
    unknown = sorted(k for k in datasets if k not in DATASET_KINDS and not REAL_SLICE.fullmatch(str(k)))
    if unknown:
        raise ValueError(
            f"{path}: unknown dataset kind(s) {unknown}; expected {list(DATASET_KINDS)} or real_<name>"
        )
    default = ingestion.get("default_dataset")
    if default not in datasets:
        raise ValueError(
            f"{path}: ingestion.default_dataset={default!r} is not one of {sorted(datasets)}"
        )
    _validate_attachments(path, ingestion, datasets)


def _validate_attachments(path: Path, ingestion: dict, datasets: dict) -> None:
    """`ingestion.attachments` maps a dataset kind to its attachment specs. Only the shape is
    checked here (causal_engine.pipeline.modalities checks the rest when a run uses them)."""
    attachments = ingestion.get("attachments")
    if not attachments:
        return
    if not isinstance(attachments, dict):
        raise ValueError(f"{path}: ingestion.attachments must map dataset kind -> list of attachment specs")
    for kind, specs in attachments.items():
        if kind not in datasets:
            raise ValueError(f"{path}: ingestion.attachments names dataset {kind!r}, which is not in ingestion.datasets")
        if not isinstance(specs, list) or not all(isinstance(s, dict) and {"name", "kind", "directory"} <= set(s) for s in specs):
            raise ValueError(f"{path}: each attachment under {kind!r} needs name, kind and directory")
        names = [s["name"] for s in specs]
        if len(names) != len(set(names)):
            raise ValueError(f"{path}: attachment names under {kind!r} must be unique, got {names}")


class ConfigLoader:
    """Loads and validates configs/*.yaml. Construct once, reuse -- YAML is
    only read from disk on load()/reload()."""

    def __init__(self, configs_dir: str | Path = DEFAULT_CONFIGS_DIR):
        self.configs_dir = Path(configs_dir)
        self._domains: dict[str, DomainConfig] = {}
        self.load()

    def load(self) -> None:
        self._domains = {}
        for path in sorted(self.configs_dir.glob("*.yaml")):
            domain = self._load_one(path)
            self._domains[domain.id] = domain

    reload = load  # explicit alias -- reload() reads the exact same files load() did

    def _load_one(self, path: Path) -> DomainConfig:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw = _resolve_env(raw)

        domain_section = raw.get("domain")
        if not domain_section:
            raise ValueError(f"{path}: missing required top-level 'domain' section")

        if "ingestion" in raw:
            _validate_datasets(path, raw["ingestion"])

        extra = {k: v for k, v in raw.items() if k not in _KNOWN_TOP_KEYS}
        # Stage functions receive only `extra`, and need the domain's id/name
        # (per-domain Feast paths, domain wording in Stage 5/7 prompts).
        extra["domain"] = dict(domain_section)

        return DomainConfig(
            **domain_section,
            pipeline=raw.get("pipeline"),
            extra=extra,
        )

    def get_domain(self, domain_id: str) -> DomainConfig:
        try:
            return self._domains[domain_id]
        except KeyError:
            raise KeyError(
                f"Unknown domain '{domain_id}'. Known domains: {sorted(self._domains)}"
            ) from None

    def list_domains(self, runnable_only: bool = False) -> list[DomainConfig]:
        domains = list(self._domains.values())
        if runnable_only:
            domains = [d for d in domains if d.is_runnable]
        return sorted(domains, key=lambda d: d.id)
