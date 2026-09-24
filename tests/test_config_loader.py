"""Stage-0 infra: rootcause/utils/config_loader.py (ported from Prism)."""

from __future__ import annotations

import pytest

from rootcause.utils.config_loader import ConfigLoader, _resolve_env


def test_loads_employee_attrition_domain(config_loader: ConfigLoader):
    domain = config_loader.get_domain("employee_attrition")
    assert domain.id == "employee_attrition"
    assert domain.status == "working"
    assert domain.is_runnable is True
    assert domain.input_type == "tabular"


def test_extra_contains_every_stage_section(domain_config: dict):
    expected_sections = {
        "ingestion",
        "feature_store",
        "causal_discovery",
        "effect_estimation",
        "counterfactuals",
        "interventions",
        "explanation",
    }
    assert expected_sections.issubset(domain_config.keys())
    # "domain" is consumed into DomainConfig's own fields, not left in extra
    assert "domain" not in domain_config


def test_unknown_domain_raises_keyerror(config_loader: ConfigLoader):
    with pytest.raises(KeyError, match="Unknown domain"):
        config_loader.get_domain("not_a_real_domain")


def test_list_domains_runnable_only(config_loader: ConfigLoader):
    all_domains = config_loader.list_domains()
    runnable = config_loader.list_domains(runnable_only=True)
    assert {d.id for d in runnable}.issubset({d.id for d in all_domains})
    assert all(d.is_runnable for d in runnable)


def test_env_var_resolution_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_TEST_VAR", raising=False)
    assert _resolve_env("${SOME_UNSET_TEST_VAR:fallback}") == "fallback"


def test_env_var_resolution_prefers_actual_env_value(monkeypatch):
    monkeypatch.setenv("SOME_UNSET_TEST_VAR", "actual")
    assert _resolve_env("${SOME_UNSET_TEST_VAR:fallback}") == "actual"


def test_env_var_resolution_recurses_into_nested_structures(monkeypatch):
    monkeypatch.setenv("SOME_UNSET_TEST_VAR", "actual")
    resolved = _resolve_env({"a": ["${SOME_UNSET_TEST_VAR:x}", {"b": "${MISSING:y}"}]})
    assert resolved == {"a": ["actual", {"b": "y"}]}


def test_missing_domain_section_raises(tmp_path):
    bad_yaml = tmp_path / "broken.yaml"
    bad_yaml.write_text("ingestion:\n  file_type: csv\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required top-level 'domain' section"):
        ConfigLoader(configs_dir=tmp_path)
