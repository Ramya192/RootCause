"""Domain wording (rootcause/utils/vocabulary.py) and its use in Stage 7."""

from __future__ import annotations

import copy

import numpy as np
import openai
import pandas as pd

from rootcause.pipeline import explanation
from rootcause.pipeline.preprocessing import feature_columns_of
from rootcause.utils.vocabulary import domain_vocabulary


def test_employee_config_wording(domain_config):
    v = domain_vocabulary(domain_config)
    assert (v.entity, v.entities, v.outcome) == ("employee", "employees", "attrition")


def test_defaults_when_nothing_configured():
    cfg = {"domain": {"id": "x"}, "ingestion": {"outcome_column": "defaulted"}}
    v = domain_vocabulary(cfg)
    assert (v.entity, v.entities, v.outcome) == ("record", "records", "defaulted")


def test_plural_defaults_to_noun_plus_s_and_missing_domain_section_is_ok():
    cfg = {"domain": {"id": "x", "entity_noun": "claim"}, "ingestion": {"outcome_column": "fraud"}}
    assert domain_vocabulary(cfg).entities == "claims"
    assert domain_vocabulary({"ingestion": {"outcome_column": "y"}}).entity == "record"


def test_feature_columns_of_includes_encoded_columns_and_excludes_id_and_outcome():
    cfg = {"feature_store": {"entity_id_column": "pid"}, "ingestion": {"outcome_column": "y"}}
    df = pd.DataFrame(columns=["pid", "age", "housing__rent", "age__missing", "y"])
    assert feature_columns_of(df, cfg) == ["age", "housing__rent", "age__missing"]


def _custom_domain(domain_config):
    cfg = copy.deepcopy(domain_config)
    cfg["domain"].update(entity_noun="applicant", entity_noun_plural="applicants", outcome_label="loan default")
    return cfg


def test_shap_covers_encoded_columns_not_just_configured_ones(domain_config):
    """Regression risk from item 2: one-hot/indicator columns aren't in feature_columns."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "employee_id": range(200),
            "compensation": rng.normal(size=200),
            "housing__rent": rng.integers(0, 2, 200).astype(float),
            "compensation__missing": rng.integers(0, 2, 200).astype(float),
            "attrition": rng.integers(0, 2, 200),
        }
    )
    summary = explanation._shap_summary(df, domain_config)
    assert set(summary) == {"compensation", "housing__rent", "compensation__missing"}


def test_llm_prompt_and_template_use_domain_wording(monkeypatch, domain_config):
    cfg = _custom_domain(domain_config)
    captured = {}

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages):
                    captured["prompt"] = messages[0]["content"]
                    msg = type("M", (), {"content": " ok "})()
                    return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)
    out = explanation._llm_narrative(cfg, [], [], [], {"compensation": 1.0})
    assert out == "ok"
    assert "loan default among applicants" in captured["prompt"]
    assert "employee" not in captured["prompt"].lower()
    assert "Top-line drivers of loan default" in captured["prompt"]  # facts block uses it too
