"""carclaims domain: the real file, the prepare script's decoding choices, the config's
consistency with the encoded columns, and one end-to-end run.

carclaims is observational with no ground truth and no synthetic twin, so the tests pin what
the data and config CLAIM (counts, rates, which columns exist) rather than any estimate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from rootcause.evaluation import harness
from rootcause.evaluation.scms import SCM_REGISTRY
from rootcause.pipeline import preprocessing, runner

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import prepare_carclaims as prepare  # noqa: E402

DOMAIN = "carclaims"
LEVERS = ("police_report_filed", "agent_internal", "deductible_100usd")


@pytest.fixture(scope="module")
def claims_config(config_loader):
    return config_loader.get_domain(DOMAIN).extra


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "data" / "carclaims" / "carclaims.csv")


@pytest.fixture(scope="module")
def claims_df(claims_config) -> pd.DataFrame:
    return pd.read_csv(runner.resolve_data_path(claims_config, "real"))


# --- the real file ------------------------------------------------------------------------


def test_raw_file_is_the_published_15420_claims_with_5_99_percent_fraud(raw_df):
    assert raw_df.shape == (15420, 33) and not raw_df.isna().any().any()
    assert (raw_df["FraudFound"] == "Yes").sum() == 923
    assert raw_df["PolicyNumber"].is_unique


def test_committed_claims_file_is_what_the_prepare_script_produces(raw_df):
    committed = pd.read_csv(REPO_ROOT / "data" / "carclaims" / "claims.csv")
    pd.testing.assert_frame_equal(prepare.prepare(raw_df), committed)


def test_prepared_file_has_the_documented_shape_and_no_calendar_or_age_columns(claims_df):
    assert list(claims_df.columns) == prepare.COLUMNS
    assert claims_df["claim_id"].is_unique and not claims_df.isna().any().any()
    assert claims_df["fraud_found"].sum() == 923
    dropped = {"month", "week_of_month", "day_of_week", "age", "rep_number", "policy_type"}
    assert not dropped & set(claims_df.columns)


def test_grouped_makes_and_corrected_spellings(raw_df, claims_df):
    assert (claims_df["make"] == "other").sum() == 704  # twelve makes with < 300 claims
    assert set(claims_df["make"]) == {"pontiac", "toyota", "honda", "mazda", "chevrolet", "acura", "ford", "other"}
    assert (claims_df["make"] == "acura").sum() == (raw_df["Make"] == "Accura").sum() == 472


def test_binary_and_scaled_columns_keep_the_source_counts(claims_df):
    assert claims_df["police_report_filed"].sum() == 428
    assert claims_df["agent_internal"].sum() == 241
    assert (claims_df["deductible_100usd"] != 4.0).sum() == 582  # 96% of claims have a $400 deductible
    assert set(claims_df["deductible_100usd"]) == {3.0, 4.0, 5.0, 7.0}


def test_prepare_rejects_a_value_it_does_not_decode(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "MaritalStatus"] = "Separated"
    with pytest.raises(ValueError, match="MaritalStatus.*Separated"):
        prepare.prepare(bad)
    bad = raw_df.copy()
    bad.loc[0, "Sex"] = "Unknown"
    with pytest.raises(ValueError, match="Sex.*Unknown"):
        prepare.prepare(bad)


def test_prepare_rejects_duplicate_claim_ids_and_wrong_columns(raw_df):
    dup = raw_df.copy()
    dup.loc[1, "PolicyNumber"] = dup.loc[0, "PolicyNumber"]
    with pytest.raises(ValueError, match="not unique"):
        prepare.prepare(dup)
    with pytest.raises(ValueError, match="expected columns"):
        prepare.prepare(raw_df.drop(columns="Year"))


def test_documented_disparity_and_fault_signal_are_in_the_data(claims_df):
    by_sex = claims_df.groupby("female")["fraud_found"].mean()
    assert by_sex[1] == pytest.approx(0.043, abs=0.001) and by_sex[0] == pytest.approx(0.063, abs=0.001)
    assert by_sex.min() / by_sex.max() == pytest.approx(0.69, abs=0.01)
    by_fault = claims_df.groupby("fault_policy_holder")["fraud_found"].mean()
    assert by_fault[1] == pytest.approx(0.079, abs=0.001) and by_fault[0] == pytest.approx(0.009, abs=0.001)


# --- config -------------------------------------------------------------------------------


def test_config_has_a_real_dataset_only_and_no_ground_truth(config_loader, claims_config):
    assert config_loader.get_domain(DOMAIN).is_runnable
    assert runner.available_datasets(claims_config) == ["multimodal", "real"]
    assert runner.resolve_dataset(claims_config) == "real"
    assert runner.resolve_data_path(claims_config, "real").exists()
    assert not [key for key in SCM_REGISTRY if key[0] == DOMAIN]  # nothing to score against


def test_config_adjusts_each_lever_for_every_other_encoded_column(claims_config, claims_df):
    features = preprocessing.preprocess(claims_df, claims_config).feature_columns
    treatments = claims_config["effect_estimation"]["treatments"]
    assert [t["name"] for t in treatments] == list(LEVERS)
    for treatment in treatments:
        assert sorted(treatment["confounders"]) == sorted(set(features) - {treatment["name"]}), treatment["name"]
        assert len(treatment["confounders"]) == len(set(treatment["confounders"]))


def test_config_discovery_variables_and_priors_are_consistent(claims_config, claims_df):
    features = set(preprocessing.preprocess(claims_df, claims_config).feature_columns)
    discovery = claims_config["causal_discovery"]
    assert set(discovery["variables"]) <= features | {"fraud_found"}
    assert not discovery.get("required_edges")  # nothing is asserted on observational data
    for a, b in discovery["forbidden_edges"]:
        assert a in discovery["variables"] and b in discovery["variables"]
    assert ["fraud_found", "female"] in discovery["forbidden_edges"]
    assert ["female", "policyholder_age_group"] in discovery["forbidden_edges"]


def test_config_levers_candidates_and_sensitive_attribute_line_up(claims_config, claims_df):
    treatments = {t["name"] for t in claims_config["effect_estimation"]["treatments"]}
    candidates = claims_config["interventions"]["candidates"]
    assert {c["target_variable"] for c in candidates} == treatments
    assert claims_config["counterfactuals"]["treatment"] in treatments
    assert claims_config["interventions"]["sensitive_attribute"] == "female"
    assert set(claims_df["female"]) == {0, 1}
    assert all(c["cost"] > 0 for c in candidates)


def test_config_encoding_covers_every_prepared_column(claims_config, claims_df):
    """Every prepared column is either a feature, the id or the outcome: none silently unused."""
    fs = claims_config["feature_store"]
    used = {fs["entity_id_column"], claims_config["ingestion"]["outcome_column"], *fs["feature_columns"]}
    used |= {c["name"] for c in fs["categorical_columns"]}
    assert used == set(claims_df.columns)


def test_ordinal_orders_and_onehot_levels_match_the_data(claims_config, claims_df):
    for cfg in claims_config["feature_store"]["categorical_columns"]:
        observed = set(claims_df[cfg["name"]].astype(str))
        declared = set(map(str, cfg.get("order") or cfg["levels"]))
        assert observed == declared, cfg["name"]


# --- through the real pipeline ------------------------------------------------------------


def test_subsample_runs_end_to_end_and_the_sex_flag_fires(
    claims_config, claims_df, isolated_feast_root, tmp_path
):
    """A fixed 3,000-row sample with a cheap placebo: enough to exercise 33 encoded columns
    through Stages 1-6 without the 2-minute full-file cost."""
    sample = claims_df.sample(3000, random_state=7)
    path = tmp_path / "claims_sample.csv"
    sample.to_csv(path, index=False)
    cfg = {
        **claims_config,
        "ingestion": {**claims_config["ingestion"], "datasets": {"real": str(path)}},
        "effect_estimation": {**claims_config["effect_estimation"], "refutation_simulations": 5},
    }

    real, *rest = harness.evaluate_dataset(DOMAIN, cfg, "real")

    assert rest == [] and real.truth is None and not real.scored_against_truth and real.reference == []
    (run,) = real.runs
    assert run.error is None, run.error
    assert run.n_rows == len(sample) and set(run.estimated_ates) == set(LEVERS)
    assert run.fairness_passed is False and 0.5 < run.fairness_ratio < 0.85
    assert run.top_choice in {c["id"] for c in claims_config["interventions"]["candidates"]}
    assert not {"ate_mae", "edge_precision", "fairness_flag_correct"} & set(run.values)
