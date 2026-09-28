"""Illinois Workplace Wellness domain: the real-RCT file and its synthetic mirror.

The real file must be what the prepare script says it is (right rows, arms, no
post-assignment columns), the mirror must be reproducible from its SCM and carry
the planted effect, the trial reference must be computed correctly, and both
datasets must run through the real pipeline and be scored the way the report claims.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rootcause.evaluation import benchmarks, harness, report
from rootcause.evaluation.scms import (
    SCM_REGISTRY,
    WELLNESS_N_ROWS,
    WELLNESS_SEED,
    WELLNESS_TREAT_LOGIT,
    illinois_wellness_scm,
)
from rootcause.pipeline import runner

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import generate_synthetic_wellness_data as generator  # noqa: E402
import prepare_illinois_wellness as prepare  # noqa: E402

DOMAIN = "illinois_wellness"


@pytest.fixture(autouse=True)
def small_truth_draws(monkeypatch):
    monkeypatch.setattr(harness, "TRUTH_DRAWS", 300_000)


@pytest.fixture(scope="module")
def wellness_config(config_loader):
    return config_loader.get_domain(DOMAIN).extra


@pytest.fixture(scope="module")
def real_df(wellness_config) -> pd.DataFrame:
    return pd.read_csv(runner.resolve_data_path(wellness_config, "real"))


@pytest.fixture(scope="module")
def synthetic_df(wellness_config) -> pd.DataFrame:
    return pd.read_csv(runner.resolve_data_path(wellness_config, "synthetic"))


# --- the real file ------------------------------------------------------------------------


def test_real_file_is_the_randomized_trial_with_the_published_arm_sizes(real_df):
    assert len(real_df) == 4834
    assert list(real_df.columns) == prepare.COLUMNS
    assert real_df["treat"].value_counts().to_dict() == {1: 3300, 0: 1534}
    assert real_df["employee_id"].is_unique
    assert not real_df["terminated_0119"].isna().any()  # the outcome is fully observed


def test_real_file_keeps_only_variables_measured_before_assignment(real_df):
    allowed = {"employee_id", "treat", "terminated_0119", *prepare.COVARIATES}
    assert set(real_df.columns) == allowed
    for post_treatment in ("gym_0816_0717", "hra_c_yr1", "prod_index_yr1", "prod_index_yr2", "terminated_0717"):
        assert post_treatment not in real_df.columns


def test_committed_real_file_is_what_the_prepare_script_produces():
    data_dir = REPO_ROOT / "data" / "illinois_wellness"
    raw = pd.read_csv(data_dir / "firm_admin.csv")
    pd.testing.assert_frame_equal(prepare.prepare(raw), pd.read_csv(data_dir / "wellness.csv"))


def test_real_trial_arms_are_balanced_on_pre_treatment_covariates(real_df):
    for col in ("male", "age50", "age37_49", "white"):
        diff = real_df.groupby("treat")[col].mean().diff().iloc[-1]
        assert abs(diff) < 0.03, col


# --- the synthetic mirror -----------------------------------------------------------------


def test_committed_synthetic_file_is_exactly_what_the_scm_generates(synthetic_df):
    pd.testing.assert_frame_equal(generator.generate(), synthetic_df)
    assert WELLNESS_N_ROWS == len(synthetic_df) and WELLNESS_SEED == 42


def test_mirror_has_the_real_schema_and_roughly_the_real_marginals(real_df, synthetic_df):
    assert list(synthetic_df.columns) == list(real_df.columns)
    big = illinois_wellness_scm().sample(100_000, 1)
    for col in ("male", "age50", "age37_49", "white", "treat"):
        assert big[col].mean() == pytest.approx(real_df[col].mean(), abs=0.01), col
    assert big["sickleave_0815_0716"].mean() == pytest.approx(real_df["sickleave_0815_0716"].mean(), rel=0.15)
    assert (big["gym_0815_0716"] == 0).mean() == pytest.approx((real_df["gym_0815_0716"] == 0).mean(), abs=0.03)

    def corr(d):
        return d["prod_index_yr0"].corr(d["sickleave_0815_0716"])

    assert corr(big) == pytest.approx(corr(real_df), abs=0.1)


def test_mirror_base_termination_rate_matches_the_real_file(real_df):
    big = illinois_wellness_scm().sample(200_000, 2)
    assert big["terminated_0119"].mean() == pytest.approx(real_df["terminated_0119"].mean(), abs=0.015)


def test_mirror_age_dummies_are_mutually_exclusive_and_treatment_is_randomized():
    df = illinois_wellness_scm().sample(50_000, 3)
    assert not ((df["age50"] == 1) & (df["age37_49"] == 1)).any()
    for col in df.columns.drop(["treat", "terminated_0119"]):  # the outcome is meant to depend on treat
        assert abs(df["treat"].corr(df[col])) < 0.02, col


def test_mirror_plants_the_stated_effect_and_the_truth_is_do_one_minus_do_zero():
    scm = illinois_wellness_scm()
    ate = scm.true_effect_of_switching("treat", "terminated_0119", n=400_000)

    assert ate < 0 and ate == pytest.approx(-0.0475, abs=0.006)
    assert WELLNESS_TREAT_LOGIT < 0
    null = illinois_wellness_scm(treat_logit=0.0)
    assert null.true_effect_of_switching("treat", "terminated_0119", n=400_000) == 0


def test_binary_nodes_are_detected_and_only_they_get_the_switching_effect():
    scm = illinois_wellness_scm()
    assert scm.is_binary("treat") and scm.is_binary("male") and scm.is_binary("terminated_0119")
    assert not scm.is_binary("prod_index_yr0") and not scm.is_binary("sickleave_0815_0716")


def test_harness_true_effect_scales_a_binary_switch_by_the_shift():
    scm = illinois_wellness_scm()
    one = harness._true_effect(scm, "treat", "terminated_0119", 1.0)
    assert harness._true_effect(scm, "treat", "terminated_0119", 2.0) == pytest.approx(2 * one)
    # a continuous node still uses a shift
    assert harness._true_effect(scm, "prod_index_yr0", "terminated_0119", 1.0) < 0


def test_mirror_graph_has_no_edge_into_treatment_and_a_treatment_to_outcome_edge():
    edges = set(illinois_wellness_scm().edges())
    assert ("treat", "terminated_0119") in edges
    assert not any(child == "treat" for _, child in edges)
    assert not any("age_group" in e for e in edges)  # latent, so not part of the observed truth
    assert SCM_REGISTRY[(DOMAIN, "synthetic")] is illinois_wellness_scm


# --- the trial reference ------------------------------------------------------------------


def test_difference_in_means_matches_a_hand_computation():
    treat = pd.Series([1, 1, 1, 0, 0, 0])
    y = pd.Series([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    ref = benchmarks.difference_in_means(treat, y, "t", "y")

    assert ref.estimate == pytest.approx(2 / 3 - 1 / 3)
    se = np.sqrt((1 / 3) / 3 + (1 / 3) / 3)  # both arms: variance 1/3 (ddof=1), n=3
    assert ref.std_error == pytest.approx(se)
    assert ref.ci_low == pytest.approx(ref.estimate - 1.959964 * se)
    assert (ref.n_treated, ref.n_control) == (3, 3)
    assert ref.contains(0.0) and not ref.significant


def test_difference_in_means_drops_missing_outcomes_but_reports_how_many():
    treat = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
    y = pd.Series([1.0, 1.0, np.nan, np.nan, 0.0, 0.0, 0.0, 1.0])

    ref = benchmarks.difference_in_means(treat, y, "t", "y")

    assert (ref.n_treated, ref.n_control) == (2, 4)
    assert ref.outcome_missing_treated == 0.5 and ref.outcome_missing_control == 0.0


def test_a_large_clear_effect_is_significant_and_an_arm_with_one_outcome_is_rejected():
    treat = pd.Series([1] * 50 + [0] * 50)
    y = pd.Series([1.0] * 50 + [0.0] * 50) + pd.Series(np.random.default_rng(0).normal(0, 0.1, 100))
    assert benchmarks.difference_in_means(treat, y, "t", "y").significant
    with pytest.raises(ValueError, match="at least two"):
        benchmarks.difference_in_means(pd.Series([1, 0, 0]), pd.Series([1.0, 0.0, 1.0]), "t", "y")


def test_reference_on_the_real_file_is_the_published_null(wellness_config):
    (ref,) = benchmarks.experimental_reference(
        runner.resolve_data_path(wellness_config, "real"), wellness_config
    )

    assert (ref.n_treated, ref.n_control) == (3300, 1534)
    assert abs(ref.estimate) < 0.01 and not ref.significant
    assert ref.outcome_missing_treated == ref.outcome_missing_control == 0.0
    assert (DOMAIN, "real") in benchmarks.RCT_DATASETS and (DOMAIN, "synthetic") not in benchmarks.RCT_DATASETS


def test_reference_skips_a_treatment_that_is_not_binary(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"dose": [0.5, 1.5, 2.5, 3.5], "y": [0, 1, 0, 1]}).to_csv(path, index=False)
    cfg = {"effect_estimation": {"outcome": "y", "treatments": [{"name": "dose"}]}}
    assert benchmarks.experimental_reference(path, cfg) == []


# --- config -------------------------------------------------------------------------------


def test_config_registers_both_datasets_and_defaults_to_the_real_trial(config_loader, wellness_config):
    assert config_loader.get_domain(DOMAIN).is_runnable
    assert runner.available_datasets(wellness_config) == ["multimodal", "real", "synthetic"]
    assert runner.resolve_dataset(wellness_config) == "real"
    for kind in ("real", "synthetic"):
        assert runner.resolve_data_path(wellness_config, kind).exists()


def test_config_uses_only_pre_assignment_columns_and_never_asserts_the_effect(wellness_config, real_df):
    features = wellness_config["feature_store"]["feature_columns"]
    assert set(features) <= set(real_df.columns)
    assert not any(tag in c for c in features for tag in ("0816", "yr1", "yr2", "hra", "0717"))

    discovery = wellness_config["causal_discovery"]
    assert not discovery.get("required_edges"), "asserting treat -> outcome would assume the answer"
    assert set(discovery["variables"]) <= set(features) | {wellness_config["ingestion"]["outcome_column"]}
    for a, b in discovery["forbidden_edges"]:
        assert a in discovery["variables"] and b in discovery["variables"]
    assert ["male", "treat"] in discovery["forbidden_edges"]  # randomization: nothing causes assignment


def test_config_treatments_line_up_with_the_intervention_candidates(wellness_config):
    treatments = {t["name"] for t in wellness_config["effect_estimation"]["treatments"]}
    targets = {c["target_variable"] for c in wellness_config["interventions"]["candidates"]}
    assert treatments == targets == {"treat"} == {wellness_config["counterfactuals"]["treatment"]}


# --- through the real pipeline ------------------------------------------------------------


def test_real_trial_runs_end_to_end_and_the_pipeline_agrees_with_the_null(wellness_config, isolated_feast_root):
    real, *rest = harness.evaluate_dataset(DOMAIN, wellness_config, "real")

    assert rest == [] and not real.scored_against_truth and real.truth is None
    (run,) = real.runs
    assert run.error is None, run.error
    assert run.n_rows == 4834  # nothing dropped: 74 missing prod_index_yr0 values are imputed, not dropped
    (ref,) = real.reference
    assert ref.contains(run.estimated_ates["treat"])
    assert run.estimated_ates["treat"] == pytest.approx(ref.estimate, abs=0.01)
    assert run.refutation_passed["treat"] is False  # "no evidence of an effect" is the correct answer here
    assert run.mean_cates["treat"] == pytest.approx(ref.estimate, abs=0.02)
    assert ("treat", "terminated_0119") not in run.predicted_edges

    md = report.render_markdown([real], {"replicates": 0, "n_rows": None})
    assert "against the randomized trial" in md
    assert "the trial finds no effect" in md and "(agree)" in md
    assert "Trial: treated − control" in md


def test_synthetic_mirror_recovers_its_planted_effect_and_is_scored_against_truth(
    wellness_config, isolated_feast_root
):
    committed, replicates = harness.evaluate_dataset(DOMAIN, wellness_config, "synthetic", replicates=1)

    assert committed.reference == []  # the mirror is scored against its SCM, not a trial
    assert committed.scored_against_truth and ("treat", "terminated_0119") in committed.truth.edges
    (run,) = committed.runs
    assert run.error is None, run.error
    true_ate = committed.truth.effects["treat"]
    assert true_ate == pytest.approx(-0.0475, abs=0.006)
    assert abs(run.estimated_ates["treat"] - true_ate) < 0.03
    assert run.refutation_passed["treat"] is True  # a real effect stands out from the placebo
    assert run.effect_rows[0].sign_agrees

    # replicates default to the committed file's size, not the harness's 2000
    (rep,) = replicates.runs
    assert rep.error is None, rep.error
    assert rep.n_rows == WELLNESS_N_ROWS
