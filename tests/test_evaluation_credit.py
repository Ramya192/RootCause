"""German Credit domain: the real file, its semi-synthetic twin, the fairness check and the
bootstrap stability runs.

The real file must be what the prepare script says it is, the semi-synthetic file must be
reproducible from its SCM and carry exactly the planted effects on top of REAL covariates,
the harness must score the fairness verdict against the SCM's population ratio, and real data
(no SCM) must get bootstrap stability runs rather than accuracy numbers it cannot have.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from causal_engine.evaluation import harness, report
from causal_engine.evaluation.scm import SCM, Node, bernoulli_of
from causal_engine.evaluation.scms import (
    CREDIT_LEVER_LOGITS,
    CREDIT_N_ROWS,
    CREDIT_SEED,
    CREDIT_YOUNG_LOGIT,
    SCM_REGISTRY,
    german_credit_semi_synthetic_scm,
)
from causal_engine.models.schemas import CausalGraph, EffectEstimate, InterventionRecommendation
from causal_engine.pipeline import preprocessing, runner

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))
import generate_semi_synthetic_credit_data as generator  # noqa: E402
import prepare_german_credit as prepare  # noqa: E402

DOMAIN = "german_credit"
LEVERS = ("duration_months", "credit_amount_kdm", "installment_rate")


@pytest.fixture(autouse=True)
def small_truth_draws(monkeypatch):
    monkeypatch.setattr(harness, "TRUTH_DRAWS", 200_000)


@pytest.fixture(scope="module")
def credit_config(config_loader):
    return config_loader.get_domain(DOMAIN).extra


@pytest.fixture(scope="module")
def real_df(credit_config) -> pd.DataFrame:
    return pd.read_csv(runner.resolve_data_path(credit_config, "real"))


@pytest.fixture(scope="module")
def semi_df(credit_config) -> pd.DataFrame:
    return pd.read_csv(runner.resolve_data_path(credit_config, "semi_synthetic"))


def quick(config: dict, simulations: int = 5) -> dict:
    """Same config with a cheaper placebo test (each simulation refits a 40-column regression)."""
    effects = {**config["effect_estimation"], "refutation_simulations": simulations}
    return {**config, "effect_estimation": effects}


# --- the real file ------------------------------------------------------------------------


def test_real_file_is_the_1000_statlog_applicants_with_the_published_class_split(real_df):
    assert len(real_df) == 1000 and list(real_df.columns) == prepare.COLUMNS
    assert real_df["applicant_id"].is_unique and not real_df.isna().any().any()
    assert real_df["default"].sum() == 300  # 700 good / 300 bad in the UCI documentation
    assert real_df["foreign_worker"].sum() == 963 and real_df["female"].sum() == 310


def test_committed_real_file_is_what_the_prepare_script_produces():
    data_dir = REPO_ROOT / "data" / "german_credit"
    raw = pd.read_csv(data_dir / "raw" / "german.data", sep=" ", header=None, names=prepare.RAW_COLUMNS)
    pd.testing.assert_frame_equal(prepare.prepare(raw), pd.read_csv(data_dir / "credit.csv"))


def test_prepare_derives_the_documented_columns(real_df):
    assert (real_df["young"] == (real_df["age"] < 25)).all() and real_df["young"].sum() == 149
    assert real_df["credit_amount_kdm"].max() == pytest.approx(18.424)  # 18,424 DM in the raw file
    assert (real_df["purpose"] == "other").sum() == 55  # domestic appliances, repairs, retraining, others
    assert set(real_df["employment_since"]) == {"unemployed", "lt_1y", "1_to_4y", "4_to_7y", "ge_7y"}


def test_prepare_rejects_a_code_the_codebook_does_not_define():
    raw = pd.read_csv(REPO_ROOT / "data" / "german_credit" / "raw" / "german.data", sep=" ", header=None, names=prepare.RAW_COLUMNS)
    raw.loc[0, "housing"] = "A159"
    with pytest.raises(ValueError, match="housing.*A159"):
        prepare.prepare(raw)


def test_real_age_disparity_is_the_one_the_config_documents(real_df):
    rates = real_df.groupby("young")["default"].mean()
    assert rates[1] == pytest.approx(0.409, abs=0.001) and rates[0] == pytest.approx(0.281, abs=0.001)
    assert rates.min() / rates.max() == pytest.approx(0.686, abs=0.001)


# --- the semi-synthetic twin --------------------------------------------------------------


def test_committed_semi_synthetic_file_is_exactly_what_the_scm_generates(semi_df):
    pd.testing.assert_frame_equal(generator.generate(), semi_df)
    assert (CREDIT_N_ROWS, CREDIT_SEED) == (len(semi_df), 42)


def test_semi_synthetic_has_the_real_schema_and_every_covariate_row_is_a_real_applicant(real_df, semi_df):
    assert list(semi_df.columns) == list(real_df.columns)
    covariates = [c for c in real_df.columns if c not in ("applicant_id", "default")]
    sample = german_credit_semi_synthetic_scm().sample(5000, 3)
    merged = sample[covariates].merge(real_df[covariates].drop_duplicates(), how="left", indicator=True)
    assert (merged["_merge"] == "both").all()  # no covariate value was invented


def test_only_the_outcome_is_simulated_so_the_real_covariate_distributions_are_kept(real_df):
    big = german_credit_semi_synthetic_scm().sample(200_000, 1)
    for column in ("duration_months", "credit_amount_kdm", "installment_rate", "age", "young", "female"):
        assert big[column].mean() == pytest.approx(real_df[column].mean(), rel=0.02), column
    corr = lambda d: d["duration_months"].corr(d["credit_amount_kdm"])  # noqa: E731
    assert corr(big) == pytest.approx(corr(real_df), abs=0.02)  # the real correlation between levers survives
    assert big["checking_status"].value_counts(normalize=True)["none"] == pytest.approx(0.394, abs=0.005)


def test_simulated_outcome_reproduces_the_real_default_rates_by_age_group(real_df):
    big = german_credit_semi_synthetic_scm().sample(400_000, 2)
    rates = big.groupby("young")["default"].mean()
    assert big["default"].mean() == pytest.approx(0.30, abs=0.005)
    assert rates[1] == pytest.approx(0.409, abs=0.01) and rates[0] == pytest.approx(0.281, abs=0.005)


def test_planted_lever_effects_are_what_a_logistic_fit_recovers():
    """An independent check on the mechanism: fit the outcome model on a large draw and read the
    lever coefficients back."""
    big = german_credit_semi_synthetic_scm().sample(200_000, 4)
    design = pd.get_dummies(
        big[["checking_status", "credit_history", "savings", "housing", "other_plans"]], drop_first=True
    ).astype(float)
    design["employment"] = big["employment_since"].map({"unemployed": 0, "lt_1y": 1, "1_to_4y": 2, "4_to_7y": 3, "ge_7y": 4})
    for column in (*LEVERS, "young", "foreign_worker"):
        design[column] = big[column]
    fit = sm.Logit(big["default"], sm.add_constant(design)).fit(disp=0)
    for lever, planted in CREDIT_LEVER_LOGITS.items():
        assert fit.params[lever] == pytest.approx(planted, abs=0.4 * planted), lever
    assert fit.params["young"] == pytest.approx(CREDIT_YOUNG_LOGIT, abs=0.08)


def test_true_effects_are_positive_and_ordered_per_unit_and_do_leaves_covariates_alone():
    scm = german_credit_semi_synthetic_scm()
    effects = {lever: scm.true_effect(lever, "default", 1.0, n=400_000) for lever in LEVERS}

    assert all(e > 0 for e in effects.values())
    assert effects["installment_rate"] > effects["credit_amount_kdm"] > effects["duration_months"]
    assert effects["duration_months"] == pytest.approx(0.0047, abs=0.0008)

    base = scm.sample(20_000, 5)
    moved = scm.sample(20_000, 5, shift={"duration_months": 12})
    other = base.columns.drop(["duration_months", "default"])
    pd.testing.assert_frame_equal(base[other], moved[other])  # a lever shift changes only itself and the outcome
    assert (moved["duration_months"] - base["duration_months"] == 12).all()
    assert moved["default"].mean() > base["default"].mean()


def test_graph_is_declared_unknown_and_registered(credit_config):
    scm = german_credit_semi_synthetic_scm()
    assert scm.graph_known is False and "row" not in scm.observed
    assert {child for _, child in scm.edges()} == {"default"}  # only the outcome's parents are modelled
    assert SCM_REGISTRY[(DOMAIN, "semi_synthetic")] is german_credit_semi_synthetic_scm
    assert (DOMAIN, "real") not in SCM_REGISTRY
    assert SCM(nodes=(Node("a", *bernoulli_of((), lambda v: 0.5)),)).graph_known is True


# --- config -------------------------------------------------------------------------------


def test_config_registers_both_datasets_and_defaults_to_the_real_file(config_loader, credit_config):
    assert config_loader.get_domain(DOMAIN).is_runnable
    assert runner.available_datasets(credit_config) == ["real", "semi_synthetic"]
    assert runner.resolve_dataset(credit_config) == "real"
    for kind in ("real", "semi_synthetic"):
        assert runner.resolve_data_path(credit_config, kind).exists()


def test_config_adjusts_each_lever_for_every_other_encoded_column(credit_config, real_df):
    features = preprocessing.preprocess(real_df, credit_config).feature_columns
    treatments = credit_config["effect_estimation"]["treatments"]
    assert [t["name"] for t in treatments] == list(LEVERS)
    for treatment in treatments:
        assert sorted(treatment["confounders"]) == sorted(set(features) - {treatment["name"]}), treatment["name"]
        assert len(treatment["confounders"]) == len(set(treatment["confounders"]))  # no typos, no repeats


def test_config_discovery_variables_and_priors_are_consistent(credit_config, real_df):
    features = set(preprocessing.preprocess(real_df, credit_config).feature_columns)
    discovery = credit_config["causal_discovery"]
    assert set(discovery["variables"]) <= features | {"default"}
    assert not discovery.get("required_edges")  # nothing is asserted on observational data
    for a, b in discovery["forbidden_edges"]:
        assert a in discovery["variables"] and b in discovery["variables"]
    assert ["default", "duration_months"] in discovery["forbidden_edges"]


def test_config_levers_candidates_and_sensitive_attribute_line_up(credit_config, real_df):
    treatments = {t["name"] for t in credit_config["effect_estimation"]["treatments"]}
    candidates = credit_config["interventions"]["candidates"]
    assert {c["target_variable"] for c in candidates} == treatments
    assert credit_config["counterfactuals"]["treatment"] in treatments
    assert credit_config["interventions"]["sensitive_attribute"] == "young"
    assert set(real_df["young"]) == {0, 1}
    for cand in candidates:  # each candidate moves its lever in the direction that lowers default
        assert cand["expected_shift"] < 0 and cand["cost"] > 0


def test_ground_truth_covers_every_treatment_candidate_and_the_fairness_ratio(credit_config):
    truth = harness.compute_truth(german_credit_semi_synthetic_scm(), credit_config)

    assert truth.edges is None  # the graph among real covariates is unknown
    assert set(truth.effects) == set(LEVERS)
    assert set(truth.roi) == {c["id"] for c in credit_config["interventions"]["candidates"]}
    assert all(r > 0 for r in truth.roi.values())
    assert truth.fairness_ratio == pytest.approx(0.686, abs=0.02)
    # the ROI order is set by the (illustrative) costs and the planted effects
    assert max(truth.roi, key=truth.roi.get) == "lower_installment_burden"


# --- fairness truth and scoring -----------------------------------------------------------


def _two_group_scm(rate_group0: float, rate_group1: float) -> SCM:
    def outcome(values, rng, n):
        p = np.where(values["group"] == 1, rate_group1, rate_group0)
        return (rng.uniform(0.0, 1.0, n) < p).astype(int)

    parents, mechanism = bernoulli_of((), lambda v: 0.5)
    return SCM(nodes=(Node("group", parents, mechanism), Node("y", ("group",), outcome)))


def test_true_fairness_ratio_is_lowest_over_highest_group_rate():
    cfg = {"interventions": {"sensitive_attribute": "group"}, "effect_estimation": {"outcome": "y"}}

    assert harness._true_fairness_ratio(_two_group_scm(0.10, 0.20), cfg) == pytest.approx(0.5, abs=0.01)
    assert harness._true_fairness_ratio(_two_group_scm(0.20, 0.20), cfg) == pytest.approx(1.0, abs=0.02)
    missing = {"interventions": {"sensitive_attribute": "no_such_column"}, "effect_estimation": {"outcome": "y"}}
    assert harness._true_fairness_ratio(_two_group_scm(0.1, 0.2), missing) is None


def _outputs(fairness_ratio: float, passed: bool) -> runner.AnalysisOutputs:
    rec = InterventionRecommendation(
        id="shorten_loan_term", target_variable="duration_months", expected_effect=0.03, cost=130.0,
        roi=2e-4, fairness_ratio=fairness_ratio, fairness_pass=passed, rank=1,
    )
    effect = EffectEstimate(treatment="duration_months", outcome="default", ate=0.005, estimator="e")
    return runner.AnalysisOutputs(
        raw=pd.DataFrame({"x": [1, 2]}), features=pd.DataFrame(),
        graph=CausalGraph(nodes=[], edges=[], algorithm="pc"), effects=[effect],
        counterfactuals=[], recommendations=[rec],
    )


def test_scoring_compares_the_fairness_verdict_with_the_population_and_skips_an_unknown_graph(credit_config):
    truth = harness.Truth(edges=None, effects={"duration_months": 0.005}, roi={}, fairness_ratio=0.75)

    flagged = harness.score_run(_outputs(0.70, False), credit_config, truth)
    assert flagged.values["fairness_flag_correct"] == 1.0  # both below the 0.8 threshold
    assert flagged.values["fairness_ratio_abs_err"] == pytest.approx(0.05)
    assert "edge_precision" not in flagged.values and "shd" not in flagged.values  # graph not scored
    assert flagged.fairness_ratio == 0.70 and flagged.fairness_passed is False and flagged.top_choice == "shorten_loan_term"

    lucky = harness.score_run(_outputs(0.85, True), credit_config, truth)
    assert lucky.values["fairness_flag_correct"] == 0.0  # sample passed, population would be flagged

    no_truth = harness.score_run(_outputs(0.70, False), credit_config, None)
    assert no_truth.fairness_ratio == 0.70 and "fairness_flag_correct" not in no_truth.values


def test_scoring_still_scores_the_graph_when_it_is_known(credit_config):
    truth = harness.Truth(edges=[("a", "b")], effects={}, roi={})
    scored = harness.score_run(_outputs(0.9, True), credit_config, truth)
    assert "edge_precision" in scored.values and "fairness_flag_correct" not in scored.values


# --- bootstrap stability for real data ----------------------------------------------------


def test_bootstrap_resamples_the_real_rows_with_unique_ids_and_fixed_seeds(credit_config, real_df, monkeypatch):
    seen = []

    def capture(label, cfg, path, truth, with_baselines=False):
        seen.append((label, pd.read_csv(path), truth))
        return harness.RunResult(label=label, n_rows=1)

    monkeypatch.setattr(harness, "_run_one", capture)
    runs = harness._run_bootstrap({}, credit_config, runner.resolve_data_path(credit_config, "real"), 2)

    assert [r.label for r in runs] == ["bootstrap 1000", "bootstrap 1001"]
    (_, first, truth), (_, second, _) = seen
    assert truth is None and len(first) == len(real_df)
    assert list(first["applicant_id"]) == list(range(1, 1001))
    assert not first.drop(columns="applicant_id").equals(second.drop(columns="applicant_id"))
    resampled = first.drop(columns="applicant_id").drop_duplicates()
    original = real_df.drop(columns="applicant_id")
    assert len(resampled.merge(original, how="inner")) == len(resampled)  # only real rows
    assert len(resampled) < len(original)  # with replacement: some rows repeat, some are absent

    seen.clear()
    harness._run_bootstrap({}, credit_config, runner.resolve_data_path(credit_config, "real"), 1)
    pd.testing.assert_frame_equal(seen[0][1], first)  # the same seed gives the same resample


def test_datasets_with_an_scm_get_scm_replicates_not_a_bootstrap(monkeypatch, tmp_path, credit_config):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)  # hide the data files
    monkeypatch.setattr(harness, "_run_replicates", lambda *a, **k: [])
    monkeypatch.setattr(harness, "_run_bootstrap", lambda *a, **k: pytest.fail("bootstrap used on an SCM dataset"))

    scenarios = harness.evaluate_dataset(DOMAIN, credit_config, "semi_synthetic", replicates=2)

    assert [s.source for s in scenarios] == ["committed file", "SCM replicates"]


def test_real_data_without_replicates_gets_no_bootstrap_and_missing_files_get_none(monkeypatch, tmp_path, credit_config):
    monkeypatch.setattr(harness, "_run_one", lambda *a, **k: harness.RunResult(label="x", n_rows=1))
    monkeypatch.setattr(harness, "_run_bootstrap", lambda *a, **k: [harness.RunResult(label="b", n_rows=1)])

    assert [s.source for s in harness.evaluate_dataset(DOMAIN, credit_config, "real")] == ["committed file"]
    with_boot = harness.evaluate_dataset(DOMAIN, credit_config, "real", replicates=3)
    assert [s.source for s in with_boot] == ["committed file", harness.BOOTSTRAP_SOURCE]
    assert with_boot[1].scored_against_truth is False and with_boot[1].truth is None

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)  # file absent: nothing to resample
    assert [s.source for s in harness.evaluate_dataset(DOMAIN, credit_config, "real", replicates=3)] == ["committed file"]


# --- report -------------------------------------------------------------------------------


def _run(label, ate, ratio, passed, choice, placebo=True) -> harness.RunResult:
    return harness.RunResult(
        label=label, n_rows=1000, estimated_ates={"duration_months": ate},
        refutation_passed={"duration_months": placebo}, refutation_p_values={"duration_months": 0.01},
        fairness_ratio=ratio, fairness_passed=passed, top_choice=choice,
    )


def test_report_renders_fairness_and_stability_sections_from_run_results():
    committed = harness.ScenarioResult(DOMAIN, "real", "committed file", False, runs=[_run("committed file", 0.0047, 0.686, False, "a")])
    boot = harness.ScenarioResult(
        DOMAIN, "real", harness.BOOTSTRAP_SOURCE, False,
        runs=[_run("b1", 0.0050, 0.70, False, "a"), _run("b2", 0.0040, 0.82, True, "a"), _run("b3", -0.0010, 0.65, False, "b", False)],
    )

    md = report.render_markdown([committed, boot], {"replicates": 3, "n_rows": None})

    assert "## Fairness check (Stage 6)" in md
    assert "german_credit / real — stability under resampling" in md
    assert "flagged below the threshold in 2/3" in md  # ratios 0.70, 0.82, 0.65 against 0.8: two of three flagged
    assert "(full sample 0.686)" in md and "a ×2, b ×1 (full sample: a)" in md
    assert "| duration_months | +0.0047 |" in md and "2/3" in md  # same sign as the full-sample estimate: 2 of 3
    assert "Bootstrap stability" in md  # the definition is explained


def test_report_fairness_table_shows_the_truth_and_how_often_the_verdict_matches():
    truth = harness.Truth(edges=None, effects={}, roi={}, fairness_ratio=0.686)
    right = harness.RunResult(label="s", n_rows=1, fairness_ratio=0.70, fairness_passed=False,
                              values={"fairness_flag_correct": 1.0, "fairness_ratio_abs_err": 0.014})
    wrong = harness.RunResult(label="t", n_rows=1, fairness_ratio=0.83, fairness_passed=True,
                              values={"fairness_flag_correct": 0.0, "fairness_ratio_abs_err": 0.144})
    scenario = harness.ScenarioResult(DOMAIN, "semi_synthetic", "SCM replicates", True, runs=[right, wrong], truth=truth)

    md = report.render_markdown([scenario], {"replicates": 2, "n_rows": None})

    assert "| 0.686 |" in md and "1/2" in md  # true ratio; one flagged; verdict matches truth in one of two
    assert "Edge P" in md and "stability under resampling" not in md


# --- through the real pipeline ------------------------------------------------------------


def test_real_credit_data_runs_end_to_end_and_the_age_flag_fires(credit_config, isolated_feast_root):
    real, *rest = harness.evaluate_dataset(DOMAIN, quick(credit_config), "real")

    assert rest == [] and real.truth is None and not real.scored_against_truth and real.reference == []
    (run,) = real.runs
    assert run.error is None, run.error
    assert run.n_rows == 1000 and set(run.estimated_ates) == set(LEVERS)
    assert all(ate > 0 for ate in run.estimated_ates.values())  # longer / bigger / heavier loans default more
    assert run.fairness_ratio == pytest.approx(0.686, abs=0.001) and run.fairness_passed is False
    assert run.top_choice in {c["id"] for c in credit_config["interventions"]["candidates"]}
    assert not {"ate_mae", "edge_precision", "fairness_flag_correct"} & set(run.values)  # nothing to score against


def test_semi_synthetic_runs_end_to_end_scored_on_effects_ranking_and_fairness_but_not_the_graph(
    credit_config, isolated_feast_root
):
    committed, replicates = harness.evaluate_dataset(DOMAIN, quick(credit_config), "semi_synthetic", replicates=1)

    assert committed.scored_against_truth and committed.truth.edges is None
    (run,) = committed.runs
    assert run.error is None, run.error
    assert {"ate_mae", "top1_correct", "rank_tau", "roi_regret", "fairness_flag_correct"} <= set(run.values)
    assert not {"edge_precision", "edge_recall", "shd"} & set(run.values)
    assert run.values["ate_mae"] < 0.01  # true effects are 0.005-0.05; n=1000 is noisy but not this noisy
    assert all(row.sign_agrees for row in run.effect_rows)

    (rep,) = replicates.runs
    assert rep.error is None and rep.n_rows == CREDIT_N_ROWS

    md = report.render_markdown([committed, replicates], {"replicates": 1, "n_rows": None})
    assert "## Fairness check (Stage 6)" in md and "Verdict matches truth" in md
    assert "Every run recovered" not in md  # no graph section for an unknown graph
