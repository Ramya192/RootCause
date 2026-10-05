"""Stage 5 effect heterogeneity: the subgroup means must recover heterogeneity that is
really there, the truth and trial references behind them must be right, and the report
must test the DIFFERENCE between subgroups rather than eyeball each interval."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_engine.evaluation import benchmarks, harness, metrics, report
from causal_engine.evaluation.scm import SCM, Node, additive_noise, bernoulli_of
from causal_engine.models.schemas import CounterfactualResult, SubgroupEffect
from causal_engine.pipeline import counterfactuals, runner

DOMAIN = "illinois_wellness"


@pytest.fixture(autouse=True)
def small_truth_draws(monkeypatch):
    monkeypatch.setattr(harness, "TRUTH_DRAWS", 200_000)


def _cf_config(subgroups=None) -> dict:
    cfg = {"treatment": "w", "outcome": "y", "treatment_threshold": 0.5}
    if subgroups is not None:
        cfg["subgroups"] = subgroups
    return {
        "counterfactuals": cfg,
        "feature_store": {"entity_id_column": "id"},
        "ingestion": {"outcome_column": "y"},
        "domain": {},
    }


def _hetero_frame(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """The program helps only where g == 1 (effect -0.5); g == 0 is untouched."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    w = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    y = -0.5 * w * g + 0.1 * x + rng.normal(0, 0.2, n)
    return pd.DataFrame({"id": range(n), "w": w, "g": g, "x": x, "y": y})


# --- Stage 5 ----------------------------------------------------------------------------


def test_subgroup_means_recover_heterogeneity_that_is_there():
    [result] = counterfactuals.estimate_counterfactuals(_hetero_frame(), _cf_config(["g"]))

    by_key = {s.key: s for s in result.subgroups}
    assert set(by_key) == {"g=0", "g=1"}
    assert by_key["g=0"].mean_cate == pytest.approx(0.0, abs=0.03)
    assert by_key["g=1"].mean_cate == pytest.approx(-0.5, abs=0.03)
    assert by_key["g=0"].n + by_key["g=1"].n == 6000
    assert result.cate_std > 0.15  # the per-unit effects really are spread out
    assert result.mean_cate == pytest.approx(-0.25, abs=0.03)


def test_no_subgroups_configured_gives_none_and_keeps_the_old_output(feature_df, domain_config):
    [result] = counterfactuals.estimate_counterfactuals(feature_df, domain_config)
    assert result.subgroups == []
    assert result.cate_std is not None and result.cate_std >= 0


def test_subgroups_must_be_binary_columns_in_the_data():
    with pytest.raises(ValueError, match="not binary"):
        counterfactuals.estimate_counterfactuals(_hetero_frame(), _cf_config(["x"]))
    with pytest.raises(ValueError, match="not in the data"):
        counterfactuals.estimate_counterfactuals(_hetero_frame(), _cf_config(["nope"]))


def test_a_level_absent_from_the_data_is_skipped_not_reported_empty():
    frame = _hetero_frame().assign(g=1)
    [result] = counterfactuals.estimate_counterfactuals(frame, _cf_config(["g"]))
    assert [s.key for s in result.subgroups] == ["g=1"]


def test_schema_defaults_and_subgroup_key():
    plain = CounterfactualResult(treatment="t", outcome="y", meta_learner="m", mean_cate=0.1, description="d")
    assert plain.cate_std is None and plain.subgroups == []
    assert SubgroupEffect(variable="male", level="1", n=5, mean_cate=0.2).key == "male=1"


# --- true subgroup effects from the SCM ---------------------------------------------------


def _toy_scm(g_after_t: bool = False) -> SCM:
    def y_of(v):
        return 2.0 * v["t"] * v["g"]

    nodes = [Node("t", *(bernoulli_of((), lambda v: 0.5)[0], bernoulli_of((), lambda v: 0.5)[1]))]
    if g_after_t:
        nodes.append(Node("g", *additive_noise(("t",), lambda v: v["t"], noise_sd=0.0)))
    else:
        nodes.insert(0, Node("g", *bernoulli_of((), lambda v: 0.5)))
    nodes.append(Node("y", *additive_noise(("t", "g"), y_of, noise_sd=0.0)))
    return SCM(nodes=tuple(nodes))


def test_true_subgroup_effects_are_the_within_group_do_contrast():
    effects = _toy_scm().true_subgroup_effects("t", "y", "g", n=20_000)
    assert effects == {"0": pytest.approx(0.0), "1": pytest.approx(2.0)}


def test_a_subgroup_variable_that_the_treatment_changes_is_rejected():
    scm = _toy_scm(g_after_t=True)
    assert scm.is_binary("g")
    with pytest.raises(ValueError, match="affected by"):
        scm.true_subgroup_effects("t", "y", "g", n=2000)


def test_a_non_binary_subgroup_variable_is_rejected():
    with pytest.raises(ValueError, match="not a binary node"):
        _toy_scm().true_subgroup_effects("t", "y", "y", n=2000)


def test_wellness_mirror_effect_varies_by_baseline_risk_even_though_the_logit_effect_is_constant():
    from causal_engine.evaluation.scms import illinois_wellness_scm

    effects = illinois_wellness_scm().true_subgroup_effects("treat", "terminated_0119", "age37_49", n=400_000)
    # the 37-49 band has the lowest baseline termination, so the same logit shift moves it least
    assert effects["1"] > effects["0"] and effects["1"] == pytest.approx(-0.035, abs=0.008)


# --- trial subgroup references ------------------------------------------------------------


def test_subgroup_references_are_within_subgroup_differences_in_means(tmp_path):
    path = tmp_path / "d.csv"
    _hetero_frame(4000).assign(w=lambda d: d["w"]).to_csv(path, index=False)
    cfg = _cf_config(["g"]) | {"effect_estimation": {"outcome": "y", "treatments": []}}

    refs = benchmarks.experimental_subgroup_references(path, cfg)

    by = {r.subgroup: r for r in refs}
    assert set(by) == {"g=0", "g=1"}
    assert by["g=0"].estimate == pytest.approx(0.0, abs=0.03) and not by["g=0"].significant
    assert by["g=1"].estimate == pytest.approx(-0.5, abs=0.03) and by["g=1"].significant
    assert by["g=1"].n_treated + by["g=1"].n_control == int((_hetero_frame(4000)["g"] == 1).sum())


def test_no_subgroup_references_without_the_config_key_or_a_binary_treatment(tmp_path):
    path = tmp_path / "d.csv"
    _hetero_frame(100).to_csv(path, index=False)
    assert benchmarks.experimental_subgroup_references(path, _cf_config()) == []
    cont = _cf_config(["g"])
    cont["counterfactuals"]["treatment"] = "x"  # continuous treatment: no arms to difference
    assert benchmarks.experimental_subgroup_references(path, cont) == []


# --- through the real pipeline ------------------------------------------------------------


@pytest.fixture(scope="module")
def wellness_config(config_loader):
    return config_loader.get_domain(DOMAIN).extra


def test_real_trial_subgroup_effects_match_the_trials_own_within_group_estimates(
    wellness_config, isolated_feast_root
):
    real, *_ = harness.evaluate_dataset(DOMAIN, wellness_config, "real")
    (run,) = real.runs
    assert run.error is None, run.error
    assert set(run.subgroup_cates) == set(r.subgroup for r in real.subgroup_reference)
    assert len(real.subgroup_reference) == 8

    # Stage 5's linear T-learner reproduces the trial's within-subgroup estimates
    for ref in real.subgroup_reference:
        assert run.subgroup_cates[ref.subgroup] == pytest.approx(ref.estimate, abs=0.02), ref.subgroup
        assert ref.contains(run.subgroup_cates[ref.subgroup])

    md = report.render_markdown([real], {"replicates": 0, "n_rows": None})
    assert "Effect heterogeneity (Stage 5 subgroups)" in md
    assert "Bonferroni p" in md and "| male |" in md
    male = {r.subgroup: r for r in real.subgroup_reference}
    assert male["male=1"].estimate > male["male=0"].estimate  # men and women differ in sign in this file


def test_single_candidate_domain_reports_no_ranking_metrics(wellness_config, isolated_feast_root):
    committed, *_ = harness.evaluate_dataset(DOMAIN, wellness_config, "synthetic")
    (run,) = committed.runs
    assert run.error is None, run.error
    assert run.ranking is None
    assert not {"top1_correct", "rank_tau", "roi_regret"} & set(run.values)


def test_synthetic_mirror_subgroup_effects_are_scored_against_the_scm(wellness_config, isolated_feast_root):
    committed, *_ = harness.evaluate_dataset(DOMAIN, wellness_config, "synthetic")
    (run,) = committed.runs

    assert set(committed.truth.subgroup_effects) == {f"{v}={lvl}" for v in ("male", "age50", "age37_49", "white") for lvl in "01"}
    assert len(run.subgroup_rows) == 8 and "subgroup_cate_mae" in run.values
    assert run.values["subgroup_cate_mae"] < 0.05

    md = report.render_markdown([committed], {"replicates": 0, "n_rows": None})
    assert "Mean absolute error across subgroups" in md and "| Subgroup | True effect |" in md


def test_config_cost_is_the_papers_first_year_cost_per_person_assigned(wellness_config):
    [candidate] = wellness_config["interventions"]["candidates"]
    assert candidate["cost"] == 152  # $271 per participant x 0.56 who completed screening


# --- the report tests the difference, not each interval ------------------------------------


def _fake_trial_result(est0: float, est1: float, se: float = 0.1) -> harness.ScenarioResult:
    def ref(level, est):
        return benchmarks.ExperimentalReference(
            treatment="treat", outcome="y", n_treated=100, n_control=50, estimate=est, std_error=se,
            ci_low=est - 1.96 * se, ci_high=est + 1.96 * se, outcome_missing_treated=0.0,
            outcome_missing_control=0.0, subgroup=f"g={level}",
        )

    run = harness.RunResult("committed file", 300)
    run.estimated_ates = {"treat": 0.15}
    run.subgroup_cates = {"g=0": est0, "g=1": est1}
    run.mean_cates = {"treat": 0.15}
    run.cate_stds = {"treat": 0.2}
    return harness.ScenarioResult(
        "d", "real", "committed file", False, runs=[run],
        reference=[], subgroup_reference=[ref(0, est0), ref(1, est1)],
    )


def test_interaction_p_value_uses_the_standard_error_of_the_difference():
    sc = _fake_trial_result(0.0, 0.3)  # difference 0.3, se sqrt(0.1^2 + 0.1^2) = 0.1414 -> z 2.12 -> p 0.0339
    detail = report._heterogeneity_detail(sc)

    assert "| g | +0.3000 ± 0.2772 | 0.0339 | 0.0339 | +0.3000 |" in detail
    assert "1 of 1 variables differ at an adjusted p < 0.05" in detail


def test_each_interval_can_include_zero_while_the_difference_is_still_significant():
    # both subgroup intervals cover 0 (est +-0.15, se 0.1) but the two differ by 0.3: p 0.034
    detail = report._heterogeneity_detail(_fake_trial_result(-0.15, 0.15))
    assert detail.count("| yes |") == 2
    assert "1 of 1 variables differ" in detail


def test_no_difference_is_reported_as_no_difference():
    detail = report._heterogeneity_detail(_fake_trial_result(0.05, 0.05))
    assert "0 of 1 variables differ" in detail and "| 1.0000 |" in detail


def test_subgroup_fields_serialise_to_json():
    import json

    sc = _fake_trial_result(0.0, 0.3)
    sc.runs[0].subgroup_rows = [metrics.EffectRow("g=1", 0.3, 0.2)]
    payload = json.loads(report.render_json([sc], {"replicates": 0}))
    assert payload["scenarios"][0]["runs"][0]["subgroup_cates"] == {"g=0": 0.0, "g=1": 0.3}
    assert payload["scenarios"][0]["subgroup_reference"][0]["subgroup"] == "g=0"
