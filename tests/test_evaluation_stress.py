"""Stress scenarios: the SCMs must do what their docstrings claim (a confounder
really confounds, a U-shape really has no linear signal), each scenario must run
through the real pipeline, and the report/CLI must handle them."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from causal_engine.evaluation import __main__ as cli
from causal_engine.evaluation import harness, metrics, report, stress, stress_report
from causal_engine.evaluation.scm import additive_noise
from causal_engine.evaluation.scms import (
    attrition_scm,
    confounded_attrition_scm,
    nonlinear_monotone_attrition_scm,
    nonlinear_u_shaped_attrition_scm,
)


@pytest.fixture(autouse=True)
def small_truth_draws(monkeypatch):
    monkeypatch.setattr(harness, "TRUTH_DRAWS", 200_000)


def _ols_slope(y: np.ndarray, x: np.ndarray, *controls: np.ndarray) -> float:
    design = np.column_stack([np.ones_like(x), x, *controls])
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


# --- SCM building blocks ----------------------------------------------------------------


def test_additive_noise_applies_the_function_then_noise():
    parents, mechanism = additive_noise(("x",), lambda v: v["x"] ** 2, noise_sd=0.0)
    assert parents == ("x",)
    x = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(mechanism({"x": x}, np.random.default_rng(0), 3), [1.0, 4.0, 9.0])


# --- confounded variant -----------------------------------------------------------------


def test_hidden_and_observed_confounder_draw_identical_data_apart_from_the_column():
    hidden = confounded_attrition_scm(0.6, hidden=True).sample(500, 7)
    observed = confounded_attrition_scm(0.6, hidden=False).sample(500, 7)

    assert "seniority" not in hidden.columns and "seniority" in observed.columns
    pd.testing.assert_frame_equal(hidden, observed[hidden.columns])


def test_confounder_correlates_with_compensation_at_the_stated_strength_and_keeps_unit_variance():
    df = confounded_attrition_scm(0.6, hidden=False).sample(100_000, 1)
    assert df["seniority"].corr(df["compensation"]) == pytest.approx(0.6, abs=0.01)
    assert df["compensation"].std() == pytest.approx(1.0, abs=0.01)


def test_confounded_edges_include_the_confounder_only_when_it_is_observed():
    baseline = {tuple(e) for e in attrition_scm().edges()}
    observed = {tuple(e) for e in confounded_attrition_scm(0.6, hidden=False).edges()}
    hidden = {tuple(e) for e in confounded_attrition_scm(0.6, hidden=True).edges()}

    assert observed == baseline | {("seniority", "compensation"), ("seniority", "attrition")}
    assert hidden == baseline  # a path through a latent node is not an observed edge


def test_ignoring_the_confounder_biases_the_compensation_slope_and_adjusting_removes_it():
    scm = confounded_attrition_scm(0.6, hidden=False)
    df = scm.sample(300_000, 2)
    true = scm.true_effect("compensation", "attrition", n=400_000, seed=3)

    unadjusted = _ols_slope(df["attrition"].to_numpy(float), df["compensation"].to_numpy())
    adjusted = _ols_slope(
        df["attrition"].to_numpy(float), df["compensation"].to_numpy(), df["seniority"].to_numpy()
    )

    assert unadjusted < true - 0.04  # pay looks much more protective than it is
    assert adjusted == pytest.approx(true, abs=0.01)  # (~0.003 residual: linear fit to a sigmoid)


def test_true_effect_of_the_confounded_treatment_is_the_same_whether_or_not_it_is_hidden():
    hidden = confounded_attrition_scm(0.6, hidden=True).true_effect("compensation", "attrition", n=100_000)
    observed = confounded_attrition_scm(0.6, hidden=False).true_effect("compensation", "attrition", n=100_000)
    assert hidden == observed


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_confounder_strength_must_be_in_zero_one(bad):
    with pytest.raises(ValueError, match="strength"):
        confounded_attrition_scm(bad, hidden=True)


# --- nonlinear variants -----------------------------------------------------------------


def test_u_shaped_workload_has_no_linear_signal_yet_a_real_effect():
    scm = nonlinear_u_shaped_attrition_scm()
    df = scm.sample(200_000, 1)

    assert abs(df["workload"].corr(df["burnout"])) < 0.01
    # E[0.6*((w+1)^2 - w^2)] = 0.6*E[2w + 1] = 0.6 -- and it reaches attrition
    assert scm.true_effect("workload", "burnout", n=200_000) == pytest.approx(0.6, abs=0.01)
    assert scm.true_effect("workload", "attrition", n=200_000) > 0.05


def test_u_shaped_variant_keeps_the_baseline_attrition_rate():
    base = attrition_scm().sample(200_000, 1)["attrition"].mean()
    u_shaped = nonlinear_u_shaped_attrition_scm().sample(200_000, 1)["attrition"].mean()
    assert u_shaped == pytest.approx(base, abs=0.02)


def test_monotone_variant_has_diminishing_returns_to_pay():
    scm = nonlinear_monotone_attrition_scm()
    # E[tanh(c + 1) - tanh(c)] for c ~ N(0,1) is about 0.52, well under the slope of 1 at c = 0
    effect = scm.true_effect("compensation", "job_satisfaction", n=200_000)
    assert 0.3 < effect < 0.7
    df = scm.sample(50_000, 2)
    assert df["workload"].corr(df["burnout"]) > 0.5  # monotone, so still strongly linear-visible


# --- scenario registry ------------------------------------------------------------------


def test_scenario_ids_are_unique_and_baseline_comes_first():
    ids = [s.id for s in stress.STRESS_SCENARIOS]
    assert len(ids) == len(set(ids))
    assert ids[0] == "baseline"


def test_select_returns_all_or_the_named_in_registry_order_and_rejects_unknown():
    assert [s.id for s in stress.select()] == [s.id for s in stress.STRESS_SCENARIOS]
    picked = stress.select(["small_n_250", "baseline"])
    assert [s.id for s in picked] == ["baseline", "small_n_250"]
    with pytest.raises(ValueError, match="unknown stress scenario"):
        stress.select(["nope"])


def test_declare_confounder_rewrites_a_copy_of_the_config(domain_config):
    before = copy.deepcopy(domain_config)
    cfg = stress.declare_confounder(domain_config)

    assert domain_config == before  # the shared session config is not mutated
    assert cfg["feature_store"]["feature_columns"][-1] == "seniority"
    variables = cfg["causal_discovery"]["variables"]
    assert variables.index("seniority") == variables.index("attrition") - 1
    confounders = {t["name"]: t["confounders"] for t in cfg["effect_estimation"]["treatments"]}
    assert confounders == {"compensation": ["seniority"], "manager_quality": [], "workload": []}


@pytest.mark.parametrize("scenario", stress.STRESS_SCENARIOS, ids=lambda s: s.id)
def test_every_scenario_produces_the_columns_its_configured_pipeline_needs(scenario, domain_config):
    cfg = scenario.configure(domain_config)
    df = scenario.scm().sample(50, 0)
    needed = {
        *cfg["feature_store"]["feature_columns"],
        cfg["effect_estimation"]["outcome"],
        cfg["interventions"]["sensitive_attribute"],
        *cfg["causal_discovery"]["variables"],
    }
    assert needed <= set(df.columns)


# --- running through the real pipeline --------------------------------------------------


def test_small_n_scenario_runs_end_to_end_at_its_own_size(config_loader, isolated_feast_root):
    (result,) = harness.run_stress(1, ["small_n_250"], loader=config_loader)

    assert result.stress_id == "small_n_250" and result.stress_factor == "sample size"
    assert result.scored_against_truth and result.truth is not None
    (run,) = result.runs
    assert run.error is None, run.error
    assert run.n_rows == 250 and run.label == f"seed {harness.REPLICATE_SEED_BASE}"
    assert {"edge_recall", "ate_mae", "top1_correct"} <= set(run.values)


def test_observed_confounder_scenario_adjusts_through_the_real_pipeline(config_loader, isolated_feast_root):
    (result,) = harness.run_stress(1, ["observed_confounder_0.6"], loader=config_loader)

    (run,) = result.runs
    assert run.error is None, run.error
    assert ("seniority", "compensation") in result.truth.edges
    assert {r.treatment for r in run.effect_rows} == {"compensation", "manager_quality", "workload"}
    # compensation is adjusted for seniority, so it lands near the truth; ignoring it is off by ~0.07
    comp = next(r for r in run.effect_rows if r.treatment == "compensation")
    assert comp.abs_error < 0.04


# --- reporting --------------------------------------------------------------------------


def _fake_result(stress_id="hidden_confounder_0.6", biases=(-0.07, -0.08)) -> harness.ScenarioResult:
    truth = harness.Truth(edges=[("a", "b")], effects={"compensation": -0.11}, roi={})
    runs = []
    for i, bias in enumerate(biases):
        run = harness.RunResult(f"seed {1000 + i}", 500, values={"refutation_pass_rate": 1.0, "edge_precision": 0.75})
        run.effect_rows = [metrics.EffectRow("compensation", -0.11 + bias, -0.11)]
        run.predicted_edges = [("a", "b"), ("b", "a")] if i == 0 else [("a", "b")]
        runs.append(run)
    return harness.ScenarioResult(
        "employee_attrition", f"stress:{stress_id}", "SCM replicates", True, runs=runs, truth=truth,
        stress_id=stress_id, stress_factor="hidden confounder", stress_description="seniority is hidden",
    )


def test_stress_report_shows_signed_bias_edge_failures_and_the_refuter_caveat():
    md = stress_report.render_markdown([_fake_result()], {"replicates": 2})

    assert "| hidden_confounder_0.6 | 500 | 2/2 |" in md
    assert "Bias: compensation" in md
    assert "-0.075 ± 0.007" in md  # mean and sd of the two signed biases
    assert "Spurious edges: b->a (1/2)" in md
    assert "seniority is hidden" in md
    assert "ROI regret" in md
    assert "staying at 1.0 next to a large bias is expected" in md


def test_roi_regret_is_a_percentage_of_the_true_best_so_a_near_tie_reads_as_small():
    truth = harness.Truth(edges=[], effects={}, roi={"a": 10.0, "b": 9.8})
    runs = [
        harness.RunResult("s1", 100, values={"roi_regret": 0.0}),
        harness.RunResult("s2", 100, values={"roi_regret": 0.2}),  # picked b, gave up 2% of a's ROI
        harness.RunResult("s3", 0, error="boom"),  # failed runs are ignored
    ]
    sc = harness.ScenarioResult("d", "stress:x", "SCM replicates", True, runs=runs, truth=truth)

    assert stress_report._regret_percent(sc) == pytest.approx(1.0)
    assert stress_report._regret_percent(harness.ScenarioResult("d", "s", "src", True, runs=runs)) is None


def test_stress_report_lists_failed_runs():
    result = _fake_result()
    result.runs.append(harness.RunResult("seed 9", 0, error="ValueError: boom"))
    md = stress_report.render_markdown([result], {"replicates": 3})
    assert "hidden_confounder_0.6 / seed 9: ValueError: boom" in md


def test_stress_results_serialise_to_json_without_callables():
    payload = json.loads(report.render_json([_fake_result()], {"replicates": 2}))
    (scenario,) = payload["scenarios"]
    assert scenario["stress_id"] == "hidden_confounder_0.6"
    assert "function" not in json.dumps(payload)


# --- CLI --------------------------------------------------------------------------------


def test_cli_rejects_scenario_without_stress_and_stress_without_replicates(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--scenario", "baseline"])
    with pytest.raises(SystemExit):
        cli.main(["--stress", "--replicates", "0"])
    assert "--scenario only applies with --stress" in capsys.readouterr().err


def test_cli_stress_writes_stress_files_and_leaves_results_files_alone(tmp_path, config_loader, isolated_feast_root):
    out = tmp_path / "eval"
    code = cli.main(["--stress", "--scenario", "small_n_250", "--replicates", "1", "--out", str(out)])

    assert code == 0
    assert (out / "stress.md").read_text(encoding="utf-8").startswith("# RootCause stress tests")
    assert json.loads((out / "stress.json").read_text(encoding="utf-8"))["settings"] == {"replicates": 1}
    assert not (out / "results.md").exists()
