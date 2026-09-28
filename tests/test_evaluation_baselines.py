"""Naive baselines: the regressions and the SHAP ranking must do what their
docstrings claim (all-columns regression loses a mediated effect, treatment-only
regression is fooled by a confounder), and the comparison must run through the
harness, report and CLI."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rootcause.evaluation import __main__ as cli
from rootcause.evaluation import baselines, baselines_report, harness, metrics, report
from rootcause.evaluation.scms import attrition_scm, confounded_attrition_scm


@pytest.fixture(autouse=True)
def small_truth_draws(monkeypatch):
    monkeypatch.setattr(harness, "TRUTH_DRAWS", 200_000)


def _features(scm, n: int, seed: int = 3) -> pd.DataFrame:
    """A Stage-2-shaped frame: entity id, the observed numeric columns, the outcome."""
    df = scm.sample(n, seed).drop(columns=["gender"])
    df.insert(0, "employee_id", range(1, n + 1))
    return df


def _ate(estimates, treatment: str) -> float:
    return next(e.ate for e in estimates if e.treatment == treatment)


# --- regressions -----------------------------------------------------------------------


def test_all_columns_regression_loses_an_effect_that_runs_through_mediators(domain_config):
    features = _features(attrition_scm(), 60_000)

    all_cols = baselines.regression_effects(features, domain_config, adjust_for_all=True)
    alone = baselines.regression_effects(features, domain_config, adjust_for_all=False)

    # Pay acts on attrition only through job_satisfaction; holding it fixed leaves ~nothing.
    assert abs(_ate(all_cols, "compensation")) < 0.01
    # Without adjustment there is no confounder, so the total effect (about -0.12) comes back.
    assert _ate(alone, "compensation") == pytest.approx(-0.124, abs=0.01)
    assert all_cols[0].estimator == "naive.regression_all_columns"
    assert alone[0].estimator == "naive.regression_treatment_only"


def test_treatment_only_regression_is_biased_by_a_confounder_it_cannot_see(domain_config):
    features = _features(confounded_attrition_scm(0.6, hidden=True), 60_000)

    alone = baselines.regression_effects(features, domain_config, adjust_for_all=False)

    # true compensation effect is about -0.110; seniority makes pay look far more protective
    assert _ate(alone, "compensation") < -0.16


def test_regression_estimates_cover_every_configured_treatment(domain_config):
    features = _features(attrition_scm(), 2000)
    effects = baselines.regression_effects(features, domain_config, adjust_for_all=True)
    assert [e.treatment for e in effects] == ["compensation", "manager_quality", "workload"]


# --- SHAP ranking ----------------------------------------------------------------------


def test_shap_recommendations_rank_by_importance_per_cost_and_ignore_the_shift(domain_config):
    features = _features(attrition_scm(), 3000)

    recs, importances = baselines.shap_recommendations(features, domain_config)

    assert [r.rank for r in recs] == [1, 2, 3]
    assert [r.roi for r in recs] == sorted((r.roi for r in recs), reverse=True)
    for r in recs:
        assert r.roi == pytest.approx(importances[r.target_variable] / r.cost)
    # importance has no direction, so a candidate that lowers its variable is not penalised
    workload = next(r for r in recs if r.target_variable == "workload")
    assert workload.expected_effect > 0


def test_shap_puts_its_weight_on_the_mediators_not_the_levers(domain_config):
    features = _features(attrition_scm(), 5000)

    _, importances = baselines.shap_recommendations(features, domain_config)

    top = max(importances, key=importances.get)
    assert top in {"job_satisfaction", "burnout"}
    assert min(importances["job_satisfaction"], importances["burnout"]) > 3 * max(
        importances["compensation"], importances["manager_quality"], importances["workload"]
    )


# --- scoring ---------------------------------------------------------------------------


def test_score_baselines_scores_all_three_against_the_truth(domain_config):
    features = _features(attrition_scm(), 3000)
    raw = features.assign(gender=["A", "B"] * 1500)
    effects = {"compensation": -0.124, "manager_quality": -0.137, "workload": 0.121}
    roi = {"manager_training": 9.1e-6, "workload_rebalancing": 6.0e-6, "compensation_adjustment": 3.1e-6}

    scored = baselines.score_baselines(raw, features, domain_config, effects, roi)

    assert set(scored) == {baselines.REGRESSION_ALL, baselines.REGRESSION_TREATMENT_ONLY, baselines.SHAP_IMPORTANCE}
    alone = scored[baselines.REGRESSION_TREATMENT_ONLY]
    assert {r.treatment for r in alone.effect_rows} == set(effects)
    assert alone.values["ate_sign_agreement"] == 1.0
    assert alone.ranking.top1_correct
    shap_run = scored[baselines.SHAP_IMPORTANCE]
    assert shap_run.effect_rows == [] and "ate_mae" not in shap_run.values  # SHAP has no ATE
    assert shap_run.top_feature in {"job_satisfaction", "burnout"}
    assert shap_run.values["top_feature_is_lever"] == 0.0
    assert set(shap_run.importances) >= {"compensation", "job_satisfaction"}


# --- running through the harness -------------------------------------------------------


def test_baselines_are_off_unless_asked_for(config_loader, isolated_feast_root):
    (result,) = harness.run_stress(1, ["small_n_250"], loader=config_loader)
    assert result.runs[0].baselines == {}


def test_baselines_run_on_the_same_draw_as_the_pipeline(config_loader, isolated_feast_root):
    (result,) = harness.run_stress(1, ["small_n_250"], loader=config_loader, with_baselines=True)

    (run,) = result.runs
    assert run.error is None, run.error
    assert set(run.baselines) == {
        baselines.REGRESSION_ALL, baselines.REGRESSION_TREATMENT_ONLY, baselines.SHAP_IMPORTANCE
    }
    # no confounder declared in this scenario, so the pipeline IS a treatment-only regression
    alone = run.baselines[baselines.REGRESSION_TREATMENT_ONLY]
    for ours, theirs in zip(run.effect_rows, alone.effect_rows):
        assert ours.treatment == theirs.treatment
        assert ours.estimated == pytest.approx(theirs.estimated, abs=1e-6)


# --- reporting -------------------------------------------------------------------------


def _fake_result() -> harness.ScenarioResult:
    truth = harness.Truth(
        edges=[], effects={"compensation": -0.10}, roi={"cheap": 10.0, "dear": 5.0}
    )
    rank = lambda chosen, ok: metrics.RankingScore(chosen, "cheap", ok, 1.0 if ok else -1.0, 0.0 if ok else 5.0)  # noqa: E731
    runs = []
    for i in range(2):
        run = harness.RunResult(f"seed {1000 + i}", 500)
        run.effect_rows = [metrics.EffectRow("compensation", -0.10, -0.10)]
        run.ranking = rank("cheap", True)
        naive = baselines.BaselineRun(baselines.REGRESSION_ALL)
        naive.effect_rows = [metrics.EffectRow("compensation", 0.01, -0.10)]
        naive.ranking = rank("dear", False)
        shap_run = baselines.BaselineRun(
            baselines.SHAP_IMPORTANCE,
            ranking=rank("dear", False),
            importances={"compensation": 0.1, "job_satisfaction": 0.8},
            top_feature="job_satisfaction",
            values={"top_feature_is_lever": 0.0},
        )
        run.baselines = {baselines.REGRESSION_ALL: naive, baselines.SHAP_IMPORTANCE: shap_run}
        runs.append(run)
    return harness.ScenarioResult(
        "employee_attrition", "stress:x", "SCM replicates", True, runs=runs, truth=truth,
        stress_id="x", stress_factor="f", stress_description="d",
    )


def test_report_lines_the_methods_up_per_scenario_with_bias_sign_and_regret():
    md = baselines_report.render_markdown([_fake_result()], {"replicates": 2})

    assert "| x | RootCause pipeline | +0.000 ± 0.000 | 2/2 | 2/2 | 1.00 ± 0.00 | 0.0% |" in md
    # +0.01 against a -0.10 truth: bias +0.110, wrong sign both times, gave up half the best ROI
    assert "| x | Regression, all columns | +0.110 ± 0.000 | 0/2 | 0/2 | -1.00 ± 0.00 | 50.0% |" in md
    assert "| x | SHAP importance ÷ cost | – | – | 0/2 |" in md
    assert "| x | cheap | cheap ×2 | dear ×2 | – | dear ×2 |" in md
    assert "| x | 0.100 | 0.800 | 0/2 |" in md  # SHAP weight sits on the mediator; top feature is never a lever
    assert "hidden_confounder_*" in md  # the limits are stated in the report


def test_report_skips_methods_that_were_not_run():
    result = _fake_result()
    for run in result.runs:
        run.baselines = {}
    md = baselines_report.render_markdown([result], {"replicates": 2})
    assert "RootCause pipeline" in md and "Regression, all columns" not in md.split("## Rank-1")[0]


def test_baseline_results_serialise_to_json():
    payload = json.loads(report.render_json([_fake_result()], {"replicates": 2}))
    run = payload["scenarios"][0]["runs"][0]
    assert run["baselines"]["shap_importance"]["top_feature"] == "job_satisfaction"


# --- CLI -------------------------------------------------------------------------------


def test_cli_rejects_stress_together_with_baselines(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--stress", "--baselines"])
    assert "pick one" in capsys.readouterr().err


def test_cli_baselines_writes_baselines_files_only(tmp_path, config_loader, isolated_feast_root):
    out = tmp_path / "eval"
    code = cli.main(["--baselines", "--scenario", "small_n_250", "--replicates", "1", "--out", str(out)])

    assert code == 0
    assert (out / "baselines.md").read_text(encoding="utf-8").startswith("# RootCause vs naive baselines")
    payload = json.loads((out / "baselines.json").read_text(encoding="utf-8"))
    assert payload["scenarios"][0]["runs"][0]["baselines"]
    assert not (out / "stress.md").exists() and not (out / "results.md").exists()


# --- parallel scenarios ----------------------------------------------------------------


def test_parallel_workers_give_the_same_results_as_a_serial_run(config_loader, isolated_feast_root):
    ids = ["small_n_500", "small_n_250"]  # registry order
    serial = harness.run_stress(1, ids, loader=config_loader, with_baselines=True)
    parallel = harness.run_stress(1, ids, with_baselines=True, workers=2)

    assert [sc.stress_id for sc in parallel] == ids  # registry order kept
    for a, b in zip(serial, parallel):
        (run_a,), (run_b,) = a.runs, b.runs
        assert run_b.error is None, run_b.error
        # Compare what the methods estimated, not the scores: workers use the full 1M-draw truth,
        # while this module's autouse fixture shrinks it for the serial run.
        assert run_a.predicted_edges == run_b.predicted_edges
        assert [r.estimated for r in run_a.effect_rows] == pytest.approx([r.estimated for r in run_b.effect_rows])
        for name, base_a in run_a.baselines.items():
            base_b = run_b.baselines[name]
            assert [r.estimated for r in base_a.effect_rows] == pytest.approx([r.estimated for r in base_b.effect_rows])
            assert base_a.importances == pytest.approx(base_b.importances)  # SHAP needs a fixed column order


def test_workers_are_validated(config_loader):
    with pytest.raises(ValueError, match="workers must be >= 1"):
        harness.run_stress(1, ["baseline"], workers=0)
    with pytest.raises(ValueError, match="custom loader"):
        harness.run_stress(1, ["small_n_250", "small_n_500"], loader=config_loader, workers=2)


def test_cli_validates_workers(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--baselines", "--workers", "0"])
    with pytest.raises(SystemExit):
        cli.main(["--workers", "2"])
    err = capsys.readouterr().err
    assert "--workers must be >= 1" in err and "--workers only applies with --stress or --baselines" in err
