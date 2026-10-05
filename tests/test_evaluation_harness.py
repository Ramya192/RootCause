"""The evaluation harness: truth computation, per-run scoring, replicates,
aggregation and the report. Uses the real pipeline outputs from the shared
session fixtures rather than mocks; only the Monte Carlo size is reduced."""

from __future__ import annotations

import json

import pytest

from causal_engine.evaluation import harness, report
from causal_engine.evaluation.scms import attrition_scm
from causal_engine.pipeline import counterfactuals, interventions, runner


@pytest.fixture(autouse=True)
def small_truth_draws(monkeypatch):
    monkeypatch.setattr(harness, "TRUTH_DRAWS", 200_000)


@pytest.fixture(scope="module")
def truth(domain_config):
    # module-scoped fixtures can't use the function-scoped monkeypatch above
    original, harness.TRUTH_DRAWS = harness.TRUTH_DRAWS, 200_000
    try:
        return harness.compute_truth(attrition_scm(), domain_config)
    finally:
        harness.TRUTH_DRAWS = original


@pytest.fixture(scope="module")
def outputs(raw_df, feature_df, causal_graph, effect_estimates, domain_config):
    return runner.AnalysisOutputs(
        raw=raw_df,
        features=feature_df,
        graph=causal_graph,
        effects=effect_estimates,
        counterfactuals=counterfactuals.estimate_counterfactuals(feature_df, domain_config),
        recommendations=interventions.rank_interventions(raw_df, effect_estimates, domain_config),
    )


# --- truth ----------------------------------------------------------------------------


def test_truth_has_edges_effects_and_roi_for_the_configured_domain(truth, ground_truth):
    assert set(truth.edges) == {tuple(e) for e in ground_truth["edges"]}
    assert set(truth.effects) == {"compensation", "manager_quality", "workload"}
    assert truth.effects["compensation"] < 0 < truth.effects["workload"]
    assert set(truth.roi) == {"manager_training", "compensation_adjustment", "workload_rebalancing"}


def test_true_roi_order_follows_effect_per_dollar(truth):
    # manager_training (-0.137 for $15k) > workload_rebalancing (-0.120 for $20k) > comp (-0.124 for $40k)
    order = sorted(truth.roi, key=truth.roi.get, reverse=True)
    assert order == ["manager_training", "workload_rebalancing", "compensation_adjustment"]


# --- scoring one run ------------------------------------------------------------------


def test_score_run_against_truth_reports_every_metric(outputs, truth, domain_config):
    run = harness.score_run(outputs, domain_config, truth)

    assert run.n_rows == 2000
    assert run.values["edge_recall"] == 1.0 and run.values["edge_precision"] == 1.0
    assert run.values["shd"] == 0.0
    assert run.values["discovered_recall"] == 1.0
    assert run.values["ate_mae"] < 0.02  # ~0.003 observed; loose bound, this is one noisy draw
    assert run.values["ate_sign_agreement"] == 1.0
    assert run.values["top1_correct"] == 1.0
    assert run.values["rank_tau"] == pytest.approx(1.0)
    assert len(run.effect_rows) == 3
    assert run.ranking.chosen == "manager_training"


def test_score_run_without_truth_reports_only_ground_truth_free_metrics(outputs, domain_config):
    run = harness.score_run(outputs, domain_config, truth=None)
    assert set(run.values) == {"refutation_pass_rate"}
    assert run.effect_rows == [] and run.ranking is None


def test_a_failing_run_is_recorded_not_raised(tmp_path, domain_config):
    run = harness._run_one("bad", domain_config, tmp_path / "does_not_exist.csv", truth=None)
    assert run.error and "does_not_exist" in run.error
    assert run.values == {}


# --- aggregation ----------------------------------------------------------------------


def test_summarize_takes_mean_and_sample_sd_and_skips_errors_and_undefined_values():
    runs = [
        harness.RunResult("a", 10, values={"m": 1.0, "u": None}),
        harness.RunResult("b", 10, values={"m": 3.0, "u": None}),
        harness.RunResult("c", 0, error="boom"),
    ]
    summary = harness.summarize(runs)
    assert summary["m"] == (2.0, pytest.approx(1.4142135), 2)
    assert "u" not in summary  # never defined -> absent, not 0


def test_summarize_single_run_has_zero_sd():
    assert harness.summarize([harness.RunResult("a", 5, values={"m": 0.5})])["m"] == (0.5, 0.0, 1)


# --- running datasets -----------------------------------------------------------------


def test_missing_data_file_is_reported_as_skipped_not_crashed(monkeypatch, tmp_path, config_loader):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)  # committed CSV now "missing"

    (scenario,) = harness.run_evaluation(domains=["employee_attrition"], loader=config_loader)

    assert scenario.skipped_reason and "not found" in scenario.skipped_reason
    assert scenario.runs == []


def test_filters_select_only_the_requested_domain_and_dataset(monkeypatch, tmp_path, config_loader):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    assert harness.run_evaluation(domains=["no_such_domain"], loader=config_loader) == []
    assert harness.run_evaluation(datasets=["no_such_kind"], loader=config_loader) == []

    # with the data files hidden, each selected (domain, dataset) is a skipped scenario
    only = harness.run_evaluation(datasets=["semi_synthetic"], loader=config_loader)
    assert {(s.domain_id, s.dataset) for s in only} == {
        ("german_credit", "semi_synthetic"), ("freddie_mac", "semi_synthetic"),
    }
    assert all(s.skipped_reason for s in only)


def test_a_twin_whose_source_download_is_absent_is_skipped_not_a_crash(monkeypatch, tmp_path, config_loader):
    from causal_engine.evaluation import scms

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(scms, "FREDDIE_DATA_PATH", tmp_path / "loans_2007.csv")
    scms._real_freddie_loans.cache_clear()
    try:
        (scenario,) = harness.run_evaluation(
            domains=["freddie_mac"], datasets=["semi_synthetic"], replicates=2, loader=config_loader
        )
    finally:
        scms._real_freddie_loans.cache_clear()
    assert scenario.skipped_reason and "not found" in scenario.skipped_reason
    assert not scenario.runs


def test_replicates_are_fresh_draws_scored_against_the_scm(
    monkeypatch, tmp_path, config_loader, isolated_feast_root
):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)  # skip the committed file; run the replicate path only

    scenarios = harness.run_evaluation(
        domains=["employee_attrition"], replicates=1, n_rows=1000, loader=config_loader
    )

    committed, replicates = scenarios
    assert committed.skipped_reason
    assert replicates.source == "SCM replicates" and replicates.scored_against_truth
    (run,) = replicates.runs
    assert run.error is None, run.error
    assert run.label == f"seed {harness.REPLICATE_SEED_BASE}"
    assert run.n_rows == 1000
    assert {"edge_recall", "ate_mae", "top1_correct"} <= set(run.values)


# --- report ---------------------------------------------------------------------------


def _scenario(runs, **kw):
    return harness.ScenarioResult("dom", "synthetic", kw.pop("source", "SCM replicates"), True, runs=runs, **kw)


def test_report_formats_single_run_replicates_and_top1_counts():
    single = {"edge_recall": (0.5, 0.0, 1)}
    many = {"edge_recall": (0.93, 0.14, 10), "top1_correct": (0.8, 0.4, 10)}
    assert report._fmt(single, "edge_recall") == "0.50"
    assert report._fmt(many, "edge_recall") == "0.93 ± 0.14"
    assert report._fmt(many, "top1_correct") == "8/10"
    assert report._fmt(many, "shd") == report.MISSING


def test_report_renders_skipped_and_failed_scenarios_and_is_valid_json(outputs, truth, domain_config):
    good = harness.score_run(outputs, domain_config, truth)
    good.label = "committed file"
    results = [
        _scenario([good], source="committed file", truth=truth),
        harness.ScenarioResult("dom", "real", "committed file", False, skipped_reason="data file not found: x.csv"),
        _scenario([harness.RunResult("seed 7", 0, error="ValueError: boom")]),
    ]
    settings = {"replicates": 0, "n_rows": 2000}

    md = report.render_markdown(results, settings)

    assert "| dom / synthetic | committed file | 1/1 |" in md
    assert "| dom / real | committed file | skipped |" in md
    assert "| dom / synthetic | SCM replicates | 0/1 |" in md
    assert "data file not found: x.csv" in md and "seed 7: ValueError: boom" in md
    assert "no placebo can detect an unmeasured confounder" in md  # the refuter's limit travels with every report
    assert json.loads(report.render_json(results, settings))["settings"] == settings
