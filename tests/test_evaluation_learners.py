"""The Stage 5 learner comparison on the Illinois trial (causal_engine/evaluation/learners.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from causal_engine.evaluation import benchmarks, harness, learners


def _run(learner="t_learner", base="linear", mean=0.0, women=-0.03, men=0.05, sd=0.04) -> learners.LearnerRun:
    return learners.LearnerRun(
        learner=learner, base=base, mean_cate=mean, cate_std=sd,
        subgroup_cates={"male=0": women, "male=1": men, "age50=0": 0.0, "age50=1": 0.0},
        extreme_propensity_share=None if learner in ("t_learner", "s_learner") else 0.0, seconds=0.1,
    )


def test_sex_reversal_needs_women_negative_and_men_positive():
    assert learners._sex_reversal(_run(women=-0.03, men=0.05))
    assert not learners._sex_reversal(_run(women=0.002, men=0.002))  # a constant effect, as a linear S-learner gives
    assert not learners._sex_reversal(_run(women=0.03, men=-0.05))  # the wrong way round
    assert not learners._sex_reversal(learners.LearnerRun("t_learner", "linear", 0, 0, {}, None, 0))  # no subgroups reported


def test_subgroup_mae_uses_only_subgroups_both_sides_have():
    run = _run(women=-0.03, men=0.05)
    truth = {"male=0": -0.03, "male=1": 0.03, "missing=1": 9.0}
    assert learners._subgroup_mae(run, truth) == pytest.approx(0.01)
    assert np.isnan(learners._subgroup_mae(run, {"other=1": 1.0}))


def test_specs_cover_every_learner_with_both_base_models():
    assert len(learners.SPECS) == 10 and {b for _, b in learners.SPECS} == {"linear", "gbm"}
    assert {l for l, _ in learners.SPECS} == {"t_learner", "s_learner", "x_learner", "r_learner", "dr_learner"}


def test_resample_keeps_the_size_reassigns_unique_ids_and_is_seeded():
    raw = pd.DataFrame({"pid": range(1, 51), "x": np.arange(50)})
    a, b = learners.resample(raw, "pid", 1), learners.resample(raw, "pid", 1)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 50 and list(a["pid"]) == list(range(1, 51))
    assert a["x"].nunique() < 50  # with replacement
    assert not a.equals(learners.resample(raw, "pid", 2))


def test_illinois_config_declares_randomized_assignment_for_the_propensity_learners(config_loader):
    cfg = config_loader.get_domain(learners.DOMAIN).extra
    assert cfg["counterfactuals"]["propensity"] == "constant"
    assert cfg["counterfactuals"]["subgroups"][0] == "male"


def _result() -> learners.Result:
    reference = [
        benchmarks.ExperimentalReference("treat", "y", 100, 100, 0.002, 0.012, -0.022, 0.026, 0.0, 0.0),
        benchmarks.ExperimentalReference("treat", "y", 50, 50, -0.033, 0.017, -0.066, 0.0, 0.0, 0.0, subgroup="male=0"),
        benchmarks.ExperimentalReference("treat", "y", 50, 50, 0.050, 0.018, 0.015, 0.085, 0.0, 0.0, subgroup="male=1"),
    ]
    runs = lambda **kw: [_run(l, b, **kw) for l, b in learners.SPECS]  # noqa: E731
    flat = [_run("s_learner", "linear", women=0.002, men=0.002, sd=0.0)]
    return learners.Result(
        reference=reference,
        real=learners.Batch("real trial", runs()[:1] + flat),
        bootstrap=[learners.Batch("b", runs()), learners.Batch("c", runs(women=0.01, men=0.02))],
        mirror=[learners.Batch("m", runs(mean=-0.04, women=-0.05, men=-0.045))],
        mirror_truth_ate=-0.0475,
        mirror_truth_subgroups={"male=0": -0.049, "male=1": -0.0455},
        settings={"replicates": 2, "n_rows": 4834},
    )


def test_report_shows_the_trial_reference_the_sex_pattern_and_the_mirror_bias():
    md = learners.render_markdown(_result())

    assert "**+0.0020** [-0.0220, +0.0260]" in md  # the trial's overall difference in means
    assert "male=0 -0.0330 [-0.0660, +0.0000]" in md  # and its subgroup estimates
    assert "| t_learner / linear | +0.0000 | yes |" in md  # a mean of 0.0 sits inside the trial interval
    assert "reproduced" in md and "not seen" in md  # the hand-built flat S-learner run is the "not seen" one
    assert "| s_learner / linear | +0.0000 | yes | +0.0020 | +0.0020 | not seen |" in md
    assert "| t_learner / linear | +0.0075 | " in md  # mirror bias: -0.0400 estimate minus -0.0475 truth, one draw so no sd


def test_json_round_trips_the_structure():
    payload = json.loads(learners.render_json(_result()))
    assert payload["settings"]["replicates"] == 2
    assert len(payload["bootstrap"]) == 2 and len(payload["real"]["runs"]) == 2
    assert payload["mirror_truth"]["ate"] == pytest.approx(-0.0475)


def test_end_to_end_on_two_learners_and_one_replicate(monkeypatch):
    """The real Illinois file through two learners, one bootstrap and one mirror draw."""
    monkeypatch.setattr(learners, "SPECS", [("t_learner", "linear"), ("s_learner", "linear")])
    monkeypatch.setattr(harness, "TRUTH_DRAWS", 100_000)

    result = learners.run(replicates=1)

    real = {r.spec: r for r in result.real.runs}
    assert set(real) == {"t_learner / linear", "s_learner / linear"}
    assert learners._sex_reversal(real["t_learner / linear"])  # the T-learner sees the trial's sex pattern
    assert real["s_learner / linear"].cate_std == pytest.approx(0.0, abs=1e-12)  # ... and the linear S-learner cannot
    assert abs(real["t_learner / linear"].mean_cate - result.reference[0].estimate) < 0.01
    assert result.mirror_truth_ate < 0 and "male=0" in result.mirror_truth_subgroups
    assert len(result.bootstrap) == 1 and len(result.mirror) == 1
    md = learners.render_markdown(result)
    assert md.count("t_learner / linear") >= 3  # real table, bootstrap table, mirror table
