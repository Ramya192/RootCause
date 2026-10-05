"""The SCM behind the evaluation harness: it must reproduce the committed
dataset exactly (so the ground truth is about THAT data), and do() must do
what it claims."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_engine.evaluation.scm import SCM, Node, linear_gaussian, logistic_binary
from causal_engine.evaluation.scms import N_ROWS, SEED, attrition_scm


def _node(name, spec, latent=False):
    parents, mechanism = spec
    return Node(name=name, parents=parents, mechanism=mechanism, latent=latent)


def test_attrition_scm_reproduces_the_committed_csv(data_path):
    committed = pd.read_csv(data_path).drop(columns="employee_id")
    sampled = attrition_scm().sample(N_ROWS, SEED)

    pd.testing.assert_frame_equal(sampled[committed.columns], committed)


def test_attrition_scm_edges_match_ground_truth_json(ground_truth):
    assert {tuple(e) for e in attrition_scm().edges()} == {tuple(e) for e in ground_truth["edges"]}


def test_same_seed_is_reproducible_and_seeds_differ():
    scm = attrition_scm()
    a, b, c = scm.sample(200, 1), scm.sample(200, 1), scm.sample(200, 2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["compensation"].equals(c["compensation"])


def test_shift_moves_the_node_and_its_descendants_but_not_its_ancestors():
    scm = attrition_scm()
    base = scm.sample(500, 3)
    moved = scm.sample(500, 3, shift={"manager_quality": 1.0})

    np.testing.assert_allclose(moved["manager_quality"] - base["manager_quality"], 1.0)
    # job_satisfaction = 0.6*comp + 0.5*manager_quality + noise -> +0.5 exactly (same noise)
    np.testing.assert_allclose(moved["job_satisfaction"] - base["job_satisfaction"], 0.5)
    np.testing.assert_allclose(moved["burnout"] - base["burnout"], -0.2)
    pd.testing.assert_series_equal(moved["compensation"], base["compensation"])
    pd.testing.assert_series_equal(moved["workload"], base["workload"])


def test_set_to_overrides_the_natural_value():
    df = attrition_scm().sample(100, 4, set_to={"workload": 2.0})
    assert (df["workload"] == 2.0).all()


def test_intervening_does_not_disturb_the_noise_of_other_nodes():
    # Common random numbers: the intervened node still consumes its draw, so every
    # node after it sees the same exogenous noise as in the un-intervened run.
    scm = attrition_scm()
    base = scm.sample(300, 5)
    hard = scm.sample(300, 5, set_to={"compensation": 0.0})
    # manager_quality and workload are drawn AFTER compensation; gender after those
    pd.testing.assert_series_equal(hard["manager_quality"], base["manager_quality"])
    pd.testing.assert_series_equal(hard["workload"], base["workload"])
    # job_satisfaction differs only through compensation: same noise, so the gap is exact
    np.testing.assert_allclose(
        hard["job_satisfaction"] - base["job_satisfaction"], -0.6 * base["compensation"]
    )


def test_unknown_or_conflicting_intervention_is_rejected():
    scm = attrition_scm()
    with pytest.raises(KeyError):
        scm.sample(10, 0, shift={"nope": 1.0})
    with pytest.raises(ValueError, match="both"):
        scm.sample(10, 0, shift={"workload": 1.0}, set_to={"workload": 0.0})


def test_true_effect_matches_a_hand_computed_linear_model():
    # y = 2*x + noise, so shifting x by 3 must move E[y] by exactly 6
    scm = SCM(
        nodes=(
            _node("x", linear_gaussian()),
            _node("y", linear_gaussian(x=2.0, noise_sd=1.0)),
        )
    )
    assert scm.true_effect("x", "y", shift=3.0, n=50_000) == pytest.approx(6.0, abs=1e-9)


def test_true_effect_is_total_effect_through_mediators_not_the_direct_coefficient():
    # x -> m -> y: x has NO direct edge to y; its total effect is 0.5 * 4 = 2
    scm = SCM(
        nodes=(
            _node("x", linear_gaussian()),
            _node("m", linear_gaussian(x=0.5)),
            _node("y", linear_gaussian(m=4.0)),
        )
    )
    assert scm.true_effect("x", "y", n=50_000) == pytest.approx(2.0, abs=1e-9)


def test_true_effect_on_the_attrition_outcome_has_the_right_signs_and_size():
    scm = attrition_scm()
    kwargs = dict(n=400_000, seed=11)
    comp = scm.true_effect("compensation", "attrition", **kwargs)
    mgr = scm.true_effect("manager_quality", "attrition", **kwargs)
    load = scm.true_effect("workload", "attrition", **kwargs)

    assert comp < 0 and mgr < 0 and load > 0
    # Independent check (scratch simulation with 2M draws): -0.124 / -0.137 / +0.120
    assert comp == pytest.approx(-0.124, abs=0.004)
    assert mgr == pytest.approx(-0.137, abs=0.004)
    assert load == pytest.approx(0.120, abs=0.004)


def test_true_effect_of_a_non_ancestor_is_exactly_zero():
    scm = SCM(
        nodes=(
            _node("x", linear_gaussian()),
            _node("noise_only", linear_gaussian()),
            _node("y", logistic_binary(x=1.0)),
        )
    )
    # paired draws: shifting a non-parent cannot change y at all, not even by noise
    assert scm.true_effect("noise_only", "y", n=20_000) == 0.0


def test_latent_node_is_simulated_but_hidden_and_excluded_from_edges():
    scm = SCM(
        nodes=(
            _node("u", linear_gaussian(), latent=True),
            _node("t", linear_gaussian(u=1.0, noise_sd=0.1)),
            _node("y", linear_gaussian(u=1.0, noise_sd=0.1)),
        )
    )
    df = scm.sample(5_000, 0)
    assert list(df.columns) == ["t", "y"]
    assert scm.edges() == []  # u is hidden, and t does not cause y
    # the hidden confounder makes t and y correlated in the data, yet do(t) does nothing to y
    assert df["t"].corr(df["y"]) > 0.9
    assert scm.true_effect("t", "y", n=5_000) == 0.0


def test_nodes_must_be_topologically_ordered_and_uniquely_named():
    with pytest.raises(ValueError, match="topological"):
        SCM(nodes=(_node("y", linear_gaussian(x=1.0)), _node("x", linear_gaussian())))
    with pytest.raises(ValueError, match="duplicate"):
        SCM(nodes=(_node("x", linear_gaussian()), _node("x", linear_gaussian())))
