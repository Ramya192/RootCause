"""Scoring functions, checked against small hand-computed cases."""

from __future__ import annotations

import pytest

from causal_engine.evaluation import metrics
from causal_engine.models.schemas import EffectEstimate, InterventionRecommendation


def _est(treatment, ate, passed=None):
    return EffectEstimate(
        treatment=treatment, outcome="y", ate=ate, estimator="e", refuter="r", refutation_passed=passed
    )


def _rec(id_, rank):
    return InterventionRecommendation(
        id=id_, target_variable="t", expected_effect=0.0, cost=1.0, roi=0.0, rank=rank
    )


# --- structural Hamming distance ---------------------------------------------------


def test_shd_counts_a_reversal_once_and_missing_or_extra_edges_once_each():
    true = [("a", "b"), ("b", "c")]
    pred = [("b", "a"), ("b", "c"), ("c", "d")]  # a-b reversed, c->d spurious
    assert metrics.structural_hamming_distance(pred, true) == 2


def test_shd_is_zero_for_identical_graphs_and_symmetric():
    g = [("a", "b"), ("b", "c")]
    assert metrics.structural_hamming_distance(g, g) == 0
    h = [("a", "b")]
    assert metrics.structural_hamming_distance(g, h) == metrics.structural_hamming_distance(h, g) == 1


# --- graph score --------------------------------------------------------------------


def test_score_graph_precision_recall_f1():
    s = metrics.score_graph(pred := [("a", "b"), ("x", "y")], true := [("a", "b"), ("b", "c")])
    assert s.precision == 0.5 and s.recall == 0.5 and s.f1 == 0.5
    assert s.true_edges_found == 1 and s.true_edges == 2
    assert s.shd == metrics.structural_hamming_distance(pred, true) == 2


def test_score_graph_excluding_priors_measures_only_what_was_discovered():
    # a->b was supplied as a prior; of the rest, x->y is wrong and b->c was missed
    s = metrics.score_graph(
        [("a", "b"), ("x", "y")], [("a", "b"), ("b", "c")], priors=[("a", "b")]
    )
    assert s.discovered_precision == 0.0
    assert s.discovered_recall == 0.0


def test_score_graph_priors_do_not_inflate_the_discovered_scores():
    true = [("a", "b"), ("b", "c"), ("c", "d")]
    with_priors_only = metrics.score_graph([("a", "b")], true, priors=[("a", "b")])
    assert with_priors_only.recall == pytest.approx(1 / 3)  # headline looks non-trivial...
    assert with_priors_only.discovered_recall == 0.0  # ...but nothing was discovered
    assert with_priors_only.discovered_precision is None  # and no non-prior edge was predicted


def test_score_graph_undefined_values_are_none_not_fake_numbers():
    s = metrics.score_graph([], [("a", "b")])
    assert s.precision is None and s.f1 is None
    assert s.recall == 0.0
    assert metrics.score_graph([("a", "b")], [], ).recall is None


# --- effects ------------------------------------------------------------------------


def test_score_effects_pairs_estimates_with_truth_and_skips_unknown_treatments():
    rows = metrics.score_effects(
        [_est("t1", -0.10), _est("t2", 0.30), _est("no_truth", 1.0)], {"t1": -0.12, "t2": 0.25}
    )
    assert [r.treatment for r in rows] == ["t1", "t2"]
    assert rows[0].abs_error == pytest.approx(0.02)
    assert rows[1].abs_error == pytest.approx(0.05)
    assert all(r.sign_agrees for r in rows)


def test_effect_row_detects_a_wrong_sign():
    assert not metrics.EffectRow("t", estimated=0.05, true=-0.05).sign_agrees


def test_an_estimate_of_exactly_zero_has_no_sign_so_it_agrees_with_neither_direction():
    assert not metrics.EffectRow("t", estimated=0.0, true=-0.05).sign_agrees
    assert not metrics.EffectRow("t", estimated=0.0, true=0.05).sign_agrees
    assert metrics.EffectRow("t", estimated=-1e-9, true=-0.05).sign_agrees


# --- refutation ---------------------------------------------------------------------


def test_refutation_pass_rate_ignores_estimates_with_no_refuter():
    ests = [_est("a", 1, True), _est("b", 1, False), _est("c", 1, None)]
    assert metrics.refutation_pass_rate(ests) == 0.5


def test_refutation_pass_rate_is_none_when_nothing_was_refuted():
    assert metrics.refutation_pass_rate([_est("a", 1, None)]) is None


# --- intervention ranking -----------------------------------------------------------


def test_ranking_identical_order_scores_perfectly():
    s = metrics.score_ranking(
        [_rec("A", 1), _rec("B", 2), _rec("C", 3)], {"A": 3.0, "B": 2.0, "C": 1.0}
    )
    assert s.top1_correct and s.chosen == s.true_best == "A"
    assert s.kendall_tau == pytest.approx(1.0)
    assert s.roi_regret == 0.0


def test_ranking_wrong_top_choice_reports_regret_and_tau():
    # pipeline: A > B > C.  truth: B (3) > C (2) > A (1).  pairs: AB disc., AC disc., BC conc. -> tau = -1/3
    s = metrics.score_ranking(
        [_rec("A", 1), _rec("B", 2), _rec("C", 3)], {"A": 1.0, "B": 3.0, "C": 2.0}
    )
    assert not s.top1_correct
    assert (s.chosen, s.true_best) == ("A", "B")
    assert s.kendall_tau == pytest.approx(-1 / 3)
    assert s.roi_regret == pytest.approx(2.0)


def test_ranking_only_compares_candidates_present_on_both_sides():
    s = metrics.score_ranking([_rec("A", 1), _rec("B", 2), _rec("X", 3)], {"A": 2.0, "B": 1.0})
    assert s.top1_correct and s.kendall_tau == pytest.approx(1.0)


def test_ranking_is_none_without_overlap_and_tau_none_for_a_single_candidate():
    assert metrics.score_ranking([_rec("A", 1)], {"Z": 1.0}) is None
    single = metrics.score_ranking([_rec("A", 1)], {"A": 1.0})
    assert single.top1_correct and single.kendall_tau is None
