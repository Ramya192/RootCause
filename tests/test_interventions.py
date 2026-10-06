"""Stage 6: Interventions (causal_engine/pipeline/interventions.py).

`test_recommendations_not_empty_regression` guards the exact silent-failure
bug documented in memory: effect_estimation.treatments and
interventions.candidates[].target_variable must name the same variables, or
rank_interventions() silently returns an empty list instead of raising.
"""

from __future__ import annotations

from causal_engine.models.schemas import EffectEstimate
from causal_engine.pipeline import interventions


def test_recommendations_not_empty_regression(raw_df, effect_estimates, domain_config):
    recs = interventions.rank_interventions(raw_df, effect_estimates, domain_config)
    candidates = domain_config["interventions"]["candidates"]
    assert len(recs) == len(candidates)


def test_ranks_are_contiguous_and_sorted_by_roi_desc(raw_df, effect_estimates, domain_config):
    recs = interventions.rank_interventions(raw_df, effect_estimates, domain_config)
    assert [r.rank for r in recs] == list(range(1, len(recs) + 1))
    rois = [r.roi for r in recs]
    assert rois == sorted(rois, reverse=True)


def test_gender_population_is_fair_by_construction(raw_df, effect_estimates, domain_config):
    # gender is deliberately not wired into any structural equation in
    # scripts/data/generate_synthetic_attrition_data.py, so the fairness check
    # should find no disparate impact.
    recs = interventions.rank_interventions(raw_df, effect_estimates, domain_config)
    assert all(r.fairness_pass for r in recs)
    assert all(r.fairness_ratio >= domain_config["interventions"]["fairness_threshold"] for r in recs)


def test_skips_candidates_with_no_matching_effect_estimate(raw_df, domain_config):
    # Only one effect estimate, for a treatment none of the configured
    # candidates target -- every candidate should be skipped, not guessed.
    partial_estimates = [
        EffectEstimate(treatment="not_a_real_lever", outcome="attrition", ate=-1.0, estimator="x")
    ]
    recs = interventions.rank_interventions(raw_df, partial_estimates, domain_config)
    assert recs == []


def test_a_candidate_with_no_effect_estimate_is_skipped_with_a_warning(
    raw_df, effect_estimates, domain_config, caplog
):
    dropped = effect_estimates[0].treatment
    with caplog.at_level("WARNING", logger=interventions.logger.name):
        recs = interventions.rank_interventions(raw_df, effect_estimates[1:], domain_config)
    assert dropped not in [r.target_variable for r in recs]
    assert any(dropped in record.getMessage() and "skipped" in record.getMessage() for record in caplog.records)


def test_every_candidate_helps_in_the_attrition_simulation(raw_df, effect_estimates, domain_config):
    recs = interventions.rank_interventions(raw_df, effect_estimates, domain_config)
    assert all(r.recommended and r.expected_effect > 0 for r in recs)


def test_an_action_that_would_raise_the_outcome_is_listed_but_not_recommended(raw_df, effect_estimates, domain_config):
    candidate = domain_config["interventions"]["candidates"][0]
    wrong_way = 0.1 if candidate["expected_shift"] > 0 else -0.1  # the shift now moves the outcome up
    flipped = [
        e.model_copy(update={"ate": wrong_way}) if e.treatment == candidate["target_variable"] else e
        for e in effect_estimates
    ]
    recs = interventions.rank_interventions(raw_df, flipped, domain_config)
    bad = next(r for r in recs if r.id == candidate["id"])
    assert not bad.recommended and bad.expected_effect < 0 and bad.roi < 0
    assert recs[-1].id == bad.id  # ranked last
    assert all(r.recommended for r in recs if r.id != bad.id)


def test_a_non_positive_cost_is_a_config_error_not_a_silent_zero_roi(raw_df, effect_estimates, domain_config):
    import copy

    import pytest

    broken = copy.deepcopy(domain_config)
    broken["interventions"]["candidates"][0]["cost"] = 0
    with pytest.raises(ValueError, match="cost must be positive"):
        interventions.rank_interventions(raw_df, effect_estimates, broken)
