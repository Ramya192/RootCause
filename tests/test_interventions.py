"""Stage 6: Interventions (rootcause/pipeline/interventions.py).

`test_recommendations_not_empty_regression` guards the exact silent-failure
bug documented in memory: effect_estimation.treatments and
interventions.candidates[].target_variable must name the same variables, or
rank_interventions() silently returns an empty list instead of raising.
"""

from __future__ import annotations

from rootcause.models.schemas import EffectEstimate
from rootcause.pipeline import interventions


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
    # scripts/generate_synthetic_attrition_data.py, so the fairness check
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
