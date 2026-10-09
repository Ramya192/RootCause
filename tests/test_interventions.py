"""Stage 6: Interventions (causal_engine/pipeline/interventions.py).

`test_recommendations_not_empty_regression` guards the exact silent-failure
bug documented in memory: effect_estimation.treatments and
interventions.candidates[].target_variable must name the same variables, or
rank_interventions() silently returns an empty list instead of raising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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


# --- equalized odds (a prediction model's error rates by group) ---

def _frame(n=2000, seed=0, base_rate_a=0.3, base_rate_b=0.3):
    """Outcome independent of everything but the group's base rate; `proxy` tracks the group, `signal` tracks the outcome."""
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, n)
    rate = np.where(group == 1, base_rate_b, base_rate_a)
    y = (rng.random(n) < rate).astype(int)
    return pd.DataFrame({
        "row_id": np.arange(n),
        "group": group,
        "signal": y + rng.normal(0, 0.7, n),
        "proxy": group + rng.normal(0, 0.3, n),
        "outcome": y,
    })


def test_equalized_odds_is_small_when_the_model_treats_the_groups_alike():
    gap = interventions._equalized_odds(_frame(), "outcome", "group", "row_id")

    assert gap is not None and gap < 0.1


def test_equalized_odds_flags_a_model_that_leans_on_a_proxy_for_the_group():
    # group 1 has a much higher base rate and a proxy column, so the model flags its members more often,
    # including the ones who did not have the outcome: a large false-positive-rate gap.
    gap = interventions._equalized_odds(_frame(base_rate_a=0.1, base_rate_b=0.6), "outcome", "group", "row_id")

    assert gap is not None and gap > 0.3


def test_equalized_odds_is_not_computed_when_a_group_has_too_few_positives():
    df = _frame()
    df.loc[df["group"] == 1, "outcome"] = 0
    df.loc[df.index[df["group"] == 1][:2], "outcome"] = 1  # two positives in one group: a rate would be noise

    assert interventions._equalized_odds(df, "outcome", "group", "row_id") is None


def test_equalized_odds_is_not_computed_for_a_constant_outcome_or_a_single_group():
    df = _frame()
    assert interventions._equalized_odds(df.assign(outcome=1), "outcome", "group", "row_id") is None
    assert interventions._equalized_odds(df.assign(group=0), "outcome", "group", "row_id") is None


def test_equalized_odds_is_reproducible_and_uses_a_capped_sample_on_large_data(monkeypatch):
    monkeypatch.setattr(interventions, "_EQUALIZED_ODDS_MAX_ROWS", 500)
    df = _frame(n=3000)

    first = interventions._equalized_odds(df, "outcome", "group", "row_id")
    second = interventions._equalized_odds(df, "outcome", "group", "row_id")

    assert first == second and first is not None


def test_equalized_odds_handles_text_columns_and_missing_values():
    df = _frame()
    df["colour"] = np.where(df["signal"] > 0.5, "red", "blue")
    df.loc[df.index[:50], "signal"] = np.nan

    assert interventions._equalized_odds(df, "outcome", "group", "row_id") is not None


def test_equalized_odds_drops_id_like_text_columns_instead_of_one_hot_encoding_them(monkeypatch):
    # One distinct value per record, one-hot encoded, would be a records-by-records matrix (it exhausted memory on
    # the Freddie Mac file); it must be dropped and the model fitted on the remaining columns.
    df = _frame(n=1500)
    df["loan_ref"] = [f"L{i:06d}" for i in range(len(df))]
    seen = {}
    real_fit = interventions.cross_val_predict

    def spy(model, X, y, **kw):
        seen["columns"] = list(X.columns)
        return real_fit(model, X, y, **kw)

    monkeypatch.setattr(interventions, "cross_val_predict", spy)

    assert interventions._equalized_odds(df, "outcome", "group", "row_id") is not None
    assert seen["columns"] == ["signal", "proxy"]


def test_recommendations_carry_the_equalized_odds_result_and_it_does_not_change_the_ranking(
    raw_df, effect_estimates, domain_config
):
    recs = interventions.rank_interventions(raw_df, effect_estimates, domain_config)

    assert all(r.equalized_odds_difference is not None for r in recs)
    assert len({r.equalized_odds_difference for r in recs}) == 1  # a population-level fact, like the four-fifths ratio
    assert all(r.equalized_odds_pass == (r.equalized_odds_difference <= 0.1) for r in recs)
    assert [r.rank for r in recs] == list(range(1, len(recs) + 1))


def test_the_equalized_odds_threshold_comes_from_the_domain_config(raw_df, effect_estimates, domain_config):
    strict = {**domain_config, "interventions": {**domain_config["interventions"], "equalized_odds_threshold": 0.0}}
    lenient = {**domain_config, "interventions": {**domain_config["interventions"], "equalized_odds_threshold": 1.0}}

    assert not any(r.equalized_odds_pass for r in interventions.rank_interventions(raw_df, effect_estimates, strict))
    assert all(r.equalized_odds_pass for r in interventions.rank_interventions(raw_df, effect_estimates, lenient))


def test_recommendation_schema_allows_equalized_odds_to_be_absent():
    from causal_engine.models.schemas import InterventionRecommendation

    rec = InterventionRecommendation(id="a", target_variable="x", expected_effect=0.1, cost=1.0, roi=0.1, rank=1)

    assert rec.equalized_odds_difference is None and rec.equalized_odds_pass is None


# --- an action whose effect estimate failed its placebo check is not recommended ---


def _two_estimates(second_passed):
    return [
        EffectEstimate(treatment="manager_quality", outcome="attrition", ate=-0.135, estimator="e", refutation_passed=True),
        EffectEstimate(treatment="compensation", outcome="attrition", ate=-0.01, estimator="e", refutation_passed=second_passed,
                       refutation_p_value=0.53),
    ]


def test_an_action_whose_effect_failed_the_placebo_check_is_flagged_not_recommended(raw_df, domain_config):
    recs = {r.target_variable: r for r in interventions.rank_interventions(raw_df, _two_estimates(False), domain_config)}

    supported, unsupported = recs["manager_quality"], recs["compensation"]
    assert supported.recommended and supported.not_recommended_reason is None
    assert not unsupported.recommended
    assert unsupported.not_recommended_reason == "effect_not_distinguishable_from_noise"
    assert unsupported.expected_effect > 0  # it still shows what the (unreliable) estimate says


def test_a_failed_placebo_check_is_reported_as_noise_even_when_the_estimate_points_the_wrong_way(raw_df, domain_config):
    # compensation's ate sign is flipped so the estimate would raise the outcome (the Illinois null-trial case)
    def reason(passed):
        est = [EffectEstimate(treatment="compensation", outcome="attrition", ate=0.01, estimator="e", refutation_passed=passed)]
        (rec,) = [r for r in interventions.rank_interventions(raw_df, est, domain_config) if r.target_variable == "compensation"]
        assert not rec.recommended
        return rec.not_recommended_reason

    assert reason(False) == "effect_not_distinguishable_from_noise"  # not an invented "raises the outcome"
    assert reason(True) == "raises_outcome"  # a supported estimate that raises the outcome says so


def test_an_unsupported_action_is_still_ranked_by_roi_so_the_ranking_evaluation_is_unchanged(raw_df, domain_config):
    recs = interventions.rank_interventions(raw_df, _two_estimates(False), domain_config)
    passed = interventions.rank_interventions(raw_df, _two_estimates(True), domain_config)

    assert [r.id for r in recs] == [r.id for r in passed]
    assert all(r.recommended for r in passed)


def test_the_narrative_recommends_the_best_action_whose_effect_held_up(raw_df, domain_config):
    from causal_engine.pipeline import explanation

    estimates = [
        EffectEstimate(treatment="manager_quality", outcome="attrition", ate=-0.01, estimator="e", refutation_passed=False),
        EffectEstimate(treatment="compensation", outcome="attrition", ate=-0.02, estimator="e", refutation_passed=True),
        EffectEstimate(treatment="workload", outcome="attrition", ate=-0.12, estimator="e", refutation_passed=True),  # rebalancing would raise it
    ]
    recs = interventions.rank_interventions(raw_df, estimates, domain_config)
    best = explanation._top_recommended(recs)

    assert best.target_variable == "compensation"  # not the higher-ranked manager_quality, whose check failed
    text = explanation._template_narrative(domain_config, estimates, [], recs, {"compensation": 1.0})
    assert "Recommended action: 'compensation_adjustment'" in text


def test_every_action_unsupported_gives_the_no_recommendation_message(raw_df, domain_config):
    from causal_engine.pipeline import explanation

    estimates = [
        EffectEstimate(treatment=t, outcome="attrition", ate=-0.01, estimator="e", refutation_passed=False)
        for t in ("manager_quality", "compensation", "workload")
    ]
    recs = interventions.rank_interventions(raw_df, estimates, domain_config)

    assert not any(r.recommended for r in recs)
    assert "No candidate action is expected to reduce attrition" in explanation._template_narrative(domain_config, estimates, [], recs, {"a": 1.0})
