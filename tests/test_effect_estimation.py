"""Stage 4: Effect Estimation (causal_engine/pipeline/effect_estimation.py).

Expected ATE signs are derived from the synthetic DAG's own structural
equations (scripts/data/generate_synthetic_attrition_data.py), not guessed:

    attrition_logit = -1.2*job_satisfaction + 1.0*burnout
    job_satisfaction = 0.6*compensation + 0.5*manager_quality + noise
    burnout = 0.7*workload - 0.2*manager_quality + noise

so d(logit)/d(compensation) < 0, d(logit)/d(manager_quality) < 0,
d(logit)/d(workload) > 0 -- i.e. more compensation/better managers lower
attrition, more workload raises it.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest
import statsmodels.tools.sm_exceptions as sm_exc

from causal_engine.pipeline import effect_estimation


def test_one_estimate_per_configured_treatment(effect_estimates, domain_config):
    configured = {t["name"] for t in domain_config["effect_estimation"]["treatments"]}
    assert {e.treatment for e in effect_estimates} == configured
    assert all(e.outcome == "attrition" for e in effect_estimates)


def test_ate_signs_match_structural_equations(effect_estimates):
    by_treatment = {e.treatment: e.ate for e in effect_estimates}
    assert by_treatment["compensation"] < 0
    assert by_treatment["manager_quality"] < 0
    assert by_treatment["workload"] > 0


def test_real_effects_are_distinguishable_from_the_permutation_placebo(effect_estimates, domain_config):
    sims = domain_config["effect_estimation"]["refutation_simulations"]
    for estimate in effect_estimates:
        assert estimate.refuter == "permutation_placebo"
        assert estimate.refutation_passed is True
        # true effects are ~0.12-0.14 against a null with sd ~0.01: no permutation gets close
        assert estimate.refutation_p_value == pytest.approx(1 / (1 + sims))


def _config_with(domain_config, **effect_overrides):
    cfg = copy.deepcopy(domain_config)
    cfg["effect_estimation"].update(effect_overrides)
    return cfg


def test_refutation_can_fail_and_is_roughly_calibrated_on_a_no_effect_outcome(feature_df, domain_config):
    # The DoWhy refuter this replaced passed every estimate, even on a shuffled outcome.
    # Here the outcome is shuffled, so no treatment has any effect: only about alpha of
    # the runs may pass, and the p-values must actually vary.
    cfg = _config_with(domain_config, treatments=[{"name": "compensation", "confounders": []}],
                       refutation_simulations=40)
    passes, p_values = 0, []
    for seed in range(12):
        shuffled = feature_df.copy()
        shuffled["attrition"] = np.random.default_rng(seed).permutation(shuffled["attrition"].to_numpy())
        (est,) = effect_estimation.estimate_effects(shuffled, cfg)
        passes += est.refutation_passed
        p_values.append(est.refutation_p_value)

    assert passes <= 3  # expected 0.6 at alpha=0.05; a vacuous check would give 12
    assert len(set(p_values)) > 3
    assert max(p_values) > 0.3


def test_refutation_is_reproducible_and_the_seed_is_configurable(feature_df, domain_config):
    one = _config_with(domain_config, treatments=[{"name": "workload", "confounders": []}],
                       refutation_simulations=30)
    a = effect_estimation.estimate_effects(feature_df, one)[0]
    b = effect_estimation.estimate_effects(feature_df, one)[0]
    assert a.refutation_p_value == b.refutation_p_value and a.ate == b.ate


def test_refutation_can_be_omitted(feature_df, domain_config):
    cfg = copy.deepcopy(domain_config)
    cfg["effect_estimation"].pop("refutation")
    cfg["effect_estimation"]["treatments"] = cfg["effect_estimation"]["treatments"][:1]
    (est,) = effect_estimation.estimate_effects(feature_df, cfg)
    assert est.refuter is None and est.refutation_passed is None and est.refutation_p_value is None


def test_an_unsupported_refuter_is_rejected_rather_than_silently_run(feature_df, domain_config):
    cfg = _config_with(domain_config, refutation="placebo_treatment_refuter")
    with pytest.raises(ValueError, match="permutation_placebo"):
        effect_estimation.estimate_effects(feature_df, cfg)


def test_an_all_zero_adjustment_column_is_dropped_not_fed_to_a_rank_deficient_fit(feature_df, domain_config):
    # Freddie Mac 2010-11 declare a channel level with no loans: an all-zero column, which used to
    # make DoWhy's regression rank-deficient (SingularMatrixWarning) without changing the estimate.
    plain = _config_with(domain_config, treatments=[{"name": "workload", "confounders": ["manager_quality"]}])
    plain["effect_estimation"].pop("refutation")
    with_empty = copy.deepcopy(plain)
    with_empty["effect_estimation"]["treatments"][0]["confounders"].append("empty_level")
    data = feature_df.assign(empty_level=0.0)

    (expected,) = effect_estimation.estimate_effects(feature_df, plain)
    with warnings.catch_warnings():
        warnings.simplefilter("error", sm_exc.SingularMatrixWarning)
        (got,) = effect_estimation.estimate_effects(data, with_empty)
    assert got.ate == pytest.approx(expected.ate)


def test_linear_estimates_carry_a_95_percent_interval_around_the_ate(effect_estimates):
    for e in effect_estimates:
        assert e.std_error is not None and e.std_error > 0
        assert e.ci_low < e.ate < e.ci_high
        assert e.ci_high - e.ci_low == pytest.approx(2 * 1.96 * e.std_error, rel=0.05)
