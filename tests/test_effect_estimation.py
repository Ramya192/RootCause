"""Stage 4: Effect Estimation (rootcause/pipeline/effect_estimation.py).

Expected ATE signs are derived from the synthetic DAG's own structural
equations (scripts/generate_synthetic_attrition_data.py), not guessed:

    attrition_logit = -1.2*job_satisfaction + 1.0*burnout
    job_satisfaction = 0.6*compensation + 0.5*manager_quality + noise
    burnout = 0.7*workload - 0.2*manager_quality + noise

so d(logit)/d(compensation) < 0, d(logit)/d(manager_quality) < 0,
d(logit)/d(workload) > 0 -- i.e. more compensation/better managers lower
attrition, more workload raises it.
"""

from __future__ import annotations


def test_one_estimate_per_configured_treatment(effect_estimates, domain_config):
    configured = {t["name"] for t in domain_config["effect_estimation"]["treatments"]}
    assert {e.treatment for e in effect_estimates} == configured
    assert all(e.outcome == "attrition" for e in effect_estimates)


def test_ate_signs_match_structural_equations(effect_estimates):
    by_treatment = {e.treatment: e.ate for e in effect_estimates}
    assert by_treatment["compensation"] < 0
    assert by_treatment["manager_quality"] < 0
    assert by_treatment["workload"] > 0


def test_placebo_refutation_passes_for_every_treatment(effect_estimates):
    for estimate in effect_estimates:
        assert estimate.refuter == "placebo_treatment_refuter"
        assert estimate.refutation_passed is True
