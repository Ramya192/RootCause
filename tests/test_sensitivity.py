"""Stage 4 sensitivity analysis (causal_engine/pipeline/sensitivity.py).

The formulas are Cinelli & Hazlett (2020). Reference numbers below were produced by the
authors' `sensemakr` Python port on the same simulated data (kept here as pins, not as a
dependency): partial R^2, RV_q and RV_{q,alpha} agree exactly; the benchmark bounds agree
to about three decimals.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from causal_engine.pipeline import effect_estimation, sensitivity


def _simulate(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2, z = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    d = 0.5 * x1 + 0.4 * x2 + 0.6 * z + rng.normal(size=n)
    y = 0.3 * d + 0.5 * x1 - 0.2 * x2 + 0.7 * z + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, d=d, x1=x1, x2=x2, z=z))


@pytest.fixture(scope="module")
def sim() -> pd.DataFrame:
    return _simulate()


def test_partial_r2_and_robustness_values_match_the_reference_implementation(sim):
    result = sensitivity.linear_sensitivity(sim, "d", "y", ["x1", "x2"])  # z omitted

    assert result.estimate == pytest.approx(0.5890091, abs=1e-6)  # biased up: true effect is 0.3
    assert result.partial_r2_treatment_outcome == pytest.approx(0.2575152, abs=1e-6)
    assert result.robustness_value == pytest.approx(0.4405086, abs=1e-6)
    assert result.robustness_value_alpha == pytest.approx(0.4208975, abs=1e-6)
    assert result.robustness_value_alpha < result.robustness_value  # significance is easier to lose than the sign


def test_benchmark_bound_agrees_with_the_reference_to_three_decimals(sim):
    result = sensitivity.linear_sensitivity(sim, "d", "y", ["x1", "x2"])

    assert result.benchmark_covariate == "x1"  # the stronger of the two observed covariates
    assert result.benchmark_r2_treatment == pytest.approx(0.193546, abs=1e-3)
    assert result.benchmark_r2_outcome == pytest.approx(0.099487, abs=1e-3)
    assert result.benchmark_adjusted_estimate == pytest.approx(0.434466, abs=1e-3)
    assert result.robust_to_benchmark is True


def test_bias_formula_is_exact_at_the_true_confounder_strength(sim):
    """The omitted-variable-bias identity: with z's TRUE partial R^2s, the estimate that omits
    z minus the bound equals the estimate that includes it, to floating point."""
    short = sm.OLS(sim["y"], sm.add_constant(sim[["d", "x1", "x2"]])).fit()
    full = sm.OLS(sim["y"], sm.add_constant(sim[["d", "x1", "x2", "z"]])).fit()
    d_full = sm.OLS(sim["d"], sm.add_constant(sim[["x1", "x2", "z"]])).fit()
    r2_yz = sensitivity.partial_r2_from_t(full.tvalues["z"], full.df_resid)
    r2_dz = sensitivity.partial_r2_from_t(d_full.tvalues["z"], d_full.df_resid)

    adjusted = sensitivity.adjusted_estimate(short.params["d"], short.bse["d"], short.df_resid, r2_dz, r2_yz)

    assert adjusted == pytest.approx(full.params["d"], abs=1e-9)
    assert abs(short.params["d"] - full.params["d"]) > 0.25  # the omission mattered, so the test has teeth


def test_a_null_effect_has_zero_robustness(sim):
    rng = np.random.default_rng(1)
    noise = sim.assign(y=rng.normal(size=len(sim)))  # y unrelated to d
    result = sensitivity.linear_sensitivity(noise, "d", "y", ["x1", "x2"])

    assert result.robustness_value < 0.05 and result.robustness_value_alpha == 0.0
    assert result.partial_r2_treatment_outcome < 0.005


def test_robustness_value_edge_cases():
    assert sensitivity.robustness_value(0.0, 100) == 0.0
    assert sensitivity.robustness_value(50.0, 100) > 0.9  # an overwhelming t needs an overwhelming confounder
    assert sensitivity.robustness_value(1.9, 100, alpha=0.05) == 0.0  # t=1.9 is not significant at 5% (critical ~1.98)
    assert sensitivity.robustness_value(-3.0, 100) == sensitivity.robustness_value(3.0, 100)  # sign-free


def test_unattainable_benchmark_strength_is_reported_as_nan_not_an_error():
    r2_dz, r2_yz = sensitivity.benchmark_partial_r2(0.6, 0.1)  # kd=1 would need r2_dz = 1.5
    assert np.isnan(r2_dz) and np.isnan(r2_yz)


def test_no_covariates_gives_no_benchmark(sim):
    result = sensitivity.linear_sensitivity(sim, "d", "y", [])
    assert result.benchmark_covariate is None and result.robust_to_benchmark is None
    assert result.robustness_value > 0


def test_constant_covariate_is_skipped_rather_than_crashing(sim):
    result = sensitivity.linear_sensitivity(sim.assign(flat=1.0), "d", "y", ["flat", "x1", "x2"])
    assert result.benchmark_covariate in {"x1", "x2"}


def test_constant_covariate_changes_nothing_and_raises_no_rank_warning(sim):
    """A declared-but-unused one-hot level is an all-zero column: dropping it must leave every
    number as it was, and statsmodels must not warn about a rank-deficient design."""
    import warnings

    without = sensitivity.linear_sensitivity(sim, "d", "y", ["x1", "x2"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with_flat = sensitivity.linear_sensitivity(sim.assign(flat=0.0), "d", "y", ["flat", "x1", "x2"])
    assert with_flat == without


# --- through Stage 4 ----------------------------------------------------------------------


def test_stage4_reports_sensitivity_and_its_estimate_is_the_dowhy_estimate(feature_df, domain_config, effect_estimates):
    for estimate in effect_estimates:
        assert estimate.sensitivity is not None
        assert estimate.sensitivity.estimate == pytest.approx(estimate.ate, abs=1e-8)  # same OLS as DoWhy's
        assert 0.0 <= estimate.sensitivity.robustness_value <= 1.0
        # the attrition config declares no confounders for any treatment, so there is nothing to benchmark against
        assert estimate.sensitivity.benchmark_covariate is None and estimate.sensitivity.robust_to_benchmark is None


def test_sensitivity_can_be_turned_off_and_is_skipped_for_other_estimators(feature_df, domain_config):
    cfg = copy.deepcopy(domain_config)
    cfg["effect_estimation"]["refutation_simulations"] = 3
    cfg["effect_estimation"]["treatments"] = cfg["effect_estimation"]["treatments"][:1]
    cfg["effect_estimation"]["sensitivity"] = False

    (estimate,) = effect_estimation.estimate_effects(feature_df, cfg)

    assert estimate.sensitivity is None


def test_a_covariate_named_const_does_not_collide_with_the_intercept(sim):
    result = sensitivity.linear_sensitivity(sim.rename(columns={"x2": "const"}), "d", "y", ["x1", "const"])
    reference = sensitivity.linear_sensitivity(sim, "d", "y", ["x1", "x2"])
    assert result.estimate == pytest.approx(reference.estimate)


def test_adjusted_estimate_defaults_to_shrinking_and_takes_an_explicit_direction():
    shrunk = sensitivity.adjusted_estimate(-0.2, 0.01, 1000, 0.3, 0.1)  # default: confounder pushed it away from zero
    assert shrunk > -0.2 and shrunk < 0
    grown = sensitivity.adjusted_estimate(-0.2, 0.01, 1000, 0.3, 0.1, bias_sign=-1.0)  # it actually pushed it up (less negative)
    assert grown == pytest.approx(-0.2 + sensitivity.bias_bound(0.01, 1000, 0.3, 0.1))
    assert sensitivity.adjusted_estimate(0.2, 0.01, 1000, 0.3, 0.1) == pytest.approx(0.2 - sensitivity.bias_bound(0.01, 1000, 0.3, 0.1))


# --- the hidden-confounder check ----------------------------------------------------------


def test_hidden_confounder_check_identity_holds_and_robustness_moves_the_wrong_way(monkeypatch):
    from causal_engine.evaluation import sensitivity_eval

    monkeypatch.setattr(sensitivity_eval, "TRUTH_DRAWS", 100_000)
    result = sensitivity_eval.run(replicates=3, n_rows=1500, strengths=(0.0, 0.85))
    rows = result["rows"]

    assert all(r.identity_error < 1e-9 for r in rows)  # the OVB formula is exact at the true strength
    by_strength = {s: np.mean([r.robustness_value for r in rows if r.strength == s]) for s in (0.0, 0.85)}
    assert by_strength[0.85] > by_strength[0.0]  # a more confounded estimate LOOKS more robust
    assert all(not r.explains_away for r in rows)

    md = sensitivity_eval.render_markdown(result)
    assert "WRONG way" in md and "| 0.85 |" in md
    import json

    assert json.loads(sensitivity_eval.render_json(result))["settings"]["replicates"] == 3


def test_a_covariate_that_explains_the_treatment_exactly_is_skipped_not_a_crash(sim):
    """Perfect collinearity used to divide by zero (an adjustment column that copies the treatment)."""
    data = sim.assign(copy=sim["d"])
    result = sensitivity.linear_sensitivity(data, "d", "y", ["copy", "x1", "x2"])
    assert result.benchmark_covariate in {"x1", "x2"}
    assert np.isnan(sensitivity.benchmark_partial_r2(1.0, 0.1)[0])
    assert np.isnan(sensitivity.benchmark_partial_r2(0.1, 1.0)[0])


def test_linear_interval_matches_statsmodels_and_is_centred_on_the_estimate():
    import statsmodels.api as sm

    rng = np.random.default_rng(0)
    n = 400
    z = rng.normal(size=n)
    t = 0.5 * z + rng.normal(size=n)
    df = pd.DataFrame({"t": t, "z": z, "y": 2.0 * t + z + rng.normal(size=n)})
    se, low, high = sensitivity.linear_interval(df, "t", "y", ["z"])
    fit = sm.OLS(df["y"], sm.add_constant(df[["t", "z"]])).fit()
    assert se == pytest.approx(fit.bse["t"])
    assert (low, high) == pytest.approx(tuple(fit.conf_int().loc["t"]))
    assert low < fit.params["t"] < high and low < 2.0 < high  # the planted effect is inside
    _, low50, high50 = sensitivity.linear_interval(df, "t", "y", ["z"], alpha=0.5)
    assert (high50 - low50) < (high - low)  # a lower confidence level gives a narrower interval
