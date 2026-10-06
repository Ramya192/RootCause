"""Stage 5: Counterfactuals (causal_engine/pipeline/counterfactuals.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_engine.pipeline import counterfactuals


def test_mean_cate_sign_matches_job_satisfaction_reducing_attrition(feature_df, domain_config):
    [result] = counterfactuals.estimate_counterfactuals(feature_df, domain_config)
    assert result.treatment == "job_satisfaction"
    assert result.outcome == "attrition"
    # higher job_satisfaction (treated arm) should lower predicted attrition
    assert result.mean_cate < 0
    assert "job_satisfaction" in result.description


def test_raises_when_treatment_has_no_variation(domain_config):
    constant_df = pd.DataFrame(
        {
            "compensation": np.random.default_rng(0).normal(size=20),
            "manager_quality": np.random.default_rng(1).normal(size=20),
            "workload": np.random.default_rng(2).normal(size=20),
            "job_satisfaction": [5.0] * 20,  # all above threshold=0.0 -> no control arm
            "burnout": np.random.default_rng(3).normal(size=20),
            "attrition": np.random.default_rng(4).integers(0, 2, size=20),
        }
    )
    with pytest.raises(ValueError, match="no variation"):
        counterfactuals.estimate_counterfactuals(constant_df, domain_config)


def test_entity_id_is_not_used_as_a_covariate(feature_df, domain_config, monkeypatch):
    """Regression: Stage 5 used to feed the entity id (employee_id) to the learner."""
    seen = {}

    def spy(X, w, y, meta_learner, base_learner, propensity, clip):
        seen["n_features"] = X.shape[1]
        return np.zeros(len(y)), None

    monkeypatch.setattr(counterfactuals, "_causalml_learner", spy)
    counterfactuals.estimate_counterfactuals(feature_df, domain_config)
    # 5 features minus the treatment (job_satisfaction) = 4; the id would make it 5
    feature_cols = domain_config["feature_store"]["feature_columns"]
    assert seen["n_features"] == len(feature_cols) - 1


def test_description_uses_domain_wording(feature_df, domain_config):
    import copy

    cfg = copy.deepcopy(domain_config)
    cfg["domain"].update(entity_noun="applicant", entity_noun_plural="applicants", outcome_label="loan default")
    [result] = counterfactuals.estimate_counterfactuals(feature_df, cfg)
    assert result.description.startswith("Across all applicants,")
    assert "loan default" in result.description
    assert "employee" not in result.description


def test_description_defaults_are_neutral_not_another_domains(feature_df, domain_config):
    import copy

    cfg = copy.deepcopy(domain_config)
    cfg["domain"] = {"id": "x"}  # no wording configured
    [result] = counterfactuals.estimate_counterfactuals(feature_df, cfg)
    assert result.description.startswith("Across all records,")
    assert "employee" not in result.description


# --- learner choice -----------------------------------------------------------------------


def _with(domain_config, **overrides):
    import copy

    cfg = copy.deepcopy(domain_config)
    cfg["counterfactuals"].update(overrides)
    return cfg


@pytest.mark.parametrize("learner", sorted(counterfactuals.LEARNERS))
def test_every_meta_learner_runs_and_names_itself(feature_df, domain_config, learner):
    [result] = counterfactuals.estimate_counterfactuals(feature_df, _with(domain_config, meta_learner=learner))
    assert result.meta_learner.startswith(f"{learner} (causalml.{counterfactuals.LEARNERS[learner]}")
    assert result.cate_std is not None
    assert result.mean_cate < 0  # job satisfaction lowers attrition whichever learner estimates it
    assert abs(result.mean_cate) < 0.3  # and none of them returns an impossible probability change
    assert (result.extreme_propensity_share is not None) == (learner in counterfactuals.NEEDS_PROPENSITY)


def test_poor_overlap_is_reported_for_the_learners_that_use_a_propensity_score(feature_df, domain_config):
    """Job satisfaction is largely determined by compensation and manager quality in this data, so
    about a third of the units have a propensity score below 0.05 or above 0.95."""
    shares = {}
    for learner in ("x_learner", "r_learner", "dr_learner"):
        [result] = counterfactuals.estimate_counterfactuals(feature_df, _with(domain_config, meta_learner=learner))
        shares[learner] = result.extreme_propensity_share
    assert all(0.25 < share < 0.5 for share in shares.values())
    assert len(set(shares.values())) == 1  # same propensity model, same units flagged


def test_propensity_clip_changes_the_dr_estimate_and_is_validated(feature_df, domain_config):
    tight = counterfactuals.estimate_counterfactuals(feature_df, _with(domain_config, meta_learner="dr_learner", propensity_clip=0.001))
    loose = counterfactuals.estimate_counterfactuals(feature_df, _with(domain_config, meta_learner="dr_learner", propensity_clip=0.1))
    assert tight[0].mean_cate != loose[0].mean_cate
    with pytest.raises(ValueError, match="propensity_clip"):
        counterfactuals.estimate_counterfactuals(feature_df, _with(domain_config, meta_learner="dr_learner", propensity_clip=0.7))


def test_default_label_is_unchanged_and_extras_only_appear_when_set(feature_df, domain_config):
    [default] = counterfactuals.estimate_counterfactuals(feature_df, domain_config)
    assert default.meta_learner == "t_learner (causalml.BaseTRegressor)"
    [tuned] = counterfactuals.estimate_counterfactuals(
        feature_df, _with(domain_config, meta_learner="x_learner", base_learner="gbm", propensity="constant")
    )
    assert tuned.meta_learner == "x_learner (causalml.BaseXRegressor, gbm base, constant propensity)"


def test_linear_s_learner_gives_one_constant_effect_for_everyone():
    """With a linear model and the treatment as a feature, the effect is one coefficient."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    w = (rng.uniform(size=500) < 0.5).astype(int)
    y = 0.3 * w * X[:, 0] + X[:, 1] + rng.normal(size=500) * 0.1  # the effect depends on X[:, 0]

    s_cate = counterfactuals.estimate_cate(X, w, y, "s_learner").cate
    t_cate = counterfactuals.estimate_cate(X, w, y, "t_learner").cate

    assert np.std(s_cate) < 1e-9
    assert np.corrcoef(t_cate, 0.3 * X[:, 0])[0, 1] > 0.95  # T-learner recovers the heterogeneity S cannot express


def test_constant_propensity_passes_the_treated_share_to_the_learner(monkeypatch):
    seen = {}

    class FakeLearner:
        def __init__(self, learner):
            pass

        def fit_predict(self, X, treatment, y, p=None):
            seen["p"] = p
            return np.zeros((len(y), 1))

    from causalml.inference import meta

    monkeypatch.setattr(meta, "BaseXRegressor", FakeLearner)
    w = np.array([1, 1, 1, 0])
    counterfactuals.estimate_cate(np.zeros((4, 2)), w, np.zeros(4), "x_learner", propensity="constant")
    assert seen["p"].tolist() == [0.75] * 4

    counterfactuals.estimate_cate(np.zeros((4, 2)), w, np.zeros(4), "x_learner", propensity="constant", clip=0.4)
    assert seen["p"].tolist() == [0.6] * 4  # clipped into [0.4, 0.6]


@pytest.mark.parametrize(
    "key,value",
    [("meta_learner", "z_learner"), ("base_learner", "forest"), ("propensity", "oracle")],
)
def test_unknown_choices_raise_instead_of_silently_running_a_t_learner(feature_df, domain_config, key, value):
    with pytest.raises(ValueError, match=f"counterfactuals.{key}={value!r}"):
        counterfactuals.estimate_counterfactuals(feature_df, _with(domain_config, **{key: value}))


def test_without_causalml_only_the_linear_t_learner_has_a_fallback(monkeypatch):
    def missing(*args, **kwargs):
        raise ImportError("no causalml")

    monkeypatch.setattr(counterfactuals, "_causalml_learner", missing)
    X, w, y = np.random.default_rng(0).normal(size=(50, 2)), np.tile([0, 1], 25), np.random.default_rng(1).normal(size=50)

    fit = counterfactuals.estimate_cate(X, w, y, "t_learner")
    assert fit.label == "sklearn-fallback T-learner" and len(fit.cate) == 50 and fit.extreme_propensity_share is None
    with pytest.raises(ImportError, match="needs the `causalml` package"):
        counterfactuals.estimate_cate(X, w, y, "dr_learner")
    with pytest.raises(ImportError, match="only the linear T-learner"):
        counterfactuals.estimate_cate(X, w, y, "t_learner", base_learner="gbm")


def test_description_says_the_effect_is_averaged_over_everyone(feature_df, domain_config):
    [result] = counterfactuals.estimate_counterfactuals(feature_df, domain_config)
    assert result.description.startswith("Across all employees,")
    assert "rather than below it" in result.description and "is estimated to change" in result.description
    assert "shifted" not in result.description  # it is not the effect on only those below the threshold


def test_description_in_an_observational_domain_is_an_association(feature_df, domain_config):
    import copy

    observational = copy.deepcopy(domain_config)
    observational.setdefault("explanation", {})["observational"] = True
    [result] = counterfactuals.estimate_counterfactuals(feature_df, observational)
    assert "goes with" in result.description and "association" in result.description
    assert "is estimated to change" not in result.description
