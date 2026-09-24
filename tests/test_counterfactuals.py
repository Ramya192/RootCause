"""Stage 5: Counterfactuals (rootcause/pipeline/counterfactuals.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rootcause.pipeline import counterfactuals


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
