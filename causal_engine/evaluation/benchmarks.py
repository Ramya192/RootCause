"""Experimental references: the yardstick for REAL randomized-trial data.

Real data has no SCM, so there is no simulated true effect to score against. When the
treatment was randomized, though, the plain difference in mean outcomes between the arms
is an unbiased estimate of the average treatment effect that involves none of the
pipeline's machinery (no graph, no adjustment, no estimator library). Comparing Stage 4's
estimate with it, and with its confidence interval, is the honest check available:

  - it is an ESTIMATE with sampling error, not a known truth (unlike an SCM effect), so the
    right question is "is the pipeline's number inside the interval", not "is the error small";
  - on a null trial (the Illinois wellness study) the reference is ~0, so this is a
    false-positive check: a pipeline that reports an effect and passes its placebo test here
    is inventing one.

Only datasets listed in RCT_DATASETS get a reference; naming one is a claim that the
treatment column was randomized.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

# (domain_id, dataset kind) pairs whose treatment(s) were randomized.
RCT_DATASETS: frozenset[tuple[str, str]] = frozenset({("illinois_wellness", "real")})

Z_95 = 1.959964


@dataclass(frozen=True)
class ExperimentalReference:
    treatment: str
    outcome: str
    n_treated: int
    n_control: int
    estimate: float  # mean outcome, treated minus control
    std_error: float  # Welch (unequal variances)
    ci_low: float
    ci_high: float
    outcome_missing_treated: float  # share of the arm's rows with no outcome (dropped at ingestion)
    outcome_missing_control: float
    subgroup: Optional[str] = None  # e.g. "male=1" for a within-subgroup reference

    @property
    def significant(self) -> bool:
        """Whether the 95% interval excludes zero."""
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def contains(self, value: float) -> bool:
        return self.ci_low <= value <= self.ci_high


def difference_in_means(treatment: pd.Series, outcome: pd.Series, name: str, outcome_name: str) -> ExperimentalReference:
    """Treated-minus-control difference in mean outcome with a Welch standard error."""
    arms = {int(a): outcome[treatment == a] for a in (0, 1)}
    observed = {a: y.dropna() for a, y in arms.items()}
    n1, n0 = len(observed[1]), len(observed[0])
    if n1 < 2 or n0 < 2:
        raise ValueError(f"{name}: need at least two observed outcomes in each arm, got {n1} and {n0}")
    estimate = float(observed[1].mean() - observed[0].mean())
    se = float(np.sqrt(observed[1].var(ddof=1) / n1 + observed[0].var(ddof=1) / n0))
    return ExperimentalReference(
        treatment=name,
        outcome=outcome_name,
        n_treated=n1,
        n_control=n0,
        estimate=estimate,
        std_error=se,
        ci_low=estimate - Z_95 * se,
        ci_high=estimate + Z_95 * se,
        outcome_missing_treated=float(arms[1].isna().mean()),
        outcome_missing_control=float(arms[0].isna().mean()),
    )


def experimental_subgroup_references(data_path: str | Path, domain_config: dict) -> list[ExperimentalReference]:
    """The trial's own difference in means within each subgroup Stage 5 reports on
    (`counterfactuals.subgroups`), for the Stage 5 treatment. Subgroup intervals are wider
    than the overall one, and looking at several of them means a few will exclude 0 by
    chance alone (about 1 in 20 at 95%)."""
    from causal_engine.pipeline.counterfactuals import binary_subgroup_levels

    cfg = domain_config["counterfactuals"]
    columns = list(cfg.get("subgroups", []))
    if not columns:
        return []
    df = pd.read_csv(data_path)
    treatment, outcome = cfg["treatment"], cfg["outcome"]
    if treatment not in df.columns or not set(df[treatment].dropna().unique()) <= {0, 1}:
        return []
    y = pd.to_numeric(df[outcome])
    refs = []
    for column in columns:
        for level, mask in binary_subgroup_levels(df, column):
            ref = difference_in_means(df.loc[mask, treatment], y[mask], treatment, outcome)
            refs.append(replace(ref, subgroup=f"{column}={level}"))
    return refs


def experimental_reference(data_path: str | Path, domain_config: dict) -> list[ExperimentalReference]:
    """One reference per configured treatment that is a 0/1 column in the file."""
    cfg = domain_config["effect_estimation"]
    outcome = cfg["outcome"]
    df = pd.read_csv(data_path)
    refs = []
    for treatment_cfg in cfg["treatments"]:
        name = treatment_cfg["name"]
        if name in df.columns and set(df[name].dropna().unique()) <= {0, 1}:
            refs.append(difference_in_means(df[name], pd.to_numeric(df[outcome]), name, outcome))
    return refs
