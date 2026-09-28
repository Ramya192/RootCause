"""Stage 4 sensitivity analysis: how strong would an UNMEASURED confounder have to be?

The permutation placebo in effect_estimation.py can only say an estimate stands out
from noise. It cannot see a confounder that is not in the data. Sensitivity analysis
answers a different, honest question: given the estimate, how much of the leftover
(residual) variance of BOTH the treatment and the outcome would a hidden confounder
have to explain to erase the effect? It does not detect confounding, and it does not
say whether such a confounder exists; it turns "what if?" into a number a domain
expert can argue with.

Implements Cinelli & Hazlett (2020), "Making sense of sensitivity: extending omitted
variable bias", JRSS-B, for a linear-regression estimate (the only estimator this
stage uses that the theory covers; other estimators get no sensitivity block):

  * partial R^2 of the treatment with the outcome, given the adjustment set;
  * robustness value RV_q: the share of residual variance a confounder must explain,
    in both the treatment and the outcome, to reduce the estimate by a fraction q
    (q=1: to zero). RV_{q,alpha} does the same for statistical significance;
  * a benchmark: the worst-case adjusted estimate if an unobserved confounder were
    as strong as the observed covariate that would bias it most (the "kd = ky = 1"
    bound of the paper), so the number can be read against variables we can see.

A high RV is not proof of no confounding: it only says a confounder would need to be
strong. Strength on this scale is unusual in practice for a real hidden cause only if
the analyst has measured the important ones, which is a domain judgement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from rootcause.models.schemas import SensitivityResult

LINEAR_ESTIMATORS = ("backdoor.linear_regression",)
INTERCEPT = "__intercept__"


def partial_r2_from_t(t: float, dof: float) -> float:
    """Partial R^2 of one regressor from its t statistic and the residual degrees of freedom."""
    return float(t**2 / (t**2 + dof))


def robustness_value(t: float, dof: float, q: float = 1.0, alpha: float | None = None) -> float:
    """RV_q (alpha=None) or RV_{q,alpha}: the partial R^2 a confounder must have with BOTH
    treatment and outcome to cut the estimate by fraction `q` (or make it insignificant)."""
    f_q = q * abs(t) / np.sqrt(dof)
    if alpha is not None:
        f_crit = abs(stats.t.ppf(alpha / 2, dof - 1)) / np.sqrt(dof - 1)
        f_q = f_q - f_crit
        if f_q <= 0:
            return 0.0
    return float(0.5 * (np.sqrt(f_q**4 + 4 * f_q**2) - f_q**2))


def bias_bound(se: float, dof: float, r2_dz: float, r2_yz: float) -> float:
    """Absolute bias of the estimate from an omitted confounder with partial R^2 `r2_dz`
    with the treatment (given the covariates) and `r2_yz` with the outcome (given treatment
    and covariates)."""
    return float(np.sqrt(r2_yz * r2_dz / (1.0 - r2_dz)) * se * np.sqrt(dof))


def benchmark_partial_r2(r2_dxj: float, r2_yxj: float, kd: float = 1.0, ky: float = 1.0) -> tuple[float, float]:
    """(r2_dz, r2_yz) of a confounder `kd`/`ky` times as strong as an observed covariate
    with partial R^2 `r2_dxj` (with the treatment) and `r2_yxj` (with the outcome).

    Follows sensemakr's `ovb_partial_r2_bound`: the covariate's strength is expressed
    against the residualised treatment, and the outcome strength is inflated because the
    confounder is also correlated with the covariate. Returns None-like (nan) pair when
    the requested strength is not attainable (r2_dz would exceed 1)."""
    if r2_dxj >= 1.0 or kd * r2_dxj >= 1.0 or r2_yxj >= 1.0:
        return float("nan"), float("nan")  # a covariate that explains the treatment (or outcome) exactly
    r2_dz = kd * r2_dxj / (1.0 - r2_dxj)
    if r2_dz >= 1.0:
        return float("nan"), float("nan")
    r2_zxj_xd = kd * r2_dxj**2 / ((1.0 - kd * r2_dxj) * (1.0 - r2_dxj))
    if r2_zxj_xd >= 1.0:
        return float("nan"), float("nan")
    r2_yz = ((np.sqrt(ky) + np.sqrt(r2_zxj_xd)) / np.sqrt(1.0 - r2_zxj_xd)) ** 2 * (r2_yxj / (1.0 - r2_yxj))
    return float(r2_dz), float(min(r2_yz, 1.0))


def linear_sensitivity(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    alpha: float = 0.05,
) -> SensitivityResult:
    """Sensitivity of the OLS estimate of `treatment` on `outcome`, adjusting for `confounders`."""
    # A column with no variation adjusts for nothing and only makes the design rank-deficient (a
    # declared one-hot level that a dataset never uses, e.g. Freddie Mac's `tpo_unspecified`
    # channel in 2010-11); every other coefficient is unchanged by leaving it out.
    confounders = [c for c in confounders if data[c].nunique(dropna=False) > 1]
    y = data[outcome].to_numpy(dtype=float)
    design = data[[treatment, *confounders]].astype(float)
    design.insert(0, INTERCEPT, 1.0)  # named here so a covariate called "const" cannot collide with it
    outcome_fit = sm.OLS(y, design).fit()
    estimate = float(outcome_fit.params[treatment])
    se = float(outcome_fit.bse[treatment])
    t = float(outcome_fit.tvalues[treatment])
    dof = float(outcome_fit.df_resid)

    result = dict(
        estimate=estimate,
        partial_r2_treatment_outcome=partial_r2_from_t(t, dof),
        robustness_value=robustness_value(t, dof, 1.0),
        robustness_value_alpha=robustness_value(t, dof, 1.0, alpha),
        alpha=alpha,
    )

    # Benchmark against each observed covariate: its partial R^2 with the treatment (given
    # the other covariates) and with the outcome (given treatment and the other covariates).
    if confounders:
        treatment_design = data[confounders].astype(float)
        treatment_design.insert(0, INTERCEPT, 1.0)
        treatment_fit = sm.OLS(data[treatment].to_numpy(dtype=float), treatment_design).fit()
        worst = None
        for name in confounders:
            t_y, t_d = outcome_fit.tvalues.get(name, np.nan), treatment_fit.tvalues.get(name, np.nan)
            if not np.isfinite(t_y) or not np.isfinite(t_d):
                continue  # a constant or perfectly collinear column has no defined strength
            r2_dxj = partial_r2_from_t(float(t_d), float(treatment_fit.df_resid))
            r2_yxj = partial_r2_from_t(float(t_y), dof)
            r2_dz, r2_yz = benchmark_partial_r2(r2_dxj, r2_yxj)
            if not np.isfinite(r2_dz):
                continue
            bias = bias_bound(se, dof, r2_dz, r2_yz)
            if worst is None or bias > worst["bias"]:
                worst = dict(covariate=name, r2_dz=r2_dz, r2_yz=r2_yz, bias=bias)
        if worst is not None:
            adjusted = float(np.sign(estimate) * (abs(estimate) - worst["bias"]))
            result.update(
                benchmark_covariate=worst["covariate"],
                benchmark_r2_treatment=worst["r2_dz"],
                benchmark_r2_outcome=worst["r2_yz"],
                benchmark_bias=worst["bias"],
                benchmark_adjusted_estimate=adjusted,
                robust_to_benchmark=bool(np.sign(adjusted) == np.sign(estimate) and adjusted != 0),
            )
    return SensitivityResult(**result)


def adjusted_estimate(
    estimate: float, se: float, dof: float, r2_dz: float, r2_yz: float, bias_sign: float | None = None
) -> float:
    """The estimate corrected for an omitted confounder of known strength.

    `bias_sign` is the direction the confounder pushed the estimate: +1 up, -1 down. By
    default it is assumed to push it away from zero (sign of the estimate), the conservative
    direction for a nonzero effect, so the correction shrinks |estimate|. A confounder that
    pushed the other way needs `bias_sign` set explicitly (a simulation knows; an analyst
    does not, which is why the default is the cautious one)."""
    direction = np.sign(estimate) if bias_sign is None else np.sign(bias_sign)
    return float(estimate - direction * bias_bound(se, dof, r2_dz, r2_yz))
