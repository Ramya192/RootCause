"""The known data-generating processes the evaluation harness scores against.

`attrition_scm` is the single definition of the committed synthetic dataset
(data/employee_attrition/attrition.csv): scripts/data/generate_synthetic_attrition_data.py
samples it, and tests/test_evaluation_scm.py fails if the two ever drift apart.

    compensation        ~ N(0, 1)                                   (root)
    manager_quality     ~ N(0, 1)                                   (root)
    workload            ~ N(0, 1)                                   (root)
    gender              ~ Uniform{A, B}                             (root, causally inert)
    job_satisfaction    = 0.6*compensation + 0.5*manager_quality + N(0, 0.5)
    burnout             = 0.7*workload - 0.2*manager_quality + N(0, 0.5)
    attrition           ~ Bernoulli(sigmoid(-1.2*job_satisfaction + 1.0*burnout + N(0, 0.3)))
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from causal_engine.evaluation.scm import (
    SCM,
    Node,
    additive_noise,
    bernoulli_of,
    linear_gaussian,
    logistic_binary,
    logistic_of,
    sigmoid,
)

SEED = 42
N_ROWS = 2000


def _node(name: str, spec, latent: bool = False) -> Node:
    parents, mechanism = spec
    return Node(name=name, parents=parents, mechanism=mechanism, latent=latent)


def attrition_scm() -> SCM:
    # The node ORDER fixes the RNG stream (each node draws its noise in turn),
    # so it must match the order the original generator drew in: gender before
    # the derived variables, attrition last.
    return SCM(
        nodes=(
            _node("compensation", linear_gaussian()),
            _node("manager_quality", linear_gaussian()),
            _node("workload", linear_gaussian()),
            _gender(),
            _node("job_satisfaction", linear_gaussian(noise_sd=0.5, compensation=0.6, manager_quality=0.5)),
            _node("burnout", linear_gaussian(noise_sd=0.5, manager_quality=-0.2, workload=0.7)),
            _node(
                "attrition",
                logistic_binary(logit_noise_sd=0.3, job_satisfaction=-1.2, burnout=1.0),
            ),
        )
    )


def _gender() -> Node:
    return Node(
        name="gender",
        parents=(),
        mechanism=lambda values, rng, n: rng.choice(["A", "B"], size=n),
    )


# --- Stress variants ---------------------------------------------------------------------
# Each changes ONE thing about the baseline so a breakage can be attributed to it.
# All keep the baseline's column names (plus `seniority` where noted), so the same
# pipeline config runs on them. Scored by causal_engine.evaluation.stress.

CONFOUNDER_ATTRITION_LOGIT = -0.8  # direct effect of seniority on the attrition logit


def confounded_attrition_scm(strength: float, hidden: bool) -> SCM:
    """Baseline plus `seniority`, a common cause of compensation and attrition:

        seniority     ~ N(0, 1)
        compensation  = strength*seniority + N(0, 1 - strength^2)   (still unit variance)
        attrition     ~ Bernoulli(sigmoid(<baseline logit> - 0.8*seniority))

    `strength` is corr(seniority, compensation). Seniors are paid more and leave
    less, so an analysis that ignores seniority credits pay with retention it did
    not cause. `hidden=True` withholds the column (an unmeasured confounder); the
    two settings draw identical data apart from that one column.
    """
    if not 0.0 <= strength < 1.0:
        raise ValueError(f"strength must be in [0, 1), got {strength}")
    return SCM(
        nodes=(
            _node("seniority", linear_gaussian(), latent=hidden),
            _node(
                "compensation",
                additive_noise(("seniority",), lambda v: strength * v["seniority"], noise_sd=(1 - strength**2) ** 0.5),
            ),
            _node("manager_quality", linear_gaussian()),
            _node("workload", linear_gaussian()),
            _gender(),
            _node("job_satisfaction", linear_gaussian(noise_sd=0.5, compensation=0.6, manager_quality=0.5)),
            _node("burnout", linear_gaussian(noise_sd=0.5, manager_quality=-0.2, workload=0.7)),
            _node(
                "attrition",
                logistic_binary(
                    logit_noise_sd=0.3,
                    job_satisfaction=-1.2,
                    burnout=1.0,
                    seniority=CONFOUNDER_ATTRITION_LOGIT,
                ),
            ),
        )
    )


def _uniform_noise(sd: float):
    """Zero-mean uniform noise with standard deviation `sd` (half-width sd*sqrt(3))."""
    half = sd * 3**0.5
    return lambda rng, n: rng.uniform(-half, half, n)


def _uniform_linear(sd: float, **coefs: float):
    """`sum(coef * parent) + uniform noise with sd` -- `linear_gaussian` with non-Gaussian noise."""
    noise = _uniform_noise(sd)

    def mechanism(values, rng, n):
        return sum(c * values[p] for p, c in coefs.items()) + noise(rng, n) if coefs else noise(rng, n)

    return tuple(coefs), mechanism


def non_gaussian_attrition_scm() -> SCM:
    """The baseline structure and noise VARIANCES, with uniform instead of Gaussian noise on every
    continuous node. Same graph, same effects (linear), so nothing but the noise shape changes.
    This is the setting DirectLiNGAM's identification argument needs (linear, non-Gaussian,
    acyclic); the Gaussian baseline is exactly where it cannot tell directions apart."""
    return SCM(
        nodes=(
            _node("compensation", _uniform_linear(1.0)),
            _node("manager_quality", _uniform_linear(1.0)),
            _node("workload", _uniform_linear(1.0)),
            _gender(),
            _node("job_satisfaction", _uniform_linear(0.5, compensation=0.6, manager_quality=0.5)),
            _node("burnout", _uniform_linear(0.5, manager_quality=-0.2, workload=0.7)),
            _node("attrition", logistic_binary(logit_noise_sd=0.3, job_satisfaction=-1.2, burnout=1.0)),
        )
    )


def nonlinear_monotone_attrition_scm() -> SCM:
    """Baseline with monotone but curved mechanisms, the kind a linear model
    approximates well near the middle and badly in the tails:

        job_satisfaction = 0.5*manager_quality + tanh(compensation) + N(0, 0.5)   (diminishing returns to pay)
        burnout          = -0.2*manager_quality + 0.6*workload + 0.3*workload*|workload| + N(0, 0.5)   (overload accelerates)
    """
    return SCM(
        nodes=(
            _node("compensation", linear_gaussian()),
            _node("manager_quality", linear_gaussian()),
            _node("workload", linear_gaussian()),
            _gender(),
            _node(
                "job_satisfaction",
                additive_noise(
                    ("compensation", "manager_quality"),
                    lambda v: 0.5 * v["manager_quality"] + np.tanh(v["compensation"]),
                    noise_sd=0.5,
                ),
            ),
            _node(
                "burnout",
                additive_noise(
                    ("manager_quality", "workload"),
                    lambda v: -0.2 * v["manager_quality"] + 0.6 * v["workload"] + 0.3 * v["workload"] * np.abs(v["workload"]),
                    noise_sd=0.5,
                ),
            ),
            _node("attrition", logistic_binary(logit_noise_sd=0.3, job_satisfaction=-1.2, burnout=1.0)),
        )
    )


def nonlinear_u_shaped_attrition_scm() -> SCM:
    """Baseline except burnout is U-shaped in workload (too little AND too much
    both burn people out), centred so the base attrition rate is unchanged:

        burnout = -0.2*manager_quality + 0.6*(workload^2 - 1) + N(0, 0.5)

    workload is uncorrelated with burnout here, so a method that only sees linear
    association finds nothing, yet moving workload by +1 does change burnout.
    """
    return SCM(
        nodes=(
            _node("compensation", linear_gaussian()),
            _node("manager_quality", linear_gaussian()),
            _node("workload", linear_gaussian()),
            _gender(),
            _node("job_satisfaction", linear_gaussian(noise_sd=0.5, compensation=0.6, manager_quality=0.5)),
            _node(
                "burnout",
                additive_noise(
                    ("manager_quality", "workload"),
                    lambda v: -0.2 * v["manager_quality"] + 0.6 * (v["workload"] ** 2 - 1.0),
                    noise_sd=0.5,
                ),
            ),
            _node("attrition", logistic_binary(logit_noise_sd=0.3, job_satisfaction=-1.2, burnout=1.0)),
        )
    )


# --- Illinois Workplace Wellness mirror --------------------------------------------------
# A fully synthetic dataset with the SAME SCHEMA as the real Illinois Workplace Wellness
# RCT (data/illinois_wellness/wellness.csv): same columns, marginals matched roughly to the
# real file, treatment randomized. Unlike the real trial, its treatment effect is PLANTED,
# so Stage 3 (graph) and Stage 4 (effect) can be scored against a known truth. Nothing
# here is real data or a claim about the real program.
#
#   male, white, treat        Bernoulli roots (treat is independent of everything: an RCT)
#   age_group (latent)        <37 / 37-49 / 50+  ->  age37_49, age50 (the two dummies in the file)
#   prod_index_yr0            = 0.43*male + 0.2*age50 + N          (pre-treatment productivity)
#   sickleave_0815_0716       skewed, 25% zeros; falls with prod_index_yr0, lower for men
#   gym_0815_0716             83% zeros, heavy tail, more common among women
#   terminated_0119           ~ Bernoulli(sigmoid(b0 + covariate terms + WELLNESS_TREAT_LOGIT*treat))

WELLNESS_SEED = 42
WELLNESS_N_ROWS = 4834  # the real file's size
WELLNESS_TREAT_LOGIT = -0.30  # planted: about -4.8 percentage points on the 20% base rate
WELLNESS_TERMINATION_INTERCEPT = 0.26  # calibrated so the base rate is ~20.4%, as in the real file


def _age_group() -> Node:
    return Node(
        name="age_group",
        parents=(),
        mechanism=lambda values, rng, n: rng.choice(3, size=n, p=[0.3395, 0.3349, 0.3256]),
        latent=True,
    )


def _sickleave(values, rng, n):
    # A point mass at zero (people who took no sick leave) plus a lognormal body;
    # both the chance of any leave and its size follow pre-treatment productivity.
    prod, male = values["prod_index_yr0"], values["male"]
    any_leave = rng.uniform(0.0, 1.0, n) < 1.0 / (1.0 + np.exp(-1.1 + 0.5 * prod))
    body = np.expm1(1.85 - 0.45 * prod - 0.25 * male + rng.normal(0.0, 0.55, n))
    return np.where(any_leave, body, 0.0)


def _gym(values, rng, n):
    used = rng.uniform(0.0, 1.0, n) < (0.135 + 0.08 * values["male"])
    return np.where(used, np.expm1(rng.normal(3.19, 1.05, n)), 0.0)


def illinois_wellness_scm(treat_logit: float = WELLNESS_TREAT_LOGIT) -> SCM:
    return SCM(
        nodes=(
            _node("male", bernoulli_of((), lambda v: 0.427)),
            _age_group(),
            _node("age50", (("age_group",), lambda v, rng, n: (v["age_group"] == 2).astype(int))),
            _node("age37_49", (("age_group",), lambda v, rng, n: (v["age_group"] == 1).astype(int))),
            _node("white", bernoulli_of(("age50",), lambda v: 0.808 + 0.09 * v["age50"])),
            _node("treat", bernoulli_of((), lambda v: 0.683)),
            _node(
                "prod_index_yr0",
                additive_noise(
                    ("male", "age50"), lambda v: 0.43 * v["male"] + 0.2 * v["age50"] - 0.28, noise_sd=1.28
                ),
            ),
            _node("sickleave_0815_0716", (("prod_index_yr0", "male"), _sickleave)),
            _node("gym_0815_0716", (("male",), _gym)),
            _node(
                "terminated_0119",
                logistic_of(
                    (
                        "male", "age50", "age37_49", "white", "sickleave_0815_0716",
                        "prod_index_yr0", "treat",
                    ),
                    lambda v: WELLNESS_TERMINATION_INTERCEPT
                    - 0.18 * v["male"]
                    - 0.50 * v["age50"]
                    - 1.06 * v["age37_49"]
                    - 0.53 * v["white"]
                    - 0.35 * np.log1p(v["sickleave_0815_0716"])
                    - 0.27 * v["prod_index_yr0"]
                    + treat_logit * v["treat"],
                ),
            ),
        )
    )


# --- German Credit semi-synthetic --------------------------------------------------------
# Real applicants, simulated outcome. Every column except `default` is REAL: a draw picks one
# of the 1,000 real applicants at random (a bootstrap) and copies all their covariates,
# including the three lending levers (duration, amount, installment rate), whose real
# correlations with the other columns are therefore kept. Only `default` is simulated, from a
# logistic model whose lever effects are PLANTED, so the true effect of moving a lever is known.
#
#   default ~ Bernoulli(sigmoid(intercept + young_logit*young + <covariate terms>
#                               + 0.03*duration_months + 0.10*credit_amount_kdm + 0.30*installment_rate))
#
# The covariate terms are the real data's logistic-regression coefficients (rounded), for
# checking status, credit history, savings, housing, other plans, employment and foreign worker.
# `young_logit` and the intercept are SOLVED so the simulation reproduces the real file's
# default rate for applicants under 25 (40.9%) and for everyone else (28.1%): the age
# disparity the fairness check looks for is real in size, and planted in mechanism.
#
# What this is NOT: the lever effects are chosen (roughly the size of the real association),
# so this validates the METHOD (does Stage 4 recover a known effect from realistic confounded
# covariates?) and says nothing about what actually causes default. The graph among the real
# covariates is unknown, so Stage 3 is not scored here (SCM.graph_known is False).

CREDIT_SEED = 42
CREDIT_N_ROWS = 1000  # the real file's size
CREDIT_INTERCEPT = -4.649
CREDIT_YOUNG_LOGIT = 0.281
CREDIT_LEVER_LOGITS = {"duration_months": 0.03, "credit_amount_kdm": 0.10, "installment_rate": 0.30}
CREDIT_FOREIGN_LOGIT = 1.3
CREDIT_EMPLOYMENT_LOGIT = -0.15  # per step of the employment_since order below
CREDIT_EMPLOYMENT_ORDER = ["unemployed", "lt_1y", "1_to_4y", "4_to_7y", "ge_7y"]
CREDIT_LEVEL_LOGITS = {  # column -> level -> logit contribution (reference level = 0)
    "checking_status": {"lt_0": 1.7, "0_to_200": 1.25, "ge_200": 0.8, "none": 0.0},
    "credit_history": {
        "no_credits_or_all_paid": 0.8, "all_paid_this_bank": 0.6, "existing_paid": 0.0,
        "past_delay": 0.0, "critical_or_other_credits": -0.55,
    },
    "savings": {"lt_100": 0.0, "100_to_500": -0.15, "500_to_1000": -0.45, "ge_1000": -1.05, "unknown": -0.85},
    "housing": {"own": 0.0, "rent": 0.5, "free": 0.2},
    "other_plans": {"none": 0.0, "bank": 0.55, "stores": 0.4},
}
CREDIT_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "german_credit" / "credit.csv"
CREDIT_ID_COLUMN = "applicant_id"
CREDIT_OUTCOME = "default"


@lru_cache(maxsize=1)
def _real_credit_applicants() -> pd.DataFrame:
    """The real applicants (prepared by scripts/data/prepare_german_credit.py), minus the id and
    the real outcome, which the SCM replaces."""
    if not CREDIT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{CREDIT_DATA_PATH} not found; run scripts/data/prepare_german_credit.py first"
        )
    return pd.read_csv(CREDIT_DATA_PATH).drop(columns=[CREDIT_ID_COLUMN, CREDIT_OUTCOME])


def _real_column(column: str) -> Node:
    table = _real_credit_applicants()[column].to_numpy()
    return Node(name=column, parents=("row",), mechanism=lambda values, rng, n: table[values["row"]])


def _credit_default(values, rng, n):
    logit = CREDIT_INTERCEPT + CREDIT_YOUNG_LOGIT * values["young"] + CREDIT_FOREIGN_LOGIT * values["foreign_worker"]
    steps = pd.Series(values["employment_since"]).map(
        {level: i for i, level in enumerate(CREDIT_EMPLOYMENT_ORDER)}
    ).to_numpy()
    logit = logit + CREDIT_EMPLOYMENT_LOGIT * steps
    for column, levels in CREDIT_LEVEL_LOGITS.items():
        logit = logit + pd.Series(values[column]).map(levels).to_numpy()
    for lever, coef in CREDIT_LEVER_LOGITS.items():
        logit = logit + coef * values[lever]
    return (rng.uniform(0.0, 1.0, n) < sigmoid(logit)).astype(int)


def german_credit_semi_synthetic_scm() -> SCM:
    applicants = _real_credit_applicants()
    row = Node(
        name="row",
        parents=(),
        mechanism=lambda values, rng, n: rng.integers(0, len(applicants), n),
        latent=True,
    )
    drivers = (
        "young", "foreign_worker", "employment_since", *CREDIT_LEVEL_LOGITS, *CREDIT_LEVER_LOGITS,
    )
    return SCM(
        nodes=(
            row,
            *(_real_column(column) for column in applicants.columns),  # same column order as credit.csv
            Node(name=CREDIT_OUTCOME, parents=drivers, mechanism=_credit_default),
        ),
        graph_known=False,
    )


# --- Freddie Mac 2007 semi-synthetic twin --------------------------------------------------
#
# The REAL 2007 covariates (31,780 loans) with a SIMULATED default whose lever effects are planted,
# so the pipeline is scored where the answer is known. Same idea as the German Credit twin, and the
# same limits: the effects are chosen (roughly the size of the real, confounded association), so this
# validates the METHOD (does Stage 4 recover a known effect from realistic covariates whose
# correlations, e.g. rate with credit score, are real?) and says nothing about what causes default.
# One difference from the real data that flatters the method: here EVERY driver of default is a
# recorded column, so there is no unobserved confounding (in the real file lenders price on risk that
# the file does not hold). The graph among the covariates is unknown (SCM.graph_known is False).
# The file has no loan_sequence: that is a real Freddie Mac identifier and is not reproduced.

FREDDIE_SEED = 42
FREDDIE_N_ROWS = 31_780  # the real 2007 file's size
FREDDIE_INTERCEPT = -4.0164  # solved so the simulated default rate matches the real file's 15.53%
FREDDIE_LEVER_LOGITS = {"interest_rate": 0.9, "ltv": 0.010, "dti": 0.015}  # per percentage point / point
FREDDIE_SCORE_LOGIT = -0.007  # per credit-score point
FREDDIE_FIRST_TIME_LOGIT = -0.15
FREDDIE_INVESTMENT_LOGIT = 0.35  # occupancy == investment
FREDDIE_CASH_OUT_LOGIT = 0.30  # purpose == refi_cash_out
FREDDIE_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "freddie_mac" / "loans_2007.csv"
FREDDIE_ID_COLUMN = "loan_id"
FREDDIE_DROPPED_COLUMNS = (FREDDIE_ID_COLUMN, "loan_sequence", "default")
FREDDIE_OUTCOME = "default"


@lru_cache(maxsize=1)
def _real_freddie_loans() -> pd.DataFrame:
    """The real 2007 loans (prepared by scripts/data/prepare_freddie_mac.py), minus the ids and the real
    outcome, which the SCM replaces."""
    if not FREDDIE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{FREDDIE_DATA_PATH} not found; it is built from a registered Freddie Mac download by "
            "scripts/data/prepare_freddie_mac.py (see data/freddie_mac/README.md)"
        )
    return pd.read_csv(FREDDIE_DATA_PATH).drop(columns=list(FREDDIE_DROPPED_COLUMNS))


def _real_freddie_column(column: str) -> Node:
    table = _real_freddie_loans()[column].to_numpy()
    return Node(name=column, parents=("row",), mechanism=lambda values, rng, n: table[values["row"]])


def _freddie_default(values, rng, n):
    logit = FREDDIE_INTERCEPT + FREDDIE_SCORE_LOGIT * values["credit_score"] + FREDDIE_FIRST_TIME_LOGIT * values["first_time_buyer"]
    logit = logit + FREDDIE_INVESTMENT_LOGIT * (values["occupancy"] == "investment")
    logit = logit + FREDDIE_CASH_OUT_LOGIT * (values["purpose"] == "refi_cash_out")
    for lever, coef in FREDDIE_LEVER_LOGITS.items():
        logit = logit + coef * values[lever]
    return (rng.uniform(0.0, 1.0, n) < sigmoid(logit)).astype(int)


def freddie_mac_semi_synthetic_scm() -> SCM:
    loans = _real_freddie_loans()
    row = Node(
        name="row",
        parents=(),
        mechanism=lambda values, rng, n: rng.integers(0, len(loans), n),
        latent=True,
    )
    drivers = ("credit_score", "first_time_buyer", "occupancy", "purpose", *FREDDIE_LEVER_LOGITS)
    return SCM(
        nodes=(
            row,
            *(_real_freddie_column(column) for column in loans.columns),  # same column order as loans_2007.csv
            Node(name=FREDDIE_OUTCOME, parents=drivers, mechanism=_freddie_default),
        ),
        graph_known=False,
    )


# (domain_id, dataset kind) -> factory for the SCM behind that dataset. A pair
# absent from here has no known ground truth (e.g. real data), and the harness
# reports only the metrics that don't need one.
SCM_REGISTRY: dict[tuple[str, str], Callable[[], SCM]] = {
    ("employee_attrition", "synthetic"): attrition_scm,
    ("illinois_wellness", "synthetic"): illinois_wellness_scm,
    ("german_credit", "semi_synthetic"): german_credit_semi_synthetic_scm,
    ("freddie_mac", "semi_synthetic"): freddie_mac_semi_synthetic_scm,
}
