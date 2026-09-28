"""Stress scenarios: where does the pipeline stop being right?

The baseline attrition data is easy (linear, independent root causes, no
confounding, n=2000), so scoring well on it proves little. Each scenario here
changes ONE thing about it -- sample size, a confounder, or the shape of a
mechanism -- and is scored against the true effects simulated from its SCM, over
the same replicate seeds as the baseline, so differences are attributable.

Scenarios are for the harness, not the API: they have no dataset kind in any
domain config, and their data is drawn fresh rather than read from a file.
A scenario may also rewrite the domain config, because Stage 4 takes its
confounders from the YAML (an observed confounder has to be declared there).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

from rootcause.evaluation import scms
from rootcause.evaluation.scm import SCM

DOMAIN_ID = "employee_attrition"
BASELINE_N = 2000
CONFOUNDER = "seniority"
CONFOUNDED_TREATMENT = "compensation"


def _identity(config: dict) -> dict:
    return config


def declare_confounder(config: dict) -> dict:
    """Copy of the attrition config that knows about the measured `seniority`
    column: it is a feature, a Stage 3 variable, and compensation's adjustment set."""
    cfg = copy.deepcopy(config)
    cfg["feature_store"]["feature_columns"].append(CONFOUNDER)
    variables = cfg["causal_discovery"]["variables"]
    variables.insert(variables.index(cfg["effect_estimation"]["outcome"]), CONFOUNDER)
    for treatment in cfg["effect_estimation"]["treatments"]:
        if treatment["name"] == CONFOUNDED_TREATMENT:
            treatment["confounders"] = [CONFOUNDER]
    return cfg


def discovery_without_outcome(config: dict) -> dict:
    """Copy of the config whose Stage 3 searches only the non-outcome variables (and drops the
    priors that mention the outcome). The outcome is binary in every domain; algorithms that
    assume linear relations with continuous variables (LiNGAM) are only fair on the rest."""
    cfg = copy.deepcopy(config)
    discovery = cfg["causal_discovery"]
    outcome = cfg["effect_estimation"]["outcome"]
    discovery["variables"] = [v for v in discovery["variables"] if v != outcome]
    for key in ("required_edges", "forbidden_edges"):
        discovery[key] = [e for e in discovery.get(key, []) if outcome not in e]
    return cfg


@dataclass(frozen=True)
class StressScenario:
    id: str
    factor: str  # what is being varied, for the report's grouping
    description: str
    scm: Callable[[], SCM]
    n_rows: int = BASELINE_N
    configure: Callable[[dict], dict] = _identity
    domain_id: str = DOMAIN_ID


def _small_n(n: int) -> StressScenario:
    return StressScenario(
        id=f"small_n_{n}",
        factor="sample size",
        description=f"Baseline data-generating process, n={n} instead of {BASELINE_N}.",
        scm=scms.attrition_scm,
        n_rows=n,
    )


def _hidden(strength: float) -> StressScenario:
    return StressScenario(
        id=f"hidden_confounder_{strength:g}",
        factor="hidden confounder",
        description=(
            f"Seniority drives compensation (corr {strength:g}) and attrition, and is NOT in the data. "
            "The pipeline is configured exactly as on the baseline."
        ),
        scm=lambda: scms.confounded_attrition_scm(strength, hidden=True),
    )


STRESS_SCENARIOS: tuple[StressScenario, ...] = (
    StressScenario(
        id="baseline",
        factor="reference",
        description=f"Unmodified attrition SCM at n={BASELINE_N}: the reference every other row is compared with.",
        scm=scms.attrition_scm,
    ),
    _small_n(1000),
    _small_n(500),
    _small_n(250),
    StressScenario(
        id="observed_confounder_0.6",
        factor="observed confounder",
        description=(
            "Seniority drives compensation (corr 0.6) and attrition, is in the data, and is declared "
            "as compensation's confounder. The same data as hidden_confounder_0.6 with the column visible."
        ),
        scm=lambda: scms.confounded_attrition_scm(0.6, hidden=False),
        configure=declare_confounder,
    ),
    _hidden(0.3),
    _hidden(0.6),
    _hidden(0.85),
    StressScenario(
        id="nonlinear_monotone",
        factor="nonlinearity",
        description=(
            "Diminishing returns to pay on satisfaction (tanh) and accelerating overload on burnout "
            "(0.3*w*|w|). Monotone, so linear models are roughly right near the middle."
        ),
        scm=scms.nonlinear_monotone_attrition_scm,
    ),
    StressScenario(
        id="nonlinear_u_shaped",
        factor="nonlinearity",
        description=(
            "Burnout is U-shaped in workload (too little and too much both burn people out). "
            "workload is linearly uncorrelated with burnout but still causes it."
        ),
        scm=scms.nonlinear_u_shaped_attrition_scm,
    ),
    StressScenario(
        id="non_gaussian_noise",
        factor="noise shape",
        description=(
            "The baseline graph and effects with uniform instead of Gaussian noise. Nothing changes for "
            "PC, GES or the estimators; it is the setting LiNGAM's assumptions need, so it is the fair "
            "test of that algorithm (--algorithm lingam) -- except that the binary outcome is still in "
            "the search, which breaks its linearity assumption (see non_gaussian_continuous_only)."
        ),
        scm=scms.non_gaussian_attrition_scm,
    ),
    StressScenario(
        id="non_gaussian_continuous_only",
        factor="noise shape",
        description=(
            "Uniform noise as above, and Stage 3 searches only the five continuous variables (the binary "
            "outcome and the priors that mention it are left out): every assumption LiNGAM makes holds. "
            "Graph scores are over the four true edges among those variables, so they are not comparable "
            "with the other rows."
        ),
        scm=scms.non_gaussian_attrition_scm,
        configure=discovery_without_outcome,
    ),
)


def select(ids: list[str] | None = None) -> list[StressScenario]:
    """All scenarios, or the named ones (in registry order). Unknown ids raise."""
    if not ids:
        return list(STRESS_SCENARIOS)
    known = {s.id for s in STRESS_SCENARIOS}
    unknown = sorted(set(ids) - known)
    if unknown:
        raise ValueError(f"unknown stress scenario(s) {unknown}; available: {sorted(known)}")
    return [s for s in STRESS_SCENARIOS if s.id in set(ids)]
