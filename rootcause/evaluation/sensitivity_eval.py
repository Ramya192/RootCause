"""Does the Stage 4 sensitivity analysis do what it claims, on data where the hidden confounder is known?

The stress tests showed a hidden confounder (seniority) inflates compensation's effect while the
permutation placebo still passes. Sensitivity analysis cannot DETECT that confounder, and this
check does not pretend it can. It asks two narrower questions that have checkable answers:

  1. Is the arithmetic right? Given the confounder's TRUE partial R^2 with compensation and with
     attrition (which we can compute here because the simulation keeps the column), the
     omitted-variable-bias formula must turn the biased estimate back into the estimate that
     includes the confounder. It is an identity, so it must hold to floating point.
  2. Does the headline number mislead? The robustness value says how strong a hidden confounder
     must be to erase the effect. We compare it to how strong the real one is and show that it
     barely moves between the clean and the confounded data: it warns about what COULD be wrong,
     it does not say whether something is.

Same draws as the stress scenarios (seeds 1000+, n=2000, corr(seniority, compensation) 0/0.3/0.6/0.85).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from rootcause.evaluation.scms import confounded_attrition_scm
from rootcause.pipeline import sensitivity

TREATMENT, OUTCOME, CONFOUNDER = "compensation", "attrition", "seniority"
STRENGTHS = (0.0, 0.3, 0.6, 0.85)
SEED0 = 1000
N_ROWS = 2000
TRUTH_DRAWS = 1_000_000


@dataclass
class SensitivityRow:
    strength: float
    seed: int
    naive_estimate: float
    full_estimate: float  # the estimate that adjusts for the confounder (needs the hidden column)
    robustness_value: float
    robustness_value_alpha: float
    r2_treatment: float  # the hidden confounder's TRUE partial R^2 with compensation
    r2_outcome: float  # ... and with attrition given compensation
    adjusted_at_true_strength: float

    @property
    def identity_error(self) -> float:
        return abs(self.adjusted_at_true_strength - self.full_estimate)

    @property
    def explains_away(self) -> bool:
        """The confounder is at least as strong as the robustness value in BOTH regressions,
        the condition under which it COULD reduce the estimate to zero."""
        return self.r2_treatment >= self.robustness_value and self.r2_outcome >= self.robustness_value


def one_draw(strength: float, seed: int, n_rows: int = N_ROWS) -> SensitivityRow:
    data = confounded_attrition_scm(strength, hidden=False).sample(n_rows, seed)
    short = sm.OLS(data[OUTCOME], sm.add_constant(data[[TREATMENT]])).fit()  # what the config's model sees
    full = sm.OLS(data[OUTCOME], sm.add_constant(data[[TREATMENT, CONFOUNDER]])).fit()
    d_full = sm.OLS(data[TREATMENT], sm.add_constant(data[[CONFOUNDER]])).fit()
    r2_yz = sensitivity.partial_r2_from_t(full.tvalues[CONFOUNDER], full.df_resid)
    r2_dz = sensitivity.partial_r2_from_t(d_full.tvalues[CONFOUNDER], d_full.df_resid)
    seen = sensitivity.linear_sensitivity(data.drop(columns=[CONFOUNDER]), TREATMENT, OUTCOME, [])
    return SensitivityRow(
        strength=strength,
        seed=seed,
        naive_estimate=float(short.params[TREATMENT]),
        full_estimate=float(full.params[TREATMENT]),
        robustness_value=seen.robustness_value,
        robustness_value_alpha=seen.robustness_value_alpha,
        r2_treatment=r2_dz,
        r2_outcome=r2_yz,
        # The simulation knows which way the confounder pushed the estimate; an analyst would
        # have to assume a direction (the default is away from zero).
        adjusted_at_true_strength=sensitivity.adjusted_estimate(
            float(short.params[TREATMENT]), float(short.bse[TREATMENT]), float(short.df_resid), r2_dz, r2_yz,
            bias_sign=float(short.params[TREATMENT] - full.params[TREATMENT]),
        ),
    )


def true_effect(strength: float) -> float:
    return confounded_attrition_scm(strength, hidden=True).true_effect(TREATMENT, OUTCOME, 1.0, n=TRUTH_DRAWS, seed=1)


def run(replicates: int = 10, n_rows: int = N_ROWS, strengths=STRENGTHS) -> dict:
    truths = {s: true_effect(s) for s in strengths}
    rows = [one_draw(s, SEED0 + i, n_rows) for s in strengths for i in range(replicates)]
    return {"settings": {"replicates": replicates, "n_rows": n_rows}, "truth": truths, "rows": rows}


def _ms(values) -> str:
    arr = np.asarray(list(values), dtype=float)
    return f"{arr.mean():+.4f} ± {arr.std(ddof=1):.4f}" if len(arr) > 1 else f"{arr.mean():+.4f}"


def render_markdown(result: dict) -> str:
    rows = pd.DataFrame([{**asdict(r), "identity_error": r.identity_error, "explains": r.explains_away} for r in result["rows"]])
    truth = result["truth"]
    k = result["settings"]["replicates"]
    lines = [
        "# Sensitivity analysis check (hidden confounder with a known strength)",
        "",
        f"Generated by `python -m rootcause.evaluation --sensitivity` ({k} draws per row, n={result['settings']['n_rows']}, "
        "seeds 1000+, the stress tests' hidden-confounder data). The confounder `seniority` is withheld from the pipeline; here "
        "it is kept only so its true strength can be measured. Effects are the change in attrition probability per unit of "
        "compensation (mean ± sd across draws).",
        "",
        "| corr(seniority, pay) | True effect | Naive estimate | Bias | Robustness value | Confounder's true R² (pay / attrition) | "
        "Could it explain the effect away? | Corrected at the true strength | Identity error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for strength, g in rows.groupby("strength", sort=True):
        bias = g["naive_estimate"] - truth[strength]
        lines.append(
            f"| {strength:.2f} | {truth[strength]:+.4f} | {_ms(g['naive_estimate'])} | {_ms(bias)} | "
            f"{g['robustness_value'].mean():.3f} | {g['r2_treatment'].mean():.3f} / {g['r2_outcome'].mean():.3f} | "
            f"{int(g['explains'].sum())}/{len(g)} | {_ms(g['adjusted_at_true_strength'])} | {g['identity_error'].max():.1e} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- **Naive estimate** is what Stage 4's regression reports without the hidden column (the config declares no confounders for compensation). **Bias** is that minus the true effect from `do()`; part of it, about 0.003, is the linear-probability model versus the sigmoid, present even with no confounding (first row).",
        "- **Robustness value** is the share of residual variance a hidden confounder would need to explain in BOTH pay and attrition to reduce the estimate to zero (Cinelli & Hazlett 2020). It is a property of the estimate and its standard error, not a detector, and here it moves the WRONG way: the more the hidden confounder inflates the estimate, the larger and more precise the estimate looks, so the MORE robust it appears (0.21 with no confounding, 0.37 at 0.85). A high robustness value on a confounded estimate is not reassurance, only a statement about erasing the effect entirely.",        "- **Could it explain the effect away** counts draws where the confounder's TRUE partial R² is at least the robustness value in both regressions. Where it is 0/10 the effect is inflated but the sign and rough size survive; the analysis correctly says a much stronger confounder would be needed to erase it. That is a statement about erasing the effect, not about how much of it is real.",
        "- **Corrected at the true strength** applies the omitted-variable-bias formula with the confounder's actual partial R²s, which only a simulation knows. It should equal the estimate that adjusts for the confounder exactly (**identity error** is the largest gap across draws: floating point), and it lands near the true effect. In practice an analyst does not know those two numbers; the benchmark in `EffectEstimate.sensitivity` offers a way to guess them from observed covariates, and how good a guess it is depends on whether the hidden cause resembles the observed ones.",
        "- Corrected estimates differ from the true effect by the same ~0.003 linear-probability gap plus sampling noise.",
    ]
    return "\n".join(lines) + "\n"


def render_json(result: dict) -> str:
    import json

    payload = {
        "settings": result["settings"],
        "truth": {str(k): v for k, v in result["truth"].items()},
        "rows": [{**asdict(r), "identity_error": r.identity_error, "explains_away": r.explains_away} for r in result["rows"]],
    }
    return json.dumps(payload, indent=2)
