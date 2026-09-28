"""Which Stage 5 meta-learner recovers the treatment effect, and its heterogeneity?

Runs every learner (T, S, X, R, DR; linear and gradient-boosting base models) on the Illinois
Workplace Wellness data and scores each against the best yardstick available:

  * REAL randomized trial: the trial's own difference in means, overall and within the
    subgroups Stage 5 reports (sex, age band, race). The real finding to reproduce is a sex
    reversal (women -0.033, men +0.050), so the question is which learners see it. This is
    an estimate with sampling error, not a known truth; on a null average effect the honest
    outcome is a mean near 0.
  * BOOTSTRAP of the real trial: how much each learner's answers move when the 4,834 people
    are resampled (spread, not accuracy).
  * MIRROR simulation with planted subgroup effects: true effects known from do(), so bias
    and error are measurable.

Learners run on the encoded features directly (no Feast round trip: Stage 5 only needs the
preprocessed frame), so this is fast.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rootcause.evaluation import benchmarks, harness
from rootcause.evaluation.scms import SCM_REGISTRY
from rootcause.pipeline import counterfactuals, ingestion, preprocessing, runner
from rootcause.utils.config_loader import ConfigLoader

DOMAIN = "illinois_wellness"
BASES = ("linear", "gbm")
SPECS = [(learner, base) for base in BASES for learner in counterfactuals.LEARNERS]
SEED0 = 1000


@dataclass
class LearnerRun:
    learner: str
    base: str
    mean_cate: float
    cate_std: float
    subgroup_cates: dict[str, float]
    extreme_propensity_share: float | None
    seconds: float

    @property
    def spec(self) -> str:
        return f"{self.learner} / {self.base}"


@dataclass
class Batch:
    """All learners on one dataset (the real file, a bootstrap resample, or a simulated draw)."""

    label: str
    runs: list[LearnerRun] = field(default_factory=list)


def _config(domain_config: dict, learner: str, base: str) -> dict:
    return {**domain_config, "counterfactuals": {**domain_config["counterfactuals"], "meta_learner": learner, "base_learner": base}}


def fit_all(feature_df: pd.DataFrame, domain_config: dict) -> list[LearnerRun]:
    runs = []
    for learner, base in SPECS:
        start = time.perf_counter()
        (result,) = counterfactuals.estimate_counterfactuals(feature_df, _config(domain_config, learner, base))
        runs.append(
            LearnerRun(
                learner=learner,
                base=base,
                mean_cate=result.mean_cate,
                cate_std=result.cate_std or 0.0,
                subgroup_cates={s.key: s.mean_cate for s in result.subgroups},
                extreme_propensity_share=result.extreme_propensity_share,
                seconds=time.perf_counter() - start,
            )
        )
    return runs


def features_of(raw: pd.DataFrame, domain_config: dict) -> pd.DataFrame:
    return preprocessing.preprocess(raw, domain_config).df


def resample(raw: pd.DataFrame, id_col: str, seed: int) -> pd.DataFrame:
    boot = raw.sample(len(raw), replace=True, random_state=seed).reset_index(drop=True)
    boot[id_col] = range(1, len(boot) + 1)
    return boot


@dataclass
class Result:
    reference: list[benchmarks.ExperimentalReference]
    real: Batch
    bootstrap: list[Batch]
    mirror: list[Batch]
    mirror_truth_ate: float
    mirror_truth_subgroups: dict[str, float]
    settings: dict


def run(replicates: int = 10) -> Result:
    cfg = ConfigLoader().get_domain(DOMAIN).extra
    id_col = cfg["ingestion"]["id_column"]
    real_path = runner.resolve_data_path(cfg, "real")
    real_raw = ingestion.ingest(real_path, cfg)

    reference = benchmarks.experimental_reference(real_path, cfg) + benchmarks.experimental_subgroup_references(real_path, cfg)
    real = Batch("real trial", fit_all(features_of(real_raw, cfg), cfg))
    bootstrap = [Batch(f"bootstrap {SEED0 + i}", fit_all(features_of(resample(real_raw, id_col, SEED0 + i), cfg), cfg)) for i in range(replicates)]

    scm = SCM_REGISTRY[(DOMAIN, "synthetic")]()
    treatment, outcome = cfg["counterfactuals"]["treatment"], cfg["counterfactuals"]["outcome"]
    truth_ate = scm.true_effect_of_switching(treatment, outcome, n=harness.TRUTH_DRAWS)
    truth_sub = harness._true_subgroup_effects(scm, cfg)
    n_rows = len(real_raw)
    mirror = []
    for i in range(replicates):
        draw = scm.sample(n_rows, SEED0 + i)
        draw.insert(0, id_col, range(1, n_rows + 1))
        mirror.append(Batch(f"mirror {SEED0 + i}", fit_all(features_of(draw, cfg), cfg)))
    return Result(reference, real, bootstrap, mirror, truth_ate, truth_sub, {"replicates": replicates, "n_rows": n_rows})


# --- report -------------------------------------------------------------------------------


def _ms(values) -> str:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return "–"
    return f"{arr.mean():+.4f} ± {arr.std(ddof=1):.4f}" if len(arr) > 1 else f"{arr.mean():+.4f}"


def _ms_unsigned(values) -> str:
    arr = np.asarray(list(values), dtype=float)
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}" if len(arr) > 1 else f"{arr.mean():.4f}"


def _sex_reversal(run: LearnerRun) -> bool:
    """Women's effect negative and men's positive: the sign pattern the trial shows."""
    return run.subgroup_cates.get("male=0", 0.0) < 0.0 < run.subgroup_cates.get("male=1", 0.0)


def _subgroup_mae(run: LearnerRun, truth: dict[str, float]) -> float:
    keys = [k for k in truth if k in run.subgroup_cates]
    return float(np.mean([abs(run.subgroup_cates[k] - truth[k]) for k in keys])) if keys else float("nan")


def render_markdown(result: Result) -> str:
    k = result.settings["replicates"]
    overall = next(r for r in result.reference if r.subgroup is None)
    subgroup_refs = {r.subgroup: r for r in result.reference if r.subgroup}
    lines = [
        "# Stage 5 meta-learner comparison (Illinois Workplace Wellness)",
        "",
        f"Generated by `python -m rootcause.evaluation --learners` ({k} bootstrap resamples and {k} mirror draws, "
        f"n={result.settings['n_rows']}). Outcome: terminated by Jan 2019. Effects are changes in that probability.",
        "",
        "## Real randomized trial",
        "",
        f"Trial difference in means: **{overall.estimate:+.4f}** [{overall.ci_low:+.4f}, {overall.ci_high:+.4f}] (a null on average). "
        "Within subgroups: "
        + "; ".join(f"{name} {ref.estimate:+.4f} [{ref.ci_low:+.4f}, {ref.ci_high:+.4f}]" for name, ref in subgroup_refs.items() if name.startswith("male"))
        + ". The sex difference is the one exploratory finding; the other splits are null.",
        "",
        "| Learner / base | Mean effect | Inside trial CI? | Women (male=0) | Men (male=1) | Sex pattern | Subgroup MAE vs trial | Effect sd | Extreme propensity |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in result.real.runs:
        truth = {name: ref.estimate for name, ref in subgroup_refs.items()}
        extreme = "–" if run.extreme_propensity_share is None else f"{run.extreme_propensity_share:.0%}"
        lines.append(
            f"| {run.spec} | {run.mean_cate:+.4f} | {'yes' if overall.contains(run.mean_cate) else 'no'} | "
            f"{run.subgroup_cates.get('male=0', float('nan')):+.4f} | {run.subgroup_cates.get('male=1', float('nan')):+.4f} | "
            f"{'reproduced' if _sex_reversal(run) else 'not seen'} | {_subgroup_mae(run, truth):.4f} | {run.cate_std:.4f} | {extreme} |"
        )
    lines += [
        "",
        f"## Stability: {k} bootstrap resamples of the real trial",
        "",
        "| Learner / base | Mean effect | Women | Men | Sex pattern seen in |",
        "|---|---|---|---|---|",
    ]
    for spec in SPECS:
        runs = [r for b in result.bootstrap for r in b.runs if (r.learner, r.base) == spec]
        lines.append(
            f"| {spec[0]} / {spec[1]} | {_ms(r.mean_cate for r in runs)} | {_ms(r.subgroup_cates.get('male=0') for r in runs)} | "
            f"{_ms(r.subgroup_cates.get('male=1') for r in runs)} | {sum(_sex_reversal(r) for r in runs)}/{len(runs)} |"
        )
    truth = result.mirror_truth_subgroups
    lines += [
        "",
        f"## Mirror simulation with planted effects ({k} draws)",
        "",
        f"True average effect {result.mirror_truth_ate:+.4f}; true subgroup effects: "
        + ", ".join(f"{name} {value:+.4f}" for name, value in truth.items() if name.startswith("male"))
        + ". Bias is estimate minus truth (mean ± sd across draws); subgroup MAE averages over all reported subgroups.",
        "",
        "| Learner / base | Mean effect bias | Subgroup MAE | Effect sd |",
        "|---|---|---|---|",
    ]
    for spec in SPECS:
        runs = [r for b in result.mirror for r in b.runs if (r.learner, r.base) == spec]
        lines.append(
            f"| {spec[0]} / {spec[1]} | {_ms(r.mean_cate - result.mirror_truth_ate for r in runs)} | "
            f"{_ms_unsigned(_subgroup_mae(r, truth) for r in runs)} | {np.mean([r.cate_std for r in runs]):.4f} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- **The mirror cannot rank learners on heterogeneity.** Its planted effect is a constant on the logit scale, so the true subgroup effects are nearly equal (women and men differ by 0.004): a learner that reports one number for everyone (the linear S-learner) scores best there because there is nothing to find. The only heterogeneity with any evidence behind it is the real trial's sex split, where the linear S-learner sees nothing and T/X/R/DR see it. Mirror bias and subgroup MAE mostly measure noise.",
        "- The trial's subgroup differences are estimates with wide intervals, so \"Subgroup MAE vs trial\" measures agreement with an estimate, not with a truth. The mirror table is the one with a known answer.",
        "- **Effect sd** is the spread of the per-person effects a learner reports; it includes estimation noise, so it overstates real heterogeneity. A linear S-learner has one coefficient for the treatment, so its sd is exactly 0: it cannot express who benefits and who is harmed.",
        "- **Extreme propensity** is the share of units whose estimated propensity is below 0.05 or above 0.95 (X, R, DR only). In a randomized trial it should be 0%; the learners use the constant treated share here (`counterfactuals.propensity: constant`), which is correct for an RCT and needs no model.",
        "- Gradient-boosting bases can express nonlinear effects but are noisier at this sample size, and can reach for structure that is not there.",
    ]
    return "\n".join(lines) + "\n"


def render_json(result: Result) -> str:
    import json

    def batch(b: Batch) -> dict:
        return {"label": b.label, "runs": [asdict(r) for r in b.runs]}

    return json.dumps(
        {
            "settings": result.settings,
            "reference": [asdict(r) for r in result.reference],
            "real": batch(result.real),
            "bootstrap": [batch(b) for b in result.bootstrap],
            "mirror": [batch(b) for b in result.mirror],
            "mirror_truth": {"ate": result.mirror_truth_ate, "subgroups": result.mirror_truth_subgroups},
        },
        indent=2,
    )
