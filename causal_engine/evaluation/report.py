"""Renders harness results as a Markdown report and a JSON dump."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict
from typing import Optional

import numpy as np

from causal_engine.evaluation.harness import BOOTSTRAP_SOURCE, ScenarioResult, summarize

MISSING = "–"

DEFINITIONS = """\
## How to read this

- **Ground truth** comes from a structural causal model (SCM) the data was generated from, so it exists only for synthetic and semi-synthetic datasets. Datasets without one show `–` for every truth-based column; they are scored only on the placebo pass rate.
- **Semi-synthetic** datasets keep REAL covariates (a bootstrap draw of the real applicants) and simulate only the outcome, with lever effects planted by the SCM. That validates the method on realistic, correlated inputs, and says nothing about what really causes the outcome. The graph among real covariates is unknown, so Stage 3 is not scored there (its columns show `–`).
- **Fairness check (Stage 6)** is fairlearn's four-fifths rule on the outcome rate across the configured sensitive attribute: lowest group rate over highest, flagged below the threshold. It is a population-level descriptive statistic, not a test of discrimination or of any recommendation's effect on groups. Where an SCM exists, the *true ratio* is the same statistic on 1M simulated rows, and "verdict matches truth" is the share of runs whose pass/flag agrees with it. Small groups make the sample ratio noisy near the threshold.
- **Bootstrap stability** (real data, no SCM): the pipeline is re-run on resamples of the real rows (drawn with replacement, same size). It shows whether the estimates, the placebo verdict, the fairness flag and the top-ranked intervention survive resampling the applicants. It measures spread, not accuracy: there is no truth to be accurate against, and it cannot see bias from unmeasured confounding.
- **Edge P / R / SHD** compare Stage 3's directed edges to the SCM's edges over the variables Stage 3 searches. An edge PC left unoriented is dropped by Stage 3, so it counts as missed. SHD counts edge additions, deletions and reversals (a reversal costs 1).
- **Edge R (excl. priors)** ignores edges the domain config supplies as `required_edges`. Those are asserted, not discovered, so they flatter the headline recall; this column is what structure learning found on its own.
- **True ATE** is the change in the outcome (a probability, for a binary outcome) when everyone's treatment is shifted by +1, simulated by `do()` on the SCM with paired random draws (1M rows). This matches how Stage 6 uses an ATE (`ate × expected_shift`). For a binary treatment (0/1) "shift by +1" is not meaningful, so the true ATE is the effect of switching everyone from 0 to 1. Stage 4's linear-regression slope estimates something slightly different for a sigmoid outcome (about 0.002–0.003 on the attrition data), so a small floor on ATE error is expected and is not estimator failure.
- **ATE MAE** is the mean absolute error of the estimated vs true ATE across the treatments.
- **Refut. pass** is the share of estimates that beat a permutation placebo: Stage 4 shuffles the treatment 100 times, re-estimates, and passes an estimate whose two-sided permutation p-value is below 0.05. On a treatment with a real effect this should be near 1; on a null treatment (e.g. the Illinois RCT) failing is the *correct* outcome and reads "no evidence of an effect beyond noise". It checks that an estimate stands out from noise, not that it is unconfounded: no placebo can detect an unmeasured confounder.
- **Top-1 / Kendall τ** compare the pipeline's intervention ranking to the ranking by *true* ROI (true effect of the configured shift ÷ cost). Top-1 is whether its first choice is the true best; τ is rank agreement across all candidates (1 = identical order). With a single candidate there is nothing to rank, so both are trivial or undefined.
- **Trial reference** (real randomized data only): the difference in mean outcome between the arms, with a Welch 95% interval. It is an *estimate* with sampling error, not a known truth like a simulated `do()` effect, so the check is whether Stage 4's number falls inside the interval and whether the two agree on "is there an effect", not whether the error is small. On a null trial such as the Illinois wellness study a *failed* placebo test is the correct outcome; a passed one would be an invented effect. Stage 5's mean CATE is shown next to it as a second estimate of the same average effect.
- **Effect heterogeneity** (domains that list `counterfactuals.subgroups`): Stage 5's T-learner gives a per-unit effect; the report averages it within each level of the listed binary columns. On real randomized data the yardstick is the trial's own difference in means within that subgroup (wide intervals, and several subgroups mean some exclude 0 by chance); on simulated data it is the true effect within the subgroup from `do()`. The T-learner here is two linear regressions, so it can only express heterogeneity that is linear in the covariates, and on a probability scale even a constant logit effect varies with baseline risk. The reported sd of per-unit effects includes estimation noise, so it overstates how much the true effect varies.
- **Replicates** are fresh draws from the SCM (same size as the committed data). Values are mean ± sd across draws. Stage 5 and Stage 7 are not scored: neither has a `do()`-defined truth.
"""


def _fmt(summary: dict, key: str, digits: int = 2) -> str:
    if key not in summary:
        return MISSING
    mean, sd, n = summary[key]
    if key == "top1_correct":
        return f"{round(mean * n)}/{n}"
    return f"{mean:.{digits}f}" if n == 1 else f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _summary_table(results: list[ScenarioResult]) -> str:
    header = [
        "Domain / dataset", "Source", "Runs", "Edge P", "Edge R", "Edge R (excl. priors)",
        "SHD", "ATE MAE", "Refut. pass", "Top-1", "Kendall τ",
    ]
    rows = []
    for sc in results:
        name = f"{sc.domain_id} / {sc.dataset}"
        if sc.skipped_reason:
            rows.append([name, sc.source, "skipped"] + [MISSING] * 8)
            continue
        ok = [r for r in sc.runs if r.error is None]
        s = summarize(sc.runs)
        rows.append([
            name, sc.source, f"{len(ok)}/{len(sc.runs)}",
            _fmt(s, "edge_precision"), _fmt(s, "edge_recall"), _fmt(s, "discovered_recall"),
            _fmt(s, "shd", 1), _fmt(s, "ate_mae", 3), _fmt(s, "refutation_pass_rate"),
            _fmt(s, "top1_correct"), _fmt(s, "rank_tau"),
        ])
    return _table(header, rows)


def _effect_detail(sc: ScenarioResult) -> Optional[str]:
    by_treatment: dict[str, list[float]] = defaultdict(list)
    for run in sc.runs:
        for row in run.effect_rows:
            by_treatment[row.treatment].append(row.estimated)
    if not by_treatment or sc.truth is None:
        return None
    rows = []
    for treatment, ests in by_treatment.items():
        true = sc.truth.effects[treatment]
        mean = float(np.mean(ests))
        est = f"{mean:+.4f}" if len(ests) == 1 else f"{mean:+.4f} ± {np.std(ests, ddof=1):.4f}"
        rows.append([treatment, f"{true:+.4f}", est, f"{mean - true:+.4f}"])
    return _table(["Treatment", "True ATE (+1 shift)", "Estimated ATE", "Bias"], rows)


def _graph_detail(sc: ScenarioResult) -> Optional[str]:
    if sc.truth is None or sc.truth.edges is None:
        return None
    true = set(sc.truth.edges)
    lines = []
    for run in sc.runs:
        if run.error:
            continue
        pred = set(run.predicted_edges)
        missing, extra = sorted(true - pred), sorted(pred - true)
        if not missing and not extra:
            continue
        lines.append(
            f"- {run.label}: missing {['->'.join(e) for e in missing] or 'none'}, "
            f"spurious {['->'.join(e) for e in extra] or 'none'}"
        )
    if not lines:
        return f"Every run recovered all {len(true)} true edges with no spurious ones."
    return "\n".join(lines)


def _reference_detail(sc: ScenarioResult) -> Optional[str]:
    """Pipeline vs the randomized trial's own estimate, per binary treatment."""
    runs = [r for r in sc.runs if r.error is None]
    if not sc.reference or not runs:
        return None
    run = runs[0]
    rows, verdicts = [], []
    for ref in sc.reference:
        ate = run.estimated_ates.get(ref.treatment)
        if ate is None:
            continue
        p = run.refutation_p_values.get(ref.treatment)
        passed = run.refutation_passed.get(ref.treatment)
        cate = run.mean_cates.get(ref.treatment)
        rows.append([
            ref.treatment,
            f"{ref.estimate:+.4f} [{ref.ci_low:+.4f}, {ref.ci_high:+.4f}]",
            f"{ate:+.4f}",
            "yes" if ref.contains(ate) else "NO",
            MISSING if p is None else f"{p:.3f} ({'passed' if passed else 'failed'})",
            MISSING if cate is None else f"{cate:+.4f}",
        ])
        if passed is not None:
            agree = passed == ref.significant
            verdicts.append(
                f"- {ref.treatment}: the trial {'finds' if ref.significant else 'finds no'} effect "
                f"(95% interval {'excludes' if ref.significant else 'includes'} 0) and the pipeline's placebo test "
                f"{'passed' if passed else 'failed'} ({'agree' if agree else 'DISAGREE'})."
            )
    if not rows:
        return None
    ref = sc.reference[0]
    lines = [
        _table(
            ["Treatment", "Trial: treated − control [95% CI]", "Pipeline ATE (Stage 4)", "Inside trial CI?",
             "Placebo p", "Stage 5 mean CATE"],
            rows,
        ),
        "",
        f"Arms: {ref.n_treated} treated, {ref.n_control} control with an observed outcome; outcome missing for "
        f"{ref.outcome_missing_treated:.1%} of treated and {ref.outcome_missing_control:.1%} of control rows "
        "(rows without an outcome are dropped at ingestion, so any difference between the arms would bias the estimate).",
        "",
        *verdicts,
    ]
    return "\n".join(lines)


def _heterogeneity_detail(sc: ScenarioResult) -> Optional[str]:
    """Stage 5's mean effect within subgroups: against the trial's own subgroup estimates
    (real randomized data) or against the SCM's true subgroup effects (simulated data)."""
    runs = [r for r in sc.runs if r.error is None and r.subgroup_cates]
    if not runs:
        return None
    stds = [r.cate_stds[t] for r in runs for t in list(r.cate_stds)[:1] if r.cate_stds[t] is not None]
    spread = f"Spread of Stage 5's per-{'unit'} effects (sd): {np.mean(stds):.4f}." if stds else ""

    if sc.subgroup_reference:
        run = runs[0]
        rows = []
        by_variable: dict[str, dict[str, object]] = defaultdict(dict)
        for ref in sc.subgroup_reference:
            est = run.subgroup_cates.get(ref.subgroup)
            if est is None:
                continue
            variable, level = ref.subgroup.split("=")
            by_variable[variable][level] = (ref, est)
            rows.append([
                ref.subgroup, f"{ref.n_treated} / {ref.n_control}",
                f"{ref.estimate:+.4f} [{ref.ci_low:+.4f}, {ref.ci_high:+.4f}]",
                f"{est:+.4f}", "yes" if ref.contains(est) else "NO",
            ])
        if not rows:
            return None

        # The question heterogeneity asks is whether the effect DIFFERS between the levels of a
        # variable, not whether one level's interval excludes 0: test the difference directly.
        tested = {v: levels for v, levels in by_variable.items() if set(levels) == {"0", "1"}}
        interaction = []
        for variable, levels in tested.items():
            (r0, e0), (r1, e1) = levels["0"], levels["1"]
            diff = r1.estimate - r0.estimate
            se = math.sqrt(r1.std_error**2 + r0.std_error**2)
            p = math.erfc(abs(diff / se) / math.sqrt(2))
            interaction.append([
                variable, f"{diff:+.4f} ± {1.959964 * se:.4f}", f"{p:.4f}", f"{min(1.0, p * len(tested)):.4f}",
                f"{e1 - e0:+.4f}",
            ])
        lines = [
            _table(["Subgroup", "Trial n (treated / control)", "Trial: treated − control [95% CI]",
                    "Pipeline Stage 5 mean effect", "Inside trial CI?"], rows),
            "",
            "Does the effect differ between the two levels of each variable? (trial: effect at level 1 minus level 0, "
            "with a two-sided z-test; the adjusted p multiplies by the number of variables tested)",
            "",
            _table(["Variable", "Trial: level 1 − level 0 (± 95%)", "p", "Bonferroni p", "Pipeline Stage 5: level 1 − level 0"],
                   interaction),
            "",
            f"{sum(float(r[3]) < 0.05 for r in interaction)} of {len(interaction)} variables differ at an adjusted p < 0.05. "
            "This is exploratory: these four variables were picked for this analysis, not pre-specified by the trial. " + spread,
        ]
        return "\n".join(lines)

    if sc.truth is not None and sc.truth.subgroup_effects:
        rows = []
        for key, true in sc.truth.subgroup_effects.items():
            ests = [r.subgroup_cates[key] for r in runs if key in r.subgroup_cates]
            if not ests:
                continue
            mean = float(np.mean(ests))
            est = f"{mean:+.4f}" if len(ests) == 1 else f"{mean:+.4f} ± {np.std(ests, ddof=1):.4f}"
            rows.append([key, f"{true:+.4f}", est, f"{mean - true:+.4f}"])
        if not rows:
            return None
        s_ = summarize(sc.runs)
        mae = _fmt(s_, "subgroup_cate_mae", 4)
        return "\n".join([
            _table(["Subgroup", "True effect", "Stage 5 mean effect", "Bias"], rows),
            "",
            f"Mean absolute error across subgroups: {mae}. " + spread,
        ])
    return None


def _mean_sd(values: list[float], digits: int = 3) -> str:
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def _fairness_section(results: list[ScenarioResult]) -> Optional[str]:
    """Stage 6's four-fifths check per dataset, and against the population's true ratio where the
    dataset comes from an SCM."""
    rows = []
    for sc in results:
        runs = [r for r in sc.runs if r.error is None and r.fairness_ratio is not None]
        if sc.skipped_reason or not runs:
            continue
        true = sc.truth.fairness_ratio if sc.truth is not None else None
        matches = [r.values["fairness_flag_correct"] for r in runs if "fairness_flag_correct" in r.values]
        errors = [r.values["fairness_ratio_abs_err"] for r in runs if "fairness_ratio_abs_err" in r.values]
        rows.append([
            f"{sc.domain_id} / {sc.dataset}", sc.source, str(len(runs)),
            MISSING if true is None else f"{true:.3f}",
            _mean_sd([r.fairness_ratio for r in runs]),
            f"{sum(r.fairness_passed is False for r in runs)}/{len(runs)}",
            f"{round(sum(matches))}/{len(matches)}" if matches else MISSING,
            _mean_sd(errors) if errors else MISSING,
        ])
    if not rows:
        return None
    return _table(
        ["Domain / dataset", "Source", "Runs", "True ratio", "Stage 6 ratio", "Flagged (ratio < threshold)",
         "Verdict matches truth", "Ratio abs. error"],
        rows,
    )


def _stability_detail(sc: ScenarioResult, reference: Optional[ScenarioResult]) -> Optional[str]:
    """How much the pipeline's answers move when the real applicants are resampled (bootstrap),
    next to the full-sample answer. Stability, not accuracy: real data has no truth."""
    runs = [r for r in sc.runs if r.error is None]
    if not runs:
        return None
    full = next((r for r in (reference.runs if reference else []) if r.error is None), None)

    rows = []
    for treatment in runs[0].estimated_ates:
        ates = [r.estimated_ates[treatment] for r in runs if treatment in r.estimated_ates]
        passed = [r.refutation_passed.get(treatment) for r in runs]
        anchor = full.estimated_ates.get(treatment) if full else None
        same_sign = (
            MISSING if not anchor else f"{sum(np.sign(a) == np.sign(anchor) for a in ates)}/{len(ates)}"
        )
        rows.append([
            treatment,
            MISSING if anchor is None else f"{anchor:+.4f}",
            f"{np.mean(ates):+.4f} ± {np.std(ates, ddof=1) if len(ates) > 1 else 0.0:.4f}",
            same_sign,
            MISSING if all(p is None for p in passed) else f"{sum(p is True for p in passed)}/{len(passed)}",
        ])
    blocks = [
        _table(["Treatment", "Full-sample ATE", "Bootstrap mean ± sd", "Same sign as full sample", "Placebo passed"], rows)
    ]

    ratios = [r.fairness_ratio for r in runs if r.fairness_ratio is not None]
    if ratios:
        anchor = full.fairness_ratio if full else None
        flagged = sum(r.fairness_passed is False for r in runs if r.fairness_ratio is not None)
        blocks.append(
            f"Fairness ratio across resamples: {_mean_sd(ratios)}"
            + (f" (full sample {anchor:.3f})" if anchor is not None else "")
            + f"; flagged below the threshold in {flagged}/{len(ratios)}."
        )
    choices: dict[str, int] = defaultdict(int)
    for r in runs:
        if r.top_choice:
            choices[r.top_choice] += 1
    if choices:
        blocks.append(
            "Rank-1 intervention across resamples: "
            + ", ".join(f"{c} ×{k}" for c, k in sorted(choices.items(), key=lambda kv: -kv[1]))
            + (f" (full sample: {full.top_choice})." if full and full.top_choice else ".")
        )
    return "\n\n".join(blocks)


def _ranking_detail(sc: ScenarioResult) -> Optional[str]:
    if sc.truth is None or not sc.truth.roi:
        return None
    true_order = sorted(sc.truth.roi, key=sc.truth.roi.get, reverse=True)
    lines = ["True ROI order (true outcome reduction per unit cost): "
             + " > ".join(f"{c} ({sc.truth.roi[c]:.2e})" for c in true_order)]
    chosen = [r.ranking.chosen for r in sc.runs if r.ranking]
    if chosen:
        counts = defaultdict(int)
        for c in chosen:
            counts[c] += 1
        lines.append(
            "Pipeline rank-1 choice: " + ", ".join(f"{c} ×{k}" for c, k in sorted(counts.items(), key=lambda kv: -kv[1]))
        )
    return "\n\n".join(lines)


def render_markdown(results: list[ScenarioResult], settings: dict) -> str:
    out = [
        "# RootCause evaluation results",
        "",
        "Generated by `python -m causal_engine.evaluation` "
        f"(replicates={settings['replicates']}, n_rows={settings['n_rows'] or 'size of each dataset file'}).",
        *(["", f"**Stage 3 algorithm forced to `{settings['algorithm']}`** (`--algorithm`); the graph columns below are that algorithm's, not the configured one's."] if settings.get("algorithm") else []),
        "",
        "## Summary",
        "",
        _summary_table(results),
        "",
    ]

    fairness = _fairness_section(results)
    if fairness:
        out += ["## Fairness check (Stage 6)", "", fairness, ""]

    for sc in results:
        if sc.source == BOOTSTRAP_SOURCE and not sc.skipped_reason:
            committed = next(
                (c for c in results if (c.domain_id, c.dataset, c.source) == (sc.domain_id, sc.dataset, "committed file")),
                None,
            )
            stability = _stability_detail(sc, committed)
            if stability:
                out += [f"## {sc.domain_id} / {sc.dataset} — stability under resampling", "", stability, ""]
            continue
        detail = None if sc.skipped_reason else _reference_detail(sc)
        if detail:
            out += [f"## {sc.domain_id} / {sc.dataset} — against the randomized trial", "", detail, ""]
            hetero = _heterogeneity_detail(sc)
            if hetero:
                out += ["### Effect heterogeneity (Stage 5 subgroups)", "", hetero, ""]
        if sc.skipped_reason or not sc.scored_against_truth:
            continue
        out += [f"## {sc.domain_id} / {sc.dataset} — {sc.source}", ""]
        for block in (_effect_detail(sc), _graph_detail(sc), _ranking_detail(sc)):
            if block:
                out += [block, ""]
        hetero = _heterogeneity_detail(sc)
        if hetero:
            out += ["### Effect heterogeneity (Stage 5 subgroups)", "", hetero, ""]

    problems = [
        (sc, r) for sc in results for r in sc.runs if r.error
    ] + [(sc, None) for sc in results if sc.skipped_reason]
    if problems:
        out += ["## Skipped / failed", ""]
        for sc, run in problems:
            what = sc.skipped_reason if run is None else f"{run.label}: {run.error}"
            out.append(f"- {sc.domain_id} / {sc.dataset} ({sc.source}): {what}")
        out.append("")

    out.append(DEFINITIONS)
    return "\n".join(out)


def render_json(results: list[ScenarioResult], settings: dict) -> str:
    return json.dumps({"settings": settings, "scenarios": [asdict(r) for r in results]}, indent=2, default=str)
