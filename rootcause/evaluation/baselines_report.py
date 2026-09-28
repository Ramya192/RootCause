"""Renders the pipeline-vs-naive-baselines comparison as Markdown.

Every method is scored on the SAME draws (same seeds, same data) against the same
simulated truth, so a row-to-row difference within a scenario is the method, not luck.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from rootcause.evaluation import baselines
from rootcause.evaluation.baselines import METHOD_LABELS, SHAP_IMPORTANCE
from rootcause.evaluation.harness import ScenarioResult
from rootcause.evaluation.metrics import EffectRow, RankingScore
from rootcause.evaluation.report import MISSING, _table

METHODS = ("pipeline", baselines.REGRESSION_ALL, baselines.REGRESSION_TREATMENT_ONLY, SHAP_IMPORTANCE)

DEFINITIONS = """\
## How to read this

- **Same data, same truth.** Every method runs on the same draws (seeds 1000, 1001, …) of each scenario's SCM and is scored against effects simulated by `do()` on that SCM. Nothing here is real data, and the magnitudes depend on the effect sizes chosen for the simulation; the *pattern* (which method is right, which is wrong, and why) is the finding.
- **RootCause pipeline** is Stages 1–6 exactly as in the stress report: PC-discovered graph, DoWhy backdoor linear regression with the confounders the domain config declares, ROI ranking.
- **Regression, all columns** regresses the outcome on every observed column (mediators included) and reads the treatment's coefficient. This is what "throw everything in the model" does. Because compensation, manager quality and workload act on attrition *through* satisfaction and burnout, controlling for those two removes the very pathway that carries the effect: the coefficient is the direct effect, which is ~0 here, so it reads as "this lever does nothing".
- **Regression, treatment only** regresses the outcome on the treatment alone. With no confounder it is unbiased (and identical in spirit to the pipeline, which adjusts for nothing there); with a confounder it is biased, and cannot notice.
- **SHAP importance ÷ cost** ranks the candidate interventions by the mean |SHAP| of the variable each one targets, per dollar. It has no effect estimate (so no bias columns) and no sign: importance says how much the fitted model *uses* a variable to predict attrition, not what happens if you change it. `expected_shift` cannot enter the ranking. The table under "SHAP importances" shows where its weight goes.
- **Bias** is the mean over draws of (estimated ATE − true ATE) for each treatment, ± sd across draws; true ATE is the effect of shifting the treatment by +1. **Sign correct** is the share of (draw, treatment) estimates with the right sign, which is where a coefficient near zero shows up as a coin flip. **Top-1 / Kendall τ / ROI regret** compare the ranking with the ranking by true ROI, as in the stress report; regret is the true ROI given up by the first pick, as a % of the true best's ROI, averaged over draws.
- **What this does and does not show.** The pipeline's adjustment set is *declared in the domain config*, not learned: Stage 4 estimates each lever's total effect and adjusts only for the confounders the config names. That is what separates it from "regress on everything". So (a) against *regression, all columns* it wins because it does not condition on mediators; (b) against *regression, treatment only* it ties whenever no confounder exists, because it is the same regression: an analyst with the same causal knowledge does equally well, and the pipeline's contribution there is encoding that knowledge plus the graph, the refutation test and the ROI ranking, not a better estimator; (c) it beats treatment-only regression only where an *observed* confounder is declared. It has no way to correct a *hidden* confounder: in `hidden_confounder_*` the pipeline and treatment-only regression make the same error, by construction.
"""


def _method_runs(sc: ScenarioResult, method: str) -> list[tuple[list[EffectRow], Optional[RankingScore]]]:
    """Per successful run: (effect rows, ranking) for `method`."""
    runs = []
    for run in sc.runs:
        if run.error:
            continue
        if method == "pipeline":
            runs.append((run.effect_rows, run.ranking))
        elif method in run.baselines:
            b = run.baselines[method]
            runs.append((b.effect_rows, b.ranking))
    return runs


def _bias(runs, treatment: str) -> Optional[tuple[float, float, int]]:
    diffs = [r.estimated - r.true for rows, _ in runs for r in rows if r.treatment == treatment]
    if not diffs:
        return None
    return float(np.mean(diffs)), float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0, len(diffs)


def _sign_correct(runs) -> Optional[str]:
    flags = [r.sign_agrees for rows, _ in runs for r in rows]
    return f"{sum(flags)}/{len(flags)}" if flags else None


def _ranking_cells(sc: ScenarioResult, runs) -> tuple[str, str, str]:
    rankings = [rk for _, rk in runs if rk is not None]
    if not rankings:
        return MISSING, MISSING, MISSING
    top1 = f"{sum(rk.top1_correct for rk in rankings)}/{len(rankings)}"
    taus = [rk.kendall_tau for rk in rankings if rk.kendall_tau is not None]
    tau = MISSING
    if taus:
        tau = f"{np.mean(taus):.2f}" if len(taus) == 1 else f"{np.mean(taus):.2f} ± {np.std(taus, ddof=1):.2f}"
    regret = MISSING
    if sc.truth and sc.truth.roi:
        best = max(sc.truth.roi.values())
        regret = f"{100 * np.mean([rk.roi_regret / best for rk in rankings]):.1f}%"
    return top1, tau, regret


def _summary_table(results: list[ScenarioResult]) -> str:
    treatments = sorted({t for sc in results if sc.truth for t in sc.truth.effects})
    header = ["Scenario", "Method"] + [f"Bias: {t}" for t in treatments] + ["Sign correct", "Top-1", "Kendall τ", "ROI regret"]
    rows = []
    for sc in results:
        for method in METHODS:
            runs = _method_runs(sc, method)
            if not runs:
                continue
            cells = []
            for t in treatments:
                b = _bias(runs, t)
                cells.append(MISSING if b is None else (f"{b[0]:+.3f}" if b[2] == 1 else f"{b[0]:+.3f} ± {b[1]:.3f}"))
            top1, tau, regret = _ranking_cells(sc, runs)
            rows.append(
                [sc.stress_id or sc.dataset, METHOD_LABELS[method]] + cells + [_sign_correct(runs) or MISSING, top1, tau, regret]
            )
    return _table(header, rows)


def _rank1_table(results: list[ScenarioResult]) -> str:
    rows = []
    for sc in results:
        true_best = max(sc.truth.roi, key=sc.truth.roi.get) if sc.truth and sc.truth.roi else None
        cells = []
        for method in METHODS:
            picks: dict[str, int] = defaultdict(int)
            for _, rk in _method_runs(sc, method):
                if rk is not None:
                    picks[rk.chosen] += 1
            cells.append(", ".join(f"{c} ×{k}" for c, k in sorted(picks.items(), key=lambda kv: -kv[1])) or MISSING)
        rows.append([sc.stress_id or sc.dataset, true_best or MISSING] + cells)
    return _table(["Scenario", "True best"] + [METHOD_LABELS[m] for m in METHODS], rows)


def _shap_table(results: list[ScenarioResult]) -> str:
    features: list[str] = []
    for sc in results:
        for run in sc.runs:
            b = run.baselines.get(SHAP_IMPORTANCE)
            for f in (b.importances or {}) if b else ():
                if f not in features:
                    features.append(f)
    rows = []
    for sc in results:
        bs = [r.baselines[SHAP_IMPORTANCE] for r in sc.runs if not r.error and SHAP_IMPORTANCE in r.baselines]
        if not bs:
            continue
        means = [
            MISSING if not any(f in b.importances for b in bs)
            else f"{np.mean([b.importances[f] for b in bs if f in b.importances]):.3f}"
            for f in features
        ]
        levers = sum(b.values.get("top_feature_is_lever") == 1.0 for b in bs)
        rows.append([sc.stress_id or sc.dataset] + means + [f"{levers}/{len(bs)}"])
    return _table(["Scenario"] + features + ["Top feature is a lever"], rows)


def _true_effects_line(results: list[ScenarioResult]) -> str:
    seen: dict[str, dict[str, float]] = {}
    for sc in results:
        if sc.truth:
            seen[sc.stress_id or sc.dataset] = sc.truth.effects
    rows = []
    for name, effects in seen.items():
        rows.append([name] + [f"{v:+.4f}" for v in effects.values()])
    treatments = list(next(iter(seen.values())).keys()) if seen else []
    return _table(["Scenario"] + [f"True ATE: {t}" for t in treatments], rows)


def render_markdown(results: list[ScenarioResult], settings: dict) -> str:
    out = [
        "# RootCause vs naive baselines",
        "",
        "Generated by `python -m rootcause.evaluation --baselines` "
        f"(replicates={settings['replicates']}). The pipeline, two plain regressions and a SHAP "
        "importance ranking are run on the same simulated data and scored against the same true effects.",
        *(["", f"**Stage 3 algorithm forced to `{settings['algorithm']}`** (`--algorithm`); the graph columns below are that algorithm's, not the configured one's."] if settings.get("algorithm") else []),
        "",
        "## Summary",
        "",
        _summary_table(results),
        "",
        "## Rank-1 intervention chosen",
        "",
        _rank1_table(results),
        "",
        "## SHAP importances (mean |SHAP| over draws)",
        "",
        "`job_satisfaction` and `burnout` are mediators; the other columns are the levers the interventions move.",
        "",
        _shap_table(results),
        "",
        "## True effects",
        "",
        _true_effects_line(results),
        "",
    ]
    failed = [(sc, r) for sc in results for r in sc.runs if r.error]
    if failed:
        out += ["## Failed runs", ""]
        out += [f"- {sc.stress_id} / {r.label}: {r.error}" for sc, r in failed]
        out.append("")
    out.append(DEFINITIONS)
    return "\n".join(out)
