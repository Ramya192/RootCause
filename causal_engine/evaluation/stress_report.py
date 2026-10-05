"""Renders stress-scenario results as Markdown (the JSON dump is report.render_json).

Reuses report.py's table, effect and ranking blocks; what differs is the summary,
which reports signed bias per treatment (a confounder shows up as a systematic
bias, not as noise) and the edge-level failure pattern across draws.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from causal_engine.evaluation.harness import ScenarioResult, summarize
from causal_engine.evaluation.report import MISSING, _effect_detail, _fmt, _ranking_detail, _table

DEFINITIONS = """\
## How to read this

- Every scenario changes **one** thing about the baseline data-generating process and is run on the same replicate seeds (1000, 1001, …), so a difference from the `baseline` row is attributable to that change. Data is drawn fresh from each scenario's SCM; nothing here is real data.
- **Bias** is the mean over draws of (estimated ATE − true ATE) for each treatment, ± sd across draws. True ATE is the effect of shifting the treatment by +1 (see the main report). A bias that is a large share of the true effect with a small sd is a *systematic* error that more data would not fix.
- **Edge P / R / SHD** score Stage 3's directed edges against the SCM's edges among the *observed* variables. With a hidden confounder the latent variable is not a node, so the spurious edges PC draws in its place count against precision.
- **Refut. pass** is the share of estimates whose permutation-placebo test passed. It only says an estimate stands out from noise. A confounded estimate is just as distinguishable from noise as an unconfounded one, so this column staying at 1.0 next to a large bias is expected, not a contradiction.
- **Top-1 / Kendall τ / ROI regret** compare the intervention ranking with the ranking by true ROI. Top-1 counts every wrong first pick the same, so it makes a near-tie (two interventions whose true ROI differ by 2%) look like a failure; **ROI regret** is the true ROI given up by the pipeline's first pick, as a % of the true best's ROI, averaged over draws, and separates a near-tie (≈0%) from a costly mistake. Where one intervention dominates by a wide margin, Top-1 is a weak test; τ and the effect estimates behind it are more sensitive.
- An edge missed in *every* draw, at any n, is not a sample-size problem: PC orients only edges that a collider (or an already-oriented edge) determines and leaves the rest undirected, and Stage 3 drops undirected edges rather than guess.
- A hidden-confounder scenario and its `observed_confounder` counterpart at the same strength draw the same data; the only difference is whether the pipeline is given the `seniority` column and told to adjust for it.
- **Caveat on the U-shaped scenario:** an average +1 shift is a poor description of an intervention on a U-shaped relationship (the right move is toward the optimum, not a uniform shift), so its "true ATE" is well defined but not what an analyst would want to act on.
"""


def _bias_by_treatment(sc: ScenarioResult) -> dict[str, tuple[float, float, int]]:
    """treatment -> (mean signed bias, sample sd, runs)."""
    diffs: dict[str, list[float]] = defaultdict(list)
    for run in sc.runs:
        for row in run.effect_rows:
            diffs[row.treatment].append(row.estimated - row.true)
    return {
        t: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
        for t, v in diffs.items()
    }


def _edge_frequencies(sc: ScenarioResult) -> Optional[str]:
    """Which true edges were missed and which spurious ones appeared, and in how many runs."""
    ok = [r for r in sc.runs if r.error is None]
    if sc.truth is None or sc.truth.edges is None or not ok:
        return None
    true = set(sc.truth.edges)
    missed: dict[tuple, int] = defaultdict(int)
    spurious: dict[tuple, int] = defaultdict(int)
    for run in ok:
        pred = set(run.predicted_edges)
        for e in true - pred:
            missed[e] += 1
        for e in pred - true:
            spurious[e] += 1

    def fmt(counts: dict) -> str:
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(f"{a}->{b} ({k}/{len(ok)})" for (a, b), k in items) or "none"

    return f"- Missed true edges: {fmt(missed)}\n- Spurious edges: {fmt(spurious)}"


def _regret_percent(sc: ScenarioResult) -> Optional[float]:
    """Mean over runs of the true ROI given up by taking the pipeline's rank-1 pick
    instead of the true best, as a % of the true best's ROI. Unlike Top-1 it
    tells a near-tie (tiny regret) from a real mistake (large regret)."""
    if sc.truth is None or not sc.truth.roi:
        return None
    best = max(sc.truth.roi.values())
    regrets = [
        r.values["roi_regret"] / best
        for r in sc.runs
        if r.error is None and r.values.get("roi_regret") is not None
    ]
    return 100.0 * float(np.mean(regrets)) if regrets else None


def _summary_table(results: list[ScenarioResult]) -> str:
    treatments = sorted({t for sc in results for t in _bias_by_treatment(sc)})
    header = (
        ["Scenario", "n", "Runs", "Edge P", "Edge R", "SHD"]
        + [f"Bias: {t}" for t in treatments]
        + ["Refut. pass", "Top-1", "Kendall τ", "ROI regret"]
    )
    rows = []
    for sc in results:
        ok = [r for r in sc.runs if r.error is None]
        s = summarize(sc.runs)
        bias = _bias_by_treatment(sc)

        regret = _regret_percent(sc)

        def fmt_bias(t: str) -> str:
            if t not in bias:
                return MISSING
            mean, sd, n = bias[t]
            return f"{mean:+.3f}" if n == 1 else f"{mean:+.3f} ± {sd:.3f}"

        rows.append(
            [sc.stress_id or sc.dataset, "/".join(str(n) for n in sorted({r.n_rows for r in ok})) or MISSING,
             f"{len(ok)}/{len(sc.runs)}", _fmt(s, "edge_precision"), _fmt(s, "edge_recall"), _fmt(s, "shd", 1)]
            + [fmt_bias(t) for t in treatments]
            + [_fmt(s, "refutation_pass_rate"), _fmt(s, "top1_correct"), _fmt(s, "rank_tau"),
               MISSING if regret is None else f"{regret:.1f}%"]
        )
    return _table(header, rows)


def render_markdown(results: list[ScenarioResult], settings: dict) -> str:
    out = [
        "# RootCause stress tests",
        "",
        "Generated by `python -m causal_engine.evaluation --stress` "
        f"(replicates={settings['replicates']}). Each scenario changes one thing about the baseline "
        "attrition data and is scored against effects simulated from its own SCM.",
        *(["", f"**Stage 3 algorithm forced to `{settings['algorithm']}`** (`--algorithm`); the graph columns below are that algorithm's, not the configured one's."] if settings.get("algorithm") else []),
        "",
        "## Summary",
        "",
        _summary_table(results),
        "",
    ]
    for sc in results:
        out += [f"## {sc.stress_id} ({sc.stress_factor})", "", sc.stress_description or "", ""]
        for block in (_effect_detail(sc), _edge_frequencies(sc), _ranking_detail(sc)):
            if block:
                out += [block, ""]

    failed = [(sc, r) for sc in results for r in sc.runs if r.error]
    if failed:
        out += ["## Failed runs", ""]
        out += [f"- {sc.stress_id} / {r.label}: {r.error}" for sc, r in failed]
        out.append("")
    out.append(DEFINITIONS)
    return "\n".join(out)
