"""Runs the pipeline over every configured dataset and scores it.

For each runnable domain and each dataset kind its config lists, run Stages
1-6 on the committed data file. Where the dataset comes from a known SCM
(scms.SCM_REGISTRY), also score against ground truth, and optionally re-run on
`replicates` fresh draws from that SCM -- a single draw is an anecdote, and
recovery rates only mean something as a distribution over draws.

What is scored, and against what:
  - Stage 3 graph:     directed-edge precision/recall/SHD vs the SCM's edges
  - Stage 4 effects:   estimated ATE vs the true ATE from simulating do()
  - Stage 4 refuter:   share of estimates distinguishable from a shuffled-treatment
                       placebo (needs no ground truth)
  - Stage 6 ranking:   the pipeline's ranking vs the ranking by true ROI
Stage 5 (a T-learner on a dichotomised treatment) and Stage 7 (narrative) have
no do()-defined truth and are not scored.

  - Stage 6 fairness:  the sample's four-fifths verdict vs the SCM population's true ratio
An SCM built on real covariates (semi-synthetic) has no known graph, so Stage 3 is not scored.

Datasets with no SCM (real data) get only the ground-truth-free metrics, plus bootstrap
resamples of the file when `replicates` > 0 (stability, not accuracy). If the real data
comes from a randomized trial (benchmarks.RCT_DATASETS) it is also compared with the
trial's own difference-in-means estimate, which is an estimate with an interval, not a truth.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from rootcause.evaluation import baselines as baseline_methods
from rootcause.evaluation import benchmarks
from rootcause.evaluation import metrics
from rootcause.evaluation.scm import SCM
from rootcause.evaluation.scms import SCM_REGISTRY
from rootcause.evaluation.stress import StressScenario
from rootcause.evaluation.stress import select as select_stress
from rootcause.pipeline import runner
from rootcause.utils.config_loader import ConfigLoader

DEFAULT_N_ROWS = 2000
REPLICATE_SEED_BASE = 1000  # replicate i uses seed 1000 + i; the committed data used 42
EVAL_DATASET_PREFIX = "eval_"  # keeps harness Feast state apart from the API's
BOOTSTRAP_SOURCE = "bootstrap of committed file"  # ScenarioResult.source of the bootstrap runs
MULTIMODAL = "multimodal"  # attachments are keyed by record id: not part of the default sweep, never bootstrapped
TRUTH_DRAWS = 1_000_000  # Monte Carlo draws behind each true effect

Edge = tuple[str, str]


@dataclass(frozen=True)
class Truth:
    """What the SCM says is true for one domain config, computed once."""

    # among the variables Stage 3 searches over; None when the SCM's graph is not known
    # (real covariates), in which case Stage 3 is not scored
    edges: Optional[list[Edge]]
    effects: dict[str, float]  # treatment -> effect on outcome of shifting it by +1
    roi: dict[str, float]  # intervention id -> true ROI of its configured shift
    # "male=1" -> true effect of the Stage 5 treatment within that subgroup; empty unless the
    # config lists `counterfactuals.subgroups` and the Stage 5 treatment is binary
    subgroup_effects: dict[str, float] = field(default_factory=dict)
    # Stage 6's fairness statistic on the SCM's own population: lowest group outcome rate over
    # the highest, across the configured sensitive attribute (None if the SCM lacks it)
    fairness_ratio: Optional[float] = None


def _true_effect(scm: SCM, node: str, outcome: str, shift: float) -> float:
    """Effect on the outcome of moving `node` by `shift`. A binary node cannot be
    "shifted" (the already-treated would go to 2), so its effect is the contrast between
    switching everyone on and everyone off, scaled by `shift`."""
    if scm.is_binary(node):
        return shift * scm.true_effect_of_switching(node, outcome, n=TRUTH_DRAWS)
    return scm.true_effect(node, outcome, shift=shift, n=TRUTH_DRAWS)


def compute_truth(scm: SCM, domain_config: dict) -> Truth:
    outcome = domain_config["effect_estimation"]["outcome"]
    discovery_vars = set(domain_config["causal_discovery"]["variables"])
    observed = set(scm.observed)

    edges = [e for e in scm.edges() if set(e) <= discovery_vars] if scm.graph_known else None
    effects = {
        t["name"]: _true_effect(scm, t["name"], outcome, 1.0)
        for t in domain_config["effect_estimation"]["treatments"]
        if t["name"] in observed
    }
    roi: dict[str, float] = {}
    for cand in domain_config["interventions"]["candidates"]:
        if cand["target_variable"] not in observed or not cand["cost"]:
            continue
        # Stage 6 treats the outcome as something to reduce: benefit = -effect.
        effect = _true_effect(scm, cand["target_variable"], outcome, cand["expected_shift"])
        roi[cand["id"]] = -effect / cand["cost"]
    return Truth(
        edges=edges,
        effects=effects,
        roi=roi,
        subgroup_effects=_true_subgroup_effects(scm, domain_config),
        fairness_ratio=_true_fairness_ratio(scm, domain_config),
    )


def _true_fairness_ratio(scm: SCM, domain_config: dict) -> Optional[float]:
    """Lowest over highest group outcome rate across the sensitive attribute, from a large
    draw of the SCM: the same min/max statistic Stage 6 computes (fairlearn's
    `ratio(between_groups)`), on the population instead of a sample."""
    sensitive = domain_config["interventions"]["sensitive_attribute"]
    outcome = domain_config["effect_estimation"]["outcome"]
    if sensitive not in scm.observed or outcome not in scm.observed:
        return None
    rates = scm.sample(TRUTH_DRAWS, 0).groupby(sensitive)[outcome].mean()
    return float(rates.min() / rates.max()) if len(rates) > 1 and rates.max() > 0 else None


def _true_subgroup_effects(scm: SCM, domain_config: dict) -> dict[str, float]:
    cf = domain_config.get("counterfactuals", {})
    columns = list(cf.get("subgroups", []))
    treatment, outcome = cf.get("treatment"), cf.get("outcome")
    if not columns or treatment not in scm.observed or not scm.is_binary(treatment):
        return {}
    truth: dict[str, float] = {}
    for column in columns:
        for level, effect in scm.true_subgroup_effects(treatment, outcome, column, n=TRUTH_DRAWS).items():
            truth[f"{column}={level}"] = effect
    return truth


@dataclass
class RunResult:
    """One pipeline run on one dataset draw."""

    label: str  # "committed file" or "seed 1003"
    n_rows: int
    values: dict[str, Optional[float]] = field(default_factory=dict)  # metric -> value
    effect_rows: list[metrics.EffectRow] = field(default_factory=list)
    ranking: Optional[metrics.RankingScore] = None
    predicted_edges: list[Edge] = field(default_factory=list)
    # What the pipeline estimated, kept even with no ground truth (real data) so it can be
    # compared with an experimental reference: treatment -> Stage 4 ATE / placebo p-value,
    # and Stage 5's mean CATE per counterfactual treatment.
    estimated_ates: dict[str, float] = field(default_factory=dict)
    refutation_p_values: dict[str, Optional[float]] = field(default_factory=dict)
    refutation_passed: dict[str, Optional[bool]] = field(default_factory=dict)
    mean_cates: dict[str, float] = field(default_factory=dict)
    # Stage 6, kept even with no ground truth: the fairness statistic and whether it passed the
    # threshold, and the first-ranked intervention
    fairness_ratio: Optional[float] = None
    fairness_passed: Optional[bool] = None
    top_choice: Optional[str] = None
    # Stage 5 spread of per-unit effects, per counterfactual treatment
    cate_stds: dict[str, Optional[float]] = field(default_factory=dict)
    # Stage 5 mean effect within each subgroup, "male=1" -> value (per counterfactual treatment
    # there is one set; the harness only uses the first counterfactual's)
    subgroup_cates: dict[str, float] = field(default_factory=dict)
    subgroup_rows: list[metrics.EffectRow] = field(default_factory=list)  # vs true subgroup effects
    baselines: dict[str, baseline_methods.BaselineRun] = field(default_factory=dict)  # only with --baselines
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    domain_id: str
    dataset: str
    source: str  # "committed file" | "SCM replicates"
    scored_against_truth: bool
    runs: list[RunResult] = field(default_factory=list)
    skipped_reason: Optional[str] = None
    truth: Optional[Truth] = None
    # Set only for randomized real data (benchmarks.RCT_DATASETS): the trial's own estimate
    reference: list[benchmarks.ExperimentalReference] = field(default_factory=list)
    subgroup_reference: list[benchmarks.ExperimentalReference] = field(default_factory=list)
    # Set only for stress scenarios (plain data, so the result serialises cleanly):
    stress_id: Optional[str] = None
    stress_factor: Optional[str] = None
    stress_description: Optional[str] = None


def score_run(
    outputs: runner.AnalysisOutputs,
    domain_config: dict,
    truth: Optional[Truth],
    with_baselines: bool = False,
) -> RunResult:
    result = RunResult(label="", n_rows=len(outputs.raw))
    result.predicted_edges = list(outputs.graph.edges)
    result.values["refutation_pass_rate"] = metrics.refutation_pass_rate(outputs.effects)
    result.estimated_ates = {e.treatment: e.ate for e in outputs.effects}
    result.refutation_p_values = {e.treatment: e.refutation_p_value for e in outputs.effects}
    result.refutation_passed = {e.treatment: e.refutation_passed for e in outputs.effects}
    result.mean_cates = {c.treatment: c.mean_cate for c in outputs.counterfactuals}
    result.cate_stds = {c.treatment: c.cate_std for c in outputs.counterfactuals}
    if outputs.counterfactuals:
        result.subgroup_cates = {s.key: s.mean_cate for s in outputs.counterfactuals[0].subgroups}
    if outputs.recommendations:
        top = outputs.recommendations[0]
        result.fairness_ratio, result.fairness_passed, result.top_choice = top.fairness_ratio, top.fairness_pass, top.id
    if truth is None:
        return result

    if truth.edges is not None:
        priors = [tuple(e) for e in domain_config["causal_discovery"].get("required_edges", [])]
        g = metrics.score_graph(outputs.graph.edges, truth.edges, priors)
        result.values.update(
            edge_precision=g.precision,
            edge_recall=g.recall,
            edge_f1=g.f1,
            shd=float(g.shd),
            discovered_precision=g.discovered_precision,
            discovered_recall=g.discovered_recall,
        )

    if truth.fairness_ratio is not None and result.fairness_ratio is not None:
        threshold = domain_config["interventions"]["fairness_threshold"]
        result.values["fairness_ratio_abs_err"] = abs(result.fairness_ratio - truth.fairness_ratio)
        # Does the sample's pass/flag verdict match what the population's ratio says?
        result.values["fairness_flag_correct"] = float((result.fairness_ratio >= threshold) == (truth.fairness_ratio >= threshold))

    result.effect_rows = metrics.score_effects(outputs.effects, truth.effects)
    if result.effect_rows:
        errors = [r.abs_error for r in result.effect_rows]
        result.values["ate_mae"] = float(np.mean(errors))
        result.values["ate_max_abs_err"] = float(np.max(errors))
        result.values["ate_sign_agreement"] = float(np.mean([r.sign_agrees for r in result.effect_rows]))

    if truth.subgroup_effects:
        result.subgroup_rows = [
            metrics.EffectRow(treatment=key, estimated=result.subgroup_cates[key], true=true)
            for key, true in truth.subgroup_effects.items()
            if key in result.subgroup_cates
        ]
        if result.subgroup_rows:
            result.values["subgroup_cate_mae"] = float(np.mean([r.abs_error for r in result.subgroup_rows]))

    # With a single candidate there is nothing to rank: "top-1 correct" would be true by
    # construction and would read as a success, so it is left undefined instead.
    result.ranking = metrics.score_ranking(outputs.recommendations, truth.roi) if len(truth.roi) >= 2 else None
    if result.ranking is not None:
        result.values["top1_correct"] = float(result.ranking.top1_correct)
        result.values["rank_tau"] = result.ranking.kendall_tau
        result.values["roi_regret"] = result.ranking.roi_regret

    if with_baselines:
        result.baselines = baseline_methods.score_baselines(
            outputs.raw, outputs.features, domain_config, truth.effects, truth.roi
        )
    return result


def _run_one(
    label: str,
    domain_config: dict,
    data_path: Path,
    truth: Optional[Truth],
    with_baselines: bool = False,
) -> RunResult:
    try:
        # Feast prints progress lines to stdout; keep stdout for the report.
        with contextlib.redirect_stdout(sys.stderr):
            outputs = runner.run_analysis_stages(domain_config, data_path)
        result = score_run(outputs, domain_config, truth, with_baselines)
    except Exception as exc:  # one bad dataset must not sink the whole table
        return RunResult(label=label, n_rows=0, error=f"{type(exc).__name__}: {exc}")
    result.label = label
    return result


def evaluate_dataset(
    domain_id: str,
    domain_config: dict,
    dataset: str,
    replicates: int = 0,
    n_rows: Optional[int] = None,
) -> list[ScenarioResult]:
    """Scenarios for one (domain, dataset): the committed file, plus SCM
    replicates if the dataset has a known SCM and `replicates` > 0. Replicates are
    `n_rows` long, by default the same length as the committed file (so each dataset
    is re-drawn at its own size), or DEFAULT_N_ROWS if that file is absent."""
    cfg = runner.with_dataset(domain_config, EVAL_DATASET_PREFIX + dataset)
    factory = SCM_REGISTRY.get((domain_id, dataset))
    scm = factory() if factory else None
    truth = compute_truth(scm, domain_config) if scm else None
    scenarios: list[ScenarioResult] = []

    path = runner.resolve_data_path(domain_config, dataset)
    committed = ScenarioResult(domain_id, dataset, "committed file", scm is not None, truth=truth)
    if path.exists():
        committed.runs.append(_run_one("committed file", cfg, path, truth))
        if (domain_id, dataset) in benchmarks.RCT_DATASETS:
            committed.reference = benchmarks.experimental_reference(path, domain_config)
            committed.subgroup_reference = benchmarks.experimental_subgroup_references(path, domain_config)
    else:
        committed.skipped_reason = f"data file not found: {path}"
    scenarios.append(committed)

    if scm is not None and replicates > 0:
        if n_rows is None:
            ran = [r for r in committed.runs if r.error is None]
            n_rows = ran[0].n_rows if ran else DEFAULT_N_ROWS
        reps = ScenarioResult(domain_id, dataset, "SCM replicates", True, truth=truth)
        reps.runs = _run_replicates(scm, cfg, domain_config, n_rows, replicates, truth)
        scenarios.append(reps)
    elif scm is None and replicates > 0 and path.exists() and dataset != MULTIMODAL:
        # No SCM to redraw from, so the only replicates available are bootstrap resamples of the
        # file itself. They measure STABILITY (do the estimates and the fairness verdict survive
        # resampling the applicants?), not accuracy: there is no truth to be accurate against.
        boot = ScenarioResult(domain_id, dataset, BOOTSTRAP_SOURCE, False)
        boot.runs = _run_bootstrap(cfg, domain_config, path, replicates)
        scenarios.append(boot)
    return scenarios


def _run_bootstrap(cfg: dict, domain_config: dict, data_path: Path, replicates: int) -> list[RunResult]:
    """Pipeline runs on `replicates` bootstrap resamples (rows drawn with replacement, same
    size) of a real data file. Replicate i uses seed REPLICATE_SEED_BASE + i."""
    real = pd.read_csv(data_path)
    id_col = domain_config["ingestion"]["id_column"]
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(replicates):
            seed = REPLICATE_SEED_BASE + i
            rows = np.random.default_rng(seed).integers(0, len(real), len(real))
            resample = real.iloc[rows].reset_index(drop=True)
            resample[id_col] = range(1, len(resample) + 1)  # ids must stay unique
            csv_path = Path(tmp) / f"bootstrap_{seed}.csv"
            resample.to_csv(csv_path, index=False)
            runs.append(_run_one(f"bootstrap {seed}", cfg, csv_path, None))
    return runs


def _run_replicates(
    scm: SCM,
    cfg: dict,
    domain_config: dict,
    n_rows: int,
    replicates: int,
    truth: Truth,
    with_baselines: bool = False,
) -> list[RunResult]:
    """Pipeline runs on `replicates` fresh draws of `n_rows` from `scm`. Replicate i
    always uses seed REPLICATE_SEED_BASE + i, so scenarios are compared on the same seeds."""
    id_col = domain_config["ingestion"]["id_column"]
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(replicates):
            seed = REPLICATE_SEED_BASE + i
            df = scm.sample(n_rows, seed)
            df.insert(0, id_col, range(1, n_rows + 1))
            csv_path = Path(tmp) / f"draw_{seed}.csv"
            df.to_csv(csv_path, index=False)
            runs.append(_run_one(f"seed {seed}", cfg, csv_path, truth, with_baselines))
    return runs


def with_algorithm(domain_config: dict, algorithm: Optional[str]) -> dict:
    """Copy of the config with Stage 3 forced to `algorithm` (None leaves it as configured)."""
    if algorithm is None:
        return domain_config
    return {**domain_config, "causal_discovery": {**domain_config["causal_discovery"], "algorithm": algorithm}}


def evaluate_stress(
    domain_config: dict,
    scenario: StressScenario,
    replicates: int,
    with_baselines: bool = False,
    algorithm: Optional[str] = None,
) -> ScenarioResult:
    """Run one stress scenario: `replicates` fresh draws from its SCM, through the
    domain config as the scenario rewrites it, scored against that SCM's truth."""
    scenario_config = with_algorithm(scenario.configure(domain_config), algorithm)
    truth = compute_truth(scenario.scm(), scenario_config)
    cfg = runner.with_dataset(scenario_config, f"{EVAL_DATASET_PREFIX}stress_{scenario.id}")
    result = ScenarioResult(
        scenario.domain_id,
        f"stress:{scenario.id}",
        "SCM replicates",
        True,
        truth=truth,
        stress_id=scenario.id,
        stress_factor=scenario.factor,
        stress_description=scenario.description,
    )
    result.runs = _run_replicates(
        scenario.scm(), cfg, scenario_config, scenario.n_rows, replicates, truth, with_baselines
    )
    return result


def _stress_worker(scenario_id: str, replicates: int, with_baselines: bool, algorithm: Optional[str]) -> ScenarioResult:
    """Runs one scenario in a worker process. Takes the scenario's id, not the scenario:
    scenarios hold lambdas, which cannot be pickled across a process boundary."""
    (scenario,) = select_stress([scenario_id])
    domain_config = ConfigLoader().get_domain(scenario.domain_id).extra
    return evaluate_stress(domain_config, scenario, replicates, with_baselines, algorithm)


def run_stress(
    replicates: int,
    scenario_ids: Optional[Iterable[str]] = None,
    loader: Optional[ConfigLoader] = None,
    with_baselines: bool = False,
    workers: int = 1,
    algorithm: Optional[str] = None,
) -> list[ScenarioResult]:
    """Every stress scenario (or the named ones), in registry order. `algorithm` forces
    Stage 3's algorithm (pc / ges / lingam) instead of the one the config names.

    `workers` > 1 runs scenarios in parallel processes. That is safe because each
    scenario keeps its Feast state in its own directory (`eval_stress_<id>`), and
    the results are identical to a serial run (seeds are fixed per replicate). A
    custom `loader` cannot be shipped to workers, so it requires workers=1."""
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    scenarios = select_stress(list(scenario_ids) if scenario_ids else None)

    if workers > 1 and len(scenarios) > 1:
        if loader is not None:
            raise ValueError("a custom loader cannot be used with workers > 1")
        ids = [s.id for s in scenarios]
        # "spawn" on every platform: the Linux default (fork) deadlocked once BLAS/OpenMP threads existed
        # (seen in the Docker test run); spawn is also what Windows always does.
        with ProcessPoolExecutor(max_workers=min(workers, len(ids)), mp_context=multiprocessing.get_context("spawn")) as pool:
            return list(
                pool.map(_stress_worker, ids, [replicates] * len(ids), [with_baselines] * len(ids), [algorithm] * len(ids))
            )

    loader = loader or ConfigLoader()
    return [
        evaluate_stress(loader.get_domain(s.domain_id).extra, s, replicates, with_baselines, algorithm)
        for s in scenarios
    ]


def run_evaluation(
    domains: Optional[Iterable[str]] = None,
    datasets: Optional[Iterable[str]] = None,
    replicates: int = 0,
    n_rows: Optional[int] = None,
    loader: Optional[ConfigLoader] = None,
    algorithm: Optional[str] = None,
) -> list[ScenarioResult]:
    """Every runnable domain x every dataset kind its config lists (optionally filtered).
    `algorithm` forces Stage 3's algorithm (pc / ges / lingam) instead of the configured one."""
    loader = loader or ConfigLoader()
    wanted_domains = set(domains) if domains else None
    wanted_datasets = set(datasets) if datasets else None

    results: list[ScenarioResult] = []
    for domain in loader.list_domains(runnable_only=True):
        if wanted_domains and domain.id not in wanted_domains:
            continue
        for dataset in runner.available_datasets(domain.extra):
            if wanted_datasets and dataset not in wanted_datasets:
                continue
            if dataset == MULTIMODAL and not (wanted_datasets and MULTIMODAL in wanted_datasets):
                continue  # slow (CNN embeddings) and needs generated attachments: ask for it by name
            results.extend(
                evaluate_dataset(domain.id, with_algorithm(domain.extra, algorithm), dataset, replicates, n_rows)
            )
    return results


def summarize(runs: list[RunResult]) -> dict[str, tuple[float, float, int]]:
    """metric -> (mean, sample sd, number of runs with a defined value); failed
    runs and undefined values are excluded, and sd is 0 for a single run."""
    out: dict[str, tuple[float, float, int]] = {}
    keys = {k for r in runs if r.error is None for k in r.values}
    for key in keys:
        vals = [r.values[key] for r in runs if r.error is None and r.values.get(key) is not None]
        if vals:
            out[key] = (
                float(np.mean(vals)),
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                len(vals),
            )
    return out
