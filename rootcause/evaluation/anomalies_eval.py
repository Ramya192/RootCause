"""Does causal anomaly flagging find what ordinary outlier detectors miss?

On simulated data with a known graph, a few percent of records get ONE variable corrupted, in
one of two ways:

  swap   the variable takes the value of a random other record. Every value is still common,
         so no single-variable check can see it; it is a fault only because the value does not
         fit the record's causes (for a binary variable, the value is flipped).
  shift  the variable is pushed 3 standard deviations away (flipped, if binary). A plain outlier.

Each detector then ranks all records and is scored against which ones were corrupted:

  causal (true graph)   mechanism surprise given the TRUE parents (best case for the method)
  causal (PC graph)     the same, with the graph Stage 3 discovers from the contaminated data
  Mahalanobis           distance under the joint linear-Gaussian model (a strong baseline: it
                        sees broken correlations, but has no notion of direction or of binary
                        variables)
  isolation forest      a standard general-purpose outlier detector
  marginal z-score      the largest |z| over the variables (single-variable check)

Nothing here says how well it finds real-world anomalies; there are no labelled ones. It shows
what it can and cannot see when the faults are of a known kind.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from rootcause.evaluation.scm import SCM
from rootcause.evaluation.scms import SCM_REGISTRY, attrition_scm
from rootcause.models.schemas import CausalGraph
from rootcause.pipeline import anomalies, causal_discovery
from rootcause.utils.config_loader import ConfigLoader

SEED0 = 1000
INJECTION_SEED0 = 5000
RATE = 0.03
SHIFT_SD = 3.0
FAULTS = ("swap", "shift")
# "roots scored" is the true-graph detector that also sums the surprise of variables with no parents (the
# module's score_roots=True). The default does not: a root has no mechanism to break.
METHODS = (
    "causal (true graph)",
    "causal (PC graph)",
    "causal (true graph, roots scored)",
    "Mahalanobis",
    "isolation forest",
    "marginal z-score",
)
TARGET_KINDS = ("continuous", "binary", "root")


@dataclass(frozen=True)
class Scenario:
    id: str
    domain_id: str
    scm: Callable[[], SCM]
    n_rows: int


SCENARIOS = (
    Scenario("attrition", "employee_attrition", attrition_scm, 2000),
    Scenario("illinois_mirror", "illinois_wellness", SCM_REGISTRY[("illinois_wellness", "synthetic")], 4834),
)


@dataclass
class Row:
    scenario: str
    fault: str
    target_kind: str  # "continuous" or "binary": the kind of variable that was corrupted
    method: str
    seed: int
    roc_auc: float
    average_precision: float
    precision_at_k: float
    top1_attribution: float | None = None  # causal methods only: the top variable is the corrupted one
    family_attribution: float | None = None  # ... or one of its children (a fault spreads to them)


def _is_binary(series: pd.Series) -> bool:
    return set(pd.unique(series)) <= {0, 1}


def inject(
    df: pd.DataFrame, targets: list[str], fault: str, rate: float, seed: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(corrupted copy, boolean mask of corrupted rows, the variable corrupted in each row)."""
    rng = np.random.default_rng(seed)
    clean = df.reset_index(drop=True)  # shift sizes and swap donors come from the clean data,
    out = clean.copy()  # not from rows already corrupted in this call
    n_inj = max(1, math.ceil(rate * len(out)))
    rows = rng.choice(len(out), size=n_inj, replace=False)
    mask = pd.Series(False, index=out.index)
    victim = pd.Series("", index=out.index, dtype=object)
    for row in rows:
        variable = targets[rng.integers(len(targets))]
        column = clean[variable]
        if fault == "shift":
            new = 1 - column.iloc[row] if _is_binary(column) else column.iloc[row] + SHIFT_SD * column.std()
        else:
            donors = np.flatnonzero(column.to_numpy() != column.iloc[row])
            new = column.iloc[rng.choice(donors)]
        out.iloc[row, out.columns.get_loc(variable)] = new
        mask.iloc[row] = True
        victim.iloc[row] = variable
    return out, mask, victim


def _scores(name: str, data: pd.DataFrame, graphs: dict[str, CausalGraph]) -> tuple[np.ndarray, anomalies.AnomalyReport | None]:
    if name.startswith("causal"):
        roots_scored = "roots scored" in name
        report = anomalies.flag_anomalies(
            data, graphs["causal (true graph)" if roots_scored else name], score_roots=roots_scored
        )
        return report.scores.to_numpy(), report
    values = data.to_numpy(dtype=float)
    if name == "Mahalanobis":
        centred = values - values.mean(axis=0)
        precision = np.linalg.pinv(np.cov(values, rowvar=False))
        return np.einsum("ij,jk,ik->i", centred, precision, centred), None
    if name == "isolation forest":
        forest = IsolationForest(n_estimators=200, random_state=0).fit(values)
        return -forest.score_samples(values), None
    z = np.abs((values - values.mean(axis=0)) / np.where(values.std(axis=0) > 0, values.std(axis=0), 1.0))
    return z.max(axis=1), None


def evaluate_draw(scenario: Scenario, cfg: dict, seed: int) -> list[Row]:
    scm = scenario.scm()
    variables = list(cfg["causal_discovery"]["variables"])
    df = scm.sample(scenario.n_rows, seed)
    df.insert(0, cfg["ingestion"]["id_column"], range(1, len(df) + 1))
    true_edges = [e for e in scm.edges() if set(e) <= set(variables)]
    true_graph = CausalGraph(nodes=variables, edges=true_edges, algorithm="truth")
    children = {v: {c for p, c in true_edges if p == v} for v in variables}
    non_roots = [v for v in variables if any(c == v for _, c in true_edges)]  # these have causes to violate
    roots = [v for v in variables if v not in non_roots and any(p == v for p, _ in true_edges)]
    targets_by_kind = {
        "continuous": [v for v in non_roots if not _is_binary(df[v])],
        "binary": [v for v in non_roots if _is_binary(df[v])],
        "root": roots,  # a fault in a root has no mechanism of its own to break; only its children show it
    }

    rows: list[Row] = []
    for fault, (kind, targets) in ((f, kt) for f in FAULTS for kt in targets_by_kind.items()):
        if not targets:
            continue
        corrupted, mask, victim = inject(df, targets, fault, RATE, INJECTION_SEED0 + (seed - SEED0))
        data = corrupted[variables]
        pc_graph = causal_discovery.discover_graph(corrupted, cfg)
        graphs = {"causal (true graph)": true_graph, "causal (PC graph)": pc_graph}
        k = int(mask.sum())
        for method in METHODS:
            score, report = _scores(method, data, graphs)
            order = np.argsort(-score)[:k]
            row = Row(
                scenario=scenario.id, fault=fault, target_kind=kind, method=method, seed=seed,
                roc_auc=float(roc_auc_score(mask, score)),
                average_precision=float(average_precision_score(mask, score)),
                precision_at_k=float(mask.to_numpy()[order].mean()),
            )
            if report is not None:
                top = report.top_variable.reset_index(drop=True)[mask]
                truth = victim[mask]
                row.top1_attribution = float((top == truth).mean())
                row.family_attribution = float(np.mean([t == v or t in children[v] for t, v in zip(top, truth)]))
            rows.append(row)
    return rows


def run(replicates: int = 10) -> dict:
    loader = ConfigLoader()
    rows: list[Row] = []
    for scenario in SCENARIOS:
        cfg = loader.get_domain(scenario.domain_id).extra
        for i in range(replicates):
            rows.extend(evaluate_draw(scenario, cfg, SEED0 + i))
    return {"settings": {"replicates": replicates, "rate": RATE, "shift_sd": SHIFT_SD}, "rows": rows}


# --- report -------------------------------------------------------------------------------


def _ms(values) -> str:
    arr = np.asarray([v for v in values if v is not None and not pd.isna(v)], dtype=float)
    if len(arr) == 0:
        return "–"
    return f"{arr.mean():.2f} ± {arr.std(ddof=1):.2f}" if len(arr) > 1 else f"{arr.mean():.2f}"


def render_markdown(result: dict) -> str:
    df = pd.DataFrame([asdict(r) for r in result["rows"]])
    s = result["settings"]
    lines = [
        "# Causal anomaly flagging check (injected mechanism faults)",
        "",
        f"Generated by `python -m rootcause.evaluation --anomalies` ({s['replicates']} draws per scenario; {s['rate']:.0%} of records "
        "get one variable corrupted; mean ± sd across draws). AUC is the chance a corrupted record is scored above a clean one "
        "(0.5 = chance); P@k is the share of the k top-ranked records that are corrupted, with k the number corrupted.",
        "",
    ]
    for scenario in SCENARIOS:
        for fault, kind in ((f, k) for f in FAULTS for k in TARGET_KINDS):
            sub = df[(df.scenario == scenario.id) & (df.fault == fault) & (df.target_kind == kind)]
            if sub.empty:
                continue
            what = "value swapped with another record's" if fault == "swap" else f"pushed {SHIFT_SD:g} sd away"
            if kind == "binary":
                what = "value taken from a record with the other value" if fault == "swap" else "value flipped"
            if kind == "root":
                what += "; the variable has no parents in the graph, so only its children can show it"
            lines += [
                f"## {scenario.id}: {fault} in a {kind} variable ({what})",
                "",
                "| Detector | ROC AUC | Avg precision | P@k | Top variable = the corrupted one | ... or one of its children |",
                "|---|---|---|---|---|---|",
            ]
            for method in METHODS:
                m = sub[sub.method == method]
                lines.append(
                    f"| {method} | {_ms(m.roc_auc)} | {_ms(m.average_precision)} | {_ms(m.precision_at_k)} | "
                    f"{_ms(m.top1_attribution)} | {_ms(m.family_attribution)} |"
                )
            lines.append("")
    lines += [
        "## How to read this",
        "",
        "- **Binary variables are hard for every detector.** Flipping a binary outcome to a value that was plausible for that record leaves little to see: its surprise is a fraction of a nat when the flipped value had a 30% chance anyway. That is why the binary sections sit far below the continuous ones, and why one blended number would hide the difference.",
        "- **swap** is the case single-variable checks cannot see: the planted value is common in the data, wrong only for that record. **shift** is a plain outlier any detector should find; it is there to show the causal method does not lose on the easy case.",
        "- **Mahalanobis** is the fair competitor: it also sees broken correlations. Where it matches the causal detector (linear, roughly Gaussian data), causal flagging adds attribution (which variable) but not detection. Its advantage over Mahalanobis, if any, would have to come from non-Gaussian, binary or nonlinear mechanisms; none of these scenarios shows one.",
        "- **Attribution** is only scored for the causal detectors. A fault in one variable also makes its children look surprising (their parent changed), so the top variable is often a child; the last column counts that as a hit. It is a lead for an investigator, not a root-cause proof.",
        "- The PC graph is discovered from the contaminated data, as it would be in use. Priors in the domain config (required and forbidden edges) apply.",
        "- **Roots are not scored by default** (a root has no mechanism to break; its 'surprise' is only rarity, which a marginal check already measures). That helps on faults in non-root variables, which is most of the sections above, partly by construction: it drops noise from variables that were never corrupted. The `root` sections are the cost: a corrupted root is visible only through its children, and 'roots scored' shows what scoring them buys back.",
        "- No labelled real anomalies exist for any dataset here, so nothing on this page is a claim about detecting real fraud or attrition anomalies.",
    ]
    return "\n".join(lines) + "\n"


def render_json(result: dict) -> str:
    return json.dumps({"settings": result["settings"], "rows": [asdict(r) for r in result["rows"]]}, indent=2)


# --- real covariates: injected faults, and what the top of the list looks like ------------------
#
# The synthetic scenarios above have only continuous and binary variables. Real data has
# categorical and ordinal columns too (a deductible, an age band), and modelling those as
# Gaussian made the top of the carclaims list one rare deductible level. These runs use the REAL
# covariates of two domains, corrupt a few percent of records as above, and compare the
# type-aware mechanisms with the earlier Gaussian treatment. There is still no labelled real
# anomaly: only the injected faults are known. The graph is what Stage 3 finds on the real data
# (there is no true graph), learned once on the uncorrupted data.

REAL_SCENARIOS = (("carclaims", "real"), ("german_credit", "real"), ("freddie_mac", "real_2007"))
# "type-aware" is the module default (categorical mechanisms, roots not scored); "earlier" is what it did
# before: every non-binary column Gaussian, every variable scored.
REAL_METHODS = ("causal (type-aware)", "causal (earlier)", "Mahalanobis", "isolation forest", "marginal z-score")
CONCENTRATION_K = 100


def _real_setup(domain_id: str, dataset: str) -> tuple[dict, pd.DataFrame, CausalGraph]:
    from rootcause.pipeline import ingestion, preprocessing, runner

    cfg = runner.with_dataset(ConfigLoader().get_domain(domain_id).extra, dataset)
    raw = ingestion.ingest(runner.resolve_data_path(cfg, dataset), cfg)
    features = preprocessing.preprocess(raw, cfg).df
    return cfg, features, causal_discovery.discover_graph(features, cfg)


def _real_scores(method: str, data: pd.DataFrame, graph: CausalGraph):
    if method.startswith("causal"):
        earlier = method == "causal (earlier)"
        report = anomalies.flag_anomalies(
            data, graph, max_levels=None if earlier else anomalies.MAX_LEVELS, score_roots=earlier
        )
        return report.scores.to_numpy(), report
    return _scores(method, data, {})


def concentration(report: anomalies.AnomalyReport, k: int = CONCENTRATION_K) -> dict:
    """How much of the top `k` is one thing: the share of the top-k records whose largest surprise is
    on the single most common variable, and the share sharing that variable AND its observed value.
    A list dominated by one (variable, value) is a model artefact, not a set of distinct anomalies."""
    top = report.scores.sort_values(ascending=False).head(k).index
    variable = report.top_variable.loc[top]
    lead = variable.value_counts().index[0]
    same_variable = variable == lead
    values = pd.Series([report.data.loc[r, lead] for r in top], index=top)
    lead_value = values[same_variable].value_counts().index[0]
    return {
        "k": k,
        "lead_variable": str(lead),
        "lead_variable_share": float(same_variable.mean()),
        "lead_value": float(lead_value),
        "lead_value_share": float((same_variable & (values == lead_value)).mean()),
        "lead_value_frequency": float((report.data[lead] == lead_value).mean()),
    }


def evaluate_real_draw(domain_id: str, cfg: dict, features: pd.DataFrame, graph: CausalGraph, seed: int) -> list[Row]:
    variables = list(graph.nodes)
    base = features[variables].reset_index(drop=True)
    children = {v: {c for p, c in graph.edges if p == v} for v in variables}
    non_roots = [v for v in variables if any(c == v for _, c in graph.edges)]
    targets_by_kind: dict[str, list[str]] = {"continuous": [], "binary": [], "categorical": []}
    for v in non_roots:
        targets_by_kind[anomalies.variable_kind(base[v])].append(v)

    rows: list[Row] = []
    for fault in FAULTS:
        for kind, targets in targets_by_kind.items():
            if not targets or (fault == "shift" and kind == "categorical"):
                continue  # +3 sd of an ordinal code is a level that does not exist: trivially found, so not a test
            corrupted, mask, victim = inject(base, targets, fault, RATE, INJECTION_SEED0 + seed)
            k = int(mask.sum())
            for method in REAL_METHODS:
                score, report = _real_scores(method, corrupted, graph)
                order = np.argsort(-score)[:k]
                row = Row(
                    scenario=f"{domain_id} (real covariates)", fault=fault, target_kind=kind, method=method, seed=seed,
                    roc_auc=float(roc_auc_score(mask, score)),
                    average_precision=float(average_precision_score(mask, score)),
                    precision_at_k=float(mask.to_numpy()[order].mean()),
                )
                if report is not None:
                    top = report.top_variable.reset_index(drop=True)[mask]
                    truth = victim[mask]
                    row.top1_attribution = float((top == truth).mean())
                    row.family_attribution = float(np.mean([t == v or t in children[v] for t, v in zip(top, truth)]))
                rows.append(row)
    return rows


def run_real(replicates: int = 5, scenarios=REAL_SCENARIOS) -> dict:
    rows: list[Row] = []
    conc: list[dict] = []
    graphs: dict[str, dict] = {}
    for domain_id, dataset in scenarios:
        cfg, features, graph = _real_setup(domain_id, dataset)
        graphs[domain_id] = {"edges": len(graph.edges), "nodes": len(graph.nodes), "records": len(features)}
        data = features[list(graph.nodes)]
        for method in ("causal (type-aware)", "causal (earlier)"):
            _, report = _real_scores(method, data, graph)
            conc.append({"scenario": f"{domain_id} (real covariates)", "method": method, **concentration(report)})
        for i in range(replicates):
            rows.extend(evaluate_real_draw(domain_id, cfg, features, graph, i))
    return {
        "settings": {"replicates": replicates, "rate": RATE, "shift_sd": SHIFT_SD, "max_levels": anomalies.MAX_LEVELS},
        "rows": rows,
        "concentration": conc,
        "graphs": graphs,
    }


def render_real_markdown(result: dict) -> str:
    df = pd.DataFrame([asdict(r) for r in result["rows"]])
    s = result["settings"]
    lines = [
        "# Causal anomaly flagging on real covariates (type-aware mechanisms)",
        "",
        f"Generated by `python -m rootcause.evaluation --anomalies-real` ({s['replicates']} draws per domain; {s['rate']:.0%} of "
        "records get one variable corrupted; mean ± sd across draws). Faults are injected into the REAL covariates of each "
        "domain, and the graph is the one Stage 3 finds on the uncorrupted real data (there is no true graph). "
        f"**type-aware** models a non-binary column with at most {s['max_levels']} distinct values as categorical "
        "(multinomial logistic) and scores only variables that have parents in the graph; **earlier** treated every "
        "non-binary column as Gaussian and scored every variable, roots included. "
        "No labelled real anomalies exist: only the injected faults are known.",
        "",
        "## What the top of the list looks like (no injection)",
        "",
        f"The {CONCENTRATION_K} records the model explains least. A list dominated by one (variable, value) is a model "
        "artefact, not a set of distinct anomalies. `lead value frequency` is how common that value is in the whole data.",
        "",
        "| Domain | Mechanisms | Records on the lead variable | ... with the lead value | Lead variable = value | Lead value's frequency in the data |",
        "|---|---|---|---|---|---|",
    ]
    for c in result["concentration"]:
        lines.append(
            f"| {c['scenario']} | {c['method'].replace('causal ', '')} | {c['lead_variable_share']:.0%} | "
            f"{c['lead_value_share']:.0%} | `{c['lead_variable']}` = {c['lead_value']:g} | {c['lead_value_frequency']:.1%} |"
        )
    lines.append("")
    for scenario in dict.fromkeys(df.scenario):
        for fault, kind in ((f, k) for f in FAULTS for k in ("continuous", "binary", "categorical")):
            sub = df[(df.scenario == scenario) & (df.fault == fault) & (df.target_kind == kind)]
            if sub.empty:
                continue
            what = {"swap": "value swapped with another record's", "shift": f"pushed {SHIFT_SD:g} sd away"}[fault]
            if kind == "binary":
                what = "value taken from a record with the other value" if fault == "swap" else "value flipped"
            lines += [
                f"## {scenario}: {fault} in a {kind} variable ({what})",
                "",
                "| Detector | ROC AUC | Avg precision | P@k | Top variable = the corrupted one | ... or one of its children |",
                "|---|---|---|---|---|---|",
            ]
            for method in REAL_METHODS:
                m = sub[sub.method == method]
                lines.append(
                    f"| {method} | {_ms(m.roc_auc)} | {_ms(m.average_precision)} | {_ms(m.precision_at_k)} | "
                    f"{_ms(m.top1_attribution)} | {_ms(m.family_attribution)} |"
                )
            lines.append("")
    lines += [
        "## How to read this",
        "",
        "- The injected faults are the only ground truth; they say what the detectors can see when a fault is of a known kind, not how many real anomalies a domain has.",
        "- A categorical **shift** is not run: adding 3 sd to an ordinal code produces a level that does not exist, which any detector finds trivially. A categorical **swap** (another record's level) is the meaningful test.",
        "- Roots have no causes, so faults are injected into non-root variables only, and type-aware does not score roots (a rare level of a root is surprising by definition, which is what filled the top of the earlier list). A fault in a root would show up only through its children.",
        "- The graph is learned once, on the uncorrupted data, unlike the synthetic scenarios where it is learned from the corrupted data.",
    ]
    return "\n".join(lines) + "\n"


def render_real_json(result: dict) -> str:
    return json.dumps(
        {"settings": result["settings"], "graphs": result["graphs"], "concentration": result["concentration"],
         "rows": [asdict(r) for r in result["rows"]]},
        indent=2,
    )
