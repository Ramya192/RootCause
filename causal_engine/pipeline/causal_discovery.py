"""Stage 3: Causal Discovery.

Learns a directed graph over the domain's configured variables with one of three
algorithms (`causal_discovery.algorithm`; anything else is an error):

  pc      causal-learn's PC: conditional-independence tests. Domain priors go in as
          BackgroundKnowledge, so it can use them while orienting edges.
  ges     causal-learn's GES: greedy search over equivalence classes with a BIC score.
          It has no background-knowledge input, so priors are applied AFTER the
          search: a forbidden edge is removed, and an edge GES leaves undirected is
          oriented when the priors forbid exactly one direction.
  lingam  DirectLiNGAM (the `lingam` package): identifies direction from
          non-Gaussian noise instead of conditional independence, so it can orient
          edges PC leaves undirected -- but ONLY if the noise really is non-Gaussian
          and the relations linear, which discrete columns and many real data break.
          Forbidden edges enter as prior knowledge ("no directed path"); weights
          below `lingam_threshold` (default 0.05, on standardised data) are dropped.

Any edge left unoriented (common with a modest sample or few variables) is dropped
rather than guessed; required edges are added regardless of what the search finds,
since they are asserted as domain knowledge, not hypotheses to test.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from causal_engine.models.schemas import CausalGraph

logger = logging.getLogger(__name__)

ALGORITHMS = ("pc", "ges", "lingam")
DEFAULT_LINGAM_THRESHOLD = 0.05

Edge = tuple[str, str]


def _oriented_edges(graph, names: list[str]) -> tuple[set[Edge], set[frozenset]]:
    """(directed edges, undirected adjacencies) of a causal-learn graph. `names[i]` is the
    variable behind the graph's i-th node. Circle/circle and arrow/arrow (bidirected)
    endpoints are ambiguous and are reported as undirected."""
    by_graph_name = {node.get_name(): names[i] for i, node in enumerate(graph.get_nodes())}
    directed: set[Edge] = set()
    undirected: set[frozenset] = set()
    for edge in graph.get_graph_edges():
        n1, n2 = by_graph_name[edge.get_node1().get_name()], by_graph_name[edge.get_node2().get_name()]
        e1, e2 = edge.get_endpoint1(), edge.get_endpoint2()
        if e1 == Endpoint.TAIL and e2 == Endpoint.ARROW:
            directed.add((n1, n2))
        elif e1 == Endpoint.ARROW and e2 == Endpoint.TAIL:
            directed.add((n2, n1))
        else:
            undirected.add(frozenset((n1, n2)))
    return directed, undirected


def _pc_edges(data: np.ndarray, variables: list[str], alpha: float, required: list[Edge], forbidden: set[Edge]) -> set[Edge]:
    nodes_by_name = {name: GraphNode(name) for name in variables}
    bk = BackgroundKnowledge()
    for parent, child in required:
        bk.add_required_by_node(nodes_by_name[parent], nodes_by_name[child])
    for parent, child in forbidden:
        bk.add_forbidden_by_node(nodes_by_name[parent], nodes_by_name[child])
    cg = pc(data, alpha=alpha, node_names=variables, background_knowledge=bk, show_progress=False)
    directed, _ = _oriented_edges(cg.G, variables)  # unoriented edges are dropped rather than guessed
    return directed


def _ges_edges(data: np.ndarray, variables: list[str], forbidden: set[Edge]) -> set[Edge]:
    from causallearn.search.ScoreBased.GES import ges

    record = ges(data, score_func="local_score_BIC")
    directed, undirected = _oriented_edges(record["G"], variables)
    for pair in undirected:
        a, b = sorted(pair)
        ab_forbidden, ba_forbidden = (a, b) in forbidden, (b, a) in forbidden
        if ab_forbidden and not ba_forbidden:
            directed.add((b, a))  # the priors allow only one direction
        elif ba_forbidden and not ab_forbidden:
            directed.add((a, b))
    return directed


def _lingam_edges(data: np.ndarray, variables: list[str], forbidden: set[Edge], threshold: float) -> set[Edge]:
    try:
        import lingam
    except ImportError as exc:
        raise ImportError("causal_discovery.algorithm='lingam' needs the `lingam` package (pip install lingam)") from exc

    index = {name: i for i, name in enumerate(variables)}
    # prior_knowledge[i, j] = 0 means "no directed path from variable j to variable i"; -1 = unknown
    prior = np.full((len(variables), len(variables)), -1, dtype=int)
    for parent, child in forbidden:
        prior[index[child], index[parent]] = 0
    scale = data.std(axis=0)
    standardised = (data - data.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
    model = lingam.DirectLiNGAM(prior_knowledge=prior).fit(standardised)
    weights = model.adjacency_matrix_  # weights[i, j] = effect of variable j on variable i
    return {
        (variables[j], variables[i])
        for i in range(len(variables))
        for j in range(len(variables))
        if i != j and abs(weights[i, j]) >= threshold
    }


def graph_warnings(edges: set[Edge]) -> list[str]:
    """Plain-language notes about the structure. Today: whether the edges contain a cycle, which a
    causal DAG cannot (an algorithm's output plus the required edges can still produce one)."""
    try:
        cycle = nx.find_cycle(nx.DiGraph(sorted(edges)))
    except nx.NetworkXNoCycle:
        return []
    path = " -> ".join([cycle[0][0], *(head for _, head in cycle)])
    return [f"the edges contain a cycle ({path}); a causal graph should be acyclic, so treat this structure with caution"]


def discover_graph(feature_df: pd.DataFrame, domain_config: dict) -> CausalGraph:
    cfg = domain_config["causal_discovery"]
    algorithm = cfg.get("algorithm", "pc")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"causal_discovery.algorithm={algorithm!r} is not supported; use one of {list(ALGORITHMS)}")
    variables: list[str] = cfg["variables"]
    alpha = cfg.get("alpha", 0.05)
    required_edges = [tuple(e) for e in cfg.get("required_edges", [])]
    forbidden_edges = {tuple(e) for e in cfg.get("forbidden_edges", [])}

    unknown = sorted({v for edge in [*required_edges, *forbidden_edges] for v in edge} - set(variables))
    if unknown:
        raise ValueError(
            f"causal_discovery required_edges/forbidden_edges name {unknown}, which are not in "
            f"causal_discovery.variables {variables}"
        )

    data = feature_df[variables].to_numpy(dtype=float)
    if algorithm == "pc":
        discovered = _pc_edges(data, variables, alpha, required_edges, forbidden_edges)
    elif algorithm == "ges":
        discovered = _ges_edges(data, variables, forbidden_edges)
    else:
        discovered = _lingam_edges(data, variables, forbidden_edges, cfg.get("lingam_threshold", DEFAULT_LINGAM_THRESHOLD))

    edges = (discovered - forbidden_edges) | set(required_edges)
    warnings = graph_warnings(edges)
    for warning in warnings:
        logger.warning("causal discovery (%s): %s", algorithm, warning)
    return CausalGraph(nodes=variables, edges=sorted(edges), algorithm=algorithm, warnings=warnings)
