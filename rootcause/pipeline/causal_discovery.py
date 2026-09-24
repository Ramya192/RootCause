"""Stage 3: Causal Discovery.

Runs causal-learn's PC algorithm over the domain's configured variables,
seeded with domain priors (required_edges / forbidden_edges) as
BackgroundKnowledge so statistical structure learning doesn't have to
rediscover what a domain expert already knows -- and can't contradict it.
Any edge PC still leaves unoriented (common with a modest sample / few
variables) is dropped rather than guessed; required edges are added
regardless of what PC finds, since they're asserted as domain knowledge,
not hypotheses to test.
"""

from __future__ import annotations

import pandas as pd
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from rootcause.models.schemas import CausalGraph


def discover_graph(feature_df: pd.DataFrame, domain_config: dict) -> CausalGraph:
    cfg = domain_config["causal_discovery"]
    variables: list[str] = cfg["variables"]
    alpha = cfg.get("alpha", 0.05)
    required_edges = [tuple(e) for e in cfg.get("required_edges", [])]
    forbidden_edges = {tuple(e) for e in cfg.get("forbidden_edges", [])}

    data = feature_df[variables].to_numpy(dtype=float)
    nodes_by_name = {name: GraphNode(name) for name in variables}

    bk = BackgroundKnowledge()
    for parent, child in required_edges:
        bk.add_required_by_node(nodes_by_name[parent], nodes_by_name[child])
    for parent, child in forbidden_edges:
        bk.add_forbidden_by_node(nodes_by_name[parent], nodes_by_name[child])

    cg = pc(
        data,
        alpha=alpha,
        node_names=variables,
        background_knowledge=bk,
        show_progress=False,
    )

    discovered_edges: set[tuple[str, str]] = set()
    for edge in cg.G.get_graph_edges():
        n1, n2 = edge.get_node1().get_name(), edge.get_node2().get_name()
        e1, e2 = edge.get_endpoint1(), edge.get_endpoint2()
        if e1 == Endpoint.TAIL and e2 == Endpoint.ARROW:
            discovered_edges.add((n1, n2))
        elif e1 == Endpoint.ARROW and e2 == Endpoint.TAIL:
            discovered_edges.add((n2, n1))
        # circle/circle or arrow/arrow (unoriented or bidirected) edges are
        # ambiguous given this sample -- dropped rather than guessed.

    edges = (discovered_edges - forbidden_edges) | set(required_edges)

    return CausalGraph(nodes=variables, edges=sorted(edges), algorithm=cfg.get("algorithm", "pc"))
