"""Stage 3: Causal Discovery (causal_engine/pipeline/causal_discovery.py).

Regression test for the result documented in memory: PC + domain priors
recovers all 6 ground-truth edges exactly, with zero spurious edges, on the
committed 2000-row synthetic dataset (seed=42).
"""

from __future__ import annotations


def test_recovers_ground_truth_edges_exactly(causal_graph, ground_truth):
    expected = {tuple(e) for e in ground_truth["edges"]}
    discovered = set(causal_graph.edges)
    assert discovered == expected


def test_no_forbidden_edges_present(causal_graph, domain_config):
    forbidden = {tuple(e) for e in domain_config["causal_discovery"]["forbidden_edges"]}
    assert not (set(causal_graph.edges) & forbidden)


def test_required_edges_always_present(causal_graph, domain_config):
    required = {tuple(e) for e in domain_config["causal_discovery"]["required_edges"]}
    assert required.issubset(set(causal_graph.edges))


def test_graph_nodes_match_configured_variables(causal_graph, domain_config):
    assert causal_graph.nodes == domain_config["causal_discovery"]["variables"]
    assert causal_graph.algorithm == "pc"


def test_a_cycle_in_the_edges_is_reported_in_plain_language():
    from causal_engine.pipeline import causal_discovery

    assert causal_discovery.graph_warnings({("a", "b"), ("b", "c")}) == []
    [warning] = causal_discovery.graph_warnings({("a", "b"), ("b", "c"), ("c", "a")})
    assert "cycle" in warning and all(name in warning for name in ("a", "b", "c"))


def test_the_committed_dataset_gives_an_acyclic_graph(causal_graph):
    assert causal_graph.warnings == []
