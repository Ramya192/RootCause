"""Causal graph drawing and DOT export (causal_engine/utils/graph_plot.py) and the API route that serves it."""

from __future__ import annotations

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from causal_engine.api.app import create_app
from causal_engine.models.schemas import CausalGraph, Explanation, PipelineResult
from causal_engine.utils import graph_plot

CONFIG = {
    "effect_estimation": {"outcome": "y", "treatments": [{"name": "t"}]},
    "interventions": {"sensitive_attribute": "s"},
    "causal_discovery": {"required_edges": [["m", "y"]]},
}


def _graph() -> CausalGraph:
    return CausalGraph(
        nodes=["t", "m", "y", "s", "lonely"],
        edges=[("t", "m"), ("m", "y"), ("s", "y")],
        algorithm="pc",
    )


def test_node_roles_come_from_the_domain_config():
    roles = graph_plot.node_roles(_graph(), CONFIG)
    assert roles == {"t": "treatment", "m": "other", "y": "outcome", "s": "sensitive", "lonely": "other"}
    assert set(graph_plot.node_roles(_graph(), None).values()) == {"other"}  # no config, no roles


def test_networkx_graph_marks_required_edges_as_priors():
    g = graph_plot.to_networkx(_graph(), CONFIG)
    assert isinstance(g, nx.DiGraph) and set(g.edges) == {("t", "m"), ("m", "y"), ("s", "y")}
    assert g.edges["m", "y"]["prior"] is True and g.edges["t", "m"]["prior"] is False
    assert g.nodes["y"]["role"] == "outcome"


def test_layers_put_each_variable_below_its_deepest_parent():
    g = graph_plot.to_networkx(_graph(), CONFIG)
    g.remove_node("lonely")
    positions = graph_plot.layered_positions(g)
    assert positions["t"][1] == positions["s"][1] == 0.0  # roots share the top layer
    assert positions["m"][1] == -1.0 and positions["y"][1] == -2.0  # y is below m, its deepest parent


def test_a_cycle_falls_back_to_a_spring_layout_instead_of_crashing(tmp_path):
    cyclic = CausalGraph(nodes=["a", "b"], edges=[("a", "b"), ("b", "a")], algorithm="lingam")
    positions = graph_plot.layered_positions(graph_plot.to_networkx(cyclic))
    assert set(positions) == {"a", "b"}
    assert graph_plot.plot_causal_graph(cyclic, tmp_path / "cycle.png").stat().st_size > 1000


def test_plot_writes_a_png_and_creates_missing_directories(tmp_path):
    path = graph_plot.plot_causal_graph(_graph(), tmp_path / "nested" / "dir" / "g.png", CONFIG, title="Test")
    assert path.exists() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 5000


def test_plot_handles_a_graph_with_no_edges(tmp_path):
    empty = CausalGraph(nodes=["a", "b"], edges=[], algorithm="pc")
    assert graph_plot.plot_causal_graph(empty, tmp_path / "empty.png").read_bytes()[:4] == b"\x89PNG"


def test_labels_wrap_long_names_and_split_one_hot_levels():
    assert graph_plot._label("credit_amount_kdm") == "credit\namount kdm"
    assert graph_plot._label("checking_status__lt_0") == "checking\nstatus\nlt 0"


def test_dot_export_has_every_node_edge_and_dashes_the_priors():
    dot = graph_plot.to_dot(_graph(), CONFIG)
    assert dot.startswith("digraph causal_graph {") and dot.rstrip().endswith("}")
    assert '"m" -> "y" [style=dashed];' in dot and '"t" -> "m";' in dot
    assert '"lonely" [fillcolor=' in dot  # isolated nodes are still declared
    assert graph_plot.COLORS["treatment"] in dot


# --- API route ----------------------------------------------------------------------------


@pytest.fixture
def client(config_loader):
    def run(domain_id, config, path):
        return PipelineResult(
            domain_id=domain_id,
            causal_graph=CausalGraph(nodes=["a", "b"], edges=[("a", "b")], algorithm="pc"),
            effect_estimates=[], counterfactuals=[], recommendations=[], explanation=Explanation(narrative="n"),
        )

    return TestClient(create_app(config_loader=config_loader, runners={"direct": run, "crew": run}))


def _finished_job(client) -> str:
    import time

    job_id = client.post("/domains/employee_attrition/analyze").json()["id"]
    for _ in range(400):
        if client.get(f"/jobs/{job_id}").json()["status"] == "succeeded":
            return job_id
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_graph_route_serves_a_png_or_dot(client):
    job_id = _finished_job(client)
    png = client.get(f"/jobs/{job_id}/graph")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png" and png.content[:4] == b"\x89PNG"
    dot = client.get(f"/jobs/{job_id}/graph", params={"format": "dot"})
    assert dot.status_code == 200 and '"a" -> "b"' in dot.text
    assert client.get(f"/jobs/{job_id}/graph", params={"format": "svg"}).status_code == 422


def test_graph_route_404_for_unknown_job_and_409_before_the_job_finishes(client):
    assert client.get("/jobs/nope/graph").status_code == 404

    import threading

    release = threading.Event()

    def slow(domain_id, config, path):
        release.wait(5)
        raise RuntimeError("never mind")

    from causal_engine.utils.config_loader import ConfigLoader

    slow_client = TestClient(create_app(config_loader=ConfigLoader(), runners={"direct": slow, "crew": slow}))
    job_id = slow_client.post("/domains/employee_attrition/analyze").json()["id"]
    response = slow_client.get(f"/jobs/{job_id}/graph")
    release.set()
    assert response.status_code == 409 and "no graph yet" in response.json()["detail"]
