"""Draw the Stage 3 causal graph (NetworkX + matplotlib) or export it as Graphviz DOT.

The figure is a layered DAG: a variable sits one layer below its deepest parent, so
causes read top to bottom. Colour says what the variable is to the analysis, so the
picture answers "how does the discovered structure relate to what we are estimating":

    treatment (a lever in Stage 4)   blue
    outcome                          red
    sensitive attribute (Stage 6)    orange
    everything else                  grey

Solid edges were found by the algorithm; dashed edges are domain priors
(`required_edges`), asserted rather than discovered. Variables with no edges are not
drawn in the layers (they would all pile up in the top row); they are listed under the
figure instead, because "found nothing for X" is information, not clutter.

Rendering uses matplotlib only, so no Graphviz binary is needed. `to_dot` writes the
same graph as DOT text for anyone who has Graphviz installed.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

import networkx as nx

from causal_engine.models.schemas import CausalGraph

COLORS = {"treatment": "#4C78A8", "outcome": "#E45756", "sensitive": "#F58518", "other": "#BAB0AC"}
LEGEND = {
    "treatment": "treatment (Stage 4 lever)",
    "outcome": "outcome",
    "sensitive": "sensitive attribute",
    "other": "other variable",
}


def node_roles(graph: CausalGraph, domain_config: Optional[dict]) -> dict[str, str]:
    """`treatment` / `outcome` / `sensitive` / `other` for every node, from the domain config."""
    roles = {node: "other" for node in graph.nodes}
    if not domain_config:
        return roles
    effect = domain_config.get("effect_estimation", {})
    sensitive = domain_config.get("interventions", {}).get("sensitive_attribute")
    for treatment in effect.get("treatments", []):
        if treatment["name"] in roles:
            roles[treatment["name"]] = "treatment"
    if sensitive in roles:
        roles[sensitive] = "sensitive"
    if effect.get("outcome") in roles:
        roles[effect["outcome"]] = "outcome"
    return roles


def prior_edges(domain_config: Optional[dict]) -> set[tuple[str, str]]:
    if not domain_config:
        return set()
    return {tuple(e) for e in domain_config.get("causal_discovery", {}).get("required_edges", [])}


def to_networkx(graph: CausalGraph, domain_config: Optional[dict] = None) -> nx.DiGraph:
    """The graph with `role` on each node and `prior` (True for a required edge) on each edge."""
    roles = node_roles(graph, domain_config)
    priors = prior_edges(domain_config)
    g = nx.DiGraph()
    for node in graph.nodes:
        g.add_node(node, role=roles[node])
    for parent, child in graph.edges:
        g.add_edge(parent, child, prior=(parent, child) in priors)
    return g


NODE_SIZE = 5200


def _label(name: str) -> str:
    """A variable name as short wrapped lines that fit inside its node."""
    return "\n".join(textwrap.fill(part.replace("_", " "), width=11) for part in name.split("__"))


def layered_positions(g: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Layer by longest path from a root; centre each layer. Falls back to a spring layout if
    the graph has a cycle (an algorithm can return one) because layers are then undefined."""
    if not nx.is_directed_acyclic_graph(g):
        return {n: (float(x), float(y)) for n, (x, y) in nx.spring_layout(g, seed=0).items()}
    positions = {}
    for depth, layer in enumerate(nx.topological_generations(g)):
        layer = sorted(layer)
        for i, node in enumerate(layer):
            positions[node] = (i - (len(layer) - 1) / 2.0, -float(depth))
    return positions


def plot_causal_graph(
    graph: CausalGraph,
    path: str | Path,
    domain_config: Optional[dict] = None,
    title: Optional[str] = None,
) -> Path:
    """Write the layered figure to `path` (PNG by extension) and return the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    g = to_networkx(graph, domain_config)
    isolated = sorted(n for n in g if g.degree(n) == 0)
    connected = g.subgraph([n for n in g if g.degree(n) > 0]).copy()
    positions = layered_positions(connected) if connected.number_of_nodes() else {}

    width = max(6.0, 2.2 * (max((abs(x) for x, _ in positions.values()), default=0) * 2 + 1))
    height = max(3.0, 1.6 * (len({y for _, y in positions.values()}) or 1) + 1.4)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_axis_off()
    if connected.number_of_nodes():
        colors = [COLORS[connected.nodes[n]["role"]] for n in connected]
        nx.draw_networkx_nodes(connected, positions, node_color=colors, node_size=NODE_SIZE, edgecolors="#333333", ax=ax)
        nx.draw_networkx_labels(connected, positions, labels={n: _label(n) for n in connected}, font_size=8, ax=ax)
        for prior in (False, True):
            edges = [(u, v) for u, v, d in connected.edges(data=True) if d["prior"] == prior]
            nx.draw_networkx_edges(
                connected, positions, edgelist=edges, arrows=True, arrowstyle="-|>", arrowsize=18, node_size=NODE_SIZE,
                style="dashed" if prior else "solid", edge_color="#444444", width=1.6, ax=ax,
            )
        ax.margins(0.18)
    else:
        ax.text(0.5, 0.5, "No edges found", ha="center", va="center", fontsize=12, transform=ax.transAxes)

    handles = [Patch(facecolor=COLORS[role], edgecolor="#333333", label=LEGEND[role]) for role in COLORS]
    handles.append(Line2D([0], [0], color="#444444", lw=1.6, label=f"edge found by {graph.algorithm}"))
    if any(d["prior"] for _, _, d in g.edges(data=True)):
        handles.append(Line2D([0], [0], color="#444444", lw=1.6, ls="--", label="domain prior (required)"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False)

    caption = f"{len(graph.edges)} directed edges among {len(graph.nodes)} variables ({graph.algorithm})"
    if isolated:
        caption += "\nNo edges found for: " + ", ".join(isolated)
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=8, color="#555555")
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def to_dot(graph: CausalGraph, domain_config: Optional[dict] = None) -> str:
    """The graph as Graphviz DOT text (render with `dot -Tpng`); same colours and dashed priors."""
    g = to_networkx(graph, domain_config)
    lines = ["digraph causal_graph {", "  rankdir=TB;", '  node [shape=box, style="rounded,filled", fontname="Helvetica"];']
    for node, data in g.nodes(data=True):
        lines.append(f'  "{node}" [fillcolor="{COLORS[data["role"]]}"];')
    for parent, child, data in g.edges(data=True):
        style = ' [style=dashed]' if data["prior"] else ""
        lines.append(f'  "{parent}" -> "{child}"{style};')
    lines.append("}")
    return "\n".join(lines) + "\n"
