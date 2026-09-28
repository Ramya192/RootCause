"""Draw the Stage 3 causal graph for each domain's dataset into docs/figures/.

    python scripts/plot_causal_graphs.py                       # every domain, its default dataset, PC
    python scripts/plot_causal_graphs.py --algorithm ges       # same with GES
    python scripts/plot_causal_graphs.py --domain carclaims --dataset real

Runs Stages 1-3 only (ingestion, encoding, discovery), so it is fast. The graph is what the
algorithm found on that dataset and, on real data, is descriptive: there is no true graph to
compare it with.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rootcause.pipeline import causal_discovery, ingestion, preprocessing, runner  # noqa: E402
from rootcause.utils.config_loader import ConfigLoader  # noqa: E402
from rootcause.utils.graph_plot import plot_causal_graph, to_dot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", action="append", help="only this domain id (repeatable)")
    parser.add_argument("--dataset", help="dataset kind (default: each domain's default_dataset)")
    parser.add_argument("--algorithm", choices=causal_discovery.ALGORITHMS, default="pc")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "figures")
    parser.add_argument("--dot", action="store_true", help="also write Graphviz DOT files")
    args = parser.parse_args()

    for domain in ConfigLoader().list_domains(runnable_only=True):
        if args.domain and domain.id not in args.domain:
            continue
        cfg = {**domain.extra, "causal_discovery": {**domain.extra["causal_discovery"], "algorithm": args.algorithm}}
        dataset = runner.resolve_dataset(cfg, args.dataset)
        path = runner.resolve_data_path(cfg, dataset)
        if not path.is_file():
            print(f"skip {domain.id}/{dataset}: {path} not found")
            continue
        features = preprocessing.preprocess(ingestion.ingest(path, cfg), cfg).df
        graph = causal_discovery.discover_graph(features, cfg)
        stem = f"{domain.id}_{dataset}_{args.algorithm}"
        title = f"{domain.name} - {dataset} data"
        png = plot_causal_graph(graph, args.out / f"{stem}.png", cfg, title=title)
        print(f"{png.relative_to(REPO_ROOT)}: {len(graph.edges)} edges")
        if args.dot:
            (args.out / f"{stem}.dot").write_text(to_dot(graph, cfg), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
