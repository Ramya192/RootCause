"""List the records a domain's causal model explains least (causal anomaly flagging).

    python scripts/reports/flag_anomalies.py --domain carclaims            # top 15 on the default dataset
    python scripts/reports/flag_anomalies.py --domain german_credit --top 25 --algorithm ges

Runs Stages 1-3 (ingestion, encoding, discovery), fits each variable's mechanism given its parents in
the discovered graph, and prints the records with the largest total surprise, the variable to look at
and what it was against what its causes predicted (a probability for a binary variable, the most likely level for
a categorical one), with the probability its causes gave the observed value.

On real data there are no labelled anomalies, so this is a list of leads, not findings: "flagged" means
"not explained by this model" (a missing cause, a wrong functional form or a data-entry error look the
same), never "fraudulent" or "wrong". See causal_engine/pipeline/anomalies.py for the method and its limits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from causal_engine.pipeline import anomalies, causal_discovery, ingestion, preprocessing, runner  # noqa: E402
from causal_engine.utils.config_loader import ConfigLoader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--dataset", help="dataset kind (default: the domain's default_dataset)")
    parser.add_argument("--algorithm", choices=causal_discovery.ALGORITHMS, default="pc")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    domain = ConfigLoader().get_domain(args.domain)
    cfg = {**domain.extra, "causal_discovery": {**domain.extra["causal_discovery"], "algorithm": args.algorithm}}
    dataset = runner.resolve_dataset(cfg, args.dataset)
    cfg = runner.with_dataset(cfg, dataset)
    features = preprocessing.preprocess(ingestion.ingest(runner.resolve_data_path(cfg, dataset), cfg), cfg).df

    graph = causal_discovery.discover_graph(features, cfg)
    report = anomalies.flag_anomalies(features, graph, cfg)
    id_col = cfg["feature_store"]["entity_id_column"]
    print(f"{domain.name} / {dataset}: {len(features)} records, graph of {len(graph.edges)} edges ({graph.algorithm})")
    print(f"Records the model explains least (id column: {id_col}); a probability is shown for binary variables:\n")
    top = report.top(args.top)
    top["record"] = top["record"].astype(int)
    print(top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    share = (report.surprises[list(report.scored)].max(axis=1) > 6).mean()
    print(f"\n{share:.1%} of records have a single variable more than 6 nats more surprising than expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
