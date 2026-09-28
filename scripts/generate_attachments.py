"""Render synthetic PDF and image attachments for the three real domains.

    python scripts/generate_attachments.py                       # all three domains
    python scripts/generate_attachments.py --domain carclaims --limit 200

Writes under data/<domain>/attachments/ (git-ignored: it is regenerated deterministically from
the tabular data) and, for carclaims, the 5,000-claim row sample data/carclaims/claims_multimodal.csv.
The attachments are rendered from pre-outcome covariates only and are SYNTHETIC: see
rootcause/evaluation/attachment_synth.py for what that does and does not show.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rootcause.evaluation import attachment_synth  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", action="append", choices=sorted(attachment_synth.DOMAINS))
    parser.add_argument("--limit", type=int, help="only the first N records (for a quick look)")
    args = parser.parse_args()
    for domain in args.domain or sorted(attachment_synth.DOMAINS):
        start = time.perf_counter()
        info = attachment_synth.generate(domain, REPO_ROOT, limit=args.limit)
        print(f"{domain}: {info['records']} records in {time.perf_counter() - start:.0f}s -> {info['pdf_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
