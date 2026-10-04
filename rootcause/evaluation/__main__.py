"""`python -m rootcause.evaluation` -- run every dataset and print the results table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rootcause.evaluation import (
    anomalies_eval,
    baselines_report,
    harness,
    learners,
    multimodal_eval,
    report,
    sensitivity_eval,
    stress_report,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STRESS_REPLICATES = 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rootcause.evaluation",
        description="Run the pipeline on every configured dataset and score it against ground truth.",
    )
    parser.add_argument("--replicates", type=int, default=None,
                        help="also run on this many fresh draws from each dataset's SCM "
                        f"(default 0; {DEFAULT_STRESS_REPLICATES} with --stress/--baselines)")
    parser.add_argument("--n-rows", type=int, default=None,
                        help="rows per replicate draw (default: the size of each dataset's committed file; "
                        "ignored by --stress and --baselines, whose scenarios fix their own n)")
    parser.add_argument("--stress", action="store_true",
                        help="run the stress scenarios (small n, confounding, nonlinearity) instead of "
                        "the dataset evaluation; writes stress.md and stress.json")
    parser.add_argument("--baselines", action="store_true",
                        help="run the stress scenarios through the pipeline AND naive baselines (plain "
                        "regression, SHAP importance) on the same draws; writes baselines.md and baselines.json")
    parser.add_argument("--sensitivity", action="store_true",
                        help="check the Stage 4 sensitivity analysis against hidden confounders of known "
                        "strength; writes sensitivity.md and sensitivity.json")
    parser.add_argument("--learners", action="store_true",
                        help="compare the Stage 5 meta-learners (T/S/X/R/DR, linear and gbm bases) on the Illinois "
                        "trial, its bootstrap and its simulated mirror; writes learners.md and learners.json")
    parser.add_argument("--anomalies", action="store_true",
                        help="check causal anomaly flagging against injected mechanism faults and standard outlier "
                        "detectors; writes anomalies.md and anomalies.json")
    parser.add_argument("--anomalies-real", action="store_true",
                        help="check causal anomaly flagging on the real covariates of carclaims, German Credit and Freddie Mac 2007 "
                        "(injected faults; type-aware vs Gaussian mechanisms; how concentrated the top of the "
                        "list is); writes anomalies_real.md and anomalies_real.json")
    parser.add_argument("--multimodal", action="store_true",
                        help="check the PDF/image ingestion path on the three domains' synthetic attachments "
                        "(extraction validity, and Stages 1-6 with vs without the features); needs "
                        "scripts/generate_attachments.py to have run; writes multimodal.md and multimodal.json")
    parser.add_argument("--workers", type=int, default=1,
                        help="with --stress or --baselines, run this many scenarios in parallel processes "
                        "(default 1; results are identical to a serial run)")
    parser.add_argument("--scenario", action="append",
                        help="with --stress or --baselines, only this scenario id (repeatable)")
    parser.add_argument("--algorithm", choices=("pc", "ges", "lingam"), default=None,
                        help="force Stage 3's causal discovery algorithm instead of the configured one; results go "
                        "to <name>_<algorithm>.md/json so the default runs are not overwritten")
    parser.add_argument("--domain", action="append", help="only this domain id (repeatable)")
    parser.add_argument("--dataset", action="append", help="only this dataset kind (repeatable)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "evaluation",
                        help="directory for results.md and results.json (default %(default)s)")
    parser.add_argument("--no-write", action="store_true", help="print only; don't write files")
    args = parser.parse_args(argv)

    # The report uses ±, τ and –; a Windows console defaults to cp1252 and would crash on them.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.multimodal:
        if args.stress or args.baselines or args.sensitivity or args.learners or args.anomalies:
            parser.error("--multimodal is a separate run; do not combine it with other modes")
        results = multimodal_eval.run(domains=args.domain or multimodal_eval.DOMAINS)
        markdown = multimodal_eval.render_markdown(results)
        print(markdown)
        if not args.no_write:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "multimodal.md").write_text(markdown, encoding="utf-8")
            (args.out / "multimodal.json").write_text(multimodal_eval.render_json(results), encoding="utf-8")
            print(f"\nWrote {args.out / 'multimodal.md'} and multimodal.json", file=sys.stderr)
        return 0
    if args.anomalies_real:
        if args.stress or args.baselines or args.sensitivity or args.learners or args.anomalies or args.multimodal:
            parser.error("--anomalies-real is a separate run; do not combine it with other modes")
        result = anomalies_eval.run_real(replicates=5 if args.replicates is None else args.replicates)
        markdown = anomalies_eval.render_real_markdown(result)
        print(markdown)
        if not args.no_write:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "anomalies_real.md").write_text(markdown, encoding="utf-8")
            (args.out / "anomalies_real.json").write_text(anomalies_eval.render_real_json(result), encoding="utf-8")
            print(f"\nWrote {args.out / 'anomalies_real.md'} and anomalies_real.json", file=sys.stderr)
        return 0
    if args.anomalies:
        if args.stress or args.baselines or args.sensitivity or args.learners:
            parser.error("--anomalies is a separate run; do not combine it with other modes")
        result = anomalies_eval.run(replicates=DEFAULT_STRESS_REPLICATES if args.replicates is None else args.replicates)
        markdown = anomalies_eval.render_markdown(result)
        print(markdown)
        if not args.no_write:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "anomalies.md").write_text(markdown, encoding="utf-8")
            (args.out / "anomalies.json").write_text(anomalies_eval.render_json(result), encoding="utf-8")
            print(f"\nWrote {args.out / 'anomalies.md'} and anomalies.json", file=sys.stderr)
        return 0
    if args.learners:
        if args.stress or args.baselines or args.sensitivity:
            parser.error("--learners is a separate run; do not combine it with other modes")
        result = learners.run(replicates=DEFAULT_STRESS_REPLICATES if args.replicates is None else args.replicates)
        markdown = learners.render_markdown(result)
        print(markdown)
        if not args.no_write:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "learners.md").write_text(markdown, encoding="utf-8")
            (args.out / "learners.json").write_text(learners.render_json(result), encoding="utf-8")
            print(f"\nWrote {args.out / 'learners.md'} and learners.json", file=sys.stderr)
        return 0
    if args.sensitivity:
        if args.stress or args.baselines:
            parser.error("--sensitivity is a separate run; do not combine it with --stress/--baselines")
        result = sensitivity_eval.run(replicates=DEFAULT_STRESS_REPLICATES if args.replicates is None else args.replicates)
        markdown = sensitivity_eval.render_markdown(result)
        print(markdown)
        if not args.no_write:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "sensitivity.md").write_text(markdown, encoding="utf-8")
            (args.out / "sensitivity.json").write_text(sensitivity_eval.render_json(result), encoding="utf-8")
            print(f"\nWrote {args.out / 'sensitivity.md'} and sensitivity.json", file=sys.stderr)
        return 0
    if args.stress and args.baselines:
        parser.error("--stress and --baselines are separate runs; pick one")
    if args.stress or args.baselines:
        replicates = DEFAULT_STRESS_REPLICATES if args.replicates is None else args.replicates
        if replicates < 1:
            parser.error("--stress and --baselines need --replicates >= 1")
        results = harness.run_stress(
            replicates, scenario_ids=args.scenario, with_baselines=args.baselines, workers=args.workers,
            algorithm=args.algorithm,
        )
        settings = {"replicates": replicates, **({"algorithm": args.algorithm} if args.algorithm else {})}
        renderer = baselines_report if args.baselines else stress_report
        markdown = renderer.render_markdown(results, settings)
        stem = "baselines" if args.baselines else "stress"
    else:
        if args.scenario:
            parser.error("--scenario only applies with --stress or --baselines")
        if args.workers != 1:
            parser.error("--workers only applies with --stress or --baselines")
        replicates = args.replicates or 0
        results = harness.run_evaluation(
            domains=args.domain, datasets=args.dataset, replicates=replicates, n_rows=args.n_rows,
            algorithm=args.algorithm,
        )
        settings = {"replicates": replicates, "n_rows": args.n_rows, **({"algorithm": args.algorithm} if args.algorithm else {})}
        markdown = report.render_markdown(results, settings)
        stem = "results"
    print(markdown)
    if args.algorithm:
        stem = f"{stem}_{args.algorithm}"

    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{stem}.md").write_text(markdown, encoding="utf-8")
        (args.out / f"{stem}.json").write_text(report.render_json(results, settings), encoding="utf-8")
        print(f"\nWrote {args.out / (stem + '.md')} and {stem}.json", file=sys.stderr)

    failed = any(r.error for sc in results for r in sc.runs)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
