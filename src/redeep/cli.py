"""Command-line interface for the remote ReDeEP reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .config import load_config
from .doctor import run_doctor
from .pipeline import (
    EVAL_SPLITS,
    audit_dataset,
    calibrate_model,
    compare_model_results,
    discover_heads,
    evaluate_model,
    extract_features,
    run_all,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="redeep",
        description="Cross-backbone ReDeEP reproduction on fixed RAGTruth responses.",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Experiment YAML path (default: configs/experiment.yaml).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check data, models, and runtime.")
    doctor.add_argument("--model", help="Inspect only one configured model key.")
    doctor.add_argument(
        "--no-load-tokenizers",
        action="store_true",
        help="Run filesystem checks without importing Transformers.",
    )

    audit = subparsers.add_parser(
        "audit-data",
        help="Validate RAGTruth joins, spans, contexts, counts, and fixed dev split.",
    )
    audit.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not require the official complete Llama2 subset counts.",
    )

    discover = subparsers.add_parser(
        "discover-heads",
        help="Discover structural Copying Head candidates for one scorer.",
    )
    discover.add_argument("--model", required=True, help="Configured model key.")
    discover.add_argument("--force", action="store_true", help="Replace cached artifact.")

    extract = subparsers.add_parser(
        "extract",
        help="Extract resumable one-response token feature shards.",
    )
    extract.add_argument("--model", required=True, help="Configured model key.")
    extract.add_argument("--split", required=True, choices=EVAL_SPLITS)
    extract.add_argument("--force", action="store_true", help="Replace existing shards.")
    extract.add_argument(
        "--limit",
        type=_positive_int,
        help="Process at most N pending examples (smoke/debug only).",
    )
    extract.add_argument(
        "--num-shards",
        type=_positive_int,
        default=1,
        help="Deterministically divide the selected split across N workers.",
    )
    extract.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based worker index; must be smaller than --num-shards.",
    )

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Select heads/layers/beta using calibration-train and dev.",
    )
    calibrate.add_argument("--model", required=True, help="Configured model key.")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate frozen calibration on untouched test shards.",
    )
    evaluate.add_argument("--model", required=True, help="Configured model key.")

    compare = subparsers.add_parser(
        "compare",
        help="Run paired response-cluster bootstrap for two completed scorers.",
    )
    compare.add_argument("--first", required=True, help="First configured model key.")
    compare.add_argument("--second", required=True, help="Second configured model key.")

    all_parser = subparsers.add_parser(
        "run-all",
        help="Run discovery, three extraction splits, calibration, and evaluation.",
    )
    all_parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Configured model key; repeat for multiple. Defaults to all.",
    )
    all_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace head artifact and all feature shards.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            report = run_doctor(
                config,
                model_key=args.model,
                load_tokenizers=not args.no_load_tokenizers,
            )
            _emit(report)
            return 0 if report["ok"] else 2
        if args.command == "audit-data":
            _emit(audit_dataset(config, strict_official_counts=not args.no_strict))
            return 0
        if args.command == "discover-heads":
            _emit(discover_heads(config, args.model, force=args.force))
            return 0
        if args.command == "extract":
            _emit(
                extract_features(
                    config,
                    args.model,
                    args.split,
                    force=args.force,
                    limit=args.limit,
                    num_shards=args.num_shards,
                    shard_index=args.shard_index,
                )
            )
            return 0
        if args.command == "calibrate":
            _emit(calibrate_model(config, args.model))
            return 0
        if args.command == "evaluate":
            _emit(evaluate_model(config, args.model))
            return 0
        if args.command == "compare":
            _emit(compare_model_results(config, args.first, args.second))
            return 0
        if args.command == "run-all":
            _emit(run_all(config, model_keys=args.models, force=args.force))
            return 0
        parser.error(f"Unsupported command: {args.command}")
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
