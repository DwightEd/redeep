#!/usr/bin/env python3
"""Run the released ReDeEP score under a fixed-response token protocol."""

from __future__ import annotations

import argparse
import gc
import gzip
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

from redeep_token.data import (
    DEFAULT_SYSTEM_PROMPT,
    TASKS,
    build_teacher_forced_encoding,
    load_ragtruth_samples,
)
from redeep_token.feature_store import (
    SCHEMA_VERSION,
    feature_path,
    read_feature_record,
    write_feature_record,
)
from redeep_token.protocol import (
    apply_calibration,
    calibrate_redeep,
    compute_response_metrics,
    compute_token_metrics,
)
from redeep_token.provenance import (
    checkpoint_artifact_manifest,
    file_manifest,
    sha256_file,
)
from redeep_token.released_config import (
    ReleasedTokenConfiguration,
    load_released_llama3_token_config,
    score_rows_with_released_config,
)
from redeep_token.workflow import (
    format_markdown_metrics,
    protocol_fingerprint,
    response_ids_sha256,
    select_longest_per_task,
)


UPSTREAM_COMMIT = "4d081915b8fb4430fda65c411da61540cc73cc57"
RAGTRUTH_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
RAGTRUTH_HASHES = {
    "response.jsonl": (
        "e4c2e4ac24fff676d8984cc61c35d791"
        "612fadc58015335d97dd632375e18073"
    ),
    "source_info.jsonl": (
        "0dffc26ea9f3c1c3d7c7e8336b56ef1"
        "646e3cec876edffcca3c9c624d12d578b"
    ),
}
OFFICIAL_CANDIDATE_HEADS_SHA256 = (
    "c53edccab60a71489877aaa08e9c111736437b50028f2bcec00ef5d5525"
)
OFFICIAL_CANDIDATE_CHECKPOINT = "meta-llama/Meta-Llama-3-8B-Instruct"
EVALUATION_CODE_PATHS = (
    "run_redeep_token_eval.py",
    "redeep_token/__init__.py",
    "redeep_token/data.py",
    "redeep_token/feature_store.py",
    "redeep_token/protocol.py",
    "redeep_token/provenance.py",
    "redeep_token/released_config.py",
    "redeep_token/scoring.py",
    "redeep_token/workflow.py",
)


def verify_ragtruth(data_dir: Path) -> dict[str, str]:
    actual = {}
    for filename, expected_hash in RAGTRUTH_HASHES.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; use the pinned official RAGTruth release"
            )
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{path} has SHA256 {actual_hash}, expected {expected_hash}"
            )
        actual[filename] = actual_hash
    return actual


def excluded_qualities_from_args(
    args: argparse.Namespace,
) -> frozenset[str]:
    if args.quality_cohort == "all":
        return frozenset()
    if args.quality_cohort == "quality-good":
        return frozenset({"incorrect_refusal", "truncated"})
    raise ValueError(f"unsupported quality cohort {args.quality_cohort!r}")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_text(
            json.dumps(
                dict(value),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_candidate_heads(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> list[tuple[int, int]]:
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"{path} has SHA256 {actual_sha256}, expected released "
            f"candidate-head SHA256 {expected_sha256}"
        )
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, list):
        raise ValueError("candidate-head file must contain a list")
    candidates = []
    for index, pair in enumerate(value):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in pair)
        ):
            raise ValueError(f"candidate head {index} is invalid")
        candidates.append((int(pair[0]), int(pair[1])))
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate-head file contains duplicates")
    return sorted(candidates)


def detector_model_type(model_path: Path) -> str:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    value = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(value.get("model_type", "")).lower()
    if model_type not in {"llama", "qwen3"}:
        raise ValueError(
            f"unsupported detector architecture {model_type!r}"
        )
    return model_type


def validate_candidate_transfer(
    *,
    model_type: str,
    allow_checkpoint_transfer: bool,
    allow_cross_architecture_transfer: bool,
) -> None:
    if not allow_checkpoint_transfer:
        raise ValueError(
            "the released Copying-Head set is checkpoint-specific; "
            "checkpoint transfer requires --allow-checkpoint-transfer"
        )
    if model_type == "qwen3" and not allow_cross_architecture_transfer:
        raise ValueError(
            "Qwen3 also requires --allow-cross-architecture-transfer because "
            "the released Copying-Head candidates are from a Llama model"
        )


def validate_split_separation(
    *,
    calibration_split: str,
    test_split: str,
    calibration_samples: Sequence[Mapping[str, Any]],
    test_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove that configuration labels cannot enter target test evaluation."""

    if calibration_split == test_split:
        raise ValueError(
            "calibration and test must use different splits"
        )
    calibration_ids = {
        str(sample["id"]) for sample in calibration_samples
    }
    test_ids = {str(sample["id"]) for sample in test_samples}
    response_overlap = sorted(calibration_ids & test_ids)
    if response_overlap:
        raise ValueError(
            "calibration and test response IDs overlap: "
            f"{response_overlap[:5]}"
        )
    calibration_sources = {
        str(sample["source_id"]) for sample in calibration_samples
    }
    test_sources = {
        str(sample["source_id"]) for sample in test_samples
    }
    source_overlap = sorted(calibration_sources & test_sources)
    if source_overlap:
        raise ValueError(
            "calibration and test source IDs overlap: "
            f"{source_overlap[:5]}"
        )
    return {
        "calibration_split": calibration_split,
        "test_split": test_split,
        "response_ids_disjoint": True,
        "source_ids_disjoint": True,
        "calibration_response_count": len(calibration_samples),
        "test_response_count": len(test_samples),
    }


def runtime_transformers_source(model_type: str) -> dict[str, Any]:
    module_name = (
        "transformers.models.llama.modeling_llama"
        if model_type == "llama"
        else "transformers.models.qwen3.modeling_qwen3"
    )
    module = importlib.import_module(module_name)
    source = Path(module.__file__).resolve()
    return {
        "module": module_name,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def build_protocol(
    args: argparse.Namespace,
    *,
    candidate_heads: Sequence[Sequence[int]],
    model_type: str,
    released_configuration: ReleasedTokenConfiguration | None,
    transformers_version: str,
    torch_version: str,
    tokenizers_version: str,
    cuda_version: str | None,
    cuda_device_name: str | None,
    ragtruth_hashes: Mapping[str, str],
    split_validation: Mapping[str, Any],
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parent
    model_path = args.model_name_or_path.resolve()
    checkpoint_manifest = checkpoint_artifact_manifest(model_path)
    evaluation_code_manifest = file_manifest(
        repository_root,
        EVALUATION_CODE_PATHS,
    )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "method": "ReDeEP-Token fixed-response adaptation",
        "configuration_mode": args.configuration_mode,
        "upstream_repository": "https://github.com/Jeryi-Sun/ReDEeP-ICLR",
        "upstream_commit": UPSTREAM_COMMIT,
        "ragtruth_commit": RAGTRUTH_COMMIT,
        "ragtruth_files": dict(ragtruth_hashes),
        "detector_model": str(model_path),
        "detector_checkpoint_artifacts": checkpoint_manifest,
        "detector_architecture": model_type,
        "candidate_heads_source_checkpoint": (
            OFFICIAL_CANDIDATE_CHECKPOINT
        ),
        "candidate_heads_checkpoint_transfer": True,
        "cross_architecture_transfer": model_type != "llama",
        "checkpoint_transfer_explicitly_authorized": (
            args.allow_checkpoint_transfer
        ),
        "cross_architecture_transfer_explicitly_authorized": (
            args.allow_cross_architecture_transfer
        ),
        "generator_model": args.generator_model,
        "tasks": list(args.tasks),
        "quality_cohort": args.quality_cohort,
        "excluded_response_qualities": sorted(
            excluded_qualities_from_args(args)
        ),
        "transformers_version": transformers_version,
        "torch_version": torch_version,
        "tokenizers_version": tokenizers_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_version": cuda_version,
        "cuda_device_name": cuda_device_name,
        "transformers_model_source": runtime_transformers_source(model_type),
        "evaluation_code": evaluation_code_manifest,
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "logit_chunk_size": args.logit_chunk_size,
        "cosine_chunk_size": args.cosine_chunk_size,
        "prompt_char_limit": (
            None
            if args.no_prompt_truncation
            else args.prompt_char_limit
        ),
        "system_prompt": args.system_prompt,
        "chat_template_kwargs": (
            {"enable_thinking": False}
            if model_type == "qwen3"
            else {}
        ),
        "context_scope": "entire_chat_prefix_as_released",
        "context_top_fraction": args.context_top_fraction,
        "candidate_heads_source": (
            "released Meta-Llama-3-8B-Instruct candidate list"
        ),
        "candidate_heads_sha256": sha256_file(
            args.candidate_heads_path
        ),
        "candidate_heads": [list(pair) for pair in candidate_heads],
        "parametric_score": (
            "released KL(M||P)+KL(M||Q), vocabulary mean, x1e6"
        ),
        "causal_alignment": (
            "state at position i scores the fixed response token at i+1"
        ),
        "span_alignment": "half-open character/token overlap",
        "special_and_prompt_tokens_evaluated": False,
        "split_separation": dict(split_validation),
        "target_test_labels_used_for_configuration": not (
            split_validation.get("response_ids_disjoint") is True
            and split_validation.get("source_ids_disjoint") is True
        ),
        "fixed_hyperparameters_are_benchmark_informed": True,
        "configuration_provenance": (
            "published frozen Llama-3 artifact"
            if args.configuration_mode == "released"
            else (
                "published counts and beta; target-checkpoint components "
                "and Min-Max ranges fitted on the disjoint train split"
            )
        ),
        "calibration_split": (
            args.calibration_split
            if args.configuration_mode == "train-transfer"
            else None
        ),
        "test_split": args.test_split,
        "selection_unit": (
            args.selection_unit
            if args.configuration_mode == "train-transfer"
            else None
        ),
        "head_count_grid": (
            list(args.head_counts)
            if args.configuration_mode == "train-transfer"
            else None
        ),
        "layer_count_grid": (
            list(args.layer_counts)
            if args.configuration_mode == "train-transfer"
            else None
        ),
        "beta_grid": (
            list(args.beta_values)
            if args.configuration_mode == "train-transfer"
            else None
        ),
        "released_configuration": (
            released_configuration.manifest()
            if released_configuration is not None
            else None
        ),
    }
    protocol["fingerprint"] = protocol_fingerprint(protocol)
    return protocol


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"unsupported dtype {args.dtype}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        local_files_only=not args.allow_network,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer is required for offset alignment")
    device_map: Any
    if args.device.startswith("cuda"):
        device_map = {"": args.device}
    else:
        device_map = None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=args.attention_implementation,
        low_cpu_mem_usage=True,
        local_files_only=not args.allow_network,
    )
    if device_map is None:
        model.to(args.device)
    model.eval()
    model.config.use_cache = False
    model_type = str(getattr(model.config, "model_type", "")).lower()
    if model_type not in {"llama", "qwen3"}:
        raise ValueError(
            f"unsupported detector architecture {model_type!r}"
        )
    return model, tokenizer, model_type, transformers.__version__


def extract_records(
    *,
    samples: Sequence[Mapping[str, Any]],
    feature_directory: Path,
    extractor: Any,
    tokenizer: Any,
    protocol: Mapping[str, Any],
    overwrite: bool,
    print_every: int,
) -> dict[str, Any]:
    import torch

    started = time.time()
    extracted = 0
    resumed = 0
    token_total = 0
    maximum_sequence_length = 0
    for index, sample in enumerate(samples, start=1):
        destination = feature_path(feature_directory, str(sample["id"]))
        if destination.exists() and not overwrite:
            existing = read_feature_record(
                destination,
                expected_id=str(sample["id"]),
                expected_fingerprint=str(protocol["fingerprint"]),
            )
            token_total += len(existing["labels"])
            maximum_sequence_length = max(
                maximum_sequence_length,
                int(existing["sequence_length"]),
            )
            resumed += 1
        else:
            encoding = build_teacher_forced_encoding(
                tokenizer,
                prompt=str(sample["prompt"]),
                response=str(sample["response"]),
                labels=sample["labels"],
                prompt_char_limit=protocol["prompt_char_limit"],
                system_prompt=str(protocol["system_prompt"]),
                chat_template_kwargs=protocol["chat_template_kwargs"],
            )
            if max(encoding["token_labels"], default=0) != int(
                sample["response_label"]
            ):
                raise ValueError(
                    f"response {sample['id']} lost a hallucination span "
                    "during token alignment"
                )
            features = extractor.extract(
                input_ids=encoding["input_ids"],
                prefix_length=len(encoding["prefix_ids"]),
                score_positions=encoding["score_positions"],
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "protocol_fingerprint": protocol["fingerprint"],
                "id": str(sample["id"]),
                "source_id": str(sample["source_id"]),
                "split": str(sample["split"]),
                "task_type": str(sample["task_type"]),
                "generator_model": str(sample["model"]),
                "response_label": int(sample["response_label"]),
                "sequence_length": len(encoding["input_ids"]),
                "prefix_length": len(encoding["prefix_ids"]),
                "boundary_retokenized": bool(
                    encoding["boundary_retokenized"]
                ),
                "response_token_ids": [
                    encoding["input_ids"][position]
                    for position in encoding["response_token_positions"]
                ],
                "response_offsets": [
                    list(offset) for offset in encoding["response_offsets"]
                ],
                "score_positions": encoding["score_positions"],
                "labels": encoding["token_labels"],
                "external": features["external"],
                "parametric": features["parametric"],
            }
            write_feature_record(feature_directory, record)
            token_total += len(record["labels"])
            maximum_sequence_length = max(
                maximum_sequence_length,
                int(record["sequence_length"]),
            )
            extracted += 1
            del encoding, features, record

        if (
            index == 1
            or index == len(samples)
            or index % print_every == 0
        ):
            print(
                f"[{index}/{len(samples)}] split={sample['split']} "
                f"id={sample['id']} extracted={extracted} resumed={resumed}",
                flush=True,
            )
        if index % print_every == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {
        "num_responses": len(samples),
        "num_tokens": token_total,
        "extracted": extracted,
        "resumed": resumed,
        "maximum_sequence_length": maximum_sequence_length,
        "elapsed_seconds": time.time() - started,
    }


def read_records_for_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    feature_directory: Path,
    fingerprint: str,
) -> list[dict[str, Any]]:
    records = []
    for sample in samples:
        path = feature_path(feature_directory, str(sample["id"]))
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run --mode full to finish extraction"
            )
        records.append(
            read_feature_record(
                path,
                expected_id=str(sample["id"]),
                expected_fingerprint=fingerprint,
            )
        )
    return records


def write_predictions(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with gzip.open(
            temporary,
            mode="wt",
            encoding="utf-8",
            newline="\n",
        ) as file:
            for record in records:
                value = {
                    key: record[key]
                    for key in (
                        "id",
                        "source_id",
                        "task_type",
                        "response_offsets",
                        "response_token_ids",
                        "labels",
                        "scores",
                    )
                }
                file.write(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate(
    *,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    released_configuration: ReleasedTokenConfiguration | None,
    calibration_samples: Sequence[Mapping[str, Any]] | None,
    test_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    test_records = read_records_for_samples(
        test_samples,
        feature_directory=args.output_dir / "features" / args.test_split,
        fingerprint=str(protocol["fingerprint"]),
    )
    if args.configuration_mode == "released":
        if released_configuration is None:
            raise ValueError(
                "released mode requires the frozen official configuration"
            )
        if calibration_samples is not None:
            raise ValueError(
                "released mode must not receive calibration samples"
            )
        scored = score_rows_with_released_config(
            test_records,
            feature_heads=protocol["candidate_heads"],
            config=released_configuration,
        )
        configuration_summary = {
            **released_configuration.manifest(),
            "calibration_labels_used": False,
        }
    else:
        if calibration_samples is None:
            raise ValueError(
                "train-transfer mode requires calibration samples"
            )
        calibration_records = read_records_for_samples(
            calibration_samples,
            feature_directory=args.output_dir
            / "features"
            / args.calibration_split,
            fingerprint=str(protocol["fingerprint"]),
        )
        calibration = calibrate_redeep(
            calibration_records,
            candidate_heads=protocol["candidate_heads"],
            selection_unit=args.selection_unit,
            head_counts=args.head_counts,
            layer_counts=args.layer_counts,
            beta_values=args.beta_values,
        )
        _write_json_atomic(
            args.output_dir / "calibration.json", calibration
        )
        del calibration_records
        gc.collect()
        scored = apply_calibration(test_records, calibration)
        configuration_summary = {
            "configuration_mode": "train_transfer",
            "selection_unit": calibration["selection_unit"],
            "selected_heads": calibration["selected_heads"],
            "selected_layers": calibration["selected_layers"],
            "beta": calibration["beta"],
            "calibration_auroc": calibration["calibration_auroc"],
            "calibration_labels_used": True,
            "calibration_split": args.calibration_split,
        }
    token_metrics = compute_token_metrics(scored, tasks=args.tasks)
    response_metrics = compute_response_metrics(scored, tasks=args.tasks)
    results = {
        "token": token_metrics,
        "response_sanity": response_metrics,
        "configuration": configuration_summary,
        "protocol": {
            **protocol,
            "calibration_response_count": (
                len(calibration_samples)
                if calibration_samples is not None
                else 0
            ),
            "test_response_count": len(test_samples),
            "test_response_ids_sha256": response_ids_sha256(
                [str(sample["id"]) for sample in test_samples]
            ),
            "responses_per_task": {
                task: sum(
                    str(sample["task_type"]) == task
                    for sample in test_samples
                )
                for task in args.tasks
            },
        },
    }
    _write_json_atomic(args.output_dir / "results.json", results)
    write_predictions(args.output_dir / "predictions.jsonl.gz", scored)
    table = format_markdown_metrics(token_metrics)
    report = "\n".join(
        [
            "# ReDeEP-Token fixed-response evaluation",
            "",
            table,
            "",
            (
                "The table reports token AUROC on fixed RAGTruth responses. "
                "It is not the response-level AUC reported in the ReDeEP paper."
            ),
            "",
            (
                "The released Meta-Llama-3-8B Copying-Head candidate set is "
                "transferred to this detector checkpoint"
                + (
                    " and architecture."
                    if protocol["cross_architecture_transfer"]
                    else "."
                )
            ),
            "",
            (
                "The released configuration is frozen and no evaluation "
                "label is used for component selection or normalization."
                if args.configuration_mode == "released"
                else (
                    "The benchmark-informed component counts and beta are "
                    "fixed; component ranking and ranges use only "
                    f"{args.calibration_split}. No target test-response "
                    "label is used."
                )
            ),
            "",
        ]
    )
    (args.output_dir / "final_report.md").write_text(
        report,
        encoding="utf-8",
    )
    print(table, flush=True)
    print(
        json.dumps(
            {
                **configuration_summary,
                "response_sanity_pooled_auroc": response_metrics["overall"][
                    "auroc"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the released ReDeEP token score with Llama-3.1 on "
            "fixed Llama-2 RAGTruth responses."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full", "evaluate"),
        default="full",
    )
    parser.add_argument("--model-name-or-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--configuration-mode",
        choices=("released", "train-transfer"),
        default="train-transfer",
        help=(
            "train-transfer (default for new checkpoints) fixes the released "
            "component counts and beta, then selects components and ranges on "
            "RAGTruth train only. The fixed values are benchmark-informed, "
            "but no target test-response label is used. released applies the "
            "exact published Meta-Llama-3 artifact and is only a frozen-"
            "configuration check"
        ),
    )
    parser.add_argument(
        "--allow-checkpoint-transfer",
        action="store_true",
        help=(
            "explicitly authorize transfer of the released Meta-Llama-3-8B "
            "Copying-Head candidates to a different detector checkpoint"
        ),
    )
    parser.add_argument(
        "--allow-cross-architecture-transfer",
        action="store_true",
        help=(
            "explicitly allow the released Llama Copying-Head candidates "
            "(and, in released mode, the full configuration) to be "
            "transferred to Qwen3; this is not an official Qwen result"
        ),
    )
    parser.add_argument(
        "--candidate-heads-path",
        type=Path,
        default=(
            repository_root
            / "ReDeEP"
            / "log"
            / "test_llama3_8B"
            / "topk_heads.json"
        ),
        help=(
            "Path to an exact byte-identical copy of the released "
            "Meta-Llama-3 candidate-head artifact; its SHA-256 is enforced"
        ),
    )
    parser.add_argument(
        "--generator-model",
        default="llama-2-7b-chat",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
    )
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--quality-cohort",
        choices=("all", "quality-good"),
        default="all",
        help=(
            "Use the complete official split (default) or explicitly exclude "
            "incorrect_refusal and truncated responses."
        ),
    )
    parser.add_argument(
        "--selection-unit",
        choices=("token", "response"),
        default="token",
    )
    parser.add_argument("--head-counts", nargs="+", type=int, default=[3])
    parser.add_argument("--layer-counts", nargs="+", type=int, default=[30])
    parser.add_argument("--beta-values", nargs="+", type=float, default=[0.4])
    parser.add_argument(
        "--context-top-fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
    )
    parser.add_argument(
        "--prompt-char-limit",
        type=int,
        default=12_000,
        help="Released ReDeEP truncates the raw user prompt to 12,000 chars.",
    )
    parser.add_argument(
        "--no-prompt-truncation",
        action="store_true",
        help=(
            "Explicit protocol adaptation that scores the full prompt instead "
            "of the released 12,000-character slice."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "eager"),
        default="eager",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--logit-chunk-size", type=int, default=32)
    parser.add_argument("--cosine-chunk-size", type=int, default=16)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository_root = Path(__file__).resolve().parent
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")
    if args.prompt_char_limit <= 0:
        raise ValueError("--prompt-char-limit must be positive")
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.model_name_or_path = args.model_name_or_path.resolve()
    args.candidate_heads_path = args.candidate_heads_path.resolve()
    ragtruth_hashes = verify_ragtruth(args.data_dir)
    candidate_heads = load_candidate_heads(
        args.candidate_heads_path,
        expected_sha256=OFFICIAL_CANDIDATE_HEADS_SHA256,
    )
    model_type = detector_model_type(args.model_name_or_path)
    validate_candidate_transfer(
        model_type=model_type,
        allow_checkpoint_transfer=args.allow_checkpoint_transfer,
        allow_cross_architecture_transfer=(
            args.allow_cross_architecture_transfer
        ),
    )
    released_configuration = (
        load_released_llama3_token_config(
            repository_root,
            target_architecture=model_type,
            allow_cross_architecture_transfer=(
                args.allow_cross_architecture_transfer
            ),
        )
        if args.configuration_mode == "released"
        else None
    )
    test_samples = load_ragtruth_samples(
        data_dir=args.data_dir,
        split=args.test_split,
        generator_model=args.generator_model,
        tasks=args.tasks,
        excluded_qualities=excluded_qualities_from_args(args),
    )
    if args.configuration_mode == "train-transfer":
        calibration_samples: Sequence[Mapping[str, Any]] | None = (
            load_ragtruth_samples(
                data_dir=args.data_dir,
                split=args.calibration_split,
                generator_model=args.generator_model,
                tasks=args.tasks,
                excluded_qualities=excluded_qualities_from_args(args),
            )
        )
        split_validation = validate_split_separation(
            calibration_split=args.calibration_split,
            test_split=args.test_split,
            calibration_samples=calibration_samples,
            test_samples=test_samples,
        )
    else:
        calibration_samples = None
        split_validation = {
            "calibration_split": None,
            "test_split": args.test_split,
            "response_ids_disjoint": True,
            "source_ids_disjoint": True,
            "calibration_response_count": 0,
            "test_response_count": len(test_samples),
            "not_applicable_reason": (
                "the frozen released configuration reads no calibration labels"
            ),
        }
    import tokenizers
    import torch
    import transformers

    protocol = build_protocol(
        args,
        candidate_heads=candidate_heads,
        model_type=model_type,
        released_configuration=released_configuration,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        tokenizers_version=tokenizers.__version__,
        cuda_version=torch.version.cuda,
        cuda_device_name=(
            torch.cuda.get_device_name(args.device)
            if str(args.device).startswith("cuda")
            and torch.cuda.is_available()
            else None
        ),
        ragtruth_hashes=ragtruth_hashes,
        split_validation=split_validation,
    )

    if args.mode == "evaluate":
        protocol_path = args.output_dir / "protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError(protocol_path)
        saved_protocol = json.loads(
            protocol_path.read_text(encoding="utf-8")
        )
        if saved_protocol.get("fingerprint") != protocol["fingerprint"]:
            raise ValueError(
                "evaluate arguments, code, dataset, tokenizer, or checkpoint "
                "do not match the saved extraction protocol"
            )
    else:
        (
            model,
            tokenizer,
            loaded_model_type,
            transformers_version,
        ) = load_model_and_tokenizer(args)
        if loaded_model_type != model_type:
            raise ValueError(
                "checkpoint config changed while loading the model"
            )
        if transformers_version != protocol["transformers_version"]:
            raise ValueError("Transformers version changed while loading")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        existing_protocol_path = args.output_dir / "protocol.json"
        if existing_protocol_path.exists() and not args.overwrite:
            existing_protocol = json.loads(
                existing_protocol_path.read_text(encoding="utf-8")
            )
            if existing_protocol.get("fingerprint") != protocol["fingerprint"]:
                raise ValueError(
                    "output directory belongs to a different protocol; "
                    "choose another directory or pass --overwrite"
                )
        _write_json_atomic(existing_protocol_path, protocol)

        from redeep_token.scoring import ReDeEPFeatureExtractor

        extractor = ReDeEPFeatureExtractor(
            model,
            candidate_heads=candidate_heads,
            top_fraction=args.context_top_fraction,
            logit_chunk_size=args.logit_chunk_size,
            cosine_chunk_size=args.cosine_chunk_size,
        )
        if args.mode == "smoke":
            smoke_samples = select_longest_per_task(
                test_samples, args.tasks
            )
            summary = extract_records(
                samples=smoke_samples,
                feature_directory=args.output_dir / "smoke",
                extractor=extractor,
                tokenizer=tokenizer,
                protocol=protocol,
                overwrite=args.overwrite,
                print_every=1,
            )
            _write_json_atomic(args.output_dir / "smoke.json", summary)
            print(json.dumps(summary, indent=2), flush=True)
            return

        extraction = {}
        if args.configuration_mode == "train-transfer":
            assert calibration_samples is not None
            extraction[args.calibration_split] = extract_records(
                samples=calibration_samples,
                feature_directory=args.output_dir
                / "features"
                / args.calibration_split,
                extractor=extractor,
                tokenizer=tokenizer,
                protocol=protocol,
                overwrite=args.overwrite,
                print_every=args.print_every,
            )
        extraction[args.test_split] = extract_records(
            samples=test_samples,
            feature_directory=args.output_dir
            / "features"
            / args.test_split,
            extractor=extractor,
            tokenizer=tokenizer,
            protocol=protocol,
            overwrite=args.overwrite,
            print_every=args.print_every,
        )
        _write_json_atomic(args.output_dir / "extraction.json", extraction)
        del extractor, model, tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    evaluate(
        args=args,
        protocol=protocol,
        released_configuration=released_configuration,
        calibration_samples=calibration_samples,
        test_samples=test_samples,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "Interrupted; completed feature files remain resumable.",
            file=sys.stderr,
        )
        raise SystemExit(130)
        raise
