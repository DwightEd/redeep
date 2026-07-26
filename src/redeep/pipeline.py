"""End-to-end orchestration for resumable remote ReDeEP experiments.

Heavy dependencies are imported inside the functions that need them.  This
keeps configuration, ``doctor``, and dataset auditing usable on login nodes
without importing PyTorch or Transformers.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

from .alignment import align_teacher_forced_example
from .config import ExperimentConfig, ModelConfig
from .data import (
    EXPECTED_LLAMA2_COUNTS,
    OFFICIAL_RESPONSE_SHA256,
    OFFICIAL_SOURCE_SHA256,
    assert_official_llama2_counts,
    dataset_counts,
    deterministic_calibration_dev_split,
    load_ragtruth,
)
from .schemas import RagTruthExample
from .utils import (
    environment_snapshot,
    git_commit,
    seed_everything,
    sha256_file,
    sha256_file_canonical_newlines,
    sha256_json,
)

FEATURE_SCHEMA_VERSION = 1
EVAL_SPLITS = ("calibration_train", "dev", "test")
_DEPENDENCY_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_CALIBRATION_DEPENDENCY_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def experiment_dir(config: ExperimentConfig) -> Path:
    """Return the directory containing all artifacts for one experiment."""

    return config.output_dir / config.name


def model_dir(config: ExperimentConfig, model_key: str) -> Path:
    _require_model_key(config, model_key)
    return experiment_dir(config) / model_key


def feature_dir(config: ExperimentConfig, model_key: str, eval_split: str) -> Path:
    if eval_split not in EVAL_SPLITS:
        raise ValueError(f"eval_split must be one of {EVAL_SPLITS}, got {eval_split!r}")
    return model_dir(config, model_key) / "features" / eval_split


def feature_shard_path(
    config: ExperimentConfig,
    model_key: str,
    eval_split: str,
    response_id: str,
) -> Path:
    """Return a stable, filesystem-safe one-response Parquet path."""

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(response_id)).strip("._")
    safe = safe or "response"
    if safe != str(response_id):
        suffix = hashlib.sha256(str(response_id).encode()).hexdigest()[:10]
        safe = f"{safe}-{suffix}"
    return feature_dir(config, model_key, eval_split) / f"{safe}.parquet"


def load_experiment_examples(
    config: ExperimentConfig,
) -> tuple[RagTruthExample, ...]:
    """Load the complete fixed-generator subset described by the config."""

    return load_ragtruth(
        config.dataset.response_path,
        config.dataset.source_path,
        generator_model=config.dataset.generator_model,
        tasks=config.dataset.tasks,
    )


def partition_examples(
    config: ExperimentConfig,
    examples: Iterable[RagTruthExample],
) -> dict[str, tuple[RagTruthExample, ...]]:
    """Construct leakage-safe calibration-train/dev/test partitions."""

    examples = tuple(examples)
    train = tuple(example for example in examples if example.split == "train")
    test = tuple(example for example in examples if example.split == "test")
    calibration_train, dev = deterministic_calibration_dev_split(
        train,
        dev_per_task=config.dataset.dev_per_task,
        seed=config.dataset.split_seed,
    )
    if not config.dataset.include_truncated_test:
        test = tuple(example for example in test if example.quality == "good")
    return {
        "calibration_train": calibration_train,
        "dev": dev,
        "test": test,
    }


def audit_dataset(
    config: ExperimentConfig,
    *,
    strict_official_counts: bool = True,
) -> dict[str, Any]:
    """Validate the complete dataset contract and return machine-readable stats."""

    examples = load_experiment_examples(config)
    observed_hashes = {
        "response": sha256_file(config.dataset.response_path),
        "source": sha256_file(config.dataset.source_path),
    }
    canonical_hashes = {
        "response": sha256_file_canonical_newlines(config.dataset.response_path),
        "source": sha256_file_canonical_newlines(config.dataset.source_path),
    }
    if strict_official_counts:
        expected_hashes = {
            "response": OFFICIAL_RESPONSE_SHA256,
            "source": OFFICIAL_SOURCE_SHA256,
        }
        if canonical_hashes != expected_hashes:
            raise ValueError(
                "RAGTruth files do not content-match the pinned official release "
                "after newline normalization: "
                f"canonical={canonical_hashes}, expected={expected_hashes}"
            )
        if config.dataset.generator_model != "llama-2-7b-chat":
            raise ValueError("Official-count validation requires llama-2-7b-chat")
        if set(config.dataset.tasks) != set(EXPECTED_LLAMA2_COUNTS["train"]):
            raise ValueError("Official-count validation requires all three RAGTruth tasks")
        assert_official_llama2_counts(examples)

    partitions = partition_examples(config, examples)
    dev_ids = sorted(example.response_id for example in partitions["dev"])
    task_stats: dict[str, dict[str, dict[str, int]]] = {}
    for split in ("train", "test"):
        task_stats[split] = {}
        for task in config.dataset.tasks:
            selected = [
                example
                for example in examples
                if example.split == split and example.task == task
            ]
            task_stats[split][task] = {
                "responses": len(selected),
                "positive_responses": sum(example.response_label for example in selected),
                "merged_spans": sum(len(example.spans) for example in selected),
                "truncated": sum(example.quality != "good" for example in selected),
            }

    return {
        "ok": True,
        "generator_model": config.dataset.generator_model,
        "examples": len(examples),
        "counts": dataset_counts(examples),
        "task_stats": task_stats,
        "partitions": {key: len(value) for key, value in partitions.items()},
        "dev_per_task": dict(Counter(example.task for example in partitions["dev"])),
        "dev_response_ids_sha256": hashlib.sha256(
            "\n".join(dev_ids).encode()
        ).hexdigest(),
        "dataset_sha256": observed_hashes,
        "dataset_canonical_newline_sha256": canonical_hashes,
    }


def pending_examples(
    config: ExperimentConfig,
    model_key: str,
    eval_split: str,
    examples: Iterable[RagTruthExample],
    *,
    force: bool = False,
    copy_heads_hash: str | None = None,
) -> tuple[RagTruthExample, ...]:
    """Return examples whose per-response feature shard is not complete."""

    examples = tuple(examples)
    if force:
        return examples
    from .artifacts import shard_is_complete

    return tuple(
        example
        for example in examples
        if not shard_is_complete(
            feature_shard_path(
                config, model_key, eval_split, example.response_id
            ),
            expected_manifest=_shard_manifest(
                config,
                model_key,
                eval_split,
                example.response_id,
                copy_heads_hash=copy_heads_hash,
            ),
        )
    )


def select_work_shard(
    examples: Sequence[RagTruthExample],
    *,
    num_shards: int = 1,
    shard_index: int = 0,
) -> tuple[RagTruthExample, ...]:
    """Deterministically assign ordered examples with ``items[index::count]``."""

    if num_shards < 1:
        raise ValueError("num_shards must be at least one")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return tuple(examples[shard_index::num_shards])


def discover_heads(
    config: ExperimentConfig,
    model_key: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Discover and persist structural Copying Head candidates."""

    model_config = _require_model_key(config, model_key)
    dependencies = _experiment_dependencies(config, model_key)
    dependencies_sha256 = sha256_json(dependencies)
    output_path = model_dir(config, model_key) / "copy_heads.json"
    if output_path.is_file() and not force:
        cached = _read_json(output_path)
        if (
            cached.get("schema_version") != FEATURE_SCHEMA_VERSION
            or cached.get("config_hash") != config.digest
            or cached.get("dependencies_sha256") != dependencies_sha256
        ):
            raise ValueError(
                f"Cached Copying Head artifact {output_path} does not match the "
                "current code/config/model; rerun discover-heads with --force"
            )
        return cached

    seed_everything(config.seed)
    model, _tokenizer = _load_model_and_tokenizer(model_config, tokenizer=False)
    try:
        from .copy_heads import discover_copy_heads
        from .models import DecoderModelAdapter

        adapter = DecoderModelAdapter.from_model(model)
        discovery = discover_copy_heads(
            adapter,
            top_k=config.extraction.copying_candidate_count,
            gershgorin_sample_size=config.extraction.copying_vocab_sample_size,
            seed=config.extraction.copying_seed,
        )
        payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "artifact": str(output_path),
            "model_key": model_key,
            "model_name": model_config.name,
            "model_path": str(model_config.path),
            "model_config_sha256": _optional_file_hash(
                model_config.path / "config.json"
            ),
            "top_heads": [list(item) for item in discovery.top_heads],
            "records": discovery.to_rows(),
            "metadata": _jsonable(discovery.metadata),
            "config_hash": config.digest,
            "git_commit": _current_git_commit(),
            "dependencies": dependencies,
            "dependencies_sha256": dependencies_sha256,
            "environment": environment_snapshot(),
            "model_runtime": _model_runtime_metadata(model),
        }
        _atomic_write_json(output_path, payload)
        return payload
    finally:
        del model
        _release_accelerator_memory()


def extract_features(
    config: ExperimentConfig,
    model_key: str,
    eval_split: str,
    *,
    force: bool = False,
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    """Extract one resumable Parquet feature shard per response."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    model_config = _require_model_key(config, model_key)
    examples = load_experiment_examples(config)
    all_selected = partition_examples(config, examples)[eval_split]
    selected = select_work_shard(
        all_selected,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    heads_payload = _load_copy_heads(config, model_key)
    copy_heads_hash = sha256_json(heads_payload)
    work = pending_examples(
        config,
        model_key,
        eval_split,
        selected,
        force=force,
        copy_heads_hash=copy_heads_hash,
    )
    if limit is not None:
        work = work[:limit]

    copying_heads = tuple(
        (int(item[0]), int(item[1])) for item in heads_payload["top_heads"]
    )
    if not copying_heads:
        raise ValueError("Copying Head artifact contains no candidates")

    target_dir = feature_dir(config, model_key, eval_split)
    target_dir.mkdir(parents=True, exist_ok=True)
    if not work:
        return {
            "model_key": model_key,
            "eval_split": eval_split,
            "total_selected": len(all_selected),
            "assigned": len(selected),
            "extracted": 0,
            "skipped": len(selected),
            "num_shards": num_shards,
            "shard_index": shard_index,
            "feature_dir": str(target_dir),
        }

    seed_everything(config.seed)
    model, tokenizer = _load_model_and_tokenizer(model_config, tokenizer=True)
    try:
        import torch

        from .features import extract_token_features
        from .models import DecoderModelAdapter

        adapter = DecoderModelAdapter.from_model(model)
        model_runtime = _model_runtime_metadata(model)
        input_device = model.get_input_embeddings().weight.device
        completed = 0
        for index, example in enumerate(work, start=1):
            aligned = align_teacher_forced_example(
                example,
                tokenizer,
                model_family=model_config.family,
                system_prompt=config.dataset.system_prompt,
            )
            maximum_length = model_config.max_length or getattr(
                model.config,
                "max_position_embeddings",
                None,
            )
            if maximum_length is not None and len(aligned.input_ids) > maximum_length:
                raise ValueError(
                    f"Response {example.response_id} renders to "
                    f"{len(aligned.input_ids)} tokens, exceeding {maximum_length}"
                )
            input_ids = torch.tensor(
                [aligned.input_ids],
                dtype=torch.long,
                device=input_device,
            )
            attention_mask = torch.tensor(
                [aligned.attention_mask],
                dtype=torch.long,
                device=input_device,
            )
            whole_prefix_positions = (
                tuple(
                    position
                    for position, attended in enumerate(aligned.attention_mask)
                    if attended
                    and position < min(aligned.response_token_positions)
                )
                if config.extraction.include_legacy_whole_prefix_ecs
                else None
            )
            with torch.inference_mode():
                batch = extract_token_features(
                    adapter,
                    input_ids,
                    attention_mask,
                    aligned.predictor_positions,
                    aligned.context_token_positions,
                    copying_heads,
                    pks_layers=None,
                    pks_modes=config.extraction.jsd_modes,
                    top_fraction=config.extraction.context_top_fraction,
                    vocab_chunk_size=config.extraction.vocab_chunk_size,
                    token_chunk_size=config.extraction.token_chunk_size,
                    whole_prefix_positions=whole_prefix_positions,
                    offload_to_cpu=config.extraction.capture_to_cpu,
                )
            rows = _feature_rows(aligned, batch.columns, eval_split=eval_split)
            output_path = feature_shard_path(
                config, model_key, eval_split, example.response_id
            )
            _write_feature_shard(
                rows,
                output_path,
                manifest=_shard_manifest(
                    config,
                    model_key,
                    eval_split,
                    example.response_id,
                    copy_heads_hash=copy_heads_hash,
                ),
            )
            completed += 1
            if index == 1 or index % 10 == 0 or index == len(work):
                print(
                    f"[{model_key}/{eval_split}] {index}/{len(work)} "
                    f"response_id={example.response_id}",
                    flush=True,
                )
            del batch, rows, input_ids, attention_mask

        _write_feature_manifest(
            config,
            model_key,
            eval_split,
            heads_payload,
            model_runtime=model_runtime,
        )
        return {
            "model_key": model_key,
            "eval_split": eval_split,
            "total_selected": len(all_selected),
            "assigned": len(selected),
            "extracted": completed,
            "skipped": len(selected) - len(work),
            "num_shards": num_shards,
            "shard_index": shard_index,
            "feature_dir": str(target_dir),
        }
    finally:
        del model
        _release_accelerator_memory()


def calibrate_model(config: ExperimentConfig, model_key: str) -> dict[str, Any]:
    """Fit and freeze ReDeEP calibration using calibration-train and dev only."""

    _require_model_key(config, model_key)
    train_paths = _feature_paths(config, model_key, "calibration_train")
    dev_paths = _feature_paths(config, model_key, "dev")
    _require_nonempty_shards(train_paths, "calibration_train")
    _require_nonempty_shards(dev_paths, "dev")
    calibration_dependencies = _calibration_dependencies(
        config,
        model_key,
        train_paths=train_paths,
        dev_paths=dev_paths,
    )
    calibration_dependencies_sha256 = sha256_json(calibration_dependencies)

    from .calibration import search_calibration

    train_frame = _load_feature_frames(train_paths)
    dev_frame = _load_feature_frames(dev_paths)
    mode_reports: dict[str, Any] = {}
    for mode in _calibration_modes(config):
        result = search_calibration(
            train_frame,
            dev_frame,
            model_name=config.models[model_key].name,
            mode=mode,
            tasks=config.dataset.tasks,
            head_grid=range(
                config.calibration.head_k_min,
                config.calibration.head_k_max + 1,
            ),
            layer_grid=range(
                config.calibration.layer_k_min,
                config.calibration.layer_k_max + 1,
            ),
            beta_grid=_float_grid(
                config.calibration.beta_min,
                config.calibration.beta_max,
                config.calibration.beta_step,
            ),
            alpha=config.calibration.alpha,
        )
        output_path = _calibration_path(config, model_key, mode)
        grid_path = model_dir(config, model_key) / f"calibration_{mode}_grid.parquet"
        _atomic_write_frame(result.grid_results, grid_path)
        payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "model_key": model_key,
            "mode": mode,
            "primary": mode == config.extraction.primary_jsd_mode,
            "config_hash": config.digest,
            "git_commit": _current_git_commit(),
            "dependencies": calibration_dependencies,
            "dependencies_sha256": calibration_dependencies_sha256,
            "calibration": result.frozen.to_dict(),
            "grid_results": str(grid_path),
            "grid_results_sha256": sha256_file(grid_path),
            "grid_rows": len(result.grid_results),
        }
        _atomic_write_json(output_path, payload)
        if mode == config.extraction.primary_jsd_mode:
            _atomic_write_json(model_dir(config, model_key) / "calibration.json", payload)
        mode_reports[mode] = {
            "artifact": str(output_path),
            "grid_results": str(grid_path),
            "k_heads": result.frozen.k_heads,
            "k_layers": result.frozen.k_layers,
            "beta": result.frozen.beta,
            "selection_value": result.frozen.selection_value,
            "dependencies_sha256": calibration_dependencies_sha256,
        }
    return {
        "model_key": model_key,
        "primary_mode": config.extraction.primary_jsd_mode,
        "modes": mode_reports,
    }


def evaluate_model(config: ExperimentConfig, model_key: str) -> dict[str, Any]:
    """Apply frozen calibration to test shards and write JSON/CSV/Markdown."""

    _require_model_key(config, model_key)
    test_paths = _feature_paths(config, model_key, "test")
    _require_nonempty_shards(test_paths, "test")
    from .evaluation import evaluate_model as compute_metrics

    test_frame = _load_feature_frames(test_paths)
    frames = []
    for mode in _calibration_modes(config):
        scored = _score_test_frame(config, model_key, test_frame, mode)
        main_result = compute_metrics(
            scored,
            model_name=config.models[model_key].name,
            mode=mode,
            score_column="redeep_score",
            tasks=config.dataset.tasks,
            n_bootstrap=config.evaluation.bootstrap_samples,
            seed=config.evaluation.bootstrap_seed,
            confidence=config.evaluation.confidence_level,
        )
        main_result.insert(0, "subset", "all")
        frames.append(main_result)
        if config.evaluation.include_quality_good_sensitivity:
            quality_good = scored.loc[scored["quality"] == "good"].copy()
            sensitivity = compute_metrics(
                quality_good,
                model_name=config.models[model_key].name,
                mode=mode,
                score_column="redeep_score",
                tasks=config.dataset.tasks,
                n_bootstrap=config.evaluation.bootstrap_samples,
                seed=config.evaluation.bootstrap_seed,
                confidence=config.evaluation.confidence_level,
            )
            sensitivity.insert(0, "subset", "quality_good")
            frames.append(sensitivity)
    import pandas as pd

    result = pd.concat(frames, ignore_index=True)
    results_dir = model_dir(config, model_key) / "results"
    paths = _write_results(
        result,
        results_dir,
        metadata=_result_metadata(
            config,
            model_key,
            test_paths=test_paths,
        ),
    )
    return {
        "model_key": model_key,
        "results_dir": str(results_dir),
        "artifacts": _jsonable(paths),
        "metric_rows": len(result),
    }


def compare_model_results(
    config: ExperimentConfig,
    first_key: str,
    second_key: str,
) -> dict[str, Any]:
    """Run paired response-cluster comparisons for two completed scorers."""

    _require_model_key(config, first_key)
    _require_model_key(config, second_key)
    if first_key == second_key:
        raise ValueError("Paired comparison requires two different models")

    import pandas as pd

    from .evaluation import compare_models_paired

    frames = []
    for mode in _calibration_modes(config):
        first = _load_scored_test(config, first_key, mode)
        second = _load_scored_test(config, second_key, mode)
        main = compare_models_paired(
            first,
            second,
            first_model=config.models[first_key].name,
            second_model=config.models[second_key].name,
            tasks=config.dataset.tasks,
            n_bootstrap=config.evaluation.bootstrap_samples,
            seed=config.evaluation.bootstrap_seed,
            confidence=config.evaluation.confidence_level,
        )
        main.insert(0, "mode", mode)
        main.insert(0, "subset", "all")
        frames.append(main)
        if config.evaluation.include_quality_good_sensitivity:
            sensitivity = compare_models_paired(
                first.loc[first["quality"] == "good"].copy(),
                second.loc[second["quality"] == "good"].copy(),
                first_model=config.models[first_key].name,
                second_model=config.models[second_key].name,
                tasks=config.dataset.tasks,
                n_bootstrap=config.evaluation.bootstrap_samples,
                seed=config.evaluation.bootstrap_seed,
                confidence=config.evaluation.confidence_level,
            )
            sensitivity.insert(0, "mode", mode)
            sensitivity.insert(0, "subset", "quality_good")
            frames.append(sensitivity)
    result = pd.concat(frames, ignore_index=True)
    output_dir = experiment_dir(config) / "comparisons" / f"{first_key}_vs_{second_key}"
    paths = _write_results(
        result,
        output_dir,
        metadata={
            "experiment": config.name,
            "cross_backbone": True,
            "generator_model": config.dataset.generator_model,
            "first_model": first_key,
            "second_model": second_key,
            "modes": list(_calibration_modes(config)),
            "config_hash": config.digest,
            "git_commit": _current_git_commit(),
            "first_dependencies": _experiment_dependencies(config, first_key),
            "second_dependencies": _experiment_dependencies(config, second_key),
            "first_test_feature_set": _feature_set_fingerprint(
                _feature_paths(config, first_key, "test")
            ),
            "second_test_feature_set": _feature_set_fingerprint(
                _feature_paths(config, second_key, "test")
            ),
        },
    )
    return {
        "first_model": first_key,
        "second_model": second_key,
        "metric_rows": len(result),
        "artifacts": paths,
    }


def run_all(
    config: ExperimentConfig,
    *,
    model_keys: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run discovery, all feature splits, calibration, and evaluation."""

    selected = tuple(model_keys) if model_keys else tuple(config.models)
    if len(set(selected)) != len(selected):
        raise ValueError("run-all model keys must not contain duplicates")
    report: dict[str, Any] = {}
    for model_key in selected:
        _require_model_key(config, model_key)
        model_report: dict[str, Any] = {}
        discovered = discover_heads(config, model_key, force=force)
        model_report["discover_heads"] = {
            "artifact": discovered.get("artifact"),
            "top_heads": discovered["top_heads"],
        }
        model_report["extract"] = {}
        for eval_split in EVAL_SPLITS:
            model_report["extract"][eval_split] = extract_features(
                config,
                model_key,
                eval_split,
                force=force,
            )
        model_report["calibrate"] = calibrate_model(config, model_key)
        model_report["evaluate"] = evaluate_model(config, model_key)
        report[model_key] = model_report
    if len(selected) > 1:
        report["comparisons"] = {
            f"{first}_vs_{second}": compare_model_results(config, first, second)
            for first, second in combinations(selected, 2)
        }
    return report


def _load_model_and_tokenizer(
    config: ModelConfig,
    *,
    tokenizer: bool,
) -> tuple[Any, Any | None]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if config.dtype not in dtype_by_name:
        raise ValueError(
            f"Unsupported dtype {config.dtype!r}; expected {tuple(dtype_by_name)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Copying Head discovery and feature extraction"
        )
    if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "The selected CUDA device does not support bfloat16; set the model "
            "dtype to float16 in the experiment config"
        )
    loaded_tokenizer = (
        AutoTokenizer.from_pretrained(
            config.tokenizer_path,
            trust_remote_code=config.trust_remote_code,
            local_files_only=True,
            use_fast=True,
        )
        if tokenizer
        else None
    )
    if loaded_tokenizer is not None and not loaded_tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for offset_mapping")
    model = AutoModelForCausalLM.from_pretrained(
        config.path,
        torch_dtype=dtype_by_name[config.dtype],
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    _validate_loaded_model(model, config)
    return model, loaded_tokenizer


def _feature_rows(
    aligned: Any,
    feature_columns: Mapping[str, Any],
    *,
    eval_split: str,
) -> dict[str, list[Any]]:
    """Combine feature tensors with the unified token-level metadata schema."""

    token_count = len(aligned.response_token_positions)
    example = aligned.example
    rows: dict[str, list[Any]] = {
        "response_id": [example.response_id] * token_count,
        "source_id": [example.source_id] * token_count,
        "generator_model": [example.generator_model] * token_count,
        "task": [example.task] * token_count,
        "quality": [example.quality] * token_count,
        "source_split": [example.split] * token_count,
        "original_split": [example.split] * token_count,
        "eval_split": [eval_split] * token_count,
        "response_label": [example.response_label] * token_count,
        "token_index": list(range(token_count)),
        "token_id": [
            aligned.input_ids[position]
            for position in aligned.response_token_positions
        ],
        "token_position": list(aligned.response_token_positions),
        "predictor_position": list(aligned.predictor_positions),
        "char_start": [item[0] for item in aligned.response_offsets],
        "char_end": [item[1] for item in aligned.response_offsets],
        "token_label": list(aligned.token_labels),
    }
    for name, tensor in sorted(feature_columns.items()):
        if name in rows:
            raise ValueError(f"Feature column collides with metadata column: {name}")
        values = _tensor_values(tensor)
        if len(values) != token_count:
            raise ValueError(
                f"Feature {name!r} has {len(values)} values; expected {token_count}"
            )
        rows[name] = values
    return rows


def _tensor_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = list(value)
    return [float(item) for item in value]


def _write_feature_shard(
    columns: Mapping[str, Sequence[Any]],
    path: Path,
    *,
    manifest: Mapping[str, Any],
) -> None:
    import pandas as pd

    from .artifacts import atomic_write_feature_shard

    frame = pd.DataFrame(columns)
    atomic_write_feature_shard(frame, path, manifest)


def _atomic_write_frame(frame: Any, path: Path) -> None:
    from .artifacts import atomic_write_parquet

    atomic_write_parquet(path, frame)


def _write_feature_manifest(
    config: ExperimentConfig,
    model_key: str,
    eval_split: str,
    heads_payload: Mapping[str, Any],
    *,
    model_runtime: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "model_key": model_key,
        "eval_split": eval_split,
        "config": config.to_serializable(),
        "config_hash": config.digest,
        "git_commit": _current_git_commit(),
        "dataset_sha256": {
            "response": sha256_file(config.dataset.response_path),
            "source": sha256_file(config.dataset.source_path),
        },
        "copy_heads_sha256": sha256_json(heads_payload),
        "model_config_sha256": _optional_file_hash(
            config.models[model_key].path / "config.json"
        ),
        "environment": environment_snapshot(),
        "model_runtime": _jsonable(model_runtime),
        "dependencies": _experiment_dependencies(config, model_key),
        "dependencies_sha256": sha256_json(
            _experiment_dependencies(config, model_key)
        ),
    }
    from .artifacts import write_manifest

    write_manifest(
        feature_dir(config, model_key, eval_split) / "manifest.json",
        payload,
    )


def _load_copy_heads(config: ExperimentConfig, model_key: str) -> dict[str, Any]:
    path = model_dir(config, model_key) / "copy_heads.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Copying Head artifact {path}; run discover-heads first"
        )
    payload = _read_json(path)
    if payload.get("config_hash") != config.digest:
        raise ValueError(
            "Copying Head artifact config hash differs from current configuration"
        )
    expected_dependencies = _experiment_dependencies(config, model_key)
    if (
        payload.get("dependencies_sha256")
        != sha256_json(expected_dependencies)
    ):
        raise ValueError(
            "Copying Head artifact dependencies differ from the current "
            "code/data/model/tokenizer"
        )
    expected_model_hash = _optional_file_hash(
        config.models[model_key].path / "config.json"
    )
    if payload.get("model_config_sha256") != expected_model_hash:
        raise ValueError(
            "Copying Head artifact model config hash differs from the current model"
        )
    return payload


def _feature_paths(
    config: ExperimentConfig,
    model_key: str,
    eval_split: str,
) -> tuple[Path, ...]:
    from .artifacts import shard_is_complete

    expected_examples = partition_examples(
        config,
        load_experiment_examples(config),
    )[eval_split]
    copy_heads_hash = sha256_json(_load_copy_heads(config, model_key))
    complete: list[Path] = []
    missing: list[str] = []
    for example in expected_examples:
        path = feature_shard_path(
            config,
            model_key,
            eval_split,
            example.response_id,
        )
        if shard_is_complete(
            path,
            expected_manifest=_shard_manifest(
                config,
                model_key,
                eval_split,
                example.response_id,
                copy_heads_hash=copy_heads_hash,
            ),
        ):
            complete.append(path)
        else:
            missing.append(example.response_id)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{model_key}/{eval_split} is incomplete: {len(missing)} of "
            f"{len(expected_examples)} response shards missing or invalid "
            f"(first: {preview})"
        )
    return tuple(complete)


def _require_nonempty_shards(paths: Sequence[Path], eval_split: str) -> None:
    if not paths:
        raise FileNotFoundError(f"No {eval_split} feature shards found")


def _load_feature_frames(paths: Sequence[Path]) -> Any:
    import pandas as pd

    from .artifacts import load_feature_shard

    return pd.concat(
        (load_feature_shard(path) for path in paths),
        ignore_index=True,
    )


def _load_scored_test(
    config: ExperimentConfig,
    model_key: str,
    mode: str,
) -> Any:
    paths = _feature_paths(config, model_key, "test")
    _require_nonempty_shards(paths, "test")
    return _score_test_frame(
        config,
        model_key,
        _load_feature_frames(paths),
        mode,
    )


def _score_test_frame(
    config: ExperimentConfig,
    model_key: str,
    test_frame: Any,
    mode: str,
) -> Any:
    from .calibration import FrozenCalibration

    calibration_path = _calibration_path(config, model_key, mode)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Missing calibration artifact: {calibration_path}")
    payload = _read_json(calibration_path)
    if payload.get("config_hash") != config.digest:
        raise ValueError(
            f"Calibration artifact for {model_key}/{mode} does not match current config"
        )
    if payload.get("mode") != mode:
        raise ValueError(
            f"Calibration artifact mode is {payload.get('mode')!r}, expected {mode!r}"
        )
    expected_dependencies = _calibration_dependencies(config, model_key)
    if payload.get("dependencies_sha256") != sha256_json(expected_dependencies):
        raise ValueError(
            f"Calibration artifact for {model_key}/{mode} was fitted from "
            "different feature/code/data/model dependencies"
        )
    calibration = FrozenCalibration.from_dict(payload["calibration"])
    return calibration.score_frame(test_frame)


def _calibration_path(
    config: ExperimentConfig,
    model_key: str,
    mode: str,
) -> Path:
    return model_dir(config, model_key) / f"calibration_{mode}.json"


def _calibration_dependencies(
    config: ExperimentConfig,
    model_key: str,
    *,
    train_paths: Sequence[Path] | None = None,
    dev_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    cache_key = (config.digest, model_key)
    if cache_key in _CALIBRATION_DEPENDENCY_CACHE:
        return _CALIBRATION_DEPENDENCY_CACHE[cache_key]
    if train_paths is None:
        train_paths = _feature_paths(config, model_key, "calibration_train")
    if dev_paths is None:
        dev_paths = _feature_paths(config, model_key, "dev")
    heads_payload = _load_copy_heads(config, model_key)
    dependencies = {
        "experiment_dependencies_sha256": sha256_json(
            _experiment_dependencies(config, model_key)
        ),
        "copy_heads_sha256": sha256_json(heads_payload),
        "feature_sets": {
            "calibration_train": _feature_set_fingerprint(train_paths),
            "dev": _feature_set_fingerprint(dev_paths),
        },
    }
    _CALIBRATION_DEPENDENCY_CACHE[cache_key] = dependencies
    return dependencies


def _feature_set_fingerprint(paths: Sequence[Path]) -> dict[str, Any]:
    from .artifacts import load_manifest, shard_manifest_path

    entries: dict[str, Any] = {}
    for path in paths:
        manifest = load_manifest(shard_manifest_path(path), verify=True)
        entries[path.name] = {
            "manifest_sha256": manifest["manifest_sha256"],
            "artifact_sha256": manifest["artifact"]["sha256"],
            "rows": manifest["artifact"]["rows"],
        }
    return {
        "shards": len(entries),
        "aggregate_sha256": sha256_json(entries),
    }


def _calibration_modes(config: ExperimentConfig) -> tuple[str, ...]:
    available: list[str] = []
    if "standard" in config.extraction.jsd_modes:
        available.append("standard")
    if (
        "legacy_redeep" in config.extraction.jsd_modes
        and config.extraction.include_legacy_whole_prefix_ecs
    ):
        available.append("legacy_redeep")
    primary = config.extraction.primary_jsd_mode
    if primary not in available:
        raise ValueError(
            f"Primary mode {primary!r} lacks its required PKS/ECS feature pair"
        )
    return (primary, *(mode for mode in available if mode != primary))


def _write_results(
    frame: Any,
    output_dir: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"
    report_path = output_dir / "report.md"
    csv_temp = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(csv_temp, index=False)
        os.replace(csv_temp, csv_path)
    finally:
        csv_temp.unlink(missing_ok=True)
    records = frame.to_dict(orient="records")
    _atomic_write_json(
        json_path,
        {
            "metadata": dict(metadata or {}),
            "metrics": records,
        },
    )
    report = [
        "# ReDeEP evaluation",
        "",
        "This is a cross-backbone evaluation on fixed Llama2-7B-chat responses.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(_jsonable(metadata or {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Metrics",
        "",
        _frame_to_markdown(frame),
        "",
    ]
    report_temp = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    try:
        report_temp.write_text("\n".join(report), encoding="utf-8")
        os.replace(report_temp, report_path)
    finally:
        report_temp.unlink(missing_ok=True)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(report_path),
    }


def _result_metadata(
    config: ExperimentConfig,
    model_key: str,
    *,
    test_paths: Sequence[Path],
) -> dict[str, Any]:
    calibrations: dict[str, Any] = {}
    for mode in _calibration_modes(config):
        payload = _read_json(_calibration_path(config, model_key, mode))
        frozen = payload["calibration"]
        calibrations[mode] = {
            "artifact": str(_calibration_path(config, model_key, mode)),
            "k_heads": frozen["k_heads"],
            "k_layers": frozen["k_layers"],
            "alpha": frozen["alpha"],
            "beta": frozen["beta"],
            "selection_metric": frozen["selection_metric"],
            "selection_value": frozen["selection_value"],
            "selected_ecs_columns": frozen["selected_ecs_columns"],
            "selected_pks_columns": frozen["selected_pks_columns"],
            "dependencies_sha256": payload["dependencies_sha256"],
            "grid_results_sha256": payload["grid_results_sha256"],
        }
    heads_payload = _load_copy_heads(config, model_key)
    return {
        "experiment": config.name,
        "cross_backbone": True,
        "generator_model": config.dataset.generator_model,
        "scorer_key": model_key,
        "scorer_name": config.models[model_key].name,
        "primary_mode": config.extraction.primary_jsd_mode,
        "config_hash": config.digest,
        "git_commit": _current_git_commit(),
        "dependencies": _experiment_dependencies(config, model_key),
        "dependencies_sha256": sha256_json(
            _experiment_dependencies(config, model_key)
        ),
        "copy_heads_sha256": sha256_json(heads_payload),
        "model_runtime": heads_payload.get("model_runtime"),
        "test_feature_set": _feature_set_fingerprint(test_paths),
        "calibrations": calibrations,
    }


def _float_grid(start: float, stop: float, step: float) -> tuple[float, ...]:
    if step <= 0 or stop < start:
        raise ValueError("Invalid floating-point grid bounds")
    count = int(round((stop - start) / step))
    values = tuple(round(start + index * step, 12) for index in range(count + 1))
    return tuple(value for value in values if value <= stop + step * 1e-9)


def _frame_to_markdown(frame: Any) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for values in frame.itertuples(index=False, name=None):
        cells = [str(value).replace("|", r"\|").replace("\n", " ") for value in values]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body])


def _require_model_key(config: ExperimentConfig, model_key: str) -> ModelConfig:
    try:
        return config.models[model_key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {model_key!r}; configured models: {sorted(config.models)}"
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    from .artifacts import load_manifest

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    if "manifest_sha256" in value:
        return load_manifest(path, verify=True)
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
    from .artifacts import write_manifest

    write_manifest(path, value)


def _shard_manifest(
    config: ExperimentConfig,
    model_key: str,
    eval_split: str,
    response_id: str,
    *,
    copy_heads_hash: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "config_hash": config.digest,
        "git_commit": _current_git_commit(),
        "model_key": model_key,
        "model_name": config.models[model_key].name,
        "eval_split": eval_split,
        "response_id": str(response_id),
        "dependencies_sha256": sha256_json(
            _experiment_dependencies(config, model_key)
        ),
    }
    if copy_heads_hash is not None:
        payload["copy_heads_sha256"] = copy_heads_hash
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "item"):
        return value.item()
    return value


@lru_cache(maxsize=1)
def _current_git_commit() -> str:
    return git_commit()


def _optional_file_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _experiment_dependencies(
    config: ExperimentConfig,
    model_key: str,
) -> dict[str, Any]:
    """Return the complete dependency identity shared by all model artifacts."""

    cache_key = (config.digest, model_key)
    if cache_key in _DEPENDENCY_CACHE:
        return _DEPENDENCY_CACHE[cache_key]
    model_config = _require_model_key(config, model_key)
    dependencies = {
        "config_sha256": config.digest,
        "git_commit": _current_git_commit(),
        "implementation": _implementation_fingerprint(),
        "dataset_canonical_newline_sha256": {
            "response": sha256_file_canonical_newlines(
                config.dataset.response_path
            ),
            "source": sha256_file_canonical_newlines(
                config.dataset.source_path
            ),
        },
        "model_checkpoint": _directory_fingerprint(
            str(model_config.path),
            "model",
        ),
        "tokenizer": _directory_fingerprint(
            str(model_config.tokenizer_path),
            "tokenizer",
        ),
    }
    _DEPENDENCY_CACHE[cache_key] = dependencies
    return dependencies


@lru_cache(maxsize=1)
def _implementation_fingerprint() -> dict[str, Any]:
    package_dir = Path(__file__).resolve().parent
    repository_dir = package_dir.parent.parent
    paths = sorted(package_dir.glob("*.py"))
    pyproject = repository_dir / "pyproject.toml"
    if pyproject.is_file():
        paths.append(pyproject)
    files = {
        path.relative_to(repository_dir).as_posix(): {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }
    return {
        "files": files,
        "aggregate_sha256": sha256_json(files),
    }


@lru_cache(maxsize=16)
def _directory_fingerprint(root_value: str, kind: str) -> dict[str, Any]:
    """Hash checkpoint weights or tokenizer files once per CLI process."""

    root = Path(root_value)
    if kind not in {"model", "tokenizer"}:
        raise ValueError(f"Unsupported fingerprint kind: {kind}")
    if not root.is_dir():
        return {
            "root": str(root),
            "files": {},
            "aggregate_sha256": sha256_json({}),
        }

    model_metadata = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    tokenizer_metadata = {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }

    def selected(path: Path) -> bool:
        name = path.name
        if kind == "model":
            return (
                name in model_metadata
                or path.suffix == ".safetensors"
                or (path.suffix == ".bin" and "model" in name)
            )
        return name in tokenizer_metadata or name.startswith("tokenizer.")

    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and selected(path)
    )
    files = {
        path.relative_to(root).as_posix(): {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }
    return {
        "root": str(root),
        "files": files,
        "aggregate_sha256": sha256_json(files),
    }


def _validate_loaded_model(model: Any, config: ModelConfig) -> None:
    """Reject placements that cannot support direct mechanistic weight access."""

    import torch

    model_type = str(getattr(model.config, "model_type", "")).lower()
    if model_type != config.family:
        raise RuntimeError(
            f"Configured family {config.family!r} does not match "
            f"checkpoint model_type {model_type!r}"
        )
    meta_parameters = [
        name for name, parameter in model.named_parameters() if parameter.is_meta
    ]
    if meta_parameters:
        preview = ", ".join(meta_parameters[:3])
        raise RuntimeError(
            "The loaded model contains meta/disk-offloaded parameters, which "
            f"cannot be read directly for ReDeEP ({preview}). Allocate enough "
            "GPU memory or use a multi-GPU device map."
        )
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        placements = {str(value) for value in device_map.values()}
        if "disk" in placements:
            raise RuntimeError(
                "Disk offload is unsupported because ReDeEP directly reads "
                "projection weights."
            )
        if "cpu" in placements and torch.cuda.is_available():
            raise RuntimeError(
                "CPU offload is unsupported for the full-vocabulary ReDeEP "
                "calculation. Allocate enough GPU memory or use multiple GPUs."
            )


def _model_runtime_metadata(model: Any) -> dict[str, Any]:
    device_map = getattr(model, "hf_device_map", None)
    return {
        "model_type": str(getattr(model.config, "model_type", "")),
        "torch_dtype": str(getattr(model, "dtype", "")),
        "attn_implementation": str(
            getattr(model.config, "_attn_implementation", "")
        ),
        "hf_device_map": (
            {str(key): str(value) for key, value in device_map.items()}
            if isinstance(device_map, Mapping)
            else None
        ),
    }


def _release_accelerator_memory() -> None:
    """Release cached allocations between heavyweight stages in one process."""

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
