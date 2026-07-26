"""Configuration loading and validation for reproducible remote runs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _expand_path(value: str, base_dir: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        expanded = (base_dir / expanded).resolve()
    return expanded


@dataclass(frozen=True)
class DatasetConfig:
    response_path: Path
    source_path: Path
    generator_model: str = "llama-2-7b-chat"
    tasks: tuple[str, ...] = ("QA", "Data2txt", "Summary")
    include_truncated_test: bool = True
    dev_per_task: int = 50
    split_seed: int = 2024
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class ModelConfig:
    name: str
    path: Path
    tokenizer_path: Path
    family: str
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = False
    enable_thinking: bool = False
    max_length: int | None = None


@dataclass(frozen=True)
class ExtractionConfig:
    context_top_fraction: float = 0.10
    copying_candidate_count: int = 32
    copying_vocab_sample_size: int = 1024
    copying_seed: int = 2024
    token_chunk_size: int = 16
    # Larger than both target vocabularies, enabling the exact one-pass
    # full-vocabulary path for each small token chunk.
    vocab_chunk_size: int = 262144
    capture_to_cpu: bool = True
    jsd_modes: tuple[str, ...] = ("standard", "legacy_redeep")
    primary_jsd_mode: str = "standard"
    primary_ecs_mode: str = "context_only"
    include_legacy_whole_prefix_ecs: bool = True


@dataclass(frozen=True)
class CalibrationConfig:
    alpha: float = 1.0
    head_k_min: int = 1
    head_k_max: int = 32
    layer_k_min: int = 1
    layer_k_max: int = 32
    beta_min: float = 0.1
    beta_max: float = 1.9
    beta_step: float = 0.1
    selection_metric: str = "task_macro_token_auc"


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 42
    confidence_level: float = 0.95
    include_quality_good_sensitivity: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: Path
    seed: int
    dataset: DatasetConfig
    models: dict[str, ModelConfig]
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    config_path: Path | None = None

    def to_serializable(self) -> dict[str, Any]:
        value = asdict(self)

        def convert(item: Any) -> Any:
            if isinstance(item, Path):
                return str(item)
            if isinstance(item, tuple):
                return [convert(x) for x in item]
            if isinstance(item, dict):
                return {str(k): convert(v) for k, v in item.items()}
            if isinstance(item, list):
                return [convert(x) for x in item]
            return item

        return convert(value)

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_serializable(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _require_mapping(raw: Any, key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration key '{key}' must be a mapping")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config and resolve relative paths against that file."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML value must be a mapping")

    base_dir = config_path.parent
    dataset_raw = _require_mapping(raw, "dataset")
    models_raw = _require_mapping(raw, "models")

    dataset = DatasetConfig(
        response_path=_expand_path(str(dataset_raw["response_path"]), base_dir),
        source_path=_expand_path(str(dataset_raw["source_path"]), base_dir),
        generator_model=str(dataset_raw.get("generator_model", "llama-2-7b-chat")),
        tasks=tuple(dataset_raw.get("tasks", ("QA", "Data2txt", "Summary"))),
        include_truncated_test=bool(dataset_raw.get("include_truncated_test", True)),
        dev_per_task=int(dataset_raw.get("dev_per_task", 50)),
        split_seed=int(dataset_raw.get("split_seed", 2024)),
        system_prompt=str(
            dataset_raw.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        ),
    )

    models: dict[str, ModelConfig] = {}
    for key, item in models_raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"models.{key} must be a mapping")
        model_path = _expand_path(str(item["path"]), base_dir)
        tokenizer_path = _expand_path(str(item.get("tokenizer_path", item["path"])), base_dir)
        models[str(key)] = ModelConfig(
            name=str(item.get("name", key)),
            path=model_path,
            tokenizer_path=tokenizer_path,
            family=str(item["family"]),
            dtype=str(item.get("dtype", "bfloat16")),
            device_map=str(item.get("device_map", "auto")),
            attn_implementation=str(item.get("attn_implementation", "sdpa")),
            trust_remote_code=bool(item.get("trust_remote_code", False)),
            enable_thinking=bool(item.get("enable_thinking", False)),
            max_length=(
                int(item["max_length"]) if item.get("max_length") is not None else None
            ),
        )

    extraction_raw = raw.get("extraction", {})
    if not isinstance(extraction_raw, dict):
        raise ValueError("Configuration key 'extraction' must be a mapping")
    extraction_values = dict(extraction_raw)
    if "jsd_modes" in extraction_values:
        extraction_values["jsd_modes"] = tuple(extraction_values["jsd_modes"])
    extraction = ExtractionConfig(**extraction_values)
    calibration = CalibrationConfig(**raw.get("calibration", {}))
    evaluation = EvaluationConfig(**raw.get("evaluation", {}))
    config = ExperimentConfig(
        name=str(raw.get("name", "redeep-ragtruth-cross-backbone")),
        output_dir=_expand_path(str(raw.get("output_dir", "../outputs")), base_dir),
        seed=int(raw.get("seed", 2024)),
        dataset=dataset,
        models=models,
        extraction=extraction,
        calibration=calibration,
        evaluation=evaluation,
        config_path=config_path,
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if not config.models:
        raise ValueError("At least one model must be configured")
    if config.dataset.dev_per_task < 1:
        raise ValueError("dev_per_task must be positive")
    if not config.dataset.system_prompt.strip():
        raise ValueError("dataset.system_prompt must not be empty")
    if not config.dataset.tasks or len(set(config.dataset.tasks)) != len(
        config.dataset.tasks
    ):
        raise ValueError("dataset.tasks must be a non-empty sequence of unique names")
    supported_families = {"llama", "qwen3"}
    for key, model in config.models.items():
        if model.family not in supported_families:
            raise ValueError(
                f"models.{key}.family={model.family!r}; expected one of {supported_families}"
            )
        if model.family == "qwen3" and model.enable_thinking:
            raise ValueError(
                "Qwen3 thinking must be disabled for Llama2 response teacher forcing"
            )
        if model.attn_implementation not in {"eager", "sdpa"}:
            raise ValueError(
                f"models.{key}.attn_implementation must be 'eager' or 'sdpa'"
            )
    if not 0 < config.extraction.context_top_fraction <= 1:
        raise ValueError("context_top_fraction must be in (0, 1]")
    if config.extraction.copying_candidate_count < 1:
        raise ValueError("copying_candidate_count must be positive")
    if config.extraction.copying_vocab_sample_size < 2:
        raise ValueError("copying_vocab_sample_size must be at least two")
    if config.extraction.token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    if config.extraction.vocab_chunk_size < 1:
        raise ValueError("vocab_chunk_size must be positive")
    allowed_jsd = {"standard", "legacy_redeep"}
    unknown_jsd = set(config.extraction.jsd_modes) - allowed_jsd
    if unknown_jsd:
        raise ValueError(f"Unsupported JSD modes: {sorted(unknown_jsd)}")
    if config.extraction.primary_jsd_mode not in config.extraction.jsd_modes:
        raise ValueError("primary_jsd_mode must be included in jsd_modes")
    allowed_ecs = {"context_only", "whole_prefix"}
    if config.extraction.primary_ecs_mode not in allowed_ecs:
        raise ValueError(f"primary_ecs_mode must be one of {sorted(allowed_ecs)}")
    expected_ecs = (
        "context_only"
        if config.extraction.primary_jsd_mode == "standard"
        else "whole_prefix"
    )
    if config.extraction.primary_ecs_mode != expected_ecs:
        raise ValueError(
            f"primary_jsd_mode={config.extraction.primary_jsd_mode!r} requires "
            f"primary_ecs_mode={expected_ecs!r}"
        )
    if (
        config.extraction.primary_jsd_mode == "legacy_redeep"
        and not config.extraction.include_legacy_whole_prefix_ecs
    ):
        raise ValueError(
            "legacy_redeep cannot be primary when whole-prefix ECS is disabled"
        )
    if config.calibration.head_k_min < 1 or config.calibration.layer_k_min < 1:
        raise ValueError("K search lower bounds must be at least one")
    if config.calibration.head_k_max < config.calibration.head_k_min:
        raise ValueError("head_k_max must be at least head_k_min")
    if config.calibration.layer_k_max < config.calibration.layer_k_min:
        raise ValueError("layer_k_max must be at least layer_k_min")
    if config.calibration.head_k_max > config.extraction.copying_candidate_count:
        raise ValueError("head_k_max cannot exceed copying_candidate_count")
    if (
        config.calibration.beta_min <= 0
        or config.calibration.beta_max < config.calibration.beta_min
    ):
        raise ValueError("beta bounds must satisfy 0 < beta_min <= beta_max")
    if config.calibration.beta_step <= 0:
        raise ValueError("beta_step must be positive")
    if config.calibration.alpha <= 0:
        raise ValueError("calibration alpha must be positive")
    if config.calibration.selection_metric != "task_macro_token_auc":
        raise ValueError(
            "selection_metric must be 'task_macro_token_auc' for this protocol"
        )
    if config.evaluation.bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    if not 0 < config.evaluation.confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
