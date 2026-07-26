"""Read-only environment checks run before expensive feature extraction."""

from __future__ import annotations

from typing import Any

from .config import ExperimentConfig
from .data import OFFICIAL_RESPONSE_SHA256, OFFICIAL_SOURCE_SHA256
from .utils import (
    environment_snapshot,
    sha256_file,
    sha256_file_canonical_newlines,
)


def run_doctor(
    config: ExperimentConfig,
    model_key: str | None = None,
    *,
    load_tokenizers: bool = True,
) -> dict[str, Any]:
    selected = (
        {model_key: config.models[model_key]} if model_key else config.models
    )
    report: dict[str, Any] = {
        "ok": True,
        "environment": environment_snapshot(),
        "dataset": {},
        "models": {},
        "errors": [],
        "warnings": [],
    }

    for label, path, expected_hash in (
        (
            "response_path",
            config.dataset.response_path,
            OFFICIAL_RESPONSE_SHA256,
        ),
        (
            "source_path",
            config.dataset.source_path,
            OFFICIAL_SOURCE_SHA256,
        ),
    ):
        exists = path.is_file()
        observed_hash = sha256_file(path) if exists else None
        canonical_hash = (
            sha256_file_canonical_newlines(path) if exists else None
        )
        report["dataset"][label] = {
            "path": str(path),
            "exists": exists,
            "sha256": observed_hash,
            "canonical_newline_sha256": canonical_hash,
            "expected_sha256": expected_hash,
            "official_hash_match": canonical_hash == expected_hash if exists else False,
        }
        if not exists:
            report["ok"] = False
            report["errors"].append(f"Missing dataset file: {path}")
        elif canonical_hash != expected_hash:
            report["warnings"].append(
                f"{label} does not byte-match the pinned RAGTruth release."
            )

    if report["environment"].get("transformers") != "4.52.2":
        report["warnings"].append(
            "The reproduction is pinned to transformers==4.52.2; "
            f"found {report['environment'].get('transformers')!r}."
        )
    if report["environment"].get("cuda_available", False) and any(
        model.dtype == "bfloat16" for model in selected.values()
    ):
        try:
            import torch

            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except (ImportError, RuntimeError):
            bf16_supported = False
        report["environment"]["bf16_supported"] = bf16_supported
        if not bf16_supported:
            report["ok"] = False
            report["errors"].append(
                "A configured model uses bfloat16, but the selected CUDA device "
                "does not support it; use float16 or another GPU."
            )

    if load_tokenizers:
        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:
            report["ok"] = False
            report["errors"].append(f"Cannot import transformers: {exc}")
            return report

    for key, model in selected.items():
        item: dict[str, Any] = {
            "name": model.name,
            "family": model.family,
            "dtype": model.dtype,
            "device_map": model.device_map,
            "attn_implementation": model.attn_implementation,
            "path": str(model.path),
            "exists": model.path.is_dir(),
            "tokenizer_path": str(model.tokenizer_path),
        }
        if not model.path.is_dir():
            report["ok"] = False
            report["errors"].append(f"Missing model directory for {key}: {model.path}")
            report["models"][key] = item
            continue
        config_json = model.path / "config.json"
        item["config_sha256"] = (
            sha256_file(config_json) if config_json.is_file() else None
        )
        if load_tokenizers:
            try:
                hf_config = AutoConfig.from_pretrained(
                    model.path,
                    trust_remote_code=model.trust_remote_code,
                    local_files_only=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model.tokenizer_path,
                    trust_remote_code=model.trust_remote_code,
                    local_files_only=True,
                    use_fast=True,
                )
                item.update(
                    {
                        "model_type": hf_config.model_type,
                        "hidden_size": getattr(hf_config, "hidden_size", None),
                        "num_hidden_layers": getattr(
                            hf_config, "num_hidden_layers", None
                        ),
                        "num_attention_heads": getattr(
                            hf_config, "num_attention_heads", None
                        ),
                        "num_key_value_heads": getattr(
                            hf_config, "num_key_value_heads", None
                        ),
                        "vocab_size": getattr(hf_config, "vocab_size", None),
                        "tokenizer_fast": bool(tokenizer.is_fast),
                        "chat_template": bool(tokenizer.chat_template),
                    }
                )
                if str(hf_config.model_type).lower() != model.family:
                    report["ok"] = False
                    report["errors"].append(
                        f"{key} model_type={hf_config.model_type!r} does not "
                        f"match configured family={model.family!r}."
                    )
                if not tokenizer.is_fast:
                    report["ok"] = False
                    report["errors"].append(
                        f"{key} tokenizer is not fast; offset_mapping is required."
                    )
                if not tokenizer.chat_template:
                    report["ok"] = False
                    report["errors"].append(
                        f"{key} tokenizer does not define a chat template."
                    )
                if model.family == "qwen3" and model.enable_thinking:
                    report["ok"] = False
                    report["errors"].append(
                        "Qwen3 enable_thinking must be false for this experiment."
                    )
            except Exception as exc:  # report all model-format failures together
                report["ok"] = False
                report["errors"].append(f"Failed to inspect {key}: {type(exc).__name__}: {exc}")
        report["models"][key] = item

    if not report["environment"].get("cuda_available", False):
        report["warnings"].append(
            "CUDA is unavailable. Data audit/evaluation can run, but feature extraction cannot."
        )
    return report
