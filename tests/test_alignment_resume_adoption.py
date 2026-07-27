from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from redeep.alignment import align_teacher_forced_example
from redeep.artifacts import (
    atomic_write_feature_shard,
    load_manifest,
    sha256_file,
    write_manifest,
)
from redeep.config import load_config
from redeep.pipeline import (
    _experiment_dependencies,
    _feature_rows,
    feature_shard_path,
    load_experiment_examples,
    model_dir,
)
from redeep.utils import sha256_json

SOURCE_COMMIT = "5fd811663cd9e636edbfecd8471334144c659516"
SOURCE_IMPLEMENTATION = "d41a53712acefaa2049723ae8d845dab5abbb68c97d5471a6083ba3cbd02143f"


class CompatibleTokenizer:
    all_special_ids: list[int] = []

    def apply_chat_template(self, messages, **kwargs):
        prefix = (
            f"<SYSTEM>{messages[0]['content']}</SYSTEM>"
            f"<USER>{messages[1]['content']}</USER><ASSISTANT>"
        )
        if len(messages) == 2:
            assert kwargs["add_generation_prompt"] is True
            return prefix
        assert len(messages) == 3
        assert kwargs["add_generation_prompt"] is False
        return prefix + messages[2]["content"].strip() + "<EOT>"

    def __call__(self, text, **kwargs):
        offsets = [(index, index + 1) for index in range(len(text))]
        return {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
            "offset_mapping": offsets,
            "special_tokens_mask": [0] * len(text),
        }


class IncompatibleTokenizer(CompatibleTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        rendered = super().apply_chat_template(messages, **kwargs)
        return rendered if len(messages) == 2 else "<LEGACY>" + rendered


def _load_adoption_module():
    path = Path(__file__).parents[1] / "scripts" / "adopt_alignment_resume.py"
    spec = importlib.util.spec_from_file_location("adopt_alignment_resume", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_config(
    tmp_path: Path,
    *,
    response: str = "answer",
    family: str = "llama",
) -> Path:
    source_path = tmp_path / "source_info.jsonl"
    response_path = tmp_path / "response.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "source_id": "source-1",
                "task_type": "QA",
                "source": "fixture",
                "source_info": {"question": "q", "passages": "context"},
                "prompt": "instruction\ncontext\noutput:",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    response_path.write_text(
        json.dumps(
            {
                "id": "response-1",
                "source_id": "source-1",
                "model": "llama-2-7b-chat",
                "split": "train",
                "quality": "good",
                "labels": [],
                "response": response,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "model").mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: fixture
output_dir: outputs
dataset:
  response_path: response.jsonl
  source_path: source_info.jsonl
  dev_per_task: 1
models:
  llama31:
    family: {family}
    path: model
""",
        encoding="utf-8",
    )
    return config_path


def _write_legacy_artifacts(
    config_path: Path,
    *,
    tokenizer: CompatibleTokenizer | None = None,
) -> tuple[Path, Path]:
    config = load_config(config_path)
    current_dependencies = _experiment_dependencies(config, "llama31")
    source_dependencies = dict(current_dependencies)
    source_dependencies["git_commit"] = SOURCE_COMMIT
    source_dependencies["implementation"] = {
        **current_dependencies["implementation"],
        "aggregate_sha256": SOURCE_IMPLEMENTATION,
    }
    source_dependencies["implementation"]["files"] = {}

    copy_path = model_dir(config, "llama31") / "copy_heads.json"
    copy_payload = write_manifest(
        copy_path,
        {
            "schema_version": 1,
            "artifact": str(copy_path),
            "model_key": "llama31",
            "model_name": config.models["llama31"].name,
            "model_path": str(config.models["llama31"].path),
            "model_config_sha256": None,
            "top_heads": [[0, 0]],
            "records": [],
            "metadata": {},
            "config_hash": config.digest,
            "git_commit": SOURCE_COMMIT,
            "dependencies": source_dependencies,
            "dependencies_sha256": sha256_json(source_dependencies),
        },
    )
    copy_hash = sha256_json(copy_payload)

    shard_path = feature_shard_path(
        config,
        "llama31",
        "dev",
        "response-1",
    )
    example = load_experiment_examples(config)[0]
    aligned = align_teacher_forced_example(
        example,
        tokenizer or CompatibleTokenizer(),
        model_family=config.models["llama31"].family,
        system_prompt=config.dataset.system_prompt,
    )
    atomic_write_feature_shard(
        pd.DataFrame(_feature_rows(aligned, {}, eval_split="dev")),
        shard_path,
        {
            "schema_version": 1,
            "config_hash": config.digest,
            "git_commit": SOURCE_COMMIT,
            "model_key": "llama31",
            "model_name": config.models["llama31"].name,
            "eval_split": "dev",
            "response_id": "response-1",
            "dependencies_sha256": sha256_json(source_dependencies),
            "copy_heads_sha256": copy_hash,
        },
    )
    return copy_path, shard_path


def test_adoption_preserves_parquet_and_records_source_provenance(
    tmp_path,
    monkeypatch,
):
    module = _load_adoption_module()
    monkeypatch.setattr(module, "_load_tokenizer", lambda _config: CompatibleTokenizer())
    config_path = _fixture_config(tmp_path)
    copy_path, shard_path = _write_legacy_artifacts(config_path)
    parquet_hash = sha256_file(shard_path)
    source_copy_manifest = load_manifest(copy_path)

    dry_run = module.adopt_alignment_resume_artifacts(
        config_path,
        "llama31",
        apply=False,
    )
    assert dry_run["eligible_feature_shards"] == 1
    assert load_manifest(copy_path) == source_copy_manifest

    report = module.adopt_alignment_resume_artifacts(
        config_path,
        "llama31",
        apply=True,
    )

    assert report["adopted_feature_shards"] == 1
    assert sha256_file(shard_path) == parquet_hash
    shard_manifest = load_manifest(shard_path.with_suffix(".parquet.manifest.json"))
    assert shard_manifest["compatibility_adoption"]["source_git_commit"] == SOURCE_COMMIT
    assert shard_manifest["artifact"]["sha256"] == parquet_hash
    assert load_manifest(copy_path)["compatibility_adoption"][
        "source_git_commit"
    ] == SOURCE_COMMIT
    assert Path(report["backup"]).is_file()
    repeated = module.adopt_alignment_resume_artifacts(
        config_path,
        "llama31",
        apply=True,
    )
    assert repeated["status"] == "already_adopted"
    assert repeated["adopted_feature_shards"] == 0


def test_pre_fix_shard_with_trimmed_response_is_rejected(tmp_path, monkeypatch):
    module = _load_adoption_module()
    monkeypatch.setattr(module, "_load_tokenizer", lambda _config: CompatibleTokenizer())
    config_path = _fixture_config(tmp_path, response="answer\n")
    _write_legacy_artifacts(config_path)

    with pytest.raises(ValueError, match="outer whitespace"):
        module.adopt_alignment_resume_artifacts(
            config_path,
            "llama31",
            apply=False,
        )


def test_pre_fix_adoption_is_restricted_to_reviewed_llama_protocol(
    tmp_path,
    monkeypatch,
):
    module = _load_adoption_module()
    monkeypatch.setattr(module, "_load_tokenizer", lambda _config: CompatibleTokenizer())
    config_path = _fixture_config(tmp_path, family="qwen3")
    _write_legacy_artifacts(config_path)

    with pytest.raises(ValueError, match="reviewed only"):
        module.adopt_alignment_resume_artifacts(
            config_path,
            "llama31",
            apply=False,
        )


def test_pre_fix_adoption_rejects_chat_templates_with_different_prefixes(
    tmp_path,
    monkeypatch,
):
    module = _load_adoption_module()
    config_path = _fixture_config(tmp_path)
    _write_legacy_artifacts(config_path)
    monkeypatch.setattr(module, "_load_tokenizer", lambda _config: IncompatibleTokenizer())

    with pytest.raises(ValueError, match="legacy assistant-message render"):
        module.adopt_alignment_resume_artifacts(
            config_path,
            "llama31",
            apply=False,
        )
