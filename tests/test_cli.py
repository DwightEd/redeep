from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from redeep.artifacts import atomic_write_feature_shard
from redeep.cli import build_parser, main
from redeep.config import load_config
from redeep.pipeline import (
    EVAL_SPLITS,
    _shard_manifest,
    feature_shard_path,
    partition_examples,
    pending_examples,
    select_work_shard,
)


def _fixture_config(tmp_path: Path) -> Path:
    sources = []
    responses = []
    source_info = {
        "QA": {"question": "q", "passages": "qa context"},
        "Data2txt": {"name": "business", "open": True},
        "Summary": "summary context",
    }
    for index, (task, info) in enumerate(source_info.items()):
        context = repr(info) if task == "Data2txt" else (
            info["passages"] if task == "QA" else info
        )
        sources.append(
            {
                "source_id": str(index),
                "task_type": task,
                "source": "fixture",
                "source_info": info,
                "prompt": f"instruction\n{context}\noutput:",
            }
        )
        responses.append(
            {
                "id": str(index),
                "source_id": str(index),
                "model": "llama-2-7b-chat",
                "split": "train",
                "quality": "good",
                "labels": [],
                "response": "answer",
            }
        )
    response_path = tmp_path / "response.jsonl"
    source_path = tmp_path / "source_info.jsonl"
    response_path.write_text(
        "".join(json.dumps(row) + "\n" for row in responses),
        encoding="utf-8",
    )
    source_path.write_text(
        "".join(json.dumps(row) + "\n" for row in sources),
        encoding="utf-8",
    )
    (tmp_path / "model").mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: fixture
output_dir: outputs
dataset:
  response_path: response.jsonl
  source_path: source_info.jsonl
  dev_per_task: 1
models:
  scorer:
    family: llama
    path: model
""",
        encoding="utf-8",
    )
    return config_path


def test_parser_exposes_all_pipeline_commands():
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices")
        and action.choices
    )
    assert {
        "doctor",
        "audit-data",
        "discover-heads",
        "extract",
        "calibrate",
        "evaluate",
        "compare",
        "run-all",
    }.issubset(subparsers.choices)


def test_audit_data_cli_uses_no_heavy_model_imports(tmp_path, capsys):
    config_path = _fixture_config(tmp_path)
    exit_code = main(["--config", str(config_path), "audit-data", "--no-strict"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["examples"] == 3
    assert output["partitions"] == {
        "calibration_train": 0,
        "dev": 3,
        "test": 0,
    }


def test_doctor_cli_can_skip_transformers(tmp_path, capsys):
    config_path = _fixture_config(tmp_path)
    exit_code = main(
        ["--config", str(config_path), "doctor", "--no-load-tokenizers"]
    )
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["ok"]


def test_feature_shard_resume_and_safe_response_id(tmp_path):
    config = load_config(_fixture_config(tmp_path))
    parts = partition_examples(
        config,
        __import__("redeep.pipeline", fromlist=["load_experiment_examples"])
        .load_experiment_examples(config),
    )
    example = parts["dev"][0]
    path = feature_shard_path(config, "scorer", "dev", "unsafe/id")
    assert path.suffix == ".parquet"
    assert "/" not in path.name

    assert pending_examples(config, "scorer", "dev", [example]) == (example,)
    completed = feature_shard_path(
        config, "scorer", "dev", example.response_id
    )
    atomic_write_feature_shard(
        pd.DataFrame({"value": [1]}),
        completed,
        _shard_manifest(config, "scorer", "dev", example.response_id),
    )
    assert pending_examples(config, "scorer", "dev", [example]) == ()
    assert pending_examples(
        config, "scorer", "dev", [example], force=True
    ) == (example,)


@pytest.mark.parametrize("eval_split", EVAL_SPLITS)
def test_all_eval_splits_have_distinct_directories(tmp_path, eval_split):
    config = load_config(_fixture_config(tmp_path))
    path = feature_shard_path(config, "scorer", eval_split, "1")
    assert path.parent.name == eval_split


def test_work_shards_are_disjoint_complete_and_deterministic(tmp_path):
    config = load_config(_fixture_config(tmp_path))
    examples = partition_examples(
        config,
        __import__("redeep.pipeline", fromlist=["load_experiment_examples"])
        .load_experiment_examples(config),
    )["dev"]
    assignments = [
        select_work_shard(examples, num_shards=2, shard_index=index)
        for index in range(2)
    ]
    assert {
        example.response_id for example in assignments[0]
    }.isdisjoint(example.response_id for example in assignments[1])
    assert sorted(
        example.response_id for group in assignments for example in group
    ) == sorted(example.response_id for example in examples)
    with pytest.raises(ValueError, match="shard_index"):
        select_work_shard(examples, num_shards=2, shard_index=2)
