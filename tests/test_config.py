from __future__ import annotations

from pathlib import Path

import pytest

from redeep.config import load_config


def _write_config(tmp_path: Path, qwen_thinking: bool = False) -> Path:
    response = tmp_path / "response.jsonl"
    source = tmp_path / "source.jsonl"
    response.write_text("", encoding="utf-8")
    source.write_text("", encoding="utf-8")
    model_a = tmp_path / "llama"
    model_b = tmp_path / "qwen"
    model_a.mkdir()
    model_b.mkdir()
    path = tmp_path / "experiment.yaml"
    path.write_text(
        f"""
name: test
output_dir: out
dataset:
  response_path: response.jsonl
  source_path: source.jsonl
models:
  llama:
    family: llama
    path: llama
  qwen:
    family: qwen3
    path: qwen
    enable_thinking: {str(qwen_thinking).lower()}
""",
        encoding="utf-8",
    )
    return path


def test_load_config_resolves_paths_and_is_stable(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    first = load_config(path)
    second = load_config(path)
    assert first.dataset.response_path == tmp_path / "response.jsonl"
    assert first.models["llama"].tokenizer_path == tmp_path / "llama"
    assert first.output_dir == tmp_path / "out"
    assert isinstance(first.extraction.jsd_modes, tuple)
    assert first.digest == second.digest
    assert len(first.digest) == 64


def test_qwen_thinking_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="thinking"):
        load_config(_write_config(tmp_path, qwen_thinking=True))
