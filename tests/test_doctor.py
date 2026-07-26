from __future__ import annotations

from pathlib import Path

from redeep.config import load_config
from redeep.doctor import run_doctor
from redeep.utils import sha256_file_canonical_newlines


def test_doctor_without_optional_model_imports(tmp_path: Path) -> None:
    (tmp_path / "response.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "source.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "model").mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output_dir: out
dataset:
  response_path: response.jsonl
  source_path: source.jsonl
models:
  local:
    family: llama
    path: model
""",
        encoding="utf-8",
    )
    report = run_doctor(load_config(config_path), load_tokenizers=False)
    assert report["ok"]
    assert report["dataset"]["response_path"]["exists"]
    assert len(report["dataset"]["response_path"]["sha256"]) == 64
    assert report["models"]["local"]["exists"]


def test_official_text_hash_is_checkout_newline_independent(tmp_path: Path) -> None:
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    lf.write_bytes(b'{"id": 1}\n{"id": 2}\n')
    crlf.write_bytes(b'{"id": 1}\r\n{"id": 2}\r\n')

    assert sha256_file_canonical_newlines(lf) == sha256_file_canonical_newlines(
        crlf
    )
