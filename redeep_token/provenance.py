"""Content-addressed provenance for resumable ReDeEP feature extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


_CHECKPOINT_METADATA = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_fingerprint(files: dict[str, dict[str, int | str]]) -> str:
    payload = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_manifest(
    root: str | Path,
    relative_paths: Iterable[str | Path],
) -> dict[str, object]:
    """Hash an explicit set of files relative to one root."""

    root_path = Path(root).resolve()
    files: dict[str, dict[str, int | str]] = {}
    for relative_path in sorted(
        {Path(path).as_posix() for path in relative_paths}
    ):
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"manifest path escapes its root: {relative_path}"
            )
        path = root_path / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative_path] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not files:
        raise ValueError("a file manifest cannot be empty")
    return {
        "files": files,
        "fingerprint": _manifest_fingerprint(files),
    }


def checkpoint_artifact_manifest(
    checkpoint_directory: str | Path,
) -> dict[str, object]:
    """Hash model weights, config, tokenizer, and chat-template artifacts."""

    root = Path(checkpoint_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = {
        name for name in _CHECKPOINT_METADATA if (root / name).is_file()
    }
    for pattern in _WEIGHT_PATTERNS:
        paths.update(
            path.relative_to(root).as_posix()
            for path in root.glob(pattern)
            if path.is_file()
        )
    if "config.json" not in paths:
        raise FileNotFoundError(root / "config.json")
    if not any(
        path.endswith(".safetensors")
        or Path(path).name.startswith("pytorch_model")
        and path.endswith(".bin")
        for path in paths
    ):
        raise FileNotFoundError(
            f"no model weight artifacts were found under {root}"
        )
    return file_manifest(root, paths)
