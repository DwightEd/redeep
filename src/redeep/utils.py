"""Small reproducibility and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_canonical_newlines(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Hash text bytes after canonicalizing Git-style CRLF checkouts to LF."""

    digest = hashlib.sha256()
    pending_carriage_return = False
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            if pending_carriage_return:
                chunk = b"\r" + chunk
                pending_carriage_return = False
            if chunk.endswith(b"\r"):
                chunk = chunk[:-1]
                pending_carriage_return = True
            digest.update(chunk.replace(b"\r\n", b"\n"))
    if pending_carriage_return:
        digest.update(b"\r")
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit(cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": sys.platform,
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = torch.cuda.is_available()
        snapshot["cuda_version"] = torch.version.cuda
        snapshot["gpu_count"] = torch.cuda.device_count()
        snapshot["gpus"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except ImportError:
        snapshot["torch"] = None
        snapshot["cuda_available"] = False
    try:
        import transformers

        snapshot["transformers"] = transformers.__version__
    except ImportError:
        snapshot["transformers"] = None
    return snapshot
