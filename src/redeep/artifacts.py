"""Reproducible, atomic artifact I/O and resume validation helpers."""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MANIFEST_HASH_KEY = "manifest_sha256"


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact or its manifest fails integrity checks."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(
    paths: Iterable[str | Path],
    *,
    root: str | Path | None = None,
) -> dict[str, str]:
    """Return deterministically ordered ``relative-path -> sha256`` entries."""

    root_path = Path(root).resolve() if root is not None else None
    entries: list[tuple[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if root_path is not None:
            try:
                key = path.relative_to(root_path).as_posix()
            except ValueError as error:
                raise ValueError(f"{path} is outside hash root {root_path}") from error
        else:
            key = path.as_posix()
        entries.append((key, sha256_file(path)))
    return dict(sorted(entries))


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize supported values deterministically for hashing and manifests."""

    normalized = _jsonable(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def hash_config(config: Any) -> str:
    return sha256_bytes(canonical_json_bytes(config))


def manifest_hash(manifest: Any) -> str:
    """Hash a manifest while excluding its self-referential hash field."""

    payload = _jsonable(manifest)
    if not isinstance(payload, dict):
        raise TypeError("A manifest must serialize to a mapping")
    payload.pop(MANIFEST_HASH_KEY, None)
    return hash_config(payload)


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    indent: int = 2,
) -> Path:
    """Atomically replace a JSON file using a temporary sibling."""

    destination = Path(path)
    payload = _jsonable(value)
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(destination, content)
    return destination


def atomic_write_parquet(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    compression: str = "zstd",
    index: bool = False,
) -> Path:
    """Atomically replace a Parquet file in the destination directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, compression=compression, index=index)
        _fsync_file(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_manifest(path: str | Path, manifest: Any) -> dict[str, Any]:
    """Write a self-hashed manifest and return the stored payload."""

    payload = _jsonable(manifest)
    if not isinstance(payload, dict):
        raise TypeError("A manifest must serialize to a mapping")
    payload.pop(MANIFEST_HASH_KEY, None)
    payload[MANIFEST_HASH_KEY] = manifest_hash(payload)
    atomic_write_json(path, payload)
    return payload


def load_manifest(
    path: str | Path,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    """Load a manifest and optionally verify its embedded canonical hash."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError(f"Manifest {path} is not a JSON object")
    if verify:
        stored_hash = payload.get(MANIFEST_HASH_KEY)
        if not isinstance(stored_hash, str):
            raise ArtifactIntegrityError(f"Manifest {path} has no {MANIFEST_HASH_KEY}")
        actual_hash = manifest_hash(payload)
        if stored_hash != actual_hash:
            raise ArtifactIntegrityError(
                f"Manifest hash mismatch for {path}: expected {stored_hash}, got {actual_hash}"
            )
    return payload


def manifest_matches(
    path: str | Path,
    expected: Any,
    *,
    required_keys: Iterable[str] | None = None,
) -> bool:
    """Return whether a stored manifest matches all expected fields."""

    try:
        actual = load_manifest(path, verify=True)
    except (OSError, ValueError, json.JSONDecodeError, ArtifactIntegrityError):
        return False
    expected_payload = _jsonable(expected)
    if not isinstance(expected_payload, dict):
        raise TypeError("expected manifest must serialize to a mapping")
    expected_payload.pop(MANIFEST_HASH_KEY, None)
    actual_without_hash = dict(actual)
    actual_without_hash.pop(MANIFEST_HASH_KEY, None)
    if required_keys is not None:
        keys = tuple(required_keys)
        missing = [key for key in keys if key not in expected_payload]
        if missing:
            raise ValueError(f"required_keys absent from expected manifest: {missing}")
        expected_payload = {key: expected_payload[key] for key in keys}
    return _mapping_is_subset(expected_payload, actual_without_hash)


def shard_manifest_path(parquet_path: str | Path) -> Path:
    path = Path(parquet_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def atomic_write_feature_shard(
    frame: pd.DataFrame,
    parquet_path: str | Path,
    manifest: Any,
    *,
    compression: str = "zstd",
) -> dict[str, Any]:
    """Write a Parquet shard, then atomically publish its completion manifest.

    The manifest is the completion marker. A crash after the Parquet rename but
    before manifest publication leaves a shard that resume logic treats as
    incomplete rather than silently accepting.
    """

    path = Path(parquet_path)
    atomic_write_parquet(path, frame, compression=compression, index=False)
    payload = _jsonable(manifest)
    if not isinstance(payload, dict):
        raise TypeError("A shard manifest must serialize to a mapping")
    payload.pop(MANIFEST_HASH_KEY, None)
    payload["artifact"] = {
        "filename": path.name,
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
    }
    return write_manifest(shard_manifest_path(path), payload)


def shard_is_complete(
    parquet_path: str | Path,
    *,
    expected_manifest: Any | None = None,
    verify_file_hash: bool = True,
) -> bool:
    """Check the completion marker, manifest compatibility, and file hash."""

    path = Path(parquet_path)
    sidecar = shard_manifest_path(path)
    if not path.is_file() or path.stat().st_size == 0 or not sidecar.is_file():
        return False
    try:
        payload = load_manifest(sidecar, verify=True)
    except (OSError, ValueError, json.JSONDecodeError, ArtifactIntegrityError):
        return False
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        return False
    if artifact.get("filename") != path.name:
        return False
    if verify_file_hash and artifact.get("sha256") != sha256_file(path):
        return False
    if expected_manifest is not None and not manifest_matches(sidecar, expected_manifest):
        return False
    return True


def load_feature_shard(
    parquet_path: str | Path,
    *,
    expected_manifest: Any | None = None,
    verify_file_hash: bool = True,
) -> pd.DataFrame:
    """Load a completed shard or raise a precise integrity error."""

    path = Path(parquet_path)
    if not shard_is_complete(
        path,
        expected_manifest=expected_manifest,
        verify_file_hash=verify_file_hash,
    ):
        raise ArtifactIntegrityError(f"Incomplete or incompatible feature shard: {path}")
    frame = pd.read_parquet(path)
    manifest = load_manifest(shard_manifest_path(path), verify=True)
    artifact = manifest["artifact"]
    if int(artifact.get("rows", -1)) != len(frame):
        raise ArtifactIntegrityError(f"Row-count mismatch for feature shard: {path}")
    if [str(column) for column in frame.columns] != artifact.get("columns"):
        raise ArtifactIntegrityError(f"Column mismatch for feature shard: {path}")
    return frame


def find_completed_shards(
    directory: str | Path,
    *,
    pattern: str = "*.parquet",
    expected_manifest: Any | None = None,
    verify_file_hash: bool = True,
) -> list[Path]:
    """List valid shards in stable path order for deterministic resume."""

    directory_path = Path(directory)
    if not directory_path.exists():
        return []
    return [
        path
        for path in sorted(directory_path.glob(pattern))
        if shard_is_complete(
            path,
            expected_manifest=expected_manifest,
            verify_file_hash=verify_file_hash,
        )
    ]


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mapping_is_subset(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                return False
            if not _mapping_is_subset(expected_value, actual_value):
                return False
        elif expected_value != actual_value:
            return False
    return True


def _atomic_write_bytes(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_file(path: Path) -> None:
    # Windows requires a writable descriptor for FlushFileBuffers/os.fsync.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(directory: Path) -> None:
    # Directory fsync is not supported on Windows; the file itself is already
    # flushed before os.replace.
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            unsupported = {
                errno.EINVAL,
                getattr(errno, "ENOTSUP", -1),
                getattr(errno, "EOPNOTSUPP", -1),
            }
            if error.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)
