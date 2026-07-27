"""Atomic, resumable storage for per-response ReDeEP features."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


def feature_path(directory: str | Path, response_id: str) -> Path:
    root = Path(directory)
    identifier = str(response_id)
    readable = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in identifier
    )[:32].strip("_")
    if not readable:
        readable = "response"
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16]
    return root / f"{readable}-{digest}.json.gz"


def _validate_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "protocol_fingerprint",
        "id",
        "task_type",
        "labels",
        "external",
        "parametric",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"feature record is missing {missing}")
    if int(record["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("unsupported feature schema version")
    token_count = len(record["labels"])
    if (
        len(record["external"]) != token_count
        or len(record["parametric"]) != token_count
    ):
        raise ValueError("feature record has inconsistent token dimensions")


def write_feature_record(
    directory: str | Path,
    record: Mapping[str, Any],
) -> Path:
    """Write one complete gzip JSON record and publish it atomically."""

    _validate_record(record)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = feature_path(root, str(record["id"]))
    temporary = destination.with_name(destination.name + ".part")
    try:
        with gzip.open(
            temporary,
            mode="wt",
            encoding="utf-8",
            newline="\n",
        ) as file:
            json.dump(
                dict(record),
                file,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            file.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def read_feature_record(
    path: str | Path,
    *,
    expected_id: str | None = None,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    with gzip.open(source, mode="rt", encoding="utf-8") as file:
        record = json.load(file)
    if not isinstance(record, dict):
        raise ValueError(f"{source} does not contain an object")
    _validate_record(record)
    if expected_id is not None and str(record["id"]) != str(expected_id):
        raise ValueError(f"{source} has the wrong response id")
    if (
        expected_fingerprint is not None
        and str(record["protocol_fingerprint"])
        != str(expected_fingerprint)
    ):
        raise ValueError(f"{source} has a stale protocol fingerprint")
    return record
