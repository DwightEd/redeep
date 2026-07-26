from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from redeep.artifacts import (
    ArtifactIntegrityError,
    atomic_write_feature_shard,
    atomic_write_json,
    find_completed_shards,
    hash_config,
    load_feature_shard,
    load_manifest,
    manifest_matches,
    shard_is_complete,
    shard_manifest_path,
    write_manifest,
)


def test_atomic_json_normalizes_numpy_and_nonfinite_values(tmp_path) -> None:
    path = tmp_path / "result.json"
    atomic_write_json(path, {"auc": np.float64(0.75), "missing": float("nan")})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"auc": 0.75, "missing": None}
    assert not list(tmp_path.glob("*.tmp"))


def test_self_hashed_manifest_detects_tampering(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    stored = write_manifest(path, {"schema_version": 1, "model": "test"})
    assert load_manifest(path) == stored
    assert manifest_matches(path, {"model": "test"})
    assert hash_config({"b": 2, "a": 1}) == hash_config({"a": 1, "b": 2})

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        load_manifest(path)


def test_feature_shard_is_a_verified_resume_unit(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {
            "response_id": ["a", "a", "b"],
            "token_index": [0, 1, 0],
            "token_label": [0, 1, 0],
            "redeep_score": [0.1, 0.9, 0.2],
        }
    )
    path = tmp_path / "features-000.parquet"
    expected = {"schema_version": 1, "model": "test", "mode": "standard"}
    stored = atomic_write_feature_shard(frame, path, expected)

    assert stored["artifact"]["rows"] == 3
    assert shard_manifest_path(path).is_file()
    assert shard_is_complete(path, expected_manifest=expected)
    pd.testing.assert_frame_equal(load_feature_shard(path), frame)
    assert find_completed_shards(tmp_path, expected_manifest=expected) == [path]
    assert not shard_is_complete(path, expected_manifest={"model": "other"})


def test_feature_shard_hash_detects_data_tampering(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"value": [1, 2, 3]})
    path = tmp_path / "features.parquet"
    atomic_write_feature_shard(frame, path, {"schema_version": 1})
    with path.open("ab") as handle:
        handle.write(b"tamper")

    assert not shard_is_complete(path)
    with pytest.raises(ArtifactIntegrityError, match="Incomplete"):
        load_feature_shard(path)


def test_orphan_parquet_without_manifest_is_not_complete(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"value": [1]})
    path = tmp_path / "orphan.parquet"
    frame.to_parquet(path, index=False)
    assert not shard_is_complete(path)
