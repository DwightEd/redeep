#!/usr/bin/env python3
"""Adopt pre-fix feature shards after the verbatim-response alignment fix.

This is a narrow, auditable metadata migration. It never changes Parquet
features. It accepts only artifacts produced by the known commits immediately
preceding the alignment fix, with identical config, data, model, tokenizer,
and feature implementation dependencies.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from redeep.alignment import align_teacher_forced_example, render_teacher_forced_text
from redeep.artifacts import (
    MANIFEST_HASH_KEY,
    atomic_write_json,
    load_manifest,
    manifest_hash,
    shard_is_complete,
    shard_manifest_path,
    write_manifest,
)
from redeep.config import load_config
from redeep.pipeline import (
    EVAL_SPLITS,
    _current_git_commit,
    _experiment_dependencies,
    _feature_rows,
    _implementation_fingerprint,
    load_experiment_examples,
    model_dir,
)
from redeep.utils import sha256_json

# 72807fb and 5fd8116 share the original assistant-message rendering.
# 6cdd919 appends the raw response; cd74aeb additionally scopes prompt lookup.
SOURCE_IMPLEMENTATIONS = {
    "72807fb82a82a3f2053d8b2a7606146b0e808f1b": (
        "d41a53712acefaa2049723ae8d845dab5abbb68c97d5471a6083ba3cbd02143f"
    ),
    "5fd811663cd9e636edbfecd8471334144c659516": (
        "d41a53712acefaa2049723ae8d845dab5abbb68c97d5471a6083ba3cbd02143f"
    ),
    "6cdd9194ad6ebf6280ccebebc667835e8e97f29c": (
        "e8c698712a8fcbd0ba08ac295227411df39872dc94fd120af9c6639fceac549e"
    ),
    "cd74aeb5b9c9f75e3a9b58d54523ae90e3282e4d": (
        "aadc01baa9c72bd9e16437ec9186843c523f63d06fc57e9cc3382d54db2f59bb"
    ),
}
PRE_VERBATIM_RESPONSE_COMMITS = frozenset(tuple(SOURCE_IMPLEMENTATIONS)[:2])
TARGET_IMPLEMENTATION = (
    "aadc01baa9c72bd9e16437ec9186843c523f63d06fc57e9cc3382d54db2f59bb"
)
ADOPTION_KIND = "verbatim-response-alignment-compatible-resume-v1"


def adopt_alignment_resume_artifacts(
    config_path: str | Path,
    model_key: str,
    *,
    apply: bool,
) -> dict[str, Any]:
    """Validate and optionally adopt compatible completed feature shards."""

    config = load_config(config_path)
    root = model_dir(config, model_key)
    copy_path = root / "copy_heads.json"
    if not copy_path.is_file():
        raise FileNotFoundError(f"Missing Copying Head artifact: {copy_path}")

    target_commit = _current_git_commit()
    target_dependencies = _experiment_dependencies(config, model_key)
    target_dependencies_sha256 = sha256_json(target_dependencies)
    target_implementation = _implementation_fingerprint().get("aggregate_sha256")
    if target_implementation != TARGET_IMPLEMENTATION:
        raise ValueError(
            "Current feature implementation is not the reviewed alignment-fix "
            f"implementation: observed={target_implementation}, "
            f"expected={TARGET_IMPLEMENTATION}"
        )

    source_copy = load_manifest(copy_path, verify=True)
    if _is_current_copy_manifest(
        source_copy,
        target_commit=target_commit,
        target_dependencies_sha256=target_dependencies_sha256,
    ):
        return {
            "status": "already_adopted",
            "model_key": model_key,
            "target_git_commit": target_commit,
            "eligible_feature_shards": 0,
            "adopted_feature_shards": 0,
            "backup": None,
        }

    source_commit = _validate_source_copy_manifest(
        source_copy,
        config_hash=config.digest,
        model_key=model_key,
        target_dependencies=target_dependencies,
    )
    if source_commit in PRE_VERBATIM_RESPONSE_COMMITS and (
        config.models[model_key].family != "llama"
        or config.dataset.generator_model != "llama-2-7b-chat"
    ):
        raise ValueError(
            "Pre-verbatim-response adoption is reviewed only for the Llama scorer "
            "on the fixed llama-2-7b-chat RAGTruth subset"
        )
    source_dependencies = source_copy["dependencies"]
    source_dependencies_sha256 = sha256_json(source_dependencies)
    source_copy_sha256 = sha256_json(source_copy)

    adoption = {
        "kind": ADOPTION_KIND,
        "source_git_commit": source_commit,
        "source_copy_heads_manifest_sha256": source_copy[MANIFEST_HASH_KEY],
        "source_dependencies_sha256": source_dependencies_sha256,
        "source_copy_heads_sha256": source_copy_sha256,
        "target_git_commit": target_commit,
        "verification": (
            "Parquet hashes preserved; config, data, model, tokenizer, and feature "
            "implementation identities matched. Pre-fix shards with response outer "
            "whitespace are rejected."
        ),
    }

    target_copy = dict(source_copy)
    target_copy.pop(MANIFEST_HASH_KEY, None)
    target_copy.update(
        {
            "git_commit": target_commit,
            "dependencies": target_dependencies,
            "dependencies_sha256": target_dependencies_sha256,
            "compatibility_adoption": adoption,
        }
    )
    stored_target_copy = _stored_manifest(target_copy)
    target_copy_sha256 = sha256_json(stored_target_copy)

    examples = {
        example.response_id: example for example in load_experiment_examples(config)
    }
    tokenizer = _load_tokenizer(config.models[model_key])
    shard_updates: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    already_adopted = 0
    incomplete_shards = 0
    for eval_split in EVAL_SPLITS:
        split_dir = root / "features" / eval_split
        if not split_dir.is_dir():
            continue
        for parquet_path in sorted(split_dir.glob("*.parquet")):
            if not shard_is_complete(parquet_path):
                incomplete_shards += 1
                continue
            sidecar = shard_manifest_path(parquet_path)
            source_manifest = load_manifest(sidecar, verify=True)
            if _is_current_shard_manifest(
                source_manifest,
                target_commit=target_commit,
                target_dependencies_sha256=target_dependencies_sha256,
                target_copy_sha256=target_copy_sha256,
            ):
                already_adopted += 1
                continue
            response_id = str(source_manifest.get("response_id", ""))
            _validate_source_shard_manifest(
                source_manifest,
                source_commit=source_commit,
                source_dependencies_sha256=source_dependencies_sha256,
                source_copy_sha256=source_copy_sha256,
                config_hash=config.digest,
                model_key=model_key,
                eval_split=eval_split,
            )
            example = examples.get(response_id)
            if example is None:
                raise ValueError(
                    f"Feature shard {parquet_path} has unknown response_id={response_id!r}"
                )
            if (
                source_commit in PRE_VERBATIM_RESPONSE_COMMITS
                and example.response != example.response.strip()
            ):
                raise ValueError(
                    f"Cannot adopt pre-fix shard for response {response_id}: "
                    "the raw response has outer whitespace"
                )
            _verify_shard_equivalence(
                parquet_path,
                example,
                tokenizer,
                model_family=config.models[model_key].family,
                system_prompt=config.dataset.system_prompt,
                eval_split=eval_split,
                source_commit=source_commit,
            )

            target_manifest = {
                "schema_version": source_manifest["schema_version"],
                "config_hash": config.digest,
                "git_commit": target_commit,
                "model_key": model_key,
                "model_name": config.models[model_key].name,
                "eval_split": eval_split,
                "response_id": response_id,
                "dependencies_sha256": target_dependencies_sha256,
                "copy_heads_sha256": target_copy_sha256,
                "artifact": source_manifest["artifact"],
                "compatibility_adoption": {
                    **adoption,
                    "source_feature_manifest_sha256": source_manifest[
                        MANIFEST_HASH_KEY
                    ],
                },
            }
            shard_updates.append((sidecar, source_manifest, target_manifest))

    split_manifest_updates = _plan_split_manifest_updates(
        root,
        source_commit=source_commit,
        source_dependencies_sha256=source_dependencies_sha256,
        source_copy_sha256=source_copy_sha256,
        target_commit=target_commit,
        target_dependencies=target_dependencies,
        target_dependencies_sha256=target_dependencies_sha256,
        target_copy_sha256=target_copy_sha256,
        adoption=adoption,
    )
    backup_path = (
        root
        / (
            f"alignment-resume-backup-{source_commit[:8]}-to-"
            f"{target_commit[:8]}.json"
        )
    )
    report = {
        "status": "eligible" if not apply else "adopted",
        "model_key": model_key,
        "source_git_commit": source_commit,
        "target_git_commit": target_commit,
        "eligible_feature_shards": len(shard_updates),
        "already_adopted_feature_shards": already_adopted,
        "incomplete_feature_shards": incomplete_shards,
        "eligible_split_manifests": len(split_manifest_updates),
        "adopted_feature_shards": 0,
        "backup": str(backup_path),
    }
    if not apply:
        return report

    backup_payload = {
        "kind": ADOPTION_KIND,
        "source_git_commit": source_commit,
        "target_git_commit": target_commit,
        "copy_heads": {
            "path": str(copy_path.relative_to(root)),
            "manifest": source_copy,
        },
        "feature_sidecars": [
            {
                "path": str(path.relative_to(root)),
                "manifest": source_manifest,
            }
            for path, source_manifest, _target_manifest in shard_updates
        ],
        "split_manifests": [
            {
                "path": str(path.relative_to(root)),
                "manifest": source_manifest,
            }
            for path, source_manifest, _target_manifest in split_manifest_updates
        ],
    }
    if backup_path.exists():
        with backup_path.open("r", encoding="utf-8") as handle:
            existing_backup = json.load(handle)
        expected_backup_identity = {
            "kind": ADOPTION_KIND,
            "source_git_commit": source_commit,
            "target_git_commit": target_commit,
        }
        observed_backup_identity = {
            key: existing_backup.get(key) for key in expected_backup_identity
        }
        if observed_backup_identity != expected_backup_identity:
            raise ValueError(f"Existing adoption backup differs: {backup_path}")
        backed_up_copy = existing_backup.get("copy_heads", {}).get("manifest", {})
        if (
            backed_up_copy.get(MANIFEST_HASH_KEY)
            != source_copy.get(MANIFEST_HASH_KEY)
        ):
            raise ValueError(
                f"Existing adoption backup has a different Copying Head artifact: "
                f"{backup_path}"
            )
    else:
        atomic_write_json(backup_path, backup_payload)

    # Publish the Copying Head manifest last. Until then, the normal pipeline
    # rejects the partially migrated state and cannot mix old/new provenance.
    for path, _source_manifest, target_manifest in shard_updates:
        write_manifest(path, target_manifest)
    for path, _source_manifest, target_manifest in split_manifest_updates:
        write_manifest(path, target_manifest)
    written_copy = write_manifest(copy_path, target_copy)
    if written_copy != stored_target_copy:
        raise RuntimeError("Stored Copying Head manifest differs from migration plan")

    report["adopted_feature_shards"] = len(shard_updates)
    return report


def _validate_source_copy_manifest(
    payload: Mapping[str, Any],
    *,
    config_hash: str,
    model_key: str,
    target_dependencies: Mapping[str, Any],
) -> str:
    source_commit = str(payload.get("git_commit", ""))
    if source_commit not in SOURCE_IMPLEMENTATIONS:
        raise ValueError(
            f"Copying Head artifact commit {source_commit!r} is not an approved "
            "alignment-resume source"
        )
    if payload.get("config_hash") != config_hash:
        raise ValueError("Copying Head config hash differs from the current config")
    if payload.get("model_key") != model_key:
        raise ValueError("Copying Head model key differs from the selected model")
    source_dependencies = payload.get("dependencies")
    if not isinstance(source_dependencies, Mapping):
        raise ValueError("Copying Head artifact has no dependency mapping")
    if payload.get("dependencies_sha256") != sha256_json(source_dependencies):
        raise ValueError("Copying Head dependency hash is invalid")
    if source_dependencies.get("git_commit") != source_commit:
        raise ValueError("Copying Head commit and dependency commit differ")
    implementation = source_dependencies.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("Copying Head dependencies have no implementation identity")
    expected_implementation = SOURCE_IMPLEMENTATIONS[source_commit]
    if implementation.get("aggregate_sha256") != expected_implementation:
        raise ValueError(
            "Source implementation hash is not the reviewed implementation for "
            f"{source_commit}"
        )
    if _non_code_dependencies(source_dependencies) != _non_code_dependencies(
        target_dependencies
    ):
        raise ValueError(
            "Config, data, model, or tokenizer dependencies changed; existing "
            "features cannot be adopted"
        )
    return source_commit


def _verify_shard_equivalence(
    parquet_path: Path,
    example: Any,
    tokenizer: Any,
    *,
    model_family: str,
    system_prompt: str,
    eval_split: str,
    source_commit: str,
) -> None:
    """Prove that the old shard and current alignment use identical predictors."""

    current = align_teacher_forced_example(
        example,
        tokenizer,
        model_family=model_family,
        system_prompt=system_prompt,
    )
    expected_metadata = _feature_rows(current, {}, eval_split=eval_split)

    import pandas as pd

    observed = pd.read_parquet(
        parquet_path,
        columns=list(expected_metadata),
    )
    mismatched_columns = [
        name
        for name, expected_values in expected_metadata.items()
        if observed[name].tolist() != expected_values
    ]
    if mismatched_columns:
        raise ValueError(
            f"Feature shard {parquet_path} does not match current token alignment "
            f"for columns {mismatched_columns}"
        )

    if source_commit not in PRE_VERBATIM_RESPONSE_COMMITS:
        return
    legacy_rendered = _render_legacy_assistant_message(
        example,
        tokenizer,
        model_family=model_family,
        system_prompt=system_prompt,
    )
    current_rendered = render_teacher_forced_text(
        example,
        tokenizer,
        model_family=model_family,
        system_prompt=system_prompt,
    )
    if not legacy_rendered.startswith(current_rendered):
        raise ValueError(
            f"Feature shard {parquet_path} cannot be adopted because the legacy "
            "assistant-message render does not equal the current generation prefix "
            "plus verbatim response"
        )

    legacy_tokens = _tokenize_for_equivalence(tokenizer, legacy_rendered)
    current_tokens = _tokenize_for_equivalence(tokenizer, current_rendered)
    for field, current_values in current_tokens.items():
        legacy_values = legacy_tokens[field]
        if legacy_values[: len(current_values)] != current_values:
            raise ValueError(
                f"Feature shard {parquet_path} cannot be adopted because legacy "
                f"and current tokenization differ in {field}"
            )
    last_predictor = max(current.predictor_positions)
    if legacy_tokens["input_ids"][: last_predictor + 1] != list(
        current.input_ids[: last_predictor + 1]
    ):
        raise ValueError(
            f"Feature shard {parquet_path} has different causal predictor inputs"
        )


def _render_legacy_assistant_message(
    example: Any,
    tokenizer: Any,
    *,
    model_family: str,
    system_prompt: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example.prompt},
        {"role": "assistant", "content": example.response},
    ]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if "qwen" in model_family.lower():
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered, str):
        raise TypeError("Legacy chat template render must be a string")
    return rendered


def _tokenize_for_equivalence(tokenizer: Any, rendered: str) -> dict[str, list[Any]]:
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=False,
    )
    result: dict[str, list[Any]] = {}
    for field in (
        "input_ids",
        "attention_mask",
        "offset_mapping",
        "special_tokens_mask",
    ):
        value = encoded[field]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list) and field != "offset_mapping":
            if len(value) != 1:
                raise ValueError(f"Batched tokenizer field is unsupported: {field}")
            value = value[0]
        if (
            field == "offset_mapping"
            and value
            and isinstance(value[0], list)
            and value[0]
            and isinstance(value[0][0], list | tuple)
        ):
            if len(value) != 1:
                raise ValueError("Batched offset_mapping is unsupported")
            value = value[0]
        result[field] = [
            tuple(item) if field == "offset_mapping" else int(item)
            for item in value
        ]
    return result


def _load_tokenizer(model_config: Any) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_path,
        trust_remote_code=model_config.trust_remote_code,
        local_files_only=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for alignment adoption")
    return tokenizer


def _validate_source_shard_manifest(
    payload: Mapping[str, Any],
    *,
    source_commit: str,
    source_dependencies_sha256: str,
    source_copy_sha256: str,
    config_hash: str,
    model_key: str,
    eval_split: str,
) -> None:
    expected = {
        "schema_version": 1,
        "config_hash": config_hash,
        "git_commit": source_commit,
        "model_key": model_key,
        "eval_split": eval_split,
        "dependencies_sha256": source_dependencies_sha256,
        "copy_heads_sha256": source_copy_sha256,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Legacy feature manifest is incompatible: {mismatches}")
    artifact = payload.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("Legacy feature manifest has no artifact identity")


def _plan_split_manifest_updates(
    root: Path,
    *,
    source_commit: str,
    source_dependencies_sha256: str,
    source_copy_sha256: str,
    target_commit: str,
    target_dependencies: Mapping[str, Any],
    target_dependencies_sha256: str,
    target_copy_sha256: str,
    adoption: Mapping[str, Any],
) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    updates: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for eval_split in EVAL_SPLITS:
        path = root / "features" / eval_split / "manifest.json"
        if not path.is_file():
            continue
        source_manifest = load_manifest(path, verify=True)
        if (
            source_manifest.get("git_commit") == target_commit
            and source_manifest.get("dependencies_sha256")
            == target_dependencies_sha256
            and source_manifest.get("copy_heads_sha256") == target_copy_sha256
        ):
            continue
        expected = {
            "git_commit": source_commit,
            "dependencies_sha256": source_dependencies_sha256,
            "copy_heads_sha256": source_copy_sha256,
        }
        mismatches = {
            key: (source_manifest.get(key), value)
            for key, value in expected.items()
            if source_manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Legacy split manifest is incompatible: {mismatches}")
        target_manifest = dict(source_manifest)
        target_manifest.pop(MANIFEST_HASH_KEY, None)
        target_manifest.update(
            {
                "git_commit": target_commit,
                "dependencies": target_dependencies,
                "dependencies_sha256": target_dependencies_sha256,
                "copy_heads_sha256": target_copy_sha256,
                "compatibility_adoption": {
                    **adoption,
                    "source_split_manifest_sha256": source_manifest[
                        MANIFEST_HASH_KEY
                    ],
                },
            }
        )
        updates.append((path, source_manifest, target_manifest))
    return updates


def _is_current_copy_manifest(
    payload: Mapping[str, Any],
    *,
    target_commit: str,
    target_dependencies_sha256: str,
) -> bool:
    return (
        payload.get("git_commit") == target_commit
        and payload.get("dependencies_sha256") == target_dependencies_sha256
    )


def _is_current_shard_manifest(
    payload: Mapping[str, Any],
    *,
    target_commit: str,
    target_dependencies_sha256: str,
    target_copy_sha256: str,
) -> bool:
    return (
        payload.get("git_commit") == target_commit
        and payload.get("dependencies_sha256") == target_dependencies_sha256
        and payload.get("copy_heads_sha256") == target_copy_sha256
    )


def _non_code_dependencies(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if key not in {"git_commit", "implementation"}
    }


def _stored_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    stored = dict(payload)
    stored.pop(MANIFEST_HASH_KEY, None)
    stored[MANIFEST_HASH_KEY] = manifest_hash(stored)
    return stored


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adopt compatible ReDeEP shards created before the verbatim-response "
            "alignment fix. Dry-run is the default."
        )
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write adopted sidecar manifests after all compatibility checks pass.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = adopt_alignment_resume_artifacts(
        args.config,
        args.model,
        apply=args.apply,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
