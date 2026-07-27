"""Small deterministic helpers shared by the CLI and its tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


def protocol_fingerprint(protocol: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(protocol),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def response_ids_sha256(response_ids: Sequence[str]) -> str:
    """Hash a response-ID cohort independent of evaluation row order."""

    digest = hashlib.sha256()
    for response_id in sorted(str(value) for value in response_ids):
        encoded = response_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def select_longest_per_task(
    samples: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
) -> list[Mapping[str, Any]]:
    selected = []
    for task in tasks:
        candidates = [
            sample
            for sample in samples
            if str(sample["task_type"]) == str(task)
        ]
        if not candidates:
            raise ValueError(f"no samples are available for task {task}")
        selected.append(
            max(
                candidates,
                key=lambda sample: (
                    len(str(sample["prompt"]))
                    + len(str(sample["response"])),
                    str(sample["id"]),
                ),
            )
        )
    return selected


def format_markdown_metrics(metrics: Mapping[str, Any]) -> str:
    task_values = metrics["per_task"]
    values = [
        float(task_values["QA"]["auroc"]) * 100.0,
        float(task_values["Summary"]["auroc"]) * 100.0,
        float(task_values["Data2txt"]["auroc"]) * 100.0,
        float(metrics["task_macro_auroc"]) * 100.0,
        float(metrics["support_weighted_task_auroc"]) * 100.0,
        float(metrics["overall"]["auroc"]) * 100.0,
    ]
    return "\n".join(
        [
            (
                "| QA | Summary | Data2txt | Task-macro | "
                "Support-weighted task | Pooled |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            "| " + " | ".join(f"{value:.2f}" for value in values) + " |",
        ]
    )
