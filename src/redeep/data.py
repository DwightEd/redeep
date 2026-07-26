"""RAGTruth loading, validation, and deterministic data partitioning.

The official RAGTruth release stores prompts/sources and model responses in two
JSONL files.  This module joins them without silently normalising text: exact
character identity is required for both hallucination spans and retrieved
contexts because downstream token labels are derived from character offsets.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .schemas import CharSpan, RagTruthExample

DEFAULT_GENERATOR_MODEL = "llama-2-7b-chat"
TASKS = ("QA", "Data2txt", "Summary")
OFFICIAL_RESPONSE_SHA256 = (
    "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073"
)
OFFICIAL_SOURCE_SHA256 = (
    "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b"
)

# Counts in the official RAGTruth release after filtering to llama-2-7b-chat.
EXPECTED_LLAMA2_COUNTS: dict[str, dict[str, int]] = {
    "train": {"QA": 839, "Data2txt": 883, "Summary": 793},
    "test": {"QA": 150, "Data2txt": 150, "Summary": 150},
}


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from *path*, reporting malformed rows precisely."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            yield row


def extract_context(source_record: Mapping[str, Any]) -> str:
    """Return the retrieved context exactly as it occurs in a RAGTruth prompt."""

    task = source_record.get("task_type")
    source_info = source_record.get("source_info")
    if task == "Summary":
        if not isinstance(source_info, str):
            raise ValueError("Summary source_info must be a string")
        context = source_info
    elif task == "QA":
        if not isinstance(source_info, Mapping) or not isinstance(
            source_info.get("passages"), str
        ):
            raise ValueError("QA source_info must contain a string 'passages' field")
        context = source_info["passages"]
    elif task == "Data2txt":
        if not isinstance(source_info, Mapping):
            raise ValueError("Data2txt source_info must be a mapping")
        # RAGTruth prompts were produced with Python's insertion-ordered dict
        # representation (single quotes, True/False/None), not JSON encoding.
        context = repr(dict(source_info))
    else:
        raise ValueError(f"Unsupported RAGTruth task_type: {task!r}")

    if not context:
        raise ValueError(f"{task} context must not be empty")
    return context


def locate_unique(text: str, substring: str, *, field: str) -> tuple[int, int]:
    """Locate *substring* exactly once and return its half-open interval."""

    first = text.find(substring)
    if first < 0:
        raise ValueError(f"{field} does not occur in its containing text")
    second = text.find(substring, first + 1)
    if second >= 0:
        raise ValueError(f"{field} occurs more than once; alignment is ambiguous")
    return first, first + len(substring)


def validate_label_spans(
    response: str,
    labels: Iterable[Mapping[str, Any]],
    *,
    response_id: str = "<unknown>",
) -> tuple[CharSpan, ...]:
    """Validate RAGTruth labels against raw response text.

    RAGTruth offsets are interpreted as half-open ``[start, end)`` intervals.
    Every provided label text must exactly equal the corresponding response
    slice.  This intentionally rejects "helpful" whitespace or Unicode
    normalisation, which would invalidate the official offsets.
    """

    spans: list[CharSpan] = []
    for index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            raise ValueError(f"Response {response_id} label {index} is not an object")
        start = label.get("start")
        end = label.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError(f"Response {response_id} label {index} has non-integer offsets")
        if start < 0 or end <= start or end > len(response):
            raise ValueError(
                f"Response {response_id} label {index} has invalid span "
                f"[{start}, {end}) for response length {len(response)}"
            )
        expected_text = response[start:end]
        label_text = label.get("text")
        if not isinstance(label_text, str) or label_text != expected_text:
            raise ValueError(
                f"Response {response_id} label {index} text does not match "
                f"response[{start}:{end}]"
            )
        spans.append(
            CharSpan(
                start=start,
                end=end,
                text=label_text,
                label_type=str(label.get("label_type", "")),
                implicit_true=bool(label.get("implicit_true", False)),
                due_to_null=bool(label.get("due_to_null", False)),
            )
        )
    return tuple(spans)


def merge_char_spans(
    spans: Iterable[CharSpan],
    *,
    response: str | None = None,
) -> tuple[CharSpan, ...]:
    """Return the union of overlapping or adjacent character spans."""

    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    if not ordered:
        return ()

    merged: list[CharSpan] = []
    group: list[CharSpan] = [ordered[0]]
    group_start = ordered[0].start
    group_end = ordered[0].end

    def emit() -> None:
        label_types = tuple(
            dict.fromkeys(span.label_type for span in group if span.label_type)
        )
        text = response[group_start:group_end] if response is not None else ""
        merged.append(
            CharSpan(
                start=group_start,
                end=group_end,
                text=text,
                label_type=" | ".join(label_types),
                implicit_true=any(span.implicit_true for span in group),
                due_to_null=any(span.due_to_null for span in group),
            )
        )

    for span in ordered[1:]:
        if span.start <= group_end:
            group.append(span)
            group_end = max(group_end, span.end)
            continue
        emit()
        group = [span]
        group_start = span.start
        group_end = span.end
    emit()
    return tuple(merged)


def load_ragtruth(
    response_path: str | Path,
    source_path: str | Path,
    *,
    generator_model: str = DEFAULT_GENERATOR_MODEL,
    splits: Iterable[str] | None = None,
    tasks: Iterable[str] = TASKS,
    qualities: Iterable[str] | None = None,
    validate_context: bool = True,
) -> tuple[RagTruthExample, ...]:
    """Join and validate official RAGTruth response/source JSONL files."""

    selected_splits = set(splits) if splits is not None else None
    selected_tasks = set(tasks)
    unknown_tasks = selected_tasks.difference(TASKS)
    if unknown_tasks:
        raise ValueError(f"Unsupported task(s): {sorted(unknown_tasks)}")
    selected_qualities = set(qualities) if qualities is not None else None

    sources: dict[str, dict[str, Any]] = {}
    for source_row in read_jsonl(source_path):
        source_id = str(source_row.get("source_id", ""))
        if not source_id:
            raise ValueError("A source row is missing source_id")
        if source_id in sources:
            raise ValueError(f"Duplicate source_id: {source_id}")
        sources[source_id] = source_row

    examples: list[RagTruthExample] = []
    response_ids: set[str] = set()
    for response_row in read_jsonl(response_path):
        if response_row.get("model") != generator_model:
            continue
        split = response_row.get("split")
        quality = response_row.get("quality")
        if selected_splits is not None and split not in selected_splits:
            continue
        if selected_qualities is not None and quality not in selected_qualities:
            continue

        response_id = str(response_row.get("id", ""))
        source_id = str(response_row.get("source_id", ""))
        if not response_id:
            raise ValueError("A selected response row is missing id")
        if response_id in response_ids:
            raise ValueError(f"Duplicate selected response id: {response_id}")
        response_ids.add(response_id)
        if source_id not in sources:
            raise ValueError(f"Response {response_id} references missing source {source_id}")

        source_row = sources[source_id]
        task = source_row.get("task_type")
        if task not in selected_tasks:
            continue
        prompt = source_row.get("prompt")
        response = response_row.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError(f"Response {response_id} has a non-string prompt or response")
        context = extract_context(source_row)
        if validate_context:
            locate_unique(prompt, context, field=f"response {response_id} context")

        labels = response_row.get("labels", [])
        if not isinstance(labels, list):
            raise ValueError(f"Response {response_id} labels must be a list")
        spans = validate_label_spans(response, labels, response_id=response_id)
        spans = merge_char_spans(spans, response=response)

        examples.append(
            RagTruthExample(
                response_id=response_id,
                source_id=source_id,
                generator_model=generator_model,
                split=str(split),
                quality=str(quality),
                task=str(task),
                prompt=prompt,
                context=context,
                response=response,
                spans=spans,
                source=str(source_row.get("source", "")),
                source_info=source_row.get("source_info"),
            )
        )

    return tuple(
        sorted(
            examples,
            key=lambda item: (
                0 if item.split == "train" else 1 if item.split == "test" else 2,
                TASKS.index(item.task),
                _natural_id_key(item.response_id),
            ),
        )
    )


def dataset_counts(
    examples: Iterable[RagTruthExample],
) -> dict[str, dict[str, int]]:
    """Count examples by split and task, including zeroes for known tasks."""

    counter = Counter((example.split, example.task) for example in examples)
    splits = sorted({example.split for example in examples})
    return {
        split: {task: counter[(split, task)] for task in TASKS}
        for split in splits
    }


def assert_official_llama2_counts(examples: Iterable[RagTruthExample]) -> None:
    """Fail if examples are not the complete official Llama2-7B-chat subset."""

    observed = dataset_counts(examples)
    if observed != EXPECTED_LLAMA2_COUNTS:
        raise ValueError(
            "Unexpected llama-2-7b-chat RAGTruth counts: "
            f"observed={observed}, expected={EXPECTED_LLAMA2_COUNTS}"
        )


def deterministic_calibration_dev_split(
    examples: Iterable[RagTruthExample],
    *,
    dev_per_task: int = 50,
    seed: int = 2024,
) -> tuple[tuple[RagTruthExample, ...], tuple[RagTruthExample, ...]]:
    """Partition train examples into calibration-fit and fixed dev sets.

    Selection uses a SHA-256 rank of ``seed/task/response_id``.  It is stable
    across input order, Python versions, and processes, unlike a mutable RNG
    stream.  Non-train examples are rejected to guard against test leakage.
    """

    examples = tuple(
        sorted(
            examples,
            key=lambda example: (TASKS.index(example.task), _natural_id_key(example.response_id)),
        )
    )
    non_train = [example.response_id for example in examples if example.split != "train"]
    if non_train:
        preview = ", ".join(non_train[:3])
        raise ValueError(f"Calibration/dev split accepts train examples only: {preview}")
    if dev_per_task < 1:
        raise ValueError("dev_per_task must be positive")

    dev_ids: set[str] = set()
    for task in TASKS:
        task_examples = [example for example in examples if example.task == task]
        if len(task_examples) < dev_per_task:
            raise ValueError(
                f"Task {task} has {len(task_examples)} train examples, "
                f"fewer than dev_per_task={dev_per_task}"
            )
        ranked = sorted(
            task_examples,
            key=lambda example: (
                hashlib.sha256(
                    f"{seed}\0{task}\0{example.response_id}".encode()
                ).digest(),
                _natural_id_key(example.response_id),
            ),
        )
        dev_ids.update(example.response_id for example in ranked[:dev_per_task])

    calibration = tuple(example for example in examples if example.response_id not in dev_ids)
    dev = tuple(example for example in examples if example.response_id in dev_ids)
    return calibration, dev


def _natural_id_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)
