"""RAGTruth loading and causal token alignment for ReDeEP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TASKS = ("QA", "Summary", "Data2txt")
DEFAULT_EXCLUDED_QUALITIES = frozenset(
    {"incorrect_refusal", "truncated"}
)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"expected an object at {path}:{line_number}"
                )
            yield line_number, value


def _require_fields(
    value: Mapping[str, Any],
    fields: Sequence[str],
    *,
    path: Path,
    line_number: int,
) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(
            f"missing fields {missing} at {path}:{line_number}"
        )


def _validate_labels(
    response_id: str,
    response: str,
    labels: Any,
) -> list[dict[str, Any]]:
    if not isinstance(labels, list):
        raise ValueError(f"response {response_id!r} has non-list labels")
    validated: list[dict[str, Any]] = []
    for label_index, label in enumerate(labels):
        if not isinstance(label, dict):
            raise ValueError(
                f"response {response_id!r} label {label_index} is not an object"
            )
        if "start" not in label or "end" not in label:
            raise ValueError(
                f"response {response_id!r} label {label_index} has no span"
            )
        start = label["start"]
        end = label["end"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(response)
        ):
            raise ValueError(
                f"response {response_id!r} label {label_index} has "
                "an invalid half-open span"
            )
        if "text" in label and response[start:end] != label["text"]:
            raise ValueError(
                f"response {response_id!r} label {label_index} text "
                "does not match its span"
            )
        validated.append(dict(label))
    return validated


def load_ragtruth_samples(
    *,
    data_dir: str | Path,
    split: str,
    generator_model: str,
    tasks: Sequence[str] = TASKS,
    excluded_qualities: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Load one immutable generator subset and attach its source metadata."""

    data_path = Path(data_dir)
    source_path = data_path / "source_info.jsonl"
    response_path = data_path / "response.jsonl"
    requested_tasks = tuple(tasks)
    unsupported = sorted(set(requested_tasks) - set(TASKS))
    if unsupported:
        raise ValueError(f"unsupported RAGTruth tasks: {unsupported}")

    sources: dict[str, dict[str, Any]] = {}
    for line_number, source in _read_jsonl(source_path):
        _require_fields(
            source,
            ("source_id", "task_type", "prompt"),
            path=source_path,
            line_number=line_number,
        )
        source_id = str(source["source_id"])
        if source_id in sources:
            raise ValueError(f"duplicate source_id {source_id!r}")
        sources[source_id] = source

    samples: list[dict[str, Any]] = []
    requested_generator = generator_model.lower()
    seen_ids: set[str] = set()
    for line_number, item in _read_jsonl(response_path):
        _require_fields(
            item,
            (
                "id",
                "source_id",
                "model",
                "split",
                "quality",
                "response",
                "labels",
            ),
            path=response_path,
            line_number=line_number,
        )
        if str(item["split"]) != split:
            continue
        if str(item["model"]).lower() != requested_generator:
            continue
        if str(item["quality"]) in excluded_qualities:
            continue

        source_id = str(item["source_id"])
        try:
            source = sources[source_id]
        except KeyError as error:
            raise ValueError(
                f"response {item['id']!r} references unknown "
                f"source_id {source_id!r}"
            ) from error
        task_type = str(source["task_type"])
        if task_type not in requested_tasks:
            continue

        response_id = str(item["id"])
        if response_id in seen_ids:
            raise ValueError(f"duplicate response id {response_id!r}")
        seen_ids.add(response_id)
        response_text = item["response"]
        if not isinstance(response_text, str):
            raise ValueError(
                f"response {response_id!r} has non-string text"
            )
        labels = _validate_labels(
            response_id, response_text, item["labels"]
        )
        samples.append(
            {
                **item,
                "id": response_id,
                "source_id": source_id,
                "response": response_text,
                "labels": labels,
                "task_type": task_type,
                "prompt": str(source["prompt"]),
                "source_info": source.get("source_info"),
                "response_label": int(bool(labels)),
            }
        )
    return samples


def _single_sequence(value: Any, field_name: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError(
            f"tokenizer field {field_name!r} must contain one example"
        )
    sequence = value[0]
    if not isinstance(sequence, (list, tuple)):
        raise ValueError(
            f"tokenizer field {field_name!r} is not a sequence"
        )
    return list(sequence)


def render_redeep_prefix(
    tokenizer: Any,
    *,
    prompt: str,
    prompt_char_limit: int | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Render the official ReDeEP system/user chat prefix."""

    if prompt_char_limit is not None and prompt_char_limit <= 0:
        raise ValueError("prompt_char_limit must be positive or None")
    rendered_prompt = (
        prompt if prompt_char_limit is None else prompt[:prompt_char_limit]
    )
    prefix = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rendered_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **dict(chat_template_kwargs or {}),
    )
    if not isinstance(prefix, str):
        raise ValueError("chat template did not return text")
    return prefix


def build_teacher_forced_encoding(
    tokenizer: Any,
    *,
    prompt: str,
    response: str,
    labels: Sequence[Mapping[str, Any]],
    prompt_char_limit: int | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Tokenize a fixed response and align its spans to causal score positions."""

    prefix = render_redeep_prefix(
        tokenizer,
        prompt=prompt,
        prompt_char_limit=prompt_char_limit,
        system_prompt=system_prompt,
        chat_template_kwargs=chat_template_kwargs,
    )
    full_text = prefix + response
    full_encoding = tokenizer(
        [full_text],
        return_offsets_mapping=True,
        truncation=False,
    )
    prefix_encoding = tokenizer([prefix], truncation=False)
    input_ids = [
        int(token_id)
        for token_id in _single_sequence(
            full_encoding["input_ids"], "input_ids"
        )
    ]
    separately_tokenized_prefix_ids = [
        int(token_id)
        for token_id in _single_sequence(
            prefix_encoding["input_ids"], "prefix input_ids"
        )
    ]

    raw_offsets = _single_sequence(
        full_encoding["offset_mapping"], "offset_mapping"
    )
    if len(raw_offsets) != len(input_ids):
        raise ValueError("token IDs and offsets have different lengths")
    full_offsets: list[tuple[int, int]] = []
    for token_index, raw_offset in enumerate(raw_offsets):
        if (
            not isinstance(raw_offset, (list, tuple))
            or len(raw_offset) != 2
        ):
            raise ValueError(f"token offset {token_index} is not a pair")
        start, end = int(raw_offset[0]), int(raw_offset[1])
        if not 0 <= start <= end <= len(full_text):
            raise ValueError(
                f"token offset {token_index} lies outside the input"
            )
        full_offsets.append((start, end))

    response_start = len(prefix)
    response_end = len(full_text)
    response_token_positions: list[int] = []
    response_offsets: list[tuple[int, int]] = []
    for token_index, (start, end) in enumerate(full_offsets):
        if end <= start:
            continue
        if start < response_end and end > response_start:
            response_token_positions.append(token_index)
            response_offsets.append(
                (
                    max(start, response_start) - response_start,
                    min(end, response_end) - response_start,
                )
            )
    if not response_token_positions and response:
        raise ValueError("the tokenizer produced no response tokens")
    if any(position <= 0 for position in response_token_positions):
        raise ValueError("a response token has no causal predecessor")
    prefix_token_count = (
        response_token_positions[0]
        if response_token_positions
        else len(input_ids)
    )
    prefix_ids = input_ids[:prefix_token_count]
    boundary_retokenized = (
        input_ids[: len(separately_tokenized_prefix_ids)]
        != separately_tokenized_prefix_ids
    )

    validated_labels = _validate_labels(
        "<teacher-forced-response>", response, list(labels)
    )
    spans = [
        (int(label["start"]), int(label["end"]))
        for label in validated_labels
    ]
    token_labels = [
        int(
            any(
                token_start < span_end and token_end > span_start
                for span_start, span_end in spans
            )
        )
        for token_start, token_end in response_offsets
    ]
    return {
        "prefix": prefix,
        "full_text": full_text,
        "input_ids": input_ids,
        "prefix_ids": prefix_ids,
        "separately_tokenized_prefix_ids": separately_tokenized_prefix_ids,
        "boundary_retokenized": boundary_retokenized,
        "full_offsets": full_offsets,
        "response_token_positions": response_token_positions,
        "score_positions": [
            position - 1 for position in response_token_positions
        ],
        "response_offsets": response_offsets,
        "token_labels": token_labels,
    }
