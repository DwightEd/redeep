"""Teacher-forced chat rendering and exact character-to-token alignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .config import DEFAULT_SYSTEM_PROMPT
from .data import locate_unique
from .schemas import RagTruthExample, TokenizedExample


def render_teacher_forced_text(
    example: RagTruthExample,
    tokenizer: Any,
    *,
    model_family: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Render a prompt and fixed RAGTruth response with the target chat template.

    Only the system/user messages and assistant generation header are rendered
    by the template.  The fixed response is then appended verbatim.  Real chat
    templates commonly apply Jinja's ``trim`` filter to message content; passing
    the response as an assistant message would therefore silently remove
    leading/trailing whitespace and invalidate RAGTruth character offsets.

    Qwen3's thinking mode is explicitly disabled; otherwise a Llama2-generated
    response would be placed under a semantically different assistant format.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example.prompt},
    ]
    family = (model_family or getattr(tokenizer, "name_or_path", "")).lower()
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if "qwen" in family:
        kwargs["enable_thinking"] = False
    rendered_prefix = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered_prefix, str):
        raise TypeError("apply_chat_template(..., tokenize=False) must return a string")
    return rendered_prefix + example.response


def align_teacher_forced_example(
    example: RagTruthExample,
    tokenizer: Any,
    *,
    model_family: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> TokenizedExample:
    """Render and tokenize one example, producing causal token mappings.

    The complete rendered sequence is tokenized exactly once with offset
    mapping.  A response token at absolute sequence position ``p`` maps to
    predictor position ``p - 1``.  Token labels use interval overlap with the
    unioned RAGTruth hallucination spans.
    """

    rendered = render_teacher_forced_text(
        example,
        tokenizer,
        model_family=model_family,
        system_prompt=system_prompt,
    )
    prompt_start, prompt_end = locate_unique(
        rendered, example.prompt, field=f"response {example.response_id} rendered prompt"
    )
    context_local_start, context_local_end = locate_unique(
        example.prompt,
        example.context,
        field=f"response {example.response_id} context",
    )
    context_start = prompt_start + context_local_start
    context_end = prompt_start + context_local_end
    if not example.response:
        raise ValueError(f"response {example.response_id} must not be empty")
    response_start = len(rendered) - len(example.response)
    response_end = len(rendered)
    if response_start < prompt_end or rendered[response_start:response_end] != example.response:
        raise ValueError(
            f"response {example.response_id} is not the verbatim final assistant continuation"
        )

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=False,
    )
    if not isinstance(encoded, Mapping):
        raise TypeError("Tokenizer output must be a mapping")
    if "offset_mapping" not in encoded:
        raise ValueError(
            "Tokenizer did not return offset_mapping; a fast tokenizer is required"
        )

    input_ids = _flatten(encoded.get("input_ids"), field="input_ids")
    offsets_raw = _flatten_offsets(encoded["offset_mapping"])
    attention_mask = _flatten(
        encoded.get("attention_mask", [1] * len(input_ids)),
        field="attention_mask",
    )
    special_mask = _flatten(
        encoded.get("special_tokens_mask", [0] * len(input_ids)),
        field="special_tokens_mask",
    )
    lengths = {
        "input_ids": len(input_ids),
        "offset_mapping": len(offsets_raw),
        "attention_mask": len(attention_mask),
        "special_tokens_mask": len(special_mask),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Tokenizer fields have inconsistent sequence lengths: {lengths}")

    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    response_positions: list[int] = []
    predictor_positions: list[int] = []
    response_offsets: list[tuple[int, int]] = []
    token_labels: list[int] = []
    context_positions: list[int] = []

    for position, ((start, end), token_id, is_special, attended) in enumerate(
        zip(offsets_raw, input_ids, special_mask, attention_mask, strict=True)
    ):
        usable = (
            bool(attended)
            and not bool(is_special)
            and token_id not in special_ids
            and end > start
        )
        if not usable:
            continue
        if _overlaps(start, end, context_start, context_end):
            context_positions.append(position)
        if not _overlaps(start, end, response_start, response_end):
            continue
        if position == 0:
            raise ValueError("The first sequence token cannot have a causal predictor")

        local_start = max(start, response_start) - response_start
        local_end = min(end, response_end) - response_start
        response_positions.append(position)
        predictor_positions.append(position - 1)
        response_offsets.append((local_start, local_end))
        token_labels.append(
            int(
                any(
                    _overlaps(local_start, local_end, span.start, span.end)
                    for span in example.spans
                )
            )
        )

    if not response_positions:
        raise ValueError(
            f"No response tokens were aligned for response {example.response_id}; "
            "use a fast tokenizer and verify its chat template"
        )
    if not context_positions:
        raise ValueError(
            f"No context tokens were aligned for response {example.response_id}"
        )

    return TokenizedExample(
        example=example,
        rendered_text=rendered,
        input_ids=tuple(input_ids),
        attention_mask=tuple(attention_mask),
        response_token_positions=tuple(response_positions),
        predictor_positions=tuple(predictor_positions),
        response_offsets=tuple(response_offsets),
        token_labels=tuple(token_labels),
        context_token_positions=tuple(context_positions),
    )


def _overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and end_a > start_b


def _flatten(value: Any, *, field: str) -> list[int]:
    if value is None:
        raise ValueError(f"Tokenizer output is missing {field}")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError(f"Tokenizer field {field} must be sequence-like")
    if value and isinstance(value[0], list | tuple):
        if len(value) != 1:
            raise ValueError(f"Batched tokenizer field {field} is not supported")
        value = list(value[0])
    return [int(item) for item in value]


def _flatten_offsets(value: Any) -> list[tuple[int, int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError("Tokenizer offset_mapping must be sequence-like")
    if value and _is_batched_offsets(value):
        if len(value) != 1:
            raise ValueError("Batched offset_mapping is not supported")
        value = value[0]
    offsets: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError(f"Invalid tokenizer offset: {item!r}")
        start, end = int(item[0]), int(item[1])
        if start < 0 or end < start:
            raise ValueError(f"Invalid tokenizer offset: {(start, end)!r}")
        offsets.append((start, end))
    return offsets


def _is_batched_offsets(value: list[Any]) -> bool:
    first = value[0]
    return (
        isinstance(first, list | tuple)
        and bool(first)
        and isinstance(first[0], list | tuple)
    )
