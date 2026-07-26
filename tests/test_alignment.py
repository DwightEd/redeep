from __future__ import annotations

import re

import pytest

from redeep.alignment import align_teacher_forced_example
from redeep.schemas import CharSpan, RagTruthExample


class FakeChatTokenizer:
    """Character tokenizer with explicit chat-control special tokens."""

    name_or_path = "Qwen3-8B"
    controls = (
        "<BOS>",
        "<SYSTEM>",
        "</SYSTEM>",
        "<USER>",
        "</USER>",
        "<ASSISTANT>",
        "</ASSISTANT>",
    )
    all_special_ids = [900 + index for index in range(len(controls))]

    def __init__(self):
        self.tokenize_calls = 0
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return (
            f"<BOS><SYSTEM>{messages[0]['content']}</SYSTEM>"
            f"<USER>{messages[1]['content']}</USER>"
            f"<ASSISTANT>{messages[2]['content']}</ASSISTANT>"
        )

    def __call__(self, text, **kwargs):
        self.tokenize_calls += 1
        assert kwargs["add_special_tokens"] is False
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["truncation"] is False
        pattern = "|".join(re.escape(control) for control in self.controls)
        input_ids = []
        offsets = []
        special_mask = []
        cursor = 0
        for match in re.finditer(pattern, text):
            for position in range(cursor, match.start()):
                input_ids.append(ord(text[position]))
                offsets.append((position, position + 1))
                special_mask.append(0)
            control_index = self.controls.index(match.group())
            input_ids.append(self.all_special_ids[control_index])
            offsets.append((match.start(), match.end()))
            special_mask.append(1)
            cursor = match.end()
        for position in range(cursor, len(text)):
            input_ids.append(ord(text[position]))
            offsets.append((position, position + 1))
            special_mask.append(0)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "offset_mapping": offsets,
            "special_tokens_mask": special_mask,
        }


def _example(*, prompt="Question\nCTX\noutput:", context="CTX"):
    return RagTruthExample(
        response_id="r1",
        source_id="s1",
        generator_model="llama-2-7b-chat",
        split="test",
        quality="good",
        task="QA",
        prompt=prompt,
        context=context,
        response="ABCDE",
        spans=(CharSpan(1, 3, "BC"),),
    )


def test_one_pass_alignment_predictor_positions_and_half_open_labels():
    tokenizer = FakeChatTokenizer()

    aligned = align_teacher_forced_example(_example(), tokenizer)

    assert tokenizer.tokenize_calls == 1
    assert tokenizer.template_kwargs["enable_thinking"] is False
    assert tokenizer.template_kwargs["tokenize"] is False
    assert aligned.response_offsets == ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
    assert aligned.token_labels == (0, 1, 1, 0, 0)
    assert aligned.predictor_positions == tuple(
        position - 1 for position in aligned.response_token_positions
    )
    assert not set(aligned.response_token_positions).intersection(
        position
        for position, token_id in enumerate(aligned.input_ids)
        if token_id in tokenizer.all_special_ids
    )


def test_context_mask_contains_only_context_characters():
    tokenizer = FakeChatTokenizer()
    aligned = align_teacher_forced_example(_example(), tokenizer)

    rendered_chars = "".join(
        chr(aligned.input_ids[position]) for position in aligned.context_token_positions
    )
    assert rendered_chars == "CTX"


def test_qwen_can_be_selected_by_explicit_family():
    tokenizer = FakeChatTokenizer()
    tokenizer.name_or_path = "unknown"
    align_teacher_forced_example(_example(), tokenizer, model_family="qwen3")
    assert tokenizer.template_kwargs["enable_thinking"] is False


def test_ambiguous_context_is_rejected():
    tokenizer = FakeChatTokenizer()
    example = _example(prompt="CTX and CTX", context="CTX")
    with pytest.raises(ValueError, match="more than once"):
        align_teacher_forced_example(example, tokenizer)
