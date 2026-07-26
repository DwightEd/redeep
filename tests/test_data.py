from __future__ import annotations

import json
from collections import Counter

import pytest

from redeep.data import (
    EXPECTED_LLAMA2_COUNTS,
    assert_official_llama2_counts,
    dataset_counts,
    deterministic_calibration_dev_split,
    extract_context,
    load_ragtruth,
    merge_char_spans,
    validate_label_spans,
)
from redeep.schemas import CharSpan, RagTruthExample


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_loads_three_tasks_filters_model_and_merges_spans(tmp_path):
    info_by_task = {
        "QA": {"question": "q", "passages": "passage context"},
        "Data2txt": {"name": "A", "open": True, "value": None},
        "Summary": "summary context",
    }
    sources = []
    responses = []
    for index, (task, source_info) in enumerate(info_by_task.items()):
        context = extract_context({"task_type": task, "source_info": source_info})
        sources.append(
            {
                "source_id": str(index),
                "task_type": task,
                "source": "fixture",
                "source_info": source_info,
                "prompt": f"instruction\n{context}\noutput:",
            }
        )
        responses.append(
            {
                "id": str(index),
                "source_id": str(index),
                "model": "llama-2-7b-chat",
                "split": "train",
                "quality": "good",
                "response": "ABCDE",
                "labels": [
                    {"start": 1, "end": 3, "text": "BC", "label_type": "a"},
                    {"start": 2, "end": 4, "text": "CD", "label_type": "b"},
                ],
            }
        )
    responses.append(
        {
            **responses[0],
            "id": "ignored",
            "model": "another-model",
        }
    )
    source_path = tmp_path / "source_info.jsonl"
    response_path = tmp_path / "response.jsonl"
    _write_jsonl(source_path, sources)
    _write_jsonl(response_path, responses)

    examples = load_ragtruth(response_path, source_path)

    assert [example.task for example in examples] == ["QA", "Data2txt", "Summary"]
    assert all(example.generator_model == "llama-2-7b-chat" for example in examples)
    assert examples[0].spans == (
        CharSpan(start=1, end=4, text="BCD", label_type="a | b"),
    )
    assert dataset_counts(examples) == {
        "train": {"QA": 1, "Data2txt": 1, "Summary": 1}
    }


def test_span_contract_is_half_open_and_exact():
    spans = validate_label_spans(
        "012345",
        [{"start": 2, "end": 5, "text": "234"}],
        response_id="r",
    )
    assert spans[0].start == 2
    assert spans[0].end == 5
    with pytest.raises(ValueError, match="does not match"):
        validate_label_spans(
            "012345",
            [{"start": 2, "end": 5, "text": "2345"}],
            response_id="r",
        )
    with pytest.raises(ValueError, match="invalid span"):
        validate_label_spans(
            "012345",
            [{"start": 2, "end": 7, "text": "2345"}],
            response_id="r",
        )


def test_merge_spans_unions_overlap_adjacent_and_metadata():
    spans = (
        CharSpan(5, 8, "567", "b", due_to_null=True),
        CharSpan(1, 3, "12", "a", implicit_true=True),
        CharSpan(3, 6, "345", "a"),
        CharSpan(10, 11, "a", "c"),
    )
    assert merge_char_spans(spans, response="0123456789a") == (
        CharSpan(
            1,
            8,
            "1234567",
            "a | b",
            implicit_true=True,
            due_to_null=True,
        ),
        CharSpan(10, 11, "a", "c"),
    )


def _example(task: str, index: int) -> RagTruthExample:
    return RagTruthExample(
        response_id=f"{task}-{index}",
        source_id=f"s-{task}-{index}",
        generator_model="llama-2-7b-chat",
        split="train",
        quality="good",
        task=task,
        prompt="context",
        context="context",
        response="answer",
        spans=(),
    )


def test_calibration_dev_split_is_order_independent_and_balanced():
    examples = tuple(_example(task, i) for task in ("Summary", "QA", "Data2txt") for i in range(60))
    fit_a, dev_a = deterministic_calibration_dev_split(examples, dev_per_task=10)
    fit_b, dev_b = deterministic_calibration_dev_split(reversed(examples), dev_per_task=10)

    assert [item.response_id for item in dev_a] == [item.response_id for item in dev_b]
    assert [item.response_id for item in fit_a] == [item.response_id for item in fit_b]
    assert {item.response_id for item in fit_a}.isdisjoint(
        item.response_id for item in dev_a
    )
    assert Counter(item.task for item in dev_a) == {
        "QA": 10,
        "Data2txt": 10,
        "Summary": 10,
    }
    assert len(fit_a) == 150


def test_fixed_official_count_contract():
    assert EXPECTED_LLAMA2_COUNTS == {
        "train": {"QA": 839, "Data2txt": 883, "Summary": 793},
        "test": {"QA": 150, "Data2txt": 150, "Summary": 150},
    }
    with pytest.raises(ValueError, match="Unexpected"):
        assert_official_llama2_counts(())
