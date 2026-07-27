import json
import tempfile
import unittest
from pathlib import Path

from redeep_token.data import (
    DEFAULT_EXCLUDED_QUALITIES,
    build_teacher_forced_encoding,
    load_ragtruth_samples,
    render_redeep_prefix,
)


class CharacterTokenizer:
    """Small fast-tokenizer stand-in with one token per character."""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        return rendered + "<assistant>"

    def __call__(
        self,
        texts,
        *,
        return_offsets_mapping=False,
        truncation=False,
        **_kwargs,
    ):
        assert truncation is False
        if isinstance(texts, str):
            texts = [texts]
        input_ids = []
        offset_mapping = []
        for text in texts:
            input_ids.append([1] + [10 + ord(char) for char in text])
            offset_mapping.append(
                [(0, 0)] + [(index, index + 1) for index in range(len(text))]
            )
        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offset_mapping
        return result


class BoundaryRetokenizingTokenizer(CharacterTokenizer):
    """Mimic a BPE tokenizer whose last prefix token changes after concatenation."""

    def __call__(
        self,
        texts,
        *,
        return_offsets_mapping=False,
        truncation=False,
        **kwargs,
    ):
        result = super().__call__(
            texts,
            return_offsets_mapping=return_offsets_mapping,
            truncation=truncation,
            **kwargs,
        )
        for text, token_ids in zip(texts, result["input_ids"]):
            if text.endswith("<assistant>"):
                token_ids[-1] += 1
        return result


class ChatKwargsTokenizer(CharacterTokenizer):
    def __init__(self):
        self.chat_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_kwargs = dict(kwargs)
        kwargs.pop("enable_thinking", None)
        return super().apply_chat_template(messages, **kwargs)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class RAGTruthLoadingTests(unittest.TestCase):
    def test_filters_generator_split_quality_and_attaches_task(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            write_jsonl(
                data_dir / "source_info.jsonl",
                [
                    {
                        "source_id": "qa",
                        "task_type": "QA",
                        "prompt": "Question and retrieved passages",
                        "source_info": {},
                    },
                    {
                        "source_id": "summary",
                        "task_type": "Summary",
                        "prompt": "Article",
                        "source_info": "Article",
                    },
                ],
            )
            write_jsonl(
                data_dir / "response.jsonl",
                [
                    {
                        "id": "keep",
                        "source_id": "qa",
                        "model": "llama-2-7b-chat",
                        "split": "test",
                        "quality": "good",
                        "response": "bad answer",
                        "labels": [
                            {"start": 0, "end": 3, "text": "bad"}
                        ],
                    },
                    {
                        "id": "wrong-model",
                        "source_id": "qa",
                        "model": "gpt-4",
                        "split": "test",
                        "quality": "good",
                        "response": "answer",
                        "labels": [],
                    },
                    {
                        "id": "wrong-split",
                        "source_id": "summary",
                        "model": "llama-2-7b-chat",
                        "split": "train",
                        "quality": "good",
                        "response": "answer",
                        "labels": [],
                    },
                    {
                        "id": "bad-quality",
                        "source_id": "qa",
                        "model": "llama-2-7b-chat",
                        "split": "test",
                        "quality": "truncated",
                        "response": "answer",
                        "labels": [],
                    },
                ],
            )

            samples = load_ragtruth_samples(
                data_dir=data_dir,
                split="test",
                generator_model="llama-2-7b-chat",
                tasks=("QA", "Summary", "Data2txt"),
                excluded_qualities=DEFAULT_EXCLUDED_QUALITIES,
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["id"], "keep")
        self.assertEqual(samples[0]["task_type"], "QA")
        self.assertEqual(
            samples[0]["prompt"], "Question and retrieved passages"
        )

    def test_default_keeps_the_complete_official_split(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            write_jsonl(
                data_dir / "source_info.jsonl",
                [
                    {
                        "source_id": "qa",
                        "task_type": "QA",
                        "prompt": "Question and retrieved passages",
                    }
                ],
            )
            write_jsonl(
                data_dir / "response.jsonl",
                [
                    {
                        "id": "truncated-response",
                        "source_id": "qa",
                        "model": "llama-2-7b-chat",
                        "split": "test",
                        "quality": "truncated",
                        "response": "answer",
                        "labels": [],
                    }
                ],
            )

            samples = load_ragtruth_samples(
                data_dir=data_dir,
                split="test",
                generator_model="llama-2-7b-chat",
            )

        self.assertEqual(
            [sample["id"] for sample in samples],
            ["truncated-response"],
        )

    def test_teacher_forcing_scores_previous_position_and_aligns_spans(self):
        tokenizer = CharacterTokenizer()
        response = "bad ok"
        encoding = build_teacher_forced_encoding(
            tokenizer,
            prompt="retrieved evidence",
            response=response,
            labels=[{"start": 0, "end": 3, "text": "bad"}],
        )

        self.assertEqual(len(encoding["response_token_positions"]), len(response))
        self.assertEqual(
            encoding["score_positions"],
            [position - 1 for position in encoding["response_token_positions"]],
        )
        self.assertEqual(encoding["response_offsets"][0], (0, 1))
        self.assertEqual(encoding["response_offsets"][-1], (5, 6))
        self.assertEqual(encoding["token_labels"], [1, 1, 1, 0, 0, 0])
        self.assertEqual(
            encoding["input_ids"][: len(encoding["prefix_ids"])],
            encoding["prefix_ids"],
        )

    def test_joint_token_offsets_handle_prefix_boundary_retokenization(self):
        tokenizer = BoundaryRetokenizingTokenizer()

        encoding = build_teacher_forced_encoding(
            tokenizer,
            prompt="retrieved evidence",
            response="answer",
            labels=[],
        )

        self.assertTrue(encoding["boundary_retokenized"])
        self.assertEqual(
            encoding["prefix_ids"],
            encoding["input_ids"][: len(encoding["prefix_ids"])],
        )
        self.assertEqual(
            encoding["score_positions"][0],
            encoding["response_token_positions"][0] - 1,
        )

    def test_qwen_chat_template_options_are_forwarded_explicitly(self):
        tokenizer = ChatKwargsTokenizer()

        render_redeep_prefix(
            tokenizer,
            prompt="retrieved evidence",
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertFalse(tokenizer.chat_kwargs["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
