import json
from pathlib import Path
import tempfile
import unittest

from redeep_token.released_config import (
    OFFICIAL_CONFIG_RELATIVE_PATH,
    load_released_llama3_token_config,
)
from run_redeep_token_eval import (
    build_parser,
    load_candidate_heads,
    validate_candidate_transfer,
    validate_split_separation,
)
from redeep_token.provenance import checkpoint_artifact_manifest


class ReleasedArtifactIntegrityTest(unittest.TestCase):
    def test_modified_released_configuration_is_rejected(self):
        repository_root = Path(__file__).resolve().parents[1]
        released_path = repository_root / OFFICIAL_CONFIG_RELATIVE_PATH
        released = json.loads(released_path.read_text(encoding="utf-8"))
        released["weight"] = 0.5

        with tempfile.TemporaryDirectory() as temporary_directory:
            copied_path = (
                Path(temporary_directory) / OFFICIAL_CONFIG_RELATIVE_PATH
            )
            copied_path.parent.mkdir(parents=True)
            copied_path.write_text(
                json.dumps(released),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_released_llama3_token_config(temporary_directory)


class CrossBackboneWorkflowContractTest(unittest.TestCase):
    def _required_arguments(self):
        return [
            "--model-name-or-path",
            "model",
            "--data-dir",
            "data",
            "--output-dir",
            "output",
        ]

    def test_new_backbones_default_to_target_test_label_free_transfer(self):
        args = build_parser().parse_args(self._required_arguments())

        self.assertEqual(args.configuration_mode, "train-transfer")
        self.assertEqual(args.dtype, "float16")
        self.assertEqual(args.attention_implementation, "eager")
        self.assertEqual(args.head_counts, [3])
        self.assertEqual(args.layer_counts, [30])
        self.assertEqual(args.beta_values, [0.4])
        self.assertEqual(args.prompt_char_limit, 12_000)
        self.assertFalse(args.no_prompt_truncation)
        self.assertFalse(args.allow_checkpoint_transfer)

    def test_remote_workflow_selects_the_faithful_adaptation(self):
        repository_root = Path(__file__).resolve().parents[1]
        script = (
            repository_root
            / "scripts"
            / "run_llama31_on_llama2_ragtruth.sh"
        ).read_text(encoding="utf-8")

        for argument in (
            "--configuration-mode train-transfer",
            "--allow-checkpoint-transfer",
            "--quality-cohort all",
            "--dtype float16",
            "--attention-implementation eager",
        ):
            self.assertIn(argument, script)

    def test_every_new_checkpoint_requires_explicit_candidate_set_transfer(self):
        with self.assertRaisesRegex(ValueError, "checkpoint transfer"):
            validate_candidate_transfer(
                model_type="llama",
                allow_checkpoint_transfer=False,
                allow_cross_architecture_transfer=False,
            )

        validate_candidate_transfer(
            model_type="llama",
            allow_checkpoint_transfer=True,
            allow_cross_architecture_transfer=False,
        )

    def test_qwen_also_requires_explicit_cross_architecture_transfer(self):
        with self.assertRaisesRegex(ValueError, "cross-architecture"):
            validate_candidate_transfer(
                model_type="qwen3",
                allow_checkpoint_transfer=True,
                allow_cross_architecture_transfer=False,
            )

        validate_candidate_transfer(
            model_type="qwen3",
            allow_checkpoint_transfer=True,
            allow_cross_architecture_transfer=True,
        )

    def test_calibration_and_test_splits_must_be_disjoint(self):
        calibration = [{"id": "train-1", "source_id": "source-train"}]
        test = [{"id": "test-1", "source_id": "source-test"}]

        verified = validate_split_separation(
            calibration_split="train",
            test_split="test",
            calibration_samples=calibration,
            test_samples=test,
        )

        self.assertTrue(verified["response_ids_disjoint"])
        self.assertTrue(verified["source_ids_disjoint"])
        with self.assertRaisesRegex(ValueError, "different splits"):
            validate_split_separation(
                calibration_split="test",
                test_split="test",
                calibration_samples=calibration,
                test_samples=test,
            )
        with self.assertRaisesRegex(ValueError, "response IDs overlap"):
            validate_split_separation(
                calibration_split="train",
                test_split="test",
                calibration_samples=calibration,
                test_samples=[
                    {"id": "train-1", "source_id": "source-test"}
                ],
            )
        with self.assertRaisesRegex(ValueError, "source IDs overlap"):
            validate_split_separation(
                calibration_split="train",
                test_split="test",
                calibration_samples=calibration,
                test_samples=[
                    {"id": "test-1", "source_id": "source-train"}
                ],
            )

    def test_relocated_or_modified_candidate_file_still_requires_official_sha(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            candidate_path = Path(temporary_directory) / "heads.json"
            candidate_path.write_text("[[0, 0]]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "released candidate-head"):
                load_candidate_heads(
                    candidate_path,
                    expected_sha256=(
                        "c53edccab60a71489877aaa08e9c111736437b50028f2bcec00ef5d5525"
                    ),
                )


class CheckpointFingerprintTests(unittest.TestCase):
    def test_weight_or_tokenizer_change_invalidates_checkpoint_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory)
            (checkpoint / "config.json").write_text(
                '{"model_type": "llama"}',
                encoding="utf-8",
            )
            (checkpoint / "tokenizer_config.json").write_text(
                '{"chat_template": "template-a"}',
                encoding="utf-8",
            )
            (checkpoint / "chat_template.jinja").write_text(
                "{{ messages }}",
                encoding="utf-8",
            )
            (checkpoint / "vocab.json").write_text(
                '{"token": 0}',
                encoding="utf-8",
            )
            (checkpoint / "merges.txt").write_text(
                "#version: 0.2\n",
                encoding="utf-8",
            )
            shard = checkpoint / "model-00001-of-00001.safetensors"
            shard.write_bytes(b"weights-a")

            first = checkpoint_artifact_manifest(checkpoint)
            shard.write_bytes(b"weights-b")
            second = checkpoint_artifact_manifest(checkpoint)
            (checkpoint / "tokenizer_config.json").write_text(
                '{"chat_template": "template-b"}',
                encoding="utf-8",
            )
            third = checkpoint_artifact_manifest(checkpoint)
            (checkpoint / "chat_template.jinja").write_text(
                "{{ messages[0] }}",
                encoding="utf-8",
            )
            fourth = checkpoint_artifact_manifest(checkpoint)

        self.assertNotEqual(first["fingerprint"], second["fingerprint"])
        self.assertNotEqual(second["fingerprint"], third["fingerprint"])
        self.assertNotEqual(third["fingerprint"], fourth["fingerprint"])
        self.assertIn("chat_template.jinja", fourth["files"])
        self.assertIn("vocab.json", fourth["files"])
        self.assertIn("merges.txt", fourth["files"])


if __name__ == "__main__":
    unittest.main()
