import unittest
from pathlib import Path

from redeep_token.released_config import (
    OFFICIAL_REPOSITORY,
    OFFICIAL_UPSTREAM_COMMIT,
    load_released_llama3_token_config,
    score_rows_with_released_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CONFIG_RELATIVE_PATH = Path(
    "ReDeEP/log/test_llama3_8B/token_hyperparameter.json"
)


class ReleasedLlama3ConfigurationTests(unittest.TestCase):
    def test_loads_exact_released_heads_layers_weight_and_max_min_ranges(self):
        config = load_released_llama3_token_config(REPOSITORY_ROOT)

        self.assertEqual(
            config.selected_heads,
            ((14, 2), (15, 20), (18, 31)),
        )
        self.assertEqual(
            config.selected_layers,
            (
                25,
                27,
                30,
                23,
                29,
                28,
                26,
                24,
                22,
                31,
                18,
                21,
                17,
                0,
                20,
                16,
                13,
                19,
                5,
                1,
                15,
                6,
                4,
                3,
                14,
                11,
                10,
                9,
                2,
                12,
            ),
        )
        self.assertEqual(config.head_max_min, (2.17333984375, 0.010288238525390625))
        self.assertEqual(
            config.layers_max_min,
            (149.9652862548828, 15.020370483398438),
        )
        self.assertEqual(
            config.final_max_min,
            (0.010109477515550308, -0.14520724625468528),
        )
        self.assertEqual(config.beta, 0.4)

    def test_manifest_identifies_the_exact_official_configuration_source(self):
        config = load_released_llama3_token_config(REPOSITORY_ROOT)

        manifest = config.manifest()

        self.assertEqual(manifest["configuration_mode"], "frozen_released")
        self.assertEqual(manifest["source_repository"], OFFICIAL_REPOSITORY)
        self.assertEqual(manifest["source_commit"], OFFICIAL_UPSTREAM_COMMIT)
        self.assertEqual(
            manifest["source_file"],
            OFFICIAL_CONFIG_RELATIVE_PATH.as_posix(),
        )
        self.assertEqual(manifest["source_architecture"], "llama")
        self.assertEqual(manifest["target_architecture"], "llama")
        self.assertFalse(manifest["cross_architecture_transfer"])

    def test_qwen3_transfer_is_rejected_unless_explicitly_opted_in(self):
        with self.assertRaisesRegex(
            ValueError,
            "cross-architecture transfer.*explicit",
        ):
            load_released_llama3_token_config(
                REPOSITORY_ROOT,
                target_architecture="qwen3",
            )

        config = load_released_llama3_token_config(
            REPOSITORY_ROOT,
            target_architecture="qwen3",
            allow_cross_architecture_transfer=True,
        )

        self.assertEqual(config.target_architecture, "qwen3")
        self.assertTrue(config.cross_architecture_transfer)
        self.assertTrue(config.manifest()["cross_architecture_transfer"])


class FrozenReleasedScoringTests(unittest.TestCase):
    def test_scores_features_with_released_ranges_without_labels_or_calibration(self):
        config = load_released_llama3_token_config(REPOSITORY_ROOT)
        feature_heads = ((18, 31), (14, 2), (15, 20), (0, 0))
        parametric = [0.0] * 32
        for layer in config.selected_layers:
            parametric[layer] = 100.0 / len(config.selected_layers)
        rows_without_labels = [
            {
                "id": "fixed-response-1",
                "task_type": "QA",
                "external": [[0.9, 0.5, 0.7, 999.0]],
                "parametric": [parametric],
            }
        ]

        scored = score_rows_with_released_config(
            rows_without_labels,
            feature_heads=feature_heads,
            config=config,
        )

        external_sum = 0.5 + 0.7 + 0.9
        external_max, external_min = config.head_max_min
        parametric_max, parametric_min = config.layers_max_min
        expected = (
            (100.0 - parametric_min) / (parametric_max - parametric_min)
            - config.beta
            * (external_sum - external_min)
            / (external_max - external_min)
        )
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0]["id"], "fixed-response-1")
        self.assertAlmostEqual(scored[0]["scores"][0], expected)

    def test_rejects_features_that_do_not_contain_every_released_head(self):
        config = load_released_llama3_token_config(REPOSITORY_ROOT)
        rows = [
            {
                "external": [[0.1, 0.2]],
                "parametric": [[0.0] * 32],
            }
        ]

        with self.assertRaisesRegex(ValueError, "released selected head"):
            score_rows_with_released_config(
                rows,
                feature_heads=((14, 2), (15, 20)),
                config=config,
            )


if __name__ == "__main__":
    unittest.main()
