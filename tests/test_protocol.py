import unittest

from redeep_token.protocol import (
    apply_calibration,
    binary_auroc,
    calibrate_redeep,
    compute_response_metrics,
    compute_token_metrics,
)


class AurocTests(unittest.TestCase):
    def test_binary_auroc_handles_ties(self):
        self.assertAlmostEqual(
            binary_auroc([0, 0, 1, 1], [0.0, 0.5, 0.5, 1.0]),
            0.875,
        )

    def test_reports_per_task_macro_weighted_and_pooled(self):
        rows = [
            {"task_type": "QA", "labels": [0, 1], "scores": [0.1, 0.9]},
            {
                "task_type": "Summary",
                "labels": [0, 1],
                "scores": [0.9, 0.1],
            },
            {
                "task_type": "Data2txt",
                "labels": [0, 1],
                "scores": [0.2, 0.8],
            },
        ]

        metrics = compute_token_metrics(rows)

        self.assertEqual(metrics["per_task"]["QA"]["auroc"], 1.0)
        self.assertEqual(metrics["per_task"]["Summary"]["auroc"], 0.0)
        self.assertEqual(metrics["per_task"]["Data2txt"]["auroc"], 1.0)
        self.assertAlmostEqual(metrics["task_macro_auroc"], 2.0 / 3.0)
        self.assertAlmostEqual(
            metrics["support_weighted_task_auroc"], 2.0 / 3.0
        )
        self.assertEqual(metrics["overall"]["num_tokens"], 6)

    def test_response_sanity_uses_mean_score_and_any_span_label(self):
        rows = [
            {
                "task_type": "QA",
                "labels": [0, 0],
                "scores": [0.1, 0.3],
            },
            {
                "task_type": "QA",
                "labels": [0, 1],
                "scores": [0.7, 0.9],
            },
            {
                "task_type": "Summary",
                "labels": [0],
                "scores": [0.2],
            },
            {
                "task_type": "Summary",
                "labels": [1],
                "scores": [0.8],
            },
            {
                "task_type": "Data2txt",
                "labels": [0],
                "scores": [0.4],
            },
            {
                "task_type": "Data2txt",
                "labels": [1],
                "scores": [0.6],
            },
        ]

        metrics = compute_response_metrics(rows)

        self.assertEqual(metrics["overall"]["auroc"], 1.0)
        self.assertEqual(metrics["per_task"]["QA"]["auroc"], 1.0)
        self.assertEqual(metrics["overall"]["num_responses"], 6)


class CalibrationTests(unittest.TestCase):
    def test_selects_grounding_head_and_parametric_layer_on_calibration_only(self):
        calibration_rows = []
        labels = [0, 0, 1, 1]
        external = [
            [0.0, 0.9],
            [1.0, 0.8],
            [0.0, 0.2],
            [1.0, 0.1],
        ]
        parametric = [
            [0.1, 0.0],
            [0.2, 1.0],
            [0.8, 0.0],
            [0.9, 1.0],
        ]
        for index, label in enumerate(labels):
            calibration_rows.append(
                {
                    "id": str(index),
                    "task_type": "QA",
                    "labels": [label],
                    "external": [external[index]],
                    "parametric": [parametric[index]],
                }
            )

        calibration = calibrate_redeep(
            calibration_rows,
            candidate_heads=[(4, 16), (14, 2)],
            selection_unit="token",
            head_counts=(1,),
            layer_counts=(1,),
            beta_values=(0.5,),
        )

        self.assertEqual(calibration["selected_head_indices"], [1])
        self.assertEqual(calibration["selected_heads"], [[14, 2]])
        self.assertEqual(calibration["selected_layers"], [0])
        self.assertEqual(calibration["beta"], 0.5)
        self.assertEqual(calibration["calibration_auroc"], 1.0)

        scored = apply_calibration(calibration_rows, calibration)
        flattened_scores = [
            score for row in scored for score in row["scores"]
        ]
        self.assertEqual(binary_auroc(labels, flattened_scores), 1.0)


if __name__ == "__main__":
    unittest.main()
