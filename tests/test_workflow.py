import unittest

from redeep_token.workflow import (
    format_markdown_metrics,
    protocol_fingerprint,
    response_ids_sha256,
    select_longest_per_task,
)


class WorkflowTests(unittest.TestCase):
    def test_protocol_fingerprint_is_order_stable_but_value_sensitive(self):
        first = protocol_fingerprint({"dtype": "bfloat16", "heads": [[1, 2]]})
        second = protocol_fingerprint({"heads": [[1, 2]], "dtype": "bfloat16"})
        changed = protocol_fingerprint({"dtype": "float16", "heads": [[1, 2]]})

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_selects_longest_sample_per_task_without_reordering_tasks(self):
        samples = [
            {"id": "q-short", "task_type": "QA", "prompt": "a", "response": "b"},
            {"id": "s", "task_type": "Summary", "prompt": "abc", "response": ""},
            {"id": "q-long", "task_type": "QA", "prompt": "abcd", "response": "e"},
        ]

        selected = select_longest_per_task(samples, ("QA", "Summary"))

        self.assertEqual([row["id"] for row in selected], ["q-long", "s"])

    def test_markdown_table_uses_requested_rebuttal_columns(self):
        metrics = {
            "per_task": {
                "QA": {"auroc": 0.63},
                "Summary": {"auroc": 0.54},
                "Data2txt": {"auroc": 0.51},
            },
            "task_macro_auroc": 0.56,
            "support_weighted_task_auroc": 0.57,
            "overall": {"auroc": 0.58},
        }

        table = format_markdown_metrics(metrics)

        self.assertIn("| QA | Summary | Data2txt | Task-macro |", table)
        self.assertIn("| 63.00 | 54.00 | 51.00 | 56.00 |", table)

    def test_response_id_hash_is_order_independent_and_boundary_safe(self):
        self.assertNotEqual(
            response_ids_sha256(["1", "23"]),
            response_ids_sha256(["12", "3"]),
        )
        self.assertEqual(
            response_ids_sha256(["a", "b"]),
            response_ids_sha256(["b", "a"]),
        )


if __name__ == "__main__":
    unittest.main()
