import tempfile
import unittest
from pathlib import Path

from redeep_token.feature_store import (
    feature_path,
    read_feature_record,
    write_feature_record,
)


class FeatureStoreTests(unittest.TestCase):
    def test_round_trip_is_atomic_and_uses_stable_response_path(self):
        record = {
            "schema_version": 1,
            "protocol_fingerprint": "abc",
            "id": "response/with unsafe chars",
            "task_type": "QA",
            "labels": [0, 1],
            "external": [[0.1, 0.2], [0.3, 0.4]],
            "parametric": [[1.0, 2.0], [3.0, 4.0]],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            path = write_feature_record(directory, record)
            expected = feature_path(directory, record["id"])
            loaded = read_feature_record(
                expected,
                expected_id=record["id"],
                expected_fingerprint="abc",
            )
            leftovers = list(directory.glob("*.part"))

        self.assertEqual(path, expected)
        self.assertEqual(loaded, record)
        self.assertEqual(leftovers, [])

    def test_rejects_stale_protocol_fingerprint(self):
        record = {
            "schema_version": 1,
            "protocol_fingerprint": "old",
            "id": "1",
            "task_type": "Summary",
            "labels": [0],
            "external": [[0.1]],
            "parametric": [[0.2]],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = write_feature_record(Path(temporary_directory), record)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                read_feature_record(
                    path,
                    expected_id="1",
                    expected_fingerprint="new",
                )


if __name__ == "__main__":
    unittest.main()
