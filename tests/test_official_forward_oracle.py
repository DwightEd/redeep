import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class OfficialForwardOracleTest(unittest.TestCase):
    def test_hook_extractor_matches_released_transformers_forward(self):
        repository_root = Path(__file__).resolve().parents[1]
        official_dependencies = os.environ.get(
            "REDEEP_OFFICIAL_ORACLE_SITE_PACKAGES"
        )
        modern_dependencies = os.environ.get(
            "REDEEP_MODERN_ORACLE_SITE_PACKAGES"
        )
        if not official_dependencies or not modern_dependencies:
            self.skipTest(
                "set both ReDeEP oracle site-package environment variables"
            )

        worker = (
            repository_root / "tests" / "official_forward_oracle_worker.py"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            state_dict = temporary / "state.pt"
            released_output = temporary / "released.json"
            modern_output = temporary / "modern.json"

            released_environment = dict(os.environ)
            released_environment["PYTHONPATH"] = os.pathsep.join(
                [
                    str(repository_root / "transformers" / "src"),
                    official_dependencies,
                    modern_dependencies,
                ]
            )
            subprocess.run(
                [
                    sys.executable,
                    str(worker),
                    "--mode",
                    "released",
                    "--state-dict",
                    str(state_dict),
                    "--output",
                    str(released_output),
                ],
                cwd=repository_root,
                env=released_environment,
                check=True,
            )

            modern_environment = dict(os.environ)
            modern_environment["PYTHONPATH"] = os.pathsep.join(
                [modern_dependencies, str(repository_root)]
            )
            subprocess.run(
                [
                    sys.executable,
                    str(worker),
                    "--mode",
                    "modern",
                    "--state-dict",
                    str(state_dict),
                    "--output",
                    str(modern_output),
                ],
                cwd=repository_root,
                env=modern_environment,
                check=True,
            )

            released = json.loads(released_output.read_text(encoding="utf-8"))
            modern = json.loads(modern_output.read_text(encoding="utf-8"))

        self.assertEqual(len(released["external"]), len(modern["external"]))
        self.assertEqual(len(released["parametric"]), len(modern["parametric"]))
        for expected_row, actual_row in zip(
            released["external"], modern["external"]
        ):
            for expected, actual in zip(expected_row, actual_row):
                self.assertAlmostEqual(expected, actual, delta=2e-6)
        for expected_row, actual_row in zip(
            released["parametric"], modern["parametric"]
        ):
            for expected, actual in zip(expected_row, actual_row):
                self.assertAlmostEqual(expected, actual, delta=2e-3)


if __name__ == "__main__":
    unittest.main()
