import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "tools" / "check_environment.py"
SPEC = importlib.util.spec_from_file_location("velascene_check_environment", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
check_environment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_environment
SPEC.loader.exec_module(check_environment)


class RequirementNamesTest(unittest.TestCase):
    def test_extracts_names_and_ignores_pip_options(self):
        lines = [
            "--extra-index-url https://example.com/simple",
            "torch==2.3.1",
            "opencv-python  # image loading",
            "package_name[extra]>=1.0; python_version >= '3.10'",
            "git+https://github.com/example/project.git",
            "",
        ]

        self.assertEqual(
            check_environment.requirement_names(lines),
            ["torch", "opencv-python", "package_name"],
        )


class PathCheckTest(unittest.TestCase):
    def test_distinguishes_files_and_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "model.pth"
            checkpoint.touch()

            self.assertEqual(
                check_environment.check_path("Data", root, directory=True).status,
                "PASS",
            )
            self.assertEqual(
                check_environment.check_path("Checkpoint", checkpoint, directory=False).status,
                "PASS",
            )
            self.assertEqual(
                check_environment.check_path(
                    "Checkpoint", root / "missing.pth", directory=False
                ).status,
                "FAIL",
            )


class MainTest(unittest.TestCase):
    def test_returns_nonzero_when_a_check_fails(self):
        checks = [check_environment.Check("FAIL", "Example", "not ready")]
        with mock.patch.object(check_environment, "run_checks", return_value=checks), mock.patch(
            "sys.stdout", new_callable=io.StringIO
        ):
            self.assertEqual(check_environment.main([]), 1)


if __name__ == "__main__":
    unittest.main()
