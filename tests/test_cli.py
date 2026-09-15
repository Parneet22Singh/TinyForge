import json
import tempfile
import unittest
from pathlib import Path

from tinyforge.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_parser_exposes_complete_command_surface(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["hardware"]).command, "hardware")
        self.assertEqual(parser.parse_args(["train", "data", "--real", "--model-id", "model"]).real, True)

    def test_training_and_compression_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset.jsonl").write_text(json.dumps({"text": "a useful example"}) + "\n", encoding="utf-8")
            run = root / "run"
            self.assertEqual(main_args(["train", str(root), "--output", str(run)]), 0)
            self.assertTrue((run / "metrics.json").exists())
            compressed = root / "compressed"
            self.assertEqual(main_args(["shrink", str(run / "model"), "--output", str(compressed)]), 0)
            self.assertTrue((compressed / "compression.json").exists())


def main_args(arguments: list[str]) -> int:
    import sys
    original = sys.argv
    try:
        sys.argv = ["tinyforge", *arguments]
        return main()
    finally:
        sys.argv = original
