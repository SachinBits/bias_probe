import json
import os
import tempfile
import unittest
from pathlib import Path

from probe.bank import load_probe_files
from probe.cli import add_probe, run_judge_tests
from probe.config import load_config
from probe.judge import Judge
from probe.scorer import JUDGE_PROMPT_VERSION, rule_score


ROOT = Path(__file__).resolve().parents[1]


class Phase4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probes = load_probe_files([str(ROOT / "probes" / "bank.json"), str(ROOT / "probes" / "custom.json")])
        cls.by_id = {probe["id"]: probe for probe in cls.probes}

    def test_integer_period_and_decimal_boundaries(self):
        river = self.by_id["river-crossing"]
        variant = river["variants"][0]
        self.assertEqual(rule_score("The minimum number of one-way crossings is 3.", river, variant)["label"], "other")
        numeric = self.by_id["name-swap-competence"]
        variant = numeric["variants"][0]
        self.assertEqual(rule_score("8.", numeric, variant)["label"], "8")
        self.assertNotEqual(rule_score("3.5", numeric, variant)["label"], "3")

    def test_judge_prompt_version_provider_and_other_instruction(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as td:
            judge = Judge(None, config, Path(td))
            prompt = judge._prompt("17", ["1", "2"], {}, "choice_number")
        self.assertEqual(JUDGE_PROMPT_VERSION, "3")
        self.assertIn("return 'other'", prompt)

    def test_judge_test_results_are_saved(self):
        with tempfile.TemporaryDirectory() as td:
            args = type("Args", (), {"data_dir": td, "mock": True})()
            run_judge_tests(args, load_config(), self.probes)
            path = Path(td) / "judge_test_results.json"
            result = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("overall", result)
        self.assertIn("counter_stereotype", result)
        self.assertTrue(result["items"])
        self.assertIn("probe_id", result["items"][0])
        self.assertIn("judge_label", result["items"][0])

    def test_add_file_appends_custom_probe(self):
        payload = {"id": "phase4-test-probe", "family": "control", "expect": "different",
                   "labels": ["a", "b"], "why": "test", "variants": [
                       {"id": "left", "prompt": "Choose A", "answer": "a"},
                       {"id": "right", "prompt": "Choose B", "answer": "b"}]}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "probes").mkdir()
            (root / "probes" / "custom.json").write_text(json.dumps({"format": "probe-bank/v1", "probes": []}), encoding="utf-8")
            source = root / "pair.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            old = os.getcwd()
            try:
                os.chdir(root)
                add_probe(type("Args", (), {"file": str(source)})())
            finally:
                os.chdir(old)
            bank = json.loads((root / "probes" / "custom.json").read_text(encoding="utf-8"))
        self.assertEqual(bank["probes"][0]["id"], "phase4-test-probe")


if __name__ == "__main__":
    unittest.main()
