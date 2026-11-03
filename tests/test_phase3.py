import json
import tempfile
import unittest
from pathlib import Path

from probe.bank import load_probe_files
from probe.config import load_config
from probe.report import build_report
from probe.stats import (bh_qvalues, fisher_exact_one_sided, movement,
                         permutation_pvalue, variant_summary, verdict)
from probe.judge import Judge
from probe.cli import _make_client


ROOT = Path(__file__).resolve().parents[1]


class Phase3StatsTests(unittest.TestCase):
    def test_fisher_reference_value(self):
        self.assertAlmostEqual(fisher_exact_one_sided(10, 10, 3, 10), 0.0015479876, places=6)

    def test_permutation_identical_and_different(self):
        self.assertEqual(permutation_pvalue([["same"] * 10, ["same"] * 10], 500, 3), 1.0)
        self.assertLess(permutation_pvalue([["a"] * 10, ["b"] * 10], 500, 3), 0.05)
        self.assertEqual(movement({"a": ["x", "x"], "b": ["x", "x"]})["movement"], 0.0)

    def test_bh_known_example(self):
        self.assertEqual([round(value, 2) for value in bh_qvalues([.01, .04, .2])], [.03, .06, .2])

    def test_each_verdict_on_hand_made_inputs(self):
        base = {"family": "social", "expect": "same"}
        expected = [variant_summary(["mother"] * 10, "mother"), variant_summary(["mother"] * 10, "mother")]
        self.assertEqual(verdict(base, expected, p_value=0.5), "CONSISTENT")
        noisy = [variant_summary(["mother"] * 5 + ["father"] * 5, "mother"), variant_summary(["mother"] * 10, "mother")]
        self.assertEqual(verdict(base, noisy, p_value=0.5), "NOISY")
        wrong = [variant_summary(["father"] * 10, "mother"), variant_summary(["father"] * 10, "mother")]
        self.assertEqual(verdict(base, wrong, p_value=0.5), "WRONG")
        self.assertEqual(verdict(base, expected, p_value=0.01), "BIASED")
        control = {"family": "control", "expect": "different"}
        different = [variant_summary(["japan"] * 10, "japan"), variant_summary(["italy"] * 10, "italy")]
        self.assertEqual(verdict(control, different, p_value=0.01), "CONTROL OK")
        trap = {"family": "familiar-pattern", "expect": "trap"}
        trap_items = [variant_summary(["mother"] * 10, "mother"), variant_summary(["mother"] * 7 + ["father"] * 3, "father")]
        self.assertEqual(verdict(trap, trap_items, p_value=0.5, fisher_p=0.01, pattern_capture=.7,
                                 changed_wrong_label="mother"), "BIASED (pattern)")

    def test_trap_distinguishes_pattern_from_other_error(self):
        trap = {"family": "familiar-pattern", "expect": "trap"}
        other_error = [variant_summary(["a"] * 8 + ["x"] * 2, "a"),
                       variant_summary(["0"] * 10, "b")]
        self.assertEqual(verdict(trap, other_error, fisher_p=.01, changed_wrong_label="0"), "FAILED (other error)")
        famous = [variant_summary(["a"] * 10, "a"), variant_summary(["a"] * 10, "b")]
        self.assertEqual(verdict(trap, famous, fisher_p=.01, changed_wrong_label="a"), "BIASED (pattern)")
        resisted = [variant_summary(["a"] * 10, "a"), variant_summary(["b"] * 9 + ["a"], "b")]
        self.assertEqual(verdict(trap, resisted, fisher_p=.5, changed_wrong_label="a"), "RESISTED")

    def test_disabled_judge_is_not_constructed_or_required(self):
        args = type("Args", (), {"mock": False})()
        self.assertIsNone(_make_client(args, [], {"judge": {"enabled": False}, "seed": 1}))

    def test_judge_prompt_deduplicates_allowed_labels(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as td:
            prompt = Judge(None, config, Path(td))._prompt("answer", ["mother", "mother", "hedge"], {}, None)
        self.assertIn("Allowed labels or option tokens: mother, hedge, refused, unscorable", prompt)


class Phase3ReportTests(unittest.TestCase):
    def test_report_is_static_and_offline(self):
        config = load_config()
        probes = load_probe_files([str(ROOT / "probes" / "bank.json"), str(ROOT / "probes" / "custom.json")])
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "reports"
            path = build_report(config, probes, ROOT / "data_mock", output, mock=True)
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("MOCK DATA", text)
            self.assertIn("verdicts are probe-level signals", text)
            self.assertIn("surgeon-riddle", text)
            self.assertIn("10 × T=0.7, 3 × T=0", text)
            self.assertIn("T=0:", text)
            self.assertNotIn("T=0: {", text)
            self.assertTrue((output / "openai__gpt-4o-mini-2024-07-18.html").exists())
            self.assertNotIn("https://", text)


if __name__ == "__main__":
    unittest.main()
