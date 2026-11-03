import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from probe.bank import load_probe_files
from probe.cli import add_probe, run_judge_tests, sample_hand_labels
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

    def test_hand_label_sample_mix_and_model_key_dedup(self):
        config = load_config()
        models = [model["id"] for model in config["models"]]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for model_index, model in enumerate(models):
                slug = model.replace("/", "__").replace(":", "__")
                response_path = root / "responses" / (slug + ".jsonl")
                label_path = root / "labels" / (slug + ".jsonl")
                response_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                response_lines, label_lines = [], []
                for i in range(35):
                    probe = self.probes[i % len(self.probes)]
                    variant = probe["variants"][i % len(probe["variants"])]
                    key = "%s-key-%d" % (model_index, i)
                    text = "x" * 70 if i < 20 else "short answer"
                    source = "judge" if i < 2 else ("none" if i == 2 else "rule")
                    response = {"key": key, "model_requested": model, "probe_id": probe["id"],
                                "variant_id": variant["id"], "repeat_index": i, "params": {"temperature": 0.7},
                                "response_text": text, "status": "ok"}
                    response_lines.append(json.dumps(response))
                    label_lines.append(json.dumps({"key": key, "model": model, "probe_id": probe["id"],
                                                   "variant_id": variant["id"], "repeat_index": i,
                                                   "temperature": 0.7, "response_text": text, "label": "x",
                                                   "source": source, "rule_label": "x" if source == "rule" else None,
                                                   "judge_label": "x" if source == "judge" else None}))
                # Same model and key as an existing row, but a different probe: must be deduplicated.
                duplicate = json.loads(label_lines[0])
                duplicate["probe_id"] = self.probes[-1]["id"]
                label_lines.append(json.dumps(duplicate))
                response_path.write_text("\n".join(response_lines) + "\n", encoding="utf-8")
                label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            args = type("Args", (), {"data_dir": td, "mock": False, "n": 60})()
            sample_hand_labels(args, config, self.probes)
            with (root / "hand_labels.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            sample_hand_labels(args, config, self.probes)
            with (root / "hand_labels.csv").open(encoding="utf-8", newline="") as handle:
                repeat_rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 60)
        self.assertEqual(rows, repeat_rows)
        self.assertTrue({"model", "line", "label_source", "human_label"}.issubset(rows[0]))
        self.assertTrue(all(row["line"] for row in rows))
        self.assertTrue(all(row["human_label"] == "" for row in rows))
        self.assertEqual(len({(row["model"], row["key"]) for row in rows}), 60)
        self.assertEqual({row["model"] for row in rows}, set(models))
        self.assertGreaterEqual(sum(row["label_source"] in {"judge", "none"} for row in rows), 3)
        self.assertGreaterEqual(sum(len(row["response_text"]) > 60 for row in rows), 15)
        first = {probe["id"]: probe["variants"][0]["id"] for probe in self.probes}
        self.assertGreaterEqual(sum(row["variant_id"] != first.get(row["probe_id"]) for row in rows), 15)


if __name__ == "__main__":
    unittest.main()
