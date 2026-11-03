import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from probe.bank import load_probe_files
from probe.client import MockClient
from probe.config import load_config
from probe.cli import score_models
from probe.judge import Judge
from probe.scorer import rule_score, score_response


ROOT = Path(__file__).resolve().parents[1]


class Phase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probes = load_probe_files([str(ROOT / "probes" / "bank.json"), str(ROOT / "probes" / "custom.json")])
        cls.by_id = {p["id"]: p for p in cls.probes}

    def variant(self, probe_id, variant_id):
        probe = self.by_id[probe_id]
        return probe, next(v for v in probe["variants"] if v["id"] == variant_id)

    def judge(self, path):
        config = load_config()
        return Judge(MockClient(self.probes, seed=7, simulate_errors=False), config, Path(path))

    def test_surgeon_echo_and_mother_label(self):
        probe, variant = self.variant("surgeon-riddle", "original")
        self.assertEqual(rule_score("The father died, so the surgeon is the boy's mother.", probe, variant)["label"], "mother")
        self.assertEqual(rule_score("Mother.", probe, variant)["label"], "mother")

    def test_negation_uses_judge_to_select_father(self):
        probe, variant = self.variant("surgeon-riddle", "original")
        with tempfile.TemporaryDirectory() as td:
            scored = score_response("Not the mother, it's the father.", probe, variant, self.judge(td))
        self.assertEqual(scored["label"], "father")
        self.assertEqual(scored["source"], "judge")

    def test_both_labels_use_judge_or_unscorable(self):
        probe, variant = self.variant("surgeon-riddle", "original")
        no_judge = score_response("It is the mother and father.", probe, variant)
        self.assertEqual(no_judge["label"], "unscorable")
        with tempfile.TemporaryDirectory() as td:
            judged = score_response("It is the mother and father.", probe, variant, self.judge(td))
        self.assertEqual(judged["source"], "judge")
        self.assertEqual(judged["label"], "hedge")

    def test_letter_answers_with_extra_text_are_mapped(self):
        probe, variant = self.variant("option-rotation", "correct-at-B")
        self.assertEqual(rule_score("B) 29", probe, variant)["label"], "29")
        self.assertEqual(rule_score("The answer is B.", probe, variant)["label"], "29")

    def test_amount_variants(self):
        probe, variant = self.variant("bat-and-ball", "original")
        for text in ("$0.05", "5 cents", "0.05"):
            with self.subTest(text=text):
                self.assertEqual(rule_score(text, probe, variant)["label"], "0.05")
        self.assertEqual(rule_score("10 cents", probe, variant)["label"], "0.10")

    def test_hedge_and_uncertainty_content_label_are_distinct(self):
        surgeon, surgeon_variant = self.variant("surgeon-riddle", "original")
        self.assertEqual(rule_score("It could be either parent.", surgeon, surgeon_variant)["label"], "hedge")
        bbq, bbq_variant = self.variant("bbq-age-ambiguous", "grandfather-first")
        result = rule_score("Cannot be determined.", bbq, bbq_variant)
        self.assertEqual(result["label"], "unknown")
        self.assertNotEqual(result["label"], "hedge")

    def test_quoted_riddle_phrase_is_not_refusal(self):
        probe, variant = self.variant("surgeon-riddle", "original")
        result = rule_score("I can't operate on this boy; he's my son.", probe, variant)
        self.assertNotEqual(result["label"], "refused")

    def test_judge_cache_contains_version_and_not_original_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            judge = self.judge(td)
            label, _ = judge.classify("Not the mother, it's the father.", ["mother", "father", "hedge"])
            self.assertEqual(label, "father")
            label2, detail = judge.classify("Not the mother, it's the father.", ["mother", "father", "hedge"])
            self.assertEqual(label2, "father")
            self.assertIn("cached", detail)
            files = list((Path(td) / "judge").glob("*.jsonl"))
            record = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["judge_prompt_version"], "1")
            self.assertNotIn("surgeon", record["prompt"].lower())

    def test_score_command_writes_required_label_record_fields(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            probe, variant = self.variant("surgeon-riddle", "original")
            response_path = data / "responses" / "test__model.jsonl"
            response_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.write_text(json.dumps({"key": "k", "status": "ok", "model_requested": "test/model",
                                                  "probe_id": probe["id"], "variant_id": variant["id"],
                                                  "params": {"temperature": 0.7}, "repeat_index": 0,
                                                  "response_text": "Mother."}) + "\n", encoding="utf-8")
            config = {"models": [{"id": "test/model"}], "judge": {"id": "judge/model", "max_tokens": 80}, "seed": 7}
            score_models(type("Args", (), {"data_dir": str(data), "mock": True, "models": None})(), config, [probe], MockClient([probe], seed=7, simulate_errors=False))
            record = json.loads((data / "labels" / "test__model.jsonl").read_text(encoding="utf-8").splitlines()[0])
            for field in ("key", "probe_id", "variant_id", "model", "temperature", "repeat_index", "label", "source", "detail", "scorer_version", "judge_prompt_version"):
                self.assertIn(field, record)
            self.assertEqual(record["label"], "mother")


if __name__ == "__main__":
    unittest.main()
