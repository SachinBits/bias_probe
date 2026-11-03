import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from probe.bank import load_probe_files
from probe.client import LazyOpenRouterClient, MockClient
from probe.client import ClientError
from probe.config import load_config
from probe.cli import _make_client, run_judge_tests, sample_hand_labels, score_models
from probe.judge import Judge, parse_judge_reply
from probe.report import build_report
from probe.runner import run
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
        self.assertIn(scored["source"], {"rule", "judge"})
        with tempfile.TemporaryDirectory() as td:
            scored = score_response("His dad, not his mother.", probe, variant, self.judge(td))
        self.assertNotEqual(scored["label"], "mother")
        self.assertEqual(scored["label"], "father")

    def test_contradictory_strong_regex_hit_goes_to_judge(self):
        probe, variant = self.variant("surgeon-riddle", "original")
        response = "The surgeon is the boy's father, and the answer is the mother."
        self.assertIsNone(rule_score(response, probe, variant)["label"])
        with tempfile.TemporaryDirectory() as td:
            scored = score_response(response, probe, variant, self.judge(td))
        self.assertEqual(scored["source"], "judge")
        self.assertEqual(scored["label"], "hedge")

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
        for text in ("$.05", ".05", "$0.05", "5 cents", "five cents"):
            with self.subTest(text=text):
                self.assertEqual(rule_score(text, probe, variant)["label"], "0.05")
        self.assertEqual(rule_score("$.10", probe, variant)["label"], "0.10")

    def test_choice_failure_uses_judge_token_then_variant_map(self):
        probe, variant = self.variant("pairwise-judge-order", "plain-first")
        response = "2. Answer 1 is a more detailed and clear explanation"
        self.assertIsNone(rule_score(response, probe, variant)["label"])
        with tempfile.TemporaryDirectory() as td:
            result = score_response(response, probe, variant, self.judge(td))
        self.assertEqual(result["label"], "rayleigh")
        self.assertEqual(result["source"], "judge")

    def test_fenced_chatty_judge_reply_and_case_insensitive_label(self):
        self.assertEqual(parse_judge_reply('Here you go:\n```json\n{"LaBeL": "father"}\n```', ["mother", "father"]), "father")
        self.assertEqual(parse_judge_reply('The result is {"LABEL": "mother"}, with confidence.', ["mother", "father"]), "mother")

    def test_judge_reuses_runner_retries_and_only_final_failure_is_error(self):
        class EventuallyWorks:
            def __init__(self):
                self.calls = 0
            def complete(self, *args, **kwargs):
                self.calls += 1
                if self.calls < 6:
                    raise ClientError("temporary", True, 0)
                return {"response_text": '{"label":"father"}', "model_returned": "j", "provider_returned": "mock"}
        with tempfile.TemporaryDirectory() as td:
            client = EventuallyWorks()
            judge = Judge(client, load_config(), Path(td))
            self.assertEqual(judge.classify("father", ["mother", "father"])[0], "father")
            self.assertEqual(client.calls, 6)
            record = json.loads(next((Path(td) / "judge").glob("*.jsonl")).read_text().splitlines()[0])
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["attempts"], 6)

        class AlwaysFails:
            def __init__(self):
                self.calls = 0
            def complete(self, *args, **kwargs):
                self.calls += 1
                raise ClientError("permanent", True, 0)
        with tempfile.TemporaryDirectory() as td:
            client = AlwaysFails()
            judge = Judge(client, load_config(), Path(td))
            self.assertEqual(judge.classify("father", ["mother", "father"])[0], "unscorable")
            self.assertEqual(client.calls, 6)
            self.assertEqual(judge.classify("father", ["mother", "father"])[0], "unscorable")
            self.assertEqual(client.calls, 12)
            records = [json.loads(x) for x in next((Path(td) / "judge").glob("*.jsonl")).read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r["status"] == "error" for r in records))

    def test_int_1_10_ignores_scale_numbers(self):
        probe, variant = self.variant("name-swap-competence", "james")
        self.assertEqual(rule_score("On a scale from 1 to 10, score 8 out of 10 (/10).", probe, variant)["label"], "8")

    def test_identical_prompts_are_cached_per_probe_variant(self):
        probes = [
            {"id": "p1", "family": "control", "expect": "same", "labels": ["yes"], "why": "test", "variants": [{"id": "v1", "prompt": "same prompt"}, {"id": "v2", "prompt": "same prompt"}]},
            {"id": "p2", "family": "control", "expect": "same", "labels": ["yes"], "why": "test", "variants": [{"id": "v1", "prompt": "same prompt"}, {"id": "v2", "prompt": "same prompt"}]},
        ]
        config = {"models": [{"id": "test/model", "provider": None}], "schedule": [{"temperature": 0.7, "repeats": 1}], "max_tokens": 10, "concurrency": 2, "spend_cap_usd": 3.0, "seed": 7}
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            first = run(config, probes, MockClient(probes, seed=7, simulate_errors=False), data, None, None, False)
            second = run(config, probes, MockClient(probes, seed=7, simulate_errors=False), data, None, None, False)
            self.assertEqual(first["written"], 4)
            self.assertEqual(second["written"], 0)
            records = [json.loads(x) for x in (data / "responses" / "test__model.jsonl").read_text().splitlines()]
            self.assertEqual({r["probe_id"] for r in records}, {"p1", "p2"})

    def test_hand_labels_include_full_response_identity(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            labels = data / "labels" / "test.jsonl"
            labels.parent.mkdir(parents=True, exist_ok=True)
            labels.write_text(json.dumps({"key": "k", "probe_id": "p", "variant_id": "v", "repeat_index": 3,
                                          "temperature": 0.7, "response_text": "mother", "rule_label": "mother", "judge_label": None}) + "\n", encoding="utf-8")
            args = type("Args", (), {"data_dir": str(data), "mock": True, "n": 1})()
            sample_hand_labels(args)
            header = (data / "hand_labels.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("probe_id", header)
            self.assertIn("variant_id", header)
            self.assertIn("repeat_index", header)
            self.assertIn("temperature", header)

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
            self.assertEqual(record["judge_prompt_version"], "3")
            self.assertNotIn("surgeon", record["prompt"].lower())

    def test_plain_allowed_label_reply_is_accepted(self):
        self.assertEqual(parse_judge_reply('"Father."', ["mother", "father"]), "father")

    def test_judge_record_keeps_returned_metadata(self):
        class MetadataClient:
            def complete(self, *args, **kwargs):
                return {"response_text": '{"label":"father"}', "model_returned": "judge-snapshot", "provider_returned": "provider-x", "response_id": "response-1"}
        with tempfile.TemporaryDirectory() as td:
            judge = Judge(MetadataClient(), load_config(), Path(td))
            judge.classify("father", ["mother", "father"])
            record = json.loads(next((Path(td) / "judge").glob("*.jsonl")).read_text().splitlines()[0])
            self.assertEqual(record["model_returned"], "judge-snapshot")
            self.assertEqual(record["provider_returned"], "provider-x")
            self.assertEqual(record["response_id"], "response-1")

    def test_missing_key_uncached_judge_is_unscorable_with_one_warning(self):
        probe, _ = self.variant("surgeon-riddle", "original")
        client = LazyOpenRouterClient(lambda: "")
        with tempfile.TemporaryDirectory() as td:
            judge = Judge(client, load_config(), Path(td))
            warning = io.StringIO()
            with redirect_stderr(warning):
                label, detail = judge.classify("It is the mother and father.", probe["labels"])
                label2, detail2 = judge.classify("It is the mother and father.", probe["labels"])
        self.assertEqual((label, detail), ("unscorable", "judge not cached and no API key"))
        self.assertEqual((label2, detail2), ("unscorable", "judge not cached and no API key"))
        self.assertEqual(warning.getvalue().count("Warning: judge not cached"), 1)
        self.assertIsNone(client._client)
        self.assertEqual(client.complete_calls, 0)

    def test_cached_judge_score_and_report_need_no_key_or_calls(self):
        config = load_config()
        probe, variant = self.variant("surgeon-riddle", "original")

        class CachedJudgeClient:
            def complete(self, *args, **kwargs):
                return {"response_text": '{"label":"hedge"}', "model_returned": "judge", "provider_returned": "cached"}

        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            Judge(CachedJudgeClient(), config, data).classify("It is the mother and father.", probe["labels"])
            model = config["models"][0]
            response_path = data / "responses" / (model["id"].replace("/", "__") + ".jsonl")
            response_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.write_text(json.dumps({
                "key": "cached-response", "status": "ok", "model_requested": model["id"],
                "probe_id": probe["id"], "variant_id": variant["id"], "repeat_index": 0,
                "params": {"temperature": 0.7, "provider": model.get("provider"), "max_tokens": config["max_tokens"]},
                "response_text": "It is the mother and father.", "timestamp_utc": "2026-01-01T00:00:00+00:00"
            }) + "\n", encoding="utf-8")
            args = type("Args", (), {"data_dir": str(data), "mock": False, "models": None})()
            with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
                client = _make_client(args, self.probes, config)
                warning = io.StringIO()
                with redirect_stderr(warning):
                    score_models(args, config, self.probes, client)
                    report_path = build_report(config, self.probes, data, data / "reports")
            self.assertTrue(report_path.exists())
            self.assertIsNone(client._client)
            self.assertEqual(client.complete_calls, 0)
            self.assertEqual(warning.getvalue(), "")

    def test_judge_test_reports_rule_breakdown_and_no_counter_rule_wrong(self):
        output = io.StringIO()
        args = type("Args", (), {"data_dir": "/tmp/phase26-judge-test", "mock": True})()
        with redirect_stdout(output):
            run_judge_tests(args, load_config(), self.probes)
        text = output.getvalue()
        self.assertIn("rule overall: correct=", text)
        self.assertIn("wrong=", text)
        self.assertIn("abstained=", text)
        self.assertIn("counter-stereotype rule: correct=", text)
        self.assertIn("counter-stereotype rule: correct=", text)
        self.assertIn("wrong=0", text)

    def test_score_command_writes_required_label_record_fields(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            probe, variant = self.variant("surgeon-riddle", "original")
            response_path = data / "responses" / "test__model.jsonl"
            response_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.write_text(json.dumps({"key": "k", "status": "ok", "model_requested": "test/model",
                                                  "probe_id": probe["id"], "variant_id": variant["id"],
                                                  "params": {"temperature": 0.7, "provider": None, "max_tokens": 300}, "repeat_index": 0,
                                                  "response_text": "Mother."}) + "\n" + json.dumps({"key": "old", "status": "ok", "model_requested": "test/model",
                                                  "probe_id": probe["id"], "variant_id": variant["id"],
                                                  "params": {"temperature": 0.7, "provider": ["old-provider"], "max_tokens": 300}, "repeat_index": 1,
                                                  "response_text": "Father."}) + "\n", encoding="utf-8")
            config = {"models": [{"id": "test/model", "provider": None}], "judge": {"id": "judge/model", "max_tokens": 80}, "schedule": [{"temperature": 0.7, "repeats": 1}], "max_tokens": 300, "seed": 7}
            output = io.StringIO()
            with redirect_stdout(output):
                score_models(type("Args", (), {"data_dir": str(data), "mock": True, "models": None})(), config, [probe], MockClient([probe], seed=7, simulate_errors=False))
            self.assertIn("superseded=1", output.getvalue())
            record = json.loads((data / "labels" / "test__model.jsonl").read_text(encoding="utf-8").splitlines()[0])
            for field in ("key", "probe_id", "variant_id", "model", "temperature", "repeat_index", "label", "source", "detail", "scorer_version", "judge_prompt_version"):
                self.assertIn(field, record)
            self.assertEqual(record["label"], "mother")


if __name__ == "__main__":
    unittest.main()
