import json
import tempfile
import unittest
from pathlib import Path

from probe.bank import BankError, load_probe_files
from probe.client import ClientError, MockClient
from probe.runner import run
from probe.store import cache_key


ROOT = Path(__file__).resolve().parents[1]


class Phase1Tests(unittest.TestCase):
    def test_bank_and_stereotype_normalisation(self):
        probes = load_probe_files([str(ROOT / "probes" / "bank.json"), str(ROOT / "probes" / "custom.json")])
        self.assertEqual(len(probes), 26)
        bbq = next(p for p in probes if p["id"] == "bbq-age-ambiguous")
        winobias = next(p for p in probes if p["id"] == "winobias-pronoun")
        self.assertIsInstance(bbq["stereotype"], dict)
        self.assertIsInstance(winobias["stereotype"], dict)
        self.assertEqual(bbq["stereotype"]["grandson-first"], "grandfather")
        self.assertEqual(winobias["stereotype"]["she"], "nurse")

    def test_validator_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.json"
            p = {"format": "probe-bank/v1", "probes": [{"id": "x", "family": "social", "expect": "same", "labels": ["a"], "variants": [{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}], "why": "x"}, {"id": "x", "family": "social", "expect": "same", "labels": ["a"], "variants": [{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}], "why": "x"}]}
            path.write_text(json.dumps(p), encoding="utf-8")
            with self.assertRaises(BankError):
                load_probe_files([str(path)])

    def test_cache_key_stability_and_schema_independence(self):
        a = cache_key("m", None, "p", 0.7, 10, 1)
        b = cache_key("m", None, "p", 0.7, 10, 1)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)

    def config(self, data_dir):
        return {"models": [{"id": "test/model", "provider": None}], "schedule": [{"temperature": 0.7, "repeats": 2}], "max_tokens": 20, "concurrency": 2, "spend_cap_usd": 3.0, "seed": 12345}

    def test_resume_second_mock_run_has_zero_new_calls(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data_mock"
            probes = [next(p for p in load_probe_files([str(ROOT / "probes" / "starter.json")]) if p["id"] == "feathers-bricks")]
            first = run(self.config(data), probes, MockClient(1), data, None, None, False)
            second = run(self.config(data), probes, MockClient(1), data, None, None, False)
            self.assertGreater(first["written"], 0)
            self.assertEqual(second["written"], 0)
            self.assertEqual(second["skipped"], first["written"])

    def test_crash_mid_run_resumes_without_duplicates(self):
        class CrashClient(MockClient):
            def __init__(self):
                super().__init__(1)
                self.calls = 0
            def complete(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 3:
                    raise KeyboardInterrupt()
                return super().complete(*args, **kwargs)
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data_mock"
            probes = [next(p for p in load_probe_files([str(ROOT / "probes" / "starter.json")]) if p["id"] == "feathers-bricks")]
            with self.assertRaises(KeyboardInterrupt):
                run(self.config(data), probes, CrashClient(), data, None, None, False)
            result = run(self.config(data), probes, MockClient(1), data, None, None, False)
            self.assertGreater(result["written"], 0)
            lines = list((data / "responses" / "test__model.jsonl").read_text(encoding="utf-8").splitlines())
            keys = [json.loads(line)["key"] for line in lines if json.loads(line)["status"] == "ok"]
            self.assertEqual(len(keys), len(set(keys)))

    def test_retry_on_simulated_429(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data_mock"
            probes = [next(p for p in load_probe_files([str(ROOT / "probes" / "starter.json")]) if p["id"] == "feathers-bricks")]
            result = run(self.config(data), probes, MockClient(1, fail_first=True), data, None, None, False)
            self.assertEqual(result["errors"], 0)
            records = [json.loads(line) for line in (data / "responses" / "test__model.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(r["attempts"] >= 2 for r in records))
            self.assertTrue(all(r["schema_version"] == 1 for r in records))


if __name__ == "__main__":
    unittest.main()
