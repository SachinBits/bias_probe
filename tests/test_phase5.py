import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from probe.bank import load_probe_files
from probe.config import load_config
from probe.report import build_report
from probe.server import create_server


class Phase5DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.probes = load_probe_files(cls.config["probe_files"])

    def test_report_has_kpis_matrix_and_probe_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reports"
            build_report(self.config, self.probes, Path("data_mock"), output, mock=True)
            text = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("Probes", text)
            self.assertIn("Calls", text)
            self.assertIn("Significant biases", text)
            self.assertIn("Scorer check", text)
            self.assertEqual(text.count("class='comparison-row'"), len(self.probes))
            for probe in self.probes:
                self.assertIn("probes/%s.html" % probe["id"], text)
                self.assertTrue((output / "probes" / (probe["id"] + ".html")).exists())
            self.assertIn("verdict-badge", text)
            self.assertIn("@media (max-width:700px)", text)

    def test_local_api_mock_job_and_validation(self):
        probe = {"id": "phase5-server-test", "family": "control", "expect": "different",
                 "labels": ["1", "2", "other", "hedge", "refused"], "why": "test",
                 "variants": [{"id": "a", "prompt": "A?", "answer": "1"},
                              {"id": "b", "prompt": "B?", "answer": "2"}]}
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            reports.mkdir()
            (reports / "index.html").write_text("ok", encoding="utf-8")
            server = create_server(self.config, self.probes, port=0, reports_dir=reports,
                                   data_dir=Path(tmp) / "data_live", mock=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = "http://127.0.0.1:%d" % server.server_address[1]

            def get(path):
                return json.loads(urllib.request.urlopen(base + path).read().decode("utf-8"))

            self.assertTrue(get("/api/health")["ok"])
            self.assertNotIn("OPENROUTER_API_KEY", get("/api/config"))

            payload = {"probe": probe, "models": [self.config["models"][0]["id"]],
                       "schedule": [{"temperature": 0.0, "repeats": 1}]}
            request = urllib.request.Request(base + "/api/run", data=json.dumps(payload).encode("utf-8"),
                                              headers={"Content-Type": "application/json"}, method="POST")
            job_id = json.loads(urllib.request.urlopen(request).read().decode("utf-8"))["job_id"]
            for _ in range(100):
                job = get("/api/jobs/" + job_id)
                if job["status"] in {"finished", "error"}:
                    break
                time.sleep(0.05)
            self.assertEqual(job["status"], "finished")
            self.assertIn("models", job["result"])
            self.assertEqual(job["errors"], 0)

            bad = dict(payload)
            bad["probe"] = {"id": "bad"}
            bad_request = urllib.request.Request(base + "/api/run", data=json.dumps(bad).encode("utf-8"),
                                                  headers={"Content-Type": "application/json"}, method="POST")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(bad_request)
            self.assertEqual(caught.exception.code, 400)
            self.assertIn("missing", caught.exception.read().decode("utf-8"))
            caught.exception.close()
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
