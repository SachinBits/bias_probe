import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from html.parser import HTMLParser
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
            build_report(self.config, self.probes, Path("data"), output, mock=False)
            text = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("Probes", text)
            self.assertNotIn("<small>Calls</small>", text)
            self.assertIn("Calls:", text)
            self.assertIn("Significant biases", text)
            self.assertIn("Scorer check", text)
            self.assertIn("GPT-4o-mini answers 'mother' 10/10 even when the prompt says the surgeon is the father.", text)
            self.assertIn("Top findings", text)
            self.assertIn("Memorised riddle versus a prompt that states the answer.", text)
            self.assertEqual(text.count("class='comparison-row'"), len(self.probes))
            for probe in self.probes:
                self.assertIn("probes/%s.html" % probe["id"], text)
                self.assertTrue((output / "probes" / (probe["id"] + ".html")).exists())
            self.assertIn("verdict-badge", text)
            self.assertIn("@media (max-width:700px)", text)

    def test_live_scripts_parse_and_example_is_real_bank_json(self):
        class Scripts(HTMLParser):
            def __init__(self):
                super().__init__()
                self.items = []
                self.current = None

            def handle_starttag(self, tag, attrs):
                if tag == "script":
                    self.current = [dict(attrs).get("type", ""), ""]
                    self.items.append(self.current)

            def handle_data(self, data):
                if self.current is not None:
                    self.current[1] += data

            def handle_endtag(self, tag):
                if tag == "script":
                    self.current = None

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reports"
            build_report(self.config, self.probes, Path("data_mock"), output, mock=True)
            parser = Scripts()
            parser.feed((output / "live.html").read_text(encoding="utf-8"))
            json_blocks = {body_id: json.loads(body) for body_id, body in
                           (("example-data" if "surgeon-riddle" in body else "config-data", body)
                            for kind, body in parser.items if kind == "application/json")}
            example = json_blocks["example-data"]
            bank_probe = next(p for p in self.probes if p["id"] == "surgeon-riddle")
            self.assertEqual(example["id"], bank_probe["id"])
            self.assertEqual(example["variants"], bank_probe["variants"])
            self.assertIn("demo-surgeon-riddle", "".join(body for _kind, body in parser.items))
            self.assertNotIn("Live runs need the local server", (output / "live.html").read_text(encoding="utf-8"))
            node = shutil.which("node")
            if node:
                for kind, body in parser.items:
                    if kind:
                        continue
                    script = Path(tmp) / "inline.js"
                    script.write_text(body, encoding="utf-8")
                    checked = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
                    self.assertEqual(checked.returncode, 0, checked.stderr)

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
