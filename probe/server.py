"""Small localhost dashboard server for live probe runs.

The server deliberately has no dependency on the report renderer or a web
framework.  It reuses the normal runner and scorer, and keeps the API key on
the server side.
"""
from __future__ import annotations

import copy
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

from .bank import validate_probe
from .client import LazyOpenRouterClient, MockClient
from .config import model_slug
from .runner import run
from .scorer import score_response
from .stats import aggregate_model_stats, load_analysis_records


class DashboardState:
    def __init__(self, config, probes, reports_dir="reports", data_dir="data_live", mock=False):
        self.config = config
        self.probes = list(probes)
        self.reports_dir = Path(reports_dir).resolve()
        self.data_dir = Path(data_dir)
        self.mock = mock
        self.jobs = {}
        self.lock = threading.RLock()

    def _job_snapshot(self, job):
        with self.lock:
            result = dict(job)
            result.pop("thread", None)
            return result

    def start_run(self, probe, model_ids, schedule):
        selected = [m for m in self.config.get("models", []) if m["id"] in model_ids]
        if not selected:
            raise ValueError("at least one configured model is required")
        if not schedule or any(int(item.get("repeats", 0)) < 1 for item in schedule):
            raise ValueError("schedule must contain positive repeats")
        total = len(selected) * len(probe["variants"]) * sum(int(x["repeats"]) for x in schedule)
        if total > 200:
            raise ValueError("job exceeds the 200-call maximum")
        job_id = uuid.uuid4().hex
        job = {"job_id": job_id, "status": "queued", "done": 0, "total": total,
               "errors": 0, "spend": 0.0, "result": None, "error": None}
        with self.lock:
            self.jobs[job_id] = job
        thread = threading.Thread(target=self._run_job, args=(job_id, probe, selected, schedule), daemon=True)
        job["thread"] = thread
        thread.start()
        return job_id

    def _run_job(self, job_id, probe, selected, schedule):
        with self.lock:
            self.jobs[job_id]["status"] = "running"
        config = copy.deepcopy(self.config)
        config["models"] = selected
        config["schedule"] = schedule
        config["spend_cap_usd"] = min(float(config.get("spend_cap_usd", 0) or 0.10), 0.10)
        client = MockClient(self.probes + [probe], config.get("seed", 12345), simulate_errors=False) if self.mock else LazyOpenRouterClient()

        def progress(update):
            with self.lock:
                item = self.jobs[job_id]
                item.update(done=update.get("done", 0), errors=update.get("errors", 0),
                            spend=update.get("spend", 0.0))

        try:
            run_result = run(config, [probe], client, self.data_dir,
                             model_ids=[m["id"] for m in selected], progress_callback=progress)
            # Reuse the CLI scorer through its public helper without importing
            # the command parser.  This also exercises the normal judge cache.
            from .cli import score_models
            args = SimpleNamespace(mock=self.mock, data_dir=str(self.data_dir),
                                   models=",".join(m["id"] for m in selected))
            score_models(args, config, self.probes + [probe], client)
            records, _summary = load_analysis_records(config, self.probes + [probe], self.data_dir)
            compact = {"probe": probe, "models": {}}
            for model in selected:
                model_records = [r for r in records if r.get("model_requested") == model["id"]]
                stats = aggregate_model_stats(model_records, [probe], config, 0.7)[0]
                label_lookup = {}
                for row in model_records:
                    identity = (row.get("key"), row.get("probe_id"), row.get("variant_id"),
                                row.get("repeat_index"), (row.get("params") or {}).get("temperature"))
                    label_lookup[identity] = row.get("label_record") or {}
                raw = []
                for row in model_records:
                    identity = (row.get("key"), row.get("probe_id"), row.get("variant_id"),
                                row.get("repeat_index"), (row.get("params") or {}).get("temperature"))
                    label = label_lookup.get(identity, {})
                    raw.append({"variant_id": row.get("variant_id"), "label": label.get("label", "unscorable"),
                                "source": label.get("source", "none"), "temperature": (row.get("params") or {}).get("temperature"),
                                "repeat_index": row.get("repeat_index"), "timestamp": row.get("timestamp_utc"),
                                "provider": row.get("provider_returned"), "file": row.get("_file"),
                                "line": row.get("_line"), "text": row.get("response_text", "")})
                compact["models"][model["id"]] = {"label": model.get("label", model["id"]),
                                                    "stat": stats, "raw": raw}
            with self.lock:
                item = self.jobs[job_id]
                item.update(status="finished", done=run_result.get("written", 0),
                            errors=run_result.get("errors", 0), spend=run_result.get("spend", 0.0),
                            result=compact)
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update(status="error", error=str(exc))


def _json(handler, payload, status=200):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            path = unquote(self.path.split("?", 1)[0])
            if path == "/api/health":
                _json(self, {"ok": True})
                return
            if path == "/api/config":
                _json(self, {"models": [{"id": m["id"], "label": m.get("label", m["id"])} for m in state.config.get("models", [])],
                              "schedule": state.config.get("schedule", [])})
                return
            if path.startswith("/api/jobs/"):
                job = state.jobs.get(path.rsplit("/", 1)[-1])
                if not job:
                    _json(self, {"error": "job not found"}, 404)
                else:
                    _json(self, state._job_snapshot(job))
                return
            if path in {"/", "/index.html"}:
                target = state.reports_dir / "index.html"
            elif path in {"/live", "/live.html"}:
                target = state.reports_dir / "live.html"
            else:
                target = state.reports_dir / path.lstrip("/")
            try:
                target = target.resolve()
                if state.reports_dir not in target.parents and target != state.reports_dir:
                    raise ValueError("outside reports")
                data = target.read_bytes()
            except (OSError, ValueError):
                self.send_error(404)
                return
            content_type = "text/html; charset=utf-8" if target.suffix == ".html" else "text/plain; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                payload = _read_json(self)
                if path == "/api/run":
                    probe = validate_probe(payload.get("probe"), "request")
                    models = payload.get("models") or [m["id"] for m in state.config.get("models", [])]
                    schedule = payload.get("schedule") or state.config.get("schedule", [])
                    job_id = state.start_run(probe, models, schedule)
                    _json(self, {"job_id": job_id}, 202)
                    return
                if path == "/api/save":
                    probe = validate_probe(payload.get("probe"), "request")
                    if any(p.get("id") == probe["id"] for p in state.probes):
                        raise ValueError("duplicate probe id: %s" % probe["id"])
                    custom = Path("probes/custom.json")
                    content = json.loads(custom.read_text(encoding="utf-8")) if custom.exists() else {"format": "probe-bank/v1", "probes": []}
                    if any(p.get("id") == probe["id"] for p in content.get("probes", [])):
                        raise ValueError("duplicate probe id: %s" % probe["id"])
                    content.setdefault("probes", []).append(probe)
                    custom.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    state.probes.append(probe)
                    _json(self, {"ok": True, "probe_id": probe["id"]})
                    return
                _json(self, {"error": "not found"}, 404)
            except Exception as exc:
                _json(self, {"error": str(exc)}, 400)

    return Handler


def create_server(config, probes, host="127.0.0.1", port=8000, reports_dir="reports",
                  data_dir="data_live", mock=False):
    state = DashboardState(config, probes, reports_dir, data_dir, mock)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.dashboard_state = state
    return server

