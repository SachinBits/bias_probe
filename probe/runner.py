"""Run probe tasks with retries, resume, concurrency and spend cap."""
from __future__ import annotations

from datetime import datetime, timezone
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from .client import ClientError
from .store import JsonlStore, cache_key


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_tasks(models, probes, schedule, max_tokens):
    for model in models:
        for probe in probes:
            for variant in probe["variants"]:
                for item in schedule:
                    for repeat_index in range(item["repeats"]):
                        yield {"model": model["id"], "probe": probe, "variant": variant,
                               "temperature": item["temperature"], "repeat_index": repeat_index,
                               "max_tokens": max_tokens}


def complete_with_retries(client, model, prompt, temperature, max_tokens, provider=None, seed=0, repeat_index=0, max_attempts=6):
    """Complete one request using the runner's retry and backoff policy."""
    for attempt in range(1, max_attempts + 1):
        try:
            return client.complete(model, prompt, temperature, max_tokens, provider, seed, repeat_index), attempt
        except ClientError as exc:
            exc.attempts = attempt
            if not exc.retryable or attempt == max_attempts:
                raise
            delay = float(exc.retry_after) if exc.retry_after is not None else min(32, 2 ** (attempt - 1))
            time.sleep(delay * (0.75 + random.random() * 0.5))
        except Exception as exc:
            exc.attempts = attempt
            raise


def _call(task, client, store, run_id, provider, seed, max_attempts=6):
    probe, variant = task["probe"], task["variant"]
    key = cache_key(task["model"], provider, variant["prompt"], task["temperature"], task["max_tokens"], task["repeat_index"])
    started = time.time()
    attempts_made = 0
    try:
        result, attempts_made = complete_with_retries(client, task["model"], variant["prompt"], task["temperature"], task["max_tokens"], provider, seed, task["repeat_index"], max_attempts)
        return {"schema_version": 1, "key": key, "run_id": run_id, "timestamp_utc": _now(),
                "model_requested": task["model"], "model_returned": result.get("model_returned"),
                "provider_returned": result.get("provider_returned"), "response_id": result.get("response_id"),
                "probe_id": probe["id"], "family": probe["family"], "variant_id": variant["id"],
                "repeat_index": task["repeat_index"], "params": {"temperature": task["temperature"], "max_tokens": task["max_tokens"], "provider": provider},
                "prompt": variant["prompt"], "response_text": result.get("response_text", ""),
                "finish_reason": result.get("finish_reason"), "usage": result.get("usage", {}),
                "cost_usd": result.get("cost_usd") or 0.0, "latency_ms": result.get("latency_ms", int((time.time() - started) * 1000)),
                "status": "ok", "error": None, "attempts": attempts_made, "mock": bool(result.get("mock"))}
    except Exception as exc:
        attempts_made = getattr(exc, "attempts", 1)
        error = str(exc)
        if not isinstance(exc, ClientError):
            error = "Unexpected %s: %s" % (type(exc).__name__, exc)
    return {"schema_version": 1, "key": key, "run_id": run_id, "timestamp_utc": _now(),
            "model_requested": task["model"], "model_returned": None, "provider_returned": None,
            "response_id": None, "probe_id": probe["id"], "family": probe["family"], "variant_id": variant["id"],
            "repeat_index": task["repeat_index"], "params": {"temperature": task["temperature"], "max_tokens": task["max_tokens"], "provider": provider},
            "prompt": variant["prompt"], "response_text": "", "finish_reason": None, "usage": {}, "cost_usd": 0.0,
            "latency_ms": int((time.time() - started) * 1000), "status": "error", "error": error,
            "attempts": attempts_made, "mock": isinstance(getattr(client, "__class__", None), type) and client.__class__.__name__ == "MockClient"}


def run(config, probes, client, data_dir, model_ids=None, probe_ids=None, dry_run=False):
    selected_models = [m for m in config["models"] if not model_ids or m["id"] in model_ids]
    selected_probes = [p for p in probes if not probe_ids or p["id"] in probe_ids]
    tasks = list(build_tasks(selected_models, selected_probes, config["schedule"], config["max_tokens"]))
    run_id = str(uuid.uuid4())
    total = 0
    planned = 0
    for model in selected_models:
        store = JsonlStore(str(data_dir) + "/responses/" + model["id"].replace("/", "__").replace(":", "__") + ".jsonl")
        keys = store.ok_task_keys()
        planned += sum(1 for t in tasks if t["model"] == model["id"] and
                       (cache_key(t["model"], model.get("provider"), t["variant"]["prompt"], t["temperature"], t["max_tokens"], t["repeat_index"]), t["probe"]["id"], t["variant"]["id"]) not in keys)
    if dry_run:
        print("Planned tasks: %d; estimated new calls: %d" % (len(tasks), planned))
        return {"planned": len(tasks), "new": planned, "written": 0, "skipped": len(tasks) - planned}
    written = skipped = errors = 0
    spend = 0.0
    grouped = {}
    for task in tasks:
        grouped.setdefault(task["model"], []).append(task)
    for model in selected_models:
        provider = model.get("provider")
        store = JsonlStore(str(data_dir) + "/responses/" + model["id"].replace("/", "__").replace(":", "__") + ".jsonl")
        keys = store.ok_task_keys()
        pending = []
        for task in grouped.get(model["id"], []):
            key = cache_key(task["model"], provider, task["variant"]["prompt"], task["temperature"], task["max_tokens"], task["repeat_index"])
            task_key = (key, task["probe"]["id"], task["variant"]["id"])
            if task_key in keys:
                skipped += 1
            else:
                pending.append(task)
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
            active = {}
            next_index = 0
            cap = float(config.get("spend_cap_usd", 0) or 0)
            while next_index < len(pending) and len(active) < config["concurrency"]:
                active[pool.submit(_call, pending[next_index], client, store, run_id, provider, config["seed"])] = pending[next_index]
                next_index += 1
            while active:
                for future in as_completed(list(active)):
                    active.pop(future)
                    record = future.result()
                    store.append(record)
                    written += 1
                    errors += record["status"] == "error"
                    spend += float(record.get("cost_usd") or 0.0)
                    total += 1
                    if total % 25 == 0:
                        print("Progress: done=%d skipped=%d errors=%d spend=$%.4f" % (total, skipped, errors, spend))
                    if next_index < len(pending) and (not cap or spend < cap):
                        active[pool.submit(_call, pending[next_index], client, store, run_id, provider, config["seed"])] = pending[next_index]
                        next_index += 1
                    break
                if cap and spend >= cap and next_index < len(pending):
                    print("Spend cap reached at $%.4f; no new tasks will be submitted." % spend)
                    next_index = len(pending)
    print("Run complete: written=%d skipped=%d errors=%d spend=$%.4f" % (written, skipped, errors, spend))
    return {"planned": len(tasks), "new": planned, "written": written, "skipped": skipped, "errors": errors, "spend": spend}
