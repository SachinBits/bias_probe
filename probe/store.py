"""Append-only JSONL storage and cache keys."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


def cache_key(model, provider, prompt, temperature, max_tokens, repeat_index):
    value = {"model": model, "provider": provider, "prompt": prompt, "temperature": temperature,
             "max_tokens": max_tokens, "repeat_index": repeat_index}
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class JsonlStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def records(self):
        if not self.path.exists():
            return []
        result = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    result.append(json.loads(line))
        return result

    def ok_keys(self):
        return {r["key"] for r in self.records() if r.get("status") == "ok" and r.get("key")}

    def append(self, record):
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
