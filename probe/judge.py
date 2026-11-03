"""Cached fallback judge that receives response text but never the original prompt."""
from __future__ import annotations

import datetime
import hashlib
import json

from .client import ClientError
from .config import model_slug
from .scorer import JUDGE_PROMPT_VERSION
from .store import JsonlStore


class Judge:
    def __init__(self, client, config, data_dir):
        self.client = client
        self.config = config
        self.data_dir = data_dir
        self.judge = config.get("judge") or {}
        self.store = JsonlStore(str(data_dir) + "/judge/" + model_slug(self.judge.get("id", "judge")) + ".jsonl")

    def _prompt(self, response_text, labels, label_help):
        allowed = list(labels) + ["hedge", "refused", "unscorable"]
        return ("You are labelling a single answer. Allowed labels: %s. %s Use 'hedge' if the answer offers several possibilities without committing, "
                "'refused' if it declines, and 'unscorable' if it gives no answer. Answer text: <<< %s >>>. Reply with JSON only: {\"label\": \"...\"}"
                % (", ".join(str(x) for x in allowed), label_help or "", response_text))

    def classify(self, response_text, labels, label_help=""):
        prompt = self._prompt(response_text, labels, label_help)
        key_data = {"judge_model": self.judge.get("id"), "response_text": response_text,
                    "labels": list(labels), "label_help": label_help, "judge_prompt_version": JUDGE_PROMPT_VERSION}
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        for record in self.store.records():
            if record.get("key") == key and record.get("status") == "ok":
                return record.get("label", "unscorable"), "cached judge result"
        try:
            result = self.client.complete(self.judge["id"], prompt, 0, self.judge.get("max_tokens", 80), None, 0, 0)
            payload = json.loads(result.get("response_text", ""))
            label = str(payload.get("label", "unscorable"))
            allowed = {str(x) for x in labels} | {"hedge", "refused", "unscorable"}
            if label not in allowed:
                label = "unscorable"
            record = {"key": key, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "judge_prompt_version": JUDGE_PROMPT_VERSION, "model": self.judge.get("id"),
                      "response_text": response_text, "prompt": prompt, "label": label, "status": "ok", "error": None}
            self.store.append(record)
            return label, "judge JSON label"
        except (ClientError, ValueError, TypeError, KeyError) as exc:
            self.store.append({"key": key, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               "judge_prompt_version": JUDGE_PROMPT_VERSION, "model": self.judge.get("id"),
                               "response_text": response_text, "prompt": prompt, "label": "unscorable", "status": "error", "error": str(exc)})
            return "unscorable", "judge error: %s" % exc
