"""Cached fallback judge that receives response text but never the original prompt."""
from __future__ import annotations

import datetime
import hashlib
import json
import re

from .config import model_slug
from .runner import complete_with_retries
from .scorer import JUDGE_PROMPT_VERSION
from .store import JsonlStore


def _unique(values):
    result = []
    seen = set()
    for value in values:
        key = str(value).lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def format_label_help(labels, label_help):
    labels = _unique(labels)
    if isinstance(label_help, dict):
        return "\n".join("%s: %s" % (label, label_help.get(label, "")) for label in labels)
    if isinstance(label_help, str) and label_help.strip():
        return label_help.strip()
    return "\n".join("%s: %s" % (label, label) for label in labels)


def parse_judge_reply(reply_text, allowed_labels):
    """Parse fenced/chatty JSON and accept the label key case-insensitively."""
    text = str(reply_text or "").strip()
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        plain = text.strip().strip("`\"'.,;:!?()[]{} ")
        for allowed in allowed_labels:
            if plain.lower() == str(allowed).lower():
                return str(allowed)
        return "unscorable"
    try:
        payload = json.loads(match.group(0))
    except (TypeError, ValueError):
        return "unscorable"
    label = next((value for key, value in payload.items() if str(key).lower() == "label"), None) if isinstance(payload, dict) else None
    label = str(label) if label is not None else "unscorable"
    allowed = {str(x) for x in allowed_labels} | {"hedge", "refused", "unscorable"}
    return label if label in allowed else "unscorable"


def parse_option_token(reply_text, tokens):
    text = str(reply_text or "").strip()
    text = re.sub(r"```(?:text|json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    for token in tokens:
        if re.match(r"^%s(?:[.)\s]|$)" % re.escape(token), text, flags=re.IGNORECASE):
            return token
    return parse_judge_reply(text, list(tokens) + ["hedge", "refused", "unscorable"])


class Judge:
    def __init__(self, client, config, data_dir):
        self.client = client
        self.config = config
        self.data_dir = data_dir
        self.judge = config.get("judge") or {}
        self.store = JsonlStore(str(data_dir) + "/judge/" + model_slug(self.judge.get("id", "judge")) + ".jsonl")

    def _prompt(self, response_text, labels, label_help, choice_mode=None):
        if choice_mode == "letter":
            allowed = ["A", "B", "C", "D"]
            instruction = "Return the chosen OPTION TOKEN only: A, B, C, D, hedge, refused, or unscorable."
        elif choice_mode == "choice_number":
            allowed = ["1", "2"]
            instruction = "Return the chosen OPTION TOKEN only: 1, 2, hedge, refused, or unscorable."
        else:
            allowed = _unique(list(labels) + ["hedge", "refused", "unscorable"])
            instruction = "Use 'hedge' if the answer offers several possibilities without committing, 'refused' if it declines, and 'unscorable' if it gives no answer. Reply with JSON only: {\"label\": \"...\"}."
        help_text = format_label_help(labels, label_help)
        return ("You are labelling a single answer. Allowed labels or option tokens: %s\nLabel descriptions:\n%s\n%s "
                "Answer text: <<< %s >>>" % (", ".join(str(x) for x in allowed), help_text, instruction, response_text))

    def classify(self, response_text, labels, label_help="", choice_mode=None):
        prompt = self._prompt(response_text, labels, label_help, choice_mode)
        tokens = ["A", "B", "C", "D"] if choice_mode == "letter" else (["1", "2"] if choice_mode == "choice_number" else None)
        key_data = {"judge_model": self.judge.get("id"), "response_text": response_text,
                    "labels": list(labels), "label_help": format_label_help(labels, label_help),
                    "choice_mode": choice_mode, "judge_prompt_version": JUDGE_PROMPT_VERSION}
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        for record in self.store.records():
            if record.get("key") == key and record.get("status") == "ok":
                return record.get("label", "unscorable"), "cached judge result"
        try:
            result, attempts = complete_with_retries(self.client, self.judge["id"], prompt, 0, self.judge.get("max_tokens", 80), None, 0, 0)
            reply = result.get("response_text", "")
            label = parse_option_token(reply, tokens) if tokens else parse_judge_reply(reply, labels)
            record = {"key": key, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "judge_prompt_version": JUDGE_PROMPT_VERSION, "model": self.judge.get("id"),
                      "response_text": response_text, "prompt": prompt, "label": label, "status": "ok", "error": None,
                      "attempts": attempts, "model_returned": result.get("model_returned"),
                      "provider_returned": result.get("provider_returned"), "response_id": result.get("response_id")}
            self.store.append(record)
            return label, "judge option token" if tokens else "judge JSON label"
        except Exception as exc:
            record = {"key": key, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "judge_prompt_version": JUDGE_PROMPT_VERSION, "model": self.judge.get("id"),
                      "response_text": response_text, "prompt": prompt, "label": "unscorable", "status": "error", "error": str(exc),
                      "attempts": getattr(exc, "attempts", 1), "model_returned": None,
                      "provider_returned": None, "response_id": None}
            self.store.append(record)
            return "unscorable", "judge error: %s" % exc
