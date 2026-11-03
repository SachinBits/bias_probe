"""OpenRouter and deterministic mock clients."""
from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.error
import urllib.request


class ClientError(Exception):
    def __init__(self, message, retryable=False, retry_after=None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class OpenRouterClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def complete(self, model, prompt, temperature, max_tokens, provider=None, seed=0, repeat_index=0):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature, "max_tokens": max_tokens, "usage": {"include": True}}
        if provider:
            body["provider"] = {"order": provider, "allow_fallbacks": False} if isinstance(provider, list) else provider
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json", "X-Title": "bias-probe"},
            method="POST")
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or exc.code >= 500
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                body = exc.read(300).decode("utf-8", errors="replace")
            except Exception:
                body = "<response body unavailable>"
            body = body.replace(self.api_key, "[REDACTED]")
            message = "HTTP %s: %s" % (exc.code, body[:300])
            raise ClientError(message, retryable, retry_after) from exc
        except Exception as exc:
            raise ClientError(str(exc), True) from exc
        if not isinstance(payload, dict):
            raise ClientError("invalid JSON response: expected an object", True)
        if payload.get("error"):
            error = payload.get("error")
            message = error if isinstance(error, str) else json.dumps(error, ensure_ascii=False)
            raise ClientError("OpenRouter error: %s" % message[:300], True)
        choices = payload.get("choices")
        if not choices:
            raise ClientError("OpenRouter response has no choices", True)
        choice = choices[0] or {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")
        content = message.get("content")
        if (content is None or not str(content).strip()) and not finish_reason:
            raise ClientError("OpenRouter response has empty message content", True)
        return {"model_returned": payload.get("model", model), "provider_returned": payload.get("provider"),
                "response_id": payload.get("id"), "response_text": content or "",
                "finish_reason": finish_reason, "usage": payload.get("usage") or {},
                "cost_usd": (payload.get("usage") or {}).get("cost"), "latency_ms": int((time.time() - started) * 1000), "mock": False}


class MockClient:
    def __init__(self, probes=None, seed=12345, fail_first=False, simulate_errors=True):
        self.probes = probes or []
        self.seed = seed
        self.fail_first = fail_first
        self.simulate_errors = simulate_errors
        self.failed = set()

    def complete(self, model, prompt, temperature, max_tokens, provider=None, seed=0, repeat_index=0):
        key = (model, prompt, temperature, repeat_index)
        digest = hashlib.sha256((str(self.seed) + prompt + str(temperature) + str(repeat_index)).encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        if (self.fail_first or (self.simulate_errors and rng.random() < 0.03)) and key not in self.failed:
            self.failed.add(key)
            raise ClientError("simulated HTTP 429", True, 0)
        probe, variant = self._find_prompt(prompt)
        roll = rng.random()
        if roll < 0.03:
            text = "I can't answer that."
        elif roll < 0.08:
            text = "It could be either."
        else:
            text = self._answer_text(probe, variant, rng, temperature)
        return {"model_returned": model, "provider_returned": "mock", "response_id": "mock-" + digest[:12],
                "response_text": text, "finish_reason": "stop", "usage": {"cost": 0}, "cost_usd": 0.0,
                "latency_ms": 0, "mock": True}

    def _find_prompt(self, prompt):
        for probe in self.probes:
            for variant in probe.get("variants", []):
                if variant.get("prompt") == prompt:
                    return probe, variant
        return None, {"prompt": prompt}

    @staticmethod
    def _content_labels(probe):
        if not probe:
            return []
        return [x for x in probe.get("labels", []) if x not in {"other", "refused", "hedge"} and not str(x).startswith("numeric")]

    def _answer_text(self, probe, variant, rng, temperature):
        if not probe:
            return "The answer is correct."
        parse = probe.get("parse")
        answer = variant.get("answer", probe.get("answer"))
        variants = probe.get("variants", [])
        if probe.get("expect") == "trap" and variant is not variants[0] and variants:
            original = variants[0].get("answer", probe.get("answer"))
            if original and rng.random() < 0.40:
                answer = original
            elif answer is None:
                answer = original
        elif answer is not None and rng.random() >= 0.75:
            alternatives = [x for x in self._content_labels(probe) if str(x).lower() != str(answer).lower()]
            if alternatives:
                answer = rng.choice(alternatives)
        if parse == "int_1_10":
            base = 7 + (1 if variant.get("id", "").endswith("ahmed") else 0)
            return str(max(1, min(10, base + rng.choice([-1, 0, 0, 1]))))
        if answer is None:
            labels = self._content_labels(probe)
            answer = labels[0] if labels else "unknown"
        if parse == "integer":
            return str(answer)
        if parse == "amount":
            amount = str(answer)
            return rng.choice(["$" + amount, amount, "5 cents" if amount == "0.05" else "10 cents"])
        if parse in {"letter", "choice_number"}:
            mapping = variant.get("map", {})
            key = next((k for k, value in mapping.items() if str(value).lower() == str(answer).lower()), None)
            key = key or next(iter(mapping), "A" if parse == "letter" else "1")
            if parse == "letter":
                return rng.choice([key, key + ") " + str(answer), "The answer is " + key + "."])
            return rng.choice([key, "Choice " + key])
        if str(answer).lower() == "mother":
            return rng.choice(["mother", "The surgeon is the boy's mother."])
        if str(answer).lower() == "father":
            return rng.choice(["father", "Not the mother, it's the father."])
        if str(answer).lower() == "bricks":
            return rng.choice(["bricks", "**Bricks**"])
        if str(answer).lower() == "feathers":
            return "feathers"
        return rng.choice([str(answer), "The answer is " + str(answer) + "."])
