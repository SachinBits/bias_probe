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
            raise ClientError("HTTP %s" % exc.code, retryable, retry_after)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ClientError(str(exc), True)
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {"model_returned": payload.get("model", model), "provider_returned": payload.get("provider"),
                "response_id": payload.get("id"), "response_text": message.get("content", ""),
                "finish_reason": choice.get("finish_reason"), "usage": payload.get("usage") or {},
                "cost_usd": (payload.get("usage") or {}).get("cost"), "latency_ms": int((time.time() - started) * 1000), "mock": False}


class MockClient:
    def __init__(self, seed=12345, fail_first=False):
        self.seed = seed
        self.fail_first = fail_first
        self.failed = set()

    def complete(self, model, prompt, temperature, max_tokens, provider=None, seed=0, repeat_index=0):
        key = (model, prompt, temperature, repeat_index)
        if self.fail_first and key not in self.failed:
            self.failed.add(key)
            raise ClientError("simulated HTTP 429", True, 0)
        digest = hashlib.sha256((str(self.seed) + prompt + str(temperature) + str(repeat_index)).encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        lower = prompt.lower()
        if "surgeon" in lower and "father" in lower and "who is the surgeon" in lower:
            text = "The surgeon is the boy's mother." if rng.random() < (0.7 if temperature else 0.2) else "The surgeon is the boy's father."
        elif "which of these numbers is prime" in lower:
            text = "A" if rng.random() < 0.35 else "B"
        elif "scale from 1 to 10" in lower:
            text = str(max(1, min(10, 7 + rng.choice([-1, 0, 0, 1]))))
        elif "answer with the amount" in lower:
            text = "$0.05" if "more than" in lower else "$0.10"
        elif "answer with the number" in lower:
            text = "7"
        elif rng.random() < 0.04:
            text = "I can't determine that."
        else:
            text = "The answer is correct."
        return {"model_returned": model, "provider_returned": "mock", "response_id": "mock-" + digest[:12],
                "response_text": text, "finish_reason": "stop", "usage": {"cost": 0}, "cost_usd": 0.0,
                "latency_ms": 0, "mock": True}
