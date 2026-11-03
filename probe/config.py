"""Configuration and minimal .env loading."""
from __future__ import annotations

import json
import os
from pathlib import Path


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def load_config(path: str = "config.json") -> dict:
    load_env()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "__")
