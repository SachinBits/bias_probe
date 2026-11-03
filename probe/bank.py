"""Load and validate probe-bank/v1 files."""
from __future__ import annotations

import json
import re
from pathlib import Path

FAMILIES = {"familiar-pattern", "social", "position", "sycophancy", "control"}
EXPECTS = {"same", "different", "trap"}
PARSES = {"integer", "int_1_10", "amount", "letter", "choice_number"}


class BankError(ValueError):
    pass


def _normalise_stereotype(probe: dict) -> None:
    value = probe.get("stereotype")
    if isinstance(value, str):
        probe["stereotype"] = {v["id"]: value for v in probe["variants"]}
    elif isinstance(value, dict):
        probe["stereotype"] = dict(value)
    elif value is not None:
        raise BankError("stereotype must be a string or object")


def validate_probe(probe: dict, source: str = "probe") -> dict:
    required = {"id", "family", "expect", "labels", "variants", "why"}
    missing = required - set(probe)
    if missing:
        raise BankError("%s missing %s" % (source, ", ".join(sorted(missing))))
    if probe["family"] not in FAMILIES:
        raise BankError("%s has unknown family" % source)
    if probe["expect"] not in EXPECTS:
        raise BankError("%s has unknown expect" % source)
    if not isinstance(probe["labels"], list) or not probe["labels"]:
        raise BankError("%s labels must be a non-empty list" % source)
    variants = probe["variants"]
    if not isinstance(variants, list) or len(variants) < 2:
        raise BankError("%s needs at least two variants" % source)
    ids = [v.get("id") for v in variants]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise BankError("%s has invalid or duplicate variant ids" % source)
    parse = probe.get("parse")
    if parse is not None and parse not in PARSES:
        raise BankError("%s has unknown parse" % source)
    for i, variant in enumerate(variants):
        if not isinstance(variant.get("prompt"), str):
            raise BankError("%s variant %d has no prompt" % (source, i))
        if parse in {"letter", "choice_number"} and not isinstance(variant.get("map"), dict):
            raise BankError("%s requires map on every variant" % source)
    for label, patterns in probe.get("match", {}).items():
        if not isinstance(patterns, list):
            raise BankError("%s match patterns must be lists" % source)
        for pattern in patterns:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise BankError("%s invalid regex: %s" % (source, exc))
    _normalise_stereotype(probe)
    return probe


def load_file(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "probe-bank/v1":
        raise BankError("%s is not probe-bank/v1" % path)
    return [validate_probe(p, "%s:%s" % (path, p.get("id", "?"))) for p in data.get("probes", [])]


def load_probe_files(paths: list) -> list:
    probes = []
    seen = set()
    for path in paths:
        for probe in load_file(path):
            if probe["id"] in seen:
                raise BankError("duplicate probe id: %s" % probe["id"])
            seen.add(probe["id"])
            probes.append(probe)
    return probes
