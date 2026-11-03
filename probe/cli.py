"""Command-line interface for Phase 1."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .bank import load_probe_files
from .client import MockClient, OpenRouterClient
from .config import load_config
from .runner import run


def _parser():
    p = argparse.ArgumentParser(prog="python3 -m probe")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("ping", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--models")
        sp.add_argument("--probes")
        sp.add_argument("--mock", action="store_true")
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("--data-dir")
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    config = load_config()
    probes = load_probe_files(config["probe_files"])
    model_ids = set(args.models.split(",")) if args.models else None
    probe_ids = set(args.probes.split(",")) if args.probes else None
    if args.command == "ping":
        client = MockClient(probes, config["seed"]) if args.mock else OpenRouterClient(os.environ.get("OPENROUTER_API_KEY", ""))
        if not args.mock and not client.api_key:
            raise SystemExit("OPENROUTER_API_KEY is required (use --mock for offline testing)")
        for model in config["models"]:
            if model_ids and model["id"] not in model_ids:
                continue
            try:
                result = client.complete(model["id"], "Reply with OK.", 0, 8, model.get("provider"), config["seed"], 0)
                print("%s -> model_returned=%s provider=%s" % (model["id"], result.get("model_returned"), result.get("provider_returned")))
            except Exception as exc:
                print("%s -> ERROR: %s" % (model["id"], exc))
        judge = config.get("judge") or {}
        if judge.get("id"):
            try:
                result = client.complete(judge["id"], "Reply with OK.", judge.get("temperature", 0), judge.get("max_tokens", 80), None, config["seed"], 0)
                print("judge %s -> model_returned=%s provider=%s" % (judge["id"], result.get("model_returned"), result.get("provider_returned")))
            except Exception as exc:
                print("judge %s -> ERROR: %s" % (judge["id"], exc))
        return
    data_dir = Path(args.data_dir or ("data_mock" if args.mock else "data"))
    if not args.mock and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required (use --mock for offline testing)")
    client = MockClient(probes, config["seed"]) if args.mock else OpenRouterClient(os.environ["OPENROUTER_API_KEY"])
    run(config, probes, client, data_dir, model_ids, probe_ids, args.dry_run)
