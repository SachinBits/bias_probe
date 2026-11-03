"""Command-line interface for Phase 1."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from .bank import load_probe_files
from .client import MockClient, OpenRouterClient
from .config import load_config, model_slug
from .judge import Judge
from .runner import run
from .scorer import JUDGE_PROMPT_VERSION, SCORER_VERSION, rule_score, score_response
from .store import JsonlStore


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
    for name in ("score", "judge-test"):
        sp = sub.add_parser(name)
        sp.add_argument("--mock", action="store_true")
        sp.add_argument("--data-dir")
        sp.add_argument("--models")
    sp = sub.add_parser("sample-hand-labels")
    sp.add_argument("--n", type=int, default=60)
    sp.add_argument("--mock", action="store_true")
    sp.add_argument("--data-dir")
    sp = sub.add_parser("check-scorer")
    sp.add_argument("--mock", action="store_true")
    sp.add_argument("--data-dir")
    return p


def _make_client(args, probes, config):
    if args.mock:
        return MockClient(probes, config["seed"], simulate_errors=False)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required (use --mock for offline testing)")
    return OpenRouterClient(key)


def _data_dir(args):
    return Path(args.data_dir or ("data_mock" if args.mock else "data"))


def _probe_maps(probes):
    by_id = {p["id"]: p for p in probes}
    variants = {(p["id"], v["id"]): v for p in probes for v in p["variants"]}
    return by_id, variants


def score_models(args, config, probes, client):
    data_dir = _data_dir(args)
    by_id, variants = _probe_maps(probes)
    judge = Judge(client, config, data_dir)
    model_ids = set(args.models.split(",")) if getattr(args, "models", None) else None
    results = []
    for model in config["models"]:
        if model_ids and model["id"] not in model_ids:
            continue
        response_path = data_dir / "responses" / (model_slug(model["id"]) + ".jsonl")
        label_path = data_dir / "labels" / (model_slug(model["id"]) + ".jsonl")
        if label_path.exists():
            label_path.unlink()
        label_store = JsonlStore(label_path)
        counts = defaultdict(int)
        if response_path.exists():
            for response in JsonlStore(response_path).records():
                if response.get("status") != "ok":
                    continue
                probe = by_id.get(response.get("probe_id"))
                variant = variants.get((response.get("probe_id"), response.get("variant_id")))
                if not probe or not variant:
                    continue
                scored = score_response(response.get("response_text", ""), probe, variant, judge)
                rule = rule_score(response.get("response_text", ""), probe, variant)
                record = {"key": response.get("key"), "probe_id": response.get("probe_id"), "variant_id": response.get("variant_id"),
                          "model": response.get("model_requested"), "temperature": response.get("params", {}).get("temperature"),
                          "repeat_index": response.get("repeat_index"), "label": scored["label"], "source": scored["source"],
                          "detail": scored["detail"], "scorer_version": SCORER_VERSION,
                          "judge_prompt_version": JUDGE_PROMPT_VERSION, "response_text": response.get("response_text", ""),
                          "rule_label": rule.get("label"), "judge_label": scored["label"] if scored["source"] == "judge" else None}
                label_store.append(record)
                counts[scored["source"] + ":" + scored["label"]] += 1
        results.append((model["id"], dict(counts)))
        print("%s: %s" % (model["id"], ", ".join("%s=%d" % x for x in sorted(counts.items())) or "no responses"))
    return results


def run_judge_tests(args, config, probes):
    path = Path("judge_tests") / "judge_tests.json"
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    client = _make_client(args, probes, config)
    judge = Judge(client, config, _data_dir(args))
    labels = ["mother", "father", "nurse", "doctor", "grandfather", "grandson", "unknown", "unclear", "equal", "bricks", "feathers", "canberra", "sydney", "hedge", "refused", "unscorable"]
    rule_correct = judge_correct = counter_rule = counter_judge = counter_total = 0
    for item in items:
        if item["gold"] in {"unknown", "unclear"}:
            item_labels = ["grandson", "grandfather", "unknown", "unclear", "other", "refused", "hedge"]
        elif item["gold"] in {"doctor", "nurse"}:
            item_labels = ["doctor", "nurse", "unclear", "other", "refused", "hedge"]
        elif item["gold"] == "hedge":
            item_labels = ["mother", "father", "hedge", "refused", "unscorable"]
        elif item["gold"] == "refused":
            item_labels = ["mother", "father", "refused", "hedge", "unscorable"]
        else:
            item_labels = list(dict.fromkeys([item["gold"], "mother", "father", "other", "refused", "hedge", "unscorable"]))
        probe = {"labels": item_labels}
        variant = {"id": "synthetic", "prompt": ""}
        rule = rule_score(item["answer"], probe, variant)
        judged, _ = judge.classify(item["answer"], item_labels, "")
        rule_correct += rule.get("label") == item["gold"]
        judge_correct += judged == item["gold"]
        if item.get("counter_stereotype"):
            counter_total += 1
            counter_rule += rule.get("label") == item["gold"]
            counter_judge += judged == item["gold"]
    print("rule accuracy: %d/%d (%.1f%%)" % (rule_correct, len(items), 100.0 * rule_correct / len(items)))
    print("judge accuracy: %d/%d (%.1f%%)" % (judge_correct, len(items), 100.0 * judge_correct / len(items)))
    print("counter-stereotype rule accuracy: %d/%d (%.1f%%)" % (counter_rule, counter_total, 100.0 * counter_rule / counter_total))
    print("counter-stereotype judge accuracy: %d/%d (%.1f%%)" % (counter_judge, counter_total, 100.0 * counter_judge / counter_total))


def sample_hand_labels(args):
    data_dir = _data_dir(args)
    rows = []
    for path in sorted((data_dir / "labels").glob("*.jsonl")):
        rows.extend(JsonlStore(path).records())
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("probe_id")].append(row)
    selected = []
    while len(selected) < min(args.n, len(rows)):
        added = False
        for probe_id in sorted(groups):
            if groups[probe_id]:
                selected.append(groups[probe_id].pop(0))
                added = True
                if len(selected) == args.n:
                    break
        if not added:
            break
    path = data_dir / "hand_labels.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "probe_id", "variant_id", "response_text", "rule_label", "judge_label", "human_label"])
        writer.writeheader()
        for row in selected:
            writer.writerow({"key": row.get("key"), "probe_id": row.get("probe_id"), "variant_id": row.get("variant_id"),
                             "response_text": row.get("response_text", ""), "rule_label": row.get("rule_label"),
                             "judge_label": row.get("judge_label"), "human_label": ""})
    print("Wrote %d hand-label rows to %s" % (len(selected), path))


def check_scorer(args):
    path = _data_dir(args) / "hand_labels.csv"
    if not path.exists():
        print("No hand-label file found: %s" % path)
        return
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    labelled = [r for r in rows if r.get("human_label", "").strip()]
    for name in ("rule_label", "judge_label"):
        correct = sum(r.get(name) == r.get("human_label") for r in labelled)
        print("%s accuracy: %d/%d (%.1f%%)" % (name, correct, len(labelled), 100.0 * correct / len(labelled) if labelled else 0.0))
    print("labelled rows: %d/%d" % (len(labelled), len(rows)))


def main(argv=None):
    args = _parser().parse_args(argv)
    config = load_config()
    probes = load_probe_files(config["probe_files"])
    model_ids = set(args.models.split(",")) if getattr(args, "models", None) else None
    probe_ids = set(args.probes.split(",")) if getattr(args, "probes", None) else None
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
    if args.command == "score":
        score_models(args, config, probes, _make_client(args, probes, config))
        return
    if args.command == "judge-test":
        run_judge_tests(args, config, probes)
        return
    if args.command == "sample-hand-labels":
        sample_hand_labels(args)
        return
    if args.command == "check-scorer":
        check_scorer(args)
        return
    data_dir = Path(args.data_dir or ("data_mock" if args.mock else "data"))
    if not args.mock and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required (use --mock for offline testing)")
    client = MockClient(probes, config["seed"]) if args.mock else OpenRouterClient(os.environ["OPENROUTER_API_KEY"])
    run(config, probes, client, data_dir, model_ids, probe_ids, args.dry_run)
