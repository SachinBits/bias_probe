"""Command-line interface for the bias probe pipeline."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from .bank import BankError, load_probe_files, validate_probe
from .client import LazyOpenRouterClient, MockClient, OpenRouterClient
from .config import load_config, model_slug
from .judge import Judge
from .runner import run
from .report import build_report
from .server import create_server
from .scorer import JUDGE_PROMPT_VERSION, SCORER_VERSION, contains_unnegated_label, rule_score, score_response
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
    for name in ("score", "judge-test", "report", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--mock", action="store_true")
        sp.add_argument("--data-dir")
        sp.add_argument("--models")
        if name == "all":
            sp.add_argument("--probes")
            sp.add_argument("--dry-run", action="store_true")
    sp = sub.add_parser("sample-hand-labels")
    sp.add_argument("--n", type=int, default=60)
    sp.add_argument("--mock", action="store_true")
    sp.add_argument("--data-dir")
    sp = sub.add_parser("check-scorer")
    sp.add_argument("--mock", action="store_true")
    sp.add_argument("--data-dir")
    sp = sub.add_parser("add")
    sp.add_argument("--file")
    sp = sub.add_parser("serve")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--mock", action="store_true")
    return p


def _make_client(args, probes, config):
    if not (config.get("judge") or {}).get("enabled", True):
        return None
    if args.mock:
        return MockClient(probes, config["seed"], simulate_errors=False)
    return LazyOpenRouterClient()


def _data_dir(args):
    return Path(args.data_dir or ("data_mock" if args.mock else "data"))


def _probe_maps(probes):
    by_id = {p["id"]: p for p in probes}
    variants = {(p["id"], v["id"]): v for p in probes for v in p["variants"]}
    return by_id, variants


def _response_matches_config(response, model, config):
    params = response.get("params") or {}
    temperatures = {item["temperature"] for item in config.get("schedule", [])}
    return (response.get("model_requested") == model["id"] and
            params.get("provider") == model.get("provider") and
            params.get("temperature") in temperatures and
            params.get("max_tokens") == config.get("max_tokens"))


def score_models(args, config, probes, client):
    data_dir = _data_dir(args)
    by_id, variants = _probe_maps(probes)
    judge = Judge(client, config, data_dir) if (config.get("judge") or {}).get("enabled", True) else None
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
        superseded = 0
        if response_path.exists():
            for response in JsonlStore(response_path).records():
                if response.get("status") != "ok":
                    continue
                if not _response_matches_config(response, model, config):
                    superseded += 1
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
                uncertainty = next((str(x) for x in probe.get("labels", []) if str(x).lower() in {"unknown", "unclear"}), None)
                stereotype = probe.get("stereotype")
                if isinstance(stereotype, dict):
                    stereotype = stereotype.get(response.get("variant_id"))
                if (probe.get("family") == "social" and uncertainty and stereotype and
                        scored.get("label") == uncertainty and contains_unnegated_label(response.get("response_text", ""), stereotype, probe)):
                    record["stereotype_in_reasoning"] = True
                label_store.append(record)
                counts[scored["source"] + ":" + scored["label"]] += 1
        results.append((model["id"], dict(counts)))
        summary = ", ".join("%s=%d" % x for x in sorted(counts.items())) or "no responses"
        print("%s: %s; superseded=%d" % (model["id"], summary, superseded))
    return results


def run_judge_tests(args, config, probes):
    path = Path("judge_tests") / "judge_tests.json"
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    client = _make_client(args, probes, config)
    judge = Judge(client, config, _data_dir(args)) if (config.get("judge") or {}).get("enabled", True) else None
    by_id, variants = _probe_maps(probes)
    rule_correct = rule_wrong = rule_abstained = 0
    judge_correct = judge_wrong = judge_abstained = 0
    counter_rule = counter_rule_wrong = counter_rule_abstained = counter_judge = counter_judge_wrong = counter_judge_abstained = counter_total = 0
    item_results = []
    for item in items:
        probe = by_id[item["probe_id"]]
        variant = variants.get((item["probe_id"], item.get("variant_id"))) or probe["variants"][0]
        rule = rule_score(item["answer"], probe, variant)
        choice_mode = probe.get("parse") if probe.get("parse") in {"letter", "choice_number"} else None
        if judge is None:
            judged = "unscorable"
        else:
            judged, _ = judge.classify(item["answer"], probe.get("labels", []), probe.get("label_help", ""), choice_mode=choice_mode)
        if choice_mode and judged not in {"hedge", "refused", "unscorable"}:
            mapping = variant.get("map", {})
            judged = next((str(value) for key, value in mapping.items() if str(key) == str(judged)), "other")
        item_results.append({"probe_id": item["probe_id"], "answer": item["answer"], "gold": item["gold"],
                             "rule_label": rule.get("label") or "unscorable", "judge_label": judged,
                             "counter_stereotype": bool(item.get("counter_stereotype"))})
        if rule.get("label") is None:
            rule_abstained += 1
        elif rule.get("label") == item["gold"]:
            rule_correct += 1
        else:
            rule_wrong += 1
        if judged == "unscorable":
            judge_abstained += 1
        elif judged == item["gold"]:
            judge_correct += 1
        else:
            judge_wrong += 1
        if item.get("counter_stereotype"):
            counter_total += 1
            if rule.get("label") is None:
                counter_rule_abstained += 1
            elif rule.get("label") == item["gold"]:
                counter_rule += 1
            else:
                counter_rule_wrong += 1
            if judged == "unscorable":
                counter_judge_abstained += 1
            elif judged == item["gold"]:
                counter_judge += 1
            else:
                counter_judge_wrong += 1
    print("rule overall: correct=%d wrong=%d abstained=%d total=%d" % (rule_correct, rule_wrong, rule_abstained, len(items)))
    print("judge overall: correct=%d wrong=%d abstained=%d total=%d" % (judge_correct, judge_wrong, judge_abstained, len(items)))
    print("counter-stereotype rule: correct=%d wrong=%d abstained=%d total=%d" % (counter_rule, counter_rule_wrong, counter_rule_abstained, counter_total))
    print("counter-stereotype judge: correct=%d wrong=%d abstained=%d total=%d" % (counter_judge, counter_judge_wrong, counter_judge_abstained, counter_total))
    result = {"schema_version": 1,
              "overall": {"rule": {"correct": rule_correct, "wrong": rule_wrong, "abstained": rule_abstained},
                          "judge": {"correct": judge_correct, "wrong": judge_wrong, "abstained": judge_abstained}},
              "counter_stereotype": {"rule": {"correct": counter_rule, "wrong": counter_rule_wrong, "abstained": counter_rule_abstained},
                                     "judge": {"correct": counter_judge, "wrong": counter_judge_wrong, "abstained": counter_judge_abstained}},
              "items": item_results}
    result_path = _data_dir(args) / "judge_test_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Judge-test results written to %s" % result_path)
    return result


def _response_line_index(data_dir, config):
    index = {}
    for model in config.get("models", []):
        path = data_dir / "responses" / (model_slug(model["id"]) + ".jsonl")
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    response = json.loads(line)
                except (TypeError, ValueError):
                    continue
                identity = (response.get("key"), response.get("probe_id"), response.get("variant_id"),
                            response.get("repeat_index"), (response.get("params") or {}).get("temperature"))
                index[(model["id"],) + identity] = line_number
    return index


def _hand_candidates(data_dir, config, probes):
    first_variants = {probe["id"]: probe.get("variants", [{}])[0].get("id") for probe in probes if probe.get("variants")}
    line_index = _response_line_index(data_dir, config)
    candidates_by_id = {}
    model_by_slug = {model_slug(model["id"]): model["id"] for model in config.get("models", [])}
    for path in sorted((data_dir / "labels").glob("*.jsonl")):
        fallback_model = model_by_slug.get(path.stem)
        with path.open(encoding="utf-8") as handle:
            for label_row in handle:
                try:
                    row = json.loads(label_row)
                except (TypeError, ValueError):
                    continue
                model = row.get("model") or fallback_model
                if not model:
                    continue
                source = row.get("source") if row.get("source") in {"rule", "judge"} else "none"
                identity = (model, row.get("key"), row.get("probe_id"), row.get("variant_id"),
                            row.get("repeat_index"), row.get("temperature"))
                candidate = {"key": row.get("key"), "model": model,
                             "line": line_index.get(identity, ""),
                             "probe_id": row.get("probe_id"), "variant_id": row.get("variant_id"),
                             "repeat_index": row.get("repeat_index"), "temperature": row.get("temperature"),
                             "response_text": row.get("response_text", ""), "label": row.get("label"),
                             "label_source": source, "rule_label": row.get("rule_label"),
                             "judge_label": row.get("judge_label"), "human_label": "",
                             "is_long": len(row.get("response_text", "")) > 60,
                             "is_changed": row.get("variant_id") != first_variants.get(row.get("probe_id"))}
                dedup_key = (model, row.get("key"))
                previous = candidates_by_id.get(dedup_key)
                priority = {"none": 2, "judge": 2, "rule": 1}
                if previous is None or priority[source] > priority[previous["label_source"]]:
                    candidates_by_id[dedup_key] = candidate
    return list(candidates_by_id.values())


def _pick_candidates(candidates, n, seed):
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected = []
    selected_ids = set()

    def add(row):
        identity = (row["model"], row["key"])
        if identity in selected_ids or len(selected) >= n:
            return False
        selected_ids.add(identity)
        selected.append(row)
        return True

    # Preserve every available judge/unscorable candidate before filling quotas.
    for row in shuffled:
        if row["label_source"] in {"judge", "none"} or row.get("label") == "unscorable":
            add(row)

    def fill_category(field, target):
        current = sum(bool(row.get(field)) for row in selected)
        for row in shuffled:
            if current >= target or len(selected) >= n:
                break
            if row.get(field) and (row["model"], row["key"]) not in selected_ids:
                if add(row):
                    current += 1

    fill_category("is_long", 15)
    fill_category("is_changed", 15)

    groups = defaultdict(list)
    for row in shuffled:
        if (row["model"], row["key"]) not in selected_ids:
            groups[(row["model"], row["probe_id"], row["variant_id"])].append(row)
    group_keys = list(groups)
    rng.shuffle(group_keys)
    while len(selected) < min(n, len(candidates)) and group_keys:
        progressed = False
        for group_key in list(group_keys):
            if groups[group_key]:
                if add(groups[group_key].pop()):
                    progressed = True
                if len(selected) >= n:
                    break
            if not groups[group_key]:
                group_keys.remove(group_key)
        if not progressed:
            break
    return selected


def sample_hand_labels(args, config=None, probes=None):
    config = config or load_config()
    probes = probes or load_probe_files(config["probe_files"])
    data_dir = _data_dir(args)
    candidates = _hand_candidates(data_dir, config, probes)
    selected = _pick_candidates(candidates, args.n, config.get("seed", 12345))
    path = data_dir / "hand_labels.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["key", "model", "line", "probe_id", "variant_id", "repeat_index", "temperature",
              "response_text", "label", "label_source", "rule_label", "judge_label", "human_label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in selected)
    print("Wrote %d hand-label rows to %s" % (len(selected), path))
    print("Categories: judge/unscorable=%d, long>60=%d, changed/non-first=%d" %
          (sum(row["label_source"] in {"judge", "none"} or row.get("label") == "unscorable" for row in selected),
           sum(row["is_long"] for row in selected), sum(row["is_changed"] for row in selected)))
    return selected


def check_scorer(args):
    path = _data_dir(args) / "hand_labels.csv"
    if not path.exists():
        print("No hand-label file found: %s" % path)
        return
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labelled = [r for r in rows if r.get("human_label", "").strip()]
    for source, field in (("rule", "rule_label"), ("judge", "judge_label")):
        source_rows = [row for row in labelled if (row.get("label_source") or ("judge" if row.get("judge_label") else "rule")) == source]
        correct = sum(row.get(field) == row.get("human_label") for row in source_rows)
        disagreements = [row for row in source_rows if row.get(field) != row.get("human_label")]
        print("%s vs human: %d/%d (%.1f%%); wrong=%d" %
              (source, correct, len(source_rows), 100.0 * correct / len(source_rows) if source_rows else 0.0,
               len(disagreements)))
        confusion = defaultdict(int)
        for row in disagreements:
            confusion[(row.get(field) or "missing", row.get("human_label"))] += 1
        print("%s confusion: %s" % (source, dict(confusion)))
        for row in disagreements:
            model = row.get("model", "unknown")
            line = row.get("line", "?")
            file_name = "data/responses/%s.jsonl" % model_slug(model) if model != "unknown" else "data/responses/?"
            print("DISAGREEMENT %s: %s:%s; response=%s; scorer=%s; human=%s" %
                  (source, file_name, line, row.get("response_text", ""), row.get(field, ""), row.get("human_label", "")))
    print("labelled rows: %d/%d" % (len(labelled), len(rows)))


def _prompt_value(label, required=True):
    value = input(label).strip()
    if required and not value:
        raise SystemExit("%s is required" % label.rstrip(": "))
    return value


def _split_labels(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def _probe_from_file(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("format") == "probe-bank/v1":
        candidates = payload.get("probes", [])
    elif isinstance(payload, dict) and isinstance(payload.get("probe"), dict):
        candidates = [payload["probe"]]
    elif isinstance(payload, dict):
        candidates = [payload]
    else:
        raise SystemExit("Probe file must contain one probe object or a probe bank")
    if not candidates:
        raise SystemExit("Probe file contains no probes")
    return candidates


def add_probe(args):
    if args.file:
        candidates = _probe_from_file(args.file)
    else:
        probe = {"id": _prompt_value("id: "),
                 "family": _prompt_value("family (familiar-pattern/social/position/sycophancy/control): "),
                 "expect": _prompt_value("expect (same/different/trap): "),
                 "labels": _split_labels(_prompt_value("labels (comma-separated): ")),
                 "why": _prompt_value("why: ")}
        parse = _prompt_value("parse (blank, integer, int_1_10, amount, letter, choice_number): ", False)
        if parse:
            probe["parse"] = parse
        probe["variants"] = []
        while True:
            variant_id = _prompt_value("variant id (blank to finish): ", False)
            if not variant_id:
                break
            variant = {"id": variant_id, "prompt": _prompt_value("prompt: ")}
            answer = _prompt_value("answer (optional): ", False)
            if answer:
                variant["answer"] = answer
            if parse in {"letter", "choice_number"}:
                mapping = _prompt_value("map as TOKEN=LABEL pairs (comma-separated): ")
                variant["map"] = {key.strip(): value.strip() for key, value in
                                   (pair.split("=", 1) for pair in mapping.split(",") if "=" in pair)}
            probe["variants"].append(variant)
        candidates = [probe]
    custom_path = Path("probes/custom.json")
    bank = json.loads(custom_path.read_text(encoding="utf-8")) if custom_path.exists() else {"format": "probe-bank/v1", "probes": []}
    existing = {probe.get("id") for probe in bank.get("probes", [])}
    for candidate in candidates:
        validate_probe(candidate, "add")
        if candidate["id"] in existing:
            raise SystemExit("Probe id already exists in custom bank: %s" % candidate["id"])
        bank.setdefault("probes", []).append(candidate)
        existing.add(candidate["id"])
        print("Added probe %s" % candidate["id"])
    custom_path.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for candidate in candidates:
        print("Run it with: python3 -m probe all --probes %s" % candidate["id"])


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
                result = client.complete(judge["id"], "Reply with OK.", judge.get("temperature", 0), judge.get("max_tokens", 80), judge.get("provider"), config["seed"], 0)
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
    if args.command == "add":
        add_probe(args)
        return
    if args.command == "serve":
        server = create_server(config, probes, port=args.port, mock=args.mock)
        print("Bias Probe dashboard: http://%s:%d/" % (server.server_address[0], server.server_address[1]))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return
    if args.command == "report":
        data_dir = _data_dir(args)
        path = build_report(config, probes, data_dir, "reports", args.mock)
        print("Report written to %s" % path)
        return
    if args.command == "all":
        data_dir = _data_dir(args)
        if not args.mock and not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY is required (use --mock for offline testing)")
        client = MockClient(probes, config["seed"]) if args.mock else OpenRouterClient(os.environ["OPENROUTER_API_KEY"])
        run(config, probes, client, data_dir, model_ids, probe_ids, args.dry_run)
        if not args.dry_run:
            score_models(args, config, probes, client)
            path = build_report(config, probes, data_dir, "reports", args.mock)
            print("Report written to %s" % path)
        return
    data_dir = Path(args.data_dir or ("data_mock" if args.mock else "data"))
    if not args.mock and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required (use --mock for offline testing)")
    client = MockClient(probes, config["seed"]) if args.mock else OpenRouterClient(os.environ["OPENROUTER_API_KEY"])
    run(config, probes, client, data_dir, model_ids, probe_ids, args.dry_run)
