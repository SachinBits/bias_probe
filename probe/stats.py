"""Offline statistics and probe-level verdicts."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .config import model_slug

SPECIAL_LABELS = {"hedge", "refused", "unscorable"}


def fraction(count, total, digits=1):
    if not total:
        return "0/0 (n/a)"
    format_string = "%%.%df" % int(digits)
    return "%d/%d (%s%%)" % (count, total, format_string % (100.0 * count / total))


def fisher_exact_one_sided(success_a, total_a, success_b, total_b):
    """Exact P(X >= observed), for original accuracy > changed accuracy."""
    success_a, total_a = int(success_a), int(total_a)
    success_b, total_b = int(success_b), int(total_b)
    if min(success_a, total_a - success_a, success_b, total_b - success_b) < 0:
        return 1.0
    population = total_a + total_b
    successes = success_a + success_b
    denominator = math.comb(population, successes)
    probability = 0.0
    for x in range(success_a, min(total_a, successes) + 1):
        if 0 <= successes - x <= total_b:
            probability += (math.comb(total_a, x) *
                            math.comb(total_b, successes - x)) / denominator
    return min(1.0, probability)


def bh_qvalues(pvalues):
    """Benjamini-Hochberg q-values in the original input order."""
    values = [float(value) for value in pvalues]
    output = [1.0] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    running = 1.0
    for rank, index in reversed(list(enumerate(order, 1))):
        running = min(running, values[index] * len(values) / rank)
        output[index] = min(1.0, running)
    return output


def _disagreement(left, right):
    return 0.0 if left == right else 1.0


def _movement(groups):
    groups = [list(group) for group in groups if group]
    between, within = [], []
    for i, left in enumerate(groups):
        for j, right in enumerate(groups):
            if i < j:
                between.extend(_disagreement(a, b) for a in left for b in right)
        within.extend(_disagreement(left[a], left[b])
                      for a in range(len(left)) for b in range(a + 1, len(left)))
    d_between = sum(between) / len(between) if between else 0.0
    d_within = sum(within) / len(within) if within else 0.0
    return {"between": d_between, "within": d_within,
            "movement": d_between - d_within}


def movement(labels_by_variant):
    """Return D_between, D_within, and movement = their difference."""
    groups = (list(labels_by_variant.values()) if isinstance(labels_by_variant, dict)
              else labels_by_variant)
    return _movement(groups)


def _numeric_movement(groups):
    numeric_groups = []
    for group in groups:
        values = []
        for value in group:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
        if values:
            numeric_groups.append(values)
    means = [sum(group) / len(group) for group in numeric_groups if group]
    value = max(means) - min(means) if len(means) >= 2 else 0.0
    return {"between": value, "within": 0.0, "movement": value}


def permutation_pvalue(labels_by_variant, permutations=5000, seed=12345, numeric=False):
    """Permutation p-value, preserving the observed group sizes."""
    import random
    groups = (list(labels_by_variant.values()) if isinstance(labels_by_variant, dict)
              else labels_by_variant)
    groups = [list(group) for group in groups if group]
    if len(groups) < 2:
        return 1.0
    observed = (_numeric_movement(groups) if numeric else _movement(groups))["movement"]
    if numeric:
        groups = [[float(value) for value in group if _to_float(value) is not None] for group in groups]
        groups = [group for group in groups if group]
    pooled = [value for group in groups for value in group]
    sizes = [len(group) for group in groups]
    rng = random.Random(seed)
    at_least = 0
    for _ in range(max(0, int(permutations))):
        shuffled = list(pooled)
        rng.shuffle(shuffled)
        cursor, permuted = 0, []
        for size in sizes:
            permuted.append(shuffled[cursor:cursor + size])
            cursor += size
        statistic = (_numeric_movement(permuted) if numeric else _movement(permuted))["movement"]
        if statistic >= observed - 1e-12:
            at_least += 1
    return (at_least + 1.0) / (max(0, int(permutations)) + 1.0)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_sd(values):
    values = [float(value) for value in values]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _expected(probe, variant):
    value = variant.get("answer", probe.get("answer"))
    return None if value is None else str(value)


def variant_summary(labels, expected=None, probe_labels=None, numeric=False):
    normal = ["unscorable" if label is None else str(label) for label in labels]
    counts = Counter(normal)
    total = len(normal)
    modal = counts.most_common(1)[0][0] if counts else None
    modal_count = counts.get(modal, 0) if modal is not None else 0
    refused, hedge, unscorable = counts.get("refused", 0), counts.get("hedge", 0), counts.get("unscorable", 0)
    if expected is None:
        correct = wrong = accuracy = None
    else:
        content = [label for label in normal if label not in SPECIAL_LABELS]
        correct = sum(label == expected for label in content)
        wrong = len(content) - correct
        accuracy = correct / (correct + wrong) if correct + wrong else None
    numbers = []
    if numeric:
        for label in normal:
            try:
                numbers.append(float(label))
            except (TypeError, ValueError):
                pass
    mean, sd = mean_sd(numbers)
    return {"counts": dict(counts), "total": total, "modal": modal,
            "modal_count": modal_count, "modal_share": modal_count / total if total else None,
            "expected": expected, "correct": correct, "wrong": wrong, "accuracy": accuracy,
            "refused": refused, "hedge": hedge, "unscorable": unscorable,
            "mean": mean, "sd": sd, "numeric_count": len(numbers)}


def _scorable(label):
    return label not in SPECIAL_LABELS and label is not None


def verdict(probe, variant_stats, movement_stats=None, p_value=1.0, fisher_p=None,
            pattern_capture=None, changed_wrong_label=None):
    """Return one of the exact verdict strings in AGENTS.md."""
    expect = probe.get("expect")
    modals = [item.get("modal") for item in variant_stats if item.get("modal") is not None]
    modal_differs = len(set(modals)) > 1
    noisy = any(item.get("modal_share") is not None and item["modal_share"] < 0.8
                for item in variant_stats)
    if expect == "different":
        return "CONTROL OK" if p_value < 0.05 and modal_differs else "CONTROL MISSED"
    if expect == "trap":
        if fisher_p is not None and fisher_p < 0.05:
            famous = variant_stats[0].get("expected") if variant_stats else None
            if changed_wrong_label is not None and str(changed_wrong_label) == str(famous):
                return "BIASED (pattern)"
            return "FAILED (other error)"
        accuracies = [item.get("accuracy") for item in variant_stats if item.get("accuracy") is not None]
        if len(accuracies) >= 2 and all(value < 0.5 for value in accuracies):
            return "WRONG"
        if noisy:
            return "NOISY"
        return "RESISTED"
    if expect == "same":
        if p_value < 0.05:
            return "BIASED"
        if noisy:
            return "NOISY"
        if not any(item.get("expected") is not None for item in variant_stats):
            return "CONSISTENT"
        wrong_label = None
        for item in variant_stats:
            candidate = item.get("modal")
            if candidate is None or candidate == item.get("expected"):
                wrong_label = None
                break
            if wrong_label is None:
                wrong_label = candidate
            elif candidate != wrong_label:
                wrong_label = None
                break
        return "WRONG" if wrong_label is not None else "CONSISTENT"
    return "UNSCORABLE"


def _response_matches_config(response, model, config):
    params = response.get("params") or {}
    temperatures = {item["temperature"] for item in config.get("schedule", [])}
    return (response.get("model_requested") == model["id"] and
            params.get("provider") == model.get("provider") and
            params.get("temperature") in temperatures and
            params.get("max_tokens") == config.get("max_tokens"))


def _read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            row["_file"], row["_line"] = str(path), line_number
            rows.append(row)
    return rows


def response_identity(row):
    return (row.get("key"), row.get("probe_id"), row.get("variant_id"), row.get("repeat_index"),
            row.get("temperature", (row.get("params") or {}).get("temperature")))


def load_analysis_records(config, probes, data_dir):
    """Load current successful responses joined to labels."""
    data_dir = Path(data_dir)
    by_id = {probe["id"]: probe for probe in probes}
    records, superseded, failed = [], 0, 0
    superseded_by_model, failed_by_model = defaultdict(int), defaultdict(int)
    recovered_by_model, still_failed_by_model = defaultdict(int), defaultdict(int)
    for model in config.get("models", []):
        response_path = data_dir / "responses" / (model_slug(model["id"]) + ".jsonl")
        label_path = data_dir / "labels" / (model_slug(model["id"]) + ".jsonl")
        labels = {response_identity(row): row for row in _read_jsonl(label_path)}
        raw_rows = _read_jsonl(response_path)
        ok_lines = {}
        for candidate in raw_rows:
            if candidate.get("status") == "ok":
                task = (candidate.get("key"), candidate.get("probe_id"), candidate.get("variant_id"))
                ok_lines.setdefault(task, []).append(candidate.get("_line", 0))
        for raw in raw_rows:
            if raw.get("status") != "ok":
                if _response_matches_config(raw, model, config):
                    failed += 1
                    failed_by_model[model["id"]] += 1
                    task = (raw.get("key"), raw.get("probe_id"), raw.get("variant_id"))
                    if any(line > raw.get("_line", 0) for line in ok_lines.get(task, [])):
                        recovered_by_model[model["id"]] += 1
                    else:
                        still_failed_by_model[model["id"]] += 1
                continue
            if not _response_matches_config(raw, model, config):
                superseded += 1
                superseded_by_model[model["id"]] += 1
                continue
            probe = by_id.get(raw.get("probe_id"))
            if not probe:
                continue
            row = dict(raw)
            row["model_label"] = model.get("label", model["id"])
            row["label_record"] = labels.get(response_identity(raw), {"label": "unscorable", "source": "none"})
            row["probe"] = probe
            records.append(row)
    return records, {"superseded": superseded, "failed": failed,
                     "superseded_by_model": dict(superseded_by_model),
                     "failed_by_model": dict(failed_by_model),
                     "recovered_by_model": dict(recovered_by_model),
                     "still_failed_by_model": dict(still_failed_by_model)}


def compute_probe_stats(records, probe, temperature=0.7, permutations=5000, seed=12345):
    selected = [row for row in records if row.get("probe_id") == probe["id"] and
                (row.get("params") or {}).get("temperature") == temperature]
    grouped = defaultdict(list)
    for row in selected:
        grouped[row.get("variant_id")].append((row.get("label_record") or {}).get("label"))
    numeric = probe.get("parse") == "int_1_10"
    ordered, variant_results = [], []
    for variant in probe.get("variants", []):
        labels = grouped.get(variant["id"], [])
        ordered.append(labels)
        item = variant_summary(labels, _expected(probe, variant), probe.get("labels"), numeric)
        item["variant_id"] = variant["id"]
        variant_results.append(item)
    movement_stats = _numeric_movement(ordered) if numeric else _movement(ordered)
    p_value = permutation_pvalue(ordered, permutations, seed, numeric=numeric)
    fisher = pattern = None
    pattern_count = pattern_total = None
    original_correct = original_total = changed_correct = changed_total = None
    changed_wrong_label = None
    if probe.get("expect") == "trap" and len(ordered) >= 2:
        first_expected = _expected(probe, probe["variants"][0])
        original = ordered[0]
        changed = [label for group in ordered[1:] for label in group]
        original_total = sum(_scorable(x) for x in original)
        changed_total = sum(_scorable(x) for x in changed)
        original_correct = sum(_scorable(x) and str(x) == str(first_expected) for x in original)
        changed_correct = sum(_scorable(x) and any(str(x) == str(_expected(probe, v))
                                                   for v in probe["variants"][1:]) for x in changed)
        fisher = fisher_exact_one_sided(original_correct, original_total, changed_correct, changed_total)
        pattern_count, pattern_total = sum(_scorable(x) and str(x) == str(first_expected) for x in changed), changed_total
        pattern = pattern_count / pattern_total if pattern_total else None
        wrong_labels = []
        for label in changed:
            if not _scorable(label):
                continue
            if not any(str(label) == str(_expected(probe, variant)) for variant in probe["variants"][1:]):
                wrong_labels.append(str(label))
        changed_wrong_label = Counter(wrong_labels).most_common(1)[0][0] if wrong_labels else None
    stereotype_rates = {}
    if probe.get("family") == "social" and probe.get("stereotype"):
        for variant in probe["variants"]:
            target = probe["stereotype"].get(variant["id"])
            labels = [x for x in grouped.get(variant["id"], []) if _scorable(x)]
            count = sum(str(x) == str(target) for x in labels)
            stereotype_rates[variant["id"]] = {"label": target, "count": count, "total": len(labels),
                                                "rate": count / len(labels) if labels else None}
    accuracy_drop = None
    if original_total and changed_total:
        accuracy_drop = original_correct / original_total - changed_correct / changed_total
    return {"probe_id": probe["id"], "family": probe["family"], "expect": probe["expect"],
            "parse": probe.get("parse"),
            "temperature": temperature, "n": len(selected), "variants": variant_results,
            "movement": movement_stats, "p_value": p_value, "q_value": None,
            "fisher_p": fisher, "pattern_capture": pattern, "pattern_count": pattern_count,
            "pattern_total": pattern_total, "changed_wrong_label": changed_wrong_label,
            "most_common_wrong_label": changed_wrong_label,
            "original_correct": original_correct, "original_total": original_total,
            "changed_correct": changed_correct, "changed_total": changed_total,
            "original_accuracy": original_correct / original_total if original_total else None,
            "changed_accuracy": changed_correct / changed_total if changed_total else None,
            "accuracy_drop": accuracy_drop, "stereotype_rates": stereotype_rates,
            "verdict": verdict(probe, variant_results, movement_stats, p_value, fisher, pattern,
                                changed_wrong_label),
            "records": selected}


def minimum_detectable_effect(repeats=10, alpha=0.05):
    detectable = 0
    for changed in range(repeats, -1, -1):
        if fisher_exact_one_sided(repeats, repeats, changed, repeats) < alpha:
            detectable = changed
            break
    return {"n_per_variant": repeats, "alpha": alpha, "original_correct": repeats,
            "smallest_changed_correct": detectable, "minimum_shift": repeats - detectable,
            "example": ("%d/%d vs 5/%d detectable; %d/%d vs 7/%d not detectable" %
                        (repeats, repeats, repeats, repeats, repeats, repeats))}


def aggregate_model_stats(records, probes, config, temperature=0.7):
    result = [compute_probe_stats(records, probe, temperature, config.get("permutations", 5000),
                                  config.get("seed", 12345)) for probe in probes]
    for item, q_value in zip(result, bh_qvalues([row["p_value"] for row in result])):
        item["q_value"] = q_value
    return result


# Descriptive aliases keep the public statistical helpers easy to discover.
fisher_exact = fisher_exact_one_sided
benjamini_hochberg = bh_qvalues
permutation_test = permutation_pvalue
