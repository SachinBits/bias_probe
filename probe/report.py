"""Static, offline HTML reporting for probe runs."""
from __future__ import annotations

import csv
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from .config import model_slug
from .stats import (aggregate_model_stats, compute_probe_stats, fraction,
                    load_analysis_records, minimum_detectable_effect, _response_matches_config)


LIMITS_SENTENCE = ("verdicts are probe-level signals for this model, snapshot, "
                   "temperature and wording. Never claim universal bias.")


def _e(value):
    return html.escape("" if value is None else str(value), quote=True)


def _pct(value):
    return "n/a" if value is None else "%.1f%%" % (100.0 * value)


def _format_schedule(schedule):
    parts = []
    for item in schedule:
        temperature = item.get("temperature")
        temp_text = "%g" % temperature if isinstance(temperature, (float, int)) else str(temperature)
        parts.append("%s × T=%s" % (item.get("repeats", 0), temp_text))
    return ", ".join(parts) or "none"


def _format_label_counts(counts):
    if not counts:
        return "none"
    return ", ".join("%s ×%d" % (_e(label), count)
                      for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _share_bars(counts, total):
    if not total:
        return "<span class='muted'>0/0 (n/a)</span>"
    pieces = []
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        pieces.append("<span class='bar'><b style='width:%s%%'></b><span>%s %s</span></span>" %
                      (100.0 * count / total, _e(label), _e(fraction(count, total))))
    return "".join(pieces)


def _read_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            row["_file"], row["_line"] = str(path), line_number
            rows.append(row)
    return rows


def _response_inventory(config, data_dir, report_dir):
    inventory = {}
    all_rows = []
    for model in config.get("models", []):
        path = Path(data_dir) / "responses" / (model_slug(model["id"]) + ".jsonl")
        rows = _read_rows(path)
        all_rows.extend(rows)
        current = [row for row in rows if row.get("status") == "ok" and _response_matches_config(row, model, config)]
        superseded = [row for row in rows if row.get("status") == "ok" and not _response_matches_config(row, model, config)]
        requested = Counter(row.get("model_requested") for row in current if row.get("model_requested"))
        returned = Counter(row.get("model_returned") for row in current if row.get("model_returned"))
        providers = Counter(row.get("provider_returned") for row in current if row.get("provider_returned"))
        superseded_providers = Counter(row.get("provider_returned") for row in superseded if row.get("provider_returned"))
        timestamps = [row.get("timestamp_utc") for row in current if row.get("timestamp_utc")]
        inventory[model["id"]] = {"rows": rows, "current": current, "requested": requested, "returned": returned,
                                   "providers": providers, "superseded_providers": superseded_providers,
                                   "start": min(timestamps) if timestamps else None,
                                   "end": max(timestamps) if timestamps else None,
                                   "relative": os.path.relpath(path, report_dir)}
    return inventory, all_rows


def _model_header(model, info, records, config, summary):
    def counts(counter):
        if not counter:
            return "none"
        return "; ".join("%s (%d)" % (_e(key), value) for key, value in counter.items())
    current = [row for row in records if row.get("model_requested") == model["id"]]
    cost = sum(float(row.get("cost_usd") or 0) for row in current)
    ok = len(info["current"])
    recovered = summary.get("recovered_by_model", {}).get(model["id"], 0)
    still_failed = summary.get("still_failed_by_model", {}).get(model["id"], 0)
    multiple = len(info["returned"]) > 1 or len(info["providers"]) > 1
    superseded_providers = counts(info.get("superseded_providers", Counter()))
    superseded_text = ("<br><code>superseded provider_returned</code>: %s" % superseded_providers
                       if superseded_providers != "none" else "")
    return ("<h2>%s</h2><p><code>model_requested</code>: %s<br>"
            "<code>model_returned</code>: %s<br><code>provider_returned</code>: %s%s<br>"
            "%s<br>Calls: %d ok; failed then recovered: %d; still failed: %d; current scored: %d; superseded: %d; cost: $%.6f<br>"
            "Time range: %s to %s<br>Schedule: %s</p>" %
            (_e(model.get("label", model["id"])), _e(counts(info["requested"])),
             _e(counts(info["returned"])), _e(counts(info["providers"])),
             " <strong class='warning'>MULTIPLE IDS/PROVIDERS SEEN</strong>" if multiple else "",
             superseded_text, ok, recovered, still_failed, len(current),
             summary.get("superseded_by_model", {}).get(model["id"], 0), cost,
             _e(info["start"] or "n/a"), _e(info["end"] or "n/a"), _e(_format_schedule(config.get("schedule", [])))))


CSS = """
:root { color-scheme: light dark; --bg:#fff; --fg:#17202a; --muted:#667085; --card:#f5f7fa; --line:#d0d5dd; --accent:#2457d6; }
@media (prefers-color-scheme: dark) { :root { --bg:#111827; --fg:#edf2f7; --muted:#aab4c3; --card:#1f2937; --line:#374151; --accent:#8eb4ff; } }
body { margin:0; padding:2rem; font:15px/1.5 system-ui,sans-serif; background:var(--bg); color:var(--fg); }
main { max-width:1400px; margin:auto; } a { color:var(--accent); } table { width:100%; border-collapse:collapse; margin:1rem 0 2rem; }
th,td { text-align:left; vertical-align:top; border-bottom:1px solid var(--line); padding:.55rem; } th { position:sticky; top:0; background:var(--card); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:1rem; margin:1rem 0; }
.mock { background:#b42318; color:#fff; padding:1rem; text-align:center; font-weight:800; font-size:1.2rem; }
.badge { display:inline-block; border-radius:999px; padding:.12rem .55rem; background:#dbeafe; color:#172554; font-weight:700; } .warning { color:#f04438; }
.muted { color:var(--muted); } .bar { display:block; position:relative; min-width:180px; margin:.15rem 0; background:var(--line); border-radius:4px; overflow:hidden; }
.bar b { display:block; height:1.25rem; background:var(--accent); opacity:.6; } .bar span { position:absolute; left:.3rem; top:0; font-size:.8rem; }
details { margin:.35rem 0; border:1px solid var(--line); padding:.45rem; border-radius:5px; } pre { white-space:pre-wrap; overflow-wrap:anywhere; }
code { overflow-wrap:anywhere; } .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:1rem; }
"""


def _raw_details(rows, label_lookup):
    chunks = []
    for row in rows:
        identity = (row.get("key"), row.get("probe_id"), row.get("variant_id"), row.get("repeat_index"),
                    (row.get("params") or {}).get("temperature"))
        label = label_lookup.get(identity, {})
        chunks.append("<details><summary>%s — source=%s, T=%s, repeat=%s, timestamp=%s, provider=%s; file=%s:%s</summary>"
                      "<pre>%s</pre></details>" %
                      (_e(label.get("label", "unscorable")), _e(label.get("source", "none")),
                       _e((row.get("params") or {}).get("temperature")), _e(row.get("repeat_index")),
                       _e(row.get("timestamp_utc")), _e(row.get("provider_returned")),
                       _e(row.get("_file")), _e(row.get("_line")), _e(row.get("response_text", ""))))
    return "".join(chunks)


def _verdict_text(statistic):
    verdict = statistic.get("verdict", "UNSCORABLE")
    if verdict.startswith("BIASED"):
        return "%s <small>(%s)</small>" % (verdict, "q&lt;0.05" if statistic.get("q_value", 1) < 0.05 else "p only")
    return _e(verdict)


def _metric_text(probe, statistic):
    if probe.get("expect") == "trap":
        original = fraction(statistic.get("original_correct", 0), statistic.get("original_total", 0))
        changed = fraction(statistic.get("changed_correct", 0), statistic.get("changed_total", 0))
        drop = statistic.get("accuracy_drop")
        return "accuracy drop = %s − %s = %s percentage points" % (original, changed,
                                                                    "n/a" if drop is None else "%.1f" % (100.0 * drop))
    if probe.get("parse") == "int_1_10":
        return "mean gap = %.2f points (max−min of variant means)" % statistic["movement"]["movement"]
    return "movement = %.3f (between %.3f − within %.3f)" % (statistic["movement"]["movement"],
                                                               statistic["movement"]["between"],
                                                               statistic["movement"]["within"])


def _mde_sentence(config):
    repeats = next((item.get("repeats") for item in config.get("schedule", [])
                    if item.get("temperature") == 0.7), 10)
    if repeats == 10:
        return "With 10 repeats per variant, 10/10 vs 5/10 is detectable at p &lt; 0.05, while 10/10 vs 7/10 is not."
    return "With %d repeats per variant, the minimum detectable effect is shown by the exact Fisher calculation." % repeats


def _probe_card(probe, statistic, rows, report_dir):
    rows = [row for row in rows if row.get("probe_id") == probe["id"]]
    identity = lambda row: (row.get("key"), row.get("probe_id"), row.get("variant_id"), row.get("repeat_index"),
                            (row.get("params") or {}).get("temperature"))
    labels = {identity(row): (row.get("label_record") or {}) for row in rows}
    variants = []
    for item, variant in zip(statistic["variants"], probe["variants"]):
        t0_labels = Counter((row.get("label_record") or {}).get("label", "unscorable") for row in rows
                            if row.get("variant_id") == variant["id"] and
                            (row.get("params") or {}).get("temperature") == 0.0)
        variants.append("<td><b>%s</b><br>expected: %s<br>%s<br>modal: %s (%s)<br>accuracy: %s<br>refused=%s, hedge=%s, unscorable=%s<br>T=0: %s</td>" %
                        (_e(variant["id"]), _e(item.get("expected", "n/a")),
                         _share_bars(item["counts"], item["total"]), _e(item.get("modal", "n/a")),
                         _e(fraction(item.get("modal_count", 0), item.get("total", 0))),
                         _e(fraction(item["correct"], item["correct"] + item["wrong"]) if item.get("accuracy") is not None else "n/a"),
                         item.get("refused", 0), item.get("hedge", 0), item.get("unscorable", 0),
                         _e(_format_label_counts(t0_labels))))
    raw = _raw_details(rows, labels)
    stereo_text = ""
    if statistic.get("stereotype_rates"):
        stereo_text = "; stereotype rate: " + ", ".join(
            "%s=%s" % (key, fraction(value.get("count", 0), value.get("total", 0)))
            for key, value in sorted(statistic["stereotype_rates"].items()))
    pattern_text = ""
    if statistic.get("pattern_total") is not None:
        pattern_text = "; pattern capture=%s" % fraction(statistic.get("pattern_count", 0), statistic.get("pattern_total", 0))
    return ("<article class='card'><h3>%s <span class='badge'>%s</span></h3><p>%s</p>"
            "<p>Expectation: <code>%s</code>; %s; p=%.4g; q=%.4g"
            "%s%s</p><table><tr>%s</tr></table><h4>Raw responses</h4>%s</article>" %
            (_e(probe["id"]), _verdict_text(statistic), _e(probe.get("why", "")), _e(probe.get("expect")),
             _metric_text(probe, statistic),
             statistic["p_value"], statistic["q_value"],
             ("; Fisher p=%.4g%s" % (statistic["fisher_p"], pattern_text)
              if statistic.get("fisher_p") is not None else ""), stereo_text, "".join(variants), raw))


def _page_shell(title, body, mock=False):
    banner = "<div class='mock'>MOCK DATA — offline synthetic responses</div>" if mock else ""
    head = "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
    return (head + "<title>%s</title><style>%s</style></head><body>%s<main>%s</main></body></html>" %
            (_e(title), CSS, banner, body))


def build_report(config, probes, data_dir, output_dir="reports", mock=False):
    """Build ``reports/index.html`` and one offline page per configured model."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, summary = load_analysis_records(config, probes, data_dir)
    inventory, all_rows = _response_inventory(config, data_dir, output_dir)
    mock = bool(mock or data_dir.name == "data_mock" or any(row.get("mock") for row in all_rows))
    model_stats = {}
    for model in config.get("models", []):
        current = [row for row in records if row.get("model_requested") == model["id"]]
        model_stats[model["id"]] = aggregate_model_stats(current, probes, config, 0.7)

    index_parts = ["<h1>Bias Probe report</h1><p>%s</p>" % _e(LIMITS_SENTENCE)]
    if mock:
        index_parts.append("<div class='mock'>MOCK DATA — offline synthetic responses</div>")
    index_parts.append("<h2>Run header</h2><p>Schedule: %s<br>Minimum detectable effect: %s</p>" %
                       (_e(_format_schedule(config.get("schedule", []))), _mde_sentence(config)))
    for model in config.get("models", []):
        index_parts.append(_model_header(model, inventory[model["id"]], records, config, summary))
        index_parts.append("<p><a href='%s'>Open per-model page</a></p>" % _e(model_slug(model["id"]) + ".html"))

    index_parts.append("<h2>Headline: surgeon original vs inverted</h2><table><tr><th>Model</th><th>Original</th><th>Inverted</th></tr>")
    for model in config.get("models", []):
        current = [row for row in records if row.get("model_requested") == model["id"]]
        probe = next((p for p in probes if p["id"] == "surgeon-riddle"), None)
        if not probe:
            continue
        stat = compute_probe_stats(current, probe, .7, config.get("permutations", 5000), config.get("seed", 12345))
        t0 = compute_probe_stats(current, probe, 0.0, config.get("permutations", 5000), config.get("seed", 12345))
        cells = []
        for variant, item in zip(probe["variants"][:2], stat["variants"][:2]):
            zero = next(x for x in t0["variants"] if x["variant_id"] == variant["id"])
            cells.append("<td>%s<br>modal: %s<br>T=0: %s</td>" %
                         (_share_bars(item["counts"], item["total"]), _e(item.get("modal")),
                          _e(_format_label_counts(zero["counts"]))))
        index_parts.append("<tr><td>%s</td>%s</tr>" % (_e(model.get("label", model["id"])), "".join(cells)))
    index_parts.append("</table>")

    index_parts.append("<h2>Family summary</h2><table><tr><th>Model</th><th>Family</th><th>Verdicts</th>"
                       "<th>Mean categorical movement (n)</th><th>Mean score gap (n)</th>"
                       "<th>Mean accuracy drop (n)</th></tr>")
    for model in config.get("models", []):
        grouped = defaultdict(list)
        for item in model_stats[model["id"]]:
            grouped[item["family"]].append(item)
        for family, items in sorted(grouped.items()):
            categorical = [x["movement"]["movement"] for x in items
                           if x.get("expect") == "same" and x.get("parse") != "int_1_10"]
            score_gaps = [x["movement"]["movement"] for x in items
                          if x.get("expect") == "same" and x.get("parse") == "int_1_10"]
            accuracy_drops = [x["accuracy_drop"] for x in items if x.get("expect") == "trap" and
                              x.get("accuracy_drop") is not None]
            pattern_count = sum(x.get("pattern_count", 0) or 0 for x in items)
            pattern_total = sum(x.get("pattern_total", 0) or 0 for x in items)
            categorical_text = "%.3f (n=%d)" % (sum(categorical) / len(categorical), len(categorical)) if categorical else "n/a"
            score_gap_text = "%.2f (n=%d)" % (sum(score_gaps) / len(score_gaps), len(score_gaps)) if score_gaps else "n/a"
            accuracy_drop_text = ("%.1f percentage points (n=%d)" %
                                  (100.0 * sum(accuracy_drops) / len(accuracy_drops), len(accuracy_drops))
                                  if accuracy_drops else "n/a")
            index_parts.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" %
                               (_e(model.get("label", model["id"])), _e(family), _e(dict(Counter(x["verdict"] for x in items))),
                                _e(categorical_text), _e(score_gap_text), _e(accuracy_drop_text)))
    index_parts.append("</table><h2>Model comparison</h2><table><tr><th>Probe</th>" +
                       "".join("<th>%s</th>" % _e(model.get("label", model["id"])) for model in config.get("models", [])) + "</tr>")
    by_model_probe = {model["id"]: {x["probe_id"]: x for x in model_stats[model["id"]]} for model in config.get("models", [])}
    for probe in probes:
        index_parts.append("<tr><td>%s</td>%s</tr>" %
                           (_e(probe["id"]), "".join("<td>%s; %s</td>" %
                           (_verdict_text(by_model_probe[model["id"]][probe["id"]]),
                            _e(_metric_text(probe, by_model_probe[model["id"]][probe["id"]])))
                           for model in config.get("models", []))))
    index_parts.append("</table>")

    source_counts = Counter()
    for row in records:
        label = row.get("label_record") or {}
        source = label.get("source", "none")
        source_counts["unscorable" if label.get("label") in (None, "unscorable") else source] += 1
    hand_path = data_dir / "hand_labels.csv"
    human_count = 0
    if hand_path.exists():
        with hand_path.open(encoding="utf-8") as handle:
            human_count = sum(bool(row.get("human_label", "").strip()) for row in csv.DictReader(handle))
    index_parts.append("<h2>Scorer quality</h2><p>Sources: %s<br>Human-labelled: %s<br>Judge validation: %s</p>" %
                       (_e(json.dumps(dict(source_counts), sort_keys=True)), human_count,
                        "fixture available; no saved judge-test result (real judge not validated offline)"))
    index_parts.append("<h2>Methods and limits</h2><p>Metrics use current configuration-matching successful records only. Raw records remain in JSONL and are linked from each model page. See <a href='../SCORING_POLICY.md'>SCORING_POLICY.md</a>.</p>")
    (output_dir / "index.html").write_text(_page_shell("Bias Probe report", "".join(index_parts), mock), encoding="utf-8")

    for model in config.get("models", []):
        current = [row for row in records if row.get("model_requested") == model["id"]]
        stats = {item["probe_id"]: item for item in model_stats[model["id"]]}
        body = [_model_header(model, inventory[model["id"]], records, config, summary),
                "<p><a href='index.html'>Back to report index</a></p>"]
        for probe in probes:
            body.append(_probe_card(probe, stats[probe["id"]], current, output_dir))
        path = output_dir / (model_slug(model["id"]) + ".html")
        path.write_text(_page_shell(model.get("label", model["id"]), "".join(body), mock), encoding="utf-8")
    return output_dir / "index.html"
