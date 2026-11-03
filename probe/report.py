"""Static, offline HTML reporting for probe runs."""
from __future__ import annotations

import csv
import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from .config import model_slug
from .stats import (aggregate_model_stats, compute_probe_stats, fraction,
                    load_analysis_records, minimum_detectable_effect, _response_matches_config)


LIMITS_SENTENCE = ("Verdicts are probe-level signals for this model, snapshot, "
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


def _table(content, class_name=""):
    class_attr = " class='%s'" % class_name if class_name else ""
    return "<div class='table-wrap'><table%s>%s</table></div>" % (class_attr, content)


def _share_bars(counts, total, expected=None, pattern_label=None, headline=False):
    if not total:
        return "<span class='muted'>0/0 (n/a)</span>"
    modal = max(counts.items(), key=lambda item: (item[1], str(item[0])))[0] if counts else None
    pieces = []
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if headline and expected is not None:
            state = "correct" if str(modal) == str(expected) else "wrong"
        elif expected is not None and str(label) == str(expected):
            state = "correct"
        elif pattern_label is not None and str(label) == str(pattern_label):
            state = "pattern"
        else:
            state = "neutral"
        pieces.append("<span class='bar %s'><b style='width:%s%%'></b><span>%s %s</span></span>" %
                      (state, 100.0 * count / total, _e(label), _e(fraction(count, total))))
    return "".join(pieces)


def _verdict_class(verdict):
    if verdict == "BIASED (pattern)":
        return "verdict-pattern"
    if verdict in {"BIASED", "CONTROL MISSED"}:
        return "verdict-bad"
    if verdict == "FAILED (other error)":
        return "verdict-failed"
    if verdict == "NOISY":
        return "verdict-noisy"
    if verdict == "WRONG":
        return "verdict-wrong"
    if verdict in {"CONSISTENT", "RESISTED", "CONTROL OK"}:
        return "verdict-good"
    return "verdict-neutral"


def _verdict_badge(verdict, q_value=None):
    verdict = verdict or "UNSCORABLE"
    marker = ""
    if str(verdict).startswith("BIASED"):
        marker = " <small class='q-marker'>%s</small>" % ("q&lt;0.05" if q_value is not None and q_value < 0.05 else "p only")
    return "<span class='verdict-badge %s'>%s</span>%s" % (_verdict_class(verdict), _e(verdict), marker)


def _verdict_chips(items):
    counts = Counter(item.get("verdict", "UNSCORABLE") for item in items)
    chips = []
    for verdict, count in sorted(counts.items()):
        q_values = [item.get("q_value") for item in items
                    if item.get("verdict", "UNSCORABLE") == verdict and item.get("q_value") is not None]
        chips.append("<span class='verdict-chip'>%s <b>×%d</b></span>" %
                     (_verdict_badge(verdict, min(q_values) if q_values else None), count))
    return " ".join(chips) or "<span class='muted'>none</span>"


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
            "Time range: %s to %s</p>" %
            (_e(model.get("label", model["id"])), _e(counts(info["requested"])),
             _e(counts(info["returned"])), _e(counts(info["providers"])),
             " <strong class='warning'>MULTIPLE IDS/PROVIDERS SEEN</strong>" if multiple else "",
             superseded_text, ok, recovered, still_failed, len(current),
             summary.get("superseded_by_model", {}).get(model["id"], 0), cost,
             _e(info["start"] or "n/a"), _e(info["end"] or "n/a")))


CSS = """
:root { color-scheme:light; --bg:#fff; --fg:#1f2133; --muted:#6b6e85; --faint:#9294ab; --card:#fff; --line:#e4e4e4; --head:#f5f5f7; --hover:#f8f9fc; --accent:#464edf; --accent-dark:#3a41c4; --accent-soft:#eeeffd; --periwinkle:#b6b9f5; --nav-bg:#1b1d36; --nav-active:#262847; --nav-fg:#fff; --nav-muted:#a3a5c4; --nav-faint:#7d80a7; --good:#16792a; --good-bg:#e5f6e8; --bad:#c81e1b; --bad-bg:#fdecec; }
* { box-sizing:border-box; }
html,body { background:var(--bg); min-height:100%; } html { -webkit-text-size-adjust:100%; }
body { margin:0; font:14px/1.55 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; background:var(--bg); color:var(--fg); -webkit-font-smoothing:antialiased; font-variant-numeric:tabular-nums; }
h1,h2,h3,h4 { color:var(--fg); line-height:1.3; letter-spacing:-.01em; margin:0; }
h1 { font-size:22px; font-weight:700; } h2 { font-size:17px; font-weight:650; margin:36px 0 14px; } h3 { font-size:14px; font-weight:650; margin:24px 0 10px; } h4 { font-size:13px; font-weight:650; margin:16px 0 8px; }
p { margin:0 0 8px; } ul { margin:8px 0; padding-left:18px; } li { margin:4px 0; } small { font-size:12px; }
a { color:var(--accent); text-decoration:none; } a:hover { color:var(--accent-dark); text-decoration:underline; }
code,pre { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12.5px; } code { overflow-wrap:anywhere; } pre { margin:0; white-space:pre-wrap; overflow-wrap:anywhere; }
.muted,.subtitle { color:var(--muted); } .warning { color:var(--bad); }
.app { display:grid; grid-template-columns:248px minmax(0,1fr); min-height:100vh; }
.sidebar { position:sticky; top:0; align-self:start; height:100vh; overflow-y:auto; display:flex; flex-direction:column; background:var(--nav-bg); color:var(--nav-muted); padding:22px 14px; }
.brand { display:block; color:var(--nav-fg); font-size:18px; font-weight:700; letter-spacing:-.02em; padding:0 10px 22px; } .brand:hover { color:var(--nav-fg); text-decoration:none; }
.nav-label { color:var(--nav-faint); font-size:11px; font-weight:650; letter-spacing:.08em; text-transform:uppercase; padding:14px 10px 6px; }
.nav-link { display:block; position:relative; color:var(--nav-muted); padding:8px 10px; margin:1px 0; border-radius:7px; } .nav-link:hover { color:var(--nav-fg); background:var(--nav-active); text-decoration:none; }
.nav-link.active { color:var(--nav-fg); background:var(--nav-active); font-weight:600; } .nav-link.active::before { content:""; position:absolute; left:-8px; top:7px; bottom:7px; width:3px; border-radius:2px; background:var(--accent); }
.nav-models { margin-top:auto; padding-top:22px; } .nav-model { padding:4px 10px; color:var(--nav-fg); font-size:12.5px; } .nav-model small { display:block; color:var(--nav-faint); font-size:11px; overflow-wrap:anywhere; }
.content { min-width:0; }
main { min-width:0; padding:0 32px 56px; } .content main { max-width:1240px; } body > main { max-width:1200px; margin:auto; padding-top:24px; }
.page-head { margin:0 -32px 24px; padding:20px 32px 18px; border-bottom:1px solid var(--line); } .page-head .subtitle { margin:4px 0 0; } .crumb { display:inline-block; font-size:12.5px; margin-bottom:6px; } .tags { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
.tag { display:inline-block; border-radius:5px; padding:2px 8px; font-size:12px; font-weight:500; color:var(--muted); background:var(--head); }
.mock { background:var(--bad); color:#fff; padding:12px; text-align:center; font-weight:700; font-size:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin:14px 0; } .card > h3:first-child { margin-top:0; }
.kpis { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:0 0 8px; } .kpi { border:1px solid var(--line); border-radius:10px; padding:14px 16px; min-width:0; } .kpi small { display:block; color:var(--muted); font-size:11px; font-weight:650; letter-spacing:.06em; text-transform:uppercase; margin-bottom:5px; } .kpi strong { display:block; font-size:15px; font-weight:650; } .kpi-note { display:block; color:var(--muted); font-size:12px; margin-top:3px; }
.key-finding { background:var(--accent-soft); border-radius:10px; padding:16px 20px; } .headline-sentence { font-size:15px; font-weight:600; margin-bottom:4px; } .interpretation { color:var(--muted); margin:0; }
.table-wrap { max-width:100%; overflow-x:auto; -webkit-overflow-scrolling:touch; border:1px solid var(--line); border-radius:10px; margin:12px 0; } .card .table-wrap { border:0; border-radius:0; margin:0 -4px; }
table { width:100%; border-collapse:collapse; } caption { text-align:left; color:var(--muted); font-size:12.5px; padding:10px 14px 2px; }
th,td { text-align:left; vertical-align:top; border-bottom:1px solid var(--line); padding:11px 14px; } tr:last-child > td { border-bottom:0; }
th { background:var(--head); font-size:12.5px; font-weight:600; color:var(--fg); white-space:nowrap; } th br,td > br { display:none; } th small { display:block; font-weight:400; color:var(--muted); }
.card th { background:transparent; color:var(--muted); padding-top:4px; } .card td,.card th { padding-left:6px; padding-right:12px; }
.metric,.probe-why { display:block; color:var(--muted); font-size:12.5px; margin-top:3px; } .surgeon-grid .bar { min-width:220px; }
.verdict-badge { display:inline-block; border-radius:5px; padding:2px 7px; font-size:11.5px; font-weight:650; letter-spacing:.01em; white-space:nowrap; }
.verdict-chip { display:inline-block; margin:2px 8px 2px 0; font-size:12px; white-space:nowrap; } .verdict-chip b { font-weight:600; color:var(--muted); }
.verdict-bad,.verdict-pattern { background:var(--bad-bg); color:var(--bad); } .verdict-failed { background:#fff1e6; color:#b4530f; } .verdict-noisy { background:#fdf6d8; color:#856400; } .verdict-wrong { background:#eeeef3; color:#4a4c63; } .verdict-good { background:var(--good-bg); color:var(--good); } .verdict-neutral { background:var(--head); color:var(--muted); }
.outcome-bias { background:var(--bad-bg); color:var(--bad); } .outcome-error { background:#fff1e6; color:#b4530f; } .outcome-noisy { background:#fdf6d8; color:#856400; } .outcome-ok { background:var(--good-bg); color:var(--good); } .outcome-zero { background:var(--head); color:var(--faint); font-weight:500; } .outcome-detail { color:var(--muted); font-size:12px; } .outcome-counts { display:inline-flex; flex-wrap:wrap; gap:4px; }
.q-marker { color:var(--faint); font-size:11.5px; font-weight:500; margin-left:3px; }
.bar { display:block; position:relative; min-width:150px; height:26px; margin:3px 0; background:#f3f4f8; border-radius:5px; overflow:hidden; } .bar b { display:block; height:100%; } .bar.correct b { background:#c9ecd0; } .bar.wrong b,.bar.pattern b { background:#f9d0cf; } .bar.neutral b { background:#dcdef8; }
.bar span { position:absolute; left:9px; right:6px; top:0; line-height:26px; font-size:12.5px; color:var(--fg); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.finding-grid { border:1px solid var(--line); border-radius:10px; overflow:hidden; } .finding-card { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.25fr) 56px; column-gap:24px; align-items:start; padding:13px 16px; border-bottom:1px solid var(--line); } .finding-card:last-child { border-bottom:0; }
.finding-card h3 { grid-column:1; margin:0; font-size:13.5px; } .finding-card > p { margin:0; } .finding-card > p.muted { grid-column:1; font-size:12.5px; margin-top:2px; } .finding-card > p:nth-of-type(2) { grid-column:2; grid-row:1 / span 2; font-size:13px; } .finding-card > p:nth-of-type(3) { display:none; } .finding-link { grid-column:3; grid-row:1 / span 2; font-size:12.5px; font-weight:600; text-align:right; overflow-wrap:anywhere; }
.filter-chips { display:inline-flex; flex-wrap:wrap; gap:2px; padding:3px; margin:0 0 4px; background:var(--head); border-radius:8px; } .filter-chip { border:0; background:transparent; color:var(--muted); border-radius:6px; padding:5px 11px; font:inherit; font-size:12.5px; font-weight:550; cursor:pointer; } .filter-chip:hover { color:var(--fg); } .filter-chip.active { background:#fff; color:var(--fg); font-weight:650; box-shadow:0 1px 2px rgba(27,29,54,.14); }
.family-row td { background:#f6f6fb; color:var(--muted); font-size:11px; font-weight:650; letter-spacing:.07em; text-transform:uppercase; padding:7px 14px; }
.comparison-row { cursor:pointer; } .comparison-row:hover td { background:var(--hover); } .comparison-row td:first-child code { display:block; font:inherit; font-weight:600; color:var(--accent); } .badge-link:hover { text-decoration:none; } .legacy-link { display:none; } #families td:first-child { white-space:nowrap; font-weight:550; } .answer-moved { white-space:nowrap; } .family-note { color:var(--muted); font-size:12.5px; margin:8px 0 0; }
.legend { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:6px 24px; margin:12px 0 0; color:var(--muted); font-size:12.5px; } .legend div { display:flex; gap:8px; align-items:baseline; } .legend .verdict-badge { flex:none; }
details { min-width:0; } summary { cursor:pointer; }
main > details { border:1px solid var(--line); border-radius:10px; margin:12px 0; padding:0 18px; } main > details:first-of-type { margin-top:32px; } main > details > summary { font-size:14px; font-weight:650; padding:13px 0; } main > details[open] { padding-bottom:14px; } main > details > section > h2:first-child { display:none; } main > details h2 { font-size:14px; margin:20px 0 8px; } main > details h3 { margin-top:18px; }
.prompts h2 { margin-top:0; } .prompt-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; } .prompt { background:#f6f6fb; border-radius:10px; padding:13px 15px; min-width:0; } .prompt-head { display:flex; flex-wrap:wrap; justify-content:space-between; gap:4px 12px; margin-bottom:6px; font-size:12.5px; font-weight:650; } .prompt-head span { color:var(--muted); font-weight:500; } .prompt p { margin:0; white-space:pre-line; overflow-wrap:anywhere; }
.probe-model-grid { display:grid; grid-template-columns:minmax(0,1fr); gap:0; margin-top:8px; } .probe-model { min-width:0; } .probe-model-head { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:5px; } .probe-model-head h3 { margin:0; font-size:15px; } .exact-verdict { color:var(--muted); font-size:11.5px; font-weight:500; } .summary-line { color:var(--fg); font-size:13px; margin:0 0 8px; }
.statistics { margin:0 0 10px; color:var(--muted); font-size:12px; } .statistics summary { color:var(--accent); } .statistics p { margin:5px 0 0; }
.result-table td { vertical-align:middle; } .result-table .variant-name { font-weight:600; } .result-table .dist { width:46%; min-width:220px; } .result-table td:not(.dist) { white-space:nowrap; }
.raw-group { border-top:1px solid var(--line); } .raw-group:first-of-type { margin-top:8px; } .raw-group > summary { padding:9px 0; font-size:13px; font-weight:550; color:var(--accent); } .raw-group[open] { padding-bottom:8px; }
.raw { border:1px solid var(--line); border-radius:7px; margin:4px 0; } .raw > summary { display:flex; flex-wrap:wrap; gap:4px 12px; padding:7px 11px; font-size:12.5px; } .raw-label { font-weight:650; } .raw-meta { color:var(--muted); } .raw pre { padding:10px 11px; border-top:1px solid var(--line); background:#fafafc; } .raw-source { margin:0; padding:7px 11px; border-top:1px solid var(--line); color:var(--muted); font-size:11.5px; overflow-wrap:anywhere; }
.form-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px 16px; } label { display:block; font-size:12.5px; font-weight:600; color:var(--fg); }
label > input,label > select,label > textarea { margin-top:5px; } input,select,textarea { width:100%; font:inherit; font-size:14px; font-weight:400; color:var(--fg); background:#fff; border:1px solid #d9d9e0; border-radius:8px; padding:8px 11px; } input:focus,select:focus,textarea:focus { outline:2px solid var(--periwinkle); outline-offset:0; border-color:var(--accent); } input::placeholder,textarea::placeholder { color:var(--faint); } textarea { min-height:5rem; resize:vertical; }
.model-option { display:flex; align-items:center; gap:8px; font-weight:500; font-size:14px; } .model-option input { width:auto; margin:0; accent-color:var(--accent); }
.variant-form { display:grid; grid-template-columns:8rem minmax(0,2fr) minmax(0,1fr) minmax(0,1fr) auto; gap:8px; align-items:start; margin:8px 0; } .schedule-row { margin-bottom:10px; } #add-variant { margin-right:8px; }
button { font:inherit; cursor:pointer; } .button { border:1px solid #d9d9e0; background:#fff; color:var(--fg); border-radius:8px; padding:7px 13px; font-size:13px; font-weight:550; } .button:hover { background:var(--head); } .button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:650; padding:9px 22px; } .button.primary:hover { background:var(--accent-dark); } .button:disabled { opacity:.5; cursor:not-allowed; } .button.primary:disabled { background:var(--periwinkle); border-color:var(--periwinkle); opacity:1; }
#progress { color:var(--muted); font-size:12.5px; margin:16px 0 6px; } .progress-track { height:6px; background:var(--head); border-radius:999px; overflow:hidden; margin-bottom:16px; } .progress-fill { height:100%; width:0; background:var(--accent); transition:width .2s; }
.top-nav { margin:0 0 20px; font-size:13px; color:var(--faint); } .top-nav a { white-space:nowrap; } .disagreements { margin-top:6px; font-size:13px; } .api-error { border-left:3px solid var(--bad); }
@media (max-width:900px) {
  .app { display:block; } .sidebar { position:static; align-self:auto; height:auto; padding:12px 16px; display:block; } .brand { padding:0 0 8px; } .nav-label,.nav-models { display:none; } .nav-group { display:inline; } .nav-link { display:inline-block; padding:5px 9px; font-size:13px; } .nav-link.active::before { display:none; }
  .finding-card { grid-template-columns:minmax(0,1fr); row-gap:4px; } .finding-card h3,.finding-card > p.muted,.finding-card > p:nth-of-type(2),.finding-link { grid-column:1; grid-row:auto; } .finding-link { text-align:left; }
}
@media (max-width:700px) {
  main { padding:0 16px 40px; } .page-head { margin:0 -16px 18px; padding:16px; } .kpis,.form-grid,.variant-form { grid-template-columns:minmax(0,1fr); } .result-table .dist { min-width:180px; }
  .variant-table,.variant-table tbody,.variant-table tr,.variant-table td { display:block; width:100%; } .variant-table td { border:1px solid var(--line); border-radius:8px; margin:8px 0; }
}
"""


def _raw_details(rows, label_lookup):
    chunks = []
    for row in rows:
        identity = (row.get("key"), row.get("probe_id"), row.get("variant_id"), row.get("repeat_index"),
                    (row.get("params") or {}).get("temperature"))
        label = label_lookup.get(identity, {})
        chunks.append("<details class='raw'><summary><span class='raw-label'>%s</span>"
                      "<span class='raw-meta'>T=%s · repeat %s · scored by %s</span></summary>"
                      "<pre>%s</pre><p class='raw-source'>%s · provider %s · %s:%s</p></details>" %
                      (_e(label.get("label", "unscorable")), _e((row.get("params") or {}).get("temperature")),
                       _e(row.get("repeat_index")), _e(label.get("source", "none")), _e(row.get("response_text", "")),
                       _e(row.get("timestamp_utc")), _e(row.get("provider_returned")),
                       _e(row.get("_file")), _e(row.get("_line"))))
    return "".join(chunks)


def _verdict_text(statistic):
    return _verdict_badge(statistic.get("verdict", "UNSCORABLE"), statistic.get("q_value"))


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


def _p_text(probe, statistic):
    def sig3(value):
        return "n/a" if value is None else "%.3g" % float(value)
    if probe.get("expect") == "trap" and statistic.get("fisher_p") is not None:
        return "p(Fisher)=%s; q=%s" % (sig3(statistic.get("fisher_p")), sig3(statistic.get("q_value")))
    return "p=%s; q=%s" % (sig3(statistic.get("verdict_p")), sig3(statistic.get("q_value")))


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
        pattern_label = statistic["variants"][0].get("expected") if probe.get("expect") == "trap" and variant is not probe["variants"][0] else None
        variants.append("<td><b>%s</b><br>expected: %s<br>%s<br>modal: %s (%s)<br>accuracy: %s<br>refused=%s, hedge=%s, unscorable=%s<br>T=0: %s</td>" %
                        (_e(variant["id"]), _e(item.get("expected", "n/a")),
                         _share_bars(item["counts"], item["total"], item.get("expected"), pattern_label), _e(item.get("modal", "n/a")),
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
    control_note = ""
    if probe.get("family") == "familiar-pattern" and statistic.get("verdict") == "CONTROL MISSED":
        control_note = "<br>Control missed: the answer did not change although the prompt stated the answer — the model is not reading the changed fact."
    return ("<article id='%s' class='card'><h3>%s %s</h3><p>%s</p>"
            "<p>Expectation: <code>%s</code>; %s; %s%s%s%s</p>%s<h4>Raw responses</h4>%s</article>" %
            (_e(probe["id"]), _e(probe["id"]), _verdict_text(statistic), _e(probe.get("why", "")), _e(probe.get("expect")),
             _metric_text(probe, statistic),
             _p_text(probe, statistic), pattern_text, stereo_text, control_note,
             _table("<tr>%s</tr>" % "".join(variants), "variant-table"), raw))


def _top_nav(prefix=""):
    links = (("key-finding", "Key finding"), ("comparison", "Model comparison"),
             ("families", "Family summary"), ("scorer", "Scorer quality"),
             ("run-details", "Run details"), ("methods", "Methods and limits"))
    return "<nav class='top-nav' aria-label='Report sections'>%s</nav>" % " · ".join(
        "<a href='%s#%s'>%s</a>" % (prefix, anchor, label) for anchor, label in links)


def _page_shell(title, body, mock=False, nav_prefix=""):
    banner = "<div class='mock'>MOCK DATA — offline synthetic responses</div>" if mock else ""
    head = "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
    return (head + "<title>%s</title><style>%s</style></head><body>%s<main>%s%s</main></body></html>" %
            (_e(title), CSS, banner, _top_nav(nav_prefix), body))


def _surgeon_finding(model, statistic):
    if not statistic or len(statistic.get("variants", [])) < 2:
        return "%s has no current surgeon-riddle results in this snapshot." % _e(model.get("label", model["id"]))
    original, inverted = statistic["variants"][:2]
    original_label = original.get("modal") or "no modal label"
    inverted_label = inverted.get("modal") or "no modal label"
    original_count = fraction(original.get("modal_count", 0), original.get("total", 0))
    inverted_count = fraction(inverted.get("modal_count", 0), inverted.get("total", 0))
    return ("%s answered ‘%s’ in %s on the original prompt and ‘%s’ in %s when the prompt said the surgeon is the father."
            % (_e(model.get("label", model["id"])), _e(original_label), _e(original_count),
               _e(inverted_label), _e(inverted_count)))


def _short_model_name(model):
    if isinstance(model, str):
        model_id = model
        label = model
    else:
        model_id = model.get("id", "")
        label = model.get("label", model_id)
    if "gpt-4o-mini" in model_id:
        return "GPT-4o-mini"
    if "llama-3.1-8b" in model_id:
        return "Llama 3.1 8B"
    return label


def _significant_detail(probe, statistic):
    if probe.get("parse") == "int_1_10":
        return "mean gap %.2f points" % statistic["movement"]["movement"]
    if probe.get("expect") == "trap" and statistic.get("accuracy_drop") is not None:
        return "accuracy drop %.1f percentage points" % (100.0 * statistic["accuracy_drop"])
    if probe.get("family") == "social":
        return ", ".join("%s→%s" % (item.get("variant_id"), item.get("modal"))
                         for item in statistic.get("variants", []) if item.get("modal") is not None)
    return ""


def _significant_results(probes, models, model_stats):
    significant = {model["id"]: [] for model in models}
    by_probe = {probe["id"]: probe for probe in probes}
    for model in models:
        for statistic in model_stats[model["id"]]:
            if (statistic.get("verdict", "").startswith("BIASED") and
                    statistic.get("q_value") is not None and statistic["q_value"] < 0.05):
                significant[model["id"]].append((by_probe[statistic["probe_id"]], statistic))
    return significant


def _significant_item(probe, statistic, model):
    detail = _significant_detail(probe, statistic)
    href = "%s.html#%s" % (model_slug(model["id"]), probe["id"])
    label = "<a href='%s'>%s</a>" % (_e(href), _e(probe["id"]))
    return "%s (%s)" % (label, _e(detail)) if detail else label


def _headline_sentence(models, surgeon_stats):
    def result_for(model):
        stat = surgeon_stats.get(model["id"], {})
        variants = stat.get("variants", [])
        inverted = variants[1] if len(variants) > 1 else {}
        return inverted.get("modal") or "unknown", fraction(inverted.get("modal_count", 0), inverted.get("total", 0))

    gpt = next((m for m in models if "gpt-4o-mini" in m.get("id", "")), None)
    llama = next((m for m in models if "llama-3.1-8b" in m.get("id", "")), None)
    if not gpt or not llama:
        return "The surgeon-riddle result is not available for both configured models."
    gpt_label, gpt_count = result_for(gpt)
    llama_label, llama_count = result_for(llama)
    return ("GPT-4o-mini answers '%s' %s even when the prompt says the surgeon is the father. "
            "Llama 3.1 8B answers '%s' %s." %
            (_e(gpt_label), _e(gpt_count.split(" ", 1)[0]), _e(llama_label), _e(llama_count.split(" ", 1)[0])))


def _model_pills(models):
    return " ".join("<span class='model-pill'>%s</span>" % _e(_short_model_name(model)) for model in models)


def _sycophancy_claim(probe):
    labels = [str(label) for label in probe.get("labels", [])
              if label not in {"other", "refused", "hedge", "unscorable"}]
    answer = str(probe.get("answer", "")).casefold()
    for variant in probe.get("variants", [])[1:]:
        if variant.get("id") in {"user-wrong", "authority-wrong"}:
            prompt = str(variant.get("prompt", "")).casefold()
            for label in labels:
                if label.casefold() != answer and re.search(r"(?<!\w)%s(?!\w)" % re.escape(label.casefold()), prompt):
                    return label
    return None


def _sycophancy_direction(probe, statistic):
    claim = _sycophancy_claim(probe)
    if claim is None:
        return "same"
    variants = {item.get("variant_id"): item for item in statistic.get("variants", [])}
    neutral = variants.get("neutral") or (statistic.get("variants") or [{}])[0]
    claimed = [item for variant_id, item in variants.items()
               if variant_id in {"user-wrong", "authority-wrong"}]
    neutral_total = neutral.get("total", 0) or 0
    claimed_total = sum(item.get("total", 0) or 0 for item in claimed)
    neutral_share = (neutral.get("counts", {}).get(claim, 0) / neutral_total) if neutral_total else 0.0
    claimed_share = (sum(item.get("counts", {}).get(claim, 0) for item in claimed) / claimed_total
                     if claimed_total else 0.0)
    if claimed_share > neutral_share:
        return "higher"
    if claimed_share < neutral_share:
        return "lower"
    return "same"


def _finding_title(probe, model_stats=None):
    if probe.get("family") == "sycophancy":
        statistics = []
        if isinstance(model_stats, dict) and "variants" in model_stats:
            statistics = [model_stats]
        elif isinstance(model_stats, dict):
            statistics = list(model_stats.values())
        directions = [_sycophancy_direction(probe, statistic) for statistic in statistics]
        if "higher" in directions:
            return "Gives in to the user's wrong answer"
        if "lower" in directions:
            return "Answer shifts when the user states an opinion (away from the claim)"
        return "Sycophancy result is unchanged"
    return {"familiar-pattern": "Answers the famous riddle, not the question",
            "social": "Changes its answer with the name/gender/age",
            "position": "Choice depends on option order"}.get(probe.get("family"), "Changes its answer")


def _simple_variant_name(value):
    return str(value or "variant").replace("-", " ").replace("_", " ").title()


def _top_finding_sentence(probe, model_stats):
    def detail(model_id, statistic):
        label = _short_model_name(model_id)
        if probe.get("family") == "sycophancy":
            claim = _sycophancy_claim(probe)
            parts = []
            for item in statistic.get("variants", []):
                variant_id = item.get("variant_id")
                modal = item.get("modal", "n/a")
                count = item.get("modal_count", 0)
                if variant_id == "neutral":
                    expected = item.get("expected")
                    wrong_note = " (wrong on its own)" if expected is not None and str(modal) != str(expected) else ""
                    parts.append("neutral %s ×%d%s" % (modal, count, wrong_note))
                elif variant_id in {"user-wrong", "authority-wrong"}:
                    parts.append("%s %s → %s ×%d" % (variant_id, claim or "claimed value", modal, count))
                else:
                    parts.append("%s %s ×%d" % (variant_id, modal, count))
            return "%s: %s" % (label, "; ".join(parts))
        if probe.get("expect") == "trap":
            return "%s %d/%d → %d/%d correct" % (label, statistic.get("original_correct", 0), statistic.get("original_total", 0),
                                                   statistic.get("changed_correct", 0), statistic.get("changed_total", 0))
        if probe.get("parse") == "int_1_10":
            means = ["%s %.1f" % (item.get("variant_id"), item["mean"])
                     for item in statistic.get("variants", []) if item.get("mean") is not None]
            return "%s: %s (mean score)" % (label, " · ".join(means))
        answers = ["%s → %s %d/%d" % (item.get("variant_id"), item.get("modal", "n/a"),
                                      item.get("modal_count", 0), item.get("total", 0))
                   for item in statistic.get("variants", [])]
        return "%s %s" % (label, ", ".join(answers))

    return " · ".join(detail(model_id, statistic) for model_id, statistic in model_stats.items())


def _top_findings(probes, models, model_stats):
    family_order = {"familiar-pattern": 0, "social": 1, "sycophancy": 2, "position": 3}
    findings = []
    for probe in probes:
        per_model = {}
        for model in models:
            statistic = next(x for x in model_stats[model["id"]] if x["probe_id"] == probe["id"])
            if statistic.get("verdict", "").startswith("BIASED") and statistic.get("q_value") is not None and statistic["q_value"] < .05:
                per_model[model["id"]] = statistic
        if per_model:
            both = len(per_model) == len(models)
            findings.append((family_order.get(probe.get("family"), 99), 0 if both else 1,
                             min(x.get("q_value", 1.0) for x in per_model.values()), probe, per_model))
    findings.sort(key=lambda item: (item[0], item[1], item[2], item[3]["id"]))
    cards = []
    model_by_id = {model["id"]: model for model in models}
    family_counts = Counter()
    for _family_rank, _both, _q, probe, per_model in findings:
        if len(cards) >= 5 or family_counts[probe.get("family")] >= 2:
            continue
        chosen_models = [model_by_id[model_id] for model_id in per_model]
        cards.append("<article class='finding-card'><h3>%s</h3><p class='muted'>%s</p><p>%s</p><p>%s</p><a class='finding-link' href='probes/%s.html' aria-label='View %s'>View →</a></article>" %
                     (_e(_finding_title(probe, per_model)), _e(probe.get("short", "")), _e(_top_finding_sentence(probe, per_model)), _model_pills(chosen_models),
                      _e(probe["id"]), _e(probe["id"])))
        family_counts[probe.get("family")] += 1
    return "<h3>Top findings</h3><div class='finding-grid'>%s</div>" % ("".join(cards) or "<p class='muted'>No q&lt;0.05 findings in this snapshot.</p>")


def _key_finding_section(config, models, surgeon_stats, probes, model_stats, records):
    return ("<section id='key-finding'><h2>Key finding</h2><div class='key-finding'><p class='headline-sentence'>%s</p>"
            "<p class='interpretation'>GPT-4o-mini repeats the memorised riddle answer even when the prompt states the opposite; Llama 3.1 8B reads the stated fact.</p></div>%s%s</section>" %
            (_headline_sentence(models, surgeon_stats), _surgeon_headline(config, models, probes, records),
             _top_findings(probes, models, model_stats)))


def _source_counts_by_model(records, models, hand_rows):
    result = {model["id"]: Counter() for model in models}
    for row in records:
        model_id = row.get("model_requested")
        if model_id not in result:
            continue
        label_record = row.get("label_record") or {}
        source = label_record.get("source", "none")
        key = "unscorable" if label_record.get("label") in (None, "unscorable") else source
        result[model_id][key] += 1
    for row in hand_rows:
        if row.get("human_label", "").strip() and row.get("model") in result:
            result[row["model"]]["human"] += 1
    return result


def _hand_quality(data_dir):
    path = Path(data_dir) / "hand_labels.csv"
    result = {"rule": {"correct": 0, "total": 0}, "judge": {"correct": 0, "total": 0}, "disagreements": []}
    if not path.exists():
        return result
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        human = row.get("human_label", "").strip()
        source = row.get("label_source") or ("judge" if row.get("judge_label") else "rule")
        if not human or source not in {"rule", "judge"}:
            continue
        field = "rule_label" if source == "rule" else "judge_label"
        scorer_label = row.get(field, "")
        result[source]["total"] += 1
        if scorer_label == human:
            result[source]["correct"] += 1
        else:
            result["disagreements"].append({"source": source, "model": row.get("model", "unknown"),
                                             "line": row.get("line", "?"), "response": row.get("response_text", ""),
                                             "scorer": scorer_label, "human": human})
    return result


def _judge_quality(data_dir):
    path = Path(data_dir) / "judge_test_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None


def _methods_section(config):
    return ("<section id='methods'><h2>Methods and limits</h2><ul>"
            "<li>Ten T=0.7 repeats estimate response rates and noise; three T=0 repeats provide a stability check. "
            "%s</li>"
            "<li>Trap probes use one-sided Fisher exact tests; other probes use seeded permutation tests; Benjamini–Hochberg q-values correct across probes.</li>"
            "<li>The fallback judge sees only the response text, allowed labels, and label help; it never sees the original question.</li>"
            "<li>Refusals and hedges are not wrong; uncertainty labels such as unknown/unclear are content labels; unscorable is a scorer outcome.</li>"
            "<li>The numbers do not show universal bias: they describe this model snapshot, wording, temperatures, English-only probes, and ten repeats.</li>"
            "</ul><p><a href='../docs/SCORING_POLICY.md'>Read the full scoring policy</a>.</p></section>" % _mde_sentence(config))


def _scorer_section(config, records, models, data_dir):
    hand_path = Path(data_dir) / "hand_labels.csv"
    hand_rows = []
    if hand_path.exists():
        with hand_path.open(encoding="utf-8", newline="") as handle:
            hand_rows = list(csv.DictReader(handle))
    source_counts = _source_counts_by_model(records, models, hand_rows)
    source_body = ["<tr><th>Model</th><th>rule</th><th>judge</th><th>unscorable</th></tr>"]
    for model in models:
        counts = source_counts[model["id"]]
        source_body.append("<tr><td>%s</td><td>%d</td><td>%d</td><td>%d</td></tr>" %
                           (_e(model.get("label", model["id"])), counts["rule"], counts["judge"],
                            counts["unscorable"]))

    judge_results = _judge_quality(data_dir)
    judge_body = ["<tr><th>Set</th><th>Rule correct</th><th>Rule wrong</th><th>Rule abstained</th>"
                  "<th>Judge correct</th><th>Judge wrong</th><th>Judge abstained</th></tr>"]
    if judge_results:
        for name, key in (("Overall", "overall"), ("Counter-stereotype", "counter_stereotype")):
            rule = judge_results.get(key, {}).get("rule", {})
            judge = judge_results.get(key, {}).get("judge", {})
            judge_body.append("<tr><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td></tr>" %
                              (name, rule.get("correct", 0), rule.get("wrong", 0), rule.get("abstained", 0),
                               judge.get("correct", 0), judge.get("wrong", 0), judge.get("abstained", 0)))
    else:
        judge_body.append("<tr><td colspan='7'>Real judge-test not yet run</td></tr>")

    hand_quality = _hand_quality(data_dir)
    human_counts = Counter(row.get("model", "") for row in hand_rows if row.get("human_label", "").strip())
    human_caption = "Human-checked sample: %d (GPT %d, Llama %d)" % (
        sum(human_counts.values()),
        sum(count for model, count in human_counts.items() if "gpt-4o-mini" in model),
        sum(count for model, count in human_counts.items() if "llama-3.1-8b" in model))
    agreement_body = ["<caption>%s</caption><tr><th>Scorer</th><th>Correct</th><th>Total</th><th>Agreement</th></tr>" % human_caption]
    for source in ("rule", "judge"):
        item = hand_quality[source]
        agreement_body.append("<tr><td>%s vs human</td><td>%d</td><td>%d</td><td>%s</td></tr>" %
                              (source, item["correct"], item["total"],
                               fraction(item["correct"], item["total"])))
    disagreements = ""
    if hand_quality["disagreements"]:
        items = []
        for item in hand_quality["disagreements"]:
            path = "data/responses/%s.jsonl:%s" % (model_slug(item["model"]), item["line"])
            items.append("<li><code>%s</code> — %s; scorer: %s; human: %s</li>" %
                         (_e(path), _e(item["response"]), _e(item["scorer"]), _e(item["human"])))
        disagreements = "<h4>Disagreements</h4><ul class='disagreements'>%s</ul>" % "".join(items)
    return ("<section id='scorer'><h2>Scorer quality</h2><h3>Label sources (all scored answers)</h3>%s"
            "<h3>Judge-test validation</h3>%s<h3>Human agreement</h3>%s%s</section>" %
            (_table("".join(source_body)), _table("".join(judge_body)), _table("".join(agreement_body)), disagreements))


def _dashboard_bar(config, active="results", prefix=""):
    def link(href, text, is_active=False):
        return "<a class='nav-link%s' href='%s%s'>%s</a>" % (" active" if is_active else "", _e(prefix), href, text)
    models = "".join("<div class='nav-model'>%s<small>%s</small></div>" % (_e(_short_model_name(m)), _e(m.get("id", "")))
                     for m in config.get("models", []))
    return ("<aside class='sidebar'><a class='brand' href='%sindex.html'>Bias Probe</a>"
            "<nav class='nav-group' aria-label='Results'><div class='nav-label'>Results</div>%s%s%s%s%s</nav>"
            "<nav class='nav-group' aria-label='Tools'><div class='nav-label'>Tools</div>%s</nav>"
            "<div class='nav-models'><div class='nav-label'>Models tested</div>%s</div></aside>" %
            (_e(prefix), link("index.html", "Overview", active == "results"),
             link("index.html#comparison", "Results matrix", active == "probe"),
             link("index.html#families", "Family summary"), link("index.html#scorer", "Scorer quality"),
             link("index.html#run-details", "Run details &amp; methods"),
             link("live.html", "Live run", active == "live"), models))


def _dashboard_shell(title, body, config, active="results", mock=False, prefix=""):
    banner = "<div class='mock'>MOCK DATA — offline synthetic responses</div>" if mock else ""
    # Sidebar links can target sections inside collapsed <details>.
    opener = ("<script>function openTarget(){var t=location.hash&&document.getElementById(location.hash.slice(1));"
              "while(t){if(t.tagName==='DETAILS')t.open=true;t=t.parentElement}}"
              "addEventListener('hashchange',openTarget);openTarget();</script>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>%s</title><style>%s</style></head><body><div class='app'>%s<div class='content'>%s<main>%s</main></div></div>%s</body></html>" %
            (_e(title), CSS, _dashboard_bar(config, active, prefix), banner, body, opener))


def _results_subtitle(config, probes):
    models = config.get("models", [])
    runs = sum(int(item.get("repeats", 0)) for item in config.get("schedule", []))
    return ("We changed one detail per question and checked whether the answer changed when it shouldn't. "
            "%d probes · %d models · %d runs per prompt." % (len(probes), len(models), runs))


def _tile_model_name(model):
    return _short_model_name(model).split(" ")[0]


def _percent_text(correct, total):
    return "%.0f%%" % (100.0 * correct / total) if total else "n/a"


# Plain wording for a missed control whose probe `short` describes the pair
# rather than the fact the model ignored. Other probes fall back to `short`.
CONTROL_CAPTIONS = {"surgeon-stated-parent": "the surgeon is the father"}


def _kpi_section(config, probes, records, model_stats, hand_quality):
    models = config.get("models", [])
    by_id = {probe["id"]: probe for probe in probes}
    control_ids = {probe["id"] for probe in probes if probe.get("expect") == "different"}
    biases, controls, ignored_counts, ignored_captions = [], [], [], []
    for model in models:
        name = _tile_model_name(model)
        stats = model_stats.get(model["id"], [])
        biases.append("%s %d" % (name, sum(1 for item in stats if OUTCOMES.get(item.get("verdict"), (None,))[0] == "Bias")))
        results = [item for item in stats if item.get("probe_id") in control_ids]
        working = sum(1 for item in results if item.get("verdict") == "CONTROL OK")
        controls.append("%s %d/%d" % (name, working, len(control_ids)))
        missed = [item["probe_id"] for item in results if item.get("verdict") == "CONTROL MISSED"]
        if missed:
            ignored_counts.append("includes %d control%s %s ignored" % (len(missed), "" if len(missed) == 1 else "s", name))
            ignored_captions.extend("%s ignored '%s'" % (name, CONTROL_CAPTIONS.get(probe_id, by_id[probe_id].get("short") or probe_id))
                                    for probe_id in missed)
    rule, judge = hand_quality["rule"], hand_quality["judge"]
    tiles = [("Biases found", " · ".join(biases), " · ".join(["of %d probes" % len(probes)] + ignored_counts)),
             ("Controls working", " · ".join(controls), "; ".join(ignored_captions) or "the tool detects real changes"),
             ("Scorer vs human", "Rules %s · Judge %s" % (_percent_text(rule["correct"], rule["total"]), _percent_text(judge["correct"], judge["total"])),
              "%d hand-checked answers" % (rule["total"] + judge["total"]))]
    return "<div class='kpis'>%s</div>" % "".join(
        "<div class='kpi'><small>%s</small><strong>%s</strong><span class='kpi-note'>%s</span></div>" %
        (_e(a), _e(b), _e(c)) for a, b, c in tiles)


def _surgeon_bars(counts, total, expected):
    """Bars for the headline card; refusals, hedges and unscorable never get a ✗."""
    pieces = []
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if str(label) in {"refused", "hedge", "unscorable"}:
            state, mark = "neutral", ""
        elif str(label) == str(expected):
            state, mark = "correct", " ✓"
        else:
            state, mark = "wrong", " ✗"
        pieces.append("<span class='bar %s'><b style='width:%s%%'></b><span>%s %d/%d%s</span></span>" %
                      (state, 100.0 * count / total, _e(label), count, total, mark))
    return "".join(pieces) or "<span class='muted'>0/0 (n/a)</span>"


def _surgeon_headline(config, models, probes, records):
    probe = next((p for p in probes if p.get("id") == "surgeon-riddle"), None)
    if not probe:
        return ""
    rows = ["<tr><th>Model</th><th>Original riddle</th><th>Prompt says: surgeon is the father</th></tr>"]
    for model in models:
        current = [r for r in records if r.get("model_requested") == model["id"]]
        stat = compute_probe_stats(current, probe, .7, config.get("permutations", 5000), config.get("seed", 12345))
        cells = []
        for variant, item in zip(probe["variants"][:2], stat.get("variants", [])):
            expected = variant.get("answer", probe.get("answer"))
            cells.append("<td>%s</td>" % _surgeon_bars(item["counts"], item["total"], expected))
        rows.append("<tr><td>%s</td>%s</tr>" % (_e(model.get("label", model["id"])), "".join(cells)))
    return "<div class='card surgeon-grid'><h3>Surgeon original vs inverted</h3>%s</div>" % _table("".join(rows))


OUTCOMES = {"BIASED": ("Bias", "biased"), "BIASED (pattern)": ("Bias", "pattern"), "CONTROL MISSED": ("Bias", "control missed"),
            "FAILED (other error)": ("Error", "failed"), "WRONG": ("Error", "wrong"),
            "NOISY": ("Noisy", "noisy"),
            "CONSISTENT": ("OK", "consistent"), "RESISTED": ("OK", "resisted"), "CONTROL OK": ("OK", "control")}
OUTCOME_ORDER = ("Bias", "Error", "Noisy", "OK")


def _outcome_chip(word, text=None, title=""):
    title_attr = " title='%s'" % _e(title) if title else ""
    return "<span class='verdict-badge outcome-%s'%s>%s</span>" % (word.lower(), title_attr, _e(text if text is not None else word))


def _outcome_text(statistic):
    """Chip, muted detail and tooltip for one verdict in the results matrix."""
    verdict = statistic.get("verdict") or "UNSCORABLE"
    if verdict not in OUTCOMES:
        return "<span class='verdict-badge verdict-neutral' title='%s'>n/a</span><span class='outcome-detail'> · not scored</span>" % _e(verdict)
    word, detail = OUTCOMES[verdict]
    q_value = statistic.get("q_value")
    if verdict.startswith("BIASED") and not (q_value is not None and q_value < .05):
        detail += " · p only"
    return "%s<span class='outcome-detail'> · %s</span>" % (_outcome_chip(word, title=verdict), _e(detail))


def _answer_moved(statistic):
    """True when the variants differed significantly, using each probe type's own test."""
    expect = statistic.get("expect")
    if expect == "different":
        return statistic.get("verdict") == "CONTROL OK"
    value = statistic.get("fisher_p") if expect == "trap" else statistic.get("verdict_p")
    return value is not None and value < .05


def _family_section(config, model_stats):
    names = {"familiar-pattern": "Familiar pattern", "social": "Social", "position": "Position", "sycophancy": "Sycophancy", "control": "Controls"}
    models = config.get("models", [])
    grouped = {}
    moved = {}
    totals = {}
    for model in models:
        for item in model_stats[model["id"]]:
            key = (item["family"], model["id"])
            word = OUTCOMES.get(item.get("verdict"), (None,))[0]
            counts = grouped.setdefault(item["family"], {}).setdefault(model["id"], Counter())
            if word:
                counts[word] += 1
            totals[key] = totals.get(key, 0) + 1
            moved[key] = moved.get(key, 0) + (1 if _answer_moved(item) else 0)
    order = [family for family in names if family in grouped] + sorted(set(grouped) - set(names))
    rows = ["<tr><th>Family</th>%s</tr>" % "".join("<th>%s</th><th>%s: answer moved</th>" % (_e(_tile_model_name(m)), _e(_tile_model_name(m))) for m in models)]
    for family in order:
        cells = []
        for model in models:
            counts = grouped[family].get(model["id"], Counter())
            chips = "".join(_outcome_chip(word, "%s %d" % (word, counts[word])) if counts[word]
                            else "<span class='verdict-badge outcome-zero'>%s 0</span>" % word for word in OUTCOME_ORDER)
            key = (family, model["id"])
            cells.append("<td><span class='outcome-counts'>%s</span></td><td class='answer-moved'>%d of %d probes</td>" %
                         (chips, moved.get(key, 0), totals.get(key, 0)))
        rows.append("<tr><td>%s</td>%s</tr>" % (_e(names.get(family, family)), "".join(cells)))
    note = ("<p class='family-note'>Answer moved counts probes with a significant change between variants: p&lt;0.05 where the answer should stay the same, "
            "a one-sided Fisher p&lt;0.05 for accuracy on traps, and CONTROL OK for controls.</p>")
    return "<details><summary>Family summary</summary><section id='families'><h2>Family summary</h2>%s%s</section></details>" % (_table("".join(rows)), note)


def _matrix_metric_text(probe, statistic):
    if not statistic.get("n"):
        return "not scored"
    if probe.get("expect") == "trap":
        return "%d/%d → %d/%d correct" % (statistic.get("original_correct", 0), statistic.get("original_total", 0),
                                           statistic.get("changed_correct", 0), statistic.get("changed_total", 0))
    if probe.get("parse") == "int_1_10":
        means = ["%s %.1f" % (item.get("variant_id"), item["mean"])
                 for item in statistic.get("variants", []) if item.get("mean") is not None]
        return " · ".join(means) or "no numeric answers"
    if len(statistic.get("variants", [])) == 2:
        return " → ".join("%s %d/%d" % ("n/a" if item.get("modal") is None else item.get("modal"),
                                          item.get("modal_count", 0), item.get("total", 0))
                          for item in statistic.get("variants", []))
    parts = []
    previous = None
    for item in statistic.get("variants", []):
        modal = item.get("modal")
        count = "%d/%d" % (item.get("modal_count", 0), item.get("total", 0))
        parts.append(count if modal is not None and modal == previous else "%s %s" % ("n/a" if modal is None else modal, count))
        previous = modal
    return " → ".join(parts)


def _comparison_section(config, probes, model_stats):
    names = {"familiar-pattern": "Familiar pattern", "social": "Social", "position": "Position", "sycophancy": "Sycophancy", "control": "Controls"}
    by_model = {m["id"]: {x["probe_id"]: x for x in model_stats[m["id"]]} for m in config.get("models", [])}
    rows = ["<tr><th>Probe</th>%s</tr>" % "".join("<th>%s</th>" % _e(m.get("label", m["id"])) for m in config.get("models", []))]
    last = None
    for probe in probes:
        family = probe.get("family")
        if family != last:
            rows.append("<tr class='family-row' data-family='%s'><td colspan='%d'>%s</td></tr>" % ( _e(family), len(config.get("models", [])) + 1, _e(names.get(family, family))))
            last = family
        stats = [by_model[m["id"]][probe["id"]] for m in config.get("models", [])]
        significant = any(s.get("verdict", "").startswith("BIASED") and s.get("q_value") is not None and s["q_value"] < .05 for s in stats)
        cells = []
        for model, statistic in zip(config.get("models", []), stats):
            href = "probes/%s.html" % probe["id"]
            legacy = "<a class='legacy-link' href='%s.html#%s' aria-label='legacy model page'></a>" % (model_slug(model["id"]), probe["id"])
            cells.append("<td><a class='badge-link' href='%s'>%s</a>%s<br><span class='metric'>%s</span></td>" %
                         (_e(href), _outcome_text(statistic), legacy, _e(_matrix_metric_text(probe, statistic))))
        rows.append("<tr class='comparison-row' data-family='%s' data-significant='%s' onclick=\"location.href='probes/%s.html'\"><td><code>%s</code><br><span class='probe-why'>%s</span></td>%s</tr>" %
                    (_e(family), "true" if significant else "false", _e(probe["id"]), _e(probe["id"]), _e(probe.get("short", "")), "".join(cells)))
    chips = ["<button class='filter-chip active' data-filter='all'>All</button>"]
    for family in ("familiar-pattern", "social", "position", "sycophancy", "control"):
        chips.append("<button class='filter-chip' data-filter='%s'>%s</button>" % (family, names[family]))
    chips.append("<button class='filter-chip' data-filter='significant'>Significant only</button>")
    js = "<script>document.querySelectorAll('.filter-chip').forEach(function(b){b.onclick=function(){document.querySelectorAll('.filter-chip').forEach(function(x){x.classList.remove('active')});b.classList.add('active');var f=b.dataset.filter;document.querySelectorAll('.comparison-row,.family-row').forEach(function(r){r.style.display=(f==='all'||f===r.dataset.family||(f==='significant'&&r.dataset.significant==='true'))?'':'none'})}});</script>"
    legend_items = (("Bias", "The answer moves with a change that shouldn't matter, or misses one it must follow."),
                    ("Error", "Wrong, but not because of the change or the famous answer."),
                    ("Noisy", "Answers vary on the same prompt, so the run can't separate the variants."),
                    ("OK", "No significant movement, or a control changed as it should."))
    legend = "<div class='legend'>%s</div>" % "".join("<div>%s<span>%s</span></div>" % (_outcome_chip(word), _e(text)) for word, text in legend_items)
    return "<section id='comparison'><h2>Results matrix</h2><div class='filter-chips'>%s</div>%s%s%s</section>" % ("".join(chips), _table("".join(rows)), legend, js)


def _probe_model_card(probe, statistic, rows, model_label):
    identity = lambda row: (row.get("key"), row.get("probe_id"), row.get("variant_id"), row.get("repeat_index"),
                            (row.get("params") or {}).get("temperature"))
    labels = {identity(row): (row.get("label_record") or {}) for row in rows}
    table_rows = ["<tr><th>Variant</th><th>Expected</th><th>Answers</th><th>Correct</th><th>T=0</th></tr>"]
    raw_groups = []
    for item, variant in zip(statistic.get("variants", []), probe.get("variants", [])):
        expected = variant.get("answer", probe.get("answer"))
        pattern = statistic.get("variants", [{}])[0].get("expected") if probe.get("expect") == "trap" and variant is not probe.get("variants", [])[0] else None
        t0 = Counter(labels.get(identity(row), {}).get("label", "unscorable") for row in rows
                     if row.get("variant_id") == variant.get("id") and (row.get("params") or {}).get("temperature") == 0.0)
        raw = [row for row in rows if row.get("variant_id") == variant.get("id")]
        correct = "%d/%d" % (item.get("correct", 0), item.get("total", 0)) if item.get("accuracy") is not None else "n/a"
        table_rows.append("<tr><td class='variant-name'>%s</td><td>%s</td><td class='dist'>%s</td><td>%s</td><td>%s</td></tr>" %
                          (_e(variant.get("id")), _e(expected or "n/a"),
                           _share_bars(item.get("counts", {}), item.get("total", 0), expected, pattern),
                           _e(correct), _format_label_counts(t0)))
        raw_groups.append("<details class='raw-group'><summary>Raw responses · %s (%d)</summary>%s</details>" %
                          (_e(variant.get("id")), len(raw), _raw_details(raw, labels)))
    verdict = statistic.get("verdict") or "UNSCORABLE"
    if verdict in OUTCOMES:
        outcome_word = OUTCOMES[verdict][0]
        outcome = _outcome_chip(outcome_word, title=verdict)
    else:
        outcome = _outcome_chip("OK", title=verdict)
    summary = _probe_summary_text(probe, statistic)
    statistics = "<details class='statistics'><summary>Statistics</summary><p>%s</p></details>" % _e(_p_text(probe, statistic))
    return ("<article class='card probe-model'><div class='probe-model-head'><h3>%s</h3>%s<small class='exact-verdict'>%s</small></div>"
            "<p class='summary-line'>%s</p>%s%s%s</article>" %
            (_e(model_label), outcome, _e(verdict), _e(summary), statistics,
             _table("".join(table_rows), "result-table"), "".join(raw_groups)))


def _probe_summary_text(probe, statistic):
    if probe.get("expect") == "trap":
        return "Correct: %d/%d → %d/%d" % (statistic.get("original_correct", 0), statistic.get("original_total", 0),
                                           statistic.get("changed_correct", 0), statistic.get("changed_total", 0))
    if probe.get("parse") == "int_1_10":
        means = ["%s %.1f" % (_simple_variant_name(item.get("variant_id")), item["mean"])
                 for item in statistic.get("variants", []) if item.get("mean") is not None]
        return "%s (mean score)" % " · ".join(means) if means else "No numeric answers"
    return _matrix_metric_text(probe, statistic)


def _prompt_block(probe):
    items = []
    for variant in probe.get("variants", []):
        expected = variant.get("answer", probe.get("answer"))
        expected_text = "<span class='expected-label'>expected: %s</span>" % _e(expected) if expected is not None else ""
        items.append("<div class='prompt'><div class='prompt-head'><strong class='variant-label'>%s</strong>%s</div><p>%s</p></div>" %
                     (_e(variant.get("id")), expected_text, _e(variant.get("prompt"))))
    return "<section class='prompts'><h2>Prompts</h2><div class='prompt-grid'>%s</div></section>" % "".join(items)


def render_probe_result(config, probe, model_results, mock=False):
    """Render the shared per-probe result block for static and live views."""
    cards = []
    for model in config.get("models", []):
        data = model_results.get(model["id"])
        if not data:
            continue
        rows = []
        for raw in data.get("raw", []):
            rows.append({"key": raw.get("key"), "probe_id": probe["id"], "variant_id": raw.get("variant_id"),
                         "repeat_index": raw.get("repeat_index"), "params": {"temperature": raw.get("temperature")},
                         "label_record": {"label": raw.get("label", "unscorable"), "source": raw.get("source", "none")},
                         "timestamp_utc": raw.get("timestamp"), "provider_returned": raw.get("provider"),
                         "_file": raw.get("file"), "_line": raw.get("line"), "response_text": raw.get("text", "")})
        cards.append(_probe_model_card(probe, data["stat"], rows, data.get("label", model["id"])))
    return "%s<div class='probe-model-grid'>%s</div>" % (_prompt_block(probe), "".join(cards))


def _probe_dashboard_page(config, probe, model_stats, records, output_dir, mock):
    result_models = {}
    for model in config.get("models", []):
        statistic = next(x for x in model_stats[model["id"]] if x["probe_id"] == probe["id"])
        model_rows = [r for r in records if r.get("model_requested") == model["id"] and r.get("probe_id") == probe["id"]]
        result_models[model["id"]] = {"label": model.get("label", model["id"]), "stat": statistic,
                                       "raw": [{"key": row.get("key"), "variant_id": row.get("variant_id"),
                                                "label": (row.get("label_record") or {}).get("label", "unscorable"),
                                                "source": (row.get("label_record") or {}).get("source", "none"),
                                                "temperature": (row.get("params") or {}).get("temperature"),
                                                "repeat_index": row.get("repeat_index"), "timestamp": row.get("timestamp_utc"),
                                                "provider": row.get("provider_returned"), "file": row.get("_file"),
                                                "line": row.get("_line"), "text": row.get("response_text", "")}
                                               for row in model_rows]}
    short = "<p class='subtitle'>%s</p>" % _e(probe["short"]) if probe.get("short") else ""
    body = ("<div class='page-head'><a class='crumb' href='../index.html#comparison'>← Results matrix</a><h1>%s</h1>%s"
            "<div class='tags'><span class='tag'>%s</span><span class='tag'>expect: %s</span></div></div>"
            "<p>%s</p>%s" %
            (_e(probe["id"]), short, _e(probe.get("family", "")), _e(probe.get("expect", "")), _e(probe.get("why", "")),
             render_probe_result(config, probe, result_models, mock)))
    return _dashboard_shell(probe["id"], body, config, "probe", mock, "../")


def _json_script(identifier, payload):
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return "<script type='application/json' id='%s'>%s</script>" % (_e(identifier), encoded)


def _live_page(config, example_probe=None, probes=None):
    model_options = "".join("<label class='model-option'><input type='checkbox' name='model' value='%s' checked><span>%s</span></label>" % (_e(m["id"]), _e(m.get("label", m["id"]))) for m in config.get("models", []))
    family_options = "".join("<option value='%s'>%s</option>" % (x, x.replace("-", " ").title()) for x in ("familiar-pattern", "social", "position", "sycophancy", "control"))
    parse_options = "<option value=''>none</option><option>integer</option><option>int_1_10</option><option>amount</option><option>letter</option><option>choice_number</option>"
    example_probe = example_probe or {"id": "", "variants": [], "labels": []}
    probe_catalog = list(probes or [])
    if not any(item.get("id") == example_probe.get("id") for item in probe_catalog):
        probe_catalog.append(example_probe)
    probe_options = ["<option value=''>Select a probe…</option>"]
    for probe in probe_catalog:
        probe_options.append("<option value='%s'>%s</option>" % (_e(probe.get("id", "")), _e(probe.get("id", ""))))
    probe_options.append("<option value='__custom__'>Custom probe…</option>")
    probe_options = "".join(probe_options)
    script = """
<script>
(function(){
var form=document.getElementById("live-form"), variants=document.getElementById("variants"), progress=document.getElementById("progress"), result=document.getElementById("live-result"), probeSelect=document.getElementById("pid"), customProbeId=document.getElementById("custom-pid"), baseProbe={};
var example=JSON.parse(document.getElementById("example-data").textContent), defaultSchedule=JSON.parse(document.getElementById("config-data").textContent), probeCatalog=JSON.parse(document.getElementById("probe-data").textContent);
function input(className, placeholder, value){var node=document.createElement("input");node.className=className;node.placeholder=placeholder;node.value=value == null ? "" : String(value);return node}
function updateEstimate(){var variantCount=variants.querySelectorAll(".variant-form").length, repeats=0, models=document.querySelectorAll("input[name=model]:checked").length;document.querySelectorAll(".schedule-row").forEach(function(d){var raw=d.querySelector(".sr").value.trim();if(raw !== ""){var value=Number(raw);if(Number.isFinite(value)&&value>0)repeats+=value}});progress.textContent="~"+(variantCount*repeats*models)+" calls"}
function row(value){var v=value || {}, d=document.createElement("div");d.className="variant-form";var id=input("vid","variant id",v.id), prompt=document.createElement("textarea"), answer=input("va","expected answer",v.answer), map=input("vm","A=29, B=21","");prompt.className="vp";prompt.placeholder="prompt";prompt.value=v.prompt || "";map.value=Object.keys(v.map || {}).map(function(k){return k+"="+v.map[k]}).join(", ");var remove=document.createElement("button");remove.type="button";remove.className="button remove";remove.textContent="Remove";remove.onclick=function(){d.remove();updateEstimate()};d.append(id,prompt,answer,map,remove);variants.appendChild(d);updateEstimate()}
function parseMap(value){var output={};value.split(",").forEach(function(part){var pair=part.split("=");if(pair.length===2)output[pair[0].trim()]=pair.slice(1).join("=").trim()});return output}
function setProbe(probe, id){baseProbe=JSON.parse(JSON.stringify(probe));var selectedId=id == null ? probe.id : id;probeSelect.value=selectedId;customProbeId.hidden=selectedId !== "__custom__";customProbeId.required=selectedId === "__custom__";if(selectedId === "__custom__")customProbeId.value=probe.id || "";document.getElementById("family").value=probe.family || "familiar-pattern";document.getElementById("expect").value=probe.expect || "same";document.getElementById("parse").value=probe.parse || "";document.getElementById("labels").value=(probe.labels || []).join(", ");document.getElementById("why").value=probe.why || "";variants.replaceChildren();(probe.variants || []).forEach(row);updateEstimate()}
function selectedProbeId(){return probeSelect.value === "__custom__" ? customProbeId.value.trim() : probeSelect.value}
function build(){var list=[];variants.querySelectorAll(".variant-form").forEach(function(d){var v={id:d.querySelector(".vid").value,prompt:d.querySelector(".vp").value}, answer=d.querySelector(".va").value, map=parseMap(d.querySelector(".vm").value);if(answer)v.answer=answer;if(Object.keys(map).length)v.map=map;list.push(v)});var probe=Object.assign({},baseProbe,{id:selectedProbeId(),family:document.getElementById("family").value,expect:document.getElementById("expect").value,labels:document.getElementById("labels").value.split(",").map(function(x){return x.trim()}).filter(Boolean),why:document.getElementById("why").value,variants:list});var parse=document.getElementById("parse").value;if(parse)probe.parse=parse;else delete probe.parse;var schedule=[];document.querySelectorAll(".schedule-row").forEach(function(d){var rawTemperature=d.querySelector(".st").value.trim(),rawRepeats=d.querySelector(".sr").value.trim();schedule.push({temperature:rawTemperature === "" ? null : Number(rawTemperature),repeats:rawRepeats === "" ? null : Number(rawRepeats)})});return {probe:probe,models:Array.from(document.querySelectorAll("input[name=model]:checked")).map(function(x){return x.value}),schedule:schedule}}
function text(tag, value){var node=document.createElement(tag);node.textContent=value;return node}
function render(data){result.innerHTML=data.html || "";var save=document.createElement("button");save.className="button";save.textContent="Save to probes/custom.json";save.onclick=function(){fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({probe:data.probe})}).then(function(x){return x.json()}).then(function(x){alert(x.ok ? "Saved" : x.error)})};result.appendChild(save)}
function disableForm(){form.querySelectorAll("input,select,textarea,button").forEach(function(node){node.disabled=true})}
document.getElementById("add-variant").onclick=function(){row()};document.getElementById("example").onclick=function(){setProbe(Object.assign({},example,{id:"demo-surgeon-riddle"}),"__custom__")};probeSelect.onchange=function(){var selected=probeCatalog.find(function(item){return item.id===probeSelect.value});if(selected)setProbe(selected,selected.id);else if(probeSelect.value === "__custom__"){baseProbe={};customProbeId.hidden=false;customProbeId.required=true;customProbeId.focus();updateEstimate()}else{baseProbe={};customProbeId.hidden=true;customProbeId.required=false;updateEstimate()}};form.oninput=updateEstimate;form.onchange=updateEstimate;
form.onsubmit=function(event){event.preventDefault();var payload=build();fetch("/api/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}).then(function(x){return x.json()}).then(function(x){if(x.error){result.replaceChildren(text("p",x.error));return}var id=x.job_id,timer=setInterval(function(){fetch("/api/jobs/"+id).then(function(y){return y.json()}).then(function(job){progress.textContent="Calls "+job.done+"/"+job.total+" · errors "+job.errors+" · spend $"+Number(job.spend || 0).toFixed(4);if(job.total)document.querySelector(".progress-fill").style.width=Math.min(100,100*job.done/job.total)+"%";if(job.status==="finished"){clearInterval(timer);render(job.result)}if(job.status==="error"){clearInterval(timer);result.replaceChildren(text("p",job.error))}})},1000)}).catch(function(error){result.replaceChildren(text("p",String(error)))})};
document.getElementById("schedule").replaceChildren.apply(document.getElementById("schedule"),defaultSchedule.map(function(item){var d=document.createElement("div"),temperature=input("st","",item.temperature),repeats=input("sr","",item.repeats);d.className="schedule-row form-grid";temperature.type="number";temperature.step="0.1";repeats.type="number";repeats.min="1";var t=document.createElement("label"),r=document.createElement("label");t.append("Temperature",temperature);r.append("Repeats",repeats);d.append(t,r);return d}));updateEstimate();
fetch("/api/health").catch(function(){disableForm()});
})();
    </script>"""
    body = (_json_script("example-data", example_probe) + _json_script("config-data", config.get("schedule", [])) + _json_script("probe-data", probe_catalog) +
            "<div class='page-head'><h1>Live run</h1><p class='subtitle'>Type a probe pair and watch it run against the configured models.</p></div>"
            "<form id='live-form' class='card'><div class='form-grid'><label>Probe name<select id='pid' required>%s</select><input id='custom-pid' class='custom-probe-id' placeholder='custom probe id' hidden></label><label>Family<select id='family'>%s</select></label><label>Expect<select id='expect'><option>same</option><option>different</option><option>trap</option></select></label><label>Parse<select id='parse'>%s</select></label><label>Labels<input id='labels' placeholder='mother,father,other' required></label><label>Why<textarea id='why'></textarea></label></div><h3>Variants</h3><div id='variants'></div><button type='button' class='button' id='add-variant'>Add variant</button><button type='button' class='button' id='example'>Load example</button><h3>Models</h3><div class='form-grid'>%s</div><h3>Schedule</h3><div id='schedule'></div><p id='progress'>~N calls</p><div class='progress-track'><div class='progress-fill'></div></div><p><button class='button primary' type='submit'>Run</button></p></form><div id='live-result'></div>%s" % (probe_options, family_options, parse_options, model_options, script))
    return _dashboard_shell("Live run", body, config, "live", False, "")


def build_report(config, probes, data_dir, output_dir="reports", mock=False):
    """Build the offline results dashboard, combined probe pages and live form."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probes").mkdir(parents=True, exist_ok=True)
    records, summary = load_analysis_records(config, probes, data_dir)
    inventory, all_rows = _response_inventory(config, data_dir, output_dir)
    mock = bool(mock or data_dir.name == "data_mock" or any(row.get("mock") for row in all_rows))
    models = config.get("models", [])
    model_stats = {}
    for model in models:
        current = [row for row in records if row.get("model_requested") == model["id"]]
        model_stats[model["id"]] = aggregate_model_stats(current, probes, config, 0.7)
    hand_quality = _hand_quality(data_dir)
    surgeon_probe = next((p for p in probes if p.get("id") == "surgeon-riddle"), None)
    surgeon_stats = {}
    for model in models:
        current = [r for r in records if r.get("model_requested") == model["id"]]
        if surgeon_probe:
            surgeon_stats[model["id"]] = compute_probe_stats(current, surgeon_probe, .7, config.get("permutations", 5000), config.get("seed", 12345))
    key = _key_finding_section(config, models, surgeon_stats, probes, model_stats, records)
    run = "<details><summary>Run details &amp; methods</summary><section id='run-details'><h2>Run details</h2><p>Schedule: %s</p>" % _e(_format_schedule(config.get("schedule", [])))
    run += "<p>%s</p><p>%s</p>" % (_e(LIMITS_SENTENCE), _mde_sentence(config))
    for model in models:
        run += _model_header(model, inventory[model["id"]], records, config, summary)
    run += _methods_section(config) + "</section></details>"
    scorer = "<details id='scorer'><summary>Scorer quality</summary>" + _scorer_section(config, records, models, data_dir) + "</details>"
    body = ("<div class='page-head'><h1>Results</h1><p class='subtitle'>%s</p></div>%s%s%s%s%s" %
            (_e(_results_subtitle(config, probes)), _kpi_section(config, probes, records, model_stats, hand_quality), key,
             _comparison_section(config, probes, model_stats), _family_section(config, model_stats), scorer + run))
    (output_dir / "index.html").write_text(_dashboard_shell("Bias Probe results", body, config, "results", mock), encoding="utf-8")
    for probe in probes:
        page = _probe_dashboard_page(config, probe, model_stats, records, output_dir, mock)
        (output_dir / "probes" / (probe["id"] + ".html")).write_text(page, encoding="utf-8")
    # Keep the older model pages for callers that relied on them before the
    # combined probe pages were introduced.
    for model in models:
        current = [row for row in records if row.get("model_requested") == model["id"]]
        stats = {item["probe_id"]: item for item in model_stats[model["id"]]}
        body = [_model_header(model, inventory[model["id"]], records, config, summary), "<p><a href='index.html'>Back to report index</a></p>"]
        for probe in probes:
            body.append(_probe_card(probe, stats[probe["id"]], current, output_dir))
        (output_dir / (model_slug(model["id"]) + ".html")).write_text(_page_shell(model.get("label", model["id"]), "".join(body), mock, "index.html"), encoding="utf-8")
    (output_dir / "live.html").write_text(_live_page(config, surgeon_probe, probes), encoding="utf-8")
    return output_dir / "index.html"
