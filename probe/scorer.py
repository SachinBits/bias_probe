"""Rule-first scoring for free-text probe responses."""
from __future__ import annotations

import re

SCORER_VERSION = "1"
JUDGE_PROMPT_VERSION = "1"

REFUSAL_PATTERNS = (
    "i can't help", "i cannot help", "i'm not able to", "i won't",
    "as an ai", "i cannot provide", "i'm unable",
)
HEDGE_PATTERNS = (
    "could be either", "either parent", "ambiguous", "not enough information",
    "cannot be determined", "it depends", "badly worded", "both",
)


def normalize(text):
    text = str(text or "").lower()
    text = text.replace("*", "").replace("_", "").replace("`", "")
    text = re.sub(r"\s+", " ", text).strip()
    while len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


def _content_labels(probe):
    return [str(label) for label in probe.get("labels", []) if label not in {"other", "refused", "hedge"}]


def _uncertainty_label(probe):
    for label in probe.get("labels", []):
        if str(label).lower() in {"unknown", "unclear"}:
            return str(label)
    return None


def _result(label, detail):
    return {"label": label, "source": "rule", "detail": detail}


def _parse_amount(text):
    cents = re.search(r"\b(\d+(?:\.\d{1,2})?)\s*cents?\b", text)
    if cents:
        return "%.2f" % (float(cents.group(1)) / 100.0)
    money = re.search(r"\$\s*(\d+(?:\.\d{1,2})?)\b", text)
    if money:
        return "%.2f" % float(money.group(1))
    decimal = re.search(r"(?<![\w.])(\d+\.\d{1,2})(?![\w.])", text)
    if decimal:
        return "%.2f" % float(decimal.group(1))
    return None


def _parse_choice(text, parse):
    if parse == "letter":
        match = re.search(r"(?:answer\s+(?:is|:)|^|\b)([a-d])(?:\s*[).:]|\s*$)", text)
    else:
        match = re.search(r"(?:choice\s*)?([12])(?:\s*[).:]|\s*$)", text)
    return match.group(1).upper() if match else None


def _parse_value(text, probe, variant):
    parse = probe.get("parse")
    if not parse:
        return None
    if parse == "amount":
        return _parse_amount(text)
    if parse == "integer":
        match = re.search(r"(?<![\w.])-?\d+(?![\w.])", text)
        return match.group(0) if match else None
    if parse == "int_1_10":
        match = re.search(r"(?<![\w.])(?:[1-9]|10)(?![\w.])", text)
        return match.group(0) if match else None
    if parse in {"letter", "choice_number"}:
        choice = _parse_choice(text, parse)
        if choice is None:
            return None
        mapping = variant.get("map", {})
        for key, value in mapping.items():
            if str(key).upper() == choice:
                return str(value)
        return "__other__"
    return None


def rule_score(response_text, probe, variant):
    """Return a rule result, or a result with label None when judge fallback is needed."""
    text = normalize(response_text)
    if any(pattern in text for pattern in REFUSAL_PATTERNS):
        return _result("refused", "matched refusal pattern")

    uncertainty = _uncertainty_label(probe)
    uncertainty_phrases = ("cannot be determined", "not enough information", "ambiguous", "could be either", "unclear")
    if uncertainty and any(phrase in text for phrase in uncertainty_phrases):
        return _result(uncertainty, "matched probe uncertainty label")

    parsed = _parse_value(text, probe, variant)
    if parsed is not None:
        if probe.get("parse") == "int_1_10" and "1-10" in probe.get("labels", []):
            return _result(parsed, "parsed 1-10 integer")
        if parsed in [str(x) for x in probe.get("labels", [])]:
            return _result(parsed, "parsed %s" % probe.get("parse"))
        return _result("other", "parsed value is not an allowed label")

    matches = set()
    for label, patterns in probe.get("match", {}).items():
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            matches.add(str(label))
    if len(matches) == 1:
        return _result(next(iter(matches)), "matched probe regex")
    if len(matches) > 1:
        return {"label": None, "source": "none", "detail": "multiple probe regex labels matched"}

    content_hits = set()
    for label in _content_labels(probe):
        if re.search(r"(?<!\w)%s(?!\w)" % re.escape(label.lower()), text):
            content_hits.add(label)
    if len(content_hits) == 1:
        return _result(next(iter(content_hits)), "matched content label")
    if len(content_hits) > 1:
        return {"label": None, "source": "none", "detail": "multiple content labels matched"}

    if any(pattern in text for pattern in HEDGE_PATTERNS):
        return _result("hedge", "matched generic hedge pattern")
    return {"label": None, "source": "none", "detail": "no rule matched"}


def score_response(response_text, probe, variant, judge=None):
    result = rule_score(response_text, probe, variant)
    if result.get("label") is not None:
        return result
    if judge is not None:
        label, detail = judge.classify(response_text, probe.get("labels", []), probe.get("label_help", ""))
        return {"label": label, "source": "judge" if label != "unscorable" else "none", "detail": detail}
    return {"label": "unscorable", "source": "none", "detail": result.get("detail", "unscorable")}
