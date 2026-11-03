"""Rule-first scoring for free-text probe responses."""
from __future__ import annotations

import re

SCORER_VERSION = "1"
JUDGE_PROMPT_VERSION = "2"

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


def _remove_negated_mentions(text, labels):
    masked = text
    prefixes = r"(?:(?:the|his|her|a)\s+)?"
    for label in labels:
        escaped = re.escape(str(label).lower())
        patterns = (
            r"\bnot\s+" + prefixes + escaped + r"\b",
            r"\bisn't\s+" + prefixes + escaped + r"\b",
            r"\brather\s+than\s+" + prefixes + escaped + r"\b",
            r"\binstead\s+of\s+" + prefixes + escaped + r"\b",
            r"\bno,\s*" + prefixes + escaped + r"\b",
        )
        for pattern in patterns:
            masked = re.sub(pattern, " ", masked, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", masked).strip()


def _has_answer_is_contradiction(text, matched_label, labels):
    for label in labels:
        if str(label).lower() == str(matched_label).lower():
            continue
        if re.search(r"\banswer\s+is\s+(?:the\s+)?%s\b" % re.escape(str(label).lower()), text):
            return True
    return False


def contains_unnegated_label(response_text, label, probe):
    text = _remove_negated_mentions(normalize(response_text), _content_labels(probe))
    return bool(re.search(r"(?<!\w)%s(?!\w)" % re.escape(str(label).lower()), text))


def _parse_amount(text):
    word_cents = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
    cents = re.search(r"\b(\d+(?:\.\d{1,2})?|one|two|three|four|five|ten)\s*cents?\b", text)
    if cents:
        raw = cents.group(1)
        value = word_cents[raw] if raw in word_cents else float(raw)
        return "%.2f" % (value / 100.0)
    money = re.search(r"\$\s*(?:(\d+(?:\.\d{1,2})?)|(\.\d{1,2}))\b", text)
    if money:
        value = money.group(1) or ("0" + money.group(2))
        return "%.2f" % float(value)
    decimal = re.search(r"(?<![\w.])(?:([0-9]+\.[0-9]{1,2})|(\.[0-9]{1,2}))(?![\w.])", text)
    if decimal:
        value = decimal.group(1) or ("0" + decimal.group(2))
        return "%.2f" % float(value)
    return None


def _parse_choice(text, parse):
    if parse == "letter":
        match = re.search(r"(?:answer\s+(?:is|:)|^|\b)([a-d])(?:\s*[).:]|\s*$)", text)
    else:
        match = re.fullmatch(r"(?:choice\s+|answer\s+(?:is|:)\s*)?([12])[.!]?", text)
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
        masked = re.sub(r"\b1\s*(?:to|-|–)\s*10\b|\bout\s+of\s+10\b|/\s*10\b", " ", text)
        match = re.search(r"(?<![\w.])(?:[1-9]|10)(?![\w.])", masked)
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
        matched = next(iter(matches))
        if _has_answer_is_contradiction(text, matched, _content_labels(probe)):
            return {"label": None, "source": "none", "detail": "strong regex hit contradicted by explicit answer"}
        return _result(matched, "matched probe regex")
    if len(matches) > 1:
        return {"label": None, "source": "none", "detail": "multiple probe regex labels matched"}

    content_text = _remove_negated_mentions(text, _content_labels(probe))
    content_hits = set()
    for label in _content_labels(probe):
        if re.search(r"(?<!\w)%s(?!\w)" % re.escape(label.lower()), content_text):
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
        choice_mode = probe.get("parse") if probe.get("parse") in {"letter", "choice_number"} else None
        label, detail = judge.classify(response_text, probe.get("labels", []), probe.get("label_help", ""), choice_mode=choice_mode)
        if choice_mode and label not in {"hedge", "refused", "unscorable"}:
            mapping = variant.get("map", {})
            mapped = next((str(value) for key, value in mapping.items() if str(key).upper() == str(label).upper()), None)
            label = mapped if mapped in [str(x) for x in probe.get("labels", [])] else "other"
            detail = detail + "; mapped option token"
        return {"label": label, "source": "judge" if label != "unscorable" else "none", "detail": detail}
    return {"label": "unscorable", "source": "none", "detail": result.get("detail", "unscorable")}
