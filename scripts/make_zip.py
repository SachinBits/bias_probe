#!/usr/bin/env python3
"""Build the submission archive without secrets or local-only artifacts."""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path


SECRET_RE = re.compile(r"sk-or-v1-[0-9a-fA-F]{20,}")


def _files(root):
    paths = []
    for directory in ("probe", "probes", "judge_tests", "tests", "reports", "docs", "scripts", "transcripts", "slides"):
        base = root / directory
        if base.exists():
            paths.extend(path for path in base.rglob("*") if path.is_file())
    for name in ("README.md", "AGENTS.md", ".env.example", "config.json"):
        path = root / name
        if path.exists():
            paths.append(path)
    data = root / "data"
    if data.exists():
        wanted = [data / "responses", data / "labels", data / "judge", data / "judge_test_results.json",
                  data / "hand_labels.csv", data / "hand_overrides.csv"]
        for path in wanted:
            if path.is_dir():
                paths.extend(child for child in path.rglob("*") if child.is_file())
            elif path.is_file():
                paths.append(path)
    filtered = []
    for path in sorted(set(paths)):
        relative = path.relative_to(root)
        if ("__pycache__" in relative.parts or ".git" in relative.parts or
                path.name.startswith("~$") or path.suffix.lower() == ".docx" or
                path.name == ".env" or "data_mock" in relative.parts):
            continue
        if "data_live" in relative.parts:
            continue
        filtered.append(path)
    return filtered


def make_zip(identifier, root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    output = root / (identifier + "_challenge-4.zip")
    files = _files(root)
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        match = SECRET_RE.search(text)
        if match:
            raise SystemExit("Refusing to package secret-looking key in %s" % path.relative_to(root))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    print("Wrote %s (%d files)" % (output, len(files)))
    return output


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or not argv[0].strip():
        raise SystemExit("usage: python3 scripts/make_zip.py <id>")
    make_zip(argv[0])


if __name__ == "__main__":
    main()
