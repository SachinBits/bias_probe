# Codex prompts — paste one phase at a time

Setup (once): create an empty folder `bias-probe/`, put `AGENTS.md`, `PROMPTS.md` and the `probes/` folder from this kit inside it, run `git init && git add -A && git commit -m "spec + probe bank"`, then open Codex in that folder.

---

## Phase 1 — Runner (collect data first)

```
Read AGENTS.md fully. Implement Phase 1 only:
- Project skeleton exactly as in "Project structure" (empty stubs are fine for scorer/judge/stats/report).
- config.json, .env.example, .gitignore, probes/custom.json (empty bank).
- probe/config.py, bank.py (loader + validator for all three probe files), client.py (OpenRouterClient + MockClient), store.py, runner.py, cli.py with commands: ping, run (--models, --probes, --mock, --dry-run, --data-dir).
- Retries, resume via cache key, spend cap and error records exactly as specified.
- tests/ for the bank validator, cache key stability, resume (second mock run = 0 new calls), and retry on simulated 429.
Done when: `python3 -m unittest` passes, `python3 -m probe run --mock` writes data_mock/responses/*.jsonl, a second run makes 0 new calls, and `python3 -m probe run --dry-run` prints the planned call count. Do not call the real API. Commit when done. End with: files changed, commands + results, deviations from spec.
```

**Then YOU run (real API, on your Mac):**
```bash
cp .env.example .env        # paste the key into .env
python3 -m probe ping
python3 -m probe run --dry-run
python3 -m probe run --models openai/gpt-4o-mini-2024-07-18
```
Send Claude: Codex's final message + the output of these commands + the first 3 lines of `data/responses/*.jsonl`.

---

## Phase 2 — Scorer + judge

```
Read AGENTS.md. Implement Phase 2 only: probe/scorer.py, probe/judge.py, judge_tests/judge_tests.json (~30 items incl. counter-stereotype answers), and CLI commands: score, judge-test, sample-hand-labels, check-scorer (all honour --mock / --data-dir).
The judge must never receive the original question. Judge calls are cached in data/judge/<slug>.jsonl.
Add unit tests for every scorer case listed under "Tests" in AGENTS.md, plus: "Mother." → mother; "The answer is B" with map → mapped label; "$0.05" / "5 cents" / "0.05" → 0.05; "It could be either parent." → hedge; quoted riddle text is not a refusal.
Done when unittest passes and `python3 -m probe score --mock` and `python3 -m probe judge-test --mock` run. Commit. End with files changed, commands + results, deviations, and the share of mock responses labelled by rule / judge / unscorable.
```

**Then YOU run:** `python3 -m probe score` and `python3 -m probe judge-test`. Send Claude the outputs, plus 20 lines from `data/labels/*.jsonl` where source is judge or label is unscorable.

---

## Phase 3 — Stats + report

```
Read AGENTS.md. Implement Phase 3 only: probe/stats.py and probe/report.py exactly as in "Stats and verdicts" and "Report", plus CLI commands report and all.
Implement Fisher exact (one-sided) with math.comb, permutation tests with the configured seed, Benjamini–Hochberg q-values, movement = D_between − D_within, pattern-capture and stereotype rates, minimum detectable effect, and the verdict rules word for word.
The report must be fully static, offline, with every raw response in <details>, the surgeon original-vs-inverted headline, family summary, model comparison table, scorer-quality section, and a red MOCK banner on mock data.
Tests: permutation p≈1 for identical inputs, small for 10/10 vs 0/10; Fisher 10/10 vs 3/10 one-sided ≈ 0.0015; verdict function on hand-made examples for each verdict.
Done when unittest passes and `python3 -m probe all --mock` produces data_mock reports. Commit. End with files changed, commands + results, deviations.
```

**Then YOU run:** `python3 -m probe report` and open `reports/index.html`. Send Claude a screenshot or the HTML file.

---

## Phase 4 — Second model, add-probe, docs, packaging

```
Read AGENTS.md. Implement Phase 4 only:
- CLI `add` (interactive, writes probes/custom.json, prints the command to run just that probe).
- README.md: what it is, setup (Python 3.9+, no installs), .env, every command, how to add a probe pair and run only it, how to reproduce the report from the committed data without spending, the exact models/snapshots tested.
- SCORING_POLICY.md skeleton with sections: definitions of biased/wrong/noisy per family, why 10 repeats at T=0.7 + 3 at T=0, minimum detectable effect, judge design and validation, what the numbers do not show, what we would do with a week. Leave TODO markers where real results go.
- scripts/make_zip.py: builds <id>_challenge-4.zip with code, probes, data/, reports/, docs, transcripts/ and slides/ if present; MUST exclude .env, data_mock/, __pycache__, .git; fail loudly if any file contains a string matching sk-or-v1-.
Done when unittest passes, `python3 -m probe all --mock` still works, and make_zip.py produces a zip with no key in it. Commit. End with files changed, commands + results, deviations.
```

**Then YOU run:** `python3 -m probe run` (second model, from cache for anything already done) → `python3 -m probe all` → send Claude the report + README for final review.
