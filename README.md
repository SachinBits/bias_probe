# Bias Probe

Bias Probe measures whether an LLM changes its answer when a prompt is changed.
It runs paired probes repeatedly, scores free-text answers, uses a fallback
judge only when rules cannot decide, and produces an offline HTML report. The
results are probe-level signals for a model snapshot, temperature, and wording;
they are not proof of universal bias.

## Setup

Requirements: Python 3.9 or newer and the standard library only. No packages
need to be installed.

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
python3 -m unittest
```

The `.env` file is ignored by Git. Never paste a real key into source, reports,
transcripts, or an archive.

## Models

- `openai/gpt-4o-mini-2024-07-18`, served by OpenAI.
- `meta-llama/llama-3.1-8b-instruct`, pinned to the DeepInfra provider.
- Fallback judge: `meta-llama/llama-3.3-70b-instruct`, pinned to DeepInfra.

## Commands

All commands support `--mock` where applicable for offline testing. Mock data is
stored under `data_mock/`, never in `data/`.

```bash
python3 -m probe ping
python3 -m probe run --dry-run
python3 -m probe run [--models MODEL[,MODEL]] [--probes ID[,ID]]
python3 -m probe run --mock
python3 -m probe score [--mock] [--data-dir PATH]
python3 -m probe judge-test --mock
python3 -m probe sample-hand-labels --n 60 --mock
python3 -m probe check-scorer --mock
python3 -m probe report [--mock] [--data-dir PATH]
python3 -m probe all --mock
python3 -m probe add
python3 -m probe add --file pair.json
python3 scripts/make_zip.py YOUR_ID
```

`run` uses the append-only response cache. A second run skips successful task
records and only retries failed tasks. `score` rebuilds inexpensive label files
while reusing cached judge calls. `report` is static HTML with no CDN or API
dependency.

## Reproduce the report without spending

With response data already present, these commands make no new model calls for
successful cached responses:

```bash
python3 -m probe score
python3 -m probe report
```

Fallback judge calls are cached separately, so an already-judged response is not
charged again. To reproduce everything offline, use `--mock` and `data_mock/`.

## Add a probe pair

Interactive mode asks for the id, family, expectation, labels, parse mode, and
variants. Each variant needs a prompt and may have an expected answer:

```bash
python3 -m probe add
```

For a prepared single-probe JSON file, use:

```bash
python3 -m probe add --file pair.json
```

The probe is validated and appended to `probes/custom.json`. Run only that
probe with:

```bash
python3 -m probe all --probes YOUR_PROBE_ID --mock
```

## The 401 key incident

An earlier API run returned HTTP 401 errors because the configured key was
invalid or misconfigured. Those errors were written as `status: "error"` and
were never treated as successful cache hits. After correcting the key, reruns
retry the failed tasks while successful cached tasks remain skipped, preventing
unnecessary spending. The error records do not expose the key.

## Documentation

- [AGENTS.md](AGENTS.md) — complete implementation specification.
- [PROMPTS.md](docs/PROMPTS.md) — phase prompts.
- [SCORING_POLICY.md](docs/SCORING_POLICY.md) — scoring definitions and limits.
- [build_logs.md](docs/build_logs.md) — append-only implementation history.
