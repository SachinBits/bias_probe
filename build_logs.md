# Build Logs

This file is append-only. Each completed implementation phase and review update is recorded below; existing entries must not be overwritten.

## Phase 1 Initial Implementation

### Scope

- Added the Phase 1 project skeleton from `AGENTS.md`.
- Added `config.json`, `.env.example`, `.gitignore`, and empty `probes/custom.json`.
- Implemented probe loading and validation for `probes/bank.json`, `probes/starter.json`, and `probes/custom.json`.
- Normalized `stereotype` values from either a string or a variant mapping into a dictionary.
- Implemented the OpenRouter client and deterministic offline mock client.
- Implemented append-only JSONL storage, cache keys, task scheduling, retries, resume behavior, error records, concurrency, and spend-cap handling.
- Added Phase 1 CLI commands: `ping` and `run`, including `--mock`, `--models`, `--probes`, `--dry-run`, and `--data-dir`.
- Added placeholder modules for the future scorer, judge, statistics, and report phases.

### Important decisions

- Python 3.9+ standard library only; no third-party dependencies.
- Real API keys are read from the environment or `.env`; keys are never hard-coded.
- Mock runs write to `data_mock/`, never to the real `data/` directory.
- Raw response files are append-only.
- Cache keys include model, provider, prompt, temperature, max tokens, and repeat index, but not schema versions.
- Every response record includes `schema_version: 1`.
- The runner records both requested and returned model/provider metadata.

### Verification

- `python3 -m unittest -v` passed the initial Phase 1 test suite.
- `python3 -m probe run --mock` wrote 1,560 response records across two model files.
- A second mock run wrote zero new calls and skipped all cached tasks.
- Dry-run reported 1,560 planned tasks.
- No real API calls were made.

### Initial commit

- Commit `448f205` — `Implement phase 1 probe runner`

## Phase 1 Review Improvements

### Reliability fixes

- OpenRouter HTTP 200 responses with an `error` field, missing choices, or empty content without a finish reason now raise retryable `ClientError` exceptions.
- Response parsing, JSON decoding, connection, HTTP, and incomplete-read failures are wrapped as retryable client errors.
- HTTP error messages include up to 300 characters of response body and redact the API key.
- Unexpected exceptions from a client call are captured as error records so one failed task does not stop the run.
- Error records now report the actual number of attempts made.
- Truncated or malformed JSONL lines are skipped with one warning per response file instead of crashing resume logic.
- `ping` now checks the configured judge model as well as the configured target models and prints returned model/provider or the full error.
- Timestamps now use timezone-aware UTC via `datetime.now(timezone.utc)`.

### Mock client improvements

- `MockClient` now receives the loaded probe bank and matches each prompt to its probe and variant.
- Mock responses are deterministic for a given seed, prompt, temperature, and repeat index.
- Responses include realistic formats such as bare labels, explanatory text, Markdown emphasis, mapped letter answers, monetary values, hedges, refusals, and simulated rate limits.
- Normal probes are approximately 75% correct; trap variants include an explicit familiar-pattern capture path.
- Integer rating probes return values in the 6–9 range with small variant-specific shifts.
- Tests can disable simulated failures when deterministic behavior is required.

### Tests added

- HTTP response validation tests for API errors, missing choices, and empty content.
- Client parsing and HTTP error tests, including API-key redaction.
- Unexpected client exception and real attempt-count tests.
- Truncated JSONL recovery and warning tests.
- Judge inclusion in `ping` output tests.
- Existing tests continue to cover bank validation, stereotype normalization, cache stability, retry behavior, crash recovery, and resume behavior.

### Review verification

- `python3 -m unittest -v` passed all 11 tests.
- Fresh `python3 -m probe run --mock` wrote 1,560 records.
- A second mock run wrote zero new calls and skipped 1,560 records.
- No real API calls were made.

### Review commits

- Commit `47aa173` — `Fix phase 1 runner reliability`
- Commit `ecc405e` — `Tune deterministic mock probe answers`
