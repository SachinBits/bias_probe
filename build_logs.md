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

## Phase 2 Scorer and Judge

### Scope

- Implemented `probe/scorer.py` with the rule-first pipeline: normalization, refusal detection, probe parsing, regex matching, content-label matching, uncertainty-label handling, hedge detection, and `unscorable` fallback.
- Implemented `probe/judge.py` as a fallback-only classifier at temperature 0.
- The judge receives only response text, allowed labels, and `label_help`; it never receives the original probe prompt.
- Added judge JSONL caching with `judge_prompt_version` included in the cache key.
- Added required label-record fields, including `scorer_version` and `judge_prompt_version`.
- Added 31 synthetic judge-test answers, including 10 counter-stereotype cases.
- Added CLI commands: `score`, `judge-test`, `sample-hand-labels`, and `check-scorer`.
- Extended the mock client with deterministic judge responses for offline validation.

### Important decisions

- `unknown` and `unclear` are treated as substantive content labels when present in a probe.
- Uncertainty phrases map to `unknown`/`unclear` rather than `hedge` for those probes.
- Refusals and hedges are not treated as wrong answers.
- Multiple content labels go to the judge; without a judge they become `unscorable`.
- Amounts normalize consistently: `$0.05`, `5 cents`, and `0.05` become `0.05`; `10 cents` becomes `0.10`.
- Letter answers are mapped through each variant’s option map, including answers with extra text.
- Judge cache records preserve the judge prompt and response text for auditability while excluding the original question.

### Verification

- `python3 -m unittest` → 20 tests passed.
- `python3 -m probe score --mock` → 780 responses scored per target model.
- Mock scoring distribution per model: rule `727/780 (93.2%)`, judge `14/780 (1.8%)`, unscorable `39/780 (5.0%)`.
- `python3 -m probe judge-test --mock` → rule accuracy `19/31 (61.3%)`; judge accuracy `30/31 (96.8%)`; counter-stereotype judge accuracy `9/10 (90.0%)`.
- `python3 -m probe sample-hand-labels --n 60 --mock` → wrote 60 stratified rows.
- `python3 -m probe check-scorer --mock` → correctly reported that no human labels had been filled yet.
- No real API calls were made and `data/` was not modified.

## Phase 2.5 Review Fixes and Data Fixes

### Scorer and judge fixes

- Choice probes now send unresolved responses to the judge for an option token only (`A`–`D` or `1`–`2`); the scorer maps that token through the variant’s map.
- Amount parsing now accepts `$.05`, `.05`, `$0.05`, `5 cents`, `five cents`, and `$.10`.
- Judge reply parsing now removes code fences, extracts the first JSON object, and accepts `label` with any capitalization.
- Judge requests now reuse the runner’s six-attempt retry/backoff policy. Only the final failure is written as an error record, and error records are not treated as successful cache hits.
- Judge prompts format descriptions as `label: description` lines.
- `int_1_10` parsing ignores scale/range notation such as `1 to 10`, `1-10`, `out of 10`, and `/10`.
- Judge-test fixtures now all carry real `probe_id` values and use real probe labels and `label_help`; the fixture contains 36 cases, including 11 counter-stereotype cases and the new amount, pairwise-choice, surgeon-conflict, BBQ, and WinoBias examples.

### Data and cache fixes

- Runner cache completion now requires an `ok` record matching `(key, probe_id, variant_id)`, while the cache-key formula remains unchanged.
- This prevents two probes with identical prompts from incorrectly sharing one cached response.
- Label records retain the full response identity fields: `key`, `probe_id`, `variant_id`, `repeat_index`, and `temperature`.
- `hand_labels.csv` now includes `repeat_index` and `temperature` alongside the probe and variant identifiers.
- Scoring accepts only response records matching the current model, provider, scheduled temperature, and max-token configuration; mismatches are counted as `superseded`.
- Set the Llama 3.1 8B model provider in `config.json` to `["deepinfra"]`; GPT-4o-mini configuration was unchanged.
- Existing `data/responses/` content and pre-existing bank edits were preserved and not rewritten.

### Verification

- `python3 -m unittest` → 26 tests passed.
- `python3 -m probe run --mock` → 780 new provider-specific records written and 780 existing tasks skipped.
- `python3 -m probe score --mock` → current records scored; 780 superseded records reported for the prior Llama provider configuration.
- `python3 -m probe judge-test --mock` → rule accuracy `22/36 (61.1%)`; judge accuracy `33/36 (91.7%)`; counter-stereotype judge accuracy `10/11 (90.9%)`.
- `python3 -m probe sample-hand-labels --n 60 --mock` → hand-label identity includes key, probe, variant, repeat, and temperature fields.
- No real API calls were made and `data/responses/` was not modified.
