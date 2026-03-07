# Remaining Workload Tracker (2026-03-05)

Canonical checkpoint for unfinished work. Originally updated to final closure state on 2026-03-06; expanded again on 2026-03-07 after deeper architecture and test-runtime audit.

## Status

- Project closure scope from the original schema/validation thread: **completed**.
- Additional backlog discovered after post-closure audit: **yes**.
- Current blocker class: **runtime integration debt and cleanup debt**, not schema closure.

## Final Completion Snapshot (2026-03-06)

- Schema gate hardened and explicit: `polylogue check --schemas`.
- Dedicated pipeline validation stage is in place (`acquire -> validate -> parse`), with persisted validation status fields.
- Acquisition-stage parsing path removed; validation is stage-exclusive.
- Runtime latest schemas promoted:
  - `chatgpt v4`
  - `claude-ai v2`
  - `claude-code v2`
  - `codex v9`
  - `gemini v2`
- Packaged baseline schemas refreshed from latest reviewed runtime schemas:
  - `polylogue/schemas/providers/*.schema.json.gz`
- Heavy-provider full verification (`samples=16`) completed with zero invalid records:
  - `claude-code`: `4643 total`, `4640 valid`, `0 invalid`, `3 decode_errors`
  - `codex`: `1013 total`, `1011 valid`, `0 invalid`, `2 decode_errors`
- Gemini full-provider gate confirmed:
  - `226/226 valid` (`--schema-samples all`).
- Record-mode validation/generation fixes landed:
  - non-record metadata payloads skipped in record-granularity validation sampling,
  - record-granularity schema root `required` relaxed for heterogeneous record corpora.
- Full regression suite:
  - `4519 passed, 1 skipped`.

## Canonical Evidence

- `docs/tasklist-master-2026-03-06.md`
- `docs/workload-closure-2026-03-06.md`
- `QA_SESSION.md`
- `qa_outputs/INDEX.md`
- `qa_outputs/schema-verification-heavy-full-2026-03-06.json`
- `qa_outputs/schema-verification-heavy-claude-code-full-2026-03-06.json`
- `qa_outputs/schema-verification-heavy-codex-full-2026-03-06.json`
- `qa_outputs/schema-verification-heavy-overhead-2026-03-06.json`
- `qa_outputs/Q22_schema_check_gemini_all.txt`
- `qa_outputs/Q22_schema_check_codex_300.txt`
- `qa_outputs/Q22_schema_check_claude_code_300.txt`

## Optional Follow-up (Non-Blocking)

1. Add explicit malformed-record quarantine/repair workflow for strict decode failures.
2. Keep periodic schema regeneration + privacy review cadence as data corpus evolves.

## Post-Closure Updates (2026-03-07)

### Newly Completed Since Closure

- Attachment materialization is now rollback-safe if bundle persistence fails.
  - Commit: `383bc49`
  - Files:
    - `polylogue/pipeline/prepare.py`
    - `tests/unit/pipeline/test_pipeline.py`
- Full suite with that fix in place:
  - `4533 passed in 359.52s (0:05:59)`

### Expanded Remaining Backlog

#### High Priority

1. Fix MCP server sync/async misuse.
   - `polylogue/mcp/server.py` is calling async repository APIs as if they were synchronous.
   - This is a runtime correctness problem, not just cleanup.
2. Fix MCP index operations to match the current `IndexService` API.
   - The server still uses an obsolete constructor/call shape for index rebuild/update.
3. Remove ambient singleton service state from runtime entry points.
   - `polylogue/services.py` remains a hidden dependency boundary and tests/showcase mutate private singleton slots directly.

#### Medium Priority Architecture / Refactor

4. Stop private backend leakage across module boundaries.
   - Current code still reaches through `_backend`, `_get_connection`, and parser/backend internals.
5. Reduce `Any`-typed orchestration surfaces in pipeline and query paths.
   - Especially `polylogue/pipeline/runner.py`, `polylogue/pipeline/services/*`, and `polylogue/cli/query.py`.
6. Remove convenience fallbacks that silently coerce invalid states instead of failing clearly.
7. Unify duplicated formatting logic.
   - Markdown/attachment rendering helpers still diverge between rendering paths.
8. Unify forced-plain-mode decision logic.
   - CLI setup and formatter layers still decide this separately.

#### Test-Suite Cleanup Debt

9. Reduce test dependence on private/runtime internals.
   - Examples include direct singleton resets and `_backend` mutation/assertions.
10. Rename stale observer/event test files and helpers to reflect current runtime semantics.
11. Consolidate repeated mocking patterns in CLI/MCP/storage tests behind shared fixtures/builders where practical.

#### Docs / Operator Clarity

12. Repair config/path documentation drift.
   - Docs and tests still reference stale path/env/config assumptions in places.

### Test Runtime Investigation (2026-03-07)

Measured with `nix develop -c pytest -q --durations=...`:

- `HEAD`: `4533 passed in 359.52s (0:05:59)`
- `51bdd27`: `4529 passed, 1 failed in 308.62s (0:05:08)`
- `d9ac0dc`: `4451 passed, 12 failed in 347.16s (0:05:47)`

Key findings:

1. The whole suite is slower than `51bdd27`, but not “many times slower”.
   - Measured delta vs fastest sampled recent baseline: `+50.90s` (`+16.5%`).
2. The slowdown is not dominated by one new catastrophic test.
   - Top 10 listed durations are nearly flat:
     - `HEAD`: `101.11s`
     - `51bdd27`: `99.51s`
3. The suite carries a stable `~45s` floor from three sqlite-vec retry/error tests.
   - These were already present in the sampled historical baselines.
4. Storage scale tests remain the biggest variable runtime bucket.
   - `tests/unit/storage/test_scale.py` dominates the slow-test list across all sampled revisions.
5. Additional runtime now appears spread across medium-cost tests and setup phases rather than one single regression point.
   - New visible contributors include `tests/unit/storage/test_fts5.py` setup/call cost, filter setup cost, integration health setup cost, parser regression checks, and CLI checks.

### Concrete Next Investigation For Test Runtime

1. Produce directory-level timings for:
   - `tests/unit/storage`
   - `tests/unit/cli`
   - `tests/unit/pipeline`
   - `tests/integration`
2. Remove real retry/backoff waiting from `tests/unit/storage/test_vec.py` by overriding the retry wait path in tests.
3. Review whether `tests/unit/storage/test_scale.py` should stay in the default suite at current corpus sizes or move partly into a dedicated performance slice.
4. Profile setup-heavy files (`test_fts5.py`, filter suites, health/integration setups) for avoidable repeated DB/bootstrap work.
5. Compare warm-cache vs cold-cache runs only if operator-reported “used to be much faster” still does not match the measured recent-history deltas above.
