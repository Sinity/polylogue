# Polylogue Workload + Schema Deep Dive (2026-03-05)

> Status update (2026-03-06): This is a historical analysis snapshot.
> Current authoritative closure/tracker docs are:
> - `docs/archive/2026-03-05-07-closure-wave/tasklist-master-2026-03-06.md`
> - `docs/archive/2026-03-05-07-closure-wave/workload-closure-2026-03-06.md`
> - `docs/archive/2026-03-05-07-closure-wave/remaining-workload-tracker-2026-03-05.md`

## Scope

This report captures:
- Current repository state and recent commit baseline.
- Open/remaining workload (session `1a05a1d0-5f6e-47b8-aa10-b8cd2d02da43` plus older polylogue backlog).
- Deep schema design review against real raw datasets.
- QA artifact organization status.

Canonical remaining-workload tracker:
- `docs/archive/2026-03-05-07-closure-wave/remaining-workload-tracker-2026-03-05.md`

No schema files were modified in this pass.

## Current Repository State

- Branch: `master`
- Tracking: `origin/master`
- Ahead: `8` commits (`origin/master...HEAD` = `0 8`)
- Head commit: `d10ffdc` (`fix: schema migration v16, MCP client limits, provider_meta bloat`)
- Git stash: empty
- Working tree: dirty (45 entries; many pre-existing modified tests/files plus QA/docs artifacts)

Recent commits:
1. `d10ffdc` fix: schema migration v16, MCP client limits, provider_meta bloat
2. `f434d76` fix: subagent collision, --stats perf, display cap removal
3. `d9ac0dc` perf: batch DB lookups, pre-computed sort_key, orjson + caching
4. `324b556` perf: incremental pipeline — parse tracking, mtime skip, Drive integration
5. `73913ea` fix: progress display — avoid double-counting in plain mode

## Task 22 / Task 21 Progress In This Pass

Completed now (residual Task 22 hardening):
- Replaced private-field coupling in filter-state tests with behavior assertions.
  - `tests/unit/pipeline/test_concurrency_guards.py`
- Removed internal ID-format coupling in branching test; now asserts actual parent linkage.
  - `tests/unit/pipeline/test_branching.py`
- Relaxed provider-set brittleness (allow additive providers).
  - `tests/unit/core/test_schema.py`
- Added real temp-DB MCP tool-path integration tests (search/list + invalid limit) to reduce synthetic bypass.
  - `tests/integration/test_mcp.py`
- Reduced tool-inventory test redundancy by scoping mutation inventory assertion.
  - `tests/integration/test_mcp_mutations.py`

Validation run:
- `pytest -q tests/unit/pipeline/test_concurrency_guards.py tests/unit/pipeline/test_branching.py tests/unit/core/test_schema.py tests/integration/test_mcp.py tests/integration/test_mcp_mutations.py`
- Result: `165 passed`

Task 21 (stale/legacy audit) status update:
- Key stale/incoherent schema-path design issues are identified below under “Schema Findings” and “Remaining Workload”.

## Open Workload (Authoritative)

From `~/.config/claude/tasks`:

### Session `1a05a1d0-5f6e-47b8-aa10-b8cd2d02da43`
- `15` (`pending`): further performance optimization pass.
- `21` (`in_progress`): stale/legacy code audit+cleanup.
- `22` (`pending` in task store): deep test-suite audit/remediation. Substantially advanced in repo/docs; task store not reconciled.
- `23` (`in_progress`): comprehensive manual QA completion and reconciliation.

### Older polylogue backlog still open (triaged as unresolved)
- `1f4c94be-...`: mutmut config, test infra consolidation, fuzzing harnesses, E2E workflows.
- `2aba5e29-...`: async migration completion (services/CLI/MCP), removal of old sync layers, migration tests.
- `adff8038-...`: test-suite file consolidation cleanup.
- `e5bd7b01-...`: config/tests phase for embedding config work.
- `e8b61976-...`: polish/docs, semantic API validation on real data, optimizations.

## Schema Deep Dive

## Real Data Snapshot (from live DB + source files)

Source roots:
- `inbox`: `/home/sinity/.local/share/polylogue/inbox`
- `claude-code`: `/home/sinity/.claude/projects`
- `codex`: `/home/sinity/.codex/sessions`
- `gemini`: Drive folder (`Google AI Studio`) cached under `/home/sinity/.local/share/polylogue/drive-cache/gemini`

DB counts:
- `raw_conversations`: `8821`
- `conversations`: `4809`
- providers (raw):
  - `claude-code 4643`
  - `chatgpt 2122`
  - `codex 931`
  - `claude 904`
  - `gemini 221`

Raw size profile (`raw_conversations.raw_content`):
- `claude-code`: avg ~1.55 MiB, max 170,551,677 bytes
- `chatgpt`: avg ~130.8 KiB, max 8,497,391 bytes
- `codex`: avg ~4.36 MiB, max 540,617,494 bytes
- `claude`: avg ~97.8 KiB, max 1,537,257 bytes
- `gemini`: avg ~699.9 KiB, max 27,120,305 bytes

Observed raw structural shapes (40-file sample/provider):
- `chatgpt`: JSON arrays of conversation objects; sampled keys: `id`, `conversation_id`, `title`, `is_anonymous`
- `claude`: JSON arrays; sampled keys include `uuid`, `name`, `summary`, `created_at`, `updated_at`, `account`, `chat_messages`
- `claude-code`: mostly JSONL session logs (`type`, `sessionId`, `timestamp`, `parentUuid`, etc.); one outlier JSON object (`runSettings/systemInstruction/chunkedPrompt`)
- `codex`: JSONL envelopes with keys `timestamp`, `type`, `payload`
- `gemini`: JSON objects with top-level `runSettings`, `systemInstruction`, `chunkedPrompt`

## Design Strengths

- Raw-to-parsed lineage exists and is explicit:
  - `conversations.raw_id -> raw_conversations.raw_id` FK.
- Storage schema has meaningful relational guarantees:
  - parent conversation FK, cascade deletion on messages/refs.
  - branch-type CHECK includes `subagent` (v16+).
- Pipeline stores raw first, parses later, and tracks parse status (`parsed_at`, `parse_error`).
- Schema generation includes privacy-aware annotation filters for enum-like fields (`x-polylogue-*`) and dynamic key collapsing.

## Critical Gaps (Guarantees vs Reality)

1. Validation provider alias mismatch (`claude` vs `claude-ai`)
- Runtime validation uses `SchemaValidator.for_provider(raw_record.provider_name)`.
- Raw/provider name is `claude`; schema filename is `claude-ai`.
- Result: no schema validation for Claude AI records (validation silently skipped).

2. Gemini schema models message chunks, but runtime validates full conversation documents
- Current gemini schema requires top-level `role`.
- Real gemini raw files are conversation objects with `chunkedPrompt.chunks`.
- Validation result in sample check: `120/120 invalid` (`'role' is required`).
- Guarantee impact: schema signal for gemini is effectively unusable/noisy.

3. Codex schema drift already present in live data
- Sample check: `76/120` codex samples invalid.
- Main observed mismatch: `payload.source` now object-shaped (subagent metadata), schema expects string.
- Guarantee impact: stale schema under-represents current codex payload variants.

4. Strict drift detection semantics produce false positives for dynamic-key maps
- ChatGPT drift warnings fire on `mapping.<uuid>` keys.
- This is expected dynamic structure, not semantic drift.
- Guarantee impact: drift counters are noisy and may trigger unnecessary regeneration.

5. Validation only samples first dict for list payloads (JSONL)
- For list payloads, only first dict entry is validated.
- Many structural variants later in streams are unchecked.
- Guarantee impact: limited coverage for JSONL-heavy providers (`claude-code`, `codex`).

6. Auto-regenerated versioned schemas are not consumed by runtime validator
- Parsing service can auto-register drifted schemas via `SchemaRegistry` into data-home versions.
- `SchemaValidator` reads baseline packaged schemas only (`SCHEMA_DIR`).
- Guarantee impact: regeneration has little/no runtime enforcement value today.

7. All provider schemas effectively allow additional top-level fields
- `additionalProperties` is not explicitly constrained at root in packaged schemas.
- Combined with drift logic, this weakens strict-mode guarantees for unknown fields.

## Validation Sampling Results (120 recent raw records/provider)

- `chatgpt`: `120 valid`, `0 invalid`, `50 drift` (mostly dynamic mapping-key warnings)
- `claude`: validator unavailable due alias mismatch (`no schema loaded`)
- `claude-code`: `119 valid`, `1 invalid` (outlier non-claude-code-shaped record)
- `codex`: `44 valid`, `76 invalid`
- `gemini`: `0 valid`, `120 invalid`

## Overall Assessment

The schema subsystem currently provides:
- Good metadata and generation machinery,
- Weak runtime guarantees for real-data conformance because schema identity and schema target-shape are misaligned for key providers (`claude`, `gemini`, `codex`), and validation scope is shallow for JSONL bundles.

## QA Organization Cleanup

Actions performed:
- Created artifact index with canonical/interrupted/superseded markers:
  - `qa_outputs/INDEX.md`
- Kept all existing artifacts intact (no destructive cleanup).

Canonical notes:
- `Q08_run_preview.txt` interrupted; `Q08b_preview_gemini.txt` canonical.
- `Q12_stats.txt` interrupted (`143`); `Q12b_stats_rerun.txt` canonical.
- `Q21b_parse_after_migration_fix.txt` interrupted; `Q21c` and `Q21d` canonical post-v17 proof.

## Remaining Workload (Meticulous Execution Plan)

1. Reconcile task store state with actual progress
- Mark `Task 22` as completed/in-progress-advanced with current remediation evidence.
- Keep `Task 21` and `Task 23` open.

2. Finish Task 21 with concrete stale-code remediations (non-schema-file changes first)
- Add provider alias normalization for schema validation (`claude` -> `claude-ai`) in validation path.
- Align gemini validation target with real raw shape (validate chunk items or schema wrapper).
- Decide codex `payload.source` contract (string vs object union) and update runtime expectations.
- Rework drift detection for dynamic-key `additionalProperties` objects to avoid UUID-map false positives.
- Ensure versioned schema registry output is actually selectable/used by validator (or remove dead path).

3. Finish Task 23 QA completion/reconciliation
- Keep current canonical files, archive/label interrupted ones.
- Ensure `QA_SESSION.md` + `qa_outputs/INDEX.md` remain in sync.
- Run a final focused CLI smoke pass after any schema-validation-path changes.

4. Task 15 performance follow-up (after correctness debt above)
- Re-profile parse/index hotspots on same dataset.
- Verify no regressions in acquisition skip logic and parse throughput.

5. Backlog triage closure
- Decide which legacy tasks are superseded vs still actionable and produce one canonical backlog list.
