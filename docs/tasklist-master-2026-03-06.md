# Polylogue Master Tasklist (2026-03-06)

Status: historical closure checkpoint, not the live execution queue
Role: compact archaeology snapshot for the March 5-7 closure wave

Current execution entrypoint:

- `planning-and-analysis-map-2026-03-21.md`
- `intentional-forward-program-2026-03-21.md`

Compact checkpoint for context-compaction recovery.

## Closed / Completed

- [x] Enforced explicit schema verification gate: `polylogue check --schemas`.
- [x] Added chunkable schema verification controls (`--schema-record-limit`, `--schema-record-offset`).
- [x] Hardened strict malformed-JSONL handling.
- [x] Hardened enum privacy filtering in schema generator.
- [x] Added structural-dedup heuristics for schema generation.
- [x] Cleared validator cache in test fixtures to avoid test-order leakage.
- [x] Removed acquisition-stage parsing path; validation is a dedicated stage.
- [x] Promoted runtime registry-latest schemas (`chatgpt v4`, `claude-ai v2`, `claude-code v2`, `codex v9`, `gemini v2`).
- [x] Promoted runtime-latest schemas into packaged baselines (`polylogue/schemas/providers/*.schema.json.gz`).
- [x] Restored Gemini operability (`226/226` valid with `--schemas --schema-samples all`).
- [x] Completed heavy-provider full verification (`samples=16`) with zero invalid records:
  - `claude-code`: `4643 total`, `4640 valid`, `0 invalid`, `3 decode_errors`
  - `codex`: `1013 total`, `1011 valid`, `0 invalid`, `2 decode_errors`
- [x] Added record-mode sampling fix: non-record documents are skipped in record-granularity validation.
- [x] Added record-mode root-required relaxation in schema generation to handle heterogeneous record streams.
- [x] Captured post-closure provider checks:
  - `gemini` all-record check: `226/226 valid`
  - `codex` sample check (`records=300`): `300/300 valid`
  - `claude-code` sample check (`records=300`): `300/300 valid`
- [x] Full regression suite green: `4519 passed, 1 skipped`.

## Open / Remaining

- [ ] No blocking correctness or QA items remain from this tasklist.
- [ ] Optional follow-up: normalize/quarantine malformed raw JSONL records causing decode-only residuals.

## Canonical Evidence

- Workload closure: `docs/workload-closure-2026-03-06.md`
- Remaining tracker: `docs/remaining-workload-tracker-2026-03-05.md`
- QA narrative: `QA_SESSION.md`
- QA artifact index: `qa_outputs/INDEX.md`
- Heavy full verification: `qa_outputs/schema-verification-heavy-full-2026-03-06.json`
- Heavy overhead probes: `qa_outputs/schema-verification-heavy-overhead-2026-03-06.json`
- Post-closure schema checks:
  - `qa_outputs/Q22_schema_check_gemini_all.txt`
  - `qa_outputs/Q22_schema_check_codex_300.txt`
  - `qa_outputs/Q22_schema_check_claude_code_300.txt`
