## Summary

PR 9/9. Stacked on `feature/refactor/stack-08-final-unification`.

Trailing runtime semantic fixes discovered after the substrate wave landed. Three fixes that share one semantic unit: raw records have two distinct failure modes (decode vs validation), and every writer/reader honors the split.

## Problem

Three tangled defects surfaced when the live archive was revalidated against the new semantics:

- **Stale validation state for parseable raws.** Strict schema-invalid raws were being persisted as `parse_error` / quarantine failures. The validation runtime was conflating "payload cannot be decoded" with "payload decoded but failed strict schema checks". Parseable raws that happened to fail strict schema validation showed up as permanently unparsed and unreachable to force-reparse planning.
- **Action-event orphan rows not counted.** Orphan rows were already measured and surfaced, but they did not participate in repair readiness, debt accounting, or repair detail. The readiness derived from row counts could report `ok` while repair would still report work.
- **Verifier decode-error detail lost.** `verify_raw_corpus` flattened decode errors to generic exception class names, while the live validation service preserved the real decode error text. The two surfaces disagreed on what "malformed JSONL" looked like.

Live state before the fixes: 12,524 total raws, 3 unparsed (2 parseable-but-validation-failed, 1 genuinely malformed, plus 1 hidden zero-byte Codex session exposed once the stale pair was resolved).

## Solution

- **Split raw parse failure from validation failure** (`polylogue/pipeline/services/validation_runtime.py`, `validation_flow.py`, `ingest_worker.py`, `ingest_batch.py`, `stage_models.py`, `polylogue/storage/raw_ingest_artifacts.py`). Only real decode/parse/transform failures now set `parse_error` and quarantine a raw; schema-invalid-but-decodable payloads stay in `validation_status = failed` without becoming parse failures. `_ValidationOutcome` dataclass and persisted batch outcomes carry the split explicitly.
- **Count action-event orphan rows in repair readiness** (`polylogue/storage/action_event_artifacts.py`, `tests/unit/storage/test_action_event_artifacts.py`, `tests/unit/storage/test_repair.py`). Orphan rows now participate in derived readiness, debt totals, and repair detail.
- **Preserve decode error detail in schema quarantine** (`polylogue/schemas/verification_corpus.py`, `tests/unit/core/test_schema_validation.py`). The verifier now carries the real decode error text from the live validation service instead of flattening to exception class names. Adds the missing direct regression test for empty-raw quarantine.

## Verification

- `pytest -q tests/unit/pipeline/test_ingest_batch.py tests/unit/storage/test_raw_ingest_artifacts.py tests/unit/storage/test_action_event_artifacts.py tests/unit/storage/test_repair.py tests/unit/storage/test_derived_status.py tests/unit/pipeline/test_parsing_service.py tests/unit/pipeline/test_run_sources.py tests/unit/core/test_schema_validation.py tests/unit/pipeline/test_prepare_records.py tests/unit/pipeline/test_resilience.py` → 170 passed
- `ruff check polylogue/pipeline/services/validation_runtime.py polylogue/pipeline/services/validation_flow.py polylogue/pipeline/services/ingest_worker.py polylogue/pipeline/services/ingest_batch.py polylogue/pipeline/stage_models.py polylogue/storage/raw_ingest_artifacts.py polylogue/storage/action_event_artifacts.py polylogue/schemas/verification_corpus.py`
- Live repair pass: 2 stale raws rewritten to `parsed`, 1 malformed raw quarantined with detailed decode error, 1 zero-byte Codex session explicitly quarantined. Final live state: 12,524 raws / 12,522 parsed / 2 intentional quarantines.

Commits on this branch: 3 (delta against `feature/refactor/stack-08-final-unification`).

## Stack

Base: `feature/refactor/stack-08-final-unification`. This is the last PR in the stacked series.
