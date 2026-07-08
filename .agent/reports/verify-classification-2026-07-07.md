---
created: "2026-07-07T20:40:00+02:00"
purpose: "polylogue-s7ae.6 deploy-gate evidence: full devtools verify run + failure classification ledger"
status: "complete"
project: "polylogue"
---

# Full-verify classification ledger — 2026-07-07 (polylogue-s7ae.6)

## Run record

- Command: `devtools verify --all` (full non-integration pytest, `-n 4`)
- Commit: master `848658dc3` (includes coordination commit `32ff31651`, 2026-07-04)
- Raw log: `.agent/reports/verify-full-2026-07-07.log`
- Result: **4 failed, 12725 passed, 1 skipped** in 187.55s (exit 1)
- Context: the "74% abort with many scattered failures" recorded on commit
  `32ff31651` was a **pytest-testmon noisy-baseline artifact**, not real
  failure mass. The clean `--all` run completes in ~3 minutes and has exactly
  4 failures.

## Classification table

| # | Failing node | Signature | First-bad commit | Class | Disposition |
|---|---|---|---|---|---|
| 1 | `tests/unit/storage/test_no_string_interpolated_sql.py::test_audited_sites_are_real_violations` | 6 stale `_AUDITED_SITES` line numbers | `bb2f84ff8` (2026-07-05, cost rollups; +~200 lines in archive.py) | pre-existing drift | **fixed** in PR #2556 |
| 2 | `...::test_no_unaudited_string_interpolated_sql` | new unaudited f-string SQL site `archive.py:1605` | `bb2f84ff8` | pre-existing | **fixed** in PR #2556 (site inspected: closed WHERE fragment list, user values bound via `?`) |
| 3 | `tests/unit/cli/test_plain_cli_snapshots.py::test_json_status_snapshot` | status payload gained `raw_parse_failed`, `missing_blob_source_available_count`, `missing_blob_source_missing_count` | `884efb5f9` + raw-blob-span work (2026-07-05) | pre-existing (stale syrupy snapshot) | **fixed** in PR #2556 (regenerated) |
| 4 | `tests/unit/test_logging_runtime.py::test_stdlib_bound_logger_forwards_exc_info_before_structlog_configured` | KeyError 'kwargs'; passes alone, fails after any test runs `configure_logging()` in the same worker | `ee1a51cb6` (2026-07-05, PR #2548 — the test itself) | pre-existing (test-order dependence) | **fixed** in PR #2556 (monkeypatch `_structlog_configured=False`) |

## Coordination-caused failures

**None.** No failing surface is touched by `32ff31651` (coordination/, mcp
agent tools, CLI agents group, beads hooks). All four failures date to the
2026-07-05 commit batch, which landed after the coordination merge.

## Verdict

- polylogue-s7ae.6 deploy gate: **open**. The coordination substrate carries
  zero attributable full-verify failures.
- Master full-verify baseline returns to green when PR #2556 merges
  (test-only diff; all three files re-verified green individually).
