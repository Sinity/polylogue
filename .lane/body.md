Summary

Added a closed `ok`/`error`/`no_result` outcome to admitted tool blocks and exposed it through archive records, material protocol, query payloads, and action projections.

Problem

Tool results relied on nullable error and exit-code columns, leaving outcome unknown when parser evidence was absent and allowing prose-shaped records to enter the archive without a typed verdict.

Solution

Derive outcomes from parser-provided structural fields and Claude Code execution sidecars. Pair recorded results back to tool uses, mark unpaired uses as `no_result`, reject result blocks without structural evidence, and require semantic replay for the new index projection. Updated parser fixtures, synthetic fixtures, and compatibility readers.

Verification

- `.venv/bin/python devtools/__main__.py test tests/unit/storage/test_tool_outcome.py -q` passed: 16 passed.
- `.venv/bin/python devtools/__main__.py test tests/unit/sources/test_tool_result_structural_outcome.py tests/unit/sources/test_claude_code_unread_wire_fields.py tests/unit/sources/test_tool_result_sidecars.py tests/unit/storage/test_tool_outcome.py tests/unit/storage/test_archive_tiers_archive.py tests/unit/storage/test_archive_tiers_write.py tests/unit/pipeline/test_archive_write.py tests/unit/pipeline/test_ingest_batch.py -q` passed: 297 passed.
- `.venv/bin/python devtools/__main__.py test tests/unit/sources/test_parsers_claude_ai_catalog.py tests/unit/sources/test_parsers_codex_catalog.py tests/unit/sources/test_parsers_claude_design.py tests/unit/archive/test_query_runtime_matching.py -q` passed: 66 passed.
- `.venv/bin/python devtools/__main__.py test tests/unit/storage/test_archive_tiers_ddl.py tests/unit/storage/test_index_fast_forward_lifecycle.py tests/unit/storage/test_attachment_cascade_sweep.py tests/unit/storage/test_index_fast_forward_executor.py tests/unit/archive/test_query_agg_metrics.py tests/unit/archive/test_query_multi_aggregate.py -q` passed: 86 passed.
- `nix develop --command devtools verify --quick` passed.
- Residual risk: the full corpus and live archive convergence were not run. An unrelated inherited embedding-generation test still fails on an existing recipe-hash contract mismatch.
