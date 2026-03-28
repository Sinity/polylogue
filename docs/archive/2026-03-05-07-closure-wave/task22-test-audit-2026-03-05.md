# Task 22 Audit Report (2026-03-05)

## Scope

Deep adversarial audit of test-suite quality across:
- `tests/unit/storage`
- `tests/unit/pipeline`
- `tests/unit/cli`
- `tests/unit/core`
- `tests/integration`

Audit agent: `Euclid` (explorer)

## Executive Summary

Total issues identified: 24
- Broken: 8
- Misleading: 10
- Redundant: 3
- Stylistic/high-maintenance risk: 3

Highest-risk areas:
1. Integrations
2. CLI
3. Storage/Core

## Broken (Must Fix)

1. `tests/unit/core/test_properties.py:417`
- `test_json_loads_handles_arbitrary_text`
- Tautology (`result is not None or result is None`).

2. `tests/unit/core/test_properties.py:429`
- `test_json_loads_handles_arbitrary_bytes`
- Same tautology.

3. `tests/unit/storage/test_crud.py:784`
- `test_new_extraction_is_superset`
- Tautological branch (`reasoning_found >= 0`).

4. `tests/unit/storage/test_async.py:499`
- `test_context_manager_cleanup`
- No effective post-condition assertion.

5. `tests/unit/pipeline/test_enrichment_guards.py:139`
- `test_empty_fenced_block_handled`
- `assert len(result) >= 0` tautology.

6. `tests/integration/test_facade.py:1271`
- `test_latest_render_path_handles_deleted_files`
- Tautological assertion.

7. `tests/integration/test_mcp_resources.py:538`
- `test_search_error_cases`
- Try/except allows pass in both success/failure paths.

8. `tests/unit/cli/test_run_int.py:957`
- `test_store_records_commits_within_lock`
- Assertion does not verify commit/lock ordering.

## Misleading (High Regression Risk)

1. `tests/unit/core/test_schema.py:136`
- Missing-mapping rejection test asserts only type.

2. `tests/unit/core/test_schema.py:146`
- Drift detection test asserts only type.

3. `tests/unit/core/test_filters_adv.py:398`
- Include/exclude semantics not asserted.

4. `tests/unit/core/test_filters_adv.py:734`
- Continuation/sidechain/has_branches tests assert mostly container type.

5. `tests/unit/storage/test_repository.py:347`
- Negative search path can pass with list-type-only assertion.

6. `tests/unit/storage/test_fts5.py:444`
- Injection-prevention branch checks type, not behavior/safety contract.

7. `tests/unit/pipeline/test_progress_callbacks.py:154`
8. `tests/unit/pipeline/test_progress_callbacks.py:191`
- Callback propagation tests weakly assert non-None result.

9. `tests/unit/pipeline/test_runner.py:559`
- `result.indexed in (True, False)` non-assertive.

10. `tests/unit/cli/test_search.py:33`
- Overly broad acceptable exit-code sets.

11. `tests/integration/test_workflows.py:149`
- HTML case checks markdown artifact behavior.

12. `tests/integration/test_mcp.py:511`
- Invalid limit behavior not asserted precisely.

13. `tests/integration/test_security.py:376`
14. `tests/integration/test_security.py:816`
- Weak/tautological assertions on path/unicode behaviors.

15. Integration tests with synthetic bypass:
- `tests/integration/test_mcp.py:246`
- `tests/integration/test_mcp_mutations.py:73`
- `tests/integration/test_mcp_resources.py:500`
- Heavy patching bypasses real repo/filter stack.

## Redundant

1. `tests/unit/cli/test_commands.py:168` and `tests/integration/test_facade.py:1220` overlap (`latest_render_path`).
2. `tests/integration/test_mcp_mutations.py:19` and `tests/integration/test_mcp_exports.py:210` overlap (tool inventory).
3. `tests/integration/test_mcp.py:511` and `tests/integration/test_mcp_resources.py:533` overlap (invalid-limit handling).

## Stylistic / High-Maintenance Risk

1. `tests/unit/pipeline/test_concurrency_guards.py:48`
- Asserts private internals of `ConversationFilter`.

2. `tests/unit/pipeline/test_branching.py:888`
- Asserts internal ID-prefix format (`codex:`) instead of semantic linkage.

3. `tests/unit/core/test_schema.py:276`
- Requires exact provider set equality; brittle under additive providers.

## Priority Remediation Queue

1. Replace all broken tautological/delete-proof assertions (8 items).
2. Tighten CLI exit-code and output contracts (`test_search.py`, `test_commands.py`).
3. Fix integration correctness assertions (`test_workflows.py`, `test_mcp_resources.py`, `test_facade.py`).
4. Convert at least one MCP integration path from patched internals to real temp-DB/repo wiring.
5. Deduplicate overlapping tests after behavior contracts are strengthened.

## Status

- Audit completed, no code modifications made by the audit agent itself.
- Follow-up implementation work remains open (see session workload tracker).

## Remediation Progress (2026-03-05)

Implemented in this continuation:

1. Broken-test fixes (wave 1) completed:
- `tests/unit/core/test_properties.py` (2 tautological property tests hardened)
- `tests/unit/storage/test_crud.py` (`test_new_extraction_is_superset` now has meaningful pass/skip contract)
- `tests/unit/storage/test_async.py` (`test_context_manager_cleanup` now asserts post-exit behavior)
- `tests/unit/pipeline/test_enrichment_guards.py` (empty fenced block exact-output assertion)
- `tests/integration/test_facade.py` (`latest_render_path` deleted-file handling now deterministic)
- `tests/integration/test_mcp_resources.py` (`test_search_error_cases` now enforces explicit error/success contracts)
- `tests/unit/cli/test_run_int.py` (commit-inside-lock behavior now asserted via lock-state instrumentation)

2. High-risk misleading fixes (wave 2) partially completed:
- `tests/integration/test_workflows.py` (`html` format path now validated via real HTML renderer output)
- `tests/unit/cli/test_search.py` (broad exit-code assertions tightened to deterministic expected outcomes)
- `tests/integration/test_mcp.py` (invalid limit now asserts clamp-to-1 behavior)
- `tests/unit/pipeline/test_progress_callbacks.py` (callback propagation assertions now tied to concrete run result/callback descriptors)
- `tests/unit/core/test_schema.py` (schema expectation tests now assert actual validator contract)
- `tests/unit/core/test_filters_adv.py` (branching and include/exclude behavior now asserted semantically)
- `tests/unit/storage/test_repository.py` (negative search path now asserts empty result)
- `tests/unit/storage/test_fts5.py` (safe pass-through query case now asserts exact non-rewrite behavior)
- `tests/unit/pipeline/test_runner.py` (`index` stage-only test now asserts concrete index contract)
- `tests/integration/test_security.py` (dots-only filename normalization and unicode parameter behavior now strict)

Validation after remediations:
- Full suite: `4465 passed, 1 skipped, 0 failed` (latest `pytest -r w` run reported no warning messages).

## Remediation Progress (2026-03-05, Wave 3)

Implemented in this continuation:

1. Maintenance-risk tests decoupled from private internals:
- `tests/unit/pipeline/test_concurrency_guards.py`
  - replaced `_providers`/`_fts_terms`/`_limit_count` assertions with behavior assertions against repository/filter execution paths.
- `tests/unit/pipeline/test_branching.py`
  - replaced internal ID-prefix check with semantic parent-link integrity assertions.
- `tests/unit/core/test_schema.py`
  - replaced exact-provider-set equality with core-subset contract (additive providers no longer break test).

2. Synthetic-bypass reduction with real temp-DB MCP paths:
- `tests/integration/test_mcp.py`
  - added real repository-backed integration tests for `search` and `list_conversations`.
  - converted invalid-limit validation path to assert runtime behavior with real data (`limit=-1` -> one result).

3. Redundancy cleanup:
- `tests/integration/test_mcp_mutations.py`
  - scoped server inventory assertion to mutation tools, reducing overlap with exhaustive inventory checks in export/completeness tests.

Validation for wave-3 touched files:
- `pytest -q tests/unit/pipeline/test_concurrency_guards.py tests/unit/pipeline/test_branching.py tests/unit/core/test_schema.py tests/integration/test_mcp.py tests/integration/test_mcp_mutations.py`
- Result: `165 passed`
