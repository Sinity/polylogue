---
created: "2026-04-09T12:20:00+02:00"
purpose: "Review ~/.claude/plans/foamy-doodling-cerf.md against the current polylogue codebase"
status: "complete"
project: "polylogue"
---

# Property-First Plan Review

## Context

Reviewed the plan at `~/.claude/plans/foamy-doodling-cerf.md` against the current `polylogue` repository to determine:

- whether its baseline numbers are accurate,
- which proposed branches duplicate existing law/property coverage,
- which parts are still worth doing if rewritten to match the current architecture.

## Findings

### Baseline numbers are materially wrong

- The plan claims `3` property files and `16` property tests.
- Actual repo scan found `35` test files containing `@given(...)`.
- Actual repo scan found `233` `@given(...)` decorators across `tests/unit`, `tests/property`, and `tests/integration`.
- The undercount comes from treating only `tests/property/*` as property tests while ignoring `*_laws.py`, `*_props.py`, and other Hypothesis suites under `tests/unit/*` and `tests/integration/*`.

### Xfail count is wrong

- The plan mentions `7 xfail property tests`.
- The repo no longer carries those `xfail` markers; the former property-test debt in `tests/property/test_semantic_properties.py` was removed after fixing the test harness to respect `parse_payload()` returning a conversation list.

### Several proposed areas already exist under different names

- Filter algebra already exists in `tests/unit/core/test_filters_props.py`.
- Provider/schema semantic and dispatch laws already exist in:
  - `tests/unit/sources/test_unified_semantic_laws.py`
  - `tests/unit/sources/test_source_laws.py`
  - `tests/unit/sources/test_parser_crashlessness.py`
- Provider-driven synthetic roundtrip coverage already exists in:
  - `tests/unit/core/test_synthetic_semantics.py`
  - `tests/unit/core/test_synthetic_semantic_wiring.py`
- Storage roundtrip/idempotence laws already exist in `tests/unit/storage/test_store_ops.py`.
- Tree/query/search law coverage already exists in `tests/unit/storage/test_tree_laws.py`, `tests/unit/storage/test_hybrid_laws.py`, and `tests/unit/storage/test_fts5.py`.
- MCP product-tool coverage already exists in `tests/unit/mcp/test_tool_contracts.py`.

### Some proposed architecture would add parallel abstractions instead of extending the current test system

- New `tests/property/conftest.py` for provider auto-derivation is unnecessary if the suite already uses `SyntheticCorpus.available_providers()` and `SchemaRegistry.list_providers()`.
- New `tests/infra/regression_capture.py` duplicates concerns already centered in `tests/conftest.py` for Hypothesis profile management.
- New MCP/product strategies should extend `tests/infra/mcp.py` and existing strategy modules rather than creating a second test architecture.

### Some proposed APIs/paths are directionally right but not the most canonical integration points

- Schema/provider generation should key off `SyntheticCorpus.available_providers()` or the runtime registry directly, but only where the current tests are still hardcoded.
- Product coverage should key off `PRODUCT_REGISTRY`, but existing tests already enumerate and assert all current registry-backed MCP tools.

## Outcome

### Keep

- Add missing parse → persist → hydrate coverage if the goal is true end-to-end storage fidelity rather than just parser roundtrip.
- Add foreign-provider rejection laws if the goal is explicit negative testing rather than crashlessness on schema-conformant input.
- Add more generative MCP parameter coverage if it supplements, not replaces, the existing contract suite.

### Drop or rewrite

- Do not describe the work as creating a property-testing culture from near-zero. The repo already has a large property/law suite.
- Do not delete `tests/property/test_semantic_properties.py` just because a more generic variant can exist; first merge or extend the stronger existing semantic-law suites.
- Do not create duplicate test infrastructure in `tests/property/` when the stronger pattern in this repo is domain-local laws near the relevant subsystem.

### Better framing

- Reframe the work as “close a few missing end-to-end law gaps” instead of “introduce property-first testing.”
- Count coverage by Hypothesis/law suites across the whole repo, not by `tests/property/` directory membership.
- Prefer extending existing files/modules over adding many new top-level property files.
