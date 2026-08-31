Summary

Admit deliberate unknown tool outcomes as `ToolOutcome.UNKNOWN`, preserve their parser reason, and keep action surfaces honest across Claude Code, Codex, and Hermes transcripts.

Problem

The writer refused real Claude Code and Codex results whose parser evidence deliberately recorded `not_reported` or `distrusted`. The prior lane also converted unknown action assertions to success and removed no-verdict shapes from synthetic generators.

Solution

The writer derives `unknown` from `tool_result_outcome_unknown_reason`, preserves the reason, and leaves legacy error and exit-code fields null. Codex and Hermes parser paths now retain `not_reported` for unresolved structural results. Action SQL maps the canonical value to `outcome_unknown`. The restored assertions expect `OUTCOME_UNKNOWN` and `tool_success is None`. Claude Code, Codex, and tool-heavy synthetic generators can again emit no-verdict results, with regressions for writer admission, reason preservation, action projections, and generator coverage. Index schema v82 declares semantic replay for the changed derived DDL.

Verification

- `.venv/bin/python -m devtools test tests/unit/storage/test_tool_outcome.py` passed: 18 passed.
- `.venv/bin/python -m devtools test tests/unit/sources/parsers/test_hermes_state.py tests/unit/sources/test_parsers_codex_catalog.py tests/unit/core/test_synthetic_semantics.py::TestWireFormatShape` passed: 30 passed.
- `.venv/bin/python -m devtools test tests/unit/storage/test_archive_tiers_archive.py::test_archive_facade_exposes_distinct_action_result_states tests/unit/storage/test_archive_tiers_archive.py::test_archive_action_relation_distinguishes_empty_payload_from_absent_linkage tests/unit/storage/test_archive_tiers_archive.py::test_session_action_occurrences_pair_repeated_ids_by_rank_and_page_after_pairing tests/unit/api/test_facade_contracts.py::test_get_actions_batch_pairs_session_wide_and_exposes_result_state` passed: 4 passed.
- `.venv/bin/python -m devtools verify --quick` passed all quick checks, including schema versioning.
- Real-source census through the parser and writer derivation route: 142 valid sampled Claude Code transcripts with 9,954 tool results and 0 derivation refusals; 116 Codex transcripts with 42,388 tool results and 0 derivation refusals. One sampled Claude subagent JSONL was malformed on disk and was excluded as an input parse error.

Residual risk: the complete 59 GB local JSONL corpus and live daemon convergence were not run. The malformed source file remains an input-quality issue, not a typed-outcome refusal.
