Summary

Admit deliberate unknown tool outcomes as `ToolOutcome.UNKNOWN`, preserve their parser reason, and keep action surfaces honest across Claude Code, Codex, and Hermes transcripts.

Problem

The writer refused real Claude Code and Codex results whose parser evidence deliberately recorded `not_reported` or `distrusted`. The prior lane also converted unknown action assertions to success and removed no-verdict shapes from synthetic generators.

Solution

The writer derives `unknown` from `tool_result_outcome_unknown_reason`, preserves the reason, and leaves legacy error and exit-code fields null. Codex and Hermes parser paths retain `not_reported` for unresolved structural results. Action SQL maps the canonical value to `outcome_unknown`. Claude Code, Codex, and tool-heavy synthetic generators can again emit no-verdict results. A parser-to-writer regression covers both real parser routes, and the restored assertions expect `OUTCOME_UNKNOWN` with `tool_success is None`. Index schema v82 declares semantic replay for the changed derived DDL.

Verification

- `.venv/bin/python -m devtools test tests/unit/storage/test_tool_outcome.py tests/unit/core/test_synthetic_semantics.py::TestWireFormatShape tests/unit/sources/test_claude_code_unread_wire_fields.py tests/unit/sources/test_tool_result_structural_outcome.py tests/unit/sources/parsers/test_hermes_state.py tests/unit/sources/test_parsers_codex_catalog.py::test_codex_catalog_parser_only_smoke` passed: 59 passed.
- Read-only parser census through `prepare_session_rows`: 143 sampled Claude Code files with 11,259 tool results and 0 refusals; 116 sampled Codex files with 46,414 tool results and 0 refusals. Deliberate unknown reasons were retained in both populations.
- `git fetch origin master && git rebase origin/master` passed. The lane uses schema v82, above the v80 and v81 claims present in other active worktrees.
- `.venv/bin/python -m devtools verify --quick` passed: all quick checks green, including schema versioning.

Residual risk: the complete 59 GB local JSONL corpus and live daemon convergence were not run. The malformed source file noted in the prior review remains an input-quality issue, not a typed-outcome refusal.

Reviewed-by: Claude (Opus) cross-family review
