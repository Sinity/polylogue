# Workload Closure Report (2026-03-06)

Current closure snapshot for schema/validation/QA work.

## Completed in this pass

- Promoted generated schemas into runtime registry latest:
  - `chatgpt v4`, `claude-ai v2`, `claude-code v2`, `codex v9`, `gemini v2`.
- Promoted registry-latest schemas into packaged in-repo baselines:
  - `polylogue/schemas/providers/chatgpt.schema.json.gz`
  - `polylogue/schemas/providers/claude-ai.schema.json.gz`
  - `polylogue/schemas/providers/claude-code.schema.json.gz`
  - `polylogue/schemas/providers/codex.schema.json.gz`
  - `polylogue/schemas/providers/gemini.schema.json.gz`
- Restored Gemini schema operability with registry-latest loading:
  - `polylogue check --schemas --schema-provider gemini --schema-samples all` => `226/226 valid`.
- Fixed record-mode schema verification semantics:
  - Record-granularity validation now skips non-record metadata documents instead of forcing false-invalid checks.
  - Record-granularity schema generation no longer enforces root-level `required` keys for heterogeneous record corpora.
- Added regression coverage for record-mode sample behavior.
- Ran full regression suite:
  - `nix develop -c pytest -q`
  - `4519 passed, 1 skipped`.

## Heavy-provider verification results (final)

Artifacts:
- `qa_outputs/schema-verification-heavy-full-2026-03-06.json`
- `qa_outputs/schema-verification-heavy-claude-code-full-2026-03-06.json`
- `qa_outputs/schema-verification-heavy-codex-full-2026-03-06.json`

Full-corpus (`samples=16`) status:
- `claude-code`: `4643 total`, `4640 valid`, `0 invalid`, `3 decode_errors`
- `codex`: `1013 total`, `1011 valid`, `0 invalid`, `2 decode_errors`

Interpretation:
- Schema-invalid residuals were eliminated for heavy providers.
- Remaining residuals are decode-only malformed raw lines under strict policy.

## Additional post-closure checks

Artifacts:
- `qa_outputs/Q22_schema_check_gemini_all.txt`
- `qa_outputs/Q22_schema_check_codex_300.txt`
- `qa_outputs/Q22_schema_check_claude_code_300.txt`

Results:
- `gemini` (all records): `226 valid`, `0 invalid`, `0 decode_errors`
- `codex` (`records=300`): `300 valid`, `0 invalid`, `0 decode_errors`
- `claude-code` (`records=300`): `300 valid`, `0 invalid`, `0 decode_errors`

## Overhead benchmark status (`samples=16` vs `samples=all`)

Artifact:
- `qa_outputs/schema-verification-heavy-overhead-2026-03-06.json`

Scope:
- Representative heavy-provider probes (`record_limit=10`) at two offsets/provider.

Result:
- Timings are captured and reproducible.
- Ratios vary by data shape and cache state; use artifact timings directly.

## Remaining work

1. No blocking schema/validation/QA tasks remain from this closure scope.
2. Optional follow-up: canonical handling of malformed raw lines (repair/quarantine workflow).
