# Polylogue Read-Surface Proof And Showcase Hardening Program

Date: 2026-03-22
Status: executed subprogram
Role: executed post-export proof phase covering query summary/list surfaces, stream surfaces, MCP read payloads, and matching showcase hardening

See also:

- `multi-surface-semantic-proof-program-2026-03-22.md`
- `semantic-proof-and-showcase-proof-lanes-program-2026-03-22.md`
- `testing-reliability-expansion-program-2026-03-14.md`
- `planning-and-analysis-map-2026-03-21.md`

## Purpose

Finish the next proof frontier after canonical render and export surfaces:

1. query summary/list read surfaces
2. streamed read surfaces
3. MCP read payloads
4. showcase coverage for those proof lanes

The goal was not just more tests. It was to make the real operator read
surfaces proof-bearing, and to tighten a few semantics while doing so.

## Why This Was The Frontier

After the export proof work, Polylogue could explain semantic preservation for:

- canonical markdown renders
- export document formats

It still mostly relied on narrower formatter tests and implied contracts for:

- `--list` summary outputs
- `--stream` outputs
- MCP read payloads
- showcase proof coverage of those surfaces

That left a conspicuous gap between “proof-bearing exports” and “trust-based
read surfaces”.

## Executed Outcomes

This phase delivered:

- proof-bearing query summary/list surfaces
- proof-bearing streamed query surfaces
- proof-bearing MCP summary/detail read payloads
- deterministic helper renderers shared by operator code and proof code
- stricter streamed metadata and emitted-message accounting
- showcase exercises that explicitly cover the new read-surface proof lane

## Executed Scope

### 1. Query Summary Surface Canonicalization

Extracted deterministic summary-list formatting into a pure helper so query
operators and semantic proof share the same surface:

- JSON
- YAML
- CSV
- plain text

Primary module:

- `polylogue/cli/query_output.py`

### 2. Stream Surface Canonicalization

Extracted reusable stream header/message/footer renderers and used them both in
query execution and proof generation.

This also tightened the stream contract:

- markdown and JSON-lines headers now preserve more metadata
- footer counts now reflect emitted visible messages rather than optimistic
  iteration counts

Primary module:

- `polylogue/cli/query_output.py`

### 3. Read-Surface Semantic Proof Expansion

Extended the semantic-proof suite with new canonical surfaces:

- `query_summary_json_v1`
- `query_summary_yaml_v1`
- `query_summary_csv_v1`
- `query_summary_text_v1`
- `query_stream_plaintext_v1`
- `query_stream_markdown_v1`
- `query_stream_json_lines_v1`
- `mcp_summary_json_v1`
- `mcp_detail_json_v1`

Added aliases such as:

- `query_all`
- `stream_all`
- `mcp_all`
- `read_all`

Primary module:

- `polylogue/rendering/semantic_proof.py`

### 4. Operator Surface Updates

Updated `polylogue check --semantic-proof` help and surface vocabulary so the
read-surface proof lane is first-class and operator-visible.

Primary module:

- `polylogue/cli/commands/check.py`

### 5. Showcase Hardening

Added dedicated showcase exercises for the read-surface proof lane:

- `check-semantic-proof-read-surfaces`
- `check-semantic-proof-read-surfaces-json`

Primary module:

- `polylogue/showcase/exercises.py`

## Verification

Executed focused verification covering:

- query helper rendering
- query execution streaming contracts
- semantic-proof suite expansion
- `check` proof surfaces
- showcase exercise catalog integrity
- QA/report/site/click consumers of semantic-proof summaries
- deterministic help snapshot updates

Primary touched areas:

- `polylogue/cli/query_output.py`
- `polylogue/rendering/semantic_proof.py`
- `polylogue/cli/commands/check.py`
- `polylogue/showcase/exercises.py`
- `tests/unit/cli/test_query_fmt.py`
- `tests/unit/core/test_semantic_proof.py`
- `tests/unit/showcase/test_exercise_catalog.py`
