# Polylogue Multi-Surface Semantic Proof Program

Date: 2026-03-22
Status: executed subprogram
Role: executed post-markdown semantic-proof phase covering query/export surfaces, QA composition, publication summaries, and showcase proof-lane coverage

See also:

- `semantic-proof-and-showcase-proof-lanes-program-2026-03-22.md`
- `artifact-and-semantic-proof-program-2026-03-19.md`
- `testing-reliability-expansion-program-2026-03-14.md`
- `planning-and-analysis-map-2026-03-21.md`

## Purpose

Extend Polylogue's semantic-proof chain from one canonical render surface to
the broader export/query family that operators and MCP consumers actually use.

The previous slice proved only:

- canonical repository projection -> canonical markdown

This phase proves a wider chain:

1. canonical repository projection -> canonical markdown
2. conversation export/query formatter -> JSON
3. conversation export/query formatter -> YAML
4. conversation export/query formatter -> CSV
5. conversation export/query formatter -> markdown
6. conversation export/query formatter -> HTML
7. conversation export/query formatter -> Obsidian
8. conversation export/query formatter -> Org

That makes semantic proof less of a renderer-local feature and more of a real
operator/export contract.

## Why This Is The Next Ambitious Phase

Polylogue now has:

- durable artifact/cohort proof
- package-aware schema authority
- publication manifests
- canonical markdown semantic proof
- proof-lane showcase coverage

The main remaining proof gap is that exported/query-visible surfaces still rely
on trust. They share `format_conversation()` and `export_conversation()`, but
Polylogue cannot yet explain which semantics each export surface preserves and
which ones it intentionally drops.

This phase closes that gap.

## Executed Outcomes

This phase delivered:

- expose one semantic-proof suite rather than one markdown-only report
- let `check --semantic-proof` prove multiple export/query surfaces at once
- classify loss per surface as preserved, declared loss, or critical loss
- embed multi-surface summaries into QA/session/publication outputs
- treat query/export and MCP export surfaces as proof-bearing, not just
  formatter internals
- expand showcase coverage so proof lanes exercise the broader semantic suite

## Executed Scope

### 1. Replaced Single-Surface Semantic Proof With A Surface Suite

Implemented a suite report with:

- per-surface reports
- overall suite summary
- surface-level provider summaries
- surface filtering

The markdown-only report remains a building block, not the top-level product.

### 2. Added Export-Surface Semantic Proofs

Added typed proof runners for:

- `canonical_markdown_v1`
- `export_json_v1`
- `export_yaml_v1`
- `export_csv_v1`
- `export_markdown_v1`
- `export_html_v1`
- `export_obsidian_v1`
- `export_org_v1`

Each surface now declares what it is expected to preserve and what it is
allowed to drop.

### 3. Used Explicit Loss Policies

Examples:

- JSON/YAML should preserve conversation identity, provider, title, message
  identities, role distribution, and timestamps
- CSV may intentionally drop provider/title-level metadata and typed semantic
  annotations while still preserving text-bearing message rows
- markdown/obsidian/org may intentionally drop per-message timestamps,
  attachments, and typed tool/thinking markers
- HTML should preserve visible message structure and branch presentation

### 4. Rewired Operator Surfaces

Update:

- `polylogue check --semantic-proof`
- QA/session JSON and Markdown
- publication/site semantic-proof summaries
- showcase proof-lane coverage

### 5. Tightened Export/API Cohesion

`format_conversation()` and export-facing surfaces are now proof-governed
interfaces rather than ad hoc string formatters.

## Verification

Executed focused verification covering:

- semantic proof suite aggregation
- export-surface proof rules
- `check --semantic-proof` JSON/plain output and surface filtering
- QA/session/report serialization of the suite
- publication/site manifest embedding of surface summaries
- showcase catalog coverage for the broader proof lane
- MCP/export-adjacent formatter coverage where relevant

Primary touched areas:

- `polylogue/rendering/semantic_proof.py`
- `polylogue/cli/commands/check.py`
- `polylogue/showcase/qa_runner.py`
- `polylogue/showcase/report.py`
- `polylogue/publication.py`
- `polylogue/site/publication_support.py`
- `polylogue/rendering/renderers/html.py`
