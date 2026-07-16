# 27. polylogue-9e5.3 — Column-honesty census for nullable/zero/default public fields

Priority: **P2**  
Lane: **evidence-honesty**  
Readiness: **ready-now / audit-artifact**

Depends on packet(s): polylogue-9e5.29

## Why this is urgent / critical-path

Before fixing every dishonest field, produce a field census that shows where null/zero/default semantics are unclear.

## Static diagnosis / likely mechanism

This is a read-only audit packet. It supports 9e5.29 by finding numeric/default columns whose meaning is ambiguous, especially across insight products, usage, attachments, and lineage.

## Implementation plan

Implementation shape:
1. Generate a table of public payload fields/DB columns with type, nullable, default, sample null density, sample zero density, and known evidence source.
2. Classify each as true-zero-safe, unknown-when-absent, not-applicable, text-derived, or needs-contract.
3. Emit JSON + Markdown artifact under docs/audits or `.agent/reports`.
4. File follow-up beads only for confirmed high-risk fields.

## Test plan

Tests are optional unless adding tooling. If adding a command, test on a small fixture schema/payload set and assert classifications render.

## Verification command / proof

Run the census command/artifact generation on the active fixture/live copy read-only. Review artifact against 9e5.29 field-contract work.

## Pitfalls

Do not mutate product code beyond a small audit tool. The goal is a map for follow-up patches.

## Files/functions to inspect or touch

- `polylogue/insights/* models`
- `polylogue/storage/sqlite/archive_tiers/index.py`
- `polylogue/storage/usage.py`
- `report/model schema emitters`
