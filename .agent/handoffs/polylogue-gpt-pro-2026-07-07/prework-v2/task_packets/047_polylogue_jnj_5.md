# 047. polylogue-jnj.5 — Route ops reset --session/--source through the mutation contract

Priority/type/status: **P2 / bug / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Identity resets tombstone directly before the preview/confirmation branch — a typo mutates suppression state without dry-run or JSON evidence. Require dry-run preview + --yes + stable JSON like other destructive ops.

## Existing design note

Audit-confirmed: ops reset --session/--source tombstones BEFORE the preview/confirmation branch in the reset command implementation (cli/commands/ reset path). Fix: route identity resets through the same mutation contract as other destructive ops — dry-run prints exact target rows (origin/native_id, counts), mutation requires --yes, stable JSON envelope for both. Test: typo'd session ref produces zero-target dry-run and no mutation; real ref mutates only with --yes.

## Acceptance criteria

- `polylogue ops reset --session <ref>` and `--source <ref>` print a dry-run of the exact target rows (origin/native_id + counts) BEFORE any tombstone write; no mutation occurs without `--yes` (code path confirmed: tombstone no longer runs before the preview/confirmation branch — grep the reset command implementation).
- Test: a typo'd/nonexistent session ref produces a zero-target dry-run and zero rows mutated (suppression state asserted unchanged).
- Test: a real ref with `--yes` mutates only the named targets; a stable JSON envelope is emitted for both dry-run and mutation (same shape as other destructive ops).
- `devtools test <reset command test>` green for both paths.

## Static mechanism / likely defect

Issue description localizes the mechanism: Identity resets tombstone directly before the preview/confirmation branch — a typo mutates suppression state without dry-run or JSON evidence. Require dry-run preview + --yes + stable JSON like other destructive ops. Design direction: Audit-confirmed: ops reset --session/--source tombstones BEFORE the preview/confirmation branch in the reset command implementation (cli/commands/ reset path). Fix: route identity resets through the same mutation contract as other destructive ops — dry-run prints exact target rows (origin/native_id, counts), mutation requires --yes, stable JSON envelope for both. Test: typo'd session ref produces zero-target dry-run…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Audit-confirmed: ops reset --session/--source tombstones BEFORE the preview/confirmation branch in the reset command implementation (cli/commands/ reset path).
2. Fix: route identity resets through the same mutation contract as other destructive ops — dry-run prints exact target rows (origin/native_id, counts), mutation requires --yes, stable JSON envelope for both.
3. Test: typo'd session ref produces zero-target dry-run and no mutation
4. real ref mutates only with --yes.

## Tests to add

- Acceptance proof: `polylogue ops reset --session <ref>` and `--source <ref>` print a dry-run of the exact target rows (origin/native_id + counts) BEFORE any tombstone write
- Acceptance proof: no mutation occurs without `--yes` (code path confirmed: tombstone no longer runs before the preview/confirmation branch — grep the reset command implementation).
- Acceptance proof: Test: a typo'd/nonexistent session ref produces a zero-target dry-run and zero rows mutated (suppression state asserted unchanged).
- Acceptance proof: Test: a real ref with `--yes` mutates only the named targets
- Acceptance proof: a stable JSON envelope is emitted for both dry-run and mutation (same shape as other destructive ops).
- Acceptance proof: `devtools test <reset command test>` green for both paths.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
