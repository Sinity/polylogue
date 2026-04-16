---
created: "2026-04-13T20:35:00+02:00"
purpose: "Lean current operator brief for Polylogue runtime, drift, and verification work"
status: "active"
project: "polylogue"
---

# Operator Brief

This is the live entrypoint for current Polylogue work.

Use this with:

- `active/005-thorough-vetting-log.md` for the detailed chronological record
- `plans/008-verification-architecture-plan.md` for the longer verification roadmap
- `completed/019-finished-analysis-and-landed-phases.md` to separate already-landed work from still-open work

## Current Reality

- Repo: `/realm/project/polylogue`
- Branch: `feature/chore/repo-cleanup-governance`
- Worktree: clean
- Scratch tree policy:
  - this note is the current operator brief
  - `active/005-thorough-vetting-log.md` is the append-only execution log
  - bulky superseded campaign, drift-audit, and handoff notes are archived

## Live Archive Facts

- Last known live rebuild sample:
  - full rebuild wall: `56m40.43s`
  - full rebuild peak RSS: `7109260 kB` (~6.9 GiB)
  - parse rerun wall: `20m54.02`
  - parse rerun peak RSS: `5613500 kB` (~5.35 GiB)
- Known raw state summary:
  - `raw_conversations|12524|2|12522|2`
  - interpretation:
    - `12524` raws total
    - `2` raws still have `parsed_at IS NULL`
    - `12522` parsed
    - `2` carry `parse_error`
- Remaining unparsed raws now split into:
  - one genuinely malformed JSONL case
  - one zero-length empty Codex session export

## Open Product Issues

### 1. Stale validation-state cases were repaired live

- Raw ids:
  - `030d903225ad997c04498f9f32bf13acd21dfbad08059bc8931ba116703e528b`
  - `dead737557ca50accfd9e5deb1a7660412575234f33294ae980ac400c761e74f`
- Important facts already proven:
  - stdlib `json.loads()` accepts the problematic lines
  - `build_raw_payload_envelope(...)` now reports no malformed lines
  - direct `ingest_record(...)` succeeds
  - strict schema-invalid raws no longer persist as `parse_error` / quarantine failures
- Interpretation:
  - parser tolerance is fixed
  - persisted state semantics are fixed
  - live data has been rewritten under the new semantics

### 2. Two raws should remain explicit quarantine failure cases

- Malformed raw id:
  - `57399b8676d88e84698827874e6f7cee6700f8138841ca058205af05cc73fd7e`
- Known failure:
  - line `819`
  - `Expecting ',' delimiter: line 1 column 1293 (char 1292)`
- Zero-length raw id:
  - `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Known failure:
  - source file exists
  - size `0`
  - decode failure is expected
- Interpretation:
  - both belong in explicit quarantine policy, not in stale-state handling

### 3. Heavy live runs still have real memory pressure

- Known operator-facing memory spikes:
  - full rebuild about `6.9 GiB`
  - parse rerun about `5.35 GiB`
- This remains real even after the major read-path and summary-query improvements already landed

## Open Architectural Drift With Runtime Impact

Only unresolved drift worth carrying live:

- async-first storage still coexists with a broad production sync `sqlite3` path
- sync/async schema bootstrap still has protocol-level duplication even after the shared schema runtime landed
- default DB-path, SQLite tuning, and connection-profile semantics are much cleaner now, but `polylogue.paths` still acts as a broad facade
- `polylogue.paths` still behaves as a catch-all facade after the split
- runtime config/env policy is still broader and more fragmented than ideal
- legacy inline-raw archive compatibility still relies on reset, not migration
- health naming around narrow `index` readiness is still too generic

## Current Verification / Control-Plane Position

Already-landed roots:

- shared scenario metadata
- shared operation catalogs
- corpus-spec compilation
- executable scenario sources
- shared scenario execution semantics and runtime

Interpretation:

- the verification/control-plane unification wave is real and mostly landed
- the next work should be new runtime/product semantics or broader artifact-graph expansion, not more local catalog cleanup

## Recently Landed Runtime Fixes

- raw parse-failure vs schema-validation-failure semantics are now split correctly
- action-event orphan rows now participate in readiness, debt, and repair accounting
- live archive stale raw-state rows were repaired under the new semantics
- schema verification quarantine now preserves the real decode error text instead of flattening to exception class names

## Recommended Next Moves

1. Keep the malformed and zero-length raws as explicit quarantine regression cases.
2. Continue memory reduction on the heaviest live operator workflows only when backed by measured hot paths.

## Archived Superseded Notes

Historical detail now lives in:

- `archive/005-thorough-vetting-campaign.md`
- `archive/006-architectural-drift-audit.md`
- `archive/009-context-reset-handoff.md`
