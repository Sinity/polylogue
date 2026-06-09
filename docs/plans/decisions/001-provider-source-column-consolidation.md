# ADR 001: Provider/Source Column Consolidation

**Status**: accepted  
**Date**: 2026-05-25  
**Issue**: [#1022](https://github.com/Sinity/polylogue/issues/1022)

## Context

Polylogue's schema currently carries two parallel columns on `conversations`
and `session_profiles`:

| Column | Origin | Purpose |
|--------|--------|---------|
| `provider_name` | Legacy (archive) | Storage-layer provider token |
| `source_name` | Added in #864 + follow-ups | Source-family identity token |

This is a column-doubling, not a rename. The `Source` dataclass
(`polylogue/core/sources.py`) and bidirectional `Provider` ↔ `Source`
mapping are fully landed. All four decomposition children of #1022 are
closed (#1214, #1216, #1219, #1221). The remaining work is physical
schema consolidation: pick one column, drop the other, update all mappers.

## Decision

**Drop `provider_name`. Keep `source_name` as the single canonical column.**

Rationale:

1. **Source-family is the user-facing concept.** CLI flags, MCP parameters,
   daemon source discovery, and watcher roots all operate on source-family
   tokens. The column should match what users see.
2. **`provider_name` conflates three concepts.** It mixes lab identity
   (OpenAI, Anthropic), product identity (Claude Code, Codex), and
   source-family identity into one token. `source_name` carries the
   disambiguated source family.
3. **Storage compatibility is not a concern.** Per the
   `polylogue-write-passive-read-pull` crystal, Polylogue has no in-place schema upgrade
   chain. Schema bumps are deletes-then-defines. There are no external
   consumers of the SQLite schema beyond lynchpin.
4. **Lynchpin is the only cross-repo consumer** and reads both columns. It
   can be updated atomically.

## Rejected alternatives

### Keep `provider_name`, drop `source_name`

Rejected because it preserves the conflation. Every new feature that needs
to distinguish "Claude Code session" from "Claude AI export" (both currently
`Provider.CLAUDE_CODE` vs `Provider.CLAUDE_AI`) would need a separate
disambiguation column layered on top of `provider_name`.

### Keep both columns

Rejected because column-doubling is the worst state: twice the storage, twice
the index maintenance, and every reader must decide which column to trust.
The dual-vocabulary period was for API surfaces, not for physical schema.

### Rename `provider_name` to `source_name` in-place

SQLite does not support `ALTER TABLE RENAME COLUMN` for columns with indexes
or FTS content references. An in-place rename would require rebuilding the
table, which is equivalent to a schema bump. No benefit over drop-and-replace.

## Reversal conditions

- If an external consumer (beyond lynchpin) is discovered that reads
  `provider_name` and cannot be updated, reconsider the drop.
- If source-family tokens prove insufficient for a new use case that
  genuinely needs lab-level identity, add a separate `originating_lab`
  column rather than reverting to the conflated `provider_name`.

## Migration cost

| Operation | Estimate |
|-----------|----------|
| Drop `provider_name` column + index | Schema bump (new `SCHEMA_VERSION`) |
| Update ~20 mapper files | Mechanical, <1 hour |
| Update lynchpin consumer | 2 files, <30 min |
| Re-ingest from source | Operator action (not automated) |

## Cross-repo coordination

Lynchpin reads `provider_name` in:
- `lynchpin/sources/polylogue.py`
- `lynchpin/substrate/`

These must be updated to read `source_name` in the same PR or via a
cross-linked lynchpin issue merged simultaneously. The lynchpin change
is mechanical: replace column name, no logic changes.

## References

- [#1022](https://github.com/Sinity/polylogue/issues/1022) — umbrella issue
- [#1219](https://github.com/Sinity/polylogue/issues/1219) — column policy decision (consolidated into #1022)
- [#1511](https://github.com/Sinity/polylogue/issues/1511) — write-passive cleanup (adjacent, broader scope)
- `docs/architecture.md` § "Dual Vocabulary Period: Provider and Origin"
- `polylogue/core/provider_identity.py` — vocabulary boundary table
- `polylogue/core/sources.py` — `Source` dataclass and Provider↔Source mapping
