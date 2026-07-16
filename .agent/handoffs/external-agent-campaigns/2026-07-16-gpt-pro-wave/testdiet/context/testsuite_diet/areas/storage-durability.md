---
created: 2026-07-16
purpose: Top-down audit packet for storage, rebuild, and durable-state tests
status: prepared-awaiting-corpus-foundation-and-sensitivity
project: polylogue
---

# Storage durability

Generated dossier:
[`../dossiers/incremental-rebuild-equivalence.md`](../dossiers/incremental-rebuild-equivalence.md).

## Exact first-slice authority

- `polylogue/storage/sqlite/archive_tiers/write.py` accepted normalized writes;
- `polylogue/operations/specs.py` high-level rebuild/reset operations;
- `polylogue/cli/commands/maintenance/_rebuild_index.py` public rebuild route;
- `devtools/index_fast_forward.py` derived fast-forward comparison;
- `polylogue/storage/insights/session/rebuild.py` insight rebuild;
- FTS/index materializers reached through daemon convergence.

The planned survivor is
`tests/unit/storage/test_incremental_rebuild_equivalence.py` over the semantic
micro corpus, using `tests/infra/semantic_facts.py:ArchiveFacts` only where its
facts are independent of storage internals. Existing candidate files include
`test_index_fast_forward_lifecycle.py`,
`test_session_insight_rebuild_progress.py`,
`test_convergence_stale_to_healthy.py`, and focused FTS rebuild tests. Durable
migration/backup, blob crash safety, lineage, and user-overlay tests retain
separate obligations and are excluded from this subtraction.

## Scope and scale

`tests/unit/storage` contains 127 Python files, about 49.2k nonblank lines, and
roughly 1,451 test/class declarations. This is too large for file-by-file
cleanup. Partition it by state obligation:

1. normalized write/read identity and content-hash idempotency;
2. full replace/append/lineage composition;
3. source/user durable-tier atomicity, migration, backup, and recovery;
4. derived index/FTS/insight rebuild equivalence;
5. blob acquisition/reference/GC crash windows;
6. query selection and work bounds;
7. user assertions/overlays remaining independent of reimport identity;
8. embeddings/vector state and catch-up.

## Strong existing anchors

- `tests/property/test_write_path_state_machine.py` already drives the real
  SQLite write/read routes with an independent logical transcript model. Extend
  this; do not replace it with the unused `RepositoryLifecycleHarness`.
- focused durable migration, blob integrity/GC, lineage normalization, FTS, and
  repository lifecycle suites already exist. Their unique crash/security
  obligations must be preserved even if local implementation tests disappear.
- live read/write amplification integration tests provide real counter seams
  that scale laws can reuse.

## Stronger cluster laws

- **incremental = rebuild:** acquire/write an archive incrementally, rebuild
  every derived tier from durable inputs, and compare public facts plus content
  hashes—not raw row order or private tables where representation may differ;
- **restart equivalence:** interrupt between durable commit and derived work,
  reopen, converge, and compare with uninterrupted execution;
- **idempotent replay:** identical artifacts and reordered independent artifacts
  yield identical logical archive facts and bounded additional writes;
- **overlay independence:** tags/corrections/annotations do not change ingest
  content identity or force a reimport;
- **lineage law:** prefix-sharing stores one logical prefix, unresolved parents
  later repair correctly, and cycles quarantine without corrupting readable
  tails;
- **GC safety:** every controlled interleaving of blob write, row commit, lease,
  generation snapshot, and GC preserves referenced bytes and eventually
  collects unreferenced bytes;
- **selectivity law:** irrelevant archive growth preserves results and bounded
  SQLite VM work for indexed reads.

Use deterministic failpoints/barriers at transaction and process boundaries.
Avoid sleeps and assertions about private helper call order.

## Audit targets

The source/AST census flags storage tests that inspect SQL or source text. Some
are valuable static security checks—e.g. preventing unsafe SQL construction—or
schema policy checks. Others may merely pin table/module spelling. Adjudicate
them by asking whether the invariant can be violated while behavior tests still
pass and whether the scanner parses a real risk rather than a refactoring diff.

Run per-test coverage contexts over one obligation cluster at a time, then use
focused mutation to identify repeated writer/reader tests that reach the same
arcs without distinct oracles. Do not forecast LOC until each of the eight
obligations has a named survivor set.

## First implementation slice

Start with incremental-versus-rebuild equivalence over one identical realized
workload identity and its independently planted public facts.
It crosses durable source data, normalized index state, FTS/insights, and public
reads, and can absorb many narrow rebuild/refresh/parity checks. Prove it with
mutations that skip one materializer, leave stale FTS, or incorrectly include
user overlay state in ingest identity. Only then delete dominated tests.

Sensitivity must also reorder independent source replay. The comparison is
over public/logical facts and declared content hashes, not private row order.

Permitted worker command:

```bash
devtools test tests/unit/storage/test_incremental_rebuild_equivalence.py
```
