# HANDOFF — `polylogue-wmsc`

## Outcome

This package implements one monotonic embedding-freshness authority across selection, publication, failure handling, and status reporting. The implementation introduces a storage-neutral `DerivationKey` value shape, maps archive embeddings onto a complete source/recipe/output identity, makes all four production selectors consume one exact predicate, and makes success/error/resolution writes conditional on the captured derivation generation and key.

The central behavioral rule is now:

> An eligible archive session is fresh only when the current embeddable source snapshot and configured recipe produce the same desired key as a succeeded active generation, and every materialized message row belongs to that generation with matching content and recipe identity.

`needs_reindex`, counts, timestamps, and error strings remain compatibility/status projections. None can independently prove freshness.

## Snapshot identity

- Supplied snapshot: `/mnt/data/polylogue-all.tar(133).gz`
- Snapshot SHA-256: `ec3fa193b2f99a6daee0bc620af4f69b5728834e99f8196792a17c3fbae11155`
- Mission file: `/mnt/data/misc-02-embedding-freshness(2).md`
- Mission SHA-256: `6bed92923ef359f274a4c4362d2258bb5f9fe582f195c093d982415eade05a15`
- Authoritative commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- Snapshot branch/ref: `master` / `origin/master`
- Commit subject: `fix(repair): harden raw authority convergence (#3046)`
- Commit time: `2026-07-17T18:55:47+02:00`
- Draft branch created locally: `work/polylogue-wmsc`
- Initial tracked delta: clean. The supplied project-state snapshot carried untracked/snapshot-only material; none of it is copied into this result.

`PATCH.diff` is against the exact commit above. It was applied with `git apply --check` and then applied to a detached clean worktree at that commit.

## Authority and evidence inspected

### Repository instructions and architecture

- `CLAUDE.md` / `AGENTS.md`: substrate-first ownership; derived-tier DDL is canonical; `embeddings.db` is rebuildable; schema changes use canonical DDL plus a version bump rather than an in-place migration chain; generated topology must be regenerated and checked.
- `.agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/testdiet/context/testsuite_diet/architecture/05-derived-freshness.md`: exact `DerivationKey` plus active generation is the freshness proof; domain ledgers remain separate; old receipts cannot clear newer debt; retryability is orthogonal.
- `docs/schema.md`, archive-tier bootstrap, topology projection/status, and the embedding lifecycle/status documentation.

### Beads read in full

- `polylogue-wmsc` — direct authority. Its accepted top-level value shape is `subject`, `source_identity`, `recipe_identity`, and `output_contract`; generation, producer/resource data, eligibility/privacy, lifecycle, and result integrity remain separate.
- `polylogue-303r.7` — complete embedding recipe identity: canonicalization, record/chunk selector, chunking version, provider, model/revision, dimensions, task/input type, normalization, tool implementation, and input/schema version. `output_hash` is result integrity, not lookup identity.
- `polylogue-1xc.12` — the future FTS consumer reuses the value shape only and keeps an FTS-owned ledger/lifecycle.
- `polylogue-iqd3` — terminal-error `needs_reindex` clobber race, superseded by `wmsc`.
- `polylogue-0k6` — same-ID/same-count changed-text full-replace regression, absorbed by `wmsc`.

### Source and production routes

The inspection followed the real path rather than only the named files:

- `polylogue/storage/embeddings/materialization.py`: embeddable-message relation, archive selector, exact status count, source snapshot, provider invocation, session publication, legacy fallback.
- `polylogue/storage/sqlite/archive_tiers/embedding_write.py`: vector/meta writes, session status projection, failure ledger, acknowledgement/requeue/supersession.
- `polylogue/storage/sqlite/archive_tiers/embeddings.py`: canonical embeddings DDL and version.
- `polylogue/daemon/convergence_stages.py`: per-source/session convergence and config-change reconciliation.
- `polylogue/daemon/embedding_backlog.py`: daemon bulk catch-up.
- `polylogue/cli/commands/embed.py`: operator backfill and failure resolution.
- `polylogue/storage/embeddings/preflight.py`: cost/window preflight.
- `polylogue/storage/embeddings/status_payload.py`: readiness/status counts and bounded exact detail.
- Embedding orphan reconciliation, stats, CLI/API/MCP status consumers, sqlite-vec initialization, archive-tier bootstrap, and generated topology ownership.

### Relevant history

- `d07627ebeffb41100d4da82a1390dd8645d4dcae` — added content-hash detection for same-count message edits, but only one caller consumed it.
- `b998ec4cfc28cde2606c89631a451b3b0133fd47` — refined clean archive status rows against exact content.
- `16fdb0fcacba55f9640222a3e0e52984f6c414aa` — guarded success against a model-change race; its own commit message explicitly recorded the terminal-error gap and intentionally avoided a schema generation.
- `4177544ce7c24b51ef56a8e875116c0e6ae9978c` — introduced inspectable embedding failure lifecycle and operator resolution.

Current source and the later `wmsc`/Diet decision supersede the earlier model-string-only race fix: a complete key and generation are required because model identity alone cannot cover content, selector, canonicalization, dimensions, schema, or output-contract changes.

## `DerivationKey` type specification

New file: `polylogue/storage/derivation_identity.py`.

### Value shape

```text
DerivationKey = {
  subject: DerivationSubject {
    reference: non-empty logical subject reference,
    grain: non-empty output grain,
  },
  source_identity: DerivationIdentity {
    namespace: non-empty versioned namespace,
    fields: unique field-name/value pairs sorted by field name,
  },
  recipe_identity: DerivationIdentity { ...same canonical identity shape... },
  output_contract: DerivationIdentity { ...same canonical identity shape... },
}
```

`DerivationKeyLike` is a runtime-checkable structural protocol with those four attributes and `digest()`. `polylogue-1xc.12` can consume this protocol/value vocabulary without importing embedding tables, generation state, scheduling, or lifecycle.

### Canonical encoding and digest

- UTF-8 canonical JSON with sorted object keys and compact separators.
- Identity fields are frozen in lexical field-name order; duplicate/unsorted direct construction is rejected.
- Bytes encode as `{"bytes_hex":"..."}` rather than implementation-dependent JSON coercion.
- Non-finite floats are rejected.
- Component digests and the final key use SHA-256 with explicit domain separators:
  - `polylogue.derivation-subject.v1\0`
  - `polylogue.derivation-identity.v1\0`
  - `polylogue.derivation-key.v1\0`
- The final key digest composes the four 32-byte component digests in the declared top-level order.

### Explicit exclusions

The key does **not** contain:

- attempt/generation/attempt ID;
- scheduler, producer, host, resource, timing, cost, or provider receipt data;
- authorization, privacy, retention, deletion, or other eligibility policy;
- retryability or lifecycle state;
- result/output hash or vector location.

Those remain domain-owned attempt, eligibility, lifecycle, and integrity records. No universal derivation table, scheduler, or lifecycle was added.

## Embedding identity mapping

New file: `polylogue/storage/embeddings/identity.py`.

### Subject

- Session materialization: `reference = session_id`, `grain = archive-message-vectors`.
- Per-message materialization: `reference = message_id`, `grain = archive-message-vector`.

### Source identity

The source snapshot is the exact set of currently embeddable archive messages. Each item is `(message_id, current messages.content_hash)`; pairs are sorted by message ID and length-framed into a domain-separated SHA-256 digest. The current content hash already covers the canonical stored role/type/material-origin and text/block content. This catches same-ID, same-count full replacements.

The source identity namespace is `polylogue.embedding.source.v1` and includes:

- canonicalization: `ordered-message-id-content-hash-v1`;
- `message_set_sha256`: the exact session source digest.

### Recipe identity

`EmbeddingRecipe` declares every computational field required by `polylogue-303r.7`:

1. `canonicalization = ordered-text-block-prose-v1`
2. `record_selector = authored-user-assistant-prose-v1`
3. `chunking_version = one-vector-per-message-v1`
4. `provider = voyage`
5. configured model
6. model revision semantics
7. configured dimensions
8. task
9. input type
10. normalization
11. tool implementation
12. input/index schema version

A parameterized mutation test changes each field independently and proves the recipe digest changes.

### Output contract

The separate output contract records dense-vector kind, `float32` element type, dimensions, and output schema version. Dimensions are intentionally present in both the computational recipe and output contract: they affect both the operation and the shape consumers may accept.

### SQLite identity helpers

The embedding domain registers connection-local helpers used by the shared predicate:

- `polylogue_embedding_source_hash(message_id, content_hash)` aggregate;
- `polylogue_embedding_derivation_key(session_id, source_hash, recipe_hash, output_contract_hash)`;
- `polylogue_embedding_message_key(message_id, content_hash, recipe_hash, output_contract_hash)`.

They use the same Python value semantics and normalize SQLite BLOB/TEXT identity values without reinterpreting them.

## One shared stale predicate

`_archive_embedding_freshness_predicate()` in `materialization.py` is the only v3 archive freshness classification used by selection and status counts.

### Desired snapshot

A single statement builds:

- `desired_messages`: the production `archive_embeddable_messages_relation`, including the authored user/assistant selector, material-origin checks, text-block reconstruction, and 20-character prose floor;
- `desired_sessions`: exact per-session message count and source digest.

The index path uses the existing embedding-prose/message indexes. Status and derivation state join by their primary keys; message metadata joins by its `message_id` primary key.

### Fresh

A session is fresh only when all are true:

- the derivation-state session exists;
- its derivation key equals the current desired session key;
- stored source, recipe, and output-contract hashes equal the desired identities;
- `attempt_state = succeeded`;
- compatibility status exists, is clean, and has the exact desired count;
- derivation-state count equals the desired count;
- every desired message has metadata with matching current content hash, recipe hash, message derivation key, generation, and no per-message reindex flag.

### Blocked

A session is blocked only when the exact **current** key is in `failed_terminal`. A terminal disposition for an older key does not block a newer content/config key.

### Pending

`pending = NOT fresh AND NOT blocked`.

This means missing ledger rows, source changes, recipe/output changes, retryable failures, superseded generations, missing/mismatched message metadata, and compatibility projection drift all select work. Rebuild mode deliberately selects every eligible session.

### Legacy compatibility

Pre-v3/minimal fixtures continue through the prior content-aware fallback. The public `include_stale_checks` switch was removed; no production caller can opt out. Legacy failure tables lacking v3 identity columns remain readable and operator-resolvable, but new keyed sessions refuse unscoped terminal projections.

## Four-caller predicate unification map

| Production consumer | Previous behavior | New behavior |
|---|---|---|
| Per-source/session convergence (`_archive_pending_embedding_session_ids`) | Used the content-aware selector, but config reconciliation ran against `index.db` although state lives in sibling `embeddings.db`. | Uses the same exact-key selector and reconciles the recipe on the actual sibling embeddings tier before selecting. |
| Daemon bulk backlog (`_drain_archive_embedding_backlog_once`) | Passed `include_stale_checks=False`; trusted missing row / `needs_reindex` only. | Passes the configured `EmbeddingRecipe`; exact-key predicate selects content and recipe drift. |
| Manual operator backfill (`_run_archive_backfill`) | Passed `include_stale_checks=False`. | Passes the preflight model/dimension recipe to the same selector. |
| Preflight (`build_preflight_report` → `_select_archive_pending_window`) | Passed `include_stale_checks=False`, so cost/window estimates disagreed with content-aware convergence. | Passes the configured recipe to the same selector, so estimated and executed windows share semantics. |

Additional consumers updated:

- `count_archive_embedding_session_state()` now uses the same predicate and returns fresh, pending, and exact-current blocked counts.
- Default and detail status payloads call that exact count under existing SQLite progress-handler time bounds; the legacy aggregate is only a timeout fallback.
- Daemon catch-up projections and convergence progress tests now reject status-only “freshness.”
- Topology generation classifies `derivation_identity.py` as a storage-root primitive.

## Monotonic write and failure mechanism

### Begin

`begin_embedding_attempt()` computes the desired session key and atomically inserts or advances `embedding_derivation_state.generation`, setting the exact key/source/recipe/output hashes and `attempt_state = pending`. It returns an immutable `ArchiveEmbeddingAttempt` token containing that generation and key. Starting a retry advances generation even for the same key, so concurrent older workers lose terminal-write authority.

### Compute and recheck

`embed_archive_session_sync()` reads and stages all vectors in memory. Before publication it re-reads the current source snapshot and current configured recipe. If either moved, `supersede_embedding_attempt()` conditionally advances only the still-owned generation to the newer key and leaves work pending.

### Success

`complete_embedding_attempt_success()` validates:

- every write belongs to the attempt session;
- unique message IDs;
- exact generation and recipe;
- each message key matches its content and the attempt recipe/output contract;
- the complete staged message set hashes to the attempt source identity.

Within one SQLite transaction it then rechecks the exact pending generation/key, deletes the prior session vectors/meta, writes the full replacement, marks that generation succeeded, updates compatibility status, and resolves active failures. If the guard no longer matches it returns `False` and changes no vector, metadata, status, or lifecycle row.

This also closes the split-tier full-replace trap: same message ID/content replacement leaves exactly one current vector/meta row rather than retaining or duplicating old output.

### Error

`record_embedding_failure()` / `mark_session_embedding_error()` may change derivation/status state only when the supplied attempt token still matches the exact pending generation/key. Retryability changes `attempt_state` (`failed_retryable` versus `failed_terminal`) but never changes identity.

An old attempt still produces inspectable evidence, but its receipt is immediately `superseded` and cannot touch current status. An unscoped error call after a session has acquired a derivation generation is likewise retained only as superseded evidence.

### Operator resolution

`resolve_embedding_failure()` scopes requeue/acknowledge/supersede projection to the failure receipt's generation and key. A generation-zero v2 receipt can mutate legacy status only while the session has no keyed derivation state. This prevents a late acknowledgement of a pre-v3 or older failure from clearing a newer pending generation.

## Schema and rebuild

`EMBEDDINGS_SCHEMA_VERSION` changes from `2` to `3` in the canonical DDL. No migration chain was added.

New/extended schema:

- `embedding_derivation_state`: session primary key, origin, generation, exact derivation/source/recipe/output hashes, attempt state, exact materialized message count, update time;
- `idx_embedding_derivation_pending`;
- `message_embeddings_meta`: recipe hash, per-message derivation key, generation;
- `embedding_failures`: generation, derivation key, source hash, recipe hash.

`docs/schema.md` now reports version 3 and documents the derivation/failure ledgers.

Because `embeddings.db` is a rebuildable-but-expensive tier, an existing v2 file must be moved aside and rebuilt. Do not attempt an in-place `ALTER` chain. Preserve it until postflight/audit is satisfactory because rebuilding incurs provider cost.

Suggested integrator order:

1. Stop the daemon/writers and retain a backup of `embeddings.db`, `embeddings.db-wal`, and `embeddings.db-shm` if present.
2. Apply `PATCH.diff` at commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.
3. Move the v2 embeddings tier and sidecars aside as one set.
4. Start the normal archive initialization path or run a bounded `polylogue ops embed backfill --yes ...`; fresh initialization creates v3.
5. Run focused tests and `devtools verify --quick` in the project devshell.
6. Run the read-only audit below on the live archive before and after rebuild; retain the old tier until the observed drift and expected provider budget are understood.
7. Run `polylogue ops embed status --detail` and daemon convergence postflight.

## Changed files

### New

- `polylogue/storage/derivation_identity.py`
- `polylogue/storage/embeddings/identity.py`
- `tests/unit/storage/test_embedding_freshness_invariant.py`

### Production changes

- `polylogue/storage/embeddings/materialization.py`
- `polylogue/storage/sqlite/archive_tiers/embedding_write.py`
- `polylogue/storage/sqlite/archive_tiers/embeddings.py`
- `polylogue/daemon/convergence_stages.py`
- `polylogue/daemon/embedding_backlog.py`
- `polylogue/cli/commands/embed.py`
- `polylogue/storage/embeddings/preflight.py`
- `polylogue/storage/embeddings/status_payload.py`

### Tests/docs/generated ownership

- `tests/unit/cli/test_embed_status_fast.py`
- `tests/unit/daemon/test_embedding_convergence_progress.py`
- `tests/unit/storage/test_embedding_contracts.py`
- `tests/unit/storage/test_embedding_orphan_reconcile.py`
- `docs/schema.md`
- `devtools/build_topology_projection.py`
- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`

## Acceptance matrix

| Requirement | Result | Evidence |
|---|---|---|
| Storage-neutral typed key; no universal ledger | Complete | New value/protocol file; embedding ledger remains domain-owned. |
| All four selectors share one predicate/snapshot | Complete | Four production-route stale-content tests plus removed bypass argument. |
| Every recipe field changes desired identity | Complete | Twelve-field parameterized mutation test. |
| Later source/recipe generation defeats old success/error | Complete | Source recheck, atomic success guard, config-change/old-error deterministic interleaving, stale-success writer tests. |
| Non-retryable disposition scoped to failed key | Complete | Exact-key blocked predicate and guarded failure/resolution writes. |
| Live census | Query shipped; execution unverified | Read-only production-predicate audit below. No live archive was accessed. |
| Removing caller/field/conditional write fails | Complete | Mutation-named caller tests, recipe-field test, success/error/legacy-resolution race tests. |
| FTS can reuse shape without shared lifecycle | Complete at protocol boundary | `DerivationKeyLike` is storage-neutral; no embedding storage/scheduler types leak into it. FTS ledger implementation remains `polylogue-1xc.12`. |

## Read-only live audit

The exact predicate requires connection-local identity functions, so the audit is a read-only Python harness around one SQL statement rather than a standalone `sqlite3` shell expression. It opens both tiers with `mode=ro`, sets `query_only`, registers the production predicate helpers, and compares:

- the historical bypass predicate (`missing status OR needs_reindex`);
- the former single content-aware predicate (bypass plus count/content mismatch);
- the new exact key/generation/materialization predicate.

Run from the patched repository/devshell:

```bash
python /tmp/polylogue-embedding-freshness-audit.py /path/to/archive-root
```

Save the following as `/tmp/polylogue-embedding-freshness-audit.py`:

```python
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from polylogue.config import load_polylogue_config
from polylogue.storage.embeddings.identity import EmbeddingRecipe
from polylogue.storage.embeddings.materialization import _archive_embedding_freshness_predicate

parser = argparse.ArgumentParser()
parser.add_argument("archive_root", type=Path)
args = parser.parse_args()
root = args.archive_root.resolve()
index_db = root / "index.db"
embeddings_db = root / "embeddings.db"
if not index_db.is_file() or not embeddings_db.is_file():
    raise SystemExit("archive_root must contain index.db and embeddings.db")

cfg = load_polylogue_config()
recipe = EmbeddingRecipe.current(
    model=str(cfg.embedding_model),
    dimensions=int(cfg.embedding_dimension),
)
conn = sqlite3.connect(f"{index_db.as_uri()}?mode=ro", uri=True, timeout=30.0)
try:
    conn.execute("ATTACH DATABASE ? AS embeddings", (f"{embeddings_db.as_uri()}?mode=ro",))
    conn.execute("PRAGMA query_only = ON")
    predicate = _archive_embedding_freshness_predicate(
        conn,
        status_table="embeddings.embedding_status",
        recipe=recipe,
    )
    if predicate is None:
        raise SystemExit("embeddings.db is not schema v3 or index.db lacks exact content identity")

    recipe_hash_sql = f"X'{recipe.recipe_hash.hex()}'"
    output_hash_sql = f"X'{recipe.output_contract_hash.hex()}'"
    desired_key_sql = (
        "polylogue_embedding_derivation_key("
        f"s.session_id, ds.source_hash, {recipe_hash_sql}, {output_hash_sql})"
    )
    key_current = f"""(
        d.session_id IS NOT NULL
        AND d.derivation_key = {desired_key_sql}
        AND d.source_hash = ds.source_hash
        AND d.recipe_hash = {recipe_hash_sql}
        AND d.output_contract_hash = {output_hash_sql}
    )"""
    old_bypass = "(e.session_id IS NULL OR COALESCE(e.needs_reindex, 0) = 1)"
    old_content = f"""(
        {old_bypass}
        OR (
            e.error_message IS NULL
            AND (
                COALESCE(e.message_count_embedded, 0) < ds.message_count
                OR EXISTS (
                    SELECT 1
                    FROM desired_messages AS old_dm
                    JOIN embeddings.message_embeddings_meta AS old_em
                      ON old_em.message_id = old_dm.message_id
                    WHERE old_dm.session_id = s.session_id
                      AND old_dm.content_hash IS NOT NULL
                      AND old_em.content_hash IS NOT NULL
                      AND old_em.content_hash != old_dm.content_hash
                )
            )
        )
    )"""
    same_key_materialization_drift = (
        f"({key_current} AND NOT ({predicate.fresh_sql}) AND NOT ({predicate.blocked_sql}))"
    )
    columns = (
        "eligible_sessions",
        "exact_fresh_sessions",
        "exact_pending_sessions",
        "exact_blocked_sessions",
        "old_bypass_pending_sessions",
        "old_content_aware_pending_sessions",
        "missed_by_old_bypass",
        "missed_by_old_content_aware",
        "missing_derivation_state",
        "source_identity_drift",
        "recipe_or_output_drift",
        "same_key_materialization_drift",
    )
    row = conn.execute(
        f"""
        {predicate.cte_sql}
        SELECT
            COUNT(*),
            COALESCE(SUM(CASE WHEN {predicate.fresh_sql} THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN {predicate.pending_sql} THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN {predicate.blocked_sql} THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN {old_bypass} THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN {old_content} THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN {predicate.pending_sql} AND NOT {old_bypass} THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN {predicate.pending_sql} AND NOT {old_content} THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN d.session_id IS NULL THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE
                WHEN d.session_id IS NOT NULL AND d.source_hash != ds.source_hash THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE
                WHEN d.session_id IS NOT NULL
                 AND (d.recipe_hash != {recipe_hash_sql}
                      OR d.output_contract_hash != {output_hash_sql})
                THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN {same_key_materialization_drift} THEN 1 ELSE 0 END), 0)
        FROM desired_sessions AS ds
        JOIN sessions AS s ON s.session_id = ds.session_id
        {predicate.join_sql}
        """
    ).fetchone()
    assert row is not None
    result = {name: int(value or 0) for name, value in zip(columns, row, strict=True)}
    result["partition_check"] = (
        result["exact_fresh_sessions"]
        + result["exact_pending_sessions"]
        + result["exact_blocked_sessions"]
        == result["eligible_sessions"]
    )
    print(json.dumps(result, indent=2, sort_keys=True))
finally:
    conn.close()
```

Interpretation:

- `partition_check` must be `true`; exact fresh + pending + blocked must equal eligible sessions.
- `missed_by_old_bypass` is the concrete silent-debt population hidden from daemon backlog, manual backfill, and preflight by the removed bypass.
- `missed_by_old_content_aware` is debt even the former per-source content check could not prove, normally recipe/output/ledger/generation drift or compensating metadata defects.
- `source_identity_drift` counts current content sets that differ from the ledger source snapshot.
- `recipe_or_output_drift` counts current configuration/output-contract mismatch.
- `same_key_materialization_drift` counts sessions whose desired key is current but status/message materialization cannot prove a successful complete result.
- `exact_blocked_sessions` is current-key terminal debt and is intentionally not automatic retry work.
- A v2 archive exits with an explicit schema-v3 requirement. Run the query on a copy rebuilt/initialized with the patch, or adapt the old-predicate portions to the retained v2 file for a before/after census. Do not mutate the live tier to make the query run.

The harness was exercised against a same-ID/same-count changed-content fixture. It returned one exact pending session, zero old-bypass pending sessions, one `missed_by_old_bypass`, one source-identity drift, and `partition_check=true`.

## Risks and limitations

- No operator live daemon, live archive, Voyage credentials, or production provider calls were available. The audit and rebuild/postflight are integrator work.
- The full project dev dependency environment could not be hydrated because the container had DNS failure reaching the Python package mirror. Consequently full Ruff, strict mypy, `devtools verify --quick`, and the complete test suite are unverified here.
- Focused tests used the snapshot's existing runtime and a test-only `worker_id` fixture/Hypothesis import shim because `pytest-xdist`, Hypothesis, and timeout plugins were absent. Production imports and behavior were not shimmed.
- Default status now attempts the exact predicate under a 5-second SQLite progress-handler budget, falling back to the legacy aggregate if interrupted. A very large live archive should be measured for query-plan/runtime behavior.
- The schema bump intentionally forces a rebuild rather than preserving v2 vectors in place. This is safer semantically but can have material provider cost.
- The low-level one-off vector upsert remains available for narrow direct callers, but it cannot establish v3 session freshness; only generation-guarded session publication can do so.

## Verification completed

- `174 passed` across the focused selector, CLI status, daemon convergence/readiness/metrics, API/MCP status, writer, lifecycle, and orphan-reconciliation files.
- Clean detached-worktree patch application followed by `29 passed` for the new invariant and canonical embedding-writer suites.
- Changed Python files compile with `compileall`.
- `git diff --check` passes.
- Topology projection/status regenerated; `devtools render topology-status --check` passes.
- `devtools verify topology --json`: `blocking=false`, zero orphans, zero missing, zero conflicts, zero kernel-rule findings; nine pre-existing storage-root TBD classifications remain.
- Audit harness fixture result matched the expected stale-content classification.
- `uv sync --extra dev --frozen` was attempted and failed only while downloading `virtualenv==21.2.0` because DNS resolution for `files.pythonhosted.org` was unavailable.

See `TESTS.md` for exact commands, test dependencies, anti-vacuity mutations, and unverified checks. See `EVIDENCE.md` for source/Bead/history findings and contradictions.
