"""Sharded from-empty index rebuild (polylogue-pzxm).

Two things are tested at the function level here (fast, no full rebuild
engine involved):

- :func:`shard_raw_ids` -- deterministic, exhaustive, non-overlapping
  partitioning.
- The cross-shard hard part the bead calls out explicitly: a parent/child
  pair whose two sessions are written into DIFFERENT shard databases never
  gets to resolve `session_links`/`parent_session_id`/`root_session_id`
  during either shard's own replay (the counterpart row does not exist in
  that shard yet). :func:`merge_shards_into_target` +
  :func:`resolve_cross_shard_session_graph` must produce the exact same
  resolved state a single-writer replay of both sessions (in either order)
  would have produced.

The full engine-level equivalence proof (sequential vs. sharded
``rebuild_index_from_source_sync``, PR #3469 MANIFESTS IDENTICAL pattern) and
the K=4/8 benchmark live in ``tests/benchmarks/test_sharded_rebuild.py`` --
both drive the real rebuild engine end to end and are slower.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, BranchType, Provider
from polylogue.maintenance.rebuild_index import RebuildProvenanceContext
from polylogue.maintenance.schema_inference_gate import (
    rebuild_source_revision_snapshot,
    validate_schema_inference_receipt,
)
from polylogue.maintenance.sharded_rebuild import (
    merge_shards_into_target,
    resolve_cross_shard_session_graph,
    shard_raw_ids,
)
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt


class _NoopProvenance:
    def validate(self) -> None:
        return

    def validate_cleanup(self) -> None:
        return


_NOOP_PROVENANCE = _NoopProvenance()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def test_shard_raw_ids_is_deterministic_exhaustive_and_disjoint(tmp_path: Path) -> None:
    raw_ids = [f"raw-{i}" for i in range(200)]
    # No source.db at tmp_path -- shard_raw_ids falls back to raw_id as its
    # own cohort key, exercising the no-cohort-evidence path deliberately.
    first = shard_raw_ids(tmp_path, raw_ids, 8)
    second = shard_raw_ids(tmp_path, raw_ids, 8)
    assert first == second  # deterministic: same hash, same bucket every call
    assert len(first) == 8
    flattened = [raw_id for bucket in first for raw_id in bucket]
    assert sorted(flattened) == sorted(raw_ids)  # exhaustive
    assert len(set(flattened)) == len(raw_ids)  # disjoint (no duplication)
    # Every bucket gets some share of a 200-item population across 8 buckets
    # (not a strict balance guarantee, but a degenerate all-in-one-bucket
    # hash would be a real bug worth catching here).
    assert all(bucket for bucket in first)


def test_shard_raw_ids_rejects_non_positive_shard_count(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="shard_count"):
        shard_raw_ids(tmp_path, ["a"], 0)


def _parent_session() -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="parent-session",
        updated_at="2026-01-01T00:00:01+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="p1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="parent")],
            )
        ],
    )


def _child_session() -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="child-session",
        parent_session_provider_id="parent-session",
        branch_type=BranchType.SIDECHAIN,
        updated_at="2026-01-01T00:00:02+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="c1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="child")],
            )
        ],
    )


def test_cross_shard_parent_child_resolves_after_merge(tmp_path: Path) -> None:
    """The polylogue-pzxm 'hard part': a fork split across two shards.

    Shard A gets only the parent; shard B gets only the child. Neither
    shard's own replay can resolve the link (the counterpart row does not
    exist locally), matching
    ``test_archive_tiers_writer_stores_unresolved_link_when_parent_absent``'s
    single-writer baseline for a still-missing parent. After merge + one
    ``resolve_cross_shard_session_graph`` pass, the result must match
    ``test_archive_tiers_writer_resolves_parent_link_when_parent_already_exists``'s
    single-writer baseline for an already-present parent.
    """
    shard_a = tmp_path / "shard-a" / "index.db"
    shard_b = tmp_path / "shard-b" / "index.db"
    shard_a.parent.mkdir(parents=True)
    shard_b.parent.mkdir(parents=True)

    conn_a = _connect(shard_a)
    parent_id = write_parsed_session_to_archive(conn_a, _parent_session(), bulk_fts=True, bulk_build=True)
    conn_a.commit()
    conn_a.close()

    conn_b = _connect(shard_b)
    child_id = write_parsed_session_to_archive(conn_b, _child_session(), bulk_fts=True, bulk_build=True)
    conn_b.commit()
    child_row_before = conn_b.execute(
        "SELECT parent_session_id, root_session_id FROM sessions WHERE session_id = ?", (child_id,)
    ).fetchone()
    link_row_before = conn_b.execute(
        "SELECT resolved_dst_session_id FROM session_links WHERE src_session_id = ?", (child_id,)
    ).fetchone()
    conn_b.close()

    # Sanity: within its own shard, the child's link is genuinely unresolved
    # -- this is the gap the merge + resolve pass must close, not a no-op.
    assert child_row_before["parent_session_id"] is None
    assert link_row_before["resolved_dst_session_id"] is None

    target = tmp_path / "target" / "index.db"
    target.parent.mkdir(parents=True)
    conn_target = _connect(target)
    conn_target.commit()
    conn_target.close()

    merge_shards_into_target(target, [shard_a, shard_b], provenance=cast(RebuildProvenanceContext, _NOOP_PROVENANCE))
    resolve_cross_shard_session_graph(target, provenance=cast(RebuildProvenanceContext, _NOOP_PROVENANCE))

    with sqlite3.connect(target) as conn:
        conn.row_factory = sqlite3.Row
        child_row = conn.execute(
            "SELECT parent_session_id, root_session_id, branch_type FROM sessions WHERE session_id = ?",
            (child_id,),
        ).fetchone()
        link_row = conn.execute(
            "SELECT resolved_dst_session_id, status FROM session_links WHERE src_session_id = ?",
            (child_id,),
        ).fetchone()
        thread_rows = conn.execute(
            "SELECT session_id, position FROM thread_sessions WHERE thread_id = ? ORDER BY position",
            (parent_id,),
        ).fetchall()

    assert dict(child_row) == {
        "parent_session_id": parent_id,
        "root_session_id": parent_id,
        "branch_type": "sidechain",
    }
    assert dict(link_row) == {"resolved_dst_session_id": parent_id, "status": None}
    assert [dict(row) for row in thread_rows] == [
        {"session_id": parent_id, "position": 0},
        {"session_id": child_id, "position": 1},
    ]


def test_merge_shards_into_target_raises_on_foreign_key_violation(tmp_path: Path) -> None:
    """A shard whose merged content would dangle an FK must fail loudly, not silently promote."""
    import pytest

    shard_a = tmp_path / "shard-a" / "index.db"
    shard_a.parent.mkdir(parents=True)
    conn_a = _connect(shard_a)
    write_parsed_session_to_archive(conn_a, _child_session(), bulk_fts=True, bulk_build=True)
    conn_a.commit()
    # Corrupt the shard by pointing the child's session_links row at a
    # resolved parent that does not exist in ANY shard -- this must never
    # happen from a real replay (resolution only ever sets
    # resolved_dst_session_id to an existing row's id), but proves the
    # foreign_key_check safety net actually fires rather than being dead
    # code.
    conn_a.execute("PRAGMA foreign_keys = OFF")
    conn_a.execute(
        "UPDATE sessions SET parent_session_id = 'claude-code-session:missing-parent' WHERE native_id = ?",
        ("child-session",),
    )
    conn_a.commit()
    conn_a.close()

    target = tmp_path / "target" / "index.db"
    target.parent.mkdir(parents=True)
    conn_target = _connect(target)
    conn_target.commit()
    conn_target.close()

    with pytest.raises(RuntimeError, match="foreign-key violations"):
        merge_shards_into_target(target, [shard_a], provenance=cast(RebuildProvenanceContext, _NOOP_PROVENANCE))


def test_merge_revalidates_external_evidence_before_target_mutation(tmp_path: Path) -> None:
    """A source-evidence change at the merge boundary leaves the target empty.

    Anti-vacuity: this invokes the production ``merge_shards_into_target`` with
    real index rows and a real receipt context. Removing its pre-insert
    provenance validation lets the shard row reach ``target`` before the
    caller can observe the drift.
    """
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"receipt-source"}}\n',
            source_path="receipt-source.jsonl",
            acquired_at_ms=1,
        )
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    evidence = validate_schema_inference_receipt(root, receipt_path)
    provenance = RebuildProvenanceContext(
        root=root,
        receipt_path=receipt_path,
        source_snapshot=rebuild_source_revision_snapshot(root),
        consumed_evidence=evidence,
    )

    shard = tmp_path / "shard" / "index.db"
    shard.parent.mkdir(parents=True)
    conn = _connect(shard)
    write_parsed_session_to_archive(conn, _parent_session(), bulk_fts=True, bulk_build=True)
    conn.commit()
    conn.close()
    target = tmp_path / "target" / "index.db"
    target.parent.mkdir(parents=True)
    conn = _connect(target)
    conn.commit()
    conn.close()

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    origin = receipt["ground_truth_inputs"]["origins"]["codex-session"]
    external_path = Path(origin["declared_roots"][0]) / origin["external_inventory"][0]["relative_path"]
    external_path.write_bytes(b"drifted-before-merge")

    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        merge_shards_into_target(target, [shard], provenance=provenance)

    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
