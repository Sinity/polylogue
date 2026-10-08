"""Transactional global raw-existence proof for live page admission."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import CursorAuthorityBlockedError, LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage import frontier_existence
from polylogue.storage.raw_retention import raw_frontier_blocked_raw_ids, raw_frontier_blocked_selected_paths
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_ingest_cursor
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture


def _raw(root: Path, raw_id: str, *, path: Path | None = None, logical_key: str | None = None) -> None:
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        conn.execute(
            "INSERT INTO raw_sessions(raw_id, origin, source_path, canonical_source_path, blob_hash, blob_size, acquired_at_ms, "
            "logical_source_key, revision_kind, source_revision, acquisition_generation, revision_authority) "
            "VALUES (?, 'codex-session', ?, ?, ?, 1, 1, ?, 'full', ?, 0, 'byte_proven')",
            (
                raw_id,
                str(path or root / f"{raw_id}.jsonl"),
                str((path or root / f"{raw_id}.jsonl").resolve()),
                bytes(32),
                logical_key or raw_id,
                raw_id,
            ),
        )


def _session(root: Path, raw_id: str, number: int) -> None:
    with closing(sqlite3.connect(root / "index.db")) as conn, conn:
        conn.execute(
            "INSERT INTO sessions(native_id, origin, raw_id, title, content_hash) "
            "VALUES (?, 'codex-session', ?, 'session', ?)",
            (f"session-{number}", raw_id, bytes(32)),
        )


def _head(root: Path, raw_id: str) -> None:
    with closing(sqlite3.connect(root / "index.db")) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO raw_revision_heads("
            "logical_source_key, session_id, accepted_raw_id, accepted_source_revision, "
            "accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
            "acquisition_generation, decided_at_ms) "
            "VALUES ('codex:head', 'codex-session:session-1', ?, ?, ?, 'semantic', 1, 0, 1)",
            (raw_id, raw_id, bytes(32)),
        )


def test_changed_keys_catch_unrelated_external_source_delete_after_own_writes(tmp_path: Path) -> None:
    """A prior own commit cannot acknowledge an unrelated external deletion."""
    initialize_active_archive_root(tmp_path)
    _raw(tmp_path, "unrelated")
    _session(tmp_path, "unrelated", 1)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    _raw(tmp_path, "own")
    _session(tmp_path, "own", 2)
    with sqlite3.connect(tmp_path / "source.db") as external:
        external.execute("DELETE FROM raw_sessions WHERE raw_id = 'unrelated'")
    assert "missing" in str(frontier_existence.raw_existence_block_reason(tmp_path))


def test_index_insert_and_reference_update_are_observed_from_external_connection(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _raw(tmp_path, "present")
    _session(tmp_path, "present", 1)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    with sqlite3.connect(tmp_path / "index.db") as external:
        external.execute("UPDATE sessions SET raw_id = 'absent' WHERE native_id = 'session-1'")
    assert "missing" in str(frontier_existence.raw_existence_block_reason(tmp_path))
    with sqlite3.connect(tmp_path / "index.db") as external:
        external.execute("UPDATE sessions SET raw_id = 'present' WHERE native_id = 'session-1'")
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    _session(tmp_path, "another-absent", 2)
    assert "missing" in str(frontier_existence.raw_existence_block_reason(tmp_path))


def test_head_replace_and_source_key_update_are_observed(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _raw(tmp_path, "present")
    _session(tmp_path, "present", 1)
    _head(tmp_path, "present")
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    _head(tmp_path, "missing-head")
    assert "missing" in str(frontier_existence.raw_existence_block_reason(tmp_path))
    _head(tmp_path, "present")
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    with sqlite3.connect(tmp_path / "source.db") as external:
        external.execute("UPDATE raw_sessions SET raw_id = 'moved' WHERE raw_id = 'present'")
    assert "missing" in str(frontier_existence.raw_existence_block_reason(tmp_path))


def test_rollback_does_not_publish_change_and_pruning_reproves(tmp_path: Path) -> None:
    """A truncated journal forces a full re-proof, never a trusted skip.

    Anti-vacuity: advance the old certificate past the truncation (the
    incremental branch) and the pruned re-pointing to an absent raw is never
    examined, so admission reports healthy.
    """
    initialize_active_archive_root(tmp_path)
    _raw(tmp_path, "present")
    _session(tmp_path, "present", 1)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    with sqlite3.connect(tmp_path / "source.db") as external:
        external.execute("DELETE FROM raw_sessions WHERE raw_id = 'present'")
        external.rollback()
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    # An unconsumed re-pointing to a raw that source.db lacks is journaled and
    # then pruned before this process consumes it.
    with sqlite3.connect(tmp_path / "index.db") as external:
        external.execute("UPDATE sessions SET raw_id = 'absent' WHERE native_id = 'session-1'")
        external.execute(
            "DELETE FROM raw_existence_changes WHERE sequence = (SELECT MAX(sequence) FROM raw_existence_changes)"
        )
    assert "missing" in str(frontier_existence.raw_existence_block_reason(tmp_path))


def test_consumed_watermarks_follow_the_healthy_certificate(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    assert frontier_existence.consumed_watermarks(tmp_path) is None
    _raw(tmp_path, "present")
    _session(tmp_path, "present", 1)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    source_mark, index_mark = frontier_existence.consumed_watermarks(tmp_path) or (-1, -1)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert index_mark == conn.execute("SELECT MAX(sequence) FROM raw_existence_changes").fetchone()[0]
    assert source_mark >= 0
    with sqlite3.connect(tmp_path / "source.db") as external:
        external.execute("DELETE FROM raw_sessions WHERE raw_id = 'present'")
    assert frontier_existence.raw_existence_block_reason(tmp_path) is not None
    assert frontier_existence.consumed_watermarks(tmp_path) is None


def test_fully_pruned_consumed_journal_retains_high_watermark(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _raw(tmp_path, "present")
    _session(tmp_path, "present", 1)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    with sqlite3.connect(tmp_path / "index.db") as external:
        external.execute("DELETE FROM raw_existence_changes")
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    frontier_existence._certificates.clear()
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    with sqlite3.connect(tmp_path / "source.db") as external:
        external.execute("DELETE FROM raw_sessions WHERE raw_id = 'present'")
    assert "missing" in str(frontier_existence.raw_existence_block_reason(tmp_path))


def test_trigger_change_and_tier_disappearance_revoke_certificate(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    with sqlite3.connect(tmp_path / "source.db") as external:
        # Migration 006 replaced raw_existence_delete with the frontier journal trigger.
        external.execute("DROP TRIGGER raw_existence_frontier_raw_sessions_delete")
    assert "trigger" in str(frontier_existence.raw_existence_block_reason(tmp_path))
    (tmp_path / "index.db").rename(tmp_path / "index-retired.db")
    assert "unavailable" in str(frontier_existence.raw_existence_block_reason(tmp_path))


def test_replaced_source_and_promoted_index_recheck_all_references(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _raw(tmp_path, "present")
    _session(tmp_path, "present", 1)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    promoted = tmp_path / ".index-generations" / "next" / "index.db"
    promoted.parent.mkdir(parents=True)
    with sqlite3.connect(tmp_path / "index.db") as source, sqlite3.connect(promoted) as target:
        source.backup(target)
    with sqlite3.connect(promoted) as conn:
        conn.execute("UPDATE sessions SET raw_id = 'missing-promoted' WHERE native_id = 'session-1'")
    (tmp_path / ".index-active-pointer").write_text(str(promoted), encoding="utf-8")
    assert "missing" in str(frontier_existence.raw_existence_block_reason(tmp_path))
    (tmp_path / ".index-active-pointer").unlink()
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    original = tmp_path / "source.db"
    with sqlite3.connect(original) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    original.rename(tmp_path / "source-retired.db")
    for suffix in ("-wal", "-shm"):
        (tmp_path / f"source.db{suffix}").unlink(missing_ok=True)
    replacement = tmp_path / "source-replacement.db"

    initialize_runtime_source_fixture(replacement)
    replacement.rename(original)
    assert "missing" in str(frontier_existence.raw_existence_block_reason(tmp_path))


def test_concurrent_delete_during_reconciliation_cannot_advance_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    _raw(tmp_path, "present")
    _session(tmp_path, "present", 1)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    original = frontier_existence._changed_keys
    injected = False

    def interleaved(conn: sqlite3.Connection, schema: str, low: int, high: int) -> set[str]:
        nonlocal injected
        result = original(conn, schema, low, high)
        if not injected:
            injected = True
            with sqlite3.connect(tmp_path / "source.db") as external:
                external.execute("DELETE FROM raw_sessions WHERE raw_id = 'present'")
        return result

    monkeypatch.setattr(frontier_existence, "_changed_keys", interleaved)
    assert "changed during admission" in str(frontier_existence.raw_existence_block_reason(tmp_path))
    assert "missing" in str(frontier_existence.raw_existence_block_reason(tmp_path))


def test_restart_like_cache_loss_performs_complete_check(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _raw(tmp_path, "present")
    _session(tmp_path, "present", 1)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    with sqlite3.connect(tmp_path / "source.db") as external:
        external.execute("DELETE FROM raw_sessions WHERE raw_id = 'present'")
    frontier_existence._certificates.clear()
    assert "missing" in str(frontier_existence.raw_existence_block_reason(tmp_path))


def test_fork_like_pid_change_discards_certificate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initialize_active_archive_root(tmp_path)
    _raw(tmp_path, "present")
    _session(tmp_path, "present", 1)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    complete_checks = 0
    original = frontier_existence._missing_reference

    def count_complete(conn: sqlite3.Connection) -> bool:
        nonlocal complete_checks
        complete_checks += 1
        return original(conn)

    monkeypatch.setattr(frontier_existence, "_missing_reference", count_complete)
    monkeypatch.setattr(frontier_existence, "_pid", -1)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    assert complete_checks == 1


def test_real_fork_discards_inherited_locked_certificate(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    script = """
import os, select, signal, sys, threading
from pathlib import Path
from polylogue.storage import frontier_existence

root = Path(sys.argv[1])
assert frontier_existence.raw_existence_block_reason(root) is None
held = threading.Event()
release = threading.Event()
def hold_lock():
    with frontier_existence._lock:
        held.set()
        release.wait(5)
thread = threading.Thread(target=hold_lock, daemon=True)
thread.start()
assert held.wait(5)
read_fd, write_fd = os.pipe()
child = os.fork()
if child == 0:
    os.close(read_fd)
    result = frontier_existence.raw_existence_block_reason(root)
    os.write(write_fd, b'healthy' if result is None else str(result).encode())
    os._exit(0)
os.close(write_fd)
ready, _, _ = select.select([read_fd], [], [], 3)
if not ready:
    os.kill(child, signal.SIGKILL)
    os.waitpid(child, 0)
    raise AssertionError('forked child blocked on inherited certificate lock')
result = os.read(read_fd, 256)
os.waitpid(child, 0)
release.set()
thread.join(5)
assert result == b'healthy', result
"""
    subprocess.run(
        (sys.executable, "-c", script, str(tmp_path)), check=True, capture_output=True, text=True, timeout=120
    )


def test_selected_chain_refuses_only_its_path_and_keeps_new_path_gap(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    broken = tmp_path / "broken.jsonl"
    healthy = tmp_path / "healthy.jsonl"
    new = tmp_path / "new.jsonl"
    new.write_text("x", encoding="utf-8")
    new_alias = tmp_path / "new-alias.jsonl"
    new_alias.symlink_to(new)
    _raw(tmp_path, "broken", path=broken)
    _raw(tmp_path, "healthy", path=healthy)
    _session(tmp_path, "broken", 1)
    _session(tmp_path, "healthy", 2)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET revision_kind = 'append', predecessor_raw_id = 'lost' WHERE raw_id = 'broken'"
        )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        upsert_ingest_cursor(
            conn,
            source_path=str(new_alias),
            canonical_source_path=str(new_alias.resolve()),
            updated_at_ms=1,
            byte_offset=1,
        )
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    selected = raw_frontier_blocked_selected_paths(tmp_path, (broken, healthy, new))
    assert selected.unattributed_reason is None
    assert selected.source_paths == frozenset({str(broken)})
    assert str(healthy) not in selected.source_paths
    assert str(new) not in selected.source_paths
    assert str(new_alias) in selected.gap_source_paths


def test_selected_raw_is_refused_for_another_component_broken_on_its_path(tmp_path: Path) -> None:
    """A broken chain of another logical source on the same file refuses that file.

    ``healthy`` and ``broken`` are independent components that share one
    physical path. Anti-vacuity: checking only the selected component's heads
    and sessions omits ``broken`` and admits the shared path.
    """
    initialize_active_archive_root(tmp_path)
    shared = tmp_path / "shared.jsonl"
    _raw(tmp_path, "healthy", path=shared, logical_key="codex:healthy")
    _raw(tmp_path, "broken", path=shared, logical_key="codex:broken")
    _session(tmp_path, "healthy", 1)
    _session(tmp_path, "broken", 2)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET revision_kind = 'append', predecessor_raw_id = 'lost' WHERE raw_id = 'broken'"
        )

    blocked = raw_frontier_blocked_raw_ids(tmp_path, ["healthy"])

    assert blocked.unattributed_reason is None
    assert blocked.source_paths == frozenset({str(shared)})


def test_shared_logical_key_refuses_connected_paths_without_refusing_sibling(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    first = tmp_path / "first.jsonl"
    shared = tmp_path / "shared.jsonl"
    sibling = tmp_path / "sibling.jsonl"
    _raw(tmp_path, "first", path=first, logical_key="codex:head")
    _raw(tmp_path, "shared", path=shared, logical_key="codex:head")
    _raw(tmp_path, "sibling", path=sibling)
    _session(tmp_path, "first", 1)
    _session(tmp_path, "sibling", 2)
    _head(tmp_path, "first")
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE raw_revision_heads SET accepted_frontier_kind = 'byte'")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET revision_kind = 'append', predecessor_raw_id = 'lost' WHERE raw_id = 'first'"
        )
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    selected = raw_frontier_blocked_selected_paths(tmp_path, (first, shared, sibling))
    assert selected.unattributed_reason is None
    assert selected.source_paths == frozenset({str(first), str(shared)})


def test_mixed_heads_expand_broken_byte_key_without_semantic_sibling(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    broken = tmp_path / "broken.jsonl"
    shared_byte = tmp_path / "shared-byte.jsonl"
    semantic_sibling = tmp_path / "semantic-sibling.jsonl"
    _raw(tmp_path, "broken", path=broken, logical_key="a-semantic")
    _raw(tmp_path, "shared-byte", path=shared_byte, logical_key="z-byte")
    _raw(tmp_path, "semantic-sibling", path=semantic_sibling, logical_key="a-semantic")
    _session(tmp_path, "broken", 1)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "INSERT INTO raw_session_memberships(raw_id, logical_source_key, provider_session_id, "
            "source_revision, normalized_content_hash, message_count) "
            "VALUES ('broken', 'z-byte', 'session-1', 'broken', ?, 1)",
            (bytes(32),),
        )
        conn.execute(
            "UPDATE raw_sessions SET revision_kind = 'append', predecessor_raw_id = 'lost' WHERE raw_id = 'broken'"
        )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        for key, kind in (("a-semantic", "semantic"), ("z-byte", "byte")):
            conn.execute(
                "INSERT INTO raw_revision_heads(logical_source_key, session_id, accepted_raw_id, "
                "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
                "acquisition_generation, decided_at_ms) VALUES (?, 'codex-session:session-1', "
                "'broken', 'broken', ?, ?, 1, 0, 1)",
                (key, bytes(32), kind),
            )
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    selected = raw_frontier_blocked_selected_paths(tmp_path, (shared_byte, semantic_sibling))
    assert selected.unattributed_reason is None
    assert str(shared_byte) in selected.source_paths
    assert str(semantic_sibling) not in selected.source_paths


def test_one_bad_byte_head_does_not_refuse_another_valid_key_sibling(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    first = tmp_path / "first.jsonl"
    healthy = tmp_path / "healthy.jsonl"
    _raw(tmp_path, "first", path=first, logical_key="a-good")
    _raw(tmp_path, "healthy", path=healthy, logical_key="a-good")
    _session(tmp_path, "first", 1)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        for key in ("a-good", "z-bad"):
            conn.execute(
                "INSERT INTO raw_revision_heads(logical_source_key, session_id, accepted_raw_id, "
                "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
                "acquisition_generation, decided_at_ms) VALUES (?, 'codex-session:session-1', "
                "'first', 'first', ?, 'byte', 1, 0, 1)",
                (key, bytes(32)),
            )
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    selected = raw_frontier_blocked_selected_paths(tmp_path, (healthy,))
    assert selected.unattributed_reason is None
    assert str(first) in selected.source_paths
    assert str(healthy) not in selected.source_paths


def test_selected_path_requires_ops_even_without_retained_raw(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    fresh = tmp_path / "fresh.jsonl"
    fresh.write_text("x", encoding="utf-8")
    (tmp_path / "ops.db").rename(tmp_path / "ops-retired.db")
    selected = raw_frontier_blocked_selected_paths(tmp_path, (fresh,))
    assert "unavailable" in str(selected.unattributed_reason)


def test_external_alias_raw_without_canonical_path_refuses_globally(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    real = tmp_path / "real.jsonl"
    alias = tmp_path / "alias.jsonl"
    real.write_text("x", encoding="utf-8")
    alias.symlink_to(real)
    with sqlite3.connect(tmp_path / "source.db") as external:
        external.execute(
            "INSERT INTO raw_sessions(raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms, "
            "revision_kind, predecessor_raw_id) VALUES ('external', 'codex-session', ?, ?, 1, 1, 'append', 'lost')",
            (str(alias), bytes(32)),
        )
    _session(tmp_path, "external", 1)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    selected = raw_frontier_blocked_selected_paths(tmp_path, (real,))
    assert "canonical path" in str(selected.unattributed_reason)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM raw_sessions WHERE canonical_source_path IS NULL LIMIT 1"
        ).fetchall()
    assert any("idx_raw_sessions_missing_canonical_path" in str(row) for row in plan)


def test_external_alias_cursor_without_canonical_path_refuses_globally(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    real = tmp_path / "real.jsonl"
    alias = tmp_path / "alias.jsonl"
    real.write_text("x", encoding="utf-8")
    alias.symlink_to(real)
    with sqlite3.connect(tmp_path / "ops.db") as external:
        external.execute(
            "INSERT INTO ingest_cursor(source_path, byte_offset, updated_at_ms) VALUES (?, 2, 1)", (str(alias),)
        )
    selected = raw_frontier_blocked_selected_paths(tmp_path, (real,))
    assert "canonical path" in str(selected.unattributed_reason)
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM ingest_cursor WHERE canonical_source_path IS NULL "
            "AND byte_offset IS NOT NULL AND excluded = 0 LIMIT 1"
        ).fetchall()
    assert any("idx_ingest_cursor_missing_canonical_path" in str(row) for row in plan)


def test_mixed_byte_and_membership_container_keeps_byte_cursor_refusal(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    container = tmp_path / "container.jsonl"
    sibling = tmp_path / "sibling.jsonl"
    _raw(tmp_path, "byte", path=container, logical_key="codex:head")
    _raw(tmp_path, "member", path=container, logical_key="membership:head")
    _raw(tmp_path, "sibling", path=sibling)
    _session(tmp_path, "byte", 1)
    _session(tmp_path, "sibling", 2)
    _head(tmp_path, "byte")
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE raw_revision_heads SET accepted_frontier_kind = 'byte'")
        conn.execute(
            "INSERT INTO raw_revision_heads(logical_source_key, session_id, accepted_raw_id, "
            "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
            "acquisition_generation, decided_at_ms) "
            "VALUES ('membership:head', 'codex-session:session-1', 'member', 'member', ?, 'semantic', 1, 0, 1)",
            (bytes(32),),
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "INSERT INTO raw_session_memberships(raw_id, logical_source_key, provider_session_id, "
            "source_revision, normalized_content_hash, message_count) "
            "VALUES ('member', 'membership:head', 'session-1', 'member', ?, 1)",
            (bytes(32),),
        )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        upsert_ingest_cursor(
            conn,
            source_path=str(container),
            canonical_source_path=str(container.resolve()),
            updated_at_ms=1,
            byte_offset=2,
        )
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    selected = raw_frontier_blocked_selected_paths(tmp_path, (container, sibling))
    assert selected.unattributed_reason is None
    assert selected.source_paths == frozenset({str(container)})


def test_selected_cursor_ahead_refuses_alias_and_safe_deferred_tail(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    path = tmp_path / "selected.jsonl"
    alias = tmp_path / "alias.jsonl"
    path.write_text("x", encoding="utf-8")
    alias.symlink_to(path)
    _raw(tmp_path, "selected", path=alias, logical_key="codex:head")
    _session(tmp_path, "selected", 1)
    _head(tmp_path, "selected")
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE raw_revision_heads SET accepted_frontier_kind = 'byte'")
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        upsert_ingest_cursor(
            conn, source_path=str(alias), canonical_source_path=str(alias.resolve()), updated_at_ms=1, byte_offset=2
        )
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    refused = raw_frontier_blocked_selected_paths(tmp_path, (path,))
    assert refused.unattributed_reason is None
    assert str(alias) in refused.source_paths
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("DELETE FROM ingest_cursor WHERE source_path = ?", (str(alias),))
        upsert_ingest_cursor(
            conn, source_path=str(path), canonical_source_path=str(path.resolve()), updated_at_ms=1, byte_offset=2
        )
    differently_spelled = raw_frontier_blocked_selected_paths(tmp_path, (path,))
    assert str(path) in differently_spelled.source_paths
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        upsert_ingest_cursor(
            conn,
            source_path=str(path),
            canonical_source_path=str(path.resolve()),
            updated_at_ms=2,
            byte_offset=2,
            deferred_end_offset=3,
        )
    # A deferred range does not accept its prefix: committed offset 2 is past
    # the accepted head (1), so the cursor is still ahead.
    ahead_deferred = raw_frontier_blocked_selected_paths(tmp_path, (path,))
    assert str(path) in ahead_deferred.source_paths
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        upsert_ingest_cursor(
            conn,
            source_path=str(path),
            canonical_source_path=str(path.resolve()),
            updated_at_ms=3,
            byte_offset=1,
            deferred_end_offset=3,
        )
    safe = raw_frontier_blocked_selected_paths(tmp_path, (path,))
    assert safe.unattributed_reason is None
    assert not safe.source_paths


def test_all_deferred_page_still_refuses_unrelated_missing_raw(tmp_path: Path) -> None:
    """The old all-deferred shortcut skipped an unattributed global refusal."""
    initialize_active_archive_root(tmp_path)
    selected = tmp_path / "selected.jsonl"
    selected.write_text("x\n", encoding="utf-8")
    _raw(tmp_path, "unrelated")
    _session(tmp_path, "unrelated", 1)
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        upsert_ingest_cursor(
            conn,
            source_path=str(selected),
            canonical_source_path=str(selected.resolve()),
            updated_at_ms=1,
            byte_offset=1,
            deferred_end_offset=2,
        )
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=tmp_path),),
        cursor=CursorStore(tmp_path / "ops.db"),
        parser_fingerprint="frontier-test",
    )
    processor.require_cursor_authority([selected])
    with sqlite3.connect(tmp_path / "source.db") as external:
        external.execute("DELETE FROM raw_sessions WHERE raw_id = 'unrelated'")
    with pytest.raises(CursorAuthorityBlockedError, match="missing"):
        processor.require_cursor_authority([selected])


def test_external_delete_during_selected_read_refuses_before_admission_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    selected = tmp_path / "selected.jsonl"
    selected.write_text("x\n", encoding="utf-8")
    _raw(tmp_path, "unrelated")
    _session(tmp_path, "unrelated", 1)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=tmp_path),),
        cursor=CursorStore(tmp_path / "ops.db"),
        parser_fingerprint="frontier-test",
    )
    processor.require_cursor_authority([selected])
    original = processor._blocked_source_paths

    def interleave(paths: list[Path]) -> Any:
        selected_result = original(paths)
        with sqlite3.connect(tmp_path / "source.db") as external:
            external.execute("DELETE FROM raw_sessions WHERE raw_id = 'unrelated'")
        return selected_result

    monkeypatch.setattr(processor, "_blocked_source_paths", interleave)
    with pytest.raises(CursorAuthorityBlockedError, match="missing"):
        processor.require_cursor_authority([selected])


@pytest.mark.parametrize("mutation", ["predecessor", "cursor"])
def test_selected_authority_change_during_read_refuses_same_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    initialize_active_archive_root(tmp_path)
    selected = tmp_path / "selected.jsonl"
    selected.write_text("x\n", encoding="utf-8")
    _raw(tmp_path, "selected", path=selected, logical_key="codex:head")
    _session(tmp_path, "selected", 1)
    _head(tmp_path, "selected")
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE raw_revision_heads SET accepted_frontier_kind = 'byte'")
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        upsert_ingest_cursor(
            conn,
            source_path=str(selected),
            canonical_source_path=str(selected.resolve()),
            updated_at_ms=1,
            byte_offset=1,
        )
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=tmp_path),),
        cursor=CursorStore(tmp_path / "ops.db"),
        parser_fingerprint="frontier-test",
    )
    processor.require_cursor_authority([selected])
    original = processor._blocked_source_paths

    def interleave(paths: list[Path]) -> Any:
        selected_result = original(paths)
        if mutation == "predecessor":
            with sqlite3.connect(tmp_path / "source.db") as external:
                external.execute(
                    "UPDATE raw_sessions SET revision_kind = 'append', predecessor_raw_id = 'lost' "
                    "WHERE raw_id = 'selected'"
                )
        else:
            with sqlite3.connect(tmp_path / "ops.db") as external:
                external.execute("UPDATE ingest_cursor SET byte_offset = 2 WHERE source_path = ?", (str(selected),))
        return selected_result

    monkeypatch.setattr(processor, "_blocked_source_paths", interleave)
    with pytest.raises(CursorAuthorityBlockedError, match="selected frontier authority changed"):
        processor.require_cursor_authority([selected])


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_prune_stage_deletes_only_consumed_journal_rows(tmp_path: Path) -> None:
    """Both consumers must prove a row consumed before the actual stage deletes it."""
    from contextlib import closing

    from polylogue.operations.raw_existence_journal import make_raw_existence_journal_prune_stage
    from polylogue.storage.frontier_inspection import inspect_prepared_raw_authority_frontier
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def seed() -> None:
        initialize_active_archive_root(tmp_path)
        _raw(tmp_path, "present")
        _session(tmp_path, "present", 1)

    await run_archive_fixture_write(tmp_path, seed)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        stage = make_raw_existence_journal_prune_stage(tmp_path / "index.db", compute_adapter=owner._compute_adapter)
        assert stage.check(tmp_path / "index.db") is False
        assert frontier_existence.raw_existence_block_reason(tmp_path) is None
        assert stage.check(tmp_path / "index.db") is False  # frontier inspection has not consumed these rows
        await owner.run_convergence_sync(
            "fixture.frontier.inspect",
            inspect_prepared_raw_authority_frontier,
            tmp_path,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
        assert stage.check(tmp_path / "index.db") is True
        assert await owner.run_convergence_sync("fixture.frontier.prune", stage.execute, tmp_path / "index.db") is True
        assert stage.check(tmp_path / "index.db") is False
        with closing(sqlite3.connect(tmp_path / "index.db")) as conn, conn:
            assert conn.execute("SELECT COUNT(*) FROM raw_existence_changes").fetchone()[0] == 0

        def second() -> None:
            _raw(tmp_path, "second")
            _session(tmp_path, "second", 2)

        await run_archive_fixture_write(tmp_path, second)
        assert await owner.run_convergence_sync("fixture.frontier.prune", stage.execute, tmp_path / "index.db") is True
        with closing(sqlite3.connect(tmp_path / "index.db")) as conn, conn:
            assert conn.execute("SELECT COUNT(*) FROM raw_existence_changes").fetchone()[0] == 1
        assert frontier_existence.raw_existence_block_reason(tmp_path) is None


def test_retired_symlink_alias_refuses_the_selected_real_path(tmp_path: Path) -> None:
    """Refusal is matched through the canonical paths stored at acquisition.

    Anti-vacuity: compare by re-resolving the stored alias instead, and once
    the symlink is gone the alias resolves to itself, the refusal names only
    the obsolete spelling, and the real path the watcher selects is admitted.
    """
    initialize_active_archive_root(tmp_path)
    real = tmp_path / "real.jsonl"
    alias = tmp_path / "alias.jsonl"
    real.write_text("x" * 8, encoding="utf-8")
    alias.symlink_to(real)
    _raw(tmp_path, "aliased", path=alias, logical_key="codex:aliased")
    _session(tmp_path, "aliased", 1)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            "INSERT INTO raw_revision_heads(logical_source_key, session_id, accepted_raw_id, "
            "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
            "acquisition_generation, decided_at_ms) "
            "VALUES ('codex:aliased', 'codex-session:session-1', 'aliased', 'aliased', ?, 'byte', 2, 0, 1)",
            (bytes(32),),
        )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute(
            "INSERT INTO ingest_cursor(source_path, canonical_source_path, byte_offset, updated_at_ms) "
            "VALUES (?, ?, 5, 1)",
            (str(alias), str(real.resolve())),
        )
    alias.unlink()
    selected = raw_frontier_blocked_selected_paths(tmp_path, (real,))
    assert selected.unattributed_reason is None
    assert str(real.resolve()) in selected.source_paths


def test_changed_keys_beyond_one_batch_are_all_checked(tmp_path: Path) -> None:
    """Every journaled key is examined, in batches, however many a page adds.

    Anti-vacuity: check only the first batch and the missing key, sorted
    last, is never examined, so admission reports healthy.
    """
    initialize_active_archive_root(tmp_path)
    assert frontier_existence.raw_existence_block_reason(tmp_path) is None
    count = frontier_existence._CHANGED_KEY_BATCH * 2 + 5
    for number in range(count):
        _raw(tmp_path, f"k{number:05d}")
        _session(tmp_path, f"k{number:05d}", number)
    _session(tmp_path, "zz-absent", count)
    assert "zz-absent" in str(frontier_existence.raw_existence_block_reason(tmp_path))


def test_membership_expansion_pages_selections_beyond_the_variable_limit(tmp_path: Path) -> None:
    """A path with more retained raws than SQLite binds at once is still expanded.

    Anti-vacuity: bind the whole selection in one statement and SQLite raises
    ``too many SQL variables``, so the selected frontier reports unreadable.
    """
    from polylogue.storage.sqlite.archive_tiers.revision_governance import expand_raw_membership_selection_sync

    initialize_active_archive_root(tmp_path)
    shared = tmp_path / "shared.jsonl"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        limit = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    total = limit // 2 + 10
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.executemany(
            "INSERT INTO raw_sessions(raw_id, origin, source_path, canonical_source_path, blob_hash, blob_size, "
            "acquired_at_ms, logical_source_key, revision_kind, source_revision, acquisition_generation, "
            "revision_authority) VALUES (?, 'codex-session', ?, ?, ?, 1, 1, 'codex:shared', 'full', ?, 0, 'byte_proven')",
            [(f"r{n:06d}", str(shared), str(shared), bytes(32), f"r{n:06d}") for n in range(total)],
        )
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        selected, keys = expand_raw_membership_selection_sync(conn, [f"r{n:06d}" for n in range(total)])
    assert len(selected) == total
    assert keys == ("codex:shared",)
