"""polylogue-aex0: append planning must resynthesize a lost ops.db cursor.

``ingest_cursor`` (``CursorStore``) lives in the disposable ``ops.db`` tier --
every reset (index rebuild, schema mismatch, ``polylogue ops reset``) wipes it,
forcing the next observation of a still-growing file back onto the
full-capture path even though the file itself hasn't changed shape. These
scenarios exercise ``LiveBatchProcessor._append_plan``'s secondary lookup
against ``source.db``'s durable revision-chain evidence
(``_resynthesize_cursor_from_source``): unchanged behavior when the ops.db
cursor is present, successful append recovery when it's gone but a durable
byte-proven 'full' head exists, and unchanged fallback-to-full-capture when
neither exists.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    append_source_revision,
)
from polylogue.core.enums import Provider
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.batch_support import (
    _AppendPlan,
    claude_semantic_frontier_for_prefix,
    encode_cursor_hash_authority,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import migrate_archive_tier


def _session_meta(session_id: str) -> bytes:
    return f'{{"type":"session_meta","payload":{{"id":"{session_id}"}}}}\n'.encode()


def _codex_message(text: str) -> bytes:
    return (
        f'{{"type":"response_item","payload":{{"type":"message","role":"user",'
        f'"content":[{{"type":"input_text","text":"{text}"}}]}}}}\n'
    ).encode()


def _seed_native_session(tmp_path: Path, *, session_id: str) -> None:
    """Satisfy the Codex append path's identity-lock (``_existing_provider_session_id``)."""
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, "codex-session", "unrelated-raw-id", 1, b"a" * 32, 1, 1),
        )
        conn.commit()


def _processor(tmp_path: Path, cursor: CursorStore, *, source_name: str = Provider.CODEX.value) -> LiveBatchProcessor:
    return LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name=source_name, root=tmp_path),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )


def test_append_plan_resynthesizes_lost_cursor_from_durable_full_head(tmp_path: Path) -> None:
    """No ops.db cursor, but source.db has a byte-proven 'full' head: append is attempted."""
    session_id = "resynth-proof"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "rollout-resynth-proof.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path=str(source),
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                logical_source_key=f"codex:{session_id}",
                kind=RawRevisionKind.FULL,
                source_revision="full-0",
                acquisition_generation=0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
    appended = _codex_message("grown-after-reset")
    source.write_bytes(baseline + appended)
    _seed_native_session(tmp_path, session_id=session_id)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?",
                ("unrelated-raw-id",),
            ).fetchone()[0]
            == 0
        )

    cursor = CursorStore(tmp_path / "ops.db")
    assert cursor.get_record(source) is None  # the ops.db cursor is genuinely gone
    processor = _processor(tmp_path, cursor)

    plan = processor._append_plan(source)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(baseline)
    assert plan.cursor_fingerprint == "full-0"
    assert appended in plan.payload


def test_append_plan_uses_ops_db_cursor_without_consulting_source_db(tmp_path: Path) -> None:
    """An existing ops.db cursor is used as before; the fallback is never invoked."""
    session_id = "existing-cursor-proof"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "rollout-existing-cursor.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    source.write_bytes(baseline)
    baseline_digest = hashlib.sha256(baseline).hexdigest()
    appended = _codex_message("grown")
    with source.open("ab") as handle:
        handle.write(appended)
    stat = source.stat()
    _seed_native_session(tmp_path, session_id=session_id)

    cursor = CursorStore(tmp_path / "ops.db")
    cursor.set(
        source,
        len(baseline),
        byte_offset=len(baseline),
        last_complete_newline=len(baseline),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="already-known",
        tail_hash=encode_cursor_hash_authority(baseline_digest, baseline_digest, ctime_ns=stat.st_ctime_ns),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )
    processor = _processor(tmp_path, cursor)

    def _fail_resynthesis(self: LiveBatchProcessor, path: Path) -> None:
        raise AssertionError("resynthesis must not be consulted when an ops.db cursor is already usable")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(LiveBatchProcessor, "_resynthesize_cursor_from_source", _fail_resynthesis)
        plan = processor._append_plan(source)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(baseline)
    assert appended in plan.payload


def test_append_plan_declines_when_neither_cursor_nor_durable_head_exists(tmp_path: Path) -> None:
    """A genuinely new file falls back to full capture exactly as before."""
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "rollout-brand-new.jsonl"
    source.write_bytes(_session_meta("brand-new") + _codex_message("hello"))

    cursor = CursorStore(tmp_path / "ops.db")
    processor = _processor(tmp_path, cursor)

    plan = processor._append_plan(source)

    assert plan is None


def test_append_plan_declines_resynthesis_for_an_append_kind_head(tmp_path: Path) -> None:
    """An already-accepted append-kind head must not be treated as resynthesizable.

    Resynthesizing from a stale 'full' baseline behind an already-accepted
    append chain would create a second sibling append candidate at the same
    offset and make ``plan_revision_replay`` mark the whole chain ambiguous.
    """
    session_id = "append-head-proof"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "rollout-append-head.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    first_append_delta = _codex_message("first-append")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path=str(source),
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                logical_source_key=f"codex:{session_id}",
                kind=RawRevisionKind.FULL,
                source_revision="full-0",
                acquisition_generation=0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_session_meta(session_id) + first_append_delta,
            source_path=str(source),
            acquired_at_ms=2,
            revision=RawRevisionEnvelope(
                logical_source_key=f"codex:{session_id}",
                kind=RawRevisionKind.APPEND,
                source_revision="append-1",
                acquisition_generation=1,
                predecessor_source_revision="full-0",
                predecessor_raw_id=None,
                baseline_raw_id=None,
                append_start_offset=len(baseline),
                append_end_offset=len(baseline) + len(first_append_delta),
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        # Promote the append into the accepted chain the same way the real
        # append-ingest path does, via the durable classifier.
        archive.classify_raw_revision_cohort_for_live_watch(f"codex:{session_id}")
    second_append = _codex_message("second-append")
    source.write_bytes(baseline + first_append_delta + second_append)
    _seed_native_session(tmp_path, session_id=session_id)

    cursor = CursorStore(tmp_path / "ops.db")
    processor = _processor(tmp_path, cursor)

    plan = processor._append_plan(source)

    assert plan is None


def test_source_migration_adds_legacy_append_resynthesis_receipts(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("DROP TABLE raw_legacy_append_resynthesis_receipts")
        conn.execute("PRAGMA user_version = 39")
        conn.commit()

        result = migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=None)

        assert result.applied_versions == (40,)
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'raw_legacy_append_resynthesis_receipts'"
        ).fetchone() == (1,)


def test_append_plan_reconstructs_pre_offset_append_chain_after_ops_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy append window is admitted only when its retained bytes prove it."""
    session_id = "legacy-offset-chain"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "rollout-legacy-offset-chain.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    delta = _codex_message("legacy append")
    next_delta = _codex_message("next append")
    future_delta = _codex_message("future append")
    source.write_bytes(baseline + delta + next_delta + future_delta)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path=str(source),
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                f"codex:{session_id}",
                RawRevisionKind.FULL,
                "full-0",
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        append_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=delta,
            source_path=str(source),
            source_index=-1,
            acquired_at_ms=2,
        )
        next_append_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=next_delta,
            source_path=str(source),
            source_index=-1,
            acquired_at_ms=3,
        )
    # This is the pre-offset durable shape: the payload is retained, but the
    # source revision row has no operational byte coordinates.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.executemany(
            """UPDATE raw_sessions SET logical_source_key = ?, revision_kind = 'unknown', parsed_at_ms = 1,
               source_revision = ?, acquisition_generation = 1,
               revision_authority = 'quarantined' WHERE raw_id = ?""",
            (
                (f"codex:{session_id}", "legacy-append-1", append_id),
                (f"codex:{session_id}", "legacy-append-2", next_append_id),
            ),
        )
        conn.commit()
    _seed_native_session(tmp_path, session_id=session_id)
    processor = _processor(tmp_path, CursorStore(tmp_path / "ops.db"))

    # Scoped to the watched session file: reading *it* whole is the regression
    # this guards. Unrelated reads (parser sources behind schema-identity
    # fingerprints) are not what this test is about.
    real_read_bytes = Path.read_bytes

    def fail_full_file_read(self: Path) -> bytes:
        if self == source:
            raise AssertionError("cursor resynthesis must compare retained windows incrementally")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fail_full_file_read)

    plan = processor._append_plan(source)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(baseline) + len(delta) + len(next_delta)
    assert plan.cursor_fingerprint == append_source_revision(
        append_source_revision("full-0", hashlib.sha256(delta).hexdigest()),
        hashlib.sha256(next_delta).hexdigest(),
    )
    assert future_delta in plan.payload
    assert baseline_id
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        assert archive.raw_append_revision_parent(
            f"codex:{session_id}", plan.start_offset, plan.cursor_fingerprint
        ) == (next_append_id, baseline_id, 2)
        receipt = archive.raw_legacy_append_resynthesis_receipt(next_append_id)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        promoted = conn.execute(
            "SELECT revision_authority_evidence FROM raw_sessions WHERE raw_id = ?", (next_append_id,)
        ).fetchone()
    assert promoted == ("live_source_verification_v1",)
    assert receipt == (hashlib.sha256(baseline + delta + next_delta).hexdigest(), 0)


def test_resynthesis_composes_claude_frontier_from_legacy_append_chain(tmp_path: Path) -> None:
    session_id = "legacy-claude-chain"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "legacy-claude-chain.jsonl"
    header = b'{"sessionId":"legacy-claude-chain","type":"user"}\n'
    baseline_body = b'{"sessionId":"legacy-claude-chain","type":"assistant"}\n'
    delta = b'{"sessionId":"legacy-claude-chain","type":"user"}\n'
    source.write_bytes(header + baseline_body + delta + b'{"sessionId":"legacy-claude-chain","type":"assistant"}\n')
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=header + baseline_body,
            source_path=str(source),
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                f"claude-code:{session_id}",
                RawRevisionKind.FULL,
                "full-0",
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        append_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=delta,
            source_path=str(source),
            source_index=-1,
            acquired_at_ms=2,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """UPDATE raw_sessions SET logical_source_key = ?, revision_kind = 'unknown', parsed_at_ms = 1,
               source_revision = ?, acquisition_generation = 1,
               revision_authority = 'quarantined' WHERE raw_id = ?""",
            (f"claude-code:{session_id}", "legacy-append-1", append_id),
        )
        conn.execute(
            """INSERT INTO raw_authority_parser_census(
                   raw_id, parser_fingerprint, status, logical_keys_json, detail, censused_at_ms
               ) VALUES (?, 'current', 'complete', '[]', 'parser-observed: legacy', 0)""",
            (append_id,),
        )
        conn.commit()

    cursor = _processor(
        tmp_path,
        CursorStore(tmp_path / "ops.db"),
        source_name=Provider.CLAUDE_CODE.value,
    )._resynthesize_cursor_from_source(source)

    expected_end = len(header + baseline_body + delta)
    assert cursor is not None
    assert cursor.byte_offset == expected_end
    assert cursor.tail_hash == claude_semantic_frontier_for_prefix(source, expected_end)
    # Resynthesis composes the cursor but does not yet promote the legacy
    # chain: that durable write waits until append planning actually
    # succeeds, so a plan that never applies leaves no promoted state behind.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute("SELECT 1 FROM raw_authority_parser_census WHERE raw_id = ?", (append_id,)).fetchone()
            is not None
        )


def test_append_plan_declines_legacy_reconstruction_when_final_prefix_proof_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: a rewrite between preliminary windows and promotion cannot become cursor authority."""
    session_id = "legacy-final-prefix-proof"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "rollout-legacy-final-prefix-proof.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    rewritten_baseline = _session_meta(session_id) + _codex_message("mutated!")
    delta = _codex_message("legacy append")
    future_delta = _codex_message("future append")
    source.write_bytes(baseline + delta + future_delta)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path=str(source),
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                f"codex:{session_id}",
                RawRevisionKind.FULL,
                "full-0",
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        append_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=delta,
            source_path=str(source),
            source_index=-1,
            acquired_at_ms=2,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """UPDATE raw_sessions SET logical_source_key = ?, revision_kind = 'unknown', parsed_at_ms = 1,
               source_revision = ?, acquisition_generation = 1,
               revision_authority = 'quarantined' WHERE raw_id = ?""",
            (f"codex:{session_id}", "legacy-append-1", append_id),
        )
        conn.commit()
    _seed_native_session(tmp_path, session_id=session_id)
    original_open: Any = Path.open
    source_reads = 0

    def rewrite_before_final_proof(path: Path, *args: object, **kwargs: object) -> object:
        nonlocal source_reads
        if path == source and args and args[0] == "rb":
            source_reads += 1
            if source_reads == 3:
                source.write_bytes(rewritten_baseline + delta + future_delta)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", rewrite_before_final_proof)

    plan = _processor(tmp_path, CursorStore(tmp_path / "ops.db"))._append_plan(source)

    assert plan is None


def test_append_plan_declines_legacy_reconstruction_when_append_never_materialized(tmp_path: Path) -> None:
    """Mutation: retaining an unparsed legacy delta must not skip its parser/index work."""
    session_id = "legacy-unmaterialized"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "rollout-legacy-unmaterialized.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    delta = _codex_message("unmaterialized append")
    source.write_bytes(baseline + delta + _codex_message("future append"))
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path=str(source),
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                f"codex:{session_id}",
                RawRevisionKind.FULL,
                "full-0",
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        append_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=delta,
            source_path=str(source),
            source_index=-1,
            acquired_at_ms=2,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """UPDATE raw_sessions SET logical_source_key = ?, revision_kind = 'unknown',
               source_revision = ?, acquisition_generation = 1,
               revision_authority = 'quarantined' WHERE raw_id = ?""",
            (f"codex:{session_id}", "legacy-append-1", append_id),
        )
        conn.commit()
    _seed_native_session(tmp_path, session_id=session_id)

    plan = _processor(tmp_path, CursorStore(tmp_path / "ops.db"))._append_plan(source)

    assert plan is None


def test_append_plan_declines_legacy_reconstruction_after_prefix_rewrite(tmp_path: Path) -> None:
    """A matching legacy delta cannot prove a rewritten full prefix."""
    session_id = "legacy-prefix-rewrite"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "rollout-legacy-prefix-rewrite.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    rewritten_baseline = _session_meta(session_id) + _codex_message("mutated!")
    delta = _codex_message("legacy append")
    future_delta = _codex_message("future append")
    source.write_bytes(rewritten_baseline + delta + future_delta)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path=str(source),
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                f"codex:{session_id}",
                RawRevisionKind.FULL,
                "full-0",
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        append_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=delta,
            source_path=str(source),
            source_index=-1,
            acquired_at_ms=2,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """UPDATE raw_sessions SET logical_source_key = ?, revision_kind = 'unknown',
               source_revision = ?, acquisition_generation = 1,
               revision_authority = 'quarantined' WHERE raw_id = ?""",
            (f"codex:{session_id}", "legacy-append-1", append_id),
        )
        conn.commit()
    _seed_native_session(tmp_path, session_id=session_id)

    plan = _processor(tmp_path, CursorStore(tmp_path / "ops.db"))._append_plan(source)

    assert plan is None


def test_full_head_claude_frontier_requires_the_retained_prefix(tmp_path: Path) -> None:
    """A full-head Claude cursor must prove the live prefix is the retained blob.

    The reconstructed-chain branch compares every retained blob component
    against the source byte by byte before composing a frontier. The full-head
    branch composes one straight from the live file at ``head.blob_size``, so a
    rewrite preserving that length is adopted as semantic authority for bytes
    the archive never retained.

    Anti-vacuity: removing the ``file_prefix_sha256(path, byte_offset) !=
    blob_hash_hex`` guard in ``_resynthesize_cursor_from_source`` composes a
    frontier over the rewritten body and returns a cursor, turning the final
    assertion red. The unmodified-file assertion above it pins that an intact
    prefix still resynthesizes, so the fix cannot be "always refuse".
    """
    session_id = "claude-full-head"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "claude-full-head.jsonl"
    header = b'{"sessionId":"claude-full-head","type":"user"}\n'
    body = b'{"sessionId":"claude-full-head","type":"assistant"}\n'
    rewritten_body = b'{"sessionId":"claude-full-head","type":"assistanX"}\n'
    assert len(rewritten_body) == len(body)
    source.write_bytes(header + body)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=header + body,
            source_path=str(source),
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                f"claude-code:{session_id}",
                RawRevisionKind.FULL,
                "full-0",
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
    processor = _processor(
        tmp_path,
        CursorStore(tmp_path / "ops.db"),
        source_name=Provider.CLAUDE_CODE.value,
    )

    intact = processor._resynthesize_cursor_from_source(source)
    assert intact is not None
    assert intact.tail_hash == claude_semantic_frontier_for_prefix(source, len(header) + len(body))

    source.write_bytes(header + rewritten_body)

    assert processor._resynthesize_cursor_from_source(source) is None


def test_superseded_parser_fingerprint_refuses_append_planning(tmp_path: Path) -> None:
    """A cursor stamped by superseded parser semantics must not append onto itself.

    #4539 changed tool-result outcome derivation, so records parsed under the
    previous fingerprint carry a stale outcome. Refusing the append route sends
    the source back through the full path, which reparses it -- the production
    convergence route rather than a manual rebuild.

    Anti-vacuity: reverting ``_PARSER_FINGERPRINT`` to ``"live-batched-v2"``
    makes the stale cursor match the current fingerprint, so ``_append_plan``
    returns an ``_AppendPlan`` and the assertion goes red. The companion
    assertion pins that a cursor stamped with the *current* fingerprint still
    plans an append, so the fix cannot be "never append".
    """
    session_id = "fingerprint-bump"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / f"rollout-{session_id}.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    appended = _codex_message("grown")
    source.write_bytes(baseline + appended)
    _seed_native_session(tmp_path, session_id=session_id)

    def _write_cursor(store: CursorStore, fingerprint: str) -> None:
        stat = source.stat()
        store.set(
            source,
            stat.st_size,
            byte_offset=len(baseline),
            last_complete_newline=len(baseline),
            parser_fingerprint=fingerprint,
            content_fingerprint="full-0",
            tail_hash=encode_cursor_hash_authority(
                hashlib.sha256(baseline).hexdigest(),
                hashlib.sha256(baseline).hexdigest(),
                ctime_ns=0,
            ),
            source_name=Provider.CODEX.value,
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
        )

    current_store = CursorStore(tmp_path / "ops-current.db")
    _write_cursor(current_store, live_watcher._PARSER_FINGERPRINT)
    assert isinstance(_processor(tmp_path, current_store)._append_plan(source), _AppendPlan)

    stale_store = CursorStore(tmp_path / "ops-stale.db")
    _write_cursor(stale_store, "live-batched-v2")
    assert _processor(tmp_path, stale_store)._append_plan(source) is None


def test_deferred_planning_leaves_the_legacy_chain_unpromoted(tmp_path: Path) -> None:
    """Legacy promotion is a durable write and must not outrun its append plan.

    Resynthesis reconstructs a legacy chain to build a candidate cursor, but
    planning can still defer afterwards. Promoting during resynthesis commits
    ``live_source_verification_v1`` evidence for an append that no cursor
    records and no plan applied.

    Anti-vacuity: moving ``promote_legacy_appends()`` back inside
    ``_resynthesize_cursor_from_source`` (rather than calling it just before
    ``_AppendPlan`` is returned) promotes the row during the deferred pass and
    turns the final assertion red. ``test_append_plan_reconstructs_pre_offset_
    append_chain_after_ops_reset`` pins the other side -- a plan that does
    succeed still promotes -- so the fix cannot be "never promote".
    """
    session_id = "legacy-deferred-chain"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / f"rollout-{session_id}.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    delta = _codex_message("legacy append")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path=str(source),
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                f"codex:{session_id}",
                RawRevisionKind.FULL,
                "full-0",
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        append_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=delta,
            source_path=str(source),
            source_index=-1,
            acquired_at_ms=2,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """UPDATE raw_sessions SET logical_source_key = ?, revision_kind = 'unknown', parsed_at_ms = 1,
               source_revision = ?, acquisition_generation = 1,
               revision_authority = 'quarantined' WHERE raw_id = ?""",
            (f"codex:{session_id}", "legacy-append-1", append_id),
        )
        conn.commit()
    _seed_native_session(tmp_path, session_id=session_id)
    # No complete record beyond the reconstructed chain: planning defers.
    source.write_bytes(baseline + delta + b'{"type":"response_item"')

    plan = _processor(tmp_path, CursorStore(tmp_path / "ops.db"))._append_plan(source)

    assert not isinstance(plan, _AppendPlan)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        evidence = conn.execute(
            "SELECT revision_authority_evidence FROM raw_sessions WHERE raw_id = ?", (append_id,)
        ).fetchone()
    assert evidence != ("live_source_verification_v1",)
