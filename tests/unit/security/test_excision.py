"""Tests for standalone/off-mode local excision (polylogue-27m).

Anti-vacuity: ``test_reingest_does_not_resurrect_excised_content`` exercises
the real acquire-time write chokepoint
(``write_source_raw_session``/``ContentExcisedError``) that
``polylogue.operations.canonical_archive_ingest.ingest_one_shot_archive`` relies
on for every ordinary re-ingest; removing the gate in
``write_source_raw_session`` (or reverting the ``write_pair`` skip-not-abort
handling) makes it fail.
``test_blob_ref_reingest_does_not_resurrect_excised_content`` is the sibling
reproduction against ``write_source_raw_session_blob_ref`` -- the daemon's
memory-bounded streaming write route (used when a payload was replayed from
a blob file rather than held in memory); removing the gate added there in
the polylogue-27m fix round makes it fail while the payload-in-memory
sibling above keeps passing, which is exactly the bypass an earlier revision
of this PR shipped. ``test_apply_removes_rows_from_every_tier``
exercises the real cross-tier DELETE statements against real archive-tier
schemas (not a toy replica); commenting out any one tier's delete makes the
corresponding assertion fail. ``TestLineageSafety`` exercises the real
``session_links`` schema/FK and the audited Excision operation's refuse-by-default
guard; removing the ``find_lineage_dependents`` call (or the check that uses
it) makes ``test_apply_without_cascade_refuses_and_does_not_mutate`` fail
because the parent session would be silently deleted instead of raising.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from builtins import BaseExceptionGroup
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import polylogue.security.excision as excision_module
from polylogue.core.enums import AssertionKind
from polylogue.security.excision import (
    LineageDependentsError,
)
from polylogue.storage.accepted_marker_inputs import (
    AcceptedMarkerInputExcisedError,
    MixedAcceptedMarkerInputError,
    PreparedAcceptedMarkerInput,
    append_accepted_marker_input,
    persist_pending_marker_input_sync,
    prepare_accepted_marker_input,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ContentExcisedError,
    deterministic_blob_hash,
    is_blob_hash_excised,
    write_source_raw_session,
    write_source_raw_session_blob_ref,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.excision import (
    find_lineage_dependents_from_root,
    plan_session_excision_from_root,
    resolve_session_excision_target_from_root,
)
from tests.infra.excision_embeddings import seed_excision_session as _seed_session
from tests.infra.excision_execution import execute_excision, recover_excision
from tests.infra.sync_as_async import AsyncConnectionView


def _seed_marker_carriers(
    archive_root: Path, session_id: str
) -> tuple[PreparedAcceptedMarkerInput, PreparedAcceptedMarkerInput]:
    """Seed real pending/accepted carrier bytes plus their rebuildable witnesses."""
    raw_id = resolve_session_excision_target_from_root(archive_root, session_id).raw_targets[0].raw_id
    pending = prepare_accepted_marker_input(
        raw_id, [{"session_id": session_id, "candidates": [{"body": "pending secret"}]}]
    )
    accepted = prepare_accepted_marker_input(
        raw_id,
        [{"session_id": session_id, "candidates": [{"body": "accepted secret"}]}],
        request_facts={"revision": "accepted"},
    )
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("BEGIN IMMEDIATE")
        persist_pending_marker_input_sync(conn, pending, expected_incarnation_id=str(uuid.uuid4()))
        asyncio.run(append_accepted_marker_input(AsyncConnectionView(conn), accepted))
    from tests.infra.excision_embeddings import seed_excision_marker_witnesses

    seed_excision_marker_witnesses(archive_root, (pending, accepted))
    return pending, accepted


class TestPlanSessionExcision:
    def test_not_found_for_unknown_session(self, tmp_path: Path) -> None:
        with write_lease("test.excision-unknown-session", archive_root=tmp_path):
            initialize_active_archive_root(tmp_path)
        plan = plan_session_excision_from_root(tmp_path, "codex-session:does-not-exist")
        assert plan.found is False

    def test_counts_every_tier(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="plan-1", with_embedding=True)
        plan = plan_session_excision_from_root(tmp_path, session_id)
        assert plan.found is True
        assert plan.source_raw_rows == 1
        assert plan.source_blob_refs == 1
        assert plan.index_sessions == 1
        assert plan.index_messages == 1
        assert plan.index_blocks == 1
        assert plan.embeddings_vectors == 1

    def test_dry_run_does_not_mutate(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="plan-2")
        plan_session_excision_from_root(tmp_path, session_id)
        # Session must still be readable after a plan-only call.
        target = resolve_session_excision_target_from_root(tmp_path, session_id)
        assert target.found is True


class TestApplySessionExcision:
    def test_not_found_returns_found_false(self, tmp_path: Path) -> None:
        initialize_active_archive_root(tmp_path)
        receipt = execute_excision(tmp_path, "codex-session:nope", reason="r", actor="user:local")
        assert receipt["found"] is False
        assert receipt["complete"] is True
        assert receipt["counts"] == {}
        for name in (
            "removed_blob_hashes",
            "shared_blob_hashes",
            "marker_input_digests",
            "cascaded_session_ids",
            "retained_hook_events",
            "retained_source_containers",
        ):
            assert receipt[name] == []

    @pytest.mark.uses_real_clock("waits on real OS-thread scheduling to show the excision blocks behind the slot")
    def test_apply_waits_for_an_in_flight_blob_publication(self, tmp_path: Path) -> None:
        """Excision and a publisher's reserve-then-publish are mutually exclusive.

        Anti-vacuity: drop ``exclude_archive_blob_publishers`` from
        the audited Excision operation and the excision commits while a publisher
        holds its slot between reserving and publishing, so the publisher can
        move bytes the ledger now names into the blob namespace.
        """
        import threading

        from polylogue.storage.blob_publication import _archive_blob_publisher_slot

        session_id = _seed_session(tmp_path, native_id="apply-serialized")
        finished = threading.Event()

        def excise() -> None:
            execute_excision(tmp_path, session_id, reason="r", actor="user:local")
            finished.set()

        with _archive_blob_publisher_slot(tmp_path / "source.db"):
            worker = threading.Thread(target=excise)
            worker.start()
            assert not finished.wait(timeout=1.0)
        worker.join(timeout=30)
        assert finished.is_set()

    def test_targets_are_resolved_under_the_publisher_exclusion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No batch can replace a target between its resolution and its apply.

        Anti-vacuity (Codex P1, #5696): resolve targets before taking the
        exclusion and a shared publisher slot is still available while they
        are resolved.
        """
        import fcntl

        from polylogue.storage.blob_publication import _writer_lock_path

        session_id = _seed_session(tmp_path, native_id="resolve-under-lock")
        observed: list[bool] = []
        original = excision_module._stage_excision_source_closure

        def probe(*args: object, **kwargs: object) -> object:
            with _writer_lock_path(tmp_path / "source.db").open("a+b") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    observed.append(True)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                    observed.append(False)
            return original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(excision_module, "_stage_excision_source_closure", probe)
        execute_excision(tmp_path, session_id, reason="r", actor="user:local")

        assert observed == [True]

    def test_apply_removes_rows_from_every_tier(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="apply-1", with_embedding=True)
        # The frontier journal records one existence change per acquisition
        # effect on the raw (raw row, parser census, blob ref); excision removes
        # every row naming the excised raw, however many the journal holds.
        with sqlite3.connect(tmp_path / "source.db") as source:
            raw_ids = [str(row[0]) for row in source.execute("SELECT raw_id FROM raw_sessions")]
            journal_rows = source.execute(
                "SELECT COUNT(*) FROM raw_existence_changes WHERE raw_id IN (SELECT raw_id FROM raw_sessions)"
            ).fetchone()[0]
        assert journal_rows >= 1

        receipt = execute_excision(tmp_path, session_id, reason="contained a secret", actor="user:local")
        assert receipt["found"] is True
        assert receipt["counts"]["index_sessions"] == 1
        assert receipt["counts"]["index_messages"] == 1
        assert receipt["counts"]["index_blocks"] == 1
        assert receipt["counts"]["source_raw_rows"] == 1
        # Excision's own raw and blob-ref deletes append journal rows that it
        # also removes, so the receipt counts at least the prior rows and none
        # naming the raw survive.
        assert receipt["counts"]["source_raw_existence_changes"] >= journal_rows
        with sqlite3.connect(tmp_path / "source.db") as source:
            assert source.execute(
                f"SELECT COUNT(*) FROM raw_existence_changes WHERE raw_id IN ({','.join('?' * len(raw_ids))})",
                raw_ids,
            ).fetchone() == (0,)
        assert receipt["counts"]["source_blob_refs"] == 1
        assert receipt["counts"]["embeddings_vectors"] == 1
        assert len(receipt["removed_blob_hashes"]) == 1

        index_conn = sqlite3.connect(tmp_path / "index.db")
        try:
            assert (
                index_conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]
                == 0
            )
            assert (
                index_conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
                == 0
            )
            assert (
                index_conn.execute("SELECT COUNT(*) FROM blocks WHERE session_id = ?", (session_id,)).fetchone()[0] == 0
            )
        finally:
            index_conn.close()

        source_conn = sqlite3.connect(tmp_path / "source.db")
        try:
            assert source_conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
            assert source_conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone()[0] == 0
            assert source_conn.execute("SELECT COUNT(*) FROM raw_existence_changes").fetchone()[0] == 0
        finally:
            source_conn.close()

        from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec as _load_vec

        emb_conn = sqlite3.connect(tmp_path / "embeddings.db")
        try:
            _load_vec(emb_conn)
            assert emb_conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 0
            assert (
                emb_conn.execute(
                    "SELECT COUNT(*) FROM embedding_status WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
                == 0
            )
        finally:
            emb_conn.close()

    def test_apply_writes_durable_audit_receipt(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="apply-2")
        receipt = execute_excision(tmp_path, session_id, reason="pii leak", actor="user:audit")

        user_conn = sqlite3.connect(tmp_path / "user.db")
        try:
            row = user_conn.execute(
                "SELECT kind, target_ref, author_ref, author_kind FROM assertions WHERE assertion_id = ?",
                (receipt["receipt_assertion_id"],),
            ).fetchone()
        finally:
            user_conn.close()
        assert row is not None
        assert row[0] == AssertionKind.EXCISION_RECORD.value
        assert row[1] == f"session:{session_id}"
        assert row[3] == "user"

    def test_apply_removes_content_bearing_assertions_targeting_the_session(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="apply-3")
        user_db = tmp_path / "user.db"
        initialize_archive_database(user_db, ArchiveTier.USER)
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

                upsert_assertion(
                    conn,
                    assertion_id="assertion-note:pre-existing",
                    target_ref=f"session:{session_id}",
                    kind=AssertionKind.NOTE,
                    body_text="quoting the secret span here",
                    author_ref="user:local",
                    author_kind="user",
                    now_ms=1,
                )
        finally:
            conn.close()

        execute_excision(tmp_path, session_id, reason="r", actor="user:local")

        conn = sqlite3.connect(user_db)
        try:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE assertion_id = ?", ("assertion-note:pre-existing",)
            ).fetchone()[0]
        finally:
            conn.close()
        assert remaining == 0

    @pytest.mark.parametrize(
        "inside_author,inside_scope,corruption",
        [
            (True, False, None),
            (False, True, None),
            (True, True, None),
            (False, False, None),
            (False, False, "author"),
            (False, False, "scope"),
            (True, True, "status"),
        ],
    )
    def test_apply_tombstones_marker_assertions_without_retaining_content(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        inside_author: bool,
        inside_scope: bool,
        corruption: str | None,
    ) -> None:
        session_id = _seed_session(tmp_path, native_id="apply-marker-tombstone")
        outside = _seed_session(tmp_path, native_id="marker-outside")
        with sqlite3.connect(tmp_path / "index.db") as conn:
            block_id = str(
                conn.execute("SELECT block_id FROM blocks WHERE session_id=? LIMIT 1", (session_id,)).fetchone()[0]
            )
            outside_block = str(
                conn.execute("SELECT block_id FROM blocks WHERE session_id=? LIMIT 1", (outside,)).fetchone()[0]
            )
        author = f"block:{block_id if inside_author else outside_block}"
        scope = f"session:{session_id if inside_scope else outside}"
        user_db = tmp_path / "user.db"
        initialize_archive_database(user_db, ArchiveTier.USER)
        with sqlite3.connect(user_db) as conn:
            from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

            upsert_assertion(
                conn,
                assertion_id="marker-excision-test",
                target_ref=f"block:{block_id}",
                kind=AssertionKind.NOTE,
                value={"marker_kind": "note", "arguments": {}},
                body_text="secret marker body",
                author_ref=author,
                scope_ref=scope,
                author_kind="agent",
                now_ms=1,
            )
            conn.commit()

        if corruption is not None:
            from contextlib import contextmanager

            from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealError

            original_statement = PreparedIndexMutation.user_statement
            reached: list[bool] = []

            @contextmanager
            def corrupt_marker_statement(
                seal: PreparedIndexMutation, sql: str, parameters: tuple[object, ...] = (), **kwargs: Any
            ) -> Iterator[sqlite3.Cursor]:
                if sql.startswith("UPDATE assertions SET target_ref='assertion:'"):
                    reached.append(True)
                    if corruption == "author":
                        sql = sql.replace(
                            "body_text=NULL, ", "body_text=NULL, author_ref='assertion:' || assertion_id, "
                        )
                    elif corruption == "scope":
                        sql = sql.replace("body_text=NULL, ", "body_text=NULL, scope_ref=NULL, ")
                    else:
                        sql = sql.replace("status='deleted'", "status='active'")
                with original_statement(seal, sql, parameters, **kwargs) as cursor:
                    yield cursor

            with sqlite3.connect(user_db) as user:
                original_marker = user.execute(
                    "SELECT * FROM assertions WHERE assertion_id='marker-excision-test'"
                ).fetchone()
            monkeypatch.setattr(PreparedIndexMutation, "user_statement", corrupt_marker_statement)
            with pytest.raises(ReferenceSealError):
                execute_excision(tmp_path, session_id, reason="remove marker", actor="user:local")
            assert reached == [True]
            with sqlite3.connect(user_db) as user:
                assert (
                    user.execute("SELECT * FROM assertions WHERE assertion_id='marker-excision-test'").fetchone()
                    == original_marker
                )
            with sqlite3.connect(tmp_path / "index.db") as index:
                assert index.execute("SELECT 1 FROM blocks WHERE block_id=?", (block_id,)).fetchone() == (1,)
                assert index.execute("SELECT 1 FROM blocks WHERE block_id=?", (outside_block,)).fetchone() == (1,)
            return

        execute_excision(tmp_path, session_id, reason="remove marker", actor="user:local")

        with sqlite3.connect(user_db) as conn:
            row = conn.execute(
                "SELECT target_ref, value_json, body_text, evidence_refs_json, status, author_ref, scope_ref "
                "FROM assertions WHERE assertion_id = ?",
                ("marker-excision-test",),
            ).fetchone()
        assert row == (
            "assertion:marker-excision-test",
            "{}",
            None,
            "[]",
            "deleted",
            "assertion:marker-excision-test" if inside_author else author,
            None if inside_scope else scope,
        )
        with sqlite3.connect(tmp_path / "index.db") as index:
            assert index.execute("SELECT 1 FROM blocks WHERE block_id=?", (block_id,)).fetchone() is None
            assert index.execute("SELECT 1 FROM blocks WHERE block_id=?", (outside_block,)).fetchone() == (1,)
        # An all-status claim read serializes the tombstone. With an
        # ``excision-marker:`` target, ObjectRef validation raises here.
        from polylogue import Polylogue

        claims = asyncio.run(
            Polylogue(archive_root=tmp_path).list_assertion_claim_payloads(kinds=(AssertionKind.NOTE,), statuses=None)
        )
        marker = next(claim for claim in claims if claim.assertion_id == "marker-excision-test")
        assert marker.target_ref == "assertion:marker-excision-test"
        rendered = marker.model_dump(mode="json")
        assert rendered["status"] == "deleted"
        # Self is explicit retired provenance on this deleted tombstone, not
        # a user/default author or a live agent claim.
        assert rendered["author_ref"] == ("assertion:marker-excision-test" if inside_author else author)

    def test_apply_is_idempotent(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="apply-4")
        first = execute_excision(tmp_path, session_id, reason="r", actor="user:local")
        assert first["found"] is True
        second = execute_excision(tmp_path, session_id, reason="r-again", actor="user:local")
        assert second["found"] is False  # already gone; nothing left to touch

    def test_unindexed_pending_marker_excision_tombstones_its_raw_revision(self, tmp_path: Path) -> None:
        """Pending carrier is the durable session-to-raw link before index commit.

        Anti-vacuity: omitting carrier raw ids from excision's durable closure
        leaves this raw replayable; checking tombstones by request key alone
        then permits a changed recipe to persist the same excised material.
        """
        session_id = _seed_session(tmp_path, native_id="pending-before-index")
        raw_id = resolve_session_excision_target_from_root(tmp_path, session_id).raw_targets[0].raw_id
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        pending = prepare_accepted_marker_input(
            raw_id,
            [{"session_id": session_id, "candidates": [{"body": "excised pending material"}]}],
            request_facts={"recipe": "before"},
        )
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute("BEGIN IMMEDIATE")
            persist_pending_marker_input_sync(conn, pending, expected_incarnation_id=str(uuid.uuid4()))

        target = resolve_session_excision_target_from_root(tmp_path, session_id)
        assert target.session_exists is False
        assert tuple(raw.raw_id for raw in target.raw_targets) == (raw_id,)
        assert tuple(marker.identity for marker in target.marker_input_targets) == (pending.identity,)

        receipt = execute_excision(tmp_path, session_id, reason="test", actor="user:test")
        assert receipt["found"] is True
        assert receipt["counts"]["source_raw_rows"] == 1
        assert receipt["counts"]["source_marker_inputs_pending"] == 1
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (0,)
            assert conn.execute(
                "SELECT COUNT(*) FROM pending_accepted_marker_inputs WHERE raw_id = ?", (raw_id,)
            ).fetchone() == (0,)
            assert conn.execute(
                "SELECT COUNT(*) FROM excised_marker_inputs WHERE raw_id = ?", (raw_id,)
            ).fetchone() == (1,)

            changed_request = prepare_accepted_marker_input(
                raw_id,
                [{"session_id": session_id, "candidates": [{"body": "excised pending material"}]}],
                request_facts={"recipe": "after"},
            )
            assert changed_request.identity != pending.identity
            with pytest.raises(AcceptedMarkerInputExcisedError, match="was excised"):
                persist_pending_marker_input_sync(conn, changed_request, expected_incarnation_id=str(uuid.uuid4()))
            assert conn.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)

    @pytest.mark.parametrize("fault_site", ["source_commit", "paid_commit", "foreign_paid_attempt"])
    def test_retry_after_source_commit_before_index_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_site: str
    ) -> None:
        session_id = _seed_session(tmp_path, native_id="crash-source-index", with_embedding=True)
        from polylogue.storage.sqlite.reference_seal import _PreparedExcisionEmbeddingsChild

        class SourceCommitInterruptionError(Exception):
            pass

        interruption = SourceCommitInterruptionError("after actual Source commit")
        original_apply = _PreparedExcisionEmbeddingsChild.apply
        failed = False

        def fail_paid_apply(child: _PreparedExcisionEmbeddingsChild) -> None:
            nonlocal failed
            if not failed:
                failed = True
                with sqlite3.connect(tmp_path / "source.db") as conn:
                    assert conn.execute("SELECT count(*) FROM raw_sessions").fetchone() == (0,)
                if fault_site != "source_commit":
                    original_apply(child)
                    with sqlite3.connect(tmp_path / "embeddings.db") as conn:
                        assert conn.execute("SELECT count(*) FROM message_embeddings_meta").fetchone() == (0,)
                        assert conn.execute("SELECT count(*) FROM excision_embedding_completions").fetchone() == (1,)
                raise interruption
            original_apply(child)

        monkeypatch.setattr(_PreparedExcisionEmbeddingsChild, "apply", fail_paid_apply)
        with pytest.raises(SourceCommitInterruptionError) as caught:
            execute_excision(tmp_path, session_id, reason="crash", actor="user:test")
        assert caught.value is interruption
        assert failed

        # The durable marker exists while the rebuildable lookup key remains.
        assert resolve_session_excision_target_from_root(tmp_path, session_id).found is True
        source_conn = sqlite3.connect(tmp_path / "source.db")
        try:
            assert is_blob_hash_excised(source_conn, deterministic_blob_hash(b'{"native_id": "x"}')) is True
        finally:
            source_conn.close()

        with sqlite3.connect(tmp_path / "audit.db") as conn:
            original_attempts = conn.execute("SELECT operation_id,attempt_id FROM operation_attempts").fetchall()
        if fault_site == "foreign_paid_attempt":
            from polylogue.operations.mutation_transaction import RecoveryDeferredError

            with sqlite3.connect(tmp_path / "embeddings.db") as conn:
                conn.execute(
                    "UPDATE excision_embedding_completions SET attempt_id=?",
                    ("attempt:" + "x" * 24,),
                )
            with pytest.raises(RecoveryDeferredError):
                recover_excision(tmp_path)
            with sqlite3.connect(tmp_path / "audit.db") as conn:
                assert (
                    conn.execute("SELECT operation_id,attempt_id FROM operation_attempts").fetchall()
                    == original_attempts
                )
                assert conn.execute("SELECT status FROM operation_runs").fetchall() == [("interrupted",)]
            with sqlite3.connect(tmp_path / "user.db") as conn:
                assert conn.execute("SELECT count(*) FROM assertions WHERE kind='excision_record'").fetchone() == (0,)
            with sqlite3.connect(tmp_path / "index.db") as conn:
                assert conn.execute("SELECT count(*) FROM sessions WHERE session_id=?", (session_id,)).fetchone() == (
                    1,
                )
            return
        recover_excision(tmp_path)
        with sqlite3.connect(tmp_path / "audit.db") as conn:
            assert (
                conn.execute("SELECT operation_id,attempt_id FROM operation_attempts").fetchall() == original_attempts
            )
            assert conn.execute("SELECT status FROM operation_runs").fetchall() == [("completed",)]
        with sqlite3.connect(tmp_path / "user.db") as conn:
            assert conn.execute("SELECT count(*) FROM assertions WHERE kind='excision_record'").fetchone() == (1,)
        with sqlite3.connect(tmp_path / "embeddings.db") as conn:
            assert conn.execute("SELECT count(*) FROM message_embeddings_meta").fetchone() == (0,)
            assert conn.execute("SELECT count(*) FROM excision_embedding_completions").fetchone() == (1,)
        assert resolve_session_excision_target_from_root(tmp_path, session_id).found is False

    def test_source_first_retry_cleans_marker_witnesses_from_terminal_evidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A retry uses content-free marker tombstones after source erasure.

        Anti-vacuity: resolve marker witnesses only through still-live carrier
        bytes and the retry leaves both rebuildable witnesses behind.
        """
        session_id = _seed_session(tmp_path, native_id="crash-marker-source-index")
        pending, accepted = _seed_marker_carriers(tmp_path, session_id)
        from polylogue.storage.sqlite.reference_seal import _PreparedExcisionEmbeddingsChild

        class MarkerSourceInterruptionError(Exception):
            pass

        interruption = MarkerSourceInterruptionError("after actual marker Source commit")
        original_apply = _PreparedExcisionEmbeddingsChild.apply
        failed = False

        def fail_paid_apply(child: _PreparedExcisionEmbeddingsChild) -> None:
            nonlocal failed
            if not failed:
                failed = True
                with sqlite3.connect(tmp_path / "source.db") as conn:
                    assert conn.execute("SELECT count(*) FROM excised_marker_inputs").fetchone() == (2,)
                raise interruption
            original_apply(child)

        monkeypatch.setattr(_PreparedExcisionEmbeddingsChild, "apply", fail_paid_apply)
        with pytest.raises(MarkerSourceInterruptionError) as caught:
            execute_excision(tmp_path, session_id, reason="crash", actor="user:test")
        assert caught.value is interruption
        assert failed

        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
            assert conn.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (0,)
            assert conn.execute("SELECT COUNT(*) FROM excised_marker_inputs").fetchone() == (2,)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM ingest_marker_witnesses WHERE request_key IN (?, ?)",
                (pending.identity, accepted.identity),
            ).fetchone() == (2,)

        recovery_plan = plan_session_excision_from_root(tmp_path, session_id)
        assert recovery_plan.source_marker_inputs_pending == 1
        assert recovery_plan.source_marker_inputs_accepted == 1
        assert recovery_plan.marker_input_digests == (pending.payload_sha256, accepted.payload_sha256)
        from polylogue.operations.mutation_actuators import SessionExcisionActuator, SessionExcisionArgs

        prepared = SessionExcisionActuator().prepare(
            SessionExcisionArgs(
                archive_root=tmp_path,
                session_id=session_id,
                reason="crash",
                actor="user:test",
                cascade_lineage=False,
            )
        )
        assert prepared.context["source_marker_inputs_pending"] == recovery_plan.source_marker_inputs_pending
        assert prepared.context["source_marker_inputs_accepted"] == recovery_plan.source_marker_inputs_accepted
        assert prepared.context["marker_input_digests"] == list(recovery_plan.marker_input_digests)

        with sqlite3.connect(tmp_path / "audit.db") as conn:
            original_attempts = conn.execute("SELECT operation_id,attempt_id FROM operation_attempts").fetchall()
        recover_excision(tmp_path)
        with sqlite3.connect(tmp_path / "audit.db") as conn:
            assert (
                conn.execute("SELECT operation_id,attempt_id FROM operation_attempts").fetchall() == original_attempts
            )
        with sqlite3.connect(tmp_path / "user.db") as conn:
            row = conn.execute(
                "SELECT value_json FROM assertions WHERE target_ref=? AND kind='excision_record'",
                (f"session:{session_id}",),
            ).fetchone()
            assert row is not None
            receipt = json.loads(row[0])
        assert receipt["counts"]["source_marker_inputs_pending"] == 1
        assert receipt["counts"]["source_marker_inputs_accepted"] == 1
        assert receipt["counts"]["index_marker_witnesses"] == 2
        assert receipt["marker_input_digests"] == [pending.payload_sha256, accepted.payload_sha256]
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM ingest_marker_witnesses WHERE request_key IN (?, ?)",
                (pending.identity, accepted.identity),
            ).fetchone() == (0,)

    def test_retry_after_receipt_commit_before_index_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session_id = _seed_session(tmp_path, native_id="crash-receipt-index")
        pending, accepted = _seed_marker_carriers(tmp_path, session_id)
        from polylogue.storage.sqlite.reference_seal import IndexMutationScope

        class UserCommitInterruptionError(Exception):
            pass

        interruption = UserCommitInterruptionError("after actual User commit")
        original_commit = IndexMutationScope.commit
        failed = False

        def fail_index_commit(scope: IndexMutationScope) -> None:
            nonlocal failed
            if not failed:
                failed = True
                with sqlite3.connect(tmp_path / "user.db") as conn:
                    assert conn.execute("SELECT count(*) FROM assertions WHERE kind='excision_record'").fetchone() == (
                        1,
                    )
                raise interruption
            original_commit(scope)

        monkeypatch.setattr(IndexMutationScope, "commit", fail_index_commit)
        with pytest.raises(UserCommitInterruptionError) as caught:
            execute_excision(tmp_path, session_id, reason="crash", actor="user:test")
        assert caught.value is interruption
        assert failed

        with sqlite3.connect(tmp_path / "user.db") as conn:
            stored = conn.execute(
                "SELECT value_json FROM assertions WHERE target_ref = ? AND kind = ?",
                (f"session:{session_id}", AssertionKind.EXCISION_RECORD.value),
            ).fetchone()
            assert stored is not None
            stored_value = json.loads(stored[0])
            assert stored_value["counts"]["index_marker_witnesses"] == 2

        with sqlite3.connect(tmp_path / "audit.db") as conn:
            original_attempts = conn.execute("SELECT operation_id,attempt_id FROM operation_attempts").fetchall()
        recover_excision(tmp_path)
        with sqlite3.connect(tmp_path / "audit.db") as conn:
            assert (
                conn.execute("SELECT operation_id,attempt_id FROM operation_attempts").fetchall() == original_attempts
            )
        with sqlite3.connect(tmp_path / "user.db") as conn:
            assert (
                conn.execute(
                    "SELECT value_json FROM assertions WHERE target_ref=? AND kind='excision_record'",
                    (f"session:{session_id}",),
                ).fetchone()
                == stored
            )
        assert stored_value["reason"] == "crash"
        assert stored_value["actor"] == "user:test"
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM ingest_marker_witnesses WHERE request_key IN (?, ?)",
                (pending.identity, accepted.identity),
            ).fetchone() == (0,)
        assert resolve_session_excision_target_from_root(tmp_path, session_id).found is False

    def test_reingest_does_not_resurrect_excised_content(self, tmp_path: Path) -> None:
        payload = b'{"native_id": "resurrect-me", "secret": "sk-ant-abc123"}'
        session_id = _seed_session(tmp_path, native_id="resurrect-me", payload=payload)

        execute_excision(tmp_path, session_id, reason="secret leak", actor="user:local")

        source_conn = sqlite3.connect(tmp_path / "source.db")
        source_conn.execute("PRAGMA foreign_keys = ON")
        try:
            assert is_blob_hash_excised(source_conn, deterministic_blob_hash(payload)) is True
            with pytest.raises(ContentExcisedError):
                write_source_raw_session(
                    source_conn,
                    origin="codex-session",
                    source_path="/fake/resurrect-me.jsonl",
                    canonical_source_path="/fake/resurrect-me.jsonl",
                    source_index=0,
                    payload=payload,  # identical bytes: an ordinary re-ingest of the SAME file
                    acquired_at_ms=9_999,
                    native_id="resurrect-me",
                )
            # No row was resurrected by the refused write.
            assert source_conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
        finally:
            source_conn.close()

    def test_reingested_revision_does_not_reuse_old_excision_receipt(self, tmp_path: Path) -> None:
        """A completed receipt only skips cleanup for the source revision it names.

        Anti-vacuity: key retry detection only by session id and the second
        excision leaves the newly written user assertion readable.
        """
        session_id = _seed_session(tmp_path, native_id="revision-reingest", payload=b'{"revision":1}')
        execute_excision(tmp_path, session_id, reason="first revision", actor="user:local")
        assert _seed_session(tmp_path, native_id="revision-reingest", payload=b'{"revision":2}') == session_id

        user_db = tmp_path / "user.db"
        with sqlite3.connect(user_db) as conn:
            from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

            with conn:
                upsert_assertion(
                    conn,
                    assertion_id="assertion-note:new-revision",
                    target_ref=f"session:{session_id}",
                    kind=AssertionKind.NOTE,
                    body_text="new revision secret",
                    author_ref="user:local",
                    author_kind="user",
                    now_ms=20,
                )

        execute_excision(tmp_path, session_id, reason="second revision", actor="user:local")
        with sqlite3.connect(user_db) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE assertion_id = 'assertion-note:new-revision'"
            ).fetchone() == (0,)

    def test_reingest_batch_skips_excised_file_without_aborting(self, tmp_path: Path) -> None:
        """The batch orchestration layer must skip-not-abort on ContentExcisedError.

        Anti-vacuity: treating an excised path as a failed file makes the
        canonical one-shot ingestion reject the batch instead of reporting a skip.

        Uses the shared synthetic-corpus generator (real provider-shaped
        files, the same fixture machinery as
        ``tests/unit/pipeline/test_archive_ingest_commit_batching.py``)
        rather than a hand-rolled JSONL literal, so this exercises the real
        parser/detector path, not a guessed shape.
        """
        import asyncio

        from polylogue.config import Source
        from polylogue.operations.canonical_archive_ingest import ingest_one_shot_archive
        from polylogue.scenarios import build_default_corpus_specs
        from polylogue.schemas.synthetic import SyntheticCorpus

        archive_root = tmp_path / "archive"
        specs = build_default_corpus_specs(providers=["codex"], count=1, messages_min=2, messages_max=3, seed=11)
        corpus_dir = tmp_path / "corpus"
        written = SyntheticCorpus.write_spec_artifacts(specs[0], corpus_dir, prefix="corpus")
        outside_spec = build_default_corpus_specs(
            providers=["codex"], count=1, messages_min=2, messages_max=3, seed=12
        )[0]
        outside_written = SyntheticCorpus.write_spec_artifacts(outside_spec, corpus_dir, prefix="outside")
        sources = [Source(name="codex", path=file_path) for file_path in (*written.files, *outside_written.files)]
        assert sources

        # First ingest establishes the raw row + session normally.
        result_first = asyncio.run(ingest_one_shot_archive(archive_root, sources))
        assert result_first.excised_skips == 0
        assert result_first.counts["sessions"] == 2

        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            row = index_conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()
        finally:
            index_conn.close()
        assert row is not None
        session_id = str(row[0])
        with sqlite3.connect(archive_root / "index.db") as conn:
            outside_heads = conn.execute(
                "SELECT * FROM raw_revision_heads WHERE session_id != ? ORDER BY logical_source_key", (session_id,)
            ).fetchall()
            outside_applications = conn.execute(
                "SELECT * FROM raw_revision_applications WHERE session_id != ? ORDER BY decision_id", (session_id,)
            ).fetchall()
        assert outside_heads and outside_applications
        receipt = execute_excision(archive_root, session_id, reason="test", actor="user:local")
        assert receipt["found"] is True
        with sqlite3.connect(archive_root / "index.db") as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM raw_revision_heads WHERE session_id = ?", (session_id,)
            ).fetchone() == (0,)
            assert conn.execute(
                "SELECT COUNT(*) FROM raw_revision_applications WHERE session_id = ?", (session_id,)
            ).fetchone() == (0,)
            assert (
                conn.execute(
                    "SELECT * FROM raw_revision_heads WHERE session_id != ? ORDER BY logical_source_key", (session_id,)
                ).fetchall()
                == outside_heads
            )
            assert (
                conn.execute(
                    "SELECT * FROM raw_revision_applications WHERE session_id != ? ORDER BY decision_id", (session_id,)
                ).fetchall()
                == outside_applications
            )

        # Re-ingest the SAME unmodified file: must skip (not raise/abort).
        result_second = asyncio.run(ingest_one_shot_archive(archive_root, sources))
        assert result_second.excised_skips >= 1

        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            remaining = index_conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        finally:
            index_conn.close()
        assert remaining == 0

    def test_blob_ref_reingest_does_not_resurrect_excised_content(self, tmp_path: Path) -> None:
        """Reproduces the reviewer's finding directly: write_source_raw_session_blob_ref
        is the daemon's streaming/blob-ref write route (used when a payload was
        replayed from a blob file rather than held in memory) and must gate on
        excised blob hashes exactly like write_source_raw_session does above.
        """
        payload = b'{"native_id": "resurrect-blobref", "secret": "sk-ant-abc123"}'
        session_id = _seed_session(tmp_path, native_id="resurrect-blobref", payload=payload)

        execute_excision(tmp_path, session_id, reason="secret leak", actor="user:local")

        blob_hash = deterministic_blob_hash(payload)
        source_conn = sqlite3.connect(tmp_path / "source.db")
        source_conn.execute("PRAGMA foreign_keys = ON")
        try:
            assert is_blob_hash_excised(source_conn, blob_hash) is True
            with pytest.raises(ContentExcisedError):
                write_source_raw_session_blob_ref(
                    source_conn,
                    origin="codex-session",
                    source_path="/fake/resurrect-blobref.jsonl",
                    canonical_source_path="/fake/resurrect-blobref.jsonl",
                    source_index=0,
                    blob_hash=blob_hash,
                    blob_size=len(payload),
                    acquired_at_ms=9_999,
                    native_id="resurrect-blobref",
                )
            # No row was resurrected by the refused write.
            assert source_conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
        finally:
            source_conn.close()

    def test_additional_blob_refs_cannot_resurrect_an_excised_sibling(self, tmp_path: Path) -> None:
        """A sibling attachment/sidecar reference is gated like the payload is.

        Session excision records every sibling attachment and sidecar hash, not
        just the payload's. Both raw-session writers checked only the primary
        hash, so a later ingest whose own payload is admissible could still
        carry the excised hash in ``additional_blob_refs`` -- inserting its
        ``blob_refs`` row and consuming its publication receipt, making the
        excised bytes reachable again under a new raw record.

        Anti-vacuity: removing either
        ``_assert_additional_blob_refs_admissible`` call makes the refusal
        assertion fail and leaves the ``blob_refs`` row behind. The final
        admissible-sibling write pins the other direction, so a blanket refusal
        of ``additional_blob_refs`` cannot pass.
        """
        from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceBlobRef

        excised_payload = b'{"native_id": "sibling-secret", "secret": "sk-ant-abc123"}'
        session_id = _seed_session(tmp_path, native_id="sibling-secret", payload=excised_payload)
        execute_excision(tmp_path, session_id, reason="secret leak", actor="user:local")

        excised_hash = deterministic_blob_hash(excised_payload)
        fresh_payload = b'{"native_id": "sibling-carrier"}'
        innocent_hash = deterministic_blob_hash(b"an unrelated attachment")

        source_conn = sqlite3.connect(tmp_path / "source.db")
        source_conn.execute("PRAGMA foreign_keys = ON")
        try:
            assert is_blob_hash_excised(source_conn, excised_hash) is True
            assert is_blob_hash_excised(source_conn, innocent_hash) is False
            before = source_conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone()[0]

            forbidden = ArchiveSourceBlobRef(
                blob_hash=excised_hash,
                raw_id="",
                ref_type="attachment",
                source_path="/fake/sibling-secret.attachment",
                size_bytes=len(excised_payload),
                acquired_at_ms=9_999,
            )
            with pytest.raises(ContentExcisedError):
                write_source_raw_session(
                    source_conn,
                    origin="codex-session",
                    source_path="/fake/sibling-carrier.jsonl",
                    canonical_source_path="/fake/sibling-carrier.jsonl",
                    source_index=0,
                    payload=fresh_payload,
                    acquired_at_ms=9_999,
                    native_id="sibling-carrier",
                    additional_blob_refs=(forbidden,),
                )
            with pytest.raises(ContentExcisedError):
                write_source_raw_session_blob_ref(
                    source_conn,
                    origin="codex-session",
                    source_path="/fake/sibling-carrier-streamed.jsonl",
                    canonical_source_path="/fake/sibling-carrier-streamed.jsonl",
                    source_index=0,
                    blob_hash=deterministic_blob_hash(b'{"native_id": "sibling-carrier-streamed"}'),
                    blob_size=41,
                    acquired_at_ms=9_999,
                    native_id="sibling-carrier-streamed",
                    additional_blob_refs=(forbidden,),
                )
            assert source_conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone()[0] == before
            assert (
                source_conn.execute("SELECT COUNT(*) FROM blob_refs WHERE blob_hash = ?", (excised_hash,)).fetchone()[0]
                == 0
            )

            # Opposite direction: an admissible sibling still writes.
            admissible = ArchiveSourceBlobRef(
                blob_hash=innocent_hash,
                raw_id="",
                ref_type="attachment",
                source_path="/fake/sibling-carrier.attachment",
                size_bytes=23,
                acquired_at_ms=9_999,
            )
            write_source_raw_session(
                source_conn,
                origin="codex-session",
                source_path="/fake/sibling-carrier.jsonl",
                canonical_source_path="/fake/sibling-carrier.jsonl",
                source_index=0,
                payload=fresh_payload,
                acquired_at_ms=9_999,
                native_id="sibling-carrier",
                additional_blob_refs=(admissible,),
            )
            source_conn.commit()
            assert (
                source_conn.execute("SELECT COUNT(*) FROM blob_refs WHERE blob_hash = ?", (innocent_hash,)).fetchone()[
                    0
                ]
                == 1
            )
        finally:
            source_conn.close()


class TestLineageSafety:
    """Coverage for the polylogue-27m fix-round lineage-collateral guard."""

    def _seed_lineage(self, tmp_path: Path) -> tuple[str, str]:
        """Seed a parent session and a prefix-sharing child session_links row.

        Returns ``(parent_session_id, child_session_id)``.
        """
        parent_id = _seed_session(tmp_path, native_id="lineage-parent")
        child_id = _seed_session(tmp_path, native_id="lineage-child")

        index_conn = sqlite3.connect(tmp_path / "index.db")
        index_conn.execute("PRAGMA foreign_keys = ON")
        try:
            branch_point = index_conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ?", (parent_id,)
            ).fetchone()[0]
            index_conn.execute(
                """
                INSERT INTO session_links (
                    src_session_id, dst_origin, dst_native_id, link_type,
                    resolved_dst_session_id, branch_point_message_id, inheritance,
                    status, method, confidence, evidence_json, observed_at_ms, resolved_at_ms
                ) VALUES (?, 'codex-session', 'lineage-parent', 'branch', ?, ?, 'prefix-sharing',
                          NULL, NULL, 1.0, '[]', 1000, NULL)
                """,
                (child_id, parent_id, branch_point),
            )
            index_conn.commit()
        finally:
            index_conn.close()
        return parent_id, child_id

    def test_find_lineage_dependents_returns_prefix_sharing_child(self, tmp_path: Path) -> None:
        parent_id, child_id = self._seed_lineage(tmp_path)
        assert find_lineage_dependents_from_root(tmp_path, parent_id) == (child_id,)
        # The child is not itself a lineage parent of anything.
        assert find_lineage_dependents_from_root(tmp_path, child_id) == ()

    def test_find_lineage_dependents_ignores_spawned_fresh(self, tmp_path: Path) -> None:
        parent_id = _seed_session(tmp_path, native_id="fresh-parent")
        child_id = _seed_session(tmp_path, native_id="fresh-child")
        index_conn = sqlite3.connect(tmp_path / "index.db")
        try:
            index_conn.execute(
                """
                INSERT INTO session_links (
                    src_session_id, dst_origin, dst_native_id, link_type,
                    resolved_dst_session_id, branch_point_message_id, inheritance,
                    status, method, confidence, evidence_json, observed_at_ms, resolved_at_ms
                ) VALUES (?, 'codex-session', 'fresh-parent', 'subagent', ?, NULL, 'spawned-fresh',
                          NULL, NULL, 1.0, '[]', 1000, NULL)
                """,
                (child_id, parent_id),
            )
            index_conn.commit()
        finally:
            index_conn.close()
        # spawned-fresh children don't share bytes with the parent.
        assert find_lineage_dependents_from_root(tmp_path, parent_id) == ()

    def test_plan_surfaces_lineage_dependents(self, tmp_path: Path) -> None:
        parent_id, child_id = self._seed_lineage(tmp_path)
        with pytest.raises(LineageDependentsError) as excinfo:
            plan_session_excision_from_root(tmp_path, parent_id)
        assert excinfo.value.dependent_session_ids == (child_id,)

    def test_cascade_plan_and_apply_share_one_marker_carrier(self, tmp_path: Path) -> None:
        parent_id, child_id = self._seed_lineage(tmp_path)
        raw_id = resolve_session_excision_target_from_root(tmp_path, parent_id).raw_targets[0].raw_id
        shared = prepare_accepted_marker_input(
            raw_id,
            [
                {"session_id": parent_id, "candidates": [{"body": "parent"}]},
                {"session_id": child_id, "candidates": [{"body": "child"}]},
            ],
        )
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute("BEGIN IMMEDIATE")
            persist_pending_marker_input_sync(conn, shared, expected_incarnation_id=str(uuid.uuid4()))

        plan = plan_session_excision_from_root(tmp_path, parent_id, cascade_lineage=True)
        assert plan.source_marker_inputs_pending == 1
        assert plan.source_marker_inputs_accepted == 0
        assert plan.marker_input_digests == (shared.payload_sha256,)
        receipt = execute_excision(tmp_path, parent_id, reason="r", actor="user:local", cascade_lineage=True)
        assert receipt["counts"]["source_marker_inputs_pending"] == 1
        assert receipt["marker_input_digests"] == [shared.payload_sha256]

    def test_cascade_plan_refuses_marker_carrier_shared_outside_the_lineage(self, tmp_path: Path) -> None:
        parent_id, child_id = self._seed_lineage(tmp_path)
        outsider_id = _seed_session(tmp_path, native_id="lineage-outsider")
        raw_id = resolve_session_excision_target_from_root(tmp_path, parent_id).raw_targets[0].raw_id
        shared = prepare_accepted_marker_input(
            raw_id,
            [
                {"session_id": parent_id, "candidates": [{"body": "parent"}]},
                {"session_id": child_id, "candidates": [{"body": "child"}]},
                {"session_id": outsider_id, "candidates": [{"body": "retain"}]},
            ],
        )
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute("BEGIN IMMEDIATE")
            persist_pending_marker_input_sync(conn, shared, expected_incarnation_id=str(uuid.uuid4()))

        with pytest.raises(MixedAcceptedMarkerInputError, match="retained sessions"):
            plan_session_excision_from_root(tmp_path, parent_id, cascade_lineage=True)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (3,)

    def test_apply_without_cascade_refuses_and_does_not_mutate(self, tmp_path: Path) -> None:
        parent_id, child_id = self._seed_lineage(tmp_path)
        with pytest.raises(LineageDependentsError) as excinfo:
            execute_excision(tmp_path, parent_id, reason="r", actor="user:local")
        assert excinfo.value.dependent_session_ids == (child_id,)

        index_conn = sqlite3.connect(tmp_path / "index.db")
        try:
            count = index_conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_id IN (?, ?)", (parent_id, child_id)
            ).fetchone()[0]
        finally:
            index_conn.close()
        assert count == 2  # neither session touched by the refused apply

    def test_apply_with_cascade_removes_parent_and_dependents(self, tmp_path: Path) -> None:
        parent_id, child_id = self._seed_lineage(tmp_path)
        receipt = execute_excision(tmp_path, parent_id, reason="r", actor="user:local", cascade_lineage=True)
        assert receipt["found"] is True
        assert receipt["cascaded_session_ids"] == [child_id]
        # Counts are summed across the whole cascade (parent + child).
        assert receipt["counts"]["index_sessions"] == 2
        assert receipt["counts"]["index_messages"] == 2

        index_conn = sqlite3.connect(tmp_path / "index.db")
        try:
            remaining = index_conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            index_conn.close()
        assert remaining == 0

        user_conn = sqlite3.connect(tmp_path / "user.db")
        try:
            receipt_count = user_conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE kind = ?", (AssertionKind.EXCISION_RECORD.value,)
            ).fetchone()[0]
        finally:
            user_conn.close()
        assert receipt_count == 2  # one durable audit receipt per removed session

    def test_apply_with_no_dependents_behaves_as_before(self, tmp_path: Path) -> None:
        session_id = _seed_session(tmp_path, native_id="no-lineage")
        receipt = execute_excision(tmp_path, session_id, reason="r", actor="user:local")
        assert receipt["found"] is True
        assert receipt["cascaded_session_ids"] == []


class TestAttachmentBlobHashesAreExcisedToo:
    """Coverage for the polylogue-27m fix-round sibling-blob-hash marker fix.

    ``blob_refs`` groups every blob published under one raw ingestion by a
    shared ``ref_id`` (``ref_type IN ('raw_payload', 'attachment',
    'sidecar')``). Before this fix, the canonical Excision producer only
    recorded an ``excised_content`` marker for the raw payload's own blob
    hash -- an attachment's distinct content hash was un-referenced (its
    ``blob_refs`` row deleted) but never durably marked excised, so an
    identical attachment blob re-acquired under the same raw ingestion could
    silently resurrect. Reverting the sibling-hash lookup in
    the canonical Excision producer (collapsing back to recording only
    ``raw_target.blob_hash``) makes ``test_attachment_blob_hash_recorded_in_excised_content``
    fail.
    """

    def test_attachment_blob_hash_recorded_in_excised_content(self, tmp_path: Path) -> None:
        from polylogue.storage.sqlite.archive_tiers.source_write import (
            ArchiveSourceBlobRef,
            write_source_blob_refs,
        )

        session_id = _seed_session(tmp_path, native_id="attach-1")
        target = resolve_session_excision_target_from_root(tmp_path, session_id)
        raw_id = target.raw_targets[0].raw_id
        raw_blob_hash = target.raw_targets[0].blob_hash
        attachment_blob_hash = deterministic_blob_hash(b"attachment bytes with a secret sk-ant-xyz")
        assert attachment_blob_hash != raw_blob_hash

        source_conn = sqlite3.connect(tmp_path / "source.db")
        try:
            attachment_refs = (
                ArchiveSourceBlobRef(
                    blob_hash=attachment_blob_hash,
                    ref_type="attachment",
                    source_path="attachment.png",
                    size_bytes=42,
                    acquired_at_ms=1_000,
                ),
            )
            write_source_blob_refs(source_conn, raw_id, lambda: iter(attachment_refs))
            source_conn.commit()
        finally:
            source_conn.close()

        receipt = execute_excision(tmp_path, session_id, reason="secret in attachment", actor="user:local")
        assert receipt["found"] is True
        assert raw_blob_hash.hex() in receipt["removed_blob_hashes"]
        assert attachment_blob_hash.hex() in receipt["removed_blob_hashes"]

        source_conn = sqlite3.connect(tmp_path / "source.db")
        try:
            assert is_blob_hash_excised(source_conn, raw_blob_hash) is True
            assert is_blob_hash_excised(source_conn, attachment_blob_hash) is True
            # The attachment's own blob_refs row is gone, same as the raw payload's.
            assert source_conn.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_id = ?", (raw_id,)).fetchone()[0] == 0
        finally:
            source_conn.close()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("sink_failed", [False, True])
def test_started_excision_effects_settle_before_sink_and_preserve_outside(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared: bool,
    sink_failed: bool,
) -> None:
    """Actual acquired Source + installed vec0 effects, independent of delivery."""
    import hashlib

    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    selected = _seed_session(
        tmp_path, native_id="effect-selected", payload=b"selected acquired bytes", with_embedding=True
    )
    outside = _seed_session(
        tmp_path, native_id="effect-outside", payload=b"outside acquired bytes", with_embedding=True
    )
    with sqlite3.connect(tmp_path / "index.db") as index:
        selected_messages = index.execute("SELECT message_id FROM messages WHERE session_id=?", (selected,)).fetchall()
        selected_blocks = index.execute("SELECT block_id FROM blocks WHERE session_id=?", (selected,)).fetchall()
        outside_messages = index.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY message_id", (outside,)
        ).fetchall()
        outside_blocks = index.execute(
            "SELECT * FROM blocks WHERE session_id=? ORDER BY block_id", (outside,)
        ).fetchall()
    assert selected_messages and selected_blocks and outside_messages and outside_blocks
    with sqlite3.connect(tmp_path / "embeddings.db") as paid:
        assert try_load_sqlite_vec(paid)[0]
        selected_hash = paid.execute(
            "SELECT vector_derivation_hash FROM message_embedding_refs WHERE session_id=?", (selected,)
        ).fetchone()[0]
        original_vectors = paid.execute(
            "SELECT vector_derivation_hash,embedding,model FROM message_embeddings ORDER BY vector_derivation_hash"
        ).fetchall()
        if shared:
            paid.execute(
                "UPDATE message_embedding_refs SET vector_derivation_hash=? WHERE session_id=?",
                (selected_hash, outside),
            )

    import pickle

    from polylogue.storage.sqlite.reference_seal import KnownTierMutationPermit

    actual_source_apply = KnownTierMutationPermit.apply_source_statements
    compiled = []

    def source_apply(permit: KnownTierMutationPermit, connection: sqlite3.Connection) -> Any:
        with permit._seal._owned_cursor(
            permit._seal._scratch,
            "SELECT compiled_actions FROM temp.known_tier_statements WHERE tier='source' "
            "AND sql LIKE 'DELETE FROM raw_sessions WHERE%'",
        ) as rows:
            actions = pickle.loads(rows.fetchone()[0])
            assert rows.fetchone() is None
        assert ("raw_container_coordinates", sqlite3.SQLITE_DELETE) in actions
        with permit._seal._owned_cursor(
            permit._seal._scratch,
            "SELECT count(*) FROM temp.known_tier_effects WHERE tier='source' AND table_name='raw_container_coordinates'",
        ) as rows:
            assert rows.fetchone()[0] == 0
        with permit._seal._owned_cursor(
            connection,
            "SELECT count(*) FROM temp.sqlite_schema WHERE type='trigger' AND tbl_name='raw_container_coordinates'",
        ) as rows:
            assert rows.fetchone()[0] == 6
        compiled.append(True)
        return actual_source_apply(permit, connection)

    monkeypatch.setattr(KnownTierMutationPermit, "apply_source_statements", source_apply)

    class SinkFailureError(Exception):
        pass

    failure = SinkFailureError("exact effect sink failure")
    reached = []

    def sink(summary: Any, literal: Any) -> Any:
        with sqlite3.connect(tmp_path / "index.db") as index:
            assert index.execute("SELECT session_id FROM sessions ORDER BY session_id").fetchall() == [(outside,)]
            assert index.execute("SELECT message_id FROM messages WHERE session_id=?", (selected,)).fetchall() == []
            assert index.execute("SELECT block_id FROM blocks WHERE session_id=?", (selected,)).fetchall() == []
            assert (
                index.execute("SELECT * FROM messages WHERE session_id=? ORDER BY message_id", (outside,)).fetchall()
                == outside_messages
            )
            assert (
                index.execute("SELECT * FROM blocks WHERE session_id=? ORDER BY block_id", (outside,)).fetchall()
                == outside_blocks
            )
        with sqlite3.connect(tmp_path / "source.db") as source:
            assert source.execute("SELECT native_id FROM raw_sessions").fetchall() == [("effect-outside",)]
        with sqlite3.connect(tmp_path / "user.db") as user:
            value = json.loads(
                user.execute(
                    "SELECT value_json FROM assertions WHERE assertion_id=?", (summary["receipt_assertion_id"],)
                ).fetchone()[0]
            )
            assert value["counts"] == summary["counts"]
        with sqlite3.connect(tmp_path / "embeddings.db") as paid:
            assert try_load_sqlite_vec(paid)[0]
            assert paid.execute("SELECT session_id FROM message_embedding_refs").fetchall() == [(outside,)]
            actual = paid.execute(
                "SELECT vector_derivation_hash,embedding,model FROM message_embeddings ORDER BY vector_derivation_hash"
            ).fetchall()
            assert actual == (
                original_vectors
                if shared
                else [row for row in original_vectors if row[0] != bytes(selected_hash).hex()]
            )
            assert paid.execute("SELECT count(*) FROM excision_embedding_completions").fetchone()[0] == 1
        assert summary["counts"]["index_sessions"] == 1
        assert summary["counts"]["index_messages"] == len(selected_messages)
        assert summary["counts"]["index_blocks"] == len(selected_blocks)
        assert summary["counts"]["source_raw_rows"] == 1
        assert summary["counts"]["embeddings_vectors"] == 1
        assert summary["counts"]["embeddings_vectors_gc"] == int(not shared)
        assert summary["counts"]["embeddings_outputs"] == int(not shared)
        assert summary["removed_blob_hashes_count"] == 1
        reached.append(True)
        if sink_failed:
            raise failure

    if sink_failed:
        with pytest.raises(SinkFailureError) as caught:
            execute_excision(tmp_path, selected, reason="synthetic removal", result_sink=sink)
        assert caught.value is failure
    else:
        receipt = execute_excision(tmp_path, selected, reason="synthetic removal", result_sink=sink)
        assert receipt["removed_blob_hashes"] == [hashlib.sha256(b"selected acquired bytes").hexdigest()]
        assert receipt["shared_blob_hashes"] == []
        assert receipt["complete"] is True
    assert reached == [True] and compiled == [True]
    with sqlite3.connect(tmp_path / "audit.db") as audit:
        event = audit.execute(
            "SELECT operation_id,attempt_id,detail_json FROM operation_events WHERE event_type='excision_source_committed'"
        ).fetchone()
        assert event is not None
        detail = json.loads(event[2])
        assert detail["operation_id"] == event[0]
        assert detail["attempt_id"] == event[1]
        assert detail["targets"][0]["session_id"] == selected
        assert audit.execute("SELECT state FROM operation_attempts WHERE attempt_id=?", (event[1],)).fetchone()[0] == (
            "unknown" if sink_failed else "applied"
        )
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert (
            source.execute("SELECT pending_payload_json FROM audit_continuity_control WHERE singleton=1").fetchone()[0]
            is None
        )


@pytest.mark.parametrize("fault", ["wrong_arguments", "precommit_cancel"])
def test_started_excision_refuses_before_any_domain_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from dataclasses import replace

    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.core.compute_cancel import compute_cancel
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealError

    selected = _seed_session(
        tmp_path, native_id="precommit-selected", payload=b"selected original", with_embedding=True
    )
    cancelled = []
    if fault == "wrong_arguments":
        actual_apply = excision_module._apply_started_session_excision

        def wrong(started: Any, args: Any, *, actuator: Any) -> Any:
            return actual_apply(started, replace(args, reason="foreign reason"), actuator=actuator)

        monkeypatch.setattr(excision_module, "_apply_started_session_excision", wrong)
        failure_type: type[BaseException] = ReferenceSealError
    else:
        actual_prepare = PreparedIndexMutation.prepare_excision_embeddings_child

        def cancel(seal: PreparedIndexMutation) -> Any:
            child = actual_prepare(seal)
            cancel_event = compute_cancel.get()
            assert cancel_event is not None
            cancel_event.set()
            cancelled.append(True)
            return child

        monkeypatch.setattr(PreparedIndexMutation, "prepare_excision_embeddings_child", cancel)
        failure_type = BaseExceptionGroup
    with pytest.raises(failure_type) as caught:
        execute_excision(tmp_path, selected, reason="original reason")
    if fault == "precommit_cancel":
        assert cancelled == [True]
        assert isinstance(caught.value, BaseExceptionGroup)
        assert len(caught.value.exceptions) == 2
        assert tuple(type(error) for error in caught.value.exceptions) == (
            asyncio.CancelledError,
            DaemonOperationCancelled,
        )
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT count(*) FROM raw_sessions").fetchone()[0] == 1
        assert source.execute("SELECT count(*) FROM excised_content").fetchone()[0] == 0
        assert (
            source.execute("SELECT pending_payload_json FROM audit_continuity_control WHERE singleton=1").fetchone()[0]
            is None
        )
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute("SELECT session_id FROM sessions").fetchall() == [(selected,)]
    with sqlite3.connect(tmp_path / "user.db") as user:
        assert user.execute("SELECT count(*) FROM assertions WHERE kind='excision_record'").fetchone()[0] == 0
    with sqlite3.connect(tmp_path / "embeddings.db") as paid:
        assert paid.execute("SELECT count(*) FROM message_embedding_refs").fetchone()[0] == 1
        assert paid.execute("SELECT count(*) FROM excision_embedding_completions").fetchone()[0] == 0


@pytest.mark.parametrize("attempt", ["uncaptured_row", "unrecorded_action"])
def test_started_source_compiled_dependency_does_not_authorize_outside_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attempt: str
) -> None:
    from polylogue.storage.sqlite.reference_seal import (
        KnownTierMutationPermit,
        PreparedIndexMutation,
        ReferenceSealError,
    )
    from polylogue.storage.sqlite.write_lease import current_sql_custody

    selected = _seed_session(
        tmp_path, native_id="guard-selected", payload=b"selected acquired bytes", with_embedding=True
    )
    outside = _seed_session(tmp_path, native_id="guard-outside", payload=b"outside acquired bytes", with_embedding=True)
    with sqlite3.connect(tmp_path / "source.db") as source:
        outside_raw = source.execute("SELECT raw_id FROM raw_sessions WHERE native_id='guard-outside'").fetchone()[0]
        source.execute(
            "INSERT INTO raw_container_coordinates(raw_id,coordinate_format,entry_ordinal,split_index) VALUES(?,'zip-v2',0,0)",
            (outside_raw,),
        )
    actual_cursor = PreparedIndexMutation._owned_cursor
    actual_consume = KnownTierMutationPermit._consume_native_effect
    attempted = []
    guarded = []
    sink = []

    def cursor(seal: Any, connection: sqlite3.Connection, sql: str, parameters: Any = ()) -> Any:
        custody = current_sql_custody()
        permit = None if custody is None else custody.known_tier_authority
        if (
            isinstance(permit, KnownTierMutationPermit)
            and permit.tier == "source"
            and permit._connection is connection
            and permit._active_statement_id is not None
            and sql.startswith("DELETE FROM raw_sessions WHERE")
        ):
            attempted.append(True)
            # Attempt hostile SQL on the actual original guarded writer. The
            # declaration compiled DELETE here but captured no selected row;
            # UPDATE was never in this exact statement's compiled closure.
            sql = (
                "DELETE FROM raw_container_coordinates WHERE raw_id=?"
                if attempt == "uncaptured_row"
                else "UPDATE raw_container_coordinates SET split_index=1 WHERE raw_id=?"
            )
            parameters = (outside_raw,)
        return actual_cursor(seal, connection, sql, parameters)

    def consume(
        permit: Any, connection: Any, table: str, phase: Any, operation: Any, old_rowid: Any, new_rowid: Any
    ) -> Any:
        if table == "raw_container_coordinates":
            guarded.append((phase, operation))
        return actual_consume(permit, connection, table, phase, operation, old_rowid, new_rowid)

    monkeypatch.setattr(PreparedIndexMutation, "_owned_cursor", cursor)
    monkeypatch.setattr(KnownTierMutationPermit, "_consume_native_effect", consume)
    expected = ReferenceSealError if attempt == "uncaptured_row" else sqlite3.DatabaseError
    with pytest.raises(expected):
        execute_excision(tmp_path, selected, reason="synthetic removal", result_sink=lambda *_: sink.append(True))
    assert attempted == [True] and sink == []
    assert guarded == ([("BEFORE", "DELETE")] if attempt == "uncaptured_row" else [])
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT count(*) FROM raw_sessions").fetchone()[0] == 2
        assert source.execute("SELECT raw_id,split_index FROM raw_container_coordinates").fetchall() == [
            (outside_raw, 0)
        ]
        assert source.execute("SELECT pending_payload_json FROM audit_continuity_control").fetchone()[0] is None
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute("SELECT session_id FROM sessions ORDER BY session_id").fetchall() == sorted(
            [(selected,), (outside,)]
        )
    with sqlite3.connect(tmp_path / "user.db") as user:
        assert user.execute("SELECT count(*) FROM assertions").fetchone()[0] == 0
    with sqlite3.connect(tmp_path / "embeddings.db") as paid:
        assert paid.execute("SELECT count(*) FROM message_embedding_refs").fetchone()[0] == 2
        assert paid.execute("SELECT count(*) FROM excision_embedding_completions").fetchone()[0] == 0
    with sqlite3.connect(tmp_path / "audit.db") as audit:
        assert (
            audit.execute(
                "SELECT count(*) FROM operation_events WHERE event_type='excision_source_committed'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("changed_column", ["carrier_digest", "incarnation_id", "dispositions_json"])
def test_frozen_index_marker_cells_refuse_changed_native_witness_before_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_column: str
) -> None:
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.sqlite.connection_profile import (
        NativeSQLCustodyOwner,
        native_sql_owner_for_connection,
        open_isolated_write_connection,
    )
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealError

    session_id = _seed_session(tmp_path, native_id="frozen-marker-native-cells", with_embedding=True)
    pending, _accepted = _seed_marker_carriers(tmp_path, session_id)
    original_for_excision = PreparedIndexMutation.for_excision
    changed = False

    def for_excision(*args: Any, **kwargs: Any) -> Any:
        nonlocal changed
        if not changed:

            def change_native() -> None:
                conn = open_isolated_write_connection(
                    tmp_path / "index.db", purpose="fixture foreign marker change", archive_root=tmp_path
                )
                owner = native_sql_owner_for_connection(conn) or NativeSQLCustodyOwner(conn)
                try:
                    value = {
                        "carrier_digest": "c" * 64,
                        "incarnation_id": str(uuid.uuid4()),
                        "dispositions_json": '[{"decision":"changed"}]',
                    }[changed_column]
                    conn.execute(
                        f"UPDATE ingest_marker_witnesses SET {changed_column}=? WHERE request_key=?",
                        (value, pending.identity),
                    )
                    conn.commit()
                finally:
                    owner.close()

            admit_stage_write("test.foreign-marker-change", change_native)
            changed = True
        return original_for_excision(*args, **kwargs)

    monkeypatch.setattr(PreparedIndexMutation, "for_excision", for_excision)
    with pytest.raises(ReferenceSealError):
        execute_excision(tmp_path, session_id, reason="exact marker refusal", actor="user:test")
    assert changed
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT count(*) FROM raw_sessions").fetchone() == (1,)
        assert source.execute("SELECT count(*) FROM excised_marker_inputs").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "embeddings.db") as paid:
        assert paid.execute("SELECT count(*) FROM message_embeddings_meta").fetchone() == (1,)
        assert paid.execute("SELECT count(*) FROM excision_embedding_completions").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "user.db") as user:
        assert user.execute("SELECT count(*) FROM assertions WHERE kind='excision_record'").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute("SELECT count(*) FROM sessions WHERE session_id=?", (session_id,)).fetchone() == (1,)
