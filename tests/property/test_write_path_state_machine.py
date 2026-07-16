"""Stateful write-path laws for full replace, append, and lineage (#g9f2).

The machine deliberately uses the production SQLite writer and its composed
archive reader.  Its model is just the expected logical transcript; it does
not duplicate prefix extraction, FTS maintenance, or session-link resolution.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
from hypothesis import HealthCheck, settings
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import Provider, TopologyEdgeStatus
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveSessionEnvelope,
    read_archive_session_envelope,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.async_sqlite import configure_connection
from polylogue.storage.sqlite.queries.session_links import resolve_unresolved_links_for_child
from polylogue.storage.sqlite.schema import _ensure_schema


@dataclass
class _SessionModel:
    native_id: str
    own_texts: list[str]
    parent_id: str | None = None
    prefix_length: int = 0
    updated_at: str = "2026-01-01T00:00:00Z"
    appended_batches: int = 0
    can_be_parent: bool = True


class WritePathStateMachine(RuleBasedStateMachine):
    """Exercise interleavings against the real archive write and read routes.

    A prefix-sharing child stores only ``own_texts``.  For a complete lineage
    read, its model transcript is the parent's logical prefix plus that tail.
    A hard-deleted branch point is an intentional exception: the documented
    #4ts.6 result is a non-overextending suffix with an incomplete-lineage
    # flag.
    """

    def __init__(self) -> None:
        super().__init__()
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        _ensure_schema(self._conn)
        self._models: dict[str, _SessionModel] = {}
        self._pending_children: dict[str, tuple[str, list[str], int, list[str]]] = {}
        self._next_session = 0
        self._next_text = 0
        self._next_fresh_version = 1
        self._deletion_done = False

    @initialize()
    def ingest_initial_parent(self) -> None:
        self._ingest_parent()
        self._check_invariants()

    @rule()
    def ingest_parent(self) -> None:
        self._ingest_parent()
        self._check_invariants()

    @rule()
    def ingest_child_replaying_parent_prefix(self) -> None:
        if not any(model.can_be_parent for model in self._models.values()):
            self._ingest_parent()
        parent_id = self._choose_parent_id()
        parent_logical = self._logical_texts(parent_id)
        prefix_length = 1 + (self._next_text % len(parent_logical))
        native_id = self._new_native_id("child")
        tail = [self._new_text("tail")]
        child_id = self._write(
            native_id,
            [*parent_logical[:prefix_length], *tail],
            parent_native_id=self._models[parent_id].native_id,
        )
        self._models[child_id] = _SessionModel(
            native_id=native_id,
            own_texts=tail,
            parent_id=parent_id,
            prefix_length=prefix_length,
        )
        self._check_invariants()

    @rule()
    def ingest_child_with_no_shared_prefix(self) -> None:
        """A known parent may be topology-linked without sharing content."""
        if not any(model.can_be_parent for model in self._models.values()):
            self._ingest_parent()
        parent_id = self._choose_parent_id()
        native_id = self._new_native_id("fresh-child")
        own_texts = [self._new_text("fresh")]
        child_id = self._write(
            native_id,
            own_texts,
            parent_native_id=self._models[parent_id].native_id,
        )
        self._models[child_id] = _SessionModel(
            native_id=native_id,
            own_texts=own_texts,
            parent_id=parent_id,
        )
        self._check_invariants()

    @rule()
    def ingest_child_before_parent(self) -> None:
        """Store a replaying child whole until its asserted parent arrives."""
        parent_native_id = self._new_native_id("late-parent")
        parent_texts = [self._new_text("late-parent"), self._new_text("late-variant")]
        prefix_length = 1 + (self._next_text % len(parent_texts))
        tail = [self._new_text("late-tail")]
        child_native_id = self._new_native_id("late-child")
        child_id = self._write(
            child_native_id,
            [*parent_texts[:prefix_length], *tail],
            parent_native_id=parent_native_id,
        )
        self._models[child_id] = _SessionModel(
            native_id=child_native_id,
            own_texts=[*parent_texts[:prefix_length], *tail],
            can_be_parent=False,
        )
        self._pending_children[child_id] = (parent_native_id, parent_texts, prefix_length, tail)
        self._check_invariants()

    @rule()
    def ingest_pending_parent(self) -> None:
        if not self._pending_children:
            return
        child_id = sorted(self._pending_children)[self._next_text % len(self._pending_children)]
        parent_native_id, parent_texts, prefix_length, tail = self._pending_children.pop(child_id)
        parent_id = self._write(parent_native_id, parent_texts)
        self._models[parent_id] = _SessionModel(native_id=parent_native_id, own_texts=parent_texts)
        self._models[child_id].own_texts = tail
        self._models[child_id].parent_id = parent_id
        self._models[child_id].prefix_length = prefix_length
        self._check_invariants()

    @rule()
    def reingest_with_edit(self) -> None:
        if self._deletion_done:
            return
        eligible = sorted(
            session_id
            for session_id, model in self._models.items()
            if session_id not in self._pending_children
            and (model.parent_id is None or self._models[model.parent_id].can_be_parent)
        )
        session_id = eligible[self._next_text % len(eligible)]
        model = self._models[session_id]
        updated_at = self._fresh_timestamp()
        replacement = self._new_text("edited")
        if model.parent_id is None:
            model.own_texts[-1] = replacement
            full_texts = model.own_texts
            parent_native_id = None
        else:
            model.own_texts[-1] = replacement
            full_texts = [*self._logical_texts(model.parent_id)[: model.prefix_length], *model.own_texts]
            parent_native_id = self._models[model.parent_id].native_id
        self._write(model.native_id, full_texts, parent_native_id=parent_native_id, updated_at=updated_at)
        model.updated_at = updated_at
        model.can_be_parent = True
        self._check_invariants()

    @rule()
    def merge_append(self) -> None:
        session_id = self._choose_session_id()
        model = self._models[session_id]
        appended = [self._new_text("append"), self._new_text("variant")]
        self._write(
            model.native_id,
            appended,
            merge_append=True,
            message_id_prefix=f"append-{model.appended_batches}",
            variant_batch=True,
        )
        model.own_texts.extend(appended)
        model.appended_batches += 1
        # Prefix extraction deliberately follows active (variant zero) paths;
        # an appended sibling variant is therefore not a replayable parent.
        model.can_be_parent = False
        self._check_invariants()

    @rule()
    def full_replace_with_sibling_variants(self) -> None:
        """A full replacement must preserve same-position variant ordering."""
        candidates = sorted(
            session_id
            for session_id, model in self._models.items()
            if model.parent_id is None
            and session_id not in self._pending_children
            and not any(other.parent_id == session_id for other in self._models.values())
        )
        if not candidates:
            return
        session_id = candidates[self._next_text % len(candidates)]
        model = self._models[session_id]
        variant_texts = [self._new_text("replace-primary"), self._new_text("replace-variant")]
        self._write(
            model.native_id,
            variant_texts,
            updated_at=self._fresh_timestamp(),
            message_id_prefix=f"replace-{model.appended_batches}",
            variant_batch=True,
        )
        model.own_texts = variant_texts
        model.can_be_parent = False
        rows = self._conn.execute(
            "SELECT position, variant_index FROM messages WHERE session_id = ? ORDER BY position, variant_index",
            (session_id,),
        ).fetchall()
        assert [tuple(row) for row in rows] == [(0, 0), (0, 1)]
        self._check_invariants()

    @rule()
    def stale_replace_attempt(self) -> None:
        session_id = self._choose_session_id()
        model = self._models[session_id]
        before = self._message_texts(session_id)
        stale_texts = [self._new_text("stale")]
        self._write(
            model.native_id,
            stale_texts,
            parent_native_id=self._models[model.parent_id].native_id if model.parent_id else None,
            updated_at="2000-01-01T00:00:00Z",
        )
        assert self._message_texts(session_id) == before
        self._check_invariants()

    @rule()
    def delete_parent_branch_point(self) -> None:
        """Hard-delete a parent branch point: child reads must tail-truncate."""
        if self._deletion_done:
            return
        children = [
            (session_id, model)
            for session_id, model in self._models.items()
            if model.parent_id is not None
            and model.prefix_length > 0
            and self._models[model.parent_id].parent_id is None
            and self._models[model.parent_id].can_be_parent
        ]
        if not children:
            return
        child_id, child = children[self._next_text % len(children)]
        parent_id = child.parent_id
        assert parent_id is not None
        parent_messages = read_archive_session_envelope(self._conn, parent_id).messages
        branch_point = parent_messages[child.prefix_length - 1].message_id
        self._conn.execute("DELETE FROM messages WHERE message_id = ?", (branch_point,))
        self._conn.commit()
        self._deletion_done = True
        self._models[parent_id].own_texts.pop(child.prefix_length - 1)
        self._models[parent_id].can_be_parent = False
        for candidate_id, candidate in self._models.items():
            if self._has_ancestor(candidate_id, parent_id):
                candidate.can_be_parent = False

        envelope = read_archive_session_envelope(self._conn, child_id)
        assert envelope.lineage_complete is False
        assert self._texts_from_envelope(envelope) == child.own_texts
        self._check_invariants()

    def _ingest_parent(self) -> None:
        native_id = self._new_native_id("parent")
        texts = [self._new_text("parent"), self._new_text("variant")]
        session_id = self._write(native_id, texts)
        self._models[session_id] = _SessionModel(native_id=native_id, own_texts=texts)

    def _write(
        self,
        native_id: str,
        texts: list[str],
        *,
        parent_native_id: str | None = None,
        updated_at: str = "2026-01-01T00:00:00Z",
        merge_append: bool = False,
        message_id_prefix: str = "message",
        variant_batch: bool = False,
    ) -> str:
        parsed = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id=native_id,
            parent_session_provider_id=parent_native_id,
            branch_type=BranchType.FORK if parent_native_id else None,
            updated_at=updated_at,
            messages=[
                ParsedMessage(
                    provider_message_id=f"{native_id}-{message_id_prefix}-{index}",
                    role=Role.USER if index % 2 == 0 else Role.ASSISTANT,
                    text=text,
                    # Full snapshots use active-path messages.  Append windows
                    # add a sibling variant at one position, exercising the
                    # variant-index regeneration branch without pretending a
                    # sibling is part of a lineage-replayable prefix.
                    position=0 if variant_batch else index,
                    variant_index=index if variant_batch else 0,
                )
                for index, text in enumerate(texts)
            ],
        )
        return write_parsed_session_to_archive(
            self._conn,
            parsed,
            content_hash=session_content_hash(parsed),
            merge_append=merge_append,
        )

    def _logical_texts(self, session_id: str) -> list[str]:
        model = self._models[session_id]
        if model.parent_id is None:
            return list(model.own_texts)
        return [*self._logical_texts(model.parent_id)[: model.prefix_length], *model.own_texts]

    def _choose_session_id(self) -> str:
        session_ids = sorted(session_id for session_id in self._models if session_id not in self._pending_children)
        return session_ids[self._next_text % len(session_ids)]

    def _choose_parent_id(self) -> str:
        parent_ids = sorted(session_id for session_id, model in self._models.items() if model.can_be_parent)
        return parent_ids[self._next_text % len(parent_ids)]

    def _has_ancestor(self, session_id: str, ancestor_id: str) -> bool:
        cursor = self._models[session_id].parent_id
        while cursor is not None:
            if cursor == ancestor_id:
                return True
            cursor = self._models[cursor].parent_id
        return False

    def _new_native_id(self, kind: str) -> str:
        native_id = f"{kind}-{self._next_session}"
        self._next_session += 1
        return native_id

    def _new_text(self, kind: str) -> str:
        text = f"{kind}-text-{self._next_text}"
        self._next_text += 1
        return text

    def _fresh_timestamp(self) -> str:
        timestamp = f"2027-01-01T00:00:{self._next_fresh_version:02d}Z"
        self._next_fresh_version += 1
        return timestamp

    def _message_texts(self, session_id: str) -> list[str]:
        return self._texts_from_envelope(read_archive_session_envelope(self._conn, session_id))

    @staticmethod
    def _texts_from_envelope(envelope: ArchiveSessionEnvelope) -> list[str]:
        messages = envelope.messages
        return [str(block.text) for message in messages for block in message.blocks if block.text is not None]

    def _check_invariants(self) -> None:
        status_rows = self._conn.execute("SELECT status FROM session_links").fetchall()
        assert {row[0] for row in status_rows}.issubset(
            {None, TopologyEdgeStatus.REPAIRED.value, TopologyEdgeStatus.QUARANTINED.value}
        )

        indexed_docids = {row[0] for row in self._conn.execute("SELECT id FROM messages_fts_docsize").fetchall()}
        indexable_docids = {
            row[0] for row in self._conn.execute("SELECT rowid FROM blocks WHERE search_text != ''").fetchall()
        }
        assert indexed_docids == indexable_docids

        for session_id, model in self._models.items():
            pending = self._pending_children.get(session_id)
            if pending is not None:
                self._assert_pending_link(session_id, pending[0])
                continue
            envelope = read_archive_session_envelope(self._conn, session_id)
            actual = self._texts_from_envelope(envelope)
            if envelope.lineage_complete:
                assert actual == self._logical_texts(session_id)
            else:
                # #4ts.6: a dangling branch point may only return the child
                # tail or a surviving suffix; it must never over-extend the
                # model transcript with messages beyond that logical branch.
                logical = self._logical_texts(session_id)
                assert actual == logical[len(logical) - len(actual) :]

            if model.parent_id is not None and envelope.lineage_complete:
                self._assert_resolved_link(session_id, model)
                physical = self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
                assert physical == len(model.own_texts)

            variant_rows = self._conn.execute(
                "SELECT position, variant_index FROM messages WHERE session_id = ?", (session_id,)
            ).fetchall()
            assert len({(int(row[0]), int(row[1])) for row in variant_rows}) == len(variant_rows)

    def _assert_pending_link(self, child_id: str, parent_native_id: str) -> None:
        row = self._conn.execute(
            """
            SELECT resolved_dst_session_id, branch_point_message_id, inheritance, status
            FROM session_links
            WHERE src_session_id = ? AND dst_native_id = ?
            """,
            (child_id, parent_native_id),
        ).fetchone()
        assert tuple(row) == (None, None, None, None)

    def _assert_resolved_link(self, child_id: str, model: _SessionModel) -> None:
        assert model.parent_id is not None
        row = self._conn.execute(
            """
            SELECT resolved_dst_session_id, branch_point_message_id, inheritance, status
            FROM session_links
            WHERE src_session_id = ?
            """,
            (child_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == model.parent_id
        assert row[2] == ("prefix-sharing" if model.prefix_length else "spawned-fresh")
        assert row[3] is None
        if model.prefix_length:
            parent_messages = read_archive_session_envelope(self._conn, model.parent_id).messages
            assert row[1] == parent_messages[model.prefix_length - 1].message_id
        else:
            assert row[1] is None

    def teardown(self) -> None:
        self._conn.close()


def test_repository_get_messages_composes_prefix_sharing_child() -> None:
    """The public repository route composes a prefix-sharing child transcript."""
    with tempfile.TemporaryDirectory(prefix="polylogue-write-model-", dir="/realm/tmp") as root_text:
        archive_root = Path(root_text)
        initialize_active_archive_root(archive_root)
        db_path = archive_root / "index.db"
        conn = sqlite3.connect(str(db_path))
        try:
            parent = ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="repository-parent",
                messages=[
                    ParsedMessage(provider_message_id="parent-0", role=Role.USER, text="parent prompt", position=0),
                    ParsedMessage(provider_message_id="parent-1", role=Role.ASSISTANT, text="parent reply", position=1),
                ],
            )
            child = ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="repository-child",
                parent_session_provider_id="repository-parent",
                branch_type=BranchType.FORK,
                messages=[
                    ParsedMessage(provider_message_id="child-0", role=Role.USER, text="parent prompt", position=0),
                    ParsedMessage(provider_message_id="child-1", role=Role.ASSISTANT, text="parent reply", position=1),
                    ParsedMessage(provider_message_id="child-2", role=Role.USER, text="child tail", position=2),
                ],
            )
            write_parsed_session_to_archive(conn, parent, content_hash=session_content_hash(parent))
            child_id = write_parsed_session_to_archive(conn, child, content_hash=session_content_hash(child))
        finally:
            conn.close()

        async def read_texts() -> list[str]:
            async with SessionRepository(db_path=db_path) as repository:
                messages = await repository.get_messages(child_id)
            return [str(block.text) for message in messages for block in message.blocks if block.text is not None]

        assert asyncio.run(read_texts()) == ["parent prompt", "parent reply", "child tail"]


def test_session_link_resolver_quarantines_cycle() -> None:
    """The async resolver quarantines a late link that would close a cycle."""
    with tempfile.TemporaryDirectory(prefix="polylogue-write-model-", dir="/realm/tmp") as root_text:
        archive_root = Path(root_text)
        initialize_active_archive_root(archive_root)
        db_path = archive_root / "index.db"
        conn = sqlite3.connect(str(db_path))
        try:
            parent = ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="cycle-parent",
                messages=[ParsedMessage(provider_message_id="parent-0", role=Role.USER, text="parent", position=0)],
            )
            child = ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="cycle-child",
                parent_session_provider_id="cycle-parent",
                branch_type=BranchType.FORK,
                messages=[ParsedMessage(provider_message_id="child-0", role=Role.USER, text="child", position=0)],
            )
            parent_id = write_parsed_session_to_archive(conn, parent, content_hash=session_content_hash(parent))
            write_parsed_session_to_archive(conn, child, content_hash=session_content_hash(child))
            conn.execute(
                """
                INSERT INTO session_links (
                    src_session_id, dst_origin, dst_native_id, link_type,
                    status, method, confidence, evidence_json, observed_at_ms
                ) VALUES (?, 'claude-code-session', 'cycle-child', 'fork', NULL, 'test', 1.0, '[]', 0)
                """,
                (parent_id,),
            )
            conn.commit()
        finally:
            conn.close()

        async def resolve_cycle() -> int:
            async with aiosqlite.connect(db_path) as async_conn:
                await configure_connection(async_conn)
                resolved = await resolve_unresolved_links_for_child(
                    async_conn,
                    src_session_id=parent_id,
                    resolved_at="2026-01-01T00:00:00Z",
                )
                await async_conn.commit()
            return resolved

        assert asyncio.run(resolve_cycle()) == 0
        with sqlite3.connect(str(db_path)) as verify_conn:
            row = verify_conn.execute(
                "SELECT resolved_dst_session_id, status, evidence_json FROM session_links WHERE src_session_id = ?",
                (parent_id,),
            ).fetchone()
        assert row is not None
        assert row[0] is None
        assert row[1] == TopologyEdgeStatus.QUARANTINED.value
        assert '"reason": "cycle_rejected"' in row[2]


TestWritePathStateMachine = WritePathStateMachine.TestCase
TestWritePathStateMachine.settings = settings(
    stateful_step_count=18,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
