"""Stateful write-path laws for full replace, append, and lineage (#g9f2).

The machine deliberately uses the production SQLite writer and its composed
archive reader.  Its model is just the expected logical transcript; it does
not duplicate prefix extraction, FTS maintenance, or session-link resolution.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from hypothesis import HealthCheck, settings
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import Provider, TopologyEdgeStatus
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveSessionEnvelope,
    read_archive_session_envelope,
    write_parsed_session_to_archive,
)
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
        self._next_session = 0
        self._next_text = 0
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
    def reingest_with_edit(self) -> None:
        if self._deletion_done:
            return
        eligible = sorted(
            session_id
            for session_id, model in self._models.items()
            if model.parent_id is None or self._models[model.parent_id].can_be_parent
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
        session_ids = sorted(self._models)
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
        return f"2027-01-{1 + (self._next_text % 27):02d}T00:00:00Z"

    def _message_texts(self, session_id: str) -> list[str]:
        return self._texts_from_envelope(read_archive_session_envelope(self._conn, session_id))

    @staticmethod
    def _texts_from_envelope(envelope: ArchiveSessionEnvelope) -> list[str]:
        messages = envelope.messages
        return [str(block.text) for message in messages for block in message.blocks if block.text is not None]

    def _check_invariants(self) -> None:
        legal_statuses = {status.value for status in TopologyEdgeStatus}
        status_rows = self._conn.execute(
            """
            SELECT COALESCE(
                status,
                CASE WHEN resolved_dst_session_id IS NULL THEN 'unresolved' ELSE 'resolved' END
            )
            FROM session_links
            """
        ).fetchall()
        assert {str(row[0]) for row in status_rows}.issubset(legal_statuses)

        indexed = self._conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
        indexable = self._conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0]
        assert indexed == indexable

        for session_id, model in self._models.items():
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
                physical = self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
                assert physical == len(model.own_texts)

            variant_rows = self._conn.execute(
                "SELECT position, variant_index FROM messages WHERE session_id = ?", (session_id,)
            ).fetchall()
            assert len({(int(row[0]), int(row[1])) for row in variant_rows}) == len(variant_rows)

    def teardown(self) -> None:
        self._conn.close()


TestWritePathStateMachine = WritePathStateMachine.TestCase
TestWritePathStateMachine.settings = settings(
    stateful_step_count=18,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
