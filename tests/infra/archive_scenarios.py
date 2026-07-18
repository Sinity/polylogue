"""Declarative archive scenarios for verification harnesses.

These helpers keep tests focused on semantic expectations instead of repeating
session-builder, repository, and SQL plumbing in every suite.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

from polylogue.api import Polylogue
from polylogue.core.enums import BlockType, Origin, Role
from polylogue.core.json import JSONDocument, JSONValue, require_json_document
from polylogue.core.types import AttachmentId, ContentHash, MessageId, SessionId
from polylogue.storage.hydrators import session_from_records
from polylogue.storage.runtime import AttachmentRecord, BlockRecord, MessageRecord, SessionRecord
from tests.infra.semantic_facts import SessionFacts
from tests.infra.storage_records import JSONRecord, SessionBuilder, db_setup

ScenarioProvider: TypeAlias = str
_FIXTURE_TIMESTAMP = "2026-01-01T00:00:00+00:00"


@contextmanager
def open_index_db(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open the ``index.db`` for direct read access."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def native_session_id_for(provider: str, session_id: str) -> str:
    """Native ``<origin>:<native_id>`` session id for a seeded scenario.

    Mirrors ``SessionBuilder.native_session_id``: the builder seeds each
    session with ``provider_session_id = "ext-<session_id>"``
    and the origin derived from ``provider``.
    """
    from polylogue.core.enums import Provider
    from polylogue.core.identity_law import session_id as archive_session_id
    from polylogue.core.sources import origin_from_provider

    origin = origin_from_provider(Provider.from_string(provider))
    return archive_session_id(origin.value, f"ext-{session_id}")


@dataclass(frozen=True, slots=True)
class ScenarioContentBlock:
    """Typed content-block fixture for authored archive scenarios."""

    block_type: str = "text"
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: JSONValue | None = None
    media_type: str | None = None
    metadata: JSONDocument | None = None
    semantic_type: str | None = None
    tool_result_is_error: int | None = None
    tool_result_exit_code: int | None = None

    @classmethod
    def text_block(cls, text: str, *, semantic_type: str | None = None) -> ScenarioContentBlock:
        return cls(block_type="text", text=text, semantic_type=semantic_type)

    @classmethod
    def tool_use(
        cls,
        *,
        tool_name: str,
        tool_input: JSONValue | None = None,
        tool_id: str | None = None,
    ) -> ScenarioContentBlock:
        return cls(block_type="tool_use", tool_name=tool_name, tool_id=tool_id, tool_input=tool_input)

    @classmethod
    def tool_result(
        cls,
        text: str,
        *,
        tool_name: str | None = None,
        tool_id: str | None = None,
        is_error: bool | None = None,
        exit_code: int | None = None,
    ) -> ScenarioContentBlock:
        return cls(
            block_type="tool_result",
            text=text,
            tool_name=tool_name,
            tool_id=tool_id,
            tool_result_is_error=None if is_error is None else int(is_error),
            tool_result_exit_code=exit_code,
        )

    @classmethod
    def thinking(cls, text: str) -> ScenarioContentBlock:
        return cls(block_type="thinking", text=text)

    def to_payload(self) -> JSONDocument:
        payload: JSONDocument = {"type": self.block_type}
        if self.text is not None:
            payload["text"] = self.text
        if self.tool_name is not None:
            payload["tool_name"] = self.tool_name
        if self.tool_id is not None:
            payload["tool_id"] = self.tool_id
        if self.tool_input is not None:
            payload["tool_input"] = self.tool_input
        if self.media_type is not None:
            payload["media_type"] = self.media_type
        if self.metadata is not None:
            payload["metadata"] = dict(require_json_document(self.metadata, context="scenario content block metadata"))
        if self.semantic_type is not None:
            payload["semantic_type"] = self.semantic_type
        if self.tool_result_is_error is not None:
            payload["tool_result_is_error"] = self.tool_result_is_error
        if self.tool_result_exit_code is not None:
            payload["tool_result_exit_code"] = self.tool_result_exit_code
        return payload


@dataclass(frozen=True, slots=True)
class ScenarioAttachment:
    """Attachment fixture data associated with a scenario message."""

    attachment_id: str | None = None
    mime_type: str = "application/octet-stream"
    size_bytes: int = 1024
    path: str | None = None
    provider_meta: JSONRecord | None = None


@dataclass(frozen=True, slots=True)
class ScenarioMessage:
    """Message fixture data used by ``ArchiveScenario``."""

    role: str = "user"
    text: str = "Test message"
    message_id: str | None = None
    timestamp: str | None = None
    provider_meta: JSONRecord | None = None
    blocks: Sequence[ScenarioContentBlock] = ()
    attachments: Sequence[ScenarioAttachment] = ()


@dataclass(frozen=True, slots=True)
class ArchiveScenarioSeed:
    """Result of seeding one archive scenario."""

    scenario: ArchiveScenario
    session_id: str

    def facts_from_connection(self, conn: sqlite3.Connection) -> SessionFacts:
        return self.scenario.facts_from_connection(conn)

    async def facts_from_archive(self, archive: Polylogue) -> SessionFacts:
        return await self.scenario.facts_from_archive(archive)


@dataclass(frozen=True, slots=True)
class ArchiveScenario:
    """A minimal semantic archive fixture that can be projected through surfaces."""

    name: str
    provider: ScenarioProvider = "test"
    title: str = "Test Session"
    messages: Sequence[ScenarioMessage] = field(default_factory=tuple)
    session_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: JSONRecord | None = None

    @property
    def resolved_session_id(self) -> str:
        return self.session_id or self.name

    @property
    def native_session_id(self) -> str:
        """Native ``<origin>:ext-<id>`` session id this scenario seeds under."""
        return native_session_id_for(self.provider, self.resolved_session_id)

    def seed(self, db_path: Path) -> ArchiveScenarioSeed:
        """Persist the scenario into ``db_path`` through the standard builder."""
        timestamp = self.created_at or _default_timestamp()
        builder = (
            SessionBuilder(db_path, self.resolved_session_id)
            .provider(self.provider)
            .title(self.title)
            .created_at(timestamp)
            .updated_at(self.updated_at or timestamp)
            .metadata(self.metadata)
        )
        for message_index, message in enumerate(self.messages or _default_messages(), start=1):
            message_id = message.message_id or f"m{message_index}"
            message_kwargs: dict[str, object] = {"blocks": _content_block_payloads(message.blocks)}
            if message.provider_meta is not None:
                message_kwargs["provider_meta"] = message.provider_meta
            builder.add_message(
                message_id=message_id,
                role=message.role,
                text=message.text,
                timestamp=message.timestamp,
                **message_kwargs,
            )
            for attachment in message.attachments:
                builder.add_attachment(
                    attachment_id=attachment.attachment_id,
                    message_id=message_id,
                    mime_type=attachment.mime_type,
                    size_bytes=attachment.size_bytes,
                    path=attachment.path,
                )
        builder.save()
        # Native user tags live as ``user.db`` assertions keyed by the generated
        # session id. Seed them through the same archive primitive
        # (``ArchiveStore.add_user_tags``) the public tag API uses.
        metadata_tags = self.metadata.get("tags") if self.metadata else None
        if isinstance(metadata_tags, list):
            from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

            tag_values = tuple(tag for tag in metadata_tags if isinstance(tag, str))
            if tag_values:
                with ArchiveStore(_archive_root_for_index_db(db_path)) as archive:
                    archive.add_user_tags((self.native_session_id,), tag_values)
        return ArchiveScenarioSeed(scenario=self, session_id=self.resolved_session_id)

    def facts_from_connection(self, conn: sqlite3.Connection) -> SessionFacts:
        """Read scenario facts directly from storage records."""
        conv_record, msg_records, attachment_records = self.records_from_connection(conn)
        return SessionFacts.from_records(conv_record, msg_records, attachment_records)

    def hydrated_facts_from_connection(self, conn: sqlite3.Connection) -> SessionFacts:
        """Hydrate storage records and extract domain-level scenario facts."""
        conv_record, msg_records, attachment_records = self.records_from_connection(conn)
        return SessionFacts.from_domain_session(session_from_records(conv_record, msg_records, attachment_records))

    def records_from_connection(
        self,
        conn: sqlite3.Connection,
    ) -> tuple[SessionRecord, list[MessageRecord], list[AttachmentRecord]]:
        """Read scenario storage records from an archive `index.db` connection.

        Reads the archive ``sessions`` / ``messages`` / ``blocks`` tables
        directly (keyed by the generated session id) and projects them into
        the storage-record shape consumed by ``SessionFacts.from_records``.
        """
        return read_session_records(conn, self.native_session_id)

    async def facts_from_archive(self, archive: Polylogue) -> SessionFacts:
        session = await archive.get_session(self.native_session_id)
        if session is None:
            raise AssertionError(f"Scenario session {self.resolved_session_id!r} not found through facade")
        return SessionFacts.from_domain_session(session)


def read_session_records(
    conn: sqlite3.Connection,
    session_id: str,
) -> tuple[SessionRecord, list[MessageRecord], list[AttachmentRecord]]:
    """Read storage records for one archive session id from an ``index.db`` connection.

    Reads the archive ``sessions`` / ``messages`` / ``blocks`` / attachment
    tables directly and projects them into the storage-record shape consumed
    by ``SessionFacts.from_records`` and ``session_from_records``.
    """
    session_row = conn.execute(
        "SELECT session_id, native_id, origin, title FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if session_row is None:
        raise AssertionError(f"Session {session_id!r} was not seeded")
    origin = Origin.from_string(str(session_row["origin"]))
    conv_record = SessionRecord(
        session_id=SessionId(session_id),
        native_id=str(session_row["native_id"]),
        origin=origin,
        title=session_row["title"],
        content_hash=ContentHash(session_id),
    )
    msg_rows = conn.execute(
        """
        SELECT message_id, role, position, word_count, has_tool_use, has_thinking
        FROM messages
        WHERE session_id = ?
        ORDER BY position, variant_index
        """,
        (session_id,),
    ).fetchall()
    content_blocks_by_message = _content_blocks_by_message(conn, session_id)
    msg_records = [
        MessageRecord(
            message_id=MessageId(row["message_id"]),
            session_id=SessionId(session_id),
            source_name=origin.value,
            role=Role(str(row["role"])),
            text=_message_text(content_blocks_by_message.get(str(row["message_id"]), [])),
            content_hash=ContentHash(str(row["message_id"])),
            word_count=int(row["word_count"] or 0),
            has_tool_use=int(row["has_tool_use"] or 0),
            has_thinking=int(row["has_thinking"] or 0),
            blocks=content_blocks_by_message.get(str(row["message_id"]), []),
        )
        for row in msg_rows
    ]
    attachment_records = _attachment_records_for_session(conn, session_id)
    return conv_record, msg_records, attachment_records


def _default_timestamp() -> str:
    return _FIXTURE_TIMESTAMP


def _default_messages() -> tuple[ScenarioMessage, ...]:
    return (ScenarioMessage(role="user", text="Test message"),)


def _content_block_payloads(blocks: Sequence[ScenarioContentBlock]) -> list[JSONDocument]:
    payloads: list[JSONDocument] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, ScenarioContentBlock):
            raise TypeError(
                "ArchiveScenario content blocks must use ScenarioContentBlock fixtures "
                f"(content_blocks[{index}]={block!r})"
            )
        payloads.append(block.to_payload())
    return payloads


def _message_text(blocks: Sequence[BlockRecord]) -> str | None:
    """Concatenate text-block content the way archive hydration does."""
    texts = [block.text for block in blocks if block.type == "text" and block.text]
    return "\n".join(texts) if texts else None


def _attachment_records_for_session(conn: sqlite3.Connection, session_id: str) -> list[AttachmentRecord]:
    rows = conn.execute(
        """
        SELECT
            a.attachment_id,
            ar.session_id,
            ar.message_id,
            a.media_type,
            a.byte_count,
            a.display_name,
            ar.source_url
        FROM attachment_refs ar
        JOIN attachments a ON a.attachment_id = ar.attachment_id
        WHERE ar.session_id = ?
        ORDER BY ar.message_id, a.attachment_id
        """,
        (session_id,),
    ).fetchall()
    return [_attachment_record_from_row(row, session_id) for row in rows]


def _content_blocks_by_message(conn: sqlite3.Connection, session_id: str) -> dict[str, list[BlockRecord]]:
    rows = conn.execute(
        """
        SELECT message_id, position, block_type, text, tool_name, tool_id, tool_input, semantic_type
        FROM blocks
        WHERE session_id = ?
        ORDER BY message_id, position
        """,
        (session_id,),
    ).fetchall()
    blocks_by_message: dict[str, list[BlockRecord]] = {}
    for row in rows:
        message_id = str(row["message_id"])
        # ``BlockRecord.tool_input`` carries the serialized JSON string
        # exactly as the native ``blocks.tool_input`` column stores it.
        tool_input = row["tool_input"] if isinstance(row["tool_input"], str) else None
        block = BlockRecord(
            block_id=f"{message_id}:{row['position']}",
            message_id=MessageId(message_id),
            session_id=SessionId(session_id),
            block_index=int(row["position"]),
            type=BlockType(str(row["block_type"])),
            text=row["text"],
            tool_name=row["tool_name"],
            tool_id=row["tool_id"],
            tool_input=tool_input,
            semantic_type=row["semantic_type"],
        )
        blocks_by_message.setdefault(message_id, []).append(block)
    return blocks_by_message


def _attachment_record_from_row(row: sqlite3.Row, session_id: str) -> AttachmentRecord:
    return AttachmentRecord(
        attachment_id=AttachmentId(row["attachment_id"]),
        session_id=SessionId(session_id),
        message_id=MessageId(row["message_id"]) if row["message_id"] is not None else None,
        mime_type=row["media_type"],
        size_bytes=row["byte_count"],
        display_name=row["display_name"],
        source_url=row["source_url"],
    )


def seed_archive_scenarios(db_path: Path, scenarios: Iterable[ArchiveScenario]) -> list[ArchiveScenarioSeed]:
    """Seed all scenarios into one archive database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return [scenario.seed(db_path) for scenario in scenarios]


def seed_workspace_scenarios(
    workspace_env: Mapping[str, Path],
    scenarios: Iterable[ArchiveScenario],
) -> tuple[Path, list[ArchiveScenarioSeed]]:
    """Seed scenarios into the standard workspace database path."""
    db_path = db_setup(workspace_env)
    return db_path, seed_archive_scenarios(db_path, scenarios)


def _archive_root_for_index_db(db_path: Path) -> Path:
    """Archive root for an ``.../index.db`` scenario database path."""
    return db_path.parent


def archive_for_scenario_db(db_path: Path) -> Polylogue:
    """Open the native ``Polylogue`` facade over a scenario ``index.db``.

    The archive facade is the repository surface in the archive: it
    exposes ``get_session`` / ``list_sessions`` / ``add_tag`` /
    ``list_tags`` / ``delete_session`` over ``index.db``.
    """
    return Polylogue(archive_root=_archive_root_for_index_db(db_path), db_path=db_path)


# Stable name kept for scenario consumers: the "repository" is the native
# Polylogue facade over index.db.
repository_for_scenario_db = archive_for_scenario_db


__all__ = [
    "ArchiveScenario",
    "ArchiveScenarioSeed",
    "ScenarioAttachment",
    "ScenarioContentBlock",
    "ScenarioMessage",
    "archive_for_scenario_db",
    "native_session_id_for",
    "open_index_db",
    "read_session_records",
    "repository_for_scenario_db",
    "seed_archive_scenarios",
    "seed_workspace_scenarios",
]
