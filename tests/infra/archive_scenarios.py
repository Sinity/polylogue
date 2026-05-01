"""Declarative archive scenarios for verification harnesses.

These helpers keep tests focused on semantic expectations instead of repeating
conversation-builder, repository, and SQL plumbing in every suite.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

from polylogue.core.json import JSONDocument, JSONValue, json_document, loads, require_json_document
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.queries.mappers_archive import (
    _row_to_content_block,
    _row_to_conversation,
    _row_to_message,
)
from polylogue.storage.hydrators import conversation_from_records
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.runtime import AttachmentRecord, ContentBlockRecord, ConversationRecord, MessageRecord
from polylogue.types import AttachmentId, ConversationId, MessageId
from tests.infra.semantic_facts import ConversationFacts
from tests.infra.storage_records import ConversationBuilder, JSONRecord, db_setup

ScenarioProvider: TypeAlias = str
_FIXTURE_TIMESTAMP = "2026-01-01T00:00:00+00:00"


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
    def tool_result(cls, text: str, *, tool_name: str | None = None) -> ScenarioContentBlock:
        return cls(block_type="tool_result", text=text, tool_name=tool_name)

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
    content_blocks: Sequence[ScenarioContentBlock] = ()
    attachments: Sequence[ScenarioAttachment] = ()


@dataclass(frozen=True, slots=True)
class ArchiveScenarioSeed:
    """Result of seeding one archive scenario."""

    scenario: ArchiveScenario
    conversation_id: str

    def facts_from_connection(self, conn: sqlite3.Connection) -> ConversationFacts:
        return self.scenario.facts_from_connection(conn)

    async def facts_from_repository(self, repository: ConversationRepository) -> ConversationFacts:
        return await self.scenario.facts_from_repository(repository)


@dataclass(frozen=True, slots=True)
class ArchiveScenario:
    """A minimal semantic archive fixture that can be projected through surfaces."""

    name: str
    provider: ScenarioProvider = "test"
    title: str = "Test Conversation"
    messages: Sequence[ScenarioMessage] = field(default_factory=tuple)
    conversation_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: JSONRecord | None = None

    @property
    def resolved_conversation_id(self) -> str:
        return self.conversation_id or self.name

    def seed(self, db_path: Path) -> ArchiveScenarioSeed:
        """Persist the scenario into ``db_path`` through the standard builder."""
        timestamp = self.created_at or _default_timestamp()
        builder = (
            ConversationBuilder(db_path, self.resolved_conversation_id)
            .provider(self.provider)
            .title(self.title)
            .created_at(timestamp)
            .updated_at(self.updated_at or timestamp)
            .metadata(self.metadata)
        )
        for message_index, message in enumerate(self.messages or _default_messages(), start=1):
            message_id = message.message_id or f"m{message_index}"
            message_kwargs: dict[str, object] = {"content_blocks": _content_block_payloads(message.content_blocks)}
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
                    provider_meta=attachment.provider_meta,
                )
        builder.save()
        return ArchiveScenarioSeed(scenario=self, conversation_id=self.resolved_conversation_id)

    def facts_from_connection(self, conn: sqlite3.Connection) -> ConversationFacts:
        """Read scenario facts directly from storage records."""
        conv_record, msg_records, attachment_records = self.records_from_connection(conn)
        return ConversationFacts.from_records(conv_record, msg_records, attachment_records)

    def hydrated_facts_from_connection(self, conn: sqlite3.Connection) -> ConversationFacts:
        """Hydrate storage records and extract domain-level scenario facts."""
        conv_record, msg_records, attachment_records = self.records_from_connection(conn)
        return ConversationFacts.from_domain_conversation(
            conversation_from_records(conv_record, msg_records, attachment_records)
        )

    def records_from_connection(
        self,
        conn: sqlite3.Connection,
    ) -> tuple[ConversationRecord, list[MessageRecord], list[AttachmentRecord]]:
        """Read scenario storage records from a SQLite connection."""
        conv_row = conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (self.resolved_conversation_id,),
        ).fetchone()
        if conv_row is None:
            raise AssertionError(f"Scenario conversation {self.resolved_conversation_id!r} was not seeded")
        conv_record = _row_to_conversation(conv_row)
        msg_rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY sort_key, message_id",
            (self.resolved_conversation_id,),
        ).fetchall()
        content_blocks_by_message = _content_blocks_by_message(conn, self.resolved_conversation_id)
        msg_records = [
            _row_to_message(row).model_copy(
                update={"content_blocks": content_blocks_by_message.get(str(row["message_id"]), [])}
            )
            for row in msg_rows
        ]
        attachment_records = _attachment_records_for_conversation(conn, self.resolved_conversation_id)
        return conv_record, msg_records, attachment_records

    async def facts_from_repository(self, repository: ConversationRepository) -> ConversationFacts:
        conversation = await repository.get(self.resolved_conversation_id)
        if conversation is None:
            raise AssertionError(
                f"Scenario conversation {self.resolved_conversation_id!r} not found through repository"
            )
        return ConversationFacts.from_domain_conversation(conversation)


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


def _attachment_records_for_conversation(conn: sqlite3.Connection, conversation_id: str) -> list[AttachmentRecord]:
    rows = conn.execute(
        """
        SELECT
            a.attachment_id,
            ar.conversation_id,
            ar.message_id,
            a.mime_type,
            a.size_bytes,
            a.path,
            a.provider_meta
        FROM attachment_refs ar
        JOIN attachments a ON a.attachment_id = ar.attachment_id
        WHERE ar.conversation_id = ?
        ORDER BY ar.message_id, a.attachment_id
        """,
        (conversation_id,),
    ).fetchall()
    return [_attachment_record_from_row(row) for row in rows]


def _content_blocks_by_message(conn: sqlite3.Connection, conversation_id: str) -> dict[str, list[ContentBlockRecord]]:
    rows = conn.execute(
        """
        SELECT *
        FROM content_blocks
        WHERE conversation_id = ?
        ORDER BY message_id, block_index
        """,
        (conversation_id,),
    ).fetchall()
    blocks_by_message: dict[str, list[ContentBlockRecord]] = {}
    for row in rows:
        blocks_by_message.setdefault(str(row["message_id"]), []).append(_row_to_content_block(row))
    return blocks_by_message


def _attachment_record_from_row(row: sqlite3.Row) -> AttachmentRecord:
    raw_provider_meta = row["provider_meta"]
    provider_meta: JSONRecord | None = None
    if isinstance(raw_provider_meta, str) and raw_provider_meta:
        provider_meta = {}
        for key, value in json_document(loads(raw_provider_meta)).items():
            provider_meta[key] = value
    return AttachmentRecord(
        attachment_id=AttachmentId(row["attachment_id"]),
        conversation_id=ConversationId(row["conversation_id"]),
        message_id=MessageId(row["message_id"]) if row["message_id"] is not None else None,
        mime_type=row["mime_type"],
        size_bytes=row["size_bytes"],
        path=row["path"],
        provider_meta=provider_meta,
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


def repository_for_scenario_db(db_path: Path) -> ConversationRepository:
    """Open a repository over a scenario database."""
    return ConversationRepository(backend=SQLiteBackend(db_path=db_path))


__all__ = [
    "ArchiveScenario",
    "ArchiveScenarioSeed",
    "ScenarioAttachment",
    "ScenarioContentBlock",
    "ScenarioMessage",
    "repository_for_scenario_db",
    "seed_archive_scenarios",
    "seed_workspace_scenarios",
]
