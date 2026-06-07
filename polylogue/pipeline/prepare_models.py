"""Typed models and caches for parsed-session preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from polylogue.pipeline.materialization_runtime import _timestamp_sort_key
from polylogue.storage.archive_views import ExistingSession
from polylogue.types import ContentHash, SessionId

if TYPE_CHECKING:
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


class SaveResult(BaseModel):
    sessions: int
    messages: int
    attachments: int
    session_events: int = 0
    skipped_sessions: int
    skipped_messages: int
    skipped_attachments: int
    skipped_session_events: int = 0


@dataclass
class PrepareCache:
    """Pre-loaded batch data for prepare_records."""

    existing: dict[str, ExistingSession] = field(default_factory=dict)
    known_ids: set[str] = field(default_factory=set)

    @classmethod
    async def load(cls, backend: SQLiteBackend, candidate_cids: set[str]) -> PrepareCache:
        cache = cls()
        if not candidate_cids:
            return cache

        cid_list = list(candidate_cids)
        for chunk_start in range(0, len(cid_list), 500):
            chunk = cid_list[chunk_start : chunk_start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            async with backend.connection() as conn:
                cursor = await conn.execute(
                    f"SELECT session_id, content_hash FROM sessions WHERE session_id IN ({placeholders})",
                    tuple(chunk),
                )
                rows = await cursor.fetchall()
            for row in rows:
                cid = row["session_id"]
                raw_content_hash = row["content_hash"]
                content_hash = raw_content_hash.hex() if isinstance(raw_content_hash, bytes) else str(raw_content_hash)
                cache.existing[cid] = ExistingSession(
                    session_id=cid,
                    content_hash=content_hash,
                )
                cache.known_ids.add(cid)

        return cache


@dataclass
class AttachmentMaterializationPlan:
    move_before_save: list[tuple[Path, Path]] = field(default_factory=list)
    delete_after_save: list[Path] = field(default_factory=list)


@dataclass
class TransformResult:
    session: ParsedSession
    materialization_plan: AttachmentMaterializationPlan
    content_hash: ContentHash
    candidate_cid: SessionId


@dataclass(frozen=True)
class PreparedBundle:
    session: ParsedSession
    materialization_plan: AttachmentMaterializationPlan
    content_hash: ContentHash
    cid: SessionId
    changed: bool


@dataclass(frozen=True)
class PersistedSessionResult:
    session_id: SessionId
    save_result: SaveResult
    content_changed: bool

    @property
    def counts(self) -> dict[str, int]:
        return self.save_result.model_dump()


__all__ = [
    "AttachmentMaterializationPlan",
    "PersistedSessionResult",
    "PreparedBundle",
    "PrepareCache",
    "SaveResult",
    "TransformResult",
    "_timestamp_sort_key",
]
