"""Synthetic archive pages for root transcript and summary-list contracts."""

from __future__ import annotations

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary
from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow, ArchiveMessageRow, ArchiveSessionEnvelope
from tests.infra.archive_store_double import ArchiveStoreDouble
from tests.infra.identity import archive_block_id, fixture_block_content_identity


class CliTranscriptArchive(ArchiveStoreDouble):
    session_id = "codex-session:fixture-child"

    def __init__(self, *, message_count: int = 2, gap_offset: int | None = None) -> None:
        super().__init__()
        self.gap_offset = gap_offset
        self.messages = tuple(
            ArchiveMessageRow(
                message_id=f"{self.session_id}:m{position}",
                native_id=f"m{position}",
                role="user",
                position=position,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=position == message_count - 1,
                blocks=(
                    ArchiveBlockRow(
                        block_id=archive_block_id(
                            f"{self.session_id}:m{position}",
                            content_identity=fixture_block_content_identity("text", f"tail message {position}"),
                        ),
                        message_id=f"{self.session_id}:m{position}",
                        block_type="text",
                        text=f"tail message {position}",
                        content_identity=fixture_block_content_identity("text", f"tail message {position}"),
                        content_occurrence=0,
                    ),
                ),
            )
            for position in range(message_count)
        )
        self.body_reads: list[tuple[int, int]] = []

    def resolve_session_id(self, token: str) -> str:
        if token != self.session_id:
            raise KeyError(token)
        return token

    def read_session_page(self, session_id: str, *, limit: int, offset: int) -> ArchiveSessionEnvelope:
        assert session_id == self.session_id
        self.body_reads.append((limit, offset))
        gap = offset == self.gap_offset
        return ArchiveSessionEnvelope(
            session_id=session_id,
            native_id="fixture-child",
            origin="codex-session",
            title="Fixture child",
            active_leaf_message_id=None,
            total_message_count=len(self.messages),
            messages=self.messages[offset : offset + limit],
            lineage_complete=not gap,
            lineage_truncation_reason="dangling_branch_point" if gap else None,
        )

    def read_summary(self, session_id: str) -> ArchiveSessionSummary:
        return ArchiveSessionSummary(
            session_id=session_id,
            native_id="fixture-child",
            origin="codex-session",
            title="Fixture child",
            created_at=None,
            updated_at=None,
            message_count=len(self.messages),
            word_count=0,
            tags=(),
        )

    def list_summaries(self, *, limit: int = 50, **filters: object) -> list[ArchiveSessionSummary]:
        if filters.get("session_id") != self.session_id or filters.get("offset", 0):
            return []
        return [self.read_summary(self.session_id)][:limit]

    def count_sessions(self, **kwargs: object) -> int:
        return int(kwargs.get("session_id") == self.session_id)
