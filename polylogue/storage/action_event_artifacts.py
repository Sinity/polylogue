"""Shared action-event artifact semantics for status, debt, and repair."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.maintenance_models import DerivedModelStatus


def _format_count_parts(parts: list[str]) -> str:
    if not parts:
        return ""
    return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class ActionEventArtifactState:
    """Canonical state for the action-event read model and its FTS index."""

    source_conversations: int
    materialized_conversations: int
    materialized_rows: int
    fts_rows: int
    stale_rows: int = 0
    orphan_rows: int = 0
    matches_version: bool = True

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, int | bool]) -> ActionEventArtifactState:
        return cls(
            source_conversations=int(metrics["action_source_documents"]),
            materialized_conversations=int(metrics["action_documents"]),
            materialized_rows=int(metrics["action_rows"]),
            fts_rows=int(metrics["action_fts_rows"]),
            stale_rows=int(metrics["action_stale_rows"]),
            orphan_rows=int(metrics["action_orphan_rows"]),
            matches_version=bool(metrics["action_matches_version"]),
        )

    @classmethod
    def from_status_snapshot(cls, status: Mapping[str, int | bool]) -> ActionEventArtifactState:
        return cls(
            source_conversations=int(status["valid_source_conversation_count"]),
            materialized_conversations=int(status["materialized_conversation_count"]),
            materialized_rows=int(status["count"]),
            fts_rows=int(status["action_fts_count"]),
            stale_rows=int(status["stale_count"]),
            orphan_rows=int(status["orphan_tool_block_count"]),
            matches_version=bool(status["matches_version"]),
        )

    @property
    def missing_conversations(self) -> int:
        return max(0, self.source_conversations - self.materialized_conversations)

    @property
    def pending_fts_rows(self) -> int:
        return max(0, self.materialized_rows - self.fts_rows)

    @property
    def excess_fts_rows(self) -> int:
        return max(0, self.fts_rows - self.materialized_rows)

    @property
    def rows_ready(self) -> bool:
        return self.source_conversations == 0 or (
            self.missing_conversations == 0 and self.stale_rows == 0 and self.orphan_rows == 0
        )

    @property
    def fts_ready(self) -> bool:
        return self.pending_fts_rows == 0 and self.excess_fts_rows == 0

    @property
    def ready(self) -> bool:
        return self.rows_ready and self.fts_ready

    @property
    def repair_item_count(self) -> int:
        return (
            self.missing_conversations
            + self.stale_rows
            + self.orphan_rows
            + self.pending_fts_rows
            + self.excess_fts_rows
        )

    def repair_detail(self) -> str:
        if self.repair_item_count == 0:
            return "Action-event read model ready"

        parts: list[str] = []
        if self.missing_conversations:
            parts.append(f"{self.missing_conversations:,} missing conversations")
        if self.stale_rows:
            parts.append(f"{self.stale_rows:,} stale action-event rows")
        if self.orphan_rows:
            parts.append(f"{self.orphan_rows:,} orphan action-event rows")
        if self.pending_fts_rows:
            parts.append(f"{self.pending_fts_rows:,} pending action-event FTS rows")
        if self.excess_fts_rows:
            parts.append(f"{self.excess_fts_rows:,} stale extra action-event FTS rows")
        return f"Action-event read model pending ({_format_count_parts(parts)})"

    def row_status(self) -> DerivedModelStatus:
        return DerivedModelStatus(
            name="action_events",
            ready=self.rows_ready,
            detail=(
                f"Action-event rows ready ({self.materialized_conversations:,}/{self.source_conversations:,} conversations)"
                if self.rows_ready
                else (
                    f"Action-event rows pending ({self.materialized_conversations:,}/{self.source_conversations:,} conversations; "
                    f"stale rows {self.stale_rows:,}, orphan rows {self.orphan_rows:,})"
                )
            ),
            source_documents=self.source_conversations,
            materialized_documents=self.materialized_conversations,
            materialized_rows=self.materialized_rows,
            pending_documents=self.missing_conversations,
            stale_rows=self.stale_rows,
            orphan_rows=self.orphan_rows,
            matches_version=self.matches_version,
        )

    def fts_status(self) -> DerivedModelStatus:
        if self.fts_ready:
            detail = f"Action-event FTS ready ({self.fts_rows:,}/{self.materialized_rows:,} rows)"
        else:
            parts: list[str] = []
            if self.pending_fts_rows:
                parts.append(f"{self.pending_fts_rows:,} pending")
            if self.excess_fts_rows:
                parts.append(f"{self.excess_fts_rows:,} stale extra")
            detail = (
                f"Action-event FTS pending ({self.fts_rows:,}/{self.materialized_rows:,} rows; "
                f"{_format_count_parts(parts)} rows)"
            )
        return DerivedModelStatus(
            name="action_events_fts",
            ready=self.fts_ready,
            detail=detail,
            source_rows=self.materialized_rows,
            materialized_rows=self.fts_rows,
            pending_rows=self.pending_fts_rows,
            stale_rows=self.excess_fts_rows,
            orphan_rows=self.orphan_rows,
        )


__all__ = ["ActionEventArtifactState"]
