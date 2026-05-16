"""Shell-completion aggregates exposed through ``ArchiveOperations``.

Surface-side completion callbacks (CLI repo/cwd/tool name completions)
historically reached into ``session_profiles`` / ``action_events`` via
raw SQL. These typed aggregates own the SQL inside the operation
boundary so the surface stays a leaf adapter (#860).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


class CompletionAggregate:
    """Single (value, count) pair for shell-completion aggregations."""

    __slots__ = ("value", "count")

    def __init__(self, value: str, count: int) -> None:
        self.value = value
        self.count = count

    def __repr__(self) -> str:
        return f"CompletionAggregate(value={self.value!r}, count={self.count})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompletionAggregate):
            return NotImplemented
        return self.value == other.value and self.count == other.count

    def __hash__(self) -> int:
        return hash((self.value, self.count))


class ArchiveCompletionMixin:
    """Aggregations consumed by CLI shell-completion callbacks."""

    if TYPE_CHECKING:

        @property
        def backend(self) -> SQLiteBackend: ...

    async def list_session_repo_names(
        self,
        *,
        prefix: str = "",
        limit: int = 32,
    ) -> list[CompletionAggregate]:
        """Return ``(repo_name, session_count)`` aggregates from ``session_profiles``."""
        like = f"{prefix}%"
        async with self.backend.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT
                    repo.value AS value,
                    COUNT(*) AS cnt
                FROM session_profiles AS sp,
                     json_each(COALESCE(sp.repo_names_json, '[]')) AS repo
                WHERE (? = '' OR repo.value LIKE ?)
                GROUP BY repo.value
                ORDER BY cnt DESC, repo.value ASC
                LIMIT ?
                """,
                (prefix, like, limit),
            )
            rows = await cursor.fetchall()
        return [CompletionAggregate(str(row["value"]), int(row["cnt"])) for row in rows]

    async def list_session_cwd_prefixes(
        self,
        *,
        prefix: str = "",
        limit: int = 32,
    ) -> list[CompletionAggregate]:
        """Return ``(cwd_path, session_count)`` aggregates from ``session_profiles``."""
        like = f"{prefix}%"
        async with self.backend.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT
                    cwd.value AS value,
                    COUNT(*) AS cnt
                FROM session_profiles AS sp,
                     json_each(COALESCE(json_extract(sp.evidence_payload_json, '$.cwd_paths'), '[]')) AS cwd
                WHERE (? = '' OR cwd.value LIKE ?)
                GROUP BY cwd.value
                ORDER BY cnt DESC, cwd.value ASC
                LIMIT ?
                """,
                (prefix, like, limit),
            )
            rows = await cursor.fetchall()
        return [CompletionAggregate(str(row["value"]), int(row["cnt"])) for row in rows]

    async def list_action_tool_names(
        self,
        *,
        prefix: str = "",
        limit: int = 32,
    ) -> list[CompletionAggregate]:
        """Return ``(tool_name, action_count)`` aggregates from ``action_events``."""
        like = f"{prefix}%"
        async with self.backend.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT
                    normalized_tool_name AS value,
                    COUNT(*) AS cnt
                FROM action_events
                WHERE (? = '' OR normalized_tool_name LIKE ?)
                GROUP BY normalized_tool_name
                ORDER BY cnt DESC, normalized_tool_name ASC
                LIMIT ?
                """,
                (prefix, like, limit),
            )
            rows = await cursor.fetchall()
        return [CompletionAggregate(str(row["value"]), int(row["cnt"])) for row in rows]


__all__ = ["ArchiveCompletionMixin", "CompletionAggregate"]
