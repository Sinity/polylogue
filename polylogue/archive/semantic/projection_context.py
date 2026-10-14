"""Private terminal tool semantics for one complete transcript projection."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from typing import TYPE_CHECKING

from polylogue.archive.semantic.content_projection import ContentProjectionSpec, project_content_page
from polylogue.storage.sqlite.connection_profile import scratch_connection_context

if TYPE_CHECKING:
    from polylogue.archive.message.models import Message


class _ToolSemantics(Mapping[str, str]):
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def __getitem__(self, key: str) -> str:
        with closing(
            self.connection.execute("SELECT semantic_type FROM tool_dependencies WHERE tool_id=?", (key,))
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise KeyError(key)
        return str(row[0])

    def __iter__(self) -> Iterator[str]:
        with closing(self.connection.execute("SELECT tool_id FROM tool_dependencies ORDER BY tool_id")) as cursor:
            for row in cursor:
                yield str(row[0])

    def __len__(self) -> int:
        with closing(self.connection.execute("SELECT COUNT(*) FROM tool_dependencies")) as cursor:
            return int(cursor.fetchone()[0])


class ContentProjectionContext:
    """Retain only dependency coordinates and terminal semantics, never bodies."""

    def __init__(self, connection: sqlite3.Connection, checkpoint: Callable[[], None]) -> None:
        self.connection = connection
        self._checkpoint = checkpoint
        self._semantics = _ToolSemantics(connection)
        with closing(
            connection.execute(
                "CREATE TABLE tool_dependencies(tool_id TEXT PRIMARY KEY, semantic_type TEXT NOT NULL, "
                "message_position INTEGER NOT NULL, block_position INTEGER NOT NULL) WITHOUT ROWID"
            )
        ):
            pass

    def register_page(self, messages: Sequence[Message], *, offset: int) -> None:
        for position, message in enumerate(messages, offset):
            self._checkpoint()
            for block_position, block in enumerate(message.blocks):
                if block.get("type") != "tool_use":
                    continue
                tool_id = block.get("tool_id") or block.get("id")
                semantic_type = block.get("semantic_type")
                if (
                    not isinstance(tool_id, str)
                    or not tool_id
                    or not isinstance(semantic_type, str)
                    or not semantic_type
                ):
                    continue
                # Complete transcript order gives reused IDs their existing
                # terminal last-wins semantics, independently of output filters.
                with closing(
                    self.connection.execute(
                        "INSERT INTO tool_dependencies VALUES(?,?,?,?) ON CONFLICT(tool_id) DO UPDATE SET "
                        "semantic_type=excluded.semantic_type, message_position=excluded.message_position, "
                        "block_position=excluded.block_position",
                        (tool_id, semantic_type, position, block_position),
                    )
                ):
                    pass

    def project_page(self, messages: Iterable[Message], projection: ContentProjectionSpec) -> list[Message]:
        self._checkpoint()
        return project_content_page(messages, projection, tool_semantics=self._semantics)


@contextmanager
def content_projection_context(*, checkpoint: Callable[[], None] = lambda: None) -> Iterator[ContentProjectionContext]:
    """Close native scratch on its creator on success, cancellation and failure."""
    with scratch_connection_context(prefix="polylogue-projection-", filename="dependencies.sqlite") as connection:
        yield ContentProjectionContext(connection, checkpoint)
