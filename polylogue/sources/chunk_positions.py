"""Translate parser-local references when retained chunks are concatenated."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import replace

from polylogue.core.message_owner import MessageOwnerAmbiguityError, MessageOwnerCoordinate
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSessionEvent


class ChunkPositions:
    """One chunk's coordinate translation; prepared replay spills it to disk.

    Only coordinates are retained, not parsed messages. The optional connection
    belongs to the existing preparation scratch store, never an archive tier.
    """

    def __init__(
        self, messages: Sequence[ParsedMessage], offset: int, *, conn: sqlite3.Connection | None = None
    ) -> None:
        self.offset = offset
        self.count = len(messages)
        self._end = 0
        self._conn = conn
        self._positions: dict[tuple[int, int], int | None] = {}
        self._unqualified_positions: dict[int, int | None] = {}
        if conn is not None:
            conn.execute(
                "CREATE TEMP TABLE IF NOT EXISTS chunk_positions ("
                "position INTEGER NOT NULL, variant INTEGER NOT NULL, placed INTEGER, "
                "PRIMARY KEY (position, variant)) WITHOUT ROWID"
            )
            conn.execute("DELETE FROM temp.chunk_positions")
        for ordinal, message in enumerate(messages):
            position = message.position if message.position is not None else ordinal
            self._end = max(self._end, position + 1)
            keys = {(position, message.variant_index or 0)}
            if message.owner_coordinate is not None and message.owner_coordinate.physical_key is not None:
                keys.add(message.owner_coordinate.physical_key)
            for key in keys:
                placed = offset + ordinal
                if conn is None:
                    self._positions[key] = (
                        placed if key not in self._positions or self._positions[key] == placed else None
                    )
                    self._unqualified_positions[key[0]] = (
                        placed
                        if key[0] not in self._unqualified_positions or self._unqualified_positions[key[0]] == placed
                        else None
                    )
                else:
                    conn.execute(
                        "INSERT INTO temp.chunk_positions VALUES (?, ?, ?) "
                        "ON CONFLICT(position, variant) DO UPDATE SET placed = "
                        "CASE WHEN placed = excluded.placed THEN placed ELSE NULL END",
                        (*key, placed),
                    )

    def position(self, position: int | None, variant: int | None = None) -> int | None:
        if position is None:
            return None
        if self._conn is None:
            placed = (
                self._unqualified_positions.get(position)
                if variant is None
                else self._positions.get((position, variant))
            )
        elif variant is None:
            rows = self._conn.execute(
                "SELECT DISTINCT placed FROM temp.chunk_positions WHERE position = ? LIMIT 2",
                (position,),
            ).fetchall()
            placed = rows[0][0] if len(rows) == 1 else None
        else:
            row = self._conn.execute(
                "SELECT placed FROM temp.chunk_positions WHERE position = ? AND variant = ?",
                (position, variant),
            ).fetchone()
            placed = row[0] if row is not None else None
        if placed is None:
            raise MessageOwnerAmbiguityError(f"chunk coordinate ({position}, {variant}) has no unique message")
        return int(placed)

    def owner(self, owner: MessageOwnerCoordinate | None) -> MessageOwnerCoordinate | None:
        return None if owner is None else replace(owner, position=self.position(owner.position, owner.variant_index))

    def message(self, message: ParsedMessage, ordinal: int) -> ParsedMessage:
        owner = message.owner_coordinate
        if owner is not None and owner.position is not None:
            owner = replace(owner, position=self.offset + ordinal)
        return message.model_copy(
            update={
                "position": self.offset + ordinal,
                "parent_message_position": self.position(message.parent_message_position),
                "owner_coordinate": owner,
                # The concatenation chooses its own leaf; a chunk's
                # storage-default leaf marker does not survive into it.
                "is_active_leaf": False,
                "active_leaf_fallback": False,
            }
        )

    def attachment(self, attachment: ParsedAttachment) -> ParsedAttachment:
        position = self.position(attachment.message_position, attachment.message_variant_index or 0)
        owner = self.owner(attachment.owner_coordinate)
        if position == attachment.message_position and owner == attachment.owner_coordinate:
            return attachment
        moved = attachment.model_copy(update={"message_position": position, "owner_coordinate": owner})
        if attachment.prepared_carrier_key is None:
            # Acquisition can precede composition. Keep that exact lookup key
            # through a value copy without changing the source object.
            moved._acquisition_origin = (
                attachment._acquisition_origin if attachment._acquisition_origin is not None else attachment
            )
        return moved

    def event(self, event: ParsedSessionEvent) -> ParsedSessionEvent:
        def boundary(position: int | None) -> int | None:
            # Empty compactions address the interval just beyond the preceding
            # message (or 0..-1 before the first). It remains empty after rebasing.
            if position == self._end:
                return self.offset + self.count
            if position == -1:
                return self.offset - 1
            return self.position(position)

        return event.model_copy(
            update={
                "owner_coordinate": self.owner(event.owner_coordinate),
                "boundary_start_position": boundary(event.boundary_start_position),
                "boundary_end_position": boundary(event.boundary_end_position),
                # A next-message anchor at the end of this fragment has no
                # message. Concatenation must not attach it to another input.
                "boundary_message_position": (
                    None
                    if event.boundary_message_position == self._end
                    else self.position(event.boundary_message_position)
                ),
            }
        )
