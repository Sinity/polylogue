"""Repeatable HTML rows with bodies held by one private render descriptor."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from typing import TextIO, overload

from polylogue.rendering.core_messages import RenderedMessage, branch_group_positions


class HTMLRecords:
    def __init__(self, stream: TextIO, messages: Iterator[RenderedMessage]) -> None:
        self.stream = stream
        self.positions: list[int] = []
        metadata: list[tuple[str | None, int, bool | None]] = []
        for message in messages:
            self.positions.append(stream.tell())
            for chunk in json.JSONEncoder().iterencode(asdict(message)):
                stream.write(chunk)
            stream.write("\n")
            metadata.append((message.parent_message_id, message.branch_index, message.is_active_path))
        stream.flush()
        self.mainline, self.branches = branch_group_positions(metadata)

    def row(self, position: int) -> dict[str, object]:
        self.stream.seek(self.positions[position])
        row = json.loads(self.stream.readline())
        if not isinstance(row, dict):
            raise TypeError("private HTML row is not an object")
        return row

    def __iter__(self) -> Iterator[dict[str, object]]:
        for position in self.mainline:
            row = self.row(position)
            row["branches"] = HTMLBranches(self, self.branches.get(position, []))
            yield row

    def __len__(self) -> int:
        return len(self.mainline)


class HTMLBranches(Sequence[dict[str, object]]):
    def __init__(self, owner: HTMLRecords, positions: Sequence[int]) -> None:
        self.owner = owner
        self.positions = positions

    def __len__(self) -> int:
        return len(self.positions)

    @overload
    def __getitem__(self, index: int) -> dict[str, object]: ...
    @overload
    def __getitem__(self, index: slice) -> HTMLBranches: ...

    def __getitem__(self, index: int | slice) -> dict[str, object] | HTMLBranches:
        if isinstance(index, slice):
            return HTMLBranches(self.owner, self.positions[index])
        return self.owner.row(self.positions[index])
