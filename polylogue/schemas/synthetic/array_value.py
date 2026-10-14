"""Generated array coordinates retained by the original relation owner."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, overload

if TYPE_CHECKING:
    from polylogue.schemas.synthetic.models import GeneratedValue


class GeneratedArrayOwner(Protocol):
    def array_length(self, array_id: int, generation: int) -> int: ...

    def array_item(self, array_id: int, generation: int, index: int) -> GeneratedValue: ...

    def borrow_array_item(self, array_id: int, generation: int, index: int) -> tuple[GeneratedValue, int]: ...

    def settle_array_item(
        self, array_id: int, generation: int, index: int, value: GeneratedValue, revision: int
    ) -> None: ...

    def append_array_item(self, array: GeneratedArray, value: GeneratedValue) -> None: ...

    def insert_array_item(self, array: GeneratedArray, index: int, value: GeneratedValue) -> None: ...

    def replace_array_item(self, array_id: int, generation: int, index: int, value: GeneratedValue) -> None: ...


@dataclass(frozen=True, eq=False)
class GeneratedArray(Sequence["GeneratedValue"]):
    """A view of exact sampled children, never a request to sample them again.

    Mutable child records are written back when their borrowed iteration step
    settles, so the canonical wire adapter's changes survive serialization.
    The original owner checks currency on every read and write.
    """

    owner: GeneratedArrayOwner
    array_id: int
    generation: int

    def __len__(self) -> int:
        return self.owner.array_length(self.array_id, self.generation)

    @overload
    def __getitem__(self, index: int) -> GeneratedValue: ...

    @overload
    def __getitem__(self, index: slice) -> list[GeneratedValue]: ...

    def __getitem__(self, index: int | slice) -> GeneratedValue | list[GeneratedValue]:
        length = len(self)
        if isinstance(index, slice):
            return [
                self.owner.array_item(self.array_id, self.generation, item) for item in range(*index.indices(length))
            ]
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError(index)
        return self.owner.array_item(self.array_id, self.generation, index)

    def __iter__(self) -> Generator[GeneratedValue, None, None]:
        for index in range(len(self)):
            value, revision = self.owner.borrow_array_item(self.array_id, self.generation, index)
            try:
                yield value
            finally:
                self.owner.settle_array_item(self.array_id, self.generation, index, value, revision)

    def __setitem__(self, index: int, value: GeneratedValue) -> None:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        self.owner.replace_array_item(self.array_id, self.generation, index, value)

    def append(self, value: GeneratedValue) -> None:
        self.owner.append_array_item(self, value)

    def extend(self, values: Iterable[GeneratedValue]) -> None:
        for value in values:
            self.append(value)

    def insert(self, index: int, value: GeneratedValue) -> None:
        self.owner.insert_array_item(self, index, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence) or isinstance(other, (str, bytes, bytearray)):
            return False
        return len(self) == len(other) and all(left == right for left, right in zip(self, other, strict=True))


def walk_generated_values(value: object) -> Generator[object, None, None]:
    """Visit sampled children with only original active ancestors retained.

    Native array identity is its owner/coordinate, since borrowing a child
    creates a fresh Python view. Completed Python IDs cannot prove custody
    for those views and would grow with the output size.
    """
    from polylogue.schemas.synthetic.models import GeneratedRecord

    active: set[tuple[int, ...]] = set()
    frames: list[tuple[tuple[int, ...], Iterator[object]]] = []
    current = value
    try:
        while True:
            identity: tuple[int, ...] | None = None
            children: Iterator[object] | None = None
            if isinstance(current, GeneratedArray):
                identity = (id(current.owner), 0, current.array_id, current.generation)
                children = iter(current)
            elif isinstance(current, GeneratedRecord):
                identity = (id(current.owner), 1, current.record_id, current.generation)
                children = iter(current.values())
            elif isinstance(current, Mapping):
                identity = (id(current),)
                children = iter(current.values())
            elif isinstance(current, Sequence) and not isinstance(current, str | bytes | bytearray):
                identity = (id(current),)
                children = iter(current)
            if identity is not None:
                if identity in active:
                    raise ValueError("cyclic objects have no finite JSON representation")
                active.add(identity)
                assert children is not None
                frames.append((identity, children))
            yield current
            while frames:
                identity, children = frames[-1]
                try:
                    current = next(children)
                    break
                except StopIteration:
                    frames.pop()
                    active.remove(identity)
            else:
                return
    finally:
        for _, children in reversed(frames):
            close = getattr(children, "close", None)
            if close is not None:
                close()
