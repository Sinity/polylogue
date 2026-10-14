"""Bounded representation of a sampled schema string.

Length and newline count are already drawn by the field sampler. This value
retains their neutral spelling recipe, so serialization never needs the whole
string or a list of newline positions.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

TEXT_PIECE_SIZE = 4096
TEXT_MATERIALIZATION_THRESHOLD = 1 << 20


def text_equals(value: str | SampledText | JoinedText, other: str) -> bool:
    """Compare a borrowed recipe without collecting its serialized text."""
    if len(value) != len(other):
        return False
    if isinstance(value, str):
        return value == other
    offset = 0
    for piece in value.pieces():
        if piece != other[offset : offset + len(piece)]:
            return False
        offset += len(piece)
    return offset == len(other)


@dataclass(frozen=True)
class SampledText:
    length: int
    pool: str
    offset: int = 0
    newlines: int = 0
    newline_phase: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.length, self.offset, self.newlines, self.newline_phase)):
            raise ValueError("sampled text coordinates must be exact integers")
        if type(self.pool) is not str or self.length < 0 or not self.pool or not 0 <= self.newlines <= self.length:
            raise ValueError("sampled text recipe cannot represent its declared shape")
        if "\n" in self.pool:
            raise ValueError("sampled text pool must leave newlines to its shape owner")

    def __len__(self) -> int:
        return self.length

    def pieces(self) -> Iterator[str]:
        for start in range(0, self.length, TEXT_PIECE_SIZE):
            stop = min(start + TEXT_PIECE_SIZE, self.length)
            characters: list[str] = []
            for position in range(start, stop):
                # An evenly spread neutral placement carries exactly the drawn
                # count. Inference does not claim a distribution of positions.
                slot = (position + self.newline_phase) % self.length
                line_break = (slot + 1) * self.newlines // self.length > slot * self.newlines // self.length
                characters.append("\n" if line_break else self.pool[(position + self.offset) % len(self.pool)])
            yield "".join(characters)


def sampled_text_value(recipe: SampledText) -> str | SampledText:
    """Change representation only; every sampled character is retained."""
    return "".join(recipe.pieces()) if recipe.length <= TEXT_MATERIALIZATION_THRESHOLD else recipe


@dataclass(frozen=True)
class JoinedText:
    parts: tuple[str | SampledText | JoinedText, ...]

    def __len__(self) -> int:
        return sum(len(part) for part in self.parts)

    def pieces(self) -> Iterator[str]:
        for part in self.parts:
            if isinstance(part, str):
                for start in range(0, len(part), TEXT_PIECE_SIZE):
                    yield part[start : start + TEXT_PIECE_SIZE]
            else:
                yield from part.pieces()


def joined_text_value(*parts: str | SampledText | JoinedText) -> str | JoinedText:
    value = JoinedText(parts)
    return "".join(value.pieces()) if len(value) <= TEXT_MATERIALIZATION_THRESHOLD else value
