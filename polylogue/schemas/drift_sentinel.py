"""Format-drift vocabulary and observation payload for retained validation.

The retained validator classifies complete records in precedence order:
validation failure, unresolved shape, new field, then schema-known unread
field. It emits deterministic field signatures in ``SchemaDriftObservation``;
the ops sampler and status readers share this vocabulary. New fields are
benign, while the other classifications are risky. No drift yields no
observation, and telemetry never chooses source identity or gates publication.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias

DriftClassification: TypeAlias = Literal["unseen_shape", "new_field", "field_changed", "known_field_unread"]

UNSEEN_SHAPE: DriftClassification = "unseen_shape"
NEW_FIELD: DriftClassification = "new_field"
FIELD_CHANGED: DriftClassification = "field_changed"
KNOWN_FIELD_UNREAD: DriftClassification = "known_field_unread"

# Risky classifications outrank benign ones for alerting purposes -- a
# rate dominated by field_changed/unseen_shape/known_field_unread should
# never read as "fine" just because new_field volume swamps it.
# known_field_unread is risky: it is not "the schema doesn't know about
# this yet" (the other three classifications' shared axis) but "the schema
# knows, and the parser silently drops it anyway" -- a defect, not drift.
RISKY_CLASSIFICATIONS: frozenset[DriftClassification] = frozenset({FIELD_CHANGED, UNSEEN_SHAPE, KNOWN_FIELD_UNREAD})
BENIGN_CLASSIFICATIONS: frozenset[DriftClassification] = frozenset({NEW_FIELD})


@dataclass(frozen=True, slots=True)
class DriftSignature:
    """Exact replayable UTF-8 signature, inline when small and file-backed otherwise.

    File-backed instances borrow the caller's preparation directory. The
    directory owner removes their private file after all consumers settle.
    """

    byte_count: int
    _inline: bytes | None
    _path: Path | None

    def __post_init__(self) -> None:
        if self.byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        if (self._inline is None) == (self._path is None):
            raise ValueError("a drift signature must have exactly one backing store")
        if self._inline is not None and len(self._inline) != self.byte_count:
            raise ValueError("inline drift signature byte count does not match its content")

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        directory: Path,
        inline_limit_bytes: int = 4096,
    ) -> DriftSignature:
        return cls.from_utf8_chunks(
            (text.encode("utf-8", errors="surrogatepass"),),
            directory=directory,
            inline_limit_bytes=inline_limit_bytes,
        )

    @classmethod
    def from_utf8_chunks(
        cls,
        chunks: Iterable[bytes],
        *,
        directory: Path,
        inline_limit_bytes: int = 4096,
    ) -> DriftSignature:
        if inline_limit_bytes < 0:
            raise ValueError("inline_limit_bytes must be non-negative")
        buffered = bytearray()
        byte_count = 0
        path: Path | None = None
        output: BinaryIO | None = None
        try:
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError("drift signature chunks must be bytes")
                if not chunk:
                    continue
                byte_count += len(chunk)
                if output is None and len(buffered) + len(chunk) <= inline_limit_bytes:
                    buffered.extend(chunk)
                    continue
                if output is None:
                    directory.mkdir(parents=True, exist_ok=True)
                    descriptor, filename = tempfile.mkstemp(prefix=".polylogue-drift-signature-", dir=directory)
                    path = Path(filename)
                    try:
                        output = os.fdopen(descriptor, "wb")
                    except BaseException:
                        os.close(descriptor)
                        raise
                    if buffered:
                        output.write(buffered)
                        buffered.clear()
                output.write(chunk)
            if output is not None:
                output.close()
                output = None
                assert path is not None
                return cls(byte_count=byte_count, _inline=None, _path=path)
            return cls(byte_count=byte_count, _inline=bytes(buffered), _path=None)
        except BaseException:
            if output is not None:
                with contextlib.suppress(OSError):
                    output.close()
            if path is not None:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
            raise

    def iter_utf8_chunks(self, *, chunk_bytes: int = 64 * 1024) -> Generator[bytes, None, None]:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if self._inline is not None:
            for offset in range(0, len(self._inline), chunk_bytes):
                yield self._inline[offset : offset + chunk_bytes]
            return
        if self._path is None:
            raise RuntimeError("drift signature has no backing storage")
        seen = 0
        with self._path.open("rb") as source:
            while chunk := source.read(chunk_bytes):
                seen += len(chunk)
                yield chunk
        if seen != self.byte_count:
            raise ValueError("file-backed drift signature byte count does not match its content")

    def compare(self, other: DriftSignature) -> int:
        """Compare exact encoded bytes without materializing either signature."""
        left = _ByteCursor(self.iter_utf8_chunks())
        right = _ByteCursor(other.iter_utf8_chunks())
        try:
            while True:
                left_bytes = left.peek()
                right_bytes = right.peek()
                if left_bytes is None or right_bytes is None:
                    return (left_bytes is not None) - (right_bytes is not None)
                shared = min(len(left_bytes), len(right_bytes))
                if left_bytes[:shared] != right_bytes[:shared]:
                    return -1 if left_bytes[:shared] < right_bytes[:shared] else 1
                left.consume(shared)
                right.consume(shared)
        finally:
            left.close()
            right.close()


class _ByteCursor:
    def __init__(self, chunks: Generator[bytes, None, None]) -> None:
        self._chunks = chunks
        self._current = b""
        self._offset = 0

    def peek(self) -> bytes | None:
        while self._offset == len(self._current):
            try:
                self._current = next(self._chunks)
            except StopIteration:
                return None
            self._offset = 0
        return self._current[self._offset :]

    def consume(self, count: int) -> None:
        self._offset += count

    def close(self) -> None:
        self._chunks.close()


@dataclass(frozen=True, slots=True)
class SchemaDriftObservation:
    """One classified drift signal for a single ingested raw record."""

    origin: str
    element_kind: str
    classification: DriftClassification
    unseen_key_signature: DriftSignature
    native_id_example: str
    raw_id: str


def is_risky(classification: DriftClassification) -> bool:
    return classification in RISKY_CLASSIFICATIONS


__all__ = [
    "BENIGN_CLASSIFICATIONS",
    "FIELD_CHANGED",
    "KNOWN_FIELD_UNREAD",
    "NEW_FIELD",
    "RISKY_CLASSIFICATIONS",
    "UNSEEN_SHAPE",
    "DriftClassification",
    "DriftSignature",
    "SchemaDriftObservation",
    "is_risky",
]
