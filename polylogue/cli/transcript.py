"""Private transcript records owned by one CLI read and delivery lifetime."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import TemporaryFile
from typing import TextIO

from polylogue.surfaces.outcome import OutcomeEnvelope


class TranscriptRecords:
    """Replay one record at a time from the caller's private descriptor."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.count = 0

    def append(self, record: Mapping[str, object]) -> None:
        if self.stream.closed:
            raise RuntimeError("transcript owner is closed")
        self.stream.seek(0, 2)
        for chunk in json.JSONEncoder().iterencode(record):
            self.stream.write(chunk)
        self.stream.write("\n")
        self.count += 1

    def records(self) -> Iterator[dict[str, object]]:
        if self.stream.closed:
            raise RuntimeError("transcript owner is closed")
        self.stream.flush()
        position = 0
        while True:
            self.stream.seek(position)
            line = self.stream.readline()
            if not line:
                return
            position = self.stream.tell()
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("private transcript record is not an object")
            yield value


@contextmanager
def transcript_records() -> Iterator[TranscriptRecords]:
    with TemporaryFile(mode="w+t", encoding="utf-8") as stream:
        yield TranscriptRecords(stream)


@dataclass(frozen=True, slots=True)
class TranscriptRead:
    header: dict[str, object]
    records: TranscriptRecords
    outcome: OutcomeEnvelope


class MetadataWalk:
    """Verify and retain the independently paged transcript metadata relation."""

    def __init__(self, tags: TranscriptRecords) -> None:
        self.tags = tags
        self.offset = 0
        self.page_limit = 100
        self.continuation: str | None = None
        self.complete = False
        self.directories_total: int | None = None
        self.tags_total: int | None = None
        self.first_directory: str | None = None
        self.last_tag: str | None = None

    def accept(self, page: object, directories: tuple[str, ...], tags: tuple[str, ...]) -> None:
        from polylogue.cli.operation_kernel import OperationEnvelopeError

        if not isinstance(page, Mapping):
            raise OperationEnvelopeError("session.read omitted its metadata page")
        directory_total = page.get("working_directories_total")
        tag_total = page.get("user_tags_total")
        limit = page.get("limit")
        offset = page.get("offset")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (directory_total, tag_total, limit, offset)
        ):
            raise OperationEnvelopeError("session.read returned invalid metadata coordinates")
        assert isinstance(directory_total, int) and isinstance(tag_total, int) and isinstance(limit, int)
        if limit > self.page_limit:
            raise OperationEnvelopeError("session.read widened its metadata page")
        self.page_limit = limit
        if self.directories_total is not None and (
            directory_total != self.directories_total or tag_total != self.tags_total
        ):
            raise OperationEnvelopeError("session.read changed its metadata counts")
        if offset != self.offset:
            raise OperationEnvelopeError("session.read changed its metadata offset")
        if len(directories) != min(limit, max(0, directory_total - self.offset)) or len(tags) != min(
            limit, max(0, tag_total - self.offset)
        ):
            raise OperationEnvelopeError("session.read omitted metadata rows")
        for directory in directories:
            if self.first_directory is None and directory.strip():
                self.first_directory = directory
        for tag in tags:
            if self.last_tag is not None and tag < self.last_tag:
                raise OperationEnvelopeError("session.read changed its User tag order")
            self.tags.append({"tag": tag})
            self.last_tag = tag
        self.directories_total, self.tags_total = directory_total, tag_total
        frontier = max(directory_total, tag_total)
        next_offset = page.get("next_offset")
        if next_offset is None:
            if page.get("complete") is not True or self.offset + limit < frontier or self.tags.count != tag_total:
                raise OperationEnvelopeError("session.read did not complete its metadata relation")
            self.complete = True
            return
        expected = min(self.offset + limit, frontier)
        token = page.get("continuation")
        if (
            type(next_offset) is not int
            or next_offset != expected
            or next_offset <= self.offset
            or next_offset >= frontier
            or page.get("complete") is not False
            or not isinstance(token, str)
            or not token
            or token == self.continuation
        ):
            raise OperationEnvelopeError("session.read returned non-progressing metadata")
        self.offset, self.continuation = next_offset, token

    def values(self, fallback: Iterable[str]) -> Iterator[str]:
        if not self.complete:
            raise RuntimeError("metadata relation is incomplete")
        if not self.tags.count:
            yield from fallback
            return
        for record in self.tags.records():
            tag = record.get("tag")
            if not isinstance(tag, str):
                raise TypeError("private metadata tag is not text")
            yield tag
