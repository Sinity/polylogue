"""Deterministic, private-data-free scale outlier fixtures.

The builders in this module own wire evidence only.  Acquisition, parsing,
materialization, and replay remain the production services exercised by the
consuming tests.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

_TERMINAL_WIRE_BYTES: Final = 90_822_451
_REVISION_COUNT: Final = 804
_STREAM_EVENT_COUNT: Final = 2_000_000
_GIANT_ATTACHMENT_RAW_BYTES: Final = 12 * 1024 * 1024
_NEAR_TERMINAL_PREDECESSOR_BYTES: Final = 32 * 1024 * 1024
_ORDINARY_BLOB_LIMIT_BYTES: Final = 64 * 1024 * 1024
_WHALE_BLOB_LIMIT_BYTES: Final = 8 * 1024 * 1024 * 1024
_GENERATOR_CHUNK_BYTES: Final = 1024 * 1024


@dataclass(frozen=True, slots=True)
class WhaleFixtureDimensions:
    """Expected dimensions and the resource envelope for one fixture pack."""

    fixture_id: str = "codex-whale-bounds-v2"
    revision_count: int = _REVISION_COUNT
    terminal_wire_bytes: int = _TERMINAL_WIRE_BYTES
    near_terminal_predecessor_bytes: int = _NEAR_TERMINAL_PREDECESSOR_BYTES
    stream_event_count: int = _STREAM_EVENT_COUNT
    giant_attachment_raw_bytes: int = _GIANT_ATTACHMENT_RAW_BYTES
    ordinary_blob_limit_bytes: int = _ORDINARY_BLOB_LIMIT_BYTES
    whale_blob_limit_bytes: int = _WHALE_BLOB_LIMIT_BYTES

    def manifest_dimensions(self) -> tuple[tuple[str, int | str], ...]:
        return (
            ("fixture_id", self.fixture_id),
            ("revision_count", self.revision_count),
            ("terminal_wire_bytes", self.terminal_wire_bytes),
            ("near_terminal_predecessor_bytes", self.near_terminal_predecessor_bytes),
            ("stream_event_count", self.stream_event_count),
            ("giant_attachment_raw_bytes", self.giant_attachment_raw_bytes),
            ("ordinary_blob_limit_bytes", self.ordinary_blob_limit_bytes),
            ("whale_blob_limit_bytes", self.whale_blob_limit_bytes),
        )


WHALE_FIXTURE_DIMENSIONS = WhaleFixtureDimensions()


def _wire_target_bytes(revision: int, dimensions: WhaleFixtureDimensions) -> int:
    if revision < dimensions.revision_count - 4:
        return 4_096 + revision * 640
    return {
        dimensions.revision_count - 4: dimensions.near_terminal_predecessor_bytes // 4,
        dimensions.revision_count - 3: dimensions.near_terminal_predecessor_bytes // 2,
        dimensions.revision_count - 2: dimensions.near_terminal_predecessor_bytes,
        dimensions.revision_count - 1: dimensions.terminal_wire_bytes,
    }[revision]


def _write_record(handle: BinaryIO, record: dict[str, object]) -> int:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    return handle.write(encoded)


def _write_base64_pattern(handle: BinaryIO, byte_count: int) -> int:
    """Write deterministic base64 in bounded chunks, returning encoded bytes."""
    pattern = b"POLYLOGUE_PRIVATE_FREE_ATTACHMENT_0001\n"
    remaining = byte_count
    encoded_bytes = 0
    while remaining:
        chunk_size = min(3 * 1024 * 1024, remaining)
        chunk_size -= chunk_size % 3
        if chunk_size == 0:
            chunk_size = remaining
        raw_chunk = (pattern * ((chunk_size // len(pattern)) + 1))[:chunk_size]
        encoded = base64.b64encode(raw_chunk)
        handle.write(encoded)
        encoded_bytes += len(encoded)
        remaining -= chunk_size
    return encoded_bytes


def _write_repeated_byte(handle: BinaryIO, value: bytes, byte_count: int) -> None:
    """Write one repeated byte without allocating an outlier-sized buffer."""
    if len(value) != 1:
        raise ValueError("value must contain exactly one byte")
    chunk = value * min(byte_count, _GENERATOR_CHUNK_BYTES)
    remaining = byte_count
    while remaining:
        current = chunk if remaining >= len(chunk) else chunk[:remaining]
        written = handle.write(current)
        if written is None or written <= 0:
            raise OSError("fixture writer made no progress")
        remaining -= written


def _write_terminal_attachment_record(handle: BinaryIO, session_id: str, byte_count: int) -> None:
    record = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": f"{session_id}-terminal-803",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "sanitized terminal response revision 803"},
                {"type": "input_image", "image_url": ""},
            ],
        },
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    prefix, suffix = encoded.split(b'""', maxsplit=1)
    handle.write(prefix + b'"data:image/png;base64,')
    _write_base64_pattern(handle, byte_count)
    handle.write(b'"' + suffix + b"\n")


def _write_padding_record(handle: BinaryIO, revision: int, target_bytes: int, current_bytes: int) -> None:
    template = json.dumps(
        {"payload": {"padding": "", "type": "token_count"}, "revision": revision, "type": "response_item"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    prefix, suffix = template.split(b'""', maxsplit=1)
    padding_size = target_bytes - current_bytes - len(prefix) - len(suffix) - 3
    if padding_size <= 0:
        raise AssertionError(f"padding underflow at revision {revision}: {padding_size}")
    handle.write(prefix + b'"')
    _write_repeated_byte(handle, b"x", padding_size)
    handle.write(b'"' + suffix + b"\n")


@dataclass(frozen=True, slots=True)
class CodexRevisionChainFixture:
    """A single-path 804-revision Codex chain built one snapshot at a time."""

    dimensions: WhaleFixtureDimensions = WHALE_FIXTURE_DIMENSIONS
    session_native_id: str = "codex-sanitized-804-session"

    def write_revision(self, source_path: Path, revision: int) -> int:
        if not 0 <= revision < self.dimensions.revision_count:
            raise ValueError(f"revision must be in [0, {self.dimensions.revision_count})")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = source_path.with_name(f".{source_path.name}.revision-{revision:03d}.tmp")
        previous_size = source_path.stat().st_size if source_path.exists() else 0
        try:
            with temporary_path.open("wb") as handle:
                if revision:
                    with source_path.open("rb") as previous:
                        shutil.copyfileobj(previous, handle, length=_GENERATOR_CHUNK_BYTES)
                else:
                    _write_record(
                        handle,
                        {
                            "type": "session_meta",
                            "payload": {
                                "id": self.session_native_id,
                                "timestamp": "2026-07-31T04:25:20Z",
                                "cwd": "/sanitized/codex-804",
                            },
                        },
                    )
                    for message_index, role in enumerate(("user", "assistant")):
                        _write_record(
                            handle,
                            {
                                "type": "response_item",
                                "timestamp": "2026-07-31T04:25:20Z",
                                "payload": {
                                    "type": "message",
                                    "id": f"{self.session_native_id}-message-{message_index}",
                                    "role": role,
                                    "content": [
                                        {
                                            "type": "output_text" if role == "assistant" else "input_text",
                                            "text": f"sanitized incident witness {role} baseline",
                                        }
                                    ],
                                },
                            },
                        )
                if revision in {1, 800, 801, 802}:
                    _write_record(
                        handle,
                        {
                            "type": "response_item",
                            "timestamp": f"2026-07-31T04:25:{20 + revision % 40:02d}Z",
                            "payload": {
                                "type": "message",
                                "id": f"{self.session_native_id}-milestone-{revision:03d}",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": f"sanitized parsed milestone revision {revision}",
                                    }
                                ],
                            },
                        },
                    )
                if revision == self.dimensions.revision_count - 1:
                    _write_record(
                        handle,
                        {
                            "type": "compacted",
                            "payload": {
                                "message": "sanitized compaction summary at the whale boundary",
                                "replacement_history": [{"role": "user", "content": "sanitized prior context"}],
                            },
                        },
                    )
                    _write_terminal_attachment_record(
                        handle,
                        self.session_native_id,
                        self.dimensions.giant_attachment_raw_bytes,
                    )
                current_bytes = handle.tell()
                target_bytes = _wire_target_bytes(revision, self.dimensions)
                if target_bytes <= max(previous_size, current_bytes):
                    raise AssertionError(
                        f"revision {revision} is not strictly larger: previous={previous_size}, "
                        f"current={current_bytes}, target={target_bytes}"
                    )
                _write_padding_record(handle, revision, target_bytes, current_bytes)
                final_size = handle.tell()
            if final_size != target_bytes:
                raise AssertionError(f"revision {revision} has {final_size} bytes, expected {target_bytes}")
            os.replace(temporary_path, source_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return final_size

    def iter_revisions(self, source_path: Path) -> Iterator[tuple[int, int]]:
        """Write one revision, yielding it before the next replaces the file."""
        for revision in range(self.dimensions.revision_count):
            yield revision, self.write_revision(source_path, revision)

    def write_manifest(self, path: Path, sizes: tuple[int, ...]) -> Path:
        if len(sizes) != self.dimensions.revision_count:
            raise AssertionError("revision manifest does not cover every revision")
        payload = {
            "fixture_id": self.dimensions.fixture_id,
            "dimensions": dict(self.dimensions.manifest_dimensions()),
            "session_native_id": self.session_native_id,
            "revision_sizes": list(sizes),
            "terminal_features": ["compaction", "giant-base64-attachment", "codex-stream-dispatch"],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return path


def write_codex_whale_fixture_pack(
    output_dir: Path,
    fixture: CodexRevisionChainFixture = CodexRevisionChainFixture(),
) -> tuple[Path, Path]:
    """Generate the final wire snapshot and its complete revision manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "codex-whale.jsonl"
    sizes = tuple(size for _revision, size in fixture.iter_revisions(source_path))
    manifest_path = fixture.write_manifest(output_dir / "manifest.json", sizes)
    return source_path, manifest_path


def acquire_codex_revision_chain(
    archive_root: Path,
    fixture: CodexRevisionChainFixture,
    source_path: Path,
) -> tuple[str, ...]:
    """Acquire all snapshots through ``AcquisitionService`` with one live path."""
    from polylogue.config import Source
    from polylogue.pipeline.services.acquisition import AcquisitionService
    from polylogue.storage.sqlite import SQLiteBackend

    async def _run() -> tuple[str, ...]:
        backend = SQLiteBackend(db_path=archive_root / "index.db")
        try:
            service = AcquisitionService(backend)
            raw_ids: list[str] = []
            for _revision, _size in fixture.iter_revisions(source_path):
                result = await service.acquire_sources([Source(name="codex", path=source_path)])
                raw_ids.extend(result.raw_ids)
            return tuple(raw_ids)
        finally:
            await backend.close()

    return asyncio.run(_run())


def multi_million_codex_stream(
    dimensions: WhaleFixtureDimensions = WHALE_FIXTURE_DIMENSIONS,
) -> Iterator[dict[str, object]]:
    """Yield a million-scale Codex stream without allocating each state row."""
    yield {"type": "session_meta", "payload": {"id": "codex-stream-million", "timestamp": "2026-08-06T00:00:00Z"}}
    yield {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": "codex-stream-million-message",
            "role": "user",
            "content": [{"type": "input_text", "text": "sanitized streaming boundary"}],
        },
    }
    state_record = {"record_type": "state"}
    for _ in range(dimensions.stream_event_count):
        yield state_record


__all__ = [
    "CodexRevisionChainFixture",
    "WHALE_FIXTURE_DIMENSIONS",
    "WhaleFixtureDimensions",
    "acquire_codex_revision_chain",
    "multi_million_codex_stream",
    "write_codex_whale_fixture_pack",
]
