from __future__ import annotations

import base64
import binascii
import builtins
import codecs
import hashlib
import hmac
import json
import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from .page import CompactJSONPage


class SessionError(ValueError):
    pass


_SCAN_BLOCK_BYTES = 64 * 1_024
DEFAULT_SCAN_BYTES = 8 * 1_024 * 1_024
MAX_CURSOR_BYTES = 8_192


class OpaqueSessionCursor:
    """Small signed cursors for resumable local-session scans.

    The state is deliberately only positions and already-observed metadata;
    transcript data stays in the source file.  Each use supplies its complete
    scope, so a token cannot be replayed for another principal, query or
    source observation.
    """

    def __init__(self, principal: str, cursor_key: bytes, purpose: str):
        self._key = hmac.new(cursor_key, f"{purpose}:{principal}".encode(), hashlib.sha256).digest()

    @staticmethod
    def _canonical(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    def encode(self, scope: dict[str, Any], state: dict[str, Any]) -> str:
        body = {"v": 1, "scope": scope, "state": state}
        payload = base64.urlsafe_b64encode(self._canonical(body)).decode().rstrip("=")
        mac = hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()
        value = f"{payload}.{mac}"
        if len(value.encode()) > MAX_CURSOR_BYTES:
            raise SessionError("session continuation cursor exceeds its size bound")
        return value

    def decode(self, value: Any, scope: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, str) or len(value.encode()) > MAX_CURSOR_BYTES or "." not in value:
            raise SessionError("session continuation cursor is malformed")
        payload, mac = value.rsplit(".", 1)
        expected = hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, expected):
            raise SessionError("session continuation cursor is stale")
        try:
            padded = payload + "=" * (-len(payload) % 4)
            body = json.loads(base64.urlsafe_b64decode(padded).decode())
        except (
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            binascii.Error,
        ) as exc:
            raise SessionError("session continuation cursor is malformed") from exc
        if not isinstance(body, dict) or body.get("v") != 1:
            raise SessionError("session continuation cursor is stale")
        actual_scope = body.get("scope")
        if actual_scope != scope:
            if (
                isinstance(actual_scope, dict)
                and actual_scope.get("source_revision") != scope.get("source_revision")
                and {key: value for key, value in actual_scope.items() if key != "source_revision"}
                == {key: value for key, value in scope.items() if key != "source_revision"}
            ):
                raise SessionError("session source changed after continuation began")
            raise SessionError("session continuation cursor is stale")
        state = body.get("state")
        if not isinstance(state, dict):
            raise SessionError("session continuation cursor is malformed")
        return state


@dataclass(frozen=True)
class SessionSource:
    provider: str
    root: Path


class SessionLogService:
    @staticmethod
    def default_sources(home: Path | None = None) -> tuple[SessionSource, ...]:
        home = Path.home() if home is None else home
        return (
            SessionSource("claude-code", home / ".claude" / "projects"),
            SessionSource("codex", home / ".codex" / "sessions"),
        )

    def __init__(
        self,
        *,
        max_result_bytes: int = 256_000,
        scope: str = "polylogue-raw",
        sources: tuple[SessionSource, ...] | None = None,
    ):
        self.max_result_bytes = max_result_bytes
        self.scope = scope
        configured_sources = sources or self.default_sources()
        self.sources = tuple(SessionSource(source.provider, source.root.resolve()) for source in configured_sources)

    def _source(self, provider: str) -> SessionSource:
        for source in self.sources:
            if source.provider == provider:
                return source
        raise SessionError("provider must be claude-code or codex")

    @staticmethod
    def _reference(source: SessionSource, path: Path) -> str:
        return f"{source.provider}:{path.relative_to(source.root)}"

    def _path_from_reference(self, reference: str) -> tuple[SessionSource, Path]:
        provider, separator, relative = reference.partition(":")
        if not separator or not relative:
            raise SessionError("reference must use provider:relative-path form")
        source = self._source(provider)
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SessionError("reference must remain within its provider root")
        try:
            path = (source.root / candidate).resolve(strict=True)
            root = source.root.resolve(strict=True)
        except FileNotFoundError as exc:
            raise SessionError("session source is unavailable") from exc
        if root not in path.parents or path.suffix != ".jsonl" or not path.is_file():
            raise SessionError("reference does not identify a session JSONL file")
        return source, path

    @staticmethod
    def _files(source: SessionSource) -> builtins.list[tuple[Path, os.stat_result]]:
        if not source.root.is_dir():
            raise SessionError("session source directory is unavailable")
        files: builtins.list[tuple[Path, os.stat_result]] = []

        def unavailable(exc: OSError) -> NoReturn:
            raise SessionError("session source directory is unavailable") from exc

        for directory, _, names in os.walk(source.root, onerror=unavailable):
            for name in names:
                if not name.endswith(".jsonl"):
                    continue
                path = Path(directory) / name
                try:
                    info = path.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    unavailable(exc)
                if stat.S_ISREG(info.st_mode):
                    files.append((path, info))
        files.sort(key=lambda row: (-row[1].st_mtime_ns, str(row[0])))
        return files

    def inventory(self, provider: str) -> builtins.list[dict[str, Any]]:
        """Observe metadata once, newest first; transcript bytes remain live."""
        source = self._source(provider)
        return [
            {
                "reference": self._reference(source, path),
                "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            }
            for path, info in self._files(source)
        ]

    def list(self, provider: str, limit: int = 100) -> dict[str, Any]:
        if limit < 1:
            raise SessionError("limit must be positive")
        rows = self.inventory(provider)
        return {
            "provider": provider,
            "sessions": rows[:limit],
            "truncated": len(rows) > limit,
        }

    def read(self, reference: str, offset: int = 0, max_bytes: int = 64_000) -> dict[str, Any]:
        source, path = self._path_from_reference(reference)
        if offset < 0:
            raise SessionError("offset must not be negative")
        if max_bytes < 1:
            raise SessionError("max_bytes must be positive")
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        data = data[:max_bytes]
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        content = decoder.decode(data, final=not truncated)
        consumed = len(data) - len(decoder.getstate()[0])
        if truncated and consumed == 0:
            raise SessionError("max_bytes is too small to decode the next UTF-8 sequence; retry with at least 4")
        return {
            "provider": source.provider,
            "reference": self._reference(source, path),
            "offset": offset,
            "bytes": consumed,
            "next_offset": offset + consumed if truncated else None,
            "truncated": truncated,
            "content": content,
        }

    @staticmethod
    def _source_revision(source: SessionSource, files: Sequence[tuple[Path, os.stat_result]]) -> str:
        """Identity of the exact searchable observation, not its contents."""
        rows = [
            (
                path.relative_to(source.root).as_posix(),
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
            )
            for path, info in files
        ]
        return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _scan_bytes(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SessionError("scan_bytes must be a positive integer")
        # This is a per-request work budget, not a source coverage ceiling.
        # The caller receives a continuation when it is exhausted.
        return value

    @staticmethod
    def _advance_line(state: dict[str, int], data: bytes) -> None:
        state["offset"] += len(data)
        last_newline = data.rfind(b"\n")
        if last_newline >= 0:
            state["line"] += data.count(b"\n")
            state["line_start"] = state["offset"] - len(data) + last_newline + 1

    @staticmethod
    def _line_for_match(state: dict[str, int], combined: bytes, match_index: int, replay: int) -> tuple[int, int]:
        """Return line number and byte start without retaining an entire line."""
        before = combined[:match_index]
        line = state["line"] - combined[:replay].count(b"\n") + before.count(b"\n")
        newline = before.rfind(b"\n")
        if newline >= 0:
            return line, state["offset"] - replay + newline + 1
        return line, state["line_start"]

    @staticmethod
    def _snippet(path: Path, line_start: int, match_offset: int, query_bytes: int) -> tuple[int, str]:
        start = max(line_start, match_offset - 200)
        # Keep the query intact even when the caller supplied a long literal.
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(max(2_000, match_offset - start + query_bytes))
        return start, data.decode("utf-8", errors="replace").rstrip("\r\n")[:2_000]

    def _scan_literal(
        self,
        source: SessionSource,
        files: Sequence[tuple[Path, os.stat_result]],
        query: str,
        limit: int,
        scan_bytes: int,
        cursor: str | None,
        cursor_key: bytes | None,
        purpose: str,
        one_per_file: bool,
        *,
        source_revision: str | None = None,
    ) -> dict[str, Any]:
        query_bytes = query.encode("utf-8")
        revision = source_revision if source_revision is not None else self._source_revision(source, files)
        scope = {
            "principal": self.scope,
            "provider": source.provider,
            "query_sha256": hashlib.sha256(query_bytes).hexdigest(),
            "source_revision": revision,
        }
        if cursor is not None:
            if cursor_key is None:
                raise SessionError("session continuation cursor is unavailable")
            state = OpaqueSessionCursor(self.scope, cursor_key, purpose).decode(cursor, scope)
        else:
            state = {"file": 0, "offset": 0, "line": 1, "line_start": 0}
        if (
            set(state) != {"file", "offset", "line", "line_start"}
            or any(isinstance(state[key], bool) or not isinstance(state[key], int) or state[key] < 0 for key in state)
            or state["file"] > len(files)
        ):
            raise SessionError("session continuation cursor is malformed")

        scanned = 0
        rows: builtins.list[dict[str, Any]] = []
        resume_after_last: dict[str, int] | None = None
        while state["file"] < len(files) and scanned < scan_bytes:
            path, info = files[state["file"]]
            try:
                current = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise SessionError("session source changed during search") from exc
            if (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            ) != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
                raise SessionError("session source changed during search")
            if state["offset"] >= info.st_size:
                state = {
                    "file": state["file"] + 1,
                    "offset": 0,
                    "line": 1,
                    "line_start": 0,
                }
                continue
            remaining = min(
                _SCAN_BLOCK_BYTES,
                scan_bytes - scanned,
                info.st_size - state["offset"],
            )
            with path.open("rb") as handle:
                handle.seek(state["offset"])
                data = handle.read(remaining)
                handle.seek(max(0, state["offset"] - max(0, len(query_bytes) - 1)))
                tail = handle.read(state["offset"] - handle.tell())
            if not data:
                # A concurrent shrink invalidates this otherwise immutable observation.
                raise SessionError("session source changed during search")
            scanned += len(data)
            combined = tail + data
            combined_start = state["offset"] - len(tail)
            index = 0
            found = False
            while True:
                index = combined.find(query_bytes, index)
                if index < 0:
                    break
                absolute = combined_start + index
                end = absolute + len(query_bytes)
                # Matches wholly in the replay tail were already returned.
                if end <= state["offset"]:
                    index += len(query_bytes)
                    continue
                line, line_start = self._line_for_match(state, combined, index, len(tail))
                offset, text = self._snippet(path, line_start, absolute, len(query_bytes))
                text = text[: min(2_000, max(128, self.max_result_bytes // 4))]
                row = {
                    "reference": self._reference(source, path),
                    "line": line,
                    "offset": offset,
                    "text": text,
                    "source_observation": (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),
                }
                if len(rows) >= limit:
                    # Do not consume this match: the continuation replay window
                    # makes it the first candidate on the next page.
                    # A file overview has already consumed the accepted file.
                    # A match stream instead resumes after its last match.
                    next_state = dict(state if one_per_file else (resume_after_last or state))
                    if cursor_key is None:
                        return {
                            "rows": rows,
                            "scanned_bytes": scanned,
                            "truncated": True,
                            "next_cursor": None,
                        }
                    return {
                        "rows": rows,
                        "scanned_bytes": scanned,
                        "truncated": True,
                        "next_cursor": OpaqueSessionCursor(self.scope, cursor_key, purpose).encode(scope, next_state),
                    }
                rows.append(row)
                found = True
                resume_after_last = dict(state)
                self._advance_line(resume_after_last, data[: max(0, end - state["offset"])])
                if one_per_file:
                    state = {
                        "file": state["file"] + 1,
                        "offset": 0,
                        "line": 1,
                        "line_start": 0,
                    }
                    break
                index += len(query_bytes)
            if found and one_per_file:
                continue
            self._advance_line(state, data)
            if state["offset"] >= info.st_size:
                state = {
                    "file": state["file"] + 1,
                    "offset": 0,
                    "line": 1,
                    "line_start": 0,
                }
        truncated = state["file"] < len(files)
        next_cursor = None
        if truncated and cursor_key is not None:
            next_cursor = OpaqueSessionCursor(self.scope, cursor_key, purpose).encode(scope, state)
        return {
            "rows": rows,
            "scanned_bytes": scanned,
            "truncated": truncated,
            "next_cursor": next_cursor,
        }

    def search(
        self,
        provider: str,
        query: str,
        max_results: Any = 100,
        *,
        cursor: str | None = None,
        cursor_key: bytes | None = None,
        scan_bytes: int = DEFAULT_SCAN_BYTES,
        reference: str | None = None,
    ) -> dict[str, Any]:
        source = self._source(provider)
        if not query or len(query) > 1_000:
            raise SessionError("query must contain 1-1000 characters")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise SessionError("max_results must be a positive integer")
        if reference is not None:
            selected_source, path = self._path_from_reference(reference)
            if selected_source.provider != provider:
                raise SessionError("reference provider must match search provider")
            info = path.stat(follow_symlinks=False)
            files = [(path, info)]
        else:
            files = self._files(source)
        result = self._scan_literal(
            source,
            files,
            query,
            max_results,
            self._scan_bytes(scan_bytes),
            cursor,
            cursor_key,
            "session-search",
            False,
        )
        return {
            "provider": provider,
            "matches": result["rows"],
            "scanned_bytes": result["scanned_bytes"],
            "truncated": result["truncated"],
            "next_cursor": result["next_cursor"],
        }

    def timeline(
        self,
        provider: str,
        start_ns: int | None,
        end_ns: int | None,
        query: str | None,
        max_results: Any,
        *,
        cursor: str | None = None,
        cursor_key: bytes | None = None,
        scan_bytes: int = DEFAULT_SCAN_BYTES,
    ) -> dict[str, Any]:
        if max_results < 1:
            raise SessionError("max_results must be a positive integer")
        return self.observe_timeline(provider, start_ns, end_ns, query).page(
            max_results, cursor=cursor, cursor_key=cursor_key, scan_bytes=scan_bytes
        )

    def observe_timeline(
        self,
        provider: str,
        start_ns: int | None,
        end_ns: int | None,
        query: str | None,
    ) -> _ObservedTimeline:
        """Bind one request's metadata enumeration; do not freeze live bytes."""
        source = self._source(provider)
        if start_ns is not None and start_ns < 0:
            raise SessionError("start time must not precede the Unix epoch")
        if end_ns is not None and end_ns < 0:
            raise SessionError("end time must not precede the Unix epoch")
        if start_ns is not None and end_ns is not None and start_ns > end_ns:
            raise SessionError("start time must not be after end time")
        if query is not None and (not query or len(query) > 1_000):
            raise SessionError("query must contain 1-1000 characters")
        files = tuple(
            (path, info)
            for path, info in self._files(source)
            if (start_ns is None or info.st_mtime_ns >= start_ns) and (end_ns is None or info.st_mtime_ns <= end_ns)
        )
        return _ObservedTimeline(self, source, files, query)


class _ObservedTimeline:
    """Request-owned metadata view, not a persistent cache or filesystem snapshot."""

    def __init__(
        self,
        service: SessionLogService,
        source: SessionSource,
        files: tuple[tuple[Path, os.stat_result], ...],
        query: str | None,
    ):
        self.service = service
        self.source = source
        self.files = files
        self.query = query
        self.revision = service._source_revision(source, files)
        self.by_reference = (
            {service._reference(source, path): info for path, info in files} if query is not None else {}
        )

    def page(
        self,
        max_results: int,
        *,
        cursor: str | None = None,
        cursor_key: bytes | None = None,
        scan_bytes: int = DEFAULT_SCAN_BYTES,
    ) -> dict[str, Any]:
        if max_results < 1:
            raise SessionError("max_results must be a positive integer")
        source, files, query = self.source, self.files, self.query
        if query is not None:
            result = self.service._scan_literal(
                source,
                files,
                query,
                max_results,
                self.service._scan_bytes(scan_bytes),
                cursor,
                cursor_key,
                "session-timeline",
                True,
                source_revision=self.revision,
            )
            entries = []
            for row in result["rows"]:
                info = self.by_reference[row["reference"]]
                entries.append(
                    {
                        "reference": row["reference"],
                        "bytes": info.st_size,
                        "mtime_ns": info.st_mtime_ns,
                        "source_observation": (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),
                        # Timeline is an overview.  The byte offset is still
                        # available from sessions.query for the full context.
                        "snippet": row["text"][:512],
                    }
                )
            return {
                "provider": source.provider,
                "entries": entries,
                "scanned_bytes": result["scanned_bytes"],
                "truncated": result["truncated"],
                "next_cursor": result["next_cursor"],
            }

        revision = self.revision
        scope = {
            "principal": self.service.scope,
            "provider": source.provider,
            "query_sha256": None,
            "source_revision": revision,
        }
        if cursor is not None:
            if cursor_key is None:
                raise SessionError("session continuation cursor is unavailable")
            state = OpaqueSessionCursor(self.service.scope, cursor_key, "session-timeline").decode(cursor, scope)
            if set(state) != {"file"} or not isinstance(state["file"], int) or state["file"] < 0:
                raise SessionError("session continuation cursor is malformed")
        else:
            state = {"file": 0}
        page = CompactJSONPage(max(512, self.service.max_result_bytes - 16_384))
        entries = page.items
        while state["file"] < len(files):
            path, info = files[state["file"]]
            entry = {
                "reference": self.service._reference(source, path),
                "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "source_observation": (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),
            }
            if len(entries) >= max_results or not page.try_append(entry):
                break
            state["file"] += 1
        truncated = state["file"] < len(files)
        return {
            "provider": source.provider,
            "entries": entries,
            "scanned_bytes": 0,
            "truncated": truncated,
            "next_cursor": OpaqueSessionCursor(self.service.scope, cursor_key, "session-timeline").encode(scope, state)
            if truncated and cursor_key is not None
            else None,
        }
