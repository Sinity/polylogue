from __future__ import annotations

import base64
import binascii
import builtins
import codecs
import errno
import hashlib
import hmac
import json
import os
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, NoReturn

from polylogue.paths import state_home

from .page import CompactJSONPage
from .snapshot_store import (
    SnapshotBinding,
    SnapshotStore,
    SnapshotTemporarilyUnavailableError,
    SnapshotUnavailableError,
)


class SessionError(ValueError):
    code = "session_read_failed"


class RetryableSessionError(SessionError):
    """A transient system failure; the same request or continuation can be retried."""

    code = "retryable"


class StaleContinuationError(SessionError):
    """A continuation that cannot resume its original scope; restart the search."""

    code = "stale_continuation"


_SCAN_BLOCK_BYTES = 64 * 1_024
DEFAULT_SCAN_BYTES = 8 * 1_024 * 1_024
MAX_CURSOR_BYTES = 8_192
# Per-page gap entries are bounded; the remainder is summarized in one line.
MAX_GAP_ENTRIES = 16
# v1 bound its scope to a digest of the whole enumerated population, so any
# unrelated append invalidated it. v2 names a retained population snapshot.
SNAPSHOT_CURSOR_VERSION = 2
# Open failures that describe the selected path itself. Anything else (for
# example EMFILE, ENFILE, ENOMEM, EIO) is systemic and must stay retryable.
_FILE_SPECIFIC_OPEN_ERRORS = frozenset(
    {errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EACCES, errno.EPERM, errno.ENXIO, errno.ENODEV, errno.EISDIR}
)
_SEARCH_STATE_KEYS = frozenset({"file", "offset", "line", "line_start", "after", "skipped"})


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

    def encode(self, scope: dict[str, Any], state: dict[str, Any], *, version: int = 1) -> str:
        body = {"v": version, "scope": scope, "state": state}
        payload = base64.urlsafe_b64encode(self._canonical(body)).decode().rstrip("=")
        mac = hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()
        value = f"{payload}.{mac}"
        if len(value.encode()) > MAX_CURSOR_BYTES:
            raise SessionError("session continuation cursor exceeds its size bound")
        return value

    def _body(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, str) or len(value.encode()) > MAX_CURSOR_BYTES or "." not in value:
            raise SessionError("session continuation cursor is malformed")
        payload, mac = value.rsplit(".", 1)
        expected = hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, expected):
            raise StaleContinuationError("session continuation cursor is stale")
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
        if not isinstance(body, dict) or not isinstance(body.get("state"), dict):
            raise SessionError("session continuation cursor is malformed")
        return body

    def decode(self, value: Any, scope: dict[str, Any]) -> dict[str, Any]:
        body = self._body(value)
        if body.get("v") != 1:
            raise StaleContinuationError("session continuation cursor is stale")
        actual_scope = body.get("scope")
        if actual_scope != scope:
            if (
                isinstance(actual_scope, dict)
                and actual_scope.get("source_revision") != scope.get("source_revision")
                and {key: value for key, value in actual_scope.items() if key != "source_revision"}
                == {key: value for key, value in scope.items() if key != "source_revision"}
            ):
                raise StaleContinuationError("session source changed after continuation began")
            raise StaleContinuationError("session continuation cursor is stale")
        state: dict[str, Any] = body["state"]
        return state

    def decode_snapshot(self, value: Any, scope: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Return (snapshot handle, state) for a v2 token bound to exactly ``scope``."""
        body = self._body(value)
        if body.get("v") == 1:
            raise StaleContinuationError("session continuation predates retained search snapshots; restart the search")
        actual_scope = body.get("scope")
        if body.get("v") != SNAPSHOT_CURSOR_VERSION or not isinstance(actual_scope, dict):
            raise StaleContinuationError("session continuation cursor is stale")
        handle = actual_scope.get("snapshot")
        if {key: value for key, value in actual_scope.items() if key != "snapshot"} != scope or not isinstance(
            handle, str
        ):
            raise StaleContinuationError("session continuation does not match its original search scope")
        state: dict[str, Any] = body["state"]
        return handle, state


def _identity(info: Any) -> tuple[int, int, int, int]:
    """The observation a selected file is held to; ctime alone is not a content change."""
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


class _Gaps:
    """Bounded, page-local coverage gaps plus a running skipped-file count."""

    def __init__(self) -> None:
        self.entries: builtins.list[str] = []
        self.overflow = 0

    def add(self, reference: str, reason: str) -> None:
        if len(self.entries) < MAX_GAP_ENTRIES:
            self.entries.append(f"{reference}: {reason}")
        else:
            self.overflow += 1

    @property
    def count(self) -> int:
        return len(self.entries) + self.overflow

    def as_list(self) -> builtins.list[str]:
        if self.overflow:
            return [*self.entries, f"{self.overflow} further selected files were skipped for the same reasons"]
        return list(self.entries)


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
        snapshot_dir: Path | None = None,
    ):
        self.max_result_bytes = max_result_bytes
        self._snapshots = SnapshotStore(
            snapshot_dir if snapshot_dir is not None else state_home() / "raw-session-search"
        )
        self.scope = scope
        # An explicitly empty source tuple is a meaningful isolated config.
        # Truthiness here silently re-enables the host's default locations.
        configured_sources = self.default_sources() if sources is None else sources
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
    def _snippet(handle: BinaryIO, line_start: int, match_offset: int, query_bytes: int) -> tuple[int, str]:
        start = max(line_start, match_offset - 200)
        # Keep the query intact even when the caller supplied a long literal.
        handle.seek(start)
        data = handle.read(max(2_000, match_offset - start + query_bytes))
        return start, data.decode("utf-8", errors="replace").rstrip("\r\n")[:2_000]

    def _scan_literal(
        self,
        source: SessionSource,
        files: Sequence[tuple[Path, Any]],
        query: str,
        limit: int,
        scan_bytes: int,
        state: dict[str, int],
        make_cursor: Callable[[dict[str, int]], str | None],
        one_per_file: bool,
    ) -> dict[str, Any]:
        """Scan the selected population from ``state``.

        A selected file that vanished, changed, or raced with this read is
        skipped with an explicit coverage gap: one live file must not abort
        coverage of the rest, and a changed file cannot be resumed at an
        offset its old observation defined. Matches are read through the
        same descriptor whose identity was checked before and after the read.
        """
        query_bytes = query.encode("utf-8")
        if state["file"] > len(files):
            raise SessionError("session continuation cursor is malformed")
        scanned = 0
        gaps = _Gaps()
        rows: builtins.list[dict[str, Any]] = []
        page_full_state: dict[str, int] | None = None
        systemic_error: str | None = None

        def next_file(current: dict[str, int], *, skipped: bool = False) -> dict[str, int]:
            moved = {**current, "file": current["file"] + 1, "offset": 0, "line": 1, "line_start": 0}
            if "after" in moved:
                moved["after"] = 0
            if skipped and "skipped" in moved:
                moved["skipped"] += 1
            return moved

        while state["file"] < len(files) and scanned < scan_bytes and page_full_state is None:
            path, observed = files[state["file"]]
            reference = self._reference(source, path)
            if state["offset"] and state["offset"] >= observed.st_size:
                # Reached through a block this scan already validated. A file
                # empty at selection still opens below: offset 0 proves nothing.
                state = next_file(state)
                continue
            try:
                # Non-blocking and no-follow: a path replaced by a FIFO or a
                # symlink after selection must become a gap, not a hang.
                descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
            except OSError as exc:
                if exc.errno not in _FILE_SPECIFIC_OPEN_ERRORS:
                    # Process- or system-wide (descriptor exhaustion, memory,
                    # I/O): not this file's fault. Stop here without skipping
                    # it, so the same continuation retries this file.
                    systemic_error = f"{errno.errorcode.get(exc.errno or 0, 'OSError')}: {exc.strerror}"
                    page_full_state = dict(state)
                    break
                gaps.add(reference, "selected file disappeared or became unreadable before it was searched")
                state = next_file(state, skipped=True)
                continue
            with os.fdopen(descriptor, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(observed):
                    reason = "changed after it was partially searched" if state["offset"] else "changed after selection"
                    gaps.add(reference, f"selected file {reason}; not searched")
                    state = next_file(state, skipped=True)
                    continue
                remaining = min(_SCAN_BLOCK_BYTES, scan_bytes - scanned, observed.st_size - state["offset"])
                handle.seek(state["offset"])
                data = handle.read(remaining)
                handle.seek(max(0, state["offset"] - max(0, len(query_bytes) - 1)))
                tail = handle.read(state["offset"] - handle.tell())
                rows_before = len(rows)
                combined = tail + data
                combined_start = state["offset"] - len(tail)
                index = 0
                found = False
                block_state = state
                # End of the last match accepted from this block. A full page
                # resumes at this block's start and skips through it, so the
                # continuation never re-enters an earlier, fully scanned file.
                accepted_end = 0
                while len(data) == remaining:
                    index = combined.find(query_bytes, index)
                    if index < 0:
                        break
                    absolute = combined_start + index
                    end = absolute + len(query_bytes)
                    # Matches wholly in the replay tail, or accepted before a
                    # continuation resumed inside this block, were already returned.
                    if end <= state["offset"] or end <= state.get("after", 0):
                        index += len(query_bytes)
                        continue
                    if len(rows) >= limit:
                        # Do not consume this match: the next page rescans this
                        # block and it is the first candidate there. A match
                        # stream skips what this block already returned.
                        page_full_state = dict(state)
                        if not one_per_file and "after" in page_full_state:
                            page_full_state["after"] = max(state.get("after", 0), accepted_end)
                        break
                    line, line_start = self._line_for_match(state, combined, index, len(tail))
                    offset, text = self._snippet(handle, line_start, absolute, len(query_bytes))
                    rows.append(
                        {
                            "reference": reference,
                            "line": line,
                            "offset": offset,
                            "text": text[: min(2_000, max(128, self.max_result_bytes // 4))],
                            "source_observation": _identity(observed),
                        }
                    )
                    found = True
                    accepted_end = end
                    if one_per_file:
                        break
                    index += len(query_bytes)
                if len(data) != remaining or _identity(os.fstat(handle.fileno())) != _identity(observed):
                    # A concurrent write or shrink raced this read; nothing from
                    # this block is evidence of the selected observation.
                    del rows[rows_before:]
                    page_full_state = None
                    # The read happened; it counts against this request's budget.
                    scanned += len(data)
                    gaps.add(reference, "selected file changed while it was being searched; not searched")
                    state = next_file(block_state, skipped=True)
                    continue
            scanned += len(data)
            if page_full_state is not None:
                break
            if found and one_per_file:
                state = next_file(state)
                continue
            self._advance_line(state, data)
            if state.get("after", 0) and state["offset"] >= state["after"]:
                # Only once the scan has passed every returned match; a short
                # resumed budget may stop inside the block.
                state["after"] = 0
            if state["offset"] >= observed.st_size:
                state = next_file(state)
        final_state = page_full_state if page_full_state is not None else state
        truncated = page_full_state is not None or state["file"] < len(files)
        return {
            "rows": rows,
            "scanned_bytes": scanned,
            "truncated": truncated,
            "next_cursor": make_cursor(final_state) if truncated else None,
            "gaps": [
                *gaps.as_list(),
                *(
                    [f"search paused by a transient system error ({systemic_error}); resume to retry"]
                    if systemic_error
                    else []
                ),
            ],
            "skipped_now": gaps.count,
            "state": final_state,
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
        summarize_skipped: bool = True,
    ) -> dict[str, Any]:
        """``summarize_skipped=False`` leaves the earlier-pages summary to a fan-out
        caller, which reports it once on its own terminal page."""
        source = self._source(provider)
        if not query or len(query) > 1_000:
            raise SessionError("query must contain 1-1000 characters")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise SessionError("max_results must be a positive integer")
        budget = self._scan_bytes(scan_bytes)
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        # The reference filter is part of the scope, so a continuation begun on
        # one file never resumes over another population. It is bound by
        # digest: an exact path of any valid length stays in the private
        # snapshot binding, not in the size-bounded token.
        reference_sha256 = hashlib.sha256(reference.encode()).hexdigest() if reference is not None else None
        scope = {
            "principal": self.scope,
            "provider": provider,
            "query_sha256": query_sha256,
            "reference_sha256": reference_sha256,
        }
        binding = SnapshotBinding(self.scope, provider, query_sha256, reference, source.root)
        purpose = "session-search"
        snapshot_handle: str | None = None
        files: Sequence[tuple[Path, Any]]
        if cursor is not None:
            if cursor_key is None:
                raise SessionError("session continuation cursor is unavailable")
            snapshot_handle, state = OpaqueSessionCursor(self.scope, cursor_key, purpose).decode_snapshot(cursor, scope)
            if set(state) != _SEARCH_STATE_KEYS or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in state.values()
            ):
                raise SessionError("session continuation cursor is malformed")
            try:
                files = self._snapshots.load(snapshot_handle, binding).files
            except SnapshotTemporarilyUnavailableError as exc:
                raise RetryableSessionError(
                    f"session continuation snapshot is temporarily unreadable ({exc}); retry the same continuation"
                ) from exc
            except OSError as exc:
                # The store lock itself failed to open (for example EMFILE).
                raise RetryableSessionError(
                    f"session continuation snapshot is temporarily unreadable ({exc}); retry the same continuation"
                ) from exc
            except SnapshotUnavailableError as exc:
                return {
                    "provider": provider,
                    "matches": [],
                    "scanned_bytes": 0,
                    "truncated": False,
                    "next_cursor": None,
                    "gaps": [str(exc)],
                    # The retained counts went with the snapshot; the gap says so.
                    "skipped_earlier": 0,
                    "skipped_now": 0,
                }
        else:
            if reference is not None:
                selected_source, path = self._path_from_reference(reference)
                if selected_source.provider != provider:
                    raise SessionError("reference provider must match search provider")
                files = [(path, path.stat(follow_symlinks=False))]
            else:
                files = self._files(source)
            state = {"file": 0, "offset": 0, "line": 1, "line_start": 0, "after": 0, "skipped": 0}
        prior_skipped = state["skipped"]
        issue_gaps: builtins.list[str] = []

        def make_cursor(next_state: dict[str, int]) -> str | None:
            nonlocal snapshot_handle
            if cursor_key is None:
                return None
            if snapshot_handle is None:
                # Only a search that actually continues retains its population.
                try:
                    snapshot_handle = self._snapshots.create(binding, files).handle
                except OSError as exc:
                    # A physical failure (disk, permissions) to retain the
                    # population: this page stays valid; only resuming is lost.
                    issue_gaps.append(f"continuation unavailable: {exc}")
                    return None
            return OpaqueSessionCursor(self.scope, cursor_key, purpose).encode(
                {**scope, "snapshot": snapshot_handle}, next_state, version=SNAPSHOT_CURSOR_VERSION
            )

        result = self._scan_literal(source, files, query, max_results, budget, state, make_cursor, False)
        gaps = [*result["gaps"], *issue_gaps]
        if summarize_skipped and not result["truncated"] and prior_skipped:
            gaps.append(f"{prior_skipped} selected files were skipped on earlier pages of this continuation")
        return {
            "provider": provider,
            "matches": result["rows"],
            "scanned_bytes": result["scanned_bytes"],
            "truncated": result["truncated"],
            "next_cursor": result["next_cursor"],
            "gaps": gaps,
            "skipped_earlier": prior_skipped,
            "skipped_now": result["skipped_now"],
            # A resumed completion token (empty population) is reusable as-is:
            # resuming it again scans nothing and carries the same count.
            "completion_cursor": cursor if cursor is not None and not files else None,
        }

    def completed_skips_token(self, provider: str, query: str, skipped: int, *, cursor_key: bytes) -> str | None:
        """A continuation for a finished provider that still owes a skipped count.

        A fan-out continuation outlives one provider's scan; its terminal page
        must still report files that provider skipped. The token resumes an
        empty retained population whose only state is that count.
        """
        source = self._source(provider)
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        binding = SnapshotBinding(self.scope, provider, query_sha256, None, source.root)
        try:
            handle = self._snapshots.create(binding, ()).handle
        except OSError:
            return None
        scope = {
            "principal": self.scope,
            "provider": provider,
            "query_sha256": query_sha256,
            "reference_sha256": None,
            "snapshot": handle,
        }
        state = {"file": 0, "offset": 0, "line": 1, "line_start": 0, "after": 0, "skipped": skipped}
        return OpaqueSessionCursor(self.scope, cursor_key, "session-search").encode(
            scope, state, version=SNAPSHOT_CURSOR_VERSION
        )

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
            scope: dict[str, Any] = {
                "principal": self.service.scope,
                "provider": source.provider,
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "source_revision": self.revision,
            }
            if cursor is not None:
                if cursor_key is None:
                    raise SessionError("session continuation cursor is unavailable")
                state = OpaqueSessionCursor(self.service.scope, cursor_key, "session-timeline").decode(cursor, scope)
                if set(state) != {"file", "offset", "line", "line_start"} or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in state.values()
                ):
                    raise SessionError("session continuation cursor is malformed")
            else:
                state = {"file": 0, "offset": 0, "line": 1, "line_start": 0}

            def make_cursor(next_state: dict[str, int]) -> str | None:
                if cursor_key is None:
                    return None
                return OpaqueSessionCursor(self.service.scope, cursor_key, "session-timeline").encode(scope, next_state)

            result = self.service._scan_literal(
                source,
                files,
                query,
                max_results,
                self.service._scan_bytes(scan_bytes),
                state,
                make_cursor,
                True,
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
                "gaps": result["gaps"],
                "skipped_now": result["skipped_now"],
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
