"""Who resolves a tool-output sidecar scope, and from what (polylogue-cq1ql).

Two resolvers implement
:class:`~polylogue.sources.sidecar_evidence.SidecarResolver`, and the split
between them is the point of the seam:

``FilesystemSidecarResolver``
    Acquisition. The original source tree is the input at that point, so
    enumerating it is not an ambient read -- it *is* the read. The same walk
    that acquires the transcript also acquires every file of the scope as an
    ordinary ``tool_result_sidecar`` raw artifact (declared by the Claude Code
    and gemini-cli ``OriginSpec`` artifact rules), which is what leaves
    something for the other resolver to find.

``RetainedSidecarResolver``
    Derivation. Resolves the same scope out of ``source.db`` plus the blob
    store and never touches the source tree, so a reparse of retained bytes
    reproduces the full tool text, its outcome and its ownership after the
    original transcript and sidecar directories are gone
    (``docs/design/retained-inputs-and-supersession.md`` D3/S8/S9).

A scope's durable coordinate is the sidecar directory path the transcript's
own retained ``source_path`` derives (``resolve_tool_results_dir`` /
``resolve_tool_outputs_dir``): retained rows carry the path they were acquired
from, so the coordinate reproduces without the tree. Resolution is by path
prefix over retained rows, which is the same population the directory listing
saw -- a file that was never acquired is absent from both.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from polylogue.logging import get_logger
from polylogue.sources.live.gemini_tool_output_sidecars import (
    resolve_tool_outputs_dir,
    tool_output_files_from_directory,
)
from polylogue.sources.live.tool_result_sidecars import (
    resolve_tool_results_dir,
    sibling_transcripts_from_directory,
    sidecar_files_from_directory,
)
from polylogue.sources.sidecar_evidence import (
    UNRESOLVED_SIDECAR_SCOPE,
    RetainedSidecarFile,
    RetainedSidecarScope,
    SiblingTranscript,
    iter_jsonl_records,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.tier_access import TierRefusal, open_tier_reader

logger = get_logger(__name__)

__all__ = [
    "FilesystemSidecarResolver",
    "RetainedSidecarResolver",
]


class FilesystemSidecarResolver:
    """Acquisition-time resolution from the source tree."""

    def claude_code_scope(self, source_path: str | Path | None) -> RetainedSidecarScope:
        tool_results_dir = resolve_tool_results_dir(source_path)
        if tool_results_dir is None or not tool_results_dir.is_dir():
            return UNRESOLVED_SIDECAR_SCOPE
        assert source_path is not None
        return RetainedSidecarScope(
            scope_key=str(tool_results_dir),
            files=sidecar_files_from_directory(tool_results_dir),
            siblings=sibling_transcripts_from_directory(source_path),
            available=True,
        )

    def gemini_cli_scope(self, source_path: str | Path | None, session_id: str | None) -> RetainedSidecarScope:
        tool_outputs_dir = resolve_tool_outputs_dir(source_path, session_id)
        if tool_outputs_dir is None or not tool_outputs_dir.is_dir():
            return UNRESOLVED_SIDECAR_SCOPE
        return RetainedSidecarScope(
            scope_key=str(tool_outputs_dir),
            files=tool_output_files_from_directory(tool_outputs_dir),
            available=True,
        )


@dataclass(frozen=True, slots=True)
class _RetainedRow:
    source_path: str
    blob_hash: str
    blob_size: int
    file_mtime_ms: int | None


def _prefix_range(prefix: str) -> tuple[str, str]:
    """Half-open key range covering every path under ``prefix``.

    A range comparison keeps ``idx_raw_sessions_source_path`` usable. ``LIKE``
    would not: SQLite declines its prefix optimization whenever an ``ESCAPE``
    clause is present, and a source path can legitimately contain ``%`` or
    ``_``, so the escape is not optional. On a rebuild this is the difference
    between one indexed seek and a full scan of ``raw_sessions`` per parsed
    transcript.
    """
    return prefix, prefix[:-1] + chr(ord(prefix[-1]) + 1)


class RetainedSidecarResolver:
    """Derivation-time resolution from retained bytes only.

    Read-only against the source tier so it is safe beside the daemon's single
    writer. What an unavailable store looks like is not this module's to
    decide: ``open_tier_reader`` classifies absence, schema skew and an
    unopenable file once at the seam, and any refusal yields an unresolved
    scope -- which records nothing at all rather than manufacturing absence
    evidence from infrastructure state. A query that fails *after* the seam
    admitted the tier is a corrupt store, not an absent one, and propagates:
    reporting it as zero retained members is exactly the silent truncation
    this resolver exists to prevent.
    """

    def __init__(self, archive_root: Path, *, blob_root: Path | None = None) -> None:
        self._archive_root = Path(archive_root)
        self._blob_root = blob_root

    def claude_code_scope(self, source_path: str | Path | None) -> RetainedSidecarScope:
        tool_results_dir = resolve_tool_results_dir(source_path)
        if tool_results_dir is None or source_path is None:
            return UNRESOLVED_SIDECAR_SCOPE
        with self._source_reader() as conn:
            if conn is None:
                return UNRESOLVED_SIDECAR_SCOPE
            rows = self._children_of(conn, tool_results_dir)
            if not rows:
                return UNRESOLVED_SIDECAR_SCOPE
            siblings = self._retained_siblings(conn, source_path)
        return RetainedSidecarScope(
            scope_key=str(tool_results_dir),
            files=tuple(self._as_file(row) for row in rows),
            siblings=siblings,
            available=True,
        )

    def gemini_cli_scope(self, source_path: str | Path | None, session_id: str | None) -> RetainedSidecarScope:
        tool_outputs_dir = resolve_tool_outputs_dir(source_path, session_id)
        if tool_outputs_dir is None:
            return UNRESOLVED_SIDECAR_SCOPE
        with self._source_reader() as conn:
            if conn is None:
                return UNRESOLVED_SIDECAR_SCOPE
            rows = self._children_of(conn, tool_outputs_dir)
        if not rows:
            return UNRESOLVED_SIDECAR_SCOPE
        return RetainedSidecarScope(
            scope_key=str(tool_outputs_dir),
            files=tuple(self._as_file(row) for row in rows),
            available=True,
        )

    # -- retained reads -------------------------------------------------

    @contextmanager
    def _source_reader(self) -> Iterator[sqlite3.Connection | None]:
        """Scope a read-only source-tier connection, or ``None`` on refusal.

        The refusal carries the reason the seam decided, so an unresolved
        scope traced back to here names *why* the tier did not answer rather
        than being indistinguishable from a scope holding no retained members.
        """
        source_db = self._archive_root / ARCHIVE_TIER_SPECS[ArchiveTier.SOURCE].filename
        with open_tier_reader(ArchiveTier.SOURCE, source_db) as acquired:
            if isinstance(acquired, TierRefusal):
                logger.debug(
                    "retained sidecar scope unresolved: source tier %s (%s)",
                    acquired.reason,
                    acquired.detail,
                )
                yield None
            else:
                yield acquired.connection

    def _children_of(self, conn: sqlite3.Connection, directory: Path) -> list[_RetainedRow]:
        """Latest retained row per file directly inside ``directory``.

        Direct children only, matching what a directory listing of the scope
        would have yielded: ``subagents/`` and other nested trees under the
        same prefix are separate coordinates, never scope members.
        """
        prefix = f"{directory.as_posix()}/"
        low, high = _prefix_range(prefix)
        rows = conn.execute(
            """
            SELECT source_path, hex(blob_hash), blob_size, file_mtime_ms
            FROM raw_sessions
            WHERE source_path >= ? AND source_path < ?
            ORDER BY acquired_at_ms DESC, raw_id DESC
            """,
            (low, high),
        ).fetchall()
        latest: dict[str, _RetainedRow] = {}
        for source_path, blob_hash, blob_size, file_mtime_ms in rows:
            remainder = str(source_path)[len(prefix) :]
            if not remainder or "/" in remainder:
                continue
            latest.setdefault(
                str(source_path),
                _RetainedRow(
                    source_path=str(source_path),
                    blob_hash=str(blob_hash).lower(),
                    blob_size=int(blob_size),
                    file_mtime_ms=int(file_mtime_ms) if file_mtime_ms is not None else None,
                ),
            )
        return sorted(latest.values(), key=lambda row: row.source_path)

    def _retained_siblings(self, conn: sqlite3.Connection, source_path: str | Path) -> tuple[SiblingTranscript, ...]:
        """Retained transcripts sharing this scope, excluding ``source_path``.

        The root ``.jsonl`` plus every ``subagents/agent-*.jsonl``. An append
        delta is a partial view of its file, so a full revision of the same
        path is preferred when the archive holds one -- a sibling index only
        corroborates ownership, and a partial one can only over-report debt.
        """
        path = Path(source_path)
        session_dir = path.parent.parent if path.parent.name == "subagents" else path.parent / path.stem
        root_path = session_dir.parent / f"{session_dir.name}.jsonl"
        low, high = _prefix_range(f"{(session_dir / 'subagents').as_posix()}/")
        rows = conn.execute(
            """
            SELECT source_path, hex(blob_hash), revision_kind
            FROM raw_sessions
            WHERE source_path = ? OR (source_path >= ? AND source_path < ?)
            ORDER BY
                CASE WHEN revision_kind = 'append' THEN 1 ELSE 0 END,
                blob_size DESC,
                acquired_at_ms DESC,
                raw_id DESC
            """,
            (root_path.as_posix(), low, high),
        ).fetchall()
        own = path.as_posix()
        chosen: dict[str, str] = {}
        for candidate_path, blob_hash, _revision_kind in rows:
            candidate = str(candidate_path)
            if candidate == own:
                continue
            if candidate != root_path.as_posix() and not candidate.endswith(".jsonl"):
                continue
            chosen.setdefault(candidate, str(blob_hash).lower())
        return tuple(
            SiblingTranscript(coordinate=candidate, open_records=self._records_from_blob(blob_hash))
            for candidate, blob_hash in sorted(chosen.items())
        )

    def _blob_path(self, blob_hash: str) -> Path:
        """Locate retained bytes in the archive this resolver was given.

        The blob root defaults to ``<archive_root>/blob`` rather than the
        process-wide ``blob_store_root()``: a replay is always scoped to one
        archive, and reading a second archive's ambient configuration is the
        class of coupling this seam exists to remove.
        """
        from polylogue.storage.blob_store import BlobStore

        root = self._blob_root if self._blob_root is not None else self._archive_root / "blob"
        return BlobStore(root).blob_path(blob_hash)

    def _as_file(self, row: _RetainedRow) -> RetainedSidecarFile:
        return RetainedSidecarFile(
            filename=Path(row.source_path).name,
            byte_size=row.blob_size,
            file_mtime_ms=row.file_mtime_ms,
            read_text=self._text_from_blob(row.blob_hash),
        )

    def _text_from_blob(self, blob_hash: str):  # type: ignore[no-untyped-def]
        def read() -> str:
            return self._blob_path(blob_hash).read_text(encoding="utf-8", errors="replace")

        return read

    def _records_from_blob(self, blob_hash: str):  # type: ignore[no-untyped-def]
        def open_records() -> Iterator[object]:
            with self._blob_path(blob_hash).open("rb") as handle:
                yield from iter_jsonl_records(lambda: iter(handle))

        return open_records
