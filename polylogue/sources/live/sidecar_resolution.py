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
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol

from polylogue.archive.artifact_taxonomy import ArtifactKind, strong_path_classification
from polylogue.archive.revision_authority import raw_receipt_order_sql
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider
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

#: One retained revision of a sibling transcript: blob hash, revision kind, blob
#: size, append start and end offsets, raw id, recorded predecessor raw id.
_RevisionRow = tuple[str, str, int, int | None, int | None, str, str | None]

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
    raw_id: str


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

    def __init__(
        self,
        archive_root: Path,
        *,
        blob_root: Path | None = None,
        source_read: RetainedSidecarSourceRead | None = None,
    ) -> None:
        self._archive_root = Path(archive_root)
        self._blob_root = blob_root
        self._source_read = source_read

    def claude_code_scope(self, source_path: str | Path | None) -> RetainedSidecarScope:
        tool_results_dir = resolve_tool_results_dir(source_path)
        if tool_results_dir is None or source_path is None:
            return UNRESOLVED_SIDECAR_SCOPE
        scope_key = str(tool_results_dir)
        with self._source_reader() as conn:
            if conn is None and self._source_read is None:
                return RetainedSidecarScope(scope_key=scope_key)
            rows = self._children_of(conn, tool_results_dir)
            if not rows:
                return RetainedSidecarScope(scope_key=scope_key)
            siblings = self._retained_siblings(conn, source_path)
        witness = tuple(
            (
                "file",
                row.source_path,
                row.raw_id,
                row.blob_hash,
                row.blob_size,
                row.file_mtime_ms,
            )
            for row in rows
        ) + tuple(
            ("sibling", sibling.coordinate, sibling.selection_witness, sibling.record_blobs) for sibling in siblings
        )
        return RetainedSidecarScope(
            scope_key=scope_key,
            files=tuple(self._as_file(row) for row in rows),
            siblings=siblings,
            available=True,
            witness=witness,
        )

    def gemini_cli_scope(self, source_path: str | Path | None, session_id: str | None) -> RetainedSidecarScope:
        tool_outputs_dir = resolve_tool_outputs_dir(source_path, session_id)
        if tool_outputs_dir is None:
            return UNRESOLVED_SIDECAR_SCOPE
        scope_key = str(tool_outputs_dir)
        with self._source_reader() as conn:
            if conn is None and self._source_read is None:
                return RetainedSidecarScope(scope_key=scope_key)
            rows = self._children_of(conn, tool_outputs_dir)
        if not rows:
            return RetainedSidecarScope(scope_key=scope_key)
        witness = tuple(
            ("file", row.source_path, row.raw_id, row.blob_hash, row.blob_size, row.file_mtime_ms) for row in rows
        )
        return RetainedSidecarScope(
            scope_key=scope_key,
            files=tuple(self._as_file(row) for row in rows),
            available=True,
            witness=witness,
        )

    # -- retained reads -------------------------------------------------

    @contextmanager
    def _source_reader(self) -> Iterator[sqlite3.Connection | None]:
        """Scope a read-only source-tier connection, or ``None`` on refusal.

        The refusal carries the reason the seam decided, so an unresolved
        scope traced back to here names *why* the tier did not answer rather
        than being indistinguishable from a scope holding no retained members.
        """
        if self._source_read is not None:
            yield None
            return
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

    def _children_of(self, conn: sqlite3.Connection | None, directory: Path) -> list[_RetainedRow]:
        """Latest retained row per file directly inside ``directory``.

        Direct children only, matching what a directory listing of the scope
        would have yielded: ``subagents/`` and other nested trees under the
        same prefix are separate coordinates, never scope members.
        """
        prefix = f"{directory.as_posix()}/"
        low, high = _prefix_range(prefix)
        if self._source_read is not None:
            scope = self._source_read.retained_children_rows(low, high)
        else:
            if conn is None:
                raise RuntimeError("retained sidecar query has no admitted Source reader")
            scope = _retained_rows(conn, _RETAINED_CHILDREN_SQL, (low, high))
        with scope as cursor:
            rows = cursor.fetchall()
        latest: dict[str, _RetainedRow] = {}
        for source_path, blob_hash, blob_size, file_mtime_ms, raw_id in rows:
            check_compute_cancelled()
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
                    raw_id=str(raw_id),
                ),
            )
        return sorted(latest.values(), key=lambda row: row.source_path)

    def _retained_siblings(
        self, conn: sqlite3.Connection | None, source_path: str | Path
    ) -> tuple[SiblingTranscript, ...]:
        """Retained transcripts sharing this scope, excluding ``source_path``.

        The root ``.jsonl`` plus every ``subagents/agent-*.jsonl``. A sibling
        that grew through append ingest is its newest full revision followed
        by the append chain descended from it: each step is the one append
        whose recorded predecessor is the chain's last raw and which starts
        where the chain ends. An append of another revision, or two competing
        steps, stops the chain rather than guess, so a superseded revision
        never contributes tool ids. A shorter index can only over-report
        debt, a wrong one would hide it.
        """
        path = Path(source_path)
        session_dir = path.parent.parent if path.parent.name == "subagents" else path.parent / path.stem
        root_path = session_dir.parent / f"{session_dir.name}.jsonl"
        low, high = _prefix_range(f"{(session_dir / 'subagents').as_posix()}/")
        # Revisions are ranked by their newest durable ``raw_payload``
        # receipt, as ``retained_assembly`` ranks currency: bytes that return
        # to an earlier value reuse that raw row, so ``acquired_at_ms`` is its
        # first sighting and would rank an intervening revision newest.
        if self._source_read is not None:
            scope = self._source_read.retained_sibling_rows(root_path.as_posix(), low, high)
        else:
            if conn is None:
                raise RuntimeError("retained sibling query has no admitted Source reader")
            scope = _retained_rows(conn, _RETAINED_SIBLINGS_SQL, (root_path.as_posix(), low, high))
        with scope as cursor:
            rows = cursor.fetchall()
        own = path.as_posix()
        revisions: dict[str, list[_RevisionRow]] = {}
        for (
            candidate_path,
            blob_hash,
            revision_kind,
            blob_size,
            append_start,
            append_end,
            raw_id,
            predecessor_raw_id,
        ) in rows:
            check_compute_cancelled()
            candidate = str(candidate_path)
            if candidate == own:
                continue
            if candidate != root_path.as_posix() and not candidate.endswith(".jsonl"):
                continue
            revisions.setdefault(candidate, []).append(
                (
                    str(blob_hash).lower(),
                    str(revision_kind),
                    int(blob_size or 0),
                    int(append_start) if append_start is not None else None,
                    int(append_end) if append_end is not None else None,
                    str(raw_id),
                    str(predecessor_raw_id) if predecessor_raw_id is not None else None,
                )
            )
        siblings: list[SiblingTranscript] = []
        for candidate, candidate_rows in sorted(revisions.items()):
            check_compute_cancelled()
            fulls = [row for row in candidate_rows if row[1] != "append"]
            if not fulls:
                # Only deltas retained: the largest is the best partial view.
                largest = max(candidate_rows, key=lambda row: row[2])
                siblings.append(
                    SiblingTranscript(
                        coordinate=candidate,
                        open_records=self._records_from_blobs([(largest[5], largest[0])]),
                        record_blobs=((largest[5], largest[0]),),
                        selection_witness=tuple(tuple(row) for row in candidate_rows),
                    )
                )
                continue
            baseline = fulls[-1]
            blob_hashes = [(baseline[5], baseline[0])]
            end = baseline[2]
            last_raw_id = baseline[5]
            # Rows are in receipt order. A step must be admitted after the
            # selected baseline and after the previous step: a historical
            # append of the same revision (``A -> A+X -> B -> A``) predates
            # the current ``A`` and is not part of it.
            last_position = max(index for index, row in enumerate(candidate_rows) if row is baseline)
            # Appends indexed by recorded predecessor, so each step reads only
            # the rows naming the chain's last raw: a file grown through
            # thousands of appends is walked in linear time.
            appends_by_predecessor: dict[str, list[tuple[int, _RevisionRow]]] = {}
            for index, row in enumerate(candidate_rows):
                check_compute_cancelled()
                if row[1] == "append" and row[3] is not None and row[6] is not None:
                    appends_by_predecessor.setdefault(row[6], []).append((index, row))
            while True:
                check_compute_cancelled()
                steps = [
                    (index, row)
                    for index, row in appends_by_predecessor.get(last_raw_id, ())
                    if index > last_position and row[3] == end and row[4] is not None and row[4] > end
                ]
                if len(steps) != 1:
                    break
                last_position, step = steps[0]
                blob_hashes.append((step[5], step[0]))
                assert step[4] is not None
                end = step[4]
                last_raw_id = step[5]
            siblings.append(
                SiblingTranscript(
                    coordinate=candidate,
                    open_records=self._records_from_blobs(blob_hashes),
                    record_blobs=tuple(blob_hashes),
                    selection_witness=tuple(tuple(row) for row in candidate_rows),
                )
            )
        return tuple(siblings)

    def _records_from_blobs(self, blob_hashes: list[tuple[str, str]]) -> Callable[[], Iterator[object]]:
        def open_records() -> Iterator[object]:
            for raw_id, blob_hash in blob_hashes:
                check_compute_cancelled()
                yield from self._records_from_blob(raw_id, blob_hash)()

        return open_records

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
            read_text=self._text_from_blob(row.raw_id, row.blob_hash),
            raw_id=row.raw_id,
            blob_hash=row.blob_hash,
            source_path=row.source_path,
        )

    def _text_from_blob(self, raw_id: str, blob_hash: str) -> Callable[[], str]:
        def read() -> str:
            check_compute_cancelled()
            if self._source_read is not None:
                with self._source_read.open_sidecar_payload(raw_id, bytes.fromhex(blob_hash)) as handle:
                    return handle.read().decode("utf-8", errors="replace")
            return self._blob_path(blob_hash).read_text(encoding="utf-8", errors="replace")

        return read

    def _records_from_blob(self, raw_id: str, blob_hash: str) -> Callable[[], Iterator[object]]:
        def open_records() -> Iterator[object]:
            check_compute_cancelled()
            scope = (
                self._source_read.open_sidecar_payload(raw_id, bytes.fromhex(blob_hash))
                if self._source_read is not None
                else self._blob_path(blob_hash).open("rb")
            )
            with scope as handle:
                yield from iter_jsonl_records(lambda: iter(handle))

        return open_records


_RETAINED_CHILDREN_SQL = f"""
    SELECT source_path, hex(blob_hash), blob_size, file_mtime_ms, raw_id
    FROM raw_sessions
    WHERE source_path >= ? AND source_path < ?
    ORDER BY {raw_receipt_order_sql("raw_sessions")} DESC, raw_id DESC
"""


_RETAINED_SIBLINGS_SQL = f"""
    SELECT r.source_path, hex(r.blob_hash), r.revision_kind, r.blob_size,
           r.append_start_offset, r.append_end_offset, r.raw_id, r.predecessor_raw_id
    FROM raw_sessions AS r
    WHERE r.source_path = ? OR (r.source_path >= ? AND r.source_path < ?)
    ORDER BY r.source_path, {raw_receipt_order_sql("r")}, r.raw_id
"""


class RetainedSidecarSourceRead(Protocol):
    """The resolver's two declared predicates and retained byte capability."""

    def retained_children_rows(self, low: str, high: str) -> AbstractContextManager[sqlite3.Cursor]: ...

    def retained_sibling_rows(
        self,
        root_path: str,
        low: str,
        high: str,
    ) -> AbstractContextManager[sqlite3.Cursor]: ...

    def open_sidecar_payload(self, raw_id: str, blob_hash: bytes) -> AbstractContextManager[BinaryIO]: ...


@contextmanager
def _retained_rows(
    connection: sqlite3.Connection,
    sql: str,
    parameters: tuple[object, ...],
) -> Iterator[sqlite3.Cursor]:
    """Retain the actual statement and settle it through the canonical cursor owner."""
    from polylogue.storage.io_phase_metrics import connection_cursor

    with connection_cursor(connection, sql, parameters) as cursor:
        yield cursor


def select_retained_claude_sidecar_owner_raw_ids(
    reader: PreparedSessionSourceRead,
    acquired_raw_ids: Sequence[str],
) -> tuple[str, ...]:
    """Select retained transcript owners from actual newly acquired sidecar scopes.

    The caller lends its original prepared Source reader before hydration;
    paths only select retained rows and never authorize a filesystem read.
    Canonical replay owns revision selection and subsequent publication.
    """
    scopes: set[Path] = set()
    for raw_id in acquired_raw_ids:
        check_compute_cancelled()
        provider, _blob_hash, source_path, _kind, _size = reader.raw_revision_descriptor(raw_id)
        if provider is not Provider.CLAUDE_CODE:
            continue
        path = Path(source_path)
        classification = strong_path_classification(path, provider=provider)
        if classification is not None and classification.kind is ArtifactKind.TOOL_RESULT_SIDECAR:
            scopes.add(path.parent.parent)
    owners: dict[str, None] = {}
    for session_dir in sorted(scopes):
        check_compute_cancelled()
        root_path = (session_dir.parent / f"{session_dir.name}.jsonl").as_posix()
        low, high = _prefix_range(f"{(session_dir / 'subagents').as_posix()}/")
        candidates: list[str] = []
        with reader.retained_sibling_rows(root_path, low, high) as rows:
            for row in rows:
                check_compute_cancelled()
                source_path, raw_id = str(row[0]), str(row[6])
                if source_path == root_path or source_path.endswith(".jsonl"):
                    candidates.append(raw_id)
        # Reading a descriptor retains its original blob input (a seal write),
        # which the seal refuses while a Source row read is open; read them
        # once the sibling cursor has settled.
        for raw_id in candidates:
            check_compute_cancelled()
            provider, _hash, _path, _kind, _size = reader.raw_revision_descriptor(raw_id)
            if provider is Provider.CLAUDE_CODE:
                owners.setdefault(raw_id, None)
    return tuple(owners)


if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
