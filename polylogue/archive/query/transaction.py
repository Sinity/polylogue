"""Shared bounded read transaction identity and continuation state.

Execution control protects the host; this module protects the protocol
boundary. A page carries enough opaque state to advance the same logical
request, so clients never reconstruct filters from a prose overflow hint.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from polylogue.archive.query.execution_control import (
    InterruptibleSQLiteRead,
    QueryAdmissionController,
    QueryExecutionContext,
    WorkloadClass,
    execute_archive_read,
    execute_archive_read_sync,
)
from polylogue.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
_TOKEN_VERSION = 1

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def archive_index_epoch(index_db: Path) -> str:
    """Index generation identifier: schema version, session count/rowid, and watermark.

    A continuation is bound to this exact frame (polylogue-z9gh.9 AC #5): if
    the index tier's schema version changes or the archive admits/mutates
    sessions between the page that issued a continuation and the page that
    resumes it, the two epochs differ and the resume is rejected as stale
    rather than silently replaying a moving relation with offset pagination.

    Related to, but deliberately stronger than,
    :func:`polylogue.archive.query.production_evaluator._index_epoch` (the
    rxdo.3 ``query_runs.archive_epoch``/``corpus_epoch`` convention): that
    function hashes schema version plus ``MAX(updated_at_ms)`` alone, but
    ``updated_at_ms``/``created_at_ms`` are NULL immediately after a raw
    session write and only backfilled by a later materialization stage, so a
    watermark-only epoch can fail to notice a session admitted between two
    pages. Row count and ``MAX(rowid)`` are populated unconditionally by
    every insert, so they catch new-session admission even before
    materialization runs; the watermark still catches an in-place mutation of
    an existing session that changes neither.
    """
    if not index_db.exists():
        return "index:absent"
    try:
        with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True, timeout=5.0)) as conn:
            version_row = conn.execute("PRAGMA user_version").fetchone()
            version = int(version_row[0]) if version_row is not None else 0
            frame_row = conn.execute("SELECT COUNT(*), MAX(rowid), MAX(updated_at_ms) FROM sessions").fetchone()
            count = int(frame_row[0]) if frame_row is not None and frame_row[0] is not None else 0
            max_rowid = int(frame_row[1]) if frame_row is not None and frame_row[1] is not None else 0
            watermark = int(frame_row[2]) if frame_row is not None and frame_row[2] is not None else 0
            return f"index:v{version}:{count}:{max_rowid}:{watermark}"
    except sqlite3.Error:
        logger.warning("query transaction: could not read index epoch for %s", index_db, exc_info=True)
        return "index:unknown"


class QueryContinuationStaleError(ValueError):
    """A continuation was issued against an archive frame that has since moved.

    Phase one prefers this honest, retryable error to duplicate/skip
    behavior over a moving offset relation (ADR 0001, invariant #5).
    """

    code = "query_continuation_stale"

    def __init__(self, *, issued_epoch: str, current_epoch: str) -> None:
        self.issued_epoch = issued_epoch
        self.current_epoch = current_epoch
        super().__init__(
            f"continuation was issued against archive epoch {issued_epoch!r}, but the archive is "
            f"now at {current_epoch!r}; restart the query instead of resuming to avoid duplicate or "
            "skipped rows"
        )


@dataclass(frozen=True, slots=True)
class QueryTransactionRequest:
    """Canonical request identity shared by page producers and adapters."""

    operation: str
    arguments: Mapping[str, object]
    page_size: int
    offset: int = 0
    projection: str = "default"
    stable_order: str = "canonical"
    archive_epoch: str = ""

    def __post_init__(self) -> None:
        if not self.operation.strip():
            raise ValueError("query operation must not be empty")
        if self.page_size < 1:
            raise ValueError("query page_size must be positive")
        if self.offset < 0:
            raise ValueError("query offset must not be negative")

    @property
    def query_ref(self) -> str:
        body = {
            "operation": self.operation,
            "arguments": dict(self.arguments),
            "projection": self.projection,
            "stable_order": self.stable_order,
        }
        return "query:" + hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()[:24]

    @property
    def result_ref(self) -> str:
        """Result identity: query identity bound to a declared archive epoch and order.

        Excludes offset/page-size/budget (physical delivery state), matching
        the ADR's identity split: two requests over the same logical query at
        the same archive frame share one result identity regardless of which
        page they're on.
        """
        body = {
            "query_ref": self.query_ref,
            "projection": self.projection,
            "stable_order": self.stable_order,
            "archive_epoch": self.archive_epoch,
        }
        return "result:" + hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()[:24]

    def next(self, *, offset: int) -> QueryTransactionRequest:
        return QueryTransactionRequest(
            operation=self.operation,
            arguments=self.arguments,
            page_size=self.page_size,
            offset=offset,
            projection=self.projection,
            stable_order=self.stable_order,
            archive_epoch=self.archive_epoch,
        )


def query_units_transaction_request(
    *,
    archive_root: Path,
    expression: str,
    session_filters: Mapping[str, object],
    page_size: int,
    offset: int = 0,
) -> QueryTransactionRequest:
    """Canonical ``query_units`` transaction request, bound to the live index epoch.

    The single constructor shared by the API, MCP, and daemon HTTP adapters so
    the three surfaces cannot drift into building the request differently (the
    #2472/#2470 partitioning-bug shape) and so archive-epoch binding lands in
    one place rather than three independently-maintained call sites.
    """
    return QueryTransactionRequest(
        operation="query_units",
        arguments={"expression": expression, "session_filters": dict(session_filters)},
        page_size=page_size,
        offset=max(0, offset),
        projection="terminal-unit-envelope",
        stable_order="canonical",
        archive_epoch=archive_index_epoch(Path(archive_root) / "index.db"),
    )


def validate_continuation_epoch(continuation_request: QueryTransactionRequest, *, archive_root: Path) -> None:
    """Reject a continuation whose declared archive frame has moved.

    An empty ``archive_epoch`` means the token predates epoch-binding (or was
    minted by a caller that never framed itself); such tokens are accepted
    rather than rejected outright so this check can land without breaking
    continuations issued before it existed.
    """
    if not continuation_request.archive_epoch:
        return
    current = archive_index_epoch(Path(archive_root) / "index.db")
    if continuation_request.archive_epoch != current:
        raise QueryContinuationStaleError(issued_epoch=continuation_request.archive_epoch, current_epoch=current)


@dataclass(frozen=True, slots=True)
class QueryContinuation:
    """Opaque, self-validating continuation for one advancing page."""

    request: QueryTransactionRequest
    result_ref: str

    def encode(self) -> str:
        request = self.request
        body = {
            "v": _TOKEN_VERSION,
            "result_ref": self.result_ref,
            "request": {
                "operation": request.operation,
                "arguments": dict(request.arguments),
                "page_size": request.page_size,
                "offset": request.offset,
                "projection": request.projection,
                "stable_order": request.stable_order,
                "archive_epoch": request.archive_epoch,
            },
        }
        encoded = base64.urlsafe_b64encode(_canonical_json(body).encode("utf-8")).decode("ascii").rstrip("=")
        return "q1." + encoded

    @classmethod
    def decode(cls, token: str) -> QueryContinuation:
        if not token.startswith("q1."):
            raise ValueError("invalid query continuation version")
        try:
            padded = token[3:] + "=" * (-len(token[3:]) % 4)
            body = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
            request_body = body["request"]
            if body["v"] != _TOKEN_VERSION:
                raise ValueError("unsupported query continuation version")
            request = QueryTransactionRequest(
                operation=str(request_body["operation"]),
                arguments=dict(request_body["arguments"]),
                page_size=int(request_body["page_size"]),
                offset=int(request_body["offset"]),
                projection=str(request_body["projection"]),
                stable_order=str(request_body["stable_order"]),
                archive_epoch=str(request_body.get("archive_epoch") or ""),
            )
            result_ref = str(body["result_ref"])
        except (binascii.Error, KeyError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid query continuation") from exc
        if not result_ref.startswith("result:"):
            raise ValueError("invalid query result reference")
        return cls(request=request, result_ref=result_ref)


@dataclass(frozen=True, slots=True)
class QueryResultPage(Generic[T]):
    """A page whose continuation advances by rows, not by a retry hint."""

    items: tuple[T, ...]
    request: QueryTransactionRequest
    has_more: bool
    result_ref: str

    @property
    def next_offset(self) -> int | None:
        return self.request.offset + len(self.items) if self.has_more else None

    @property
    def continuation(self) -> str | None:
        if self.next_offset is None:
            return None
        return QueryContinuation(
            request=self.request.next(offset=self.next_offset),
            result_ref=self.result_ref,
        ).encode()


class QueryTransaction:
    """Own execution context and request identity for one archive read."""

    def __init__(
        self,
        archive_root: Path,
        request: QueryTransactionRequest,
        *,
        workload_class: WorkloadClass = "interactive",
        admission_weight: int = 1,
        controller: QueryAdmissionController | None = None,
        read_timeout: float = 5.0,
    ) -> None:
        self.request = request
        self.context = QueryExecutionContext.create(
            query_text=_canonical_json(
                {
                    "operation": request.operation,
                    "arguments": dict(request.arguments),
                    "offset": request.offset,
                    "projection": request.projection,
                    "stable_order": request.stable_order,
                }
            ),
            workload_class=workload_class,
            admission_weight=admission_weight,
            owner_ref=request.operation,
        )
        self.archive_root = archive_root
        self.controller = controller
        self.read_timeout = read_timeout
        # Delegate to the request's own property rather than recomputing:
        # this used to be a second, subtly different formula (it omitted
        # archive_epoch and could diverge from what unit_results.py computed
        # for the same logical request).
        self.result_ref = request.result_ref

    async def run(self, work: Callable[[ArchiveStore], T]) -> T:
        return await execute_archive_read(
            self.archive_root,
            work,
            ctx=self.context,
            controller=self.controller,
            read_timeout=self.read_timeout,
        )

    def run_sync(self, work: Callable[[ArchiveStore], T]) -> T:
        return execute_archive_read_sync(
            self.archive_root,
            work,
            ctx=self.context,
            controller=self.controller,
            read_timeout=self.read_timeout,
        )


def _page_size(value: int | None) -> int:
    return max(1, int(value)) if value is not None else 1


async def run_archive_read(
    archive_root: Path,
    *,
    operation: str,
    arguments: Mapping[str, object],
    work: Callable[[ArchiveStore], T],
    page_size: int | None = None,
    offset: int = 0,
    projection: str = "default",
    stable_order: str = "canonical",
    workload_class: WorkloadClass = "interactive",
    admission_weight: int = 1,
    controller: QueryAdmissionController | None = None,
) -> T:
    """Run one named read through the shared transaction boundary."""
    transaction = QueryTransaction(
        archive_root,
        QueryTransactionRequest(
            operation=operation,
            arguments=dict(arguments),
            page_size=_page_size(page_size),
            offset=max(0, offset),
            projection=projection,
            stable_order=stable_order,
        ),
        workload_class=workload_class,
        admission_weight=admission_weight,
        controller=controller,
    )
    return await transaction.run(work)


def run_archive_read_sync(
    archive_root: Path,
    *,
    operation: str,
    arguments: Mapping[str, object],
    work: Callable[[ArchiveStore], T],
    page_size: int | None = None,
    offset: int = 0,
    projection: str = "default",
    stable_order: str = "canonical",
    workload_class: WorkloadClass = "interactive",
    admission_weight: int = 1,
    controller: QueryAdmissionController | None = None,
) -> T:
    """Run one named read through the shared transaction boundary synchronously."""
    transaction = QueryTransaction(
        archive_root,
        QueryTransactionRequest(
            operation=operation,
            arguments=dict(arguments),
            page_size=_page_size(page_size),
            offset=max(0, offset),
            projection=projection,
            stable_order=stable_order,
        ),
        workload_class=workload_class,
        admission_weight=admission_weight,
        controller=controller,
    )
    return transaction.run_sync(work)


@contextmanager
def archive_read_context(
    archive_root: Path,
    *,
    operation: str,
    arguments: Mapping[str, object],
    page_size: int | None = None,
    offset: int = 0,
    projection: str = "default",
    stable_order: str = "canonical",
    workload_class: WorkloadClass = "interactive",
    admission_weight: int = 1,
) -> Iterator[ArchiveStore]:
    """Expose one controlled store to callers already running in a worker thread."""
    transaction = QueryTransaction(
        archive_root,
        QueryTransactionRequest(
            operation=operation,
            arguments=dict(arguments),
            page_size=_page_size(page_size),
            offset=max(0, offset),
            projection=projection,
            stable_order=stable_order,
        ),
        workload_class=workload_class,
        admission_weight=admission_weight,
    )
    reader = InterruptibleSQLiteRead(transaction.context)
    with reader.open_context(archive_root) as archive:
        yield archive


__all__ = [
    "QueryContinuation",
    "QueryContinuationStaleError",
    "QueryResultPage",
    "QueryTransaction",
    "QueryTransactionRequest",
    "archive_index_epoch",
    "archive_read_context",
    "query_units_transaction_request",
    "run_archive_read",
    "run_archive_read_sync",
    "validate_continuation_epoch",
]
