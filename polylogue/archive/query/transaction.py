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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

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

# Shared protocol vocabulary for describing what a bounded query result does and
# does not cover. Discovery surfaces import these aliases so coverage and total
# claims use the same words as transaction/page contracts.
QueryCoverageClass = Literal[
    "exhaustive",
    "top-k",
    "sample",
    "aggregate",
    "bounded-context",
    "recursive-page",
]
QueryTotalSemantics = Literal["exact", "qualified", "aggregate", "not-applicable"]
QueryContinuationSemantics = Literal[
    "none",
    "cursor-or-offset",
    "ranked-frontier",
    "recursive-cursor",
]


@dataclass(frozen=True, slots=True)
class QueryResultSemanticsContract:
    """Truthful coverage, total, and continuation claims for one result class."""

    coverage: QueryCoverageClass
    total: QueryTotalSemantics
    continuation: QueryContinuationSemantics
    phrase: str


# V2 binds a page to the exact index+user-tier frame that supplied its rows.
# V1 remains decode-only for continuations issued before frame validation.
_TOKEN_VERSION = 2
_LEGACY_TOKEN_VERSION = 1

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class QueryArchiveEpochUnreadableError(RuntimeError):
    """The reader could not establish the archive frame for a page."""

    code = "archive_read_unavailable"


def archive_snapshot_epoch(archive: ArchiveStore) -> str:
    """Return the query-unit frame from the reader's active SQLite snapshot.

    The index and user tiers each advance a durable, trigger-maintained epoch
    only for relations that can change terminal query-unit rows or their
    session/tag scope.  Reading both components through ``archive._conn``
    after ``begin_read_snapshot()`` binds the continuation to the exact
    snapshots that will supply the result rows; it never races a second
    ``index.db`` probe against the writer.
    """
    try:
        conn = archive._conn
        index_version = int(conn.execute("PRAGMA main.user_version").fetchone()[0])
        index_epoch = int(conn.execute("SELECT epoch FROM query_unit_frame_state WHERE singleton = 1").fetchone()[0])
        user_version = int(conn.execute("PRAGMA user_tier.user_version").fetchone()[0])
        user_epoch = int(
            conn.execute("SELECT epoch FROM user_tier.query_unit_frame_state WHERE singleton = 1").fetchone()[0]
        )
    except (AttributeError, IndexError, sqlite3.Error, TypeError) as exc:
        logger.warning("query transaction: could not read archive snapshot epoch", exc_info=True)
        raise QueryArchiveEpochUnreadableError("could not establish archive frame for query continuation") from exc
    return f"archive:v1:index:v{index_version}:{index_epoch}:user:v{user_version}:{user_epoch}"


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
    continuation_version: int | None = field(default=None, compare=False)

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
            continuation_version=self.continuation_version,
        )

    def with_archive_epoch(self, archive_epoch: str) -> QueryTransactionRequest:
        """Bind this canonical request to the snapshot that produced its rows."""
        return QueryTransactionRequest(
            operation=self.operation,
            arguments=self.arguments,
            page_size=self.page_size,
            offset=self.offset,
            projection=self.projection,
            stable_order=self.stable_order,
            archive_epoch=archive_epoch,
            continuation_version=self.continuation_version,
        )


def query_units_transaction_request(
    *,
    expression: str,
    session_filters: Mapping[str, object],
    page_size: int,
    offset: int = 0,
) -> QueryTransactionRequest:
    """Build the canonical, unframed ``query_units`` transaction request.

    The single constructor shared by the API, MCP, and daemon HTTP adapters so
    the three surfaces cannot drift into building the request differently (the
    #2472/#2470 partitioning-bug shape) and so archive-epoch binding lands in
    one place rather than three independently-maintained call sites.  The
    frame is intentionally absent here: :func:`query_unit_envelope` captures
    it only after the controlled reader has opened its SQLite snapshot.
    """
    return QueryTransactionRequest(
        operation="query_units",
        arguments={"expression": expression, "session_filters": dict(session_filters)},
        page_size=page_size,
        offset=max(0, offset),
        projection="terminal-unit-envelope",
        stable_order="canonical",
    )


def validate_continuation_epoch(continuation_request: QueryTransactionRequest, *, archive: ArchiveStore) -> str:
    """Validate one decoded continuation in the same snapshot as its rows."""
    if not continuation_request.archive_epoch:
        if continuation_request.continuation_version == _LEGACY_TOKEN_VERSION:
            return archive_snapshot_epoch(archive)
        raise ValueError("query continuation is missing required archive_epoch")
    current = archive_snapshot_epoch(archive)
    if continuation_request.archive_epoch != current:
        raise QueryContinuationStaleError(issued_epoch=continuation_request.archive_epoch, current_epoch=current)
    return current


@dataclass(frozen=True, slots=True)
class QueryContinuation:
    """Opaque, self-validating continuation for one advancing page."""

    request: QueryTransactionRequest
    result_ref: str

    def encode(self) -> str:
        request = self.request
        if not request.archive_epoch:
            raise ValueError("new query continuations require an archive_epoch")
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
        return "q2." + encoded

    @classmethod
    def decode(cls, token: str) -> QueryContinuation:
        if not token.startswith(("q1.", "q2.")):
            raise ValueError("invalid query continuation version")
        try:
            padded = token[3:] + "=" * (-len(token[3:]) % 4)
            body = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
            request_body = body["request"]
            version = int(body["v"])
            expected_prefix = f"q{version}."
            if version not in {_LEGACY_TOKEN_VERSION, _TOKEN_VERSION} or not token.startswith(expected_prefix):
                raise ValueError("unsupported query continuation version")
            archive_epoch = str(request_body.get("archive_epoch") or "")
            if version == _TOKEN_VERSION and not archive_epoch:
                raise ValueError("query continuation is missing required archive_epoch")
            request = QueryTransactionRequest(
                operation=str(request_body["operation"]),
                arguments=dict(request_body["arguments"]),
                page_size=int(request_body["page_size"]),
                offset=int(request_body["offset"]),
                projection=str(request_body["projection"]),
                stable_order=str(request_body["stable_order"]),
                archive_epoch=archive_epoch,
                continuation_version=version,
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
    "QueryArchiveEpochUnreadableError",
    "QueryContinuationStaleError",
    "QueryContinuationSemantics",
    "QueryCoverageClass",
    "QueryResultPage",
    "QueryResultSemanticsContract",
    "QueryTotalSemantics",
    "QueryTransaction",
    "QueryTransactionRequest",
    "archive_snapshot_epoch",
    "archive_read_context",
    "query_units_transaction_request",
    "run_archive_read",
    "run_archive_read_sync",
    "validate_continuation_epoch",
]
