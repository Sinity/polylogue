"""Shared bounded read transaction identity and continuation state.

Execution control protects the host; this module protects the protocol
boundary. A page carries enough opaque state to advance the same logical
request, so clients never reconstruct filters from a prose overflow hint.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
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

T = TypeVar("T")
_TOKEN_VERSION = 1

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


@dataclass(frozen=True, slots=True)
class QueryTransactionRequest:
    """Canonical request identity shared by page producers and adapters."""

    operation: str
    arguments: Mapping[str, object]
    page_size: int
    offset: int = 0
    projection: str = "default"
    stable_order: str = "canonical"

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

    def next(self, *, offset: int) -> QueryTransactionRequest:
        return QueryTransactionRequest(
            operation=self.operation,
            arguments=self.arguments,
            page_size=self.page_size,
            offset=offset,
            projection=self.projection,
            stable_order=self.stable_order,
        )


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
            )
            result_ref = str(body["result_ref"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
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
        self.result_ref = (
            "result:"
            + hashlib.sha256(f"{request.query_ref}:{request.projection}:{request.stable_order}".encode()).hexdigest()[
                :24
            ]
        )

    async def run(self, work: Callable[[ArchiveStore], T]) -> T:
        return await execute_archive_read(
            self.archive_root,
            work,
            ctx=self.context,
            controller=self.controller,
        )

    def run_sync(self, work: Callable[[ArchiveStore], T]) -> T:
        return execute_archive_read_sync(
            self.archive_root,
            work,
            ctx=self.context,
            controller=self.controller,
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
    "QueryResultPage",
    "QueryTransaction",
    "QueryTransactionRequest",
    "archive_read_context",
    "run_archive_read",
    "run_archive_read_sync",
]
