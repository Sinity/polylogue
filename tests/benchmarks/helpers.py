from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Connection
from typing import Protocol, TypeVar

from polylogue.storage.repository import ConversationRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection

T = TypeVar("T")


class BenchmarkFixture(Protocol):
    def __call__(self, func: Callable[[], T]) -> T: ...


@dataclass
class BenchAsyncStore:
    loop: asyncio.AbstractEventLoop
    backend: SQLiteBackend
    repository: ConversationRepository

    def run(self, awaitable: Awaitable[T]) -> T:
        return self.loop.run_until_complete(awaitable)


@contextmanager
def open_bench_store(db_path: Path) -> Iterator[BenchAsyncStore]:
    """Open a benchmark backend/repository pair without touching private APIs."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)
    store = BenchAsyncStore(loop=loop, backend=backend, repository=repository)
    try:
        yield store
    finally:
        loop.run_until_complete(backend.close())
        loop.close()


def benchmark_store_call(
    benchmark: BenchmarkFixture,
    db_path: Path,
    operation: Callable[[BenchAsyncStore], Awaitable[T]],
) -> None:
    """Benchmark one async repository/backend operation against a seeded DB."""
    with open_bench_store(db_path) as store:
        benchmark(lambda: store.run(operation(store)))


def benchmark_connection_call(
    benchmark: BenchmarkFixture,
    db_path: Path,
    operation: Callable[[Connection], T],
) -> None:
    """Benchmark one sync sqlite/index operation against a seeded DB."""
    with open_connection(db_path) as conn:
        benchmark(lambda: operation(conn))
