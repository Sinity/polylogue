from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Connection
from typing import Any, Protocol, TypeVar, cast

from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection

T = TypeVar("T")


def _index_db_path(db_path: Path) -> Path:
    return db_path if db_path.name == "index.db" else db_path.parent / "index.db"


class BenchmarkFixture(Protocol):
    def __call__(self, func: Callable[[], T]) -> T: ...


def benchmark_one_shot(benchmark: Any, operation: Callable[..., T], *args: object) -> T:
    """Measure one mutating end-to-end operation against one fresh state.

    One round is the right shape only when repeating the operation would not
    measure the same thing -- a mutating lane whose second round runs against
    the state the first one left. It is the wrong shape for a repeatable read:
    with ``rounds=1`` pytest-benchmark reports ``stddev == 0``, and every p95
    estimator downstream (``devtools/verify_slos.py`` estimates
    ``mean + 1.645 * stddev``) then returns the single observed sample wearing
    a p95's name. Repeatable lanes use :func:`benchmark_repeated`.
    """
    return cast(T, benchmark.pedantic(operation, args=args, rounds=1, iterations=1))


#: Rounds for a repeatable lane. Five is what ``test_bench_cli_status_cold``
#: already takes, and it is the smallest count at which the catalog's
#: ``mean + 1.645 * stddev`` estimate is computed from an actual spread rather
#: than from a constant zero.
REPEATED_BENCHMARK_ROUNDS = 5


def benchmark_repeated(
    benchmark: Any,
    operation: Callable[..., T],
    *args: object,
    rounds: int = REPEATED_BENCHMARK_ROUNDS,
) -> T:
    """Measure a repeatable operation often enough to have a distribution.

    A latency budget is a claim about a distribution. A lane that observes one
    sample can report that sample honestly and can report nothing about a tail,
    so any surface it feeds must either take more rounds or decline to declare
    a p95. This is the "take more rounds" side; the returned value is the last
    round's, matching :func:`benchmark_one_shot`'s contract.
    """
    if rounds < 2:
        raise ValueError("a repeated benchmark needs at least two rounds to have a spread")
    return cast(T, benchmark.pedantic(operation, args=args, rounds=rounds, iterations=1))


@dataclass
class BenchAsyncStore:
    loop: asyncio.AbstractEventLoop
    backend: SQLiteBackend
    repository: SessionRepository

    def run(self, awaitable: Awaitable[T]) -> T:
        return self.loop.run_until_complete(awaitable)


@contextmanager
def open_bench_store(db_path: Path) -> Iterator[BenchAsyncStore]:
    """Open a benchmark backend/repository pair without touching private APIs."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=_index_db_path(db_path))
    repository = SessionRepository(backend=backend)
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
    with open_connection(_index_db_path(db_path)) as conn:
        benchmark(lambda: operation(conn))
