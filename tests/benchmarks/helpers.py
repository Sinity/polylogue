from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository


@dataclass
class BenchAsyncStore:
    loop: asyncio.AbstractEventLoop
    backend: SQLiteBackend
    repository: ConversationRepository

    def run(self, awaitable):
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
