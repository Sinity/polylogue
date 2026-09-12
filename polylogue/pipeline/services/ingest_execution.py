"""Optional execution boundary for staged acquisition and ingest publication."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

T = TypeVar("T")


class IngestExecution(Protocol):
    """The caller owns compute settlement and each bounded writer admission."""

    async def prepare(self, operation: Callable[[], T]) -> T: ...

    async def settle(self, pending: Awaitable[T]) -> T: ...

    async def publish(self, actor: str, operation: Callable[[], Awaitable[T]]) -> T: ...

    async def publish_sync(self, actor: str, operation: Callable[[], T]) -> T: ...
