"""Physically settle an original iterator before releasing its input lifetime."""

from __future__ import annotations

import sys
from builtins import BaseExceptionGroup
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Protocol, TypeVar, runtime_checkable

_T = TypeVar("_T")


@runtime_checkable
class _ClosableIterator(Protocol):
    def close(self) -> None: ...


@contextmanager
def settled_iterator(items: Iterable[_T]) -> Iterator[Iterator[_T]]:
    iterator = iter(items)
    try:
        yield iterator
    finally:
        primary = sys.exception()
        try:
            if isinstance(iterator, _ClosableIterator):
                try:
                    iterator.close()
                except BaseException as cleanup:
                    if primary is not None:
                        raise BaseExceptionGroup(
                            "iteration and original iterator close failed", [primary, cleanup]
                        ) from primary
                    raise
        finally:
            primary = None
