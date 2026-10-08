"""Count actual outside-receipt bytes read by the isolated source owner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from polylogue.sources import sqlite_export

receipt_name = os.environ["POLYLOGUE_TEST_RECEIPT_NAME"]
accounting = Path(os.environ["POLYLOGUE_TEST_RECEIPT_ACCOUNTING"])
real_open = os.open
real_fdopen = os.fdopen
receipt_descriptors: set[int] = set()


def tracked_open(path: Any, *args: Any, **kwargs: Any) -> int:
    descriptor = real_open(path, *args, **kwargs)
    if os.fsdecode(path) == receipt_name:
        receipt_descriptors.add(descriptor)
    return descriptor


class CountedReceipt:
    def __init__(self, stream: Any) -> None:
        self.stream = stream

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)

    def __enter__(self) -> CountedReceipt:
        self.stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self.stream.__exit__(*args)

    def counted(self, payload: bytes) -> bytes:
        with accounting.open("a", encoding="ascii") as output:
            output.write(f"{len(payload)}\n")
        return payload

    def read(self, size: int = -1) -> bytes:
        return self.counted(self.stream.read(size))

    def readline(self, size: int = -1) -> bytes:
        return self.counted(self.stream.readline(size))

    def __iter__(self) -> CountedReceipt:
        return self

    def __next__(self) -> bytes:
        payload = self.readline()
        if not payload:
            raise StopIteration
        return payload


def tracked_fdopen(descriptor: int, *args: Any, **kwargs: Any) -> Any:
    stream = real_fdopen(descriptor, *args, **kwargs)
    if descriptor not in receipt_descriptors:
        return stream
    receipt_descriptors.remove(descriptor)
    return CountedReceipt(stream)


with patch("os.open", tracked_open), patch("os.fdopen", tracked_fdopen):
    sqlite_export._source_worker_main()
