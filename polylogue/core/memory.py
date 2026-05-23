"""Process memory release helpers for long-running local daemons."""

from __future__ import annotations

import ctypes
import gc
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryReleaseResult:
    """Best-effort process memory release result."""

    collected_objects: int
    malloc_trim_called: bool


def release_process_memory() -> MemoryReleaseResult:
    """Release Python cycles and return free malloc arenas to the OS when possible.

    Long-running ingest daemons can transiently allocate large parser payloads,
    process-pool results, and SQLite buffers. CPython frees those objects, but
    glibc often keeps arenas resident unless asked to trim. This helper is
    deliberately best-effort: it is safe on non-glibc platforms and never turns
    memory reclamation failures into ingest failures.
    """

    collected = gc.collect()
    malloc_trim_called = False
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            malloc_trim = libc.malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            malloc_trim(0)
            malloc_trim_called = True
        except (AttributeError, OSError):
            malloc_trim_called = False
    return MemoryReleaseResult(collected_objects=collected, malloc_trim_called=malloc_trim_called)


__all__ = ["MemoryReleaseResult", "release_process_memory"]
