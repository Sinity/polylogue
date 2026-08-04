"""Atomic-read contract for acquiring bytes from a source file.

polylogue-1fijp / polylogue-2t0vp: a live source file can be mid-rewrite when
an acquisition pass opens it -- a provider CLI truncating and rewriting a
session log in place, a sync client replacing a Drive cache file, etc. A
naive ``path.read_bytes()`` can observe a torn snapshot: some bytes from the
old generation, some from the new, with no error raised. Live evidence of
this shape: a Codex rollout file recaptured 15-16x with ``blob_size`` growing
then suddenly dropping from 427MB to 10KB in one read.

The contract: capture ``(size, mtime_ns)`` before reading, read to EOF, then
re-stat. If the post-read stat disagrees with the pre-read stat, the read
raced a concurrent writer and the bytes are unsafe to persist -- retry a
bounded number of times. If the file is still unstable (or vanishes) after
every retry, raise :class:`TornReadError` rather than ever returning partial
or torn bytes. Callers that want a non-raising "best effort" read use
:func:`try_read_source_bytes_atomic`, which turns the same failure into
``None``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AtomicReadResult:
    """A payload whose bytes are proven stable across the read window."""

    payload: bytes
    size: int
    mtime_ns: int


class TornReadError(RuntimeError):
    """Raised when a source file never stabilized across every retry.

    Distinct from a plain ``OSError``/``FileNotFoundError`` propagating from
    the final attempt (a genuinely vanished/unreadable source) -- this is
    raised specifically when reads kept succeeding but disagreed with each
    other, which is the concurrent-rewrite shape this contract exists to
    catch. ``attempts`` is always ``>= 1``.
    """

    def __init__(self, *, path: str, attempts: int, reason: str) -> None:
        self.path = path
        self.attempts = attempts
        self.reason = reason
        super().__init__(f"torn read at {path!r} after {attempts} attempt(s): {reason}")


def _stat_identity(path: Path) -> tuple[int, int]:
    stat_result = path.stat()
    return stat_result.st_size, stat_result.st_mtime_ns


def read_source_bytes_atomic(
    path: Path | str,
    *,
    max_retries: int = 3,
    retry_delay_s: float = 0.05,
) -> AtomicReadResult:
    """Read ``path`` and prove the bytes were not observed mid-rewrite.

    Raises :class:`TornReadError` if the file's ``(size, mtime_ns)`` identity
    never stabilizes across ``max_retries + 1`` attempts. Any read attempt
    that hits a genuine OS error (file vanished, permission denied) on its
    *final* attempt propagates that error directly rather than being wrapped
    -- only "kept changing under us" is reported as a torn read.

    ``max_retries`` must be ``>= 0`` (0 means a single attempt, no retry).
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    resolved_path = Path(path)
    last_reason = "file identity never observed"
    for attempt in range(max_retries + 1):
        is_final_attempt = attempt == max_retries
        try:
            before_size, before_mtime_ns = _stat_identity(resolved_path)
            payload = resolved_path.read_bytes()
            after_size, after_mtime_ns = _stat_identity(resolved_path)
        except OSError:
            if is_final_attempt:
                raise
            last_reason = "source became unreadable mid-read"
            time.sleep(retry_delay_s)
            continue
        if before_size == after_size and before_mtime_ns == after_mtime_ns and len(payload) == before_size:
            return AtomicReadResult(payload=payload, size=after_size, mtime_ns=after_mtime_ns)
        last_reason = (
            f"pre-read stat (size={before_size}, mtime_ns={before_mtime_ns}) disagreed with "
            f"post-read stat (size={after_size}, mtime_ns={after_mtime_ns}, read_len={len(payload)})"
        )
        if is_final_attempt:
            break
        time.sleep(retry_delay_s)
    raise TornReadError(path=str(resolved_path), attempts=max_retries + 1, reason=last_reason)


def try_read_source_bytes_atomic(
    path: Path | str,
    *,
    max_retries: int = 3,
    retry_delay_s: float = 0.05,
) -> AtomicReadResult | None:
    """Non-raising variant of :func:`read_source_bytes_atomic`.

    Returns ``None`` on a torn read or any OS-level read failure (vanished
    file, permission error) instead of raising -- for callers in the
    opportunistic re-acquire arm (polylogue-1fijp arm 5) that must fall
    through to a typed refusal rather than propagate an exception.
    """
    try:
        return read_source_bytes_atomic(path, max_retries=max_retries, retry_delay_s=retry_delay_s)
    except (TornReadError, OSError):
        return None


__all__ = [
    "AtomicReadResult",
    "TornReadError",
    "read_source_bytes_atomic",
    "try_read_source_bytes_atomic",
]
