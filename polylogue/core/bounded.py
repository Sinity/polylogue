"""Small, typed guards for calls that cross a process or event-loop boundary.

Callers must provide a positive wall-clock budget.  Keeping the policy here
means timeout handling is uniform and makes an omitted budget easy to find in
code review and by the AST ratchet.
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from typing import IO, Any, TypeAlias, TypedDict, TypeVar, Unpack

T = TypeVar("T")


class BoundedTimeoutError(TimeoutError):
    """A cross-boundary operation exceeded its declared wall-clock budget."""

    def __init__(self, budget: float, *, operation: object | None = None) -> None:
        self.budget = budget
        self.operation = operation
        detail = f"operation exceeded {budget:g}s budget"
        if operation is not None:
            detail = f"{operation!r} {detail}"
        super().__init__(detail)


TimeoutHandler: TypeAlias = type[BaseException] | BaseException | Callable[[float], BaseException]


class _RunOptions(TypedDict, total=False):
    """The caller-controlled options accepted by ``subprocess.run``."""

    bufsize: int
    executable: str | bytes | os.PathLike[str] | os.PathLike[bytes] | None
    stdin: int | IO[Any] | None
    stdout: int | IO[Any] | None
    stderr: int | IO[Any] | None
    preexec_fn: Callable[[], object] | None
    close_fds: bool
    shell: bool
    cwd: str | bytes | os.PathLike[str] | os.PathLike[bytes] | None
    env: (
        Mapping[str, str | bytes | os.PathLike[str] | os.PathLike[bytes]]
        | Mapping[bytes, str | bytes | os.PathLike[str] | os.PathLike[bytes]]
        | None
    )
    universal_newlines: bool | None
    startupinfo: Any
    creationflags: int
    restore_signals: bool
    start_new_session: bool
    pass_fds: Collection[int]
    capture_output: bool
    encoding: str | None
    errors: str | None
    input: str | bytes | bytearray | memoryview | None
    text: bool | None
    user: str | int | None
    group: str | int | None
    extra_groups: Iterable[str | int] | None
    umask: int
    pipesize: int
    process_group: int | None


def _validate_budget(budget: float) -> float:
    if isinstance(budget, bool) or not isinstance(budget, (int, float)):
        raise TypeError("budget must be a positive finite number of seconds")
    value = float(budget)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("budget must be a positive finite number of seconds")
    return value


def run_bounded(
    argv: Sequence[str],
    budget: float,
    *,
    check_exit: bool = True,
    **kwargs: Unpack[_RunOptions],
) -> subprocess.CompletedProcess[str | bytes]:
    """Run *argv* with a required timeout and checked exit status by default.

    ``kwargs`` are the ordinary ``subprocess.run`` options (for example
    ``cwd``, ``env``, and ``capture_output``).  ``timeout`` and ``check`` are
    intentionally owned by this helper and cannot silently override the law.
    """
    seconds = _validate_budget(budget)
    if "timeout" in kwargs or "check" in kwargs:
        raise TypeError("run_bounded owns timeout and check; use budget and check_exit")
    try:
        result: subprocess.CompletedProcess[str | bytes] = subprocess.run(
            [str(argument) for argument in argv],
            timeout=seconds,
            check=check_exit,
            **kwargs,
        )
        return result
    except subprocess.TimeoutExpired as exc:
        raise BoundedTimeoutError(seconds, operation=argv) from exc


def _timeout_exception(handler: TimeoutHandler | None, budget: float) -> BaseException:
    if handler is None:
        return BoundedTimeoutError(budget)
    if isinstance(handler, BaseException):
        return handler
    if isinstance(handler, type):
        if not issubclass(handler, BaseException):
            raise TypeError("on_timeout class must derive from BaseException")
        exception_type: type[BaseException] = handler
        return exception_type()
    return handler(budget)


async def bounded(
    awaitable: Awaitable[T],
    budget: float,
    *,
    on_timeout: TimeoutHandler | None = None,
) -> T:
    """Await *awaitable* for at most *budget* seconds.

    The task is cancelled by ``asyncio.timeout`` and the cancellation is
    translated to a typed exception.  ``on_timeout`` may be an exception
    class, an exception instance, or a factory receiving the budget.
    """
    seconds = _validate_budget(budget)
    try:
        async with asyncio.timeout(seconds):
            return await awaitable
    except TimeoutError as exc:
        raise _timeout_exception(on_timeout, seconds) from exc


__all__ = ["BoundedTimeoutError", "bounded", "run_bounded"]
