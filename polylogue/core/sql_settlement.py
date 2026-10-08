"""Structural cleanup contracts for retained SQL handles."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Condition
from typing import Protocol

if "_native_sql_lifetimes" not in globals():
    _native_sql_lifetimes: ContextVar[tuple[object, ...]] = ContextVar("native_sql_artifact_lifetimes", default=())


def current_native_sql_lifetimes() -> tuple[object, ...]:
    """Dependencies explicitly captured by artifact SQL registrations."""
    return _native_sql_lifetimes.get()


@contextmanager
def retain_native_sql_lifetimes(*dependencies: object) -> Iterator[None]:
    """Carry existing scratch owners through captured compute and read contexts."""
    token = _native_sql_lifetimes.set((*current_native_sql_lifetimes(), *dependencies))
    try:
        yield
    finally:
        _native_sql_lifetimes.reset(token)


class SQLCustodyOwner(Protocol):
    def close(self) -> None: ...


class NativeSQLCustodyOwner(SQLCustodyOwner, Protocol):
    def retain_lifetime(self, dependency: object) -> None: ...


class AsyncSQLCustodyOwner(Protocol):
    def request_sql_settlement(self) -> None: ...

    async def close(self) -> None: ...


if "_native_sql_census" not in globals():
    _native_sql_census: Callable[[tuple[SQLCustodyOwner, ...] | None], tuple[SQLCustodyOwner, ...]] | None = None


def register_native_sql_census(
    census: Callable[[tuple[SQLCustodyOwner, ...] | None], tuple[SQLCustodyOwner, ...]],
) -> None:
    """Bind the existing native owner's census without importing its substrate."""
    global _native_sql_census
    if _native_sql_census is not None and _native_sql_census is not census:
        module = getattr(census, "__module__", None)
        name = getattr(census, "__qualname__", None)
        if (
            isinstance(module, str)
            and isinstance(name, str)
            and getattr(_native_sql_census, "__module__", None) == module
            and getattr(_native_sql_census, "__qualname__", None) == name
        ):
            # Reload retains the original callback and its owning module's
            # retained registry. It cannot replace that registry with emptiness.
            return
        raise RuntimeError("native SQL census already has its owning implementation")
    _native_sql_census = census


def retained_native_sql_owners() -> tuple[SQLCustodyOwner, ...]:
    """Enumerate cleanup on the creator thread after the native owner loads."""
    census = _native_sql_census
    return census(()) if census is not None else ()


def capture_native_sql_owners() -> tuple[SQLCustodyOwner, ...]:
    """Capture exact physical entry owners without transporting their handles."""
    census = _native_sql_census
    return census(None) if census is not None else ()


def _native_sql_settlement_owners(
    preserved_native_owners: tuple[SQLCustodyOwner, ...],
) -> tuple[SQLCustodyOwner, ...]:
    census = _native_sql_census
    return census(preserved_native_owners) if census is not None else ()


@dataclass(frozen=True, slots=True)
class NativeSQLSettlementEvidence:
    """Unresolved physical ownership on the native handle's creator worker."""

    owner_count: int
    failure_types: tuple[str, ...]


class SQLSettlementRetry:
    """Count explicit creator-worker retries without dropping a racing request."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._generation = 0

    def generation(self) -> int:
        with self._condition:
            return self._generation

    def request(self) -> None:
        with self._condition:
            self._generation += 1
            self._condition.notify_all()

    def wait_after(self, observed: int) -> int:
        with self._condition:
            self._condition.wait_for(lambda: self._generation > observed)
            return self._generation


def settle_native_sql(
    *,
    retry: SQLSettlementRetry,
    on_pending: Callable[[NativeSQLSettlementEvidence], None],
    on_settled: Callable[[], None],
    preserved_native_owners: tuple[SQLCustodyOwner, ...] = (),
    initial_observed_generation: int | None = None,
) -> BaseException | None:
    """Settle on the creator worker, retaining it on a failed native close.

    Runtime owners retain their existing physical future and admission until
    this returns. An explicit retry wakes the same worker; cancellation cannot
    surrender native cleanup to another thread. No elapsed budget changes the
    result, and the native owner census remains the authority for handles.
    """
    first_failure: BaseException | None = None
    # A runtime may bind its physical task mailbox before execution. Keep
    # requests delivered during that work available after the first failure;
    # nested cleanup carries the same task's consumed generation.
    observed = retry.generation() if initial_observed_generation is None else initial_observed_generation
    while True:
        failures: list[BaseException] = []
        try:
            owners = _native_sql_settlement_owners(preserved_native_owners)
        except BaseException as error:
            owners = ()
            first_failure = first_failure or error
            failures.append(error)
        for owner in owners:
            try:
                owner.close()
            except BaseException as error:
                first_failure = first_failure or error
                failures.append(error)
        try:
            owner_count = len(_native_sql_settlement_owners(preserved_native_owners))
        except BaseException as error:
            # A failed census cannot certify physical settlement.
            owner_count = -1
            first_failure = first_failure or error
            failures.append(error)
        if owner_count == 0:
            on_settled()
            return first_failure
        on_pending(NativeSQLSettlementEvidence(owner_count, tuple(type(error).__name__ for error in failures)))
        observed = retry.wait_after(observed)
