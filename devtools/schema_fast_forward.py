"""Generic, proof-gated SQLite schema fast-forward engine.

Schema transitions are data supplied to this module.  The engine owns the
common safety sequence: select a contiguous version path, apply one step in an
atomic transaction, prove its postcondition, and advance ``user_version``
last.  A new transition therefore adds a step declaration; it does not need a
version-named actuator or a copy of the fast-forward machinery.

This module intentionally does not know about any archive tier.  Durable
tiers can provide their migration steps and derived tiers can provide
clone-safe steps without introducing another per-version route.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final


class SchemaFastForwardError(RuntimeError):
    """Raised when a schema transition cannot be proven safe to apply."""


ApplyStep = Callable[[sqlite3.Connection], None]
VerifyStep = Callable[[sqlite3.Connection], None]


def _noop(_: sqlite3.Connection) -> None:
    return None


@dataclass(frozen=True, slots=True)
class SchemaFastForwardStep:
    """One version transition understood by :class:`SchemaFastForwardEngine`.

    ``apply`` may be a SQL string or a callback for a structural rebuild.  A
    callback must leave the connection transaction open; the engine commits
    only after ``verify`` and the version stamp succeed.
    """

    version: int
    apply: str | ApplyStep
    verify: VerifyStep = _noop
    name: str = ""

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("schema fast-forward versions must be positive")
        if not isinstance(self.apply, str) and not callable(self.apply):
            raise TypeError("schema fast-forward step apply must be SQL or a callable")
        if not callable(self.verify):
            raise TypeError("schema fast-forward step verify must be callable")

    @property
    def label(self) -> str:
        return self.name or f"v{self.version}"


@dataclass(frozen=True, slots=True)
class SchemaFastForwardPlan:
    """A validated contiguous path from the connection's current version."""

    tier: str
    source_version: int
    target_version: int
    steps: tuple[SchemaFastForwardStep, ...]


@dataclass(frozen=True, slots=True)
class SchemaFastForwardResult:
    """Receipt-like result for one engine execution."""

    tier: str
    source_version: int
    target_version: int
    applied_versions: tuple[int, ...]

    @property
    def changed(self) -> bool:
        return bool(self.applied_versions)


def _execute_sql(conn: sqlite3.Connection, sql: str, *, label: str) -> None:
    """Execute SQL without allowing it to escape the engine transaction."""
    statement = ""
    for line in sql.splitlines(keepends=True):
        statement += line
        if not sqlite3.complete_statement(statement):
            continue
        if statement.strip():
            try:
                conn.execute(statement)
            except sqlite3.DatabaseError as exc:
                raise SchemaFastForwardError(f"{label} failed: {exc}") from exc
        statement = ""
    if statement.strip():
        # sqlite3.complete_statement requires a semicolon, while callers often
        # provide one ordinary DDL statement without one.  Let SQLite parse
        # that final statement; it still rejects multi-statement input.
        try:
            conn.execute(statement)
        except sqlite3.DatabaseError as exc:
            raise SchemaFastForwardError(f"{label} contains an incomplete SQL statement: {exc}") from exc


class SchemaFastForwardEngine:
    """Apply declared schema steps across any contiguous version gap.

    The constructor validates declarations once.  ``plan`` derives the actual
    path from the database, so a seeded archive at any supported version uses
    the same engine and retrying an already-applied prefix is a no-op.
    """

    def __init__(
        self,
        steps: Iterable[SchemaFastForwardStep],
        *,
        tier: str = "archive",
        target_version: int | None = None,
    ) -> None:
        normalized: list[SchemaFastForwardStep] = []
        for item in steps:
            if isinstance(item, SchemaFastForwardStep):
                normalized.append(item)
                continue
            # Keeping tuple declarations accepted makes transition tables easy
            # to keep next to a DDL manifest without introducing wrapper code.
            try:
                version, apply = item
            except (TypeError, ValueError) as exc:
                raise TypeError("schema fast-forward declarations must be steps or (version, apply) pairs") from exc
            normalized.append(SchemaFastForwardStep(int(version), apply))
        ordered = tuple(sorted(normalized, key=lambda step: step.version))
        versions = tuple(step.version for step in ordered)
        if len(set(versions)) != len(versions):
            raise ValueError("schema fast-forward declarations contain duplicate versions")
        if not ordered and target_version is not None:
            raise ValueError("an empty schema fast-forward declaration cannot have a target")
        if target_version is not None and target_version < 0:
            raise ValueError("schema fast-forward target must not be negative")
        if ordered and target_version is not None and target_version < ordered[-1].version:
            raise ValueError("schema fast-forward target precedes the declared head")
        self._steps: Mapping[int, SchemaFastForwardStep] = {step.version: step for step in ordered}
        self.tier = tier
        self.target_version = target_version if target_version is not None else (ordered[-1].version if ordered else 0)

    @property
    def steps(self) -> tuple[SchemaFastForwardStep, ...]:
        return tuple(self._steps.values())

    def plan(self, conn: sqlite3.Connection, *, target_version: int | None = None) -> SchemaFastForwardPlan:
        current = _read_user_version(conn)
        target = self.target_version if target_version is None else target_version
        if target < current:
            raise SchemaFastForwardError(
                f"{self.tier} schema version {current} is newer than requested target {target}"
            )
        if target > self.target_version:
            raise SchemaFastForwardError(
                f"{self.tier} schema target {target} is newer than the declared engine head {self.target_version}"
            )
        steps: list[SchemaFastForwardStep] = []
        for version in range(current + 1, target + 1):
            try:
                steps.append(self._steps[version])
            except KeyError as exc:
                raise SchemaFastForwardError(
                    f"{self.tier} schema fast-forward path is incomplete at v{version} (v{current}->v{target})"
                ) from exc
        return SchemaFastForwardPlan(self.tier, current, target, tuple(steps))

    def execute(self, conn: sqlite3.Connection, *, target_version: int | None = None) -> SchemaFastForwardResult:
        plan = self.plan(conn, target_version=target_version)
        applied: list[int] = []
        for step in plan.steps:
            self._execute_step(conn, step)
            applied.append(step.version)
        # The plan has no steps when the archive is already at target.  This is
        # deliberately a successful idempotent result rather than an error.
        if _read_user_version(conn) != plan.target_version:
            raise SchemaFastForwardError(
                f"{self.tier} schema fast-forward ended at v{_read_user_version(conn)}, expected v{plan.target_version}"
            )
        return SchemaFastForwardResult(plan.tier, plan.source_version, plan.target_version, tuple(applied))

    def _execute_step(self, conn: sqlite3.Connection, step: SchemaFastForwardStep) -> None:
        # Each step commits independently.  A crash or failed later step leaves
        # a valid prefix which the next invocation plans from, never a stamped
        # version whose operation was only partially applied.
        savepoint = f"schema_fast_forward_{step.version}"
        own_transaction = not conn.in_transaction
        if own_transaction:
            conn.execute("BEGIN IMMEDIATE")
        else:
            conn.execute(f'SAVEPOINT "{savepoint}"')
        try:
            if isinstance(step.apply, str):
                _execute_sql(conn, step.apply, label=step.label)
            else:
                step.apply(conn)
            if not conn.in_transaction:
                raise SchemaFastForwardError(f"{step.label} escaped the engine transaction")
            step.verify(conn)
            if not conn.in_transaction:
                raise SchemaFastForwardError(f"{step.label} verification escaped the engine transaction")
            conn.execute(f"PRAGMA user_version = {step.version}")
            if _read_user_version(conn) != step.version:
                raise SchemaFastForwardError(f"{step.label} did not set user_version to v{step.version}")
            if own_transaction:
                conn.commit()
            else:
                conn.execute(f'RELEASE SAVEPOINT "{savepoint}"')
        except Exception as exc:
            if own_transaction:
                if conn.in_transaction:
                    conn.rollback()
            else:
                conn.execute(f'ROLLBACK TO SAVEPOINT "{savepoint}"')
                conn.execute(f'RELEASE SAVEPOINT "{savepoint}"')
            if isinstance(exc, SchemaFastForwardError):
                raise
            raise SchemaFastForwardError(f"{step.label} failed: {exc}") from exc


def fast_forward_schema(
    conn: sqlite3.Connection,
    steps: Iterable[SchemaFastForwardStep],
    *,
    tier: str = "archive",
    target_version: int | None = None,
) -> SchemaFastForwardResult:
    """Convenience entry point for callers that do not need an engine object."""
    return SchemaFastForwardEngine(steps, tier=tier, target_version=target_version).execute(conn)


def _read_user_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
    except sqlite3.DatabaseError as exc:
        raise SchemaFastForwardError(f"cannot read schema version: {exc}") from exc
    if row is None:
        raise SchemaFastForwardError("SQLite did not return a schema version")
    version = int(row[0])
    if version < 0:
        raise SchemaFastForwardError(f"SQLite schema version is negative: {version}")
    return version


__all__: Final = [
    "SchemaFastForwardEngine",
    "SchemaFastForwardError",
    "SchemaFastForwardPlan",
    "SchemaFastForwardResult",
    "SchemaFastForwardStep",
    "fast_forward_schema",
]
