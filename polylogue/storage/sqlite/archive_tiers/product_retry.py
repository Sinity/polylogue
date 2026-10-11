"""Typed retry transitions on the existing convergence-debt owner.

Writer module: ops.

Callers supply the admitted original Ops connection and clock. Observations
never count attempts; only an exact admitted identity may settle a result.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from polylogue.core.product_retry import ProductAttempt, ProductObservation, ProductOutcome


def ops_retry_incarnation(conn: sqlite3.Connection) -> str:
    incarnation = str(conn.execute("SELECT incarnation FROM product_retry_incarnation WHERE singleton=1").fetchone()[0])
    path = next(str(row[2]) for row in conn.execute("PRAGMA database_list") if row[1] == "main")
    if path:
        identity = Path(path).stat()
        return f"{incarnation}:{identity.st_dev}:{identity.st_ino}"
    return incarnation


def observe_product_retry(
    conn: sqlite3.Connection, *, product: str, target: str, observation: ProductObservation, now_ms: int
) -> None:
    if not observation.needs_work:
        conn.execute(
            "DELETE FROM convergence_debt WHERE stage=? AND target_type='archive' AND target_id=?",
            (product, target),
        )
        return
    conn.execute(
        "INSERT INTO convergence_debt(debt_id,stage,target_type,target_id,status,attempts,"
        "created_at_ms,updated_at_ms,demand_identity,observation_count,last_outcome) "
        "VALUES(?,?,'archive',?,'deferred',0,?,?,?,1,'blocked') "
        "ON CONFLICT(stage,target_type,target_id) DO UPDATE SET demand_identity=excluded.demand_identity,"
        "observation_count=convergence_debt.observation_count+1,updated_at_ms=excluded.updated_at_ms",
        (str(uuid.uuid4()), product, target, now_ms, now_ms, observation.demand_identity),
    )


def admit_product_retry(
    conn: sqlite3.Connection,
    *,
    product: str,
    target: str,
    demand_identity: str,
    owner: str,
    now_ms: int,
    active_attempts: frozenset[str] = frozenset(),
) -> ProductAttempt | None:
    row = conn.execute(
        "SELECT demand_identity,admitted_attempt_id,attempt_owner,next_retry_at FROM convergence_debt "
        "WHERE stage=? AND target_type='archive' AND target_id=?",
        (product, target),
    ).fetchone()
    if row is None or row[0] != demand_identity:
        return None
    if row[1] is not None and row[2] == owner and row[1] in active_attempts:
        return None
    if row[3] is not None and datetime.fromisoformat(str(row[3])).timestamp() * 1000 > now_ms:
        return None
    attempt = ProductAttempt(product, target, ops_retry_incarnation(conn), demand_identity, str(uuid.uuid4()))
    conn.execute(
        "UPDATE convergence_debt SET admitted_attempt_id=?,attempt_owner=?,attempts=attempts+1,"
        "updated_at_ms=? WHERE stage=? AND target_type='archive' AND target_id=? AND demand_identity=?",
        (attempt.attempt_id, owner, now_ms, product, target, demand_identity),
    )
    return attempt


def product_attempt_current(conn: sqlite3.Connection, attempt: ProductAttempt) -> bool:
    if ops_retry_incarnation(conn) != attempt.ops_incarnation:
        return False
    return (
        conn.execute(
            "SELECT 1 FROM convergence_debt WHERE stage=? AND target_type='archive' AND target_id=? "
            "AND demand_identity=? AND admitted_attempt_id=? AND (settled_attempt_id IS NULL OR settled_attempt_id!=?)",
            (attempt.product, attempt.target, attempt.demand_identity, attempt.attempt_id, attempt.attempt_id),
        ).fetchone()
        is not None
    )


def settle_product_retry(
    conn: sqlite3.Connection, attempt: ProductAttempt, outcome: ProductOutcome, *, now_ms: int, error: str | None = None
) -> bool:
    if not product_attempt_current(conn, attempt):
        return False
    if outcome in (ProductOutcome.CURRENT, ProductOutcome.VALID_EMPTY):
        conn.execute(
            "DELETE FROM convergence_debt WHERE stage=? AND target_type='archive' AND target_id=?",
            (attempt.product, attempt.target),
        )
        return True
    row = conn.execute(
        "SELECT failure_streak,next_retry_at FROM convergence_debt WHERE stage=? AND target_type='archive' AND target_id=?",
        (attempt.product, attempt.target),
    ).fetchone()
    streak = int(row[0]) + int(outcome is ProductOutcome.FAILED)
    retry_at = row[1]
    if outcome is ProductOutcome.FAILED:
        # Bound the scheduling delay, never valid work or its running time.
        delay_s = 60 * 2 ** min(max(streak - 1, 0), 6)
        retry_at = datetime.fromtimestamp((now_ms + min(delay_s, 3600) * 1000) / 1000, tz=UTC).isoformat()
    conn.execute(
        "UPDATE convergence_debt SET admitted_attempt_id=NULL,settled_attempt_id=?,failure_streak=?,"
        "status=?,last_outcome=?,last_error=?,next_retry_at=?,updated_at_ms=? "
        "WHERE stage=? AND target_type='archive' AND target_id=?",
        (
            attempt.attempt_id,
            streak,
            "failed" if outcome is ProductOutcome.FAILED else "deferred",
            outcome.value,
            error,
            retry_at,
            now_ms,
            attempt.product,
            attempt.target,
        ),
    )
    return True
