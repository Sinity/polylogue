"""Process-local proof that every active index raw reference exists in source.db.

The proof is deliberately narrower than raw frontier integrity: predecessor
chains and cursors are checked afresh for the selected source component.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from threading import RLock

from polylogue.core.evidence import Measured, Unavailable
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.schema_identity import DerivedTier, derived_schema_identity
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import attach_readonly_database, open_readonly_connection
from polylogue.storage.tier_access import capture_sqlite_read


@dataclass(frozen=True)
class _Certificate:
    identity: tuple[object, ...]
    source_watermark: int
    index_watermark: int


_lock = RLock()
_certificates: dict[Path, _Certificate] = {}
_pid = os.getpid()
_TRIGGER_PATTERN = re.compile(
    r"CREATE TRIGGER IF NOT EXISTS (?P<create>raw_(?:existence|frontier_cursor)_\w+)\s.*?END;"
    r"|DROP TRIGGER IF EXISTS (?P<drop>raw_(?:existence|frontier_cursor)_\w+)\s*;",
    re.S,
)


def _reset_after_fork() -> None:
    global _lock, _certificates, _pid
    # The parent may have forked while another thread held its RLock. The
    # child has no such thread, so acquiring that inherited lock would hang.
    _lock = RLock()
    _certificates = {}
    _pid = os.getpid()


os.register_at_fork(after_in_child=_reset_after_fork)


def _normalized_sql(sql: str) -> str:
    return " ".join(re.sub(r"\bIF NOT EXISTS\b", "", sql, flags=re.I).lower().rstrip("; ").split())


def _expected_triggers(ddl: str) -> dict[str, str]:
    triggers: dict[str, str] = {}
    for match in _TRIGGER_PATTERN.finditer(ddl):
        if name := match.group("drop"):
            triggers.pop(name, None)
        else:
            triggers[match.group("create")] = _normalized_sql(match.group(0))
    return triggers


_SOURCE_TRIGGERS = _expected_triggers(ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE])
_INDEX_TRIGGERS = _expected_triggers(INDEX_DDL)


@cache
def _expected_index_identity() -> str:
    return derived_schema_identity(DerivedTier.INDEX)


def _tier_identity(path: Path) -> tuple[object, ...]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_dev, stat.st_ino)


def _frame_paths(root: Path) -> tuple[Path, Path, Path]:
    paths = (root / "source.db", resolve_active_index_path(root), root / "ops.db")
    if any(not path.is_file() for path in paths):
        raise ValueError("required source, index, or ops frontier tier is unavailable")
    return paths


def _frame_versions(conn: sqlite3.Connection) -> tuple[int, int, int]:
    return (
        int(conn.execute("PRAGMA main.data_version").fetchone()[0]),
        int(conn.execute("PRAGMA index_tier.data_version").fetchone()[0]),
        int(conn.execute("PRAGMA ops_tier.data_version").fetchone()[0]),
    )


@contextmanager
def stable_selected_authority_frame(archive_root: Path) -> Iterator[None]:
    """Refuse any ordinary SQLite write during selected and global proofs.

    ``data_version`` is compared on the same read-only handle. It is only a
    conservative change detector here; journal reconciliation, never a
    data-version delta, proves the global raw-existence subset.
    """
    root = archive_root.resolve()
    try:
        source_path, index_path, ops_path = _frame_paths(root)
        before_tiers = tuple(_tier_identity(path) for path in (source_path, index_path, ops_path))
        with closing(open_readonly_connection(source_path, validate_schema=False)) as conn:
            attach_readonly_database(conn, index_path, alias="index_tier")
            attach_readonly_database(conn, ops_path, alias="ops_tier")
            if tuple(_tier_identity(path) for path in (source_path, index_path, ops_path)) != before_tiers:
                raise ValueError("selected frontier tier changed while opening authority")
            before_versions = _frame_versions(conn)
            yield
            after_versions = _frame_versions(conn)
            after_paths = _frame_paths(root)
            after_tiers = tuple(_tier_identity(path) for path in after_paths)
            if after_versions != before_versions or after_tiers != before_tiers:
                raise ValueError("selected frontier authority changed during admission")
    except sqlite3.Error as exc:
        raise ValueError(f"selected frontier authority observation failed: {exc}") from exc


def _journal_state(conn: sqlite3.Connection, schema: str, expected: dict[str, str]) -> tuple[int, int]:
    actual = {
        str(row[0]): _normalized_sql(str(row[1]))
        for row in conn.execute(
            f"SELECT name, sql FROM {schema}.sqlite_schema WHERE type = 'trigger' AND name LIKE 'raw_existence_%'"
        )
    }
    if actual != expected:
        raise ValueError(f"{schema} raw-existence trigger contract is unavailable")
    row = conn.execute(
        f"SELECT retained_floor FROM {schema}.raw_existence_journal_control WHERE singleton = 1"
    ).fetchone()
    if row is None:
        raise ValueError(f"{schema} raw-existence journal control is unavailable")
    floor = int(row[0])
    # AUTOINCREMENT retains the committed high-water mark even when every
    # consumed journal row has been pruned. MAX(sequence) would turn an empty
    # retained tail into an apparent regression and refuse a sound bootstrap.
    sequence = conn.execute(f"SELECT seq FROM {schema}.sqlite_sequence WHERE name = 'raw_existence_changes'").fetchone()
    high = int(sequence[0]) if sequence is not None else 0
    if floor < 0 or floor > high:
        raise ValueError(f"{schema} raw-existence journal coverage is invalid")
    return high, floor


def _identity(conn: sqlite3.Connection, source_path: Path, index_path: Path) -> tuple[object, ...]:
    source_schema = int(conn.execute("PRAGMA main.schema_version").fetchone()[0])
    index_schema = int(conn.execute("PRAGMA index_tier.schema_version").fetchone()[0])
    source_version = int(conn.execute("PRAGMA main.user_version").fetchone()[0])
    index_version = int(conn.execute("PRAGMA index_tier.user_version").fetchone()[0])
    index_identity = conn.execute("SELECT identity FROM index_tier.schema_identity WHERE tier = 'index'").fetchone()
    if index_identity is None:
        raise ValueError("index derived schema identity is unavailable")
    if str(index_identity[0]) != _expected_index_identity():
        raise ValueError("index derived schema identity does not match the current code")
    return (
        _tier_identity(source_path),
        _tier_identity(index_path),
        source_schema,
        index_schema,
        source_version,
        index_version,
        str(index_identity[0]),
    )


def _read_state(root: Path) -> tuple[tuple[object, ...], int, int, int, int]:
    source_path = root / "source.db"
    index_path = resolve_active_index_path(root)
    if not source_path.is_file() or not index_path.is_file():
        raise ValueError("required source or index tier is unavailable")
    opened_incarnations = (_tier_identity(source_path), _tier_identity(index_path))
    with closing(open_readonly_connection(source_path, validate_schema=False)) as conn:
        attach_readonly_database(conn, index_path, alias="index_tier")
        conn.execute("BEGIN")
        identity = _identity(conn, source_path, index_path)
        if identity[:2] != opened_incarnations:
            raise ValueError("raw-existence tier changed while opening authority")
        source_high, source_floor = _journal_state(conn, "main", _SOURCE_TRIGGERS)
        index_high, index_floor = _journal_state(conn, "index_tier", _INDEX_TRIGGERS)
        conn.commit()
    return identity, source_high, source_floor, index_high, index_floor


def _missing_reference(conn: sqlite3.Connection) -> bool:
    """Whether any active index raw reference is absent from source (full check)."""
    if conn.execute(
        "SELECT 1 FROM index_tier.sessions s WHERE s.raw_id IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM raw_sessions r WHERE r.raw_id = s.raw_id) LIMIT 1"
    ).fetchone():
        return True
    return (
        conn.execute(
            "SELECT 1 FROM index_tier.raw_revision_heads h WHERE h.accepted_raw_id IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM raw_sessions r WHERE r.raw_id = h.accepted_raw_id) LIMIT 1"
        ).fetchone()
        is not None
    )


_CHANGED_KEY_BATCH = 400


def _first_missing_reference(conn: sqlite3.Connection, raw_ids: set[str]) -> str | None:
    """One changed key still referenced by the index but absent from source.

    Checked in bounded ``IN`` batches: a page that journals thousands of keys
    costs a handful of anti-joins, not two statements per key.
    """
    ordered = sorted(raw_ids)
    for start in range(0, len(ordered), _CHANGED_KEY_BATCH):
        batch = tuple(ordered[start : start + _CHANGED_KEY_BATCH])
        marks = ",".join("?" for _ in batch)
        row = conn.execute(
            f"SELECT s.raw_id FROM index_tier.sessions s WHERE s.raw_id IN ({marks}) "
            "AND NOT EXISTS (SELECT 1 FROM raw_sessions r WHERE r.raw_id = s.raw_id) "
            f"UNION ALL SELECT h.accepted_raw_id FROM index_tier.raw_revision_heads h "
            f"WHERE h.accepted_raw_id IN ({marks}) "
            "AND NOT EXISTS (SELECT 1 FROM raw_sessions r WHERE r.raw_id = h.accepted_raw_id) LIMIT 1",
            batch * 2,
        ).fetchone()
        if row is not None:
            return str(row[0])
    return None


def _changed_keys(conn: sqlite3.Connection, schema: str, low: int, high: int) -> set[str]:
    rows = conn.execute(
        f"SELECT sequence, raw_id FROM {schema}.raw_existence_changes "
        "WHERE sequence > ? AND sequence <= ? ORDER BY sequence",
        (low, high),
    ).fetchall()
    if len(rows) != high - low or any(int(row[0]) != low + offset + 1 for offset, row in enumerate(rows)):
        raise ValueError(f"{schema} raw-existence journal has a coverage gap")
    return {str(row[1]) for row in rows}


def _prove(root: Path, old: _Certificate | None) -> _Certificate:
    source_path = root / "source.db"
    index_path = resolve_active_index_path(root)
    if not source_path.is_file() or not index_path.is_file():
        raise ValueError("required source or index tier is unavailable")
    opened_incarnations = (_tier_identity(source_path), _tier_identity(index_path))
    with closing(open_readonly_connection(source_path, validate_schema=False)) as conn:
        attach_readonly_database(conn, index_path, alias="index_tier")
        conn.execute("BEGIN")
        identity = _identity(conn, source_path, index_path)
        if identity[:2] != opened_incarnations:
            raise ValueError("raw-existence tier changed while opening authority")
        source_high, source_floor = _journal_state(conn, "main", _SOURCE_TRIGGERS)
        index_high, index_floor = _journal_state(conn, "index_tier", _INDEX_TRIGGERS)
        # A certificate whose unconsumed journal rows were pruned (by this
        # process's consumed-row owner, or a stricter one elsewhere) cannot be
        # advanced incrementally; it is re-proven from scratch instead.
        truncated = old is not None and (old.source_watermark < source_floor or old.index_watermark < index_floor)
        if old is None or old.identity != identity or truncated:
            if _missing_reference(conn):
                raise ValueError("active index raw is missing from source tier")
        else:
            if old.source_watermark > source_high or old.index_watermark > index_high:
                raise ValueError("raw-existence journal watermark regressed")
            changed = _changed_keys(conn, "main", old.source_watermark, source_high)
            changed.update(_changed_keys(conn, "index_tier", old.index_watermark, index_high))
            missing = _first_missing_reference(conn, changed)
            if missing is not None:
                raise ValueError(f"active index raw is missing from source tier: {missing}")
        conn.commit()
    candidate = _Certificate(identity, source_high, index_high)
    if _read_state(root) != (identity, source_high, source_floor, index_high, index_floor):
        raise ValueError("raw-existence authority changed during admission")
    return candidate


def raw_existence_block_reason(archive_root: Path) -> str | None:
    """Return an unattributed refusal, or refresh the healthy certificate."""
    global _pid
    root = archive_root.resolve()
    with _lock:
        if _pid != os.getpid():
            _certificates.clear()
            _pid = os.getpid()
        old = _certificates.get(root)
        try:
            evidence = capture_sqlite_read(lambda: _prove(root, old))
        except (OSError, ValueError) as exc:
            _certificates.pop(root, None)
            return f"global raw existence is unproven: {exc}"
        if isinstance(evidence, Unavailable):
            _certificates.pop(root, None)
            return f"global raw existence is unproven: {evidence.detail or evidence.reason}"
        if not isinstance(evidence, Measured):
            _certificates.pop(root, None)
            return "global raw existence is unproven: observation is incomplete"
        candidate = evidence.value
        _certificates[root] = candidate
        return None


def consumed_watermarks(archive_root: Path) -> tuple[int, int] | None:
    """The journal positions this process's healthy certificate has consumed.

    Journal rows at or below them are no longer needed by this process. The
    daemon's pruning stage deletes exactly those; a process whose certificate
    lags re-proves from scratch rather than trusting a truncated journal.
    """
    root = archive_root.resolve()
    with _lock:
        certificate = _certificates.get(root) if _pid == os.getpid() else None
    if certificate is None:
        return None
    return certificate.source_watermark, certificate.index_watermark


def _journal_tiers(root: Path) -> tuple[tuple[Path, int], tuple[Path, int]] | None:
    from polylogue.storage.frontier_inspection import read_frontier_inspection_mark

    marks = consumed_watermarks(root)
    if marks is None or not (root / "ops.db").is_file():
        return None
    with closing(open_readonly_connection(root / "ops.db")) as conn:
        mark = read_frontier_inspection_mark(conn)
    if mark is None or mark.state != "healthy":
        return None
    return (root / "source.db", min(marks[0], mark.source_watermark)), (
        resolve_active_index_path(root),
        min(marks[1], mark.index_watermark),
    )


def has_consumed_journal_rows(archive_root: Path) -> bool:
    """Whether a journal still holds rows this process's certificate consumed."""
    tiers = _journal_tiers(archive_root.resolve())
    if tiers is None:
        return False
    for tier_path, watermark in tiers:
        if not tier_path.is_file():
            continue
        with closing(open_readonly_connection(tier_path, validate_schema=False)) as conn:
            if (
                conn.execute(
                    "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'raw_existence_changes'"
                ).fetchone()
                and conn.execute(
                    "SELECT 1 FROM raw_existence_changes WHERE sequence <= ? LIMIT 1", (watermark,)
                ).fetchone()
            ):
                return True
    return False


def prune_consumed_journal_rows(archive_root: Path, *, input_demand: Callable[[int], None]) -> int:
    """Publish the canonical prepared prune under the supplied original creator."""
    from polylogue.storage.frontier_inspection import prune_prepared_frontier_journals

    return prune_prepared_frontier_journals(archive_root.resolve(), input_demand=input_demand)
