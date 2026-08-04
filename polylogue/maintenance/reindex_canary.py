"""Read-only semantic comparison for an inactive reindex canary.

The rebuild service owns creation of an inactive generation.  This module owns
the other half of the canary contract: compare the real SQLite read models in
that generation with the active index and account for every row difference.
It deliberately opens both inputs read-only and never knows how to promote,
repair, or otherwise mutate a generation.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, cast


class DifferenceOperation(StrEnum):
    """The row-level change observed between the active and candidate indexes."""

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"


class DifferenceClassification(StrEnum):
    """How the canary changelog accounts for a semantic difference."""

    EXPECTED = "expected"
    UNEXPECTED = "unexpected"


class CanarySelectionError(ValueError):
    """The requested representative canary cannot be built from this index."""


class UnclassifiedCanaryDiffError(ValueError):
    """A durable report was requested without one review per diff row."""


@dataclass(frozen=True, slots=True)
class ExpectedDifference:
    """A reviewed change signature that is allowed in a canary report.

    Matching is intentionally structural.  A bead or delta declaration names
    the affected table and may narrow the signature to an operation and/or a
    changed column.  Unmatched differences are ``UNEXPECTED`` by default, so
    the report cannot contain an unclassified bucket.
    """

    table: str
    bead_ref: str
    rationale: str
    operations: tuple[DifferenceOperation, ...] = ()
    columns: tuple[str, ...] = ()

    def matches(
        self,
        *,
        table: str,
        operation: DifferenceOperation,
        changed_columns: tuple[str, ...],
    ) -> bool:
        if self.table != table:
            return False
        if self.operations and operation not in self.operations:
            return False
        return not self.columns or bool(set(self.columns).intersection(changed_columns))


@dataclass(frozen=True, slots=True)
class RowDifference:
    """One canonical row-level difference in the canary changelog."""

    table: str
    operation: DifferenceOperation
    identity: tuple[tuple[str, object], ...]
    before: dict[str, object] | None
    after: dict[str, object] | None
    changed_columns: tuple[str, ...]
    classification: DifferenceClassification
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "operation": self.operation.value,
            "identity": dict(self.identity),
            "before": self.before,
            "after": self.after,
            "changed_columns": list(self.changed_columns),
            "classification": self.classification.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class CanaryDiffReport:
    """Complete, JSON-ready account of a read-model comparison."""

    current_index: Path
    candidate_index: Path
    session_ids: tuple[str, ...]
    compared_tables: tuple[str, ...]
    missing_tables: tuple[str, ...]
    differences: tuple[RowDifference, ...]

    @property
    def expected_count(self) -> int:
        return sum(item.classification is DifferenceClassification.EXPECTED for item in self.differences)

    @property
    def unexpected_count(self) -> int:
        return sum(item.classification is DifferenceClassification.UNEXPECTED for item in self.differences)

    @property
    def unclassified_count(self) -> int:
        """The explicit zero-bucket contract for the canary changelog."""

        return 0

    @property
    def counts_by_table(self) -> dict[str, int]:
        return dict(sorted(Counter(item.table for item in self.differences).items()))

    def to_dict(self) -> dict[str, object]:
        return {
            "current_index": str(self.current_index),
            "candidate_index": str(self.candidate_index),
            "session_ids": list(self.session_ids),
            "compared_tables": list(self.compared_tables),
            "missing_tables": list(self.missing_tables),
            "summary": {
                "difference_count": len(self.differences),
                "expected_count": self.expected_count,
                "unexpected_count": self.unexpected_count,
                "unclassified_count": self.unclassified_count,
                "counts_by_table": self.counts_by_table,
            },
            "differences": [item.to_dict() for item in self.differences],
        }


@dataclass(frozen=True, slots=True)
class CanarySelection:
    """Deterministic, read-only input selection for one canary rebuild."""

    index_path: Path
    sessions_per_origin: int
    selected_session_ids: tuple[str, ...]
    selected_raw_ids: tuple[str, ...]
    sampled_session_ids: tuple[str, ...]
    pathology_session_ids: tuple[str, ...]
    sample_session_ids: tuple[str, ...]
    origin_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "index_path": str(self.index_path),
            "sessions_per_origin": self.sessions_per_origin,
            "selected_session_ids": list(self.selected_session_ids),
            "selected_raw_ids": list(self.selected_raw_ids),
            "sampled_session_ids": list(self.sampled_session_ids),
            "pathology_session_ids": list(self.pathology_session_ids),
            "sample_session_ids": list(self.sample_session_ids),
            "origin_counts": dict(self.origin_counts),
        }


def select_canary_sessions(
    index_path: Path,
    *,
    sessions_per_origin: int = 100,
    pathology_session_ids: Iterable[str] = (),
    sample_session_ids: Iterable[str] = (),
) -> CanarySelection:
    """Select a representative raw-id set from a real active index.

    The automatic portion takes the newest deterministic ``N`` sessions for
    every origin. Explicit pathology and sample sessions are always included,
    even when they fall outside that sample. Every explicit id must resolve to
    an indexed session with a non-null ``raw_id`` because the existing rebuild
    route accepts raw ids, not summaries or synthetic session descriptions.
    """

    if sessions_per_origin <= 0:
        raise CanarySelectionError("sessions_per_origin must be positive")
    path = Path(index_path)
    pathology = tuple(dict.fromkeys(str(value) for value in pathology_session_ids))
    explicit_samples = tuple(dict.fromkeys(str(value) for value in sample_session_ids))
    explicit = set(pathology).union(explicit_samples)
    with _open_read_only(path) as connection:
        rows = connection.execute(
            """
            SELECT session_id, origin, raw_id, sort_key_ms
            FROM sessions
            ORDER BY origin, (sort_key_ms IS NULL), sort_key_ms DESC, session_id
            """
        ).fetchall()

    records: dict[str, tuple[str, str | None]] = {
        str(row["session_id"]): (
            str(row["origin"]),
            str(row["raw_id"]) if row["raw_id"] is not None else None,
        )
        for row in rows
    }
    missing = sorted(explicit.difference(records))
    if missing:
        raise CanarySelectionError(f"explicit canary session(s) are not indexed: {', '.join(missing)}")
    without_raw = sorted(session_id for session_id in explicit if records[session_id][1] is None)
    if without_raw:
        raise CanarySelectionError(
            "explicit canary session(s) have no raw_id and cannot be replayed: " + ", ".join(without_raw)
        )

    sampled: list[str] = []
    origin_seen: dict[str, int] = {}
    for row in rows:
        if row["raw_id"] is None:
            continue
        origin = str(row["origin"])
        if origin_seen.get(origin, 0) >= sessions_per_origin:
            continue
        session_id = str(row["session_id"])
        sampled.append(session_id)
        origin_seen[origin] = origin_seen.get(origin, 0) + 1

    selected = set(sampled).union(explicit)
    selected_session_ids = tuple(sorted(selected))
    selected_raw_ids = tuple(
        sorted(raw_id for session_id in selected if (raw_id := records[session_id][1]) is not None)
    )
    origin_counts = Counter(records[session_id][0] for session_id in selected)
    return CanarySelection(
        index_path=path,
        sessions_per_origin=sessions_per_origin,
        selected_session_ids=selected_session_ids,
        selected_raw_ids=selected_raw_ids,
        sampled_session_ids=tuple(sorted(sampled)),
        pathology_session_ids=tuple(sorted(pathology)),
        sample_session_ids=tuple(sorted(explicit_samples)),
        origin_counts=tuple(sorted(origin_counts.items())),
    )


@dataclass(frozen=True, slots=True)
class CanaryDifferenceReview:
    """An explicit operator classification for one diff row."""

    table: str
    operation: DifferenceOperation
    identity: tuple[tuple[str, object], ...]
    classification: DifferenceClassification
    reference: str
    rationale: str

    @classmethod
    def for_difference(
        cls,
        difference: RowDifference,
        *,
        classification: DifferenceClassification,
        reference: str,
        rationale: str,
    ) -> CanaryDifferenceReview:
        return cls(
            table=difference.table,
            operation=difference.operation,
            identity=difference.identity,
            classification=classification,
            reference=reference,
            rationale=rationale,
        )

    @property
    def key(self) -> tuple[str, DifferenceOperation, tuple[tuple[str, object], ...]]:
        return self.table, self.operation, self.identity

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "operation": self.operation.value,
            "identity": dict(self.identity),
            "classification": self.classification.value,
            "reference": self.reference,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class DurableCanaryReport:
    """The reviewed, persisted canary changelog."""

    selection: CanarySelection
    comparison: CanaryDiffReport
    reviews: tuple[CanaryDifferenceReview, ...]

    @property
    def unclassified_count(self) -> int:
        return self.comparison.unclassified_count

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "selection": self.selection.to_dict(),
            "comparison": self.comparison.to_dict(),
            "reviews": [review.to_dict() for review in self.reviews],
        }


def write_canary_report(
    output_path: Path,
    *,
    selection: CanarySelection,
    comparison: CanaryDiffReport,
    reviews: Iterable[CanaryDifferenceReview],
) -> DurableCanaryReport:
    """Persist a reviewed report, refusing any incomplete classification.

    Reviews must cover exactly the comparator's row identities. This prevents
    a report from becoming a durable green-light artifact while a diff was
    silently omitted from review. The write is atomic and touches only the
    requested report path, never either SQLite generation.
    """

    review_list = tuple(reviews)
    review_by_key: dict[tuple[str, DifferenceOperation, tuple[tuple[str, object], ...]], CanaryDifferenceReview] = {}
    duplicate_keys: list[object] = []
    for review in review_list:
        if not review.reference.strip() or not review.rationale.strip():
            raise UnclassifiedCanaryDiffError("every canary review needs a non-empty reference and rationale")
        if review.key in review_by_key:
            duplicate_keys.append(review.key)
        review_by_key[review.key] = review
    difference_keys = {
        (
            difference.table,
            difference.operation,
            difference.identity,
        )
        for difference in comparison.differences
    }
    missing_keys = difference_keys.difference(review_by_key)
    extra_keys = set(review_by_key).difference(difference_keys)
    if duplicate_keys or missing_keys or extra_keys:
        detail = [
            f"duplicate={len(duplicate_keys)}",
            f"missing={len(missing_keys)}",
            f"extra={len(extra_keys)}",
        ]
        raise UnclassifiedCanaryDiffError("canary report classification is incomplete (" + ", ".join(detail) + ")")

    reviewed_differences = tuple(
        replace(
            difference,
            classification=review_by_key[(difference.table, difference.operation, difference.identity)].classification,
            rationale=(
                f"{review_by_key[(difference.table, difference.operation, difference.identity)].reference}: "
                f"{review_by_key[(difference.table, difference.operation, difference.identity)].rationale}"
            ),
        )
        for difference in comparison.differences
    )
    reviewed_comparison = replace(comparison, differences=reviewed_differences)
    durable = DurableCanaryReport(selection=selection, comparison=reviewed_comparison, reviews=review_list)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(durable.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return durable


def load_canary_report(path: Path) -> dict[str, object]:
    """Read a durable report and reject any persisted unclassified summary."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UnclassifiedCanaryDiffError("canary report root must be an object")
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        raise UnclassifiedCanaryDiffError("canary report has no comparison object")
    summary = comparison.get("summary")
    if not isinstance(summary, dict) or summary.get("unclassified_count"):
        raise UnclassifiedCanaryDiffError("canary report contains unclassified differences")
    return cast(dict[str, object], payload)


# These are SQLite implementation details rather than semantic read models.
# FTS backing tables are virtual-table internals and generation metadata is not
# part of the session read model.  Aggregate rollups without a session key
# cannot be attributed to a partial canary and are therefore intentionally
# outside this comparator's scope.
_EXCLUDED_TABLES = frozenset(
    {
        "delegation_refresh_scope",
        "derived_refresh_guard",
        "agent_meta_sidecar_purge_receipts",
        "session_tag_rollups",
    }
)
_CORE_TABLES = ("sessions", "messages", "blocks", "session_links")
_SESSION_SCOPE_COLUMNS = (
    "session_id",
    "src_session_id",
    "parent_session_id",
    "child_session_id",
)
_VOLATILE_COLUMNS = frozenset(
    {
        "generation_id",
        "generation_owner_id",
        "materialized_at",
        "materialized_at_ms",
        "materialized_at_utc",
        "refreshed_at_ms",
    }
)


def compare_reindex_generations(
    current_index: Path,
    candidate_index: Path,
    *,
    session_ids: Iterable[str] = (),
    expected: Iterable[ExpectedDifference] = (),
) -> CanaryDiffReport:
    """Compare real generation read models without mutating either database.

    ``current_index`` is the active generation and ``candidate_index`` is the
    inactive generation produced by the existing rebuild service.  When no
    session selection is supplied, the union of session ids in both databases
    is compared.  A partial selection is useful for the N-per-origin canary
    and still reports additions/removals inside that selection.
    """

    current_path = Path(current_index)
    candidate_path = Path(candidate_index)
    if not current_path.exists():
        raise FileNotFoundError(f"current index does not exist: {current_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate index does not exist: {candidate_path}")

    reviewed = tuple(expected)
    with _open_read_only(current_path) as current, _open_read_only(candidate_path) as candidate:
        current_tables = _read_model_tables(current)
        candidate_tables = _read_model_tables(candidate)
        compared_tables = tuple(sorted(current_tables.intersection(candidate_tables)))
        missing_tables = tuple(sorted(current_tables.symmetric_difference(candidate_tables)))
        selected_sessions = _selected_session_ids(current, candidate, session_ids)
        differences: list[RowDifference] = []
        for table in compared_tables:
            differences.extend(
                _compare_table(
                    table,
                    current,
                    candidate,
                    session_ids=selected_sessions,
                    expected=reviewed,
                )
            )

    return CanaryDiffReport(
        current_index=current_path,
        candidate_index=candidate_path,
        session_ids=selected_sessions,
        compared_tables=compared_tables,
        missing_tables=missing_tables,
        differences=tuple(differences),
    )


def _open_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve(strict=True)}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _read_model_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    result: set[str] = set()
    for row in rows:
        table = str(row[0])
        if table in _EXCLUDED_TABLES or table.startswith("messages_fts") or table.startswith("blocks_command_trigram"):
            continue
        columns = _table_columns(connection, table)
        if table in _CORE_TABLES or any(column in columns for column in _SESSION_SCOPE_COLUMNS):
            result.add(table)
    return result


def _table_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    quoted = _quote_identifier(table)
    rows = connection.execute(f"PRAGMA table_xinfo({quoted})").fetchall()
    # hidden=1/2 are virtual-table shadow columns. Generated columns (3) are
    # real semantic ids and remain in the comparison.
    return tuple(str(row[1]) for row in rows if int(row[6]) not in (1, 2))


def _table_primary_key(connection: sqlite3.Connection, table: str, columns: tuple[str, ...]) -> tuple[str, ...]:
    rows = connection.execute(f"PRAGMA table_xinfo({_quote_identifier(table)})").fetchall()
    primary_key = [(int(row[5]), str(row[1])) for row in rows if int(row[5]) > 0 and int(row[6]) not in (1, 2)]
    if primary_key:
        return tuple(name for _position, name in sorted(primary_key))
    for preferred in ("session_id", "message_id", "block_id", "event_id", "policy_id"):
        if preferred in columns:
            return (preferred,)
    return columns


def _selected_session_ids(
    current: sqlite3.Connection,
    candidate: sqlite3.Connection,
    requested: Iterable[str],
) -> tuple[str, ...]:
    explicit = tuple(dict.fromkeys(str(value) for value in requested))
    if explicit:
        return tuple(sorted(explicit))
    values: set[str] = set()
    for connection in (current, candidate):
        if "sessions" not in _read_model_tables(connection):
            continue
        rows = connection.execute("SELECT session_id FROM sessions ORDER BY session_id").fetchall()
        values.update(str(row[0]) for row in rows)
    return tuple(sorted(values))


def _compare_table(
    table: str,
    current: sqlite3.Connection,
    candidate: sqlite3.Connection,
    *,
    session_ids: tuple[str, ...],
    expected: tuple[ExpectedDifference, ...],
) -> list[RowDifference]:
    current_columns = _table_columns(current, table)
    candidate_columns = _table_columns(candidate, table)
    columns = tuple(column for column in current_columns if column in candidate_columns)
    if not columns:
        return []
    scope_columns = tuple(column for column in _SESSION_SCOPE_COLUMNS if column in columns)
    if not scope_columns:
        return []
    current_rows = _table_rows(current, table, columns, scope_columns, session_ids)
    candidate_rows = _table_rows(candidate, table, columns, scope_columns, session_ids)
    keys = sorted(set(current_rows).union(candidate_rows), key=repr)
    primary_key = _table_primary_key(current, table, columns)
    differences: list[RowDifference] = []
    for key in keys:
        before = current_rows.get(key)
        after = candidate_rows.get(key)
        if before == after:
            continue
        if before is None:
            operation = DifferenceOperation.ADDED
            changed_columns = tuple(after) if after is not None else ()
        elif after is None:
            operation = DifferenceOperation.REMOVED
            changed_columns = tuple(before)
        else:
            operation = DifferenceOperation.CHANGED
            changed_columns = tuple(column for column in columns if before.get(column) != after.get(column))
        matching = next(
            (
                item
                for item in expected
                if item.matches(table=table, operation=operation, changed_columns=changed_columns)
            ),
            None,
        )
        differences.append(
            RowDifference(
                table=table,
                operation=operation,
                identity=tuple((column, key[index]) for index, column in enumerate(primary_key)),
                before=before,
                after=after,
                changed_columns=changed_columns,
                classification=(
                    DifferenceClassification.EXPECTED if matching is not None else DifferenceClassification.UNEXPECTED
                ),
                rationale=(
                    f"{matching.bead_ref}: {matching.rationale}"
                    if matching is not None
                    else "no reviewed bead or delta declaration matched this difference"
                ),
            )
        )
    return differences


def _table_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    scope_columns: tuple[str, ...],
    session_ids: tuple[str, ...],
) -> dict[tuple[object, ...], dict[str, object]]:
    selected_columns = tuple(column for column in columns if column not in _VOLATILE_COLUMNS)
    quoted_columns = ", ".join(_quote_identifier(column) for column in columns)
    query = f"SELECT {quoted_columns} FROM {_quote_identifier(table)}"
    parameters: tuple[str, ...] = ()
    if session_ids:
        placeholders = ", ".join("?" for _ in session_ids)
        query += " WHERE " + " OR ".join(f"{_quote_identifier(column)} IN ({placeholders})" for column in scope_columns)
        parameters = session_ids * len(scope_columns)
    query += " ORDER BY rowid" if table not in {"sessions", "messages", "blocks", "session_links"} else ""
    result: dict[tuple[object, ...], dict[str, object]] = {}
    primary_key = _table_primary_key(connection, table, columns)
    for row in connection.execute(query, parameters):
        normalized = {column: _normalize_value(column, row[column]) for column in selected_columns}
        key = tuple(_normalize_value(column, row[column]) for column in primary_key)
        result[key] = normalized
    return result


def _normalize_value(column: str, value: Any) -> object:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, str) and column.endswith("_json"):
        try:
            return json.dumps(json.loads(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return value
    return value


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


__all__ = [
    "CanaryDifferenceReview",
    "CanaryDiffReport",
    "CanarySelection",
    "CanarySelectionError",
    "DurableCanaryReport",
    "DifferenceClassification",
    "DifferenceOperation",
    "ExpectedDifference",
    "RowDifference",
    "UnclassifiedCanaryDiffError",
    "compare_reindex_generations",
    "load_canary_report",
    "select_canary_sessions",
    "write_canary_report",
]
