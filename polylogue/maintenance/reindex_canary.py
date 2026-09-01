"""Source-authoritative, inactive-generation canary construction.

The canary answers whether current source obligations hold for a bounded
candidate. Historical index differences are diagnostic evidence only.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast


class CanarySelectionError(ValueError):
    """The sealed source cohort cannot be selected safely."""


class DifferenceOperation(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"


@dataclass(frozen=True, slots=True)
class CanarySelection:
    index_path: Path
    sessions_per_origin: int
    selected_session_ids: tuple[str, ...]
    selected_raw_ids: tuple[str, ...]
    sampled_session_ids: tuple[str, ...] = ()
    origin_counts: tuple[tuple[str, int], ...] = ()
    source_manifest_digest: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "index_path": str(self.index_path),
            "sessions_per_origin": self.sessions_per_origin,
            "selected_session_ids": list(self.selected_session_ids),
            "selected_raw_ids": list(self.selected_raw_ids),
            "sampled_session_ids": list(self.sampled_session_ids),
            "origin_counts": dict(self.origin_counts),
            "source_manifest_digest": self.source_manifest_digest,
        }


@dataclass(frozen=True, slots=True)
class RowDifference:
    table: str
    operation: DifferenceOperation
    identity: tuple[tuple[str, object], ...]
    before: dict[str, object] | None
    after: dict[str, object] | None
    changed_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "operation": self.operation.value,
            "identity": dict(self.identity),
            "before": self.before,
            "after": self.after,
            "changed_columns": list(self.changed_columns),
        }


@dataclass(frozen=True, slots=True)
class CanaryDiffReport:
    current_index: Path
    candidate_index: Path
    session_ids: tuple[str, ...]
    compared_tables: tuple[str, ...]
    differences: tuple[RowDifference, ...]

    @property
    def counts_by_operation(self) -> dict[str, int]:
        return dict(sorted(Counter(item.operation.value for item in self.differences).items()))

    @property
    def counts_by_table(self) -> dict[str, int]:
        return dict(sorted(Counter(item.table for item in self.differences).items()))

    def to_dict(self) -> dict[str, object]:
        return {
            "current_index": str(self.current_index),
            "candidate_index": str(self.candidate_index),
            "session_ids": list(self.session_ids),
            "compared_tables": list(self.compared_tables),
            "summary": {
                "difference_count": len(self.differences),
                "counts_by_operation": self.counts_by_operation,
                "counts_by_table": self.counts_by_table,
            },
            "differences": [item.to_dict() for item in self.differences],
        }


@dataclass(frozen=True, slots=True)
class CanaryRunResult:
    selection: CanarySelection
    comparison: CanaryDiffReport
    rebuild_receipt: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "selection": self.selection.to_dict(),
            "comparison": self.comparison.to_dict(),
            "rebuild_receipt": self.rebuild_receipt,
        }


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve(strict=True)}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def select_canary_sessions(
    index_path: Path, *, sessions_per_origin: int = 100, source_manifest_digest: str | None = None
) -> CanarySelection:
    """Select a deterministic cohort from the sealed source-backed index."""
    if sessions_per_origin <= 0:
        raise CanarySelectionError("sessions_per_origin must be positive")
    with _open_read_only(Path(index_path)) as connection:
        rows = connection.execute(
            "SELECT session_id, origin, raw_id, sort_key_ms FROM sessions WHERE raw_id IS NOT NULL ORDER BY origin, (sort_key_ms IS NULL), sort_key_ms DESC, session_id"
        ).fetchall()
    selected: list[str] = []
    counts: Counter[str] = Counter()
    raw_by_session: dict[str, str] = {}
    for row in rows:
        origin = str(row["origin"])
        if counts[origin] >= sessions_per_origin:
            continue
        session_id = str(row["session_id"])
        selected.append(session_id)
        raw_by_session[session_id] = str(row["raw_id"])
        counts[origin] += 1
    if not selected:
        raise CanarySelectionError("sealed source cohort is empty")
    selected_ids = tuple(sorted(selected))
    return CanarySelection(
        Path(index_path),
        sessions_per_origin,
        selected_ids,
        tuple(sorted(raw_by_session.values())),
        selected_ids,
        tuple(sorted(counts.items())),
        source_manifest_digest,
    )


def run_reindex_canary(
    archive_root: Path,
    *,
    input_index: Path | None = None,
    schema_inference_receipt_path: Path | None,
    sessions_per_origin: int = 100,
    no_promote: bool,
) -> CanaryRunResult:
    """Build and inspect one inactive candidate through the daemon route."""
    if schema_inference_receipt_path is None:
        raise CanarySelectionError("reindex canary requires an explicit schema-inference receipt path")
    if not no_promote:
        raise CanarySelectionError("reindex canary requires --no-promote")
    from polylogue.storage.archive_identity import ArchiveLocation, TierFileIdentity

    root = Path(archive_root).resolve()
    active = ArchiveLocation.resolve(root).active_index
    current = active if input_index is None else TierFileIdentity.resolve("index", Path(input_index))
    if not current.same_file(active):
        raise CanarySelectionError("canary input index must be the configured active generation")
    selection = select_canary_sessions(current.resolved_path, sessions_per_origin=sessions_per_origin)
    from polylogue.daemon.bulk_rebuild import run_daemon_canary_rebuild
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION

    receipt = run_daemon_canary_rebuild(
        archive_root=root,
        raw_ids=selection.selected_raw_ids,
        selected_session_ids=selection.selected_session_ids,
        index_schema_version=INDEX_SCHEMA_VERSION,
        schema_inference_receipt_path=schema_inference_receipt_path,
    )
    if not isinstance(receipt, dict):
        raise CanarySelectionError("daemon candidate operation returned invalid evidence")
    generation = receipt.get("generation")
    candidate = (
        Path(generation["index_path"])
        if isinstance(generation, dict) and isinstance(generation.get("index_path"), str)
        else None
    )
    if candidate is None or not candidate.is_absolute() or not candidate.exists():
        raise CanarySelectionError("daemon candidate evidence has no inactive index")
    if isinstance(generation, dict) and generation.get("state") not in (None, "inactive"):
        raise CanarySelectionError("candidate generation is not inactive")
    return CanaryRunResult(
        selection,
        compare_reindex_generations(current.resolved_path, candidate, session_ids=selection.selected_session_ids),
        receipt,
    )


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'"
        )
        if not str(row[0]).startswith(("messages_fts", "blocks_command_trigram"))
    }


def _columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    quoted = table.replace('"', '""')
    return tuple(
        str(row[1]) for row in connection.execute(f'PRAGMA table_xinfo("{quoted}")') if int(row[6]) not in (1, 2)
    )


def compare_reindex_generations(
    current_index: Path, candidate_index: Path, *, session_ids: Iterable[str] = ()
) -> CanaryDiffReport:
    """Bounded forensic comparison. Its result cannot authorize acceptance."""
    current_path, candidate_path = Path(current_index), Path(candidate_index)
    requested = tuple(sorted(dict.fromkeys(str(value) for value in session_ids)))
    differences: list[RowDifference] = []
    with _open_read_only(current_path) as current, _open_read_only(candidate_path) as candidate:
        tables = sorted(_tables(current) & _tables(candidate))
        for table in tables:
            columns = tuple(column for column in _columns(current, table) if column in _columns(candidate, table))
            scope = tuple(column for column in ("session_id", "src_session_id", "dst_session_id") if column in columns)
            if not scope:
                continue
            select = ",".join(f'"{column.replace(chr(34), chr(34) * 2)}"' for column in columns)
            query = f'SELECT {select} FROM "{table.replace(chr(34), chr(34) * 2)}"'
            args: tuple[str, ...] = ()
            if requested:
                marks = ",".join("?" for _ in requested)
                query += " WHERE " + " OR ".join(f'"{column}" IN ({marks})' for column in scope)
                args = requested * len(scope)
            key_columns = ("session_id",) if "session_id" in columns else (columns[0],)
            key_indexes = tuple(columns.index(column) for column in key_columns)
            left = {tuple(row[index] for index in key_indexes): row for row in current.execute(query, args)}
            right = {tuple(row[index] for index in key_indexes): row for row in candidate.execute(query, args)}
            for key in sorted(set(left) | set(right), key=repr):
                if left.get(key) == right.get(key):
                    continue
                before = dict(zip(columns, left[key], strict=True)) if key in left else None
                after = dict(zip(columns, right[key], strict=True)) if key in right else None
                operation = (
                    DifferenceOperation.ADDED
                    if before is None
                    else DifferenceOperation.REMOVED
                    if after is None
                    else DifferenceOperation.CHANGED
                )
                changed = tuple(
                    column
                    for column in columns
                    if before is None or after is None or before.get(column) != after.get(column)
                )
                differences.append(
                    RowDifference(table, operation, tuple(zip(key_columns, key, strict=True)), before, after, changed)
                )
    return CanaryDiffReport(current_path, candidate_path, requested, tuple(tables), tuple(differences))


def write_canary_report(
    path: Path, *, selection: CanarySelection, comparison: CanaryDiffReport, rebuild_receipt: Mapping[str, object]
) -> dict[str, object]:
    payload = {
        "version": 1,
        "selection": selection.to_dict(),
        "comparison": comparison.to_dict(),
        "rebuild_receipt": dict(rebuild_receipt),
        "no_promotion": True,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    temporary.replace(path)
    return payload


def load_canary_report(path: Path, **_: object) -> dict[str, object]:
    payload = cast(dict[str, object], json.loads(Path(path).read_text(encoding="utf-8")))
    if payload.get("no_promotion") is not True:
        raise CanarySelectionError("canary report lacks no-promotion evidence")
    return payload


__all__ = [
    "CanaryDiffReport",
    "CanaryRunResult",
    "CanarySelection",
    "CanarySelectionError",
    "DifferenceOperation",
    "RowDifference",
    "compare_reindex_generations",
    "load_canary_report",
    "run_reindex_canary",
    "select_canary_sessions",
    "write_canary_report",
]
