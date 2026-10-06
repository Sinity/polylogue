"""Bounded inputs for canonical accepted-frontier inspection.

All reads borrow the caller's original Source/Index/Ops frame. A journal
watermark describes coverage, never acceptance of a Source mutation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.index_generation import ActiveWriterLease
    from polylogue.storage.raw_authority import RawReplayPlan
    from polylogue.storage.raw_reconciler import RawAuthorityFrontierItem
    from polylogue.storage.raw_retention import _CursorFrontierComparison, _IndexRawRevisionHead, _OpsCursorAuthority
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner
    from polylogue.storage.sqlite.reference_seal import KnownTierCell, KnownTierMutationPermit, PreparedIndexMutation


@dataclass(frozen=True, slots=True)
class FrontierJournalState:
    authority_identity: str
    source_high: int
    source_floor: int
    index_high: int
    index_floor: int
    cursor_high: int
    cursor_floor: int

    def __post_init__(self) -> None:
        for floor, high in (
            (self.source_floor, self.source_high),
            (self.index_floor, self.index_high),
            (self.cursor_floor, self.cursor_high),
        ):
            if not 0 <= floor <= high:
                raise ValueError("frontier journal coverage is invalid")


#: Broken-head samples a findings document keeps; its counts stay exact.
_FINDINGS_SAMPLE_LIMIT = 10


@dataclass(frozen=True, slots=True)
class FrontierInspectionMark:
    authority_identity: str
    source_watermark: int
    index_watermark: int
    cursor_watermark: int
    state: str
    #: The pass's recorded findings document (:class:`FrontierInspectionFindings`).
    findings_document: str | None = None


def read_frontier_inspection_mark(conn: sqlite3.Connection, *, schema: str = "main") -> FrontierInspectionMark | None:
    if schema not in {"main", "ops_tier"}:
        raise ValueError("unsupported frontier mark authority schema")
    # Missing table is unavailable evidence; callers retain the SQL failure,
    # rather than translating it to an empty healthy frontier.
    with closing(
        conn.execute(
            "SELECT authority_identity,source_watermark,index_watermark,cursor_watermark,state,detail "
            f"FROM {schema}.raw_frontier_inspection WHERE singleton=1"
        )
    ) as cursor:
        row = cursor.fetchone()
    if row is None:
        return None
    return FrontierInspectionMark(
        str(row[0]), int(row[1]), int(row[2]), int(row[3]), str(row[4]), None if row[5] is None else str(row[5])
    )


def frontier_inspection_mode(
    state: FrontierJournalState,
    mark: FrontierInspectionMark | None,
) -> Literal["full", "delta", "current"]:
    if mark is None or mark.state != "healthy" or mark.authority_identity != state.authority_identity:
        return "full"
    previous = (mark.source_watermark, mark.index_watermark, mark.cursor_watermark)
    highs = (state.source_high, state.index_high, state.cursor_high)
    floors = (state.source_floor, state.index_floor, state.cursor_floor)
    if any(not floor <= prior <= high for prior, floor, high in zip(previous, floors, highs, strict=True)):
        return "full"
    return "current" if previous == highs else "delta"


def changed_frontier_raw_ids(
    conn: sqlite3.Connection,
    state: FrontierJournalState,
    mark: FrontierInspectionMark,
) -> Iterator[str]:
    if frontier_inspection_mode(state, mark) != "delta":
        raise ValueError("changed-key inspection requires complete retained journal coverage")
    # UNION deduplicates in the existing SQLite input owner; the cursor is
    # streamed and closed even when its consumer abandons inspection.
    with closing(
        conn.execute(
            "SELECT raw_id FROM main.raw_existence_changes WHERE sequence>? AND sequence<=? "
            "UNION SELECT raw_id FROM index_tier.raw_existence_changes WHERE sequence>? AND sequence<=? "
            "ORDER BY raw_id",
            (mark.source_watermark, state.source_high, mark.index_watermark, state.index_high),
        )
    ) as cursor:
        for row in cursor:
            yield str(row[0])


def changed_frontier_paths(
    conn: sqlite3.Connection,
    state: FrontierJournalState,
    mark: FrontierInspectionMark,
) -> Iterator[str]:
    if frontier_inspection_mode(state, mark) != "delta":
        raise ValueError("changed-path inspection requires complete retained journal coverage")
    with closing(
        conn.execute(
            "SELECT DISTINCT source_path FROM ops_tier.raw_frontier_cursor_changes "
            "WHERE sequence>? AND sequence<=? ORDER BY source_path",
            (mark.cursor_watermark, state.cursor_high),
        )
    ) as cursor:
        for row in cursor:
            yield str(row[0])


class PreparedFrontierSource:
    """Finite frontier reads and receipt writes on the original Source tape."""

    def __init__(self, seal: PreparedIndexMutation) -> None:
        # The caller supplies the actual PreparedIndexMutation and its active
        # original read/Source producer windows. No connection is manufactured.
        from polylogue.storage.sqlite.archive_tiers.revision_governance import _PreparedSourceProducer

        self._producer = _PreparedSourceProducer(seal)
        self._seal = seal

    def _statement_operands(
        self,
        sql: str,
        columns: tuple[str, ...],
        values: tuple[object, ...],
    ) -> tuple[str, tuple[object, ...], dict[str, KnownTierCell]]:
        # These are the fixed canonical frontier INSERT/UPDATE statements,
        # whose positional operands all become original literal cells.
        parts = sql.split("?")
        if len(parts) != len(values) + 1 or len(columns) != len(values):
            raise ValueError("canonical frontier statement operand mismatch")
        expressions: list[str] = []
        parameters: list[object] = []
        cells = {}
        for column, value in zip(columns, values, strict=True):
            if value is not None and not isinstance(value, (int, float, str, bytes)):
                raise TypeError("frontier statement requires a canonical scalar operand")
            cell = self._seal.retain_literal_scalar(value)
            cells[column] = cell
            expression, operand = self._seal.source_literal_expression(cell)
            expressions.append(expression)
            parameters.extend(operand)
        prepared_sql = parts[0] + "".join(
            expression + part for expression, part in zip(expressions, parts[1:], strict=True)
        )
        return prepared_sql, tuple(parameters), cells

    def blob_receipt_fingerprint(self, digest: str) -> tuple[int, int, int, int, int] | None:
        blob_hash = bytes.fromhex(digest)
        self._producer._load_artifact_inputs(
            "verified_blob_receipts",
            "SELECT rowid FROM verified_blob_receipts WHERE blob_hash=?",
            (blob_hash,),
        )
        with self._seal.source_rows(
            "SELECT st_dev,st_ino,st_size,st_mtime_ns,st_ctime_ns FROM verified_blob_receipts WHERE blob_hash=?",
            (blob_hash,),
        ) as rows:
            row = rows.fetchone()
        return None if row is None else (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]))

    def record_blob_receipt(
        self,
        digest: str,
        fingerprint: tuple[int, int, int, int, int],
        *,
        observed_at_ms: int,
    ) -> None:
        from polylogue.storage.raw_reconciler import _BLOB_RECEIPT_UPSERT_SQL

        blob_hash = bytes.fromhex(digest)
        # A receipt journal entry names every original Raw sharing the blob.
        # Load that exact predicate before the canonical trigger is captured.
        self._producer._load_artifact_inputs(
            "raw_sessions",
            "SELECT rowid FROM raw_sessions WHERE blob_hash=?",
            (blob_hash,),
        )
        self._producer._load_artifact_inputs(
            "verified_blob_receipts",
            "SELECT rowid FROM verified_blob_receipts WHERE blob_hash=?",
            (blob_hash,),
        )
        self._seal.source_allocation_dependencies("verified_blob_receipts")
        key = self._seal.retain_literal_scalar(blob_hash)
        sql, parameters, cells = self._statement_operands(
            _BLOB_RECEIPT_UPSERT_SQL,
            ("blob_hash", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "verified_at_ms"),
            (blob_hash, *fingerprint, observed_at_ms),
        )
        with self._seal.source_statement(
            sql,
            parameters,
            table="verified_blob_receipts",
            writable_targets=(("verified_blob_receipts", (key,)),),
            prepared_cells=cells,
        ):
            pass

    def prune_journal_input(self, watermark: int) -> int:
        """Capture consumed original roots and their exact floor effects in bounded pages."""
        self._producer._load_artifact_inputs(
            "raw_existence_journal_control", "SELECT rowid FROM raw_existence_journal_control", ()
        )
        after = 0
        count = 0
        while after < watermark:
            self._producer._load_artifact_inputs(
                "raw_existence_changes",
                "SELECT rowid FROM raw_existence_changes WHERE sequence>? AND sequence<=? ORDER BY sequence LIMIT 256",
                (after, watermark),
            )
            with self._seal.source_rows(
                "SELECT sequence FROM raw_existence_changes WHERE sequence>? AND sequence<=? ORDER BY sequence LIMIT 256",
                (after, watermark),
            ) as rows:
                page = tuple(int(row[0]) for row in rows)
            if not page:
                break
            for sequence in page:
                key = self._seal.retain_literal_scalar(sequence)
                with self._seal.source_statement(
                    "DELETE FROM raw_existence_changes WHERE sequence=?",
                    (sequence,),
                    table="raw_existence_changes",
                    writable_targets=(("raw_existence_changes", (key,)),),
                ):
                    pass
                count += 1
            after = page[-1]
        return count

    def frontier_blocker_identity(self, *, pass_id: str, plan_id: str) -> str:
        from polylogue.storage.raw_reconciler import _frontier_blocker_identity

        def resolved_at(blocker_id: str) -> tuple[object] | None:
            self._producer._load_artifact_inputs(
                "raw_authority_blockers",
                "SELECT rowid FROM raw_authority_blockers WHERE blocker_id=?",
                (blocker_id,),
            )
            with self._seal.source_rows(
                "SELECT resolved_at_ms FROM raw_authority_blockers WHERE blocker_id=?",
                (blocker_id,),
            ) as rows:
                row = rows.fetchone()
                return None if row is None else (row[0],)

        return _frontier_blocker_identity(resolved_at, pass_id=pass_id, plan_id=plan_id)

    def publish_obligation_input(
        self, item: RawAuthorityFrontierItem, *, pass_id: str, observed_at_ms: int
    ) -> str | None:
        from polylogue.storage.raw_reconciler import (
            _FRONTIER_BLOCKER_INSERT_SQL,
            _OBLIGATION_STATES,
            _frontier_obligation_values,
        )

        if item.state not in _OBLIGATION_STATES:
            return None
        blocker_id = self.frontier_blocker_identity(pass_id=pass_id, plan_id=item.plan_id)
        self._producer._load_artifact_inputs(
            "raw_authority_blockers",
            "SELECT rowid FROM raw_authority_blockers WHERE blocker_id=?",
            (blocker_id,),
        )
        self._seal.source_allocation_dependencies("raw_authority_blockers")
        key = self._seal.retain_literal_scalar(blocker_id)
        sql, parameters, cells = self._statement_operands(
            _FRONTIER_BLOCKER_INSERT_SQL,
            (
                "blocker_id",
                "plan_input_digest",
                "observed_pass_id",
                "reason",
                "expected_json",
                "observed_json",
                "created_at_ms",
            ),
            _frontier_obligation_values(item, blocker_id=blocker_id, pass_id=pass_id, observed_at_ms=observed_at_ms),
        )
        with self._seal.source_statement(
            sql,
            parameters,
            table="raw_authority_blockers",
            writable_targets=(("raw_authority_blockers", (key,)),),
            prepared_cells=cells,
        ):
            pass
        return blocker_id

    def resolve_obligation_input(self, blocker_id: str, *, pass_id: str, observed_at_ms: int) -> None:
        from polylogue.storage.raw_reconciler import (
            _FRONTIER_BLOCKER_RESOLVE_SQL,
            _frontier_resolution_values,
        )

        self._producer._load_artifact_inputs(
            "raw_authority_blockers",
            "SELECT rowid FROM raw_authority_blockers WHERE blocker_id=?",
            (blocker_id,),
        )
        key = self._seal.retain_literal_scalar(blocker_id)
        sql, parameters, cells = self._statement_operands(
            _FRONTIER_BLOCKER_RESOLVE_SQL,
            ("resolved_at_ms", "resolution", "blocker_id"),
            _frontier_resolution_values(blocker_id, pass_id=pass_id, observed_at_ms=observed_at_ms),
        )
        with self._seal.source_statement(
            sql,
            parameters,
            table="raw_authority_blockers",
            writable_targets=(("raw_authority_blockers", (key,)),),
            prepared_cells=cells,
        ):
            pass

    def frontier_input(self, logical_key: str) -> dict[str, object] | None:
        """Read one head and its exact original Source dependencies."""
        from polylogue.storage.raw_reconciler import _FRONTIER_INDEX_INPUT_SQL, _FRONTIER_SOURCE_INPUT_SQL

        self._seal.before_index_input(
            "raw_revision_heads",
            (
                "logical_source_key",
                "session_id",
                "accepted_raw_id",
                "accepted_source_revision",
                "accepted_content_hash",
                "accepted_frontier_kind",
                "accepted_frontier",
                "acquisition_generation",
                "append_end_offset",
                "decided_at_ms",
            ),
            "SELECT rowid FROM raw_revision_heads WHERE logical_source_key=?",
            (logical_key,),
        )
        self._seal.before_index_input(
            "sessions",
            ("origin", "raw_id", "content_hash", "message_count"),
            "SELECT s.rowid FROM sessions s JOIN raw_revision_heads h ON h.session_id=s.session_id "
            "WHERE h.logical_source_key=?",
            (logical_key,),
        )
        with self._seal.original_rows("index", _FRONTIER_INDEX_INPUT_SQL, (logical_key,)) as rows:
            names = tuple(column[0] for column in rows.description or ())
            row = rows.fetchone()
            head = None if row is None else dict(zip(names, row, strict=True))
        if head is None:
            return None
        raw_id = head["accepted_raw_id"]
        for table in (
            "raw_sessions",
            "raw_session_memberships",
            "raw_membership_census",
            "raw_authority_parser_census",
        ):
            self._producer._load_artifact_inputs(
                table,
                f"SELECT rowid FROM {table} WHERE raw_id=?",
                (raw_id,),
            )
        with self._seal.source_rows(_FRONTIER_SOURCE_INPUT_SQL, (logical_key, logical_key, raw_id)) as rows:
            names = tuple(column[0] for column in rows.description or ())
            row = rows.fetchone()
            if row is not None:
                head.update(zip(names, row, strict=True))
        return head

    def terminal_artifact_paths(self, paths: set[str]) -> set[str]:
        from polylogue.storage.raw_retention import _terminal_artifact_paths_from_inputs

        def read_rows(sql: str, parameters: tuple[object, ...], batch: tuple[str, ...]) -> tuple[sqlite3.Row, ...]:
            marks = ",".join("?" for _ in batch)
            self._producer._load_artifact_inputs(
                "raw_sessions",
                f"SELECT rowid FROM raw_sessions WHERE source_path IN ({marks})",
                batch,
            )
            self._producer._load_artifact_inputs(
                "blob_refs",
                f"SELECT rowid FROM blob_refs WHERE ref_type='raw_payload' AND source_path IN ({marks})",
                batch,
            )
            for table in ("raw_artifacts", "raw_membership_census"):
                self._producer._load_artifact_inputs(
                    table,
                    f"SELECT a.rowid FROM {table} a JOIN raw_sessions r ON r.raw_id=a.raw_id "
                    f"WHERE r.source_path IN ({marks})",
                    batch,
                )
            with self._seal.source_rows(sql, parameters) as rows:
                return tuple(rows)

        return _terminal_artifact_paths_from_inputs(read_rows, paths)

    def raw_revision_chain_rows(self, raw_ids: set[str]) -> dict[str, sqlite3.Row]:
        from polylogue.storage.raw_retention import _RAW_REVISION_CHAIN_COLUMNS, _raw_revision_rows_from_inputs

        def read_rows(batch: tuple[str, ...]) -> tuple[sqlite3.Row, ...]:
            marks = ",".join("?" for _ in batch)
            self._producer._load_artifact_inputs(
                "raw_sessions",
                f"SELECT rowid FROM raw_sessions WHERE raw_id IN ({marks})",
                batch,
            )
            with self._seal.source_rows(
                f"SELECT {_RAW_REVISION_CHAIN_COLUMNS} FROM raw_sessions WHERE raw_id IN ({marks})",
                batch,
            ) as rows:
                return tuple(rows)

        return _raw_revision_rows_from_inputs(read_rows, raw_ids, allow_missing=True)

    def changed_raw_ids(self, state: FrontierJournalState, mark: FrontierInspectionMark) -> Iterator[str]:
        from heapq import merge

        from polylogue.core.compute_cancel import check_compute_cancelled

        if frontier_inspection_mode(state, mark) != "delta":
            raise ValueError("changed-key inspection requires complete retained journal coverage")

        def keys(tier: str, low: int, high: int) -> Iterator[str]:
            after = None
            while True:
                with self._seal.original_rows(
                    tier,
                    "SELECT DISTINCT raw_id FROM raw_existence_changes WHERE sequence>? AND sequence<=? "
                    "AND (? IS NULL OR raw_id>?) ORDER BY raw_id LIMIT 256",
                    (low, high, after, after),
                ) as rows:
                    page = tuple(str(row[0]) for row in rows)
                if not page:
                    return
                yield from page
                after = page[-1]

        previous = None
        for raw_id in merge(
            keys("source", mark.source_watermark, state.source_high),
            keys("index", mark.index_watermark, state.index_high),
        ):
            check_compute_cancelled()
            if raw_id != previous:
                yield raw_id
            previous = raw_id

    def frontier_keys(self) -> Iterator[str]:
        """Page original head identities, closing each cursor before hydration."""
        from polylogue.core.compute_cancel import check_compute_cancelled

        after: str | None = None
        while True:
            with self._seal.original_rows(
                "index",
                "SELECT logical_source_key FROM raw_revision_heads "
                "WHERE (? IS NULL OR logical_source_key>?) ORDER BY logical_source_key LIMIT 256",
                (after, after),
            ) as rows:
                page = tuple(str(row[0]) for row in rows)
            if not page:
                return
            for key in page:
                check_compute_cancelled()
                yield key
            after = page[-1]

    def verify_frontier_bytes(self, blob_store: BlobStore, row: dict[str, object], *, observed_at_ms: int) -> bool:
        from polylogue.storage.raw_reconciler import _verify_blob_bytes

        if row.get("raw_origin") is None:
            return False
        raw_id = str(row["accepted_raw_id"])
        digest = str(row["blob_hash"]).lower()
        original_hash, original_length = self._seal.retain_original_blob_input(raw_id)
        if original_hash.hex() != digest or original_length != row["blob_size"]:
            raise ValueError("frontier bytes differ from their original retained Raw")
        return _verify_blob_bytes(
            blob_store,
            digest,
            retained_fingerprint=self.blob_receipt_fingerprint(digest),
            record_receipt=lambda value, fingerprint: self.record_blob_receipt(
                value,
                fingerprint,
                observed_at_ms=observed_at_ms,
            ),
        )


def _iter_selected_frontier_proofs(
    source: PreparedFrontierSource,
    selection: _FrontierSelection,
    blob_store: BlobStore,
    *,
    observed_at_ms: int,
) -> Iterator[tuple[RawAuthorityFrontierItem, tuple[object, ...]]]:
    """Classify selected original heads with the canonical byte-chain reducer."""
    from polylogue.core.compute_cancel import check_compute_cancelled
    from polylogue.storage.raw_reconciler import _classify_frontier_row
    from polylogue.storage.raw_retention import _check_broken_active_chain_inputs, _IndexRawRevisionHead

    selected_keys = selection.keys()
    for logical_key in selected_keys:
        check_compute_cancelled()
        row = source.frontier_input(logical_key)
        if row is None:
            continue
        item = _classify_frontier_row(
            row,
            verified_bytes=source.verify_frontier_bytes(blob_store, row, observed_at_ms=observed_at_ms),
        )
        head = _IndexRawRevisionHead(
            logical_source_key=logical_key,
            accepted_raw_id=str(row["head_accepted_raw_id"]),
            accepted_source_revision=str(row["accepted_source_revision"]),
            accepted_frontier_kind=str(row["accepted_frontier_kind"]),
            accepted_frontier=int(cast(int, row["accepted_frontier"])),
            acquisition_generation=int(cast(int, row["head_acquisition_generation"])),
            append_end_offset=cast(int | None, row["head_append_end_offset"]),
        )
        source._producer._load_artifact_inputs(
            "raw_sessions",
            "SELECT rowid FROM raw_sessions WHERE raw_id=?",
            (head.accepted_raw_id,),
        )
        with source._seal.source_rows(
            "SELECT source_path,canonical_source_path FROM raw_sessions WHERE raw_id=?",
            (head.accepted_raw_id,),
        ) as paths:
            path_row = paths.fetchone()
        comparison_path = None if path_row is None else str(path_row[1] or path_row[0])
        selection.retain_head(head, comparison_path)
        session_raw_id = row.get("session_raw_id")
        chain = _check_broken_active_chain_inputs(
            source.raw_revision_chain_rows,
            frozenset(() if session_raw_id is None else (str(session_raw_id),)),
            (head,),
            sample_limit=1,
        )
        yield item, cast(tuple[object, ...], chain)


def _iter_selected_cursor_proofs(
    source: PreparedFrontierSource,
    selection: _FrontierSelection,
    ops: sqlite3.Connection,
) -> Iterator[tuple[_OpsCursorAuthority, _CursorFrontierComparison, bool]]:
    """Consume all selected original cursor rows through the shared decision."""
    from polylogue.core.compute_cancel import check_compute_cancelled
    from polylogue.storage.raw_retention import _classify_cursor_frontier_input, _OpsCursorAuthority

    for selected_path in selection.paths():
        after = None
        while True:
            with closing(
                ops.execute(
                    "SELECT source_path,byte_offset,deferred_end_offset,canonical_source_path FROM ingest_cursor "
                    "WHERE byte_offset IS NOT NULL AND COALESCE(excluded,0)=0 AND (source_path=? OR canonical_source_path=?) "
                    "AND (? IS NULL OR source_path>?) ORDER BY source_path LIMIT 256",
                    (selected_path, selected_path, after, after),
                )
            ) as rows:
                page = tuple(tuple(row) for row in rows)
            if not page:
                break
            for cells in page:
                check_compute_cancelled()
                path = str(cells[0])
                after = path
                if not selection.begin_cursor(path):
                    continue
                cursor = _OpsCursorAuthority(
                    path,
                    int(cells[1]),
                    None if cells[2] is None else int(cells[2]),
                    None if cells[3] is None else str(cells[3]),
                )
                comparison_path = cursor.canonical_source_path or path
                source._producer._load_artifact_inputs(
                    "raw_sessions",
                    "SELECT rowid FROM raw_sessions WHERE source_path=? OR canonical_source_path=?",
                    (path, comparison_path),
                )
                with source._seal.source_rows(
                    "SELECT 1 FROM raw_sessions WHERE source_path=? OR canonical_source_path=? LIMIT 1",
                    (path, comparison_path),
                ) as retained:
                    retained_path = retained.fetchone() is not None
                terminal_paths = source.terminal_artifact_paths({path})
                comparison = _classify_cursor_frontier_input(
                    cursor,
                    selection.byte_heads_for_path(comparison_path),
                    has_any_head=selection.has_head_path(comparison_path),
                    terminal_artifact=path in terminal_paths,
                )
                yield cursor, comparison, retained_path


def _journal_coverage(
    conn: sqlite3.Connection,
    *,
    journal: str,
    control: str,
    expected_triggers: dict[str, str],
    schema: str = "main",
) -> tuple[int, int]:
    from polylogue.storage.frontier_existence import _normalized_sql

    if schema not in {"main", "source", "source_tier", "index_tier", "ops_tier"}:
        raise ValueError("unsupported frontier journal authority schema")
    with closing(conn.execute(f"SELECT name,sql FROM {schema}.sqlite_schema WHERE type='trigger'")) as rows:
        prefix = "raw_existence_" if journal == "raw_existence_changes" else "raw_frontier_cursor_"
        actual = {str(row[0]): _normalized_sql(str(row[1])) for row in rows if str(row[0]).startswith(prefix)}
    if actual != expected_triggers:
        raise ValueError("frontier dependency journal trigger contract is unavailable")
    with closing(conn.execute(f"SELECT retained_floor FROM {schema}.{control} WHERE singleton=1")) as rows:
        floor_row = rows.fetchone()
    if floor_row is None:
        raise ValueError("frontier dependency journal coverage is unavailable")
    with closing(conn.execute(f"SELECT seq FROM {schema}.sqlite_sequence WHERE name=?", (journal,))) as rows:
        high_row = rows.fetchone()
    high, floor = (0 if high_row is None else int(high_row[0])), int(floor_row[0])
    if not 0 <= floor <= high:
        raise ValueError("frontier dependency journal coverage is invalid")
    return high, floor


def read_frontier_journal_state(
    *,
    source: sqlite3.Connection,
    index: sqlite3.Connection,
    ops: sqlite3.Connection,
    source_path: Path,
    index_path: Path,
    ops_path: Path,
    source_triggers: dict[str, str],
    index_triggers: dict[str, str],
    cursor_triggers: dict[str, str],
    source_schema: str = "main",
    index_schema: str = "main",
    ops_schema: str = "main",
) -> FrontierJournalState:
    """Measure coverage on the three original pinned handles, without reopening paths."""
    from polylogue.storage.sqlite.archive_tiers.schema_identity import DerivedTier, derived_schema_identity

    identities: list[object] = []
    for conn, path, tier, schema in (
        (source, source_path, "source", source_schema),
        (index, index_path, "index", index_schema),
        (ops, ops_path, "ops", ops_schema),
    ):
        if schema not in {"main", "source", "source_tier", "index_tier", "ops_tier"}:
            raise ValueError("unsupported frontier authority schema")
        stat = path.stat()
        with closing(conn.execute("PRAGMA database_list")) as rows:
            actual_path = next((Path(str(row[2])).resolve() for row in rows if row[1] == schema), None)
        if actual_path != path.resolve():
            raise ValueError("frontier journal handle does not own its declared tier")
        with closing(conn.execute(f"PRAGMA {schema}.schema_version")) as rows:
            schema_version = int(rows.fetchone()[0])
        with closing(conn.execute(f"PRAGMA {schema}.user_version")) as rows:
            user_version = int(rows.fetchone()[0])
        identity = None
        if tier != "source":
            with closing(conn.execute(f"SELECT identity FROM {schema}.schema_identity WHERE tier=?", (tier,))) as rows:
                row = rows.fetchone()
            if row is None or str(row[0]) != derived_schema_identity(DerivedTier(tier)):
                raise ValueError("frontier derived authority identity is unavailable")
            identity = str(row[0])
        identities.append((str(path.resolve()), stat.st_dev, stat.st_ino, schema_version, user_version, identity))
    source_high, source_floor = _journal_coverage(
        source,
        journal="raw_existence_changes",
        control="raw_existence_journal_control",
        expected_triggers=source_triggers,
        schema=source_schema,
    )
    index_high, index_floor = _journal_coverage(
        index,
        journal="raw_existence_changes",
        control="raw_existence_journal_control",
        expected_triggers=index_triggers,
        schema=index_schema,
    )
    cursor_high, cursor_floor = _journal_coverage(
        ops,
        journal="raw_frontier_cursor_changes",
        control="raw_frontier_cursor_journal_control",
        expected_triggers=cursor_triggers,
        schema=ops_schema,
    )
    identity = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
    return FrontierJournalState(identity, source_high, source_floor, index_high, index_floor, cursor_high, cursor_floor)


def declared_frontier_triggers(ddl: str) -> dict[str, str]:
    from polylogue.storage.frontier_existence import _expected_triggers

    return {str(name): str(sql) for name, sql in _expected_triggers(ddl).items()}


class _FrontierSelection:
    """Disposable, paged inspection keys owned by the original SQL parent."""

    def __init__(
        self,
        seal: PreparedIndexMutation,
        path: Path,
        *,
        scratch_directory: tempfile.TemporaryDirectory[str],
    ) -> None:
        from polylogue.storage.sqlite.connection_profile import open_scratch_connection

        self._owner = open_scratch_connection(
            path,
            terminal_parent=seal,
            scratch_directory=scratch_directory,
        )
        conn = self._owner.require_connection()
        conn.executescript(
            "CREATE TABLE selected_raws(raw_id TEXT PRIMARY KEY, expanded INTEGER NOT NULL DEFAULT 0, membership_expanded INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID;"
            "CREATE INDEX selected_raw_pending ON selected_raws(expanded,raw_id);"
            "CREATE TABLE selected_keys(logical_key TEXT PRIMARY KEY) WITHOUT ROWID;"
            "CREATE TABLE selected_paths(source_path TEXT PRIMARY KEY) WITHOUT ROWID;"
            "CREATE TABLE frontier_proofs(ordinal INTEGER PRIMARY KEY,plan_id TEXT UNIQUE NOT NULL, "
            "state TEXT NOT NULL,obligation INTEGER NOT NULL,payload TEXT NOT NULL);"
            "CREATE TABLE frontier_heads(logical_key TEXT PRIMARY KEY,raw_id TEXT NOT NULL,"
            "source_revision TEXT NOT NULL,kind TEXT NOT NULL,frontier INTEGER NOT NULL,"
            "generation INTEGER NOT NULL,append_end INTEGER,comparison_path TEXT) WITHOUT ROWID;"
            "CREATE INDEX frontier_heads_by_path ON frontier_heads(comparison_path,logical_key);"
            "CREATE TABLE checked_cursor_paths(source_path TEXT PRIMARY KEY) WITHOUT ROWID;"
        )

    def add_raw(self, raw_id: str) -> None:
        with closing(
            self._owner.require_connection().execute(
                "INSERT INTO selected_raws(raw_id) VALUES (?) ON CONFLICT DO NOTHING",
                (raw_id,),
            )
        ):
            pass

    def add_key(self, key: str) -> None:
        with closing(
            self._owner.require_connection().execute(
                "INSERT INTO selected_keys(logical_key) VALUES (?) ON CONFLICT DO NOTHING",
                (key,),
            )
        ):
            pass

    def add_path(self, path: str) -> None:
        with closing(
            self._owner.require_connection().execute(
                "INSERT INTO selected_paths(source_path) VALUES (?) ON CONFLICT DO NOTHING",
                (path,),
            )
        ):
            pass

    def membership_expanded(self, raw_id: str) -> bool:
        with closing(
            self._owner.require_connection().execute(
                "SELECT membership_expanded FROM selected_raws WHERE raw_id=?",
                (raw_id,),
            )
        ) as rows:
            return bool(rows.fetchone()[0])

    def mark_membership_expanded(self, raw_id: str) -> None:
        with closing(
            self._owner.require_connection().execute(
                "UPDATE selected_raws SET membership_expanded=1 WHERE raw_id=?",
                (raw_id,),
            )
        ):
            pass

    def next_raw(self) -> str | None:
        with closing(
            self._owner.require_connection().execute(
                "SELECT raw_id FROM selected_raws WHERE expanded=0 ORDER BY raw_id LIMIT 1",
            )
        ) as rows:
            row = rows.fetchone()
        return None if row is None else str(row[0])

    def expanded(self, raw_id: str) -> None:
        with closing(
            self._owner.require_connection().execute(
                "UPDATE selected_raws SET expanded=1 WHERE raw_id=?",
                (raw_id,),
            )
        ):
            pass

    def keys(self) -> Iterator[str]:
        after = None
        while True:
            with closing(
                self._owner.require_connection().execute(
                    "SELECT logical_key FROM selected_keys WHERE (? IS NULL OR logical_key>?) "
                    "ORDER BY logical_key LIMIT 256",
                    (after, after),
                )
            ) as rows:
                page = tuple(str(row[0]) for row in rows)
            if not page:
                return
            yield from page
            after = page[-1]

    def paths(self) -> Iterator[str]:
        after = None
        while True:
            with closing(
                self._owner.require_connection().execute(
                    "SELECT source_path FROM selected_paths WHERE (? IS NULL OR source_path>?) "
                    "ORDER BY source_path LIMIT 256",
                    (after, after),
                )
            ) as rows:
                page = tuple(str(row[0]) for row in rows)
            if not page:
                return
            yield from page
            after = page[-1]

    def retain_head(self, head: _IndexRawRevisionHead, comparison_path: str | None) -> None:
        with closing(
            self._owner.require_connection().execute(
                "INSERT INTO frontier_heads VALUES(?,?,?,?,?,?,?,?)",
                (
                    head.logical_source_key,
                    head.accepted_raw_id,
                    head.accepted_source_revision,
                    head.accepted_frontier_kind,
                    head.accepted_frontier,
                    head.acquisition_generation,
                    head.append_end_offset,
                    comparison_path,
                ),
            )
        ):
            pass

    def has_head_path(self, path: str) -> bool:
        with closing(
            self._owner.require_connection().execute(
                "SELECT 1 FROM frontier_heads WHERE comparison_path=? LIMIT 1",
                (path,),
            )
        ) as rows:
            return rows.fetchone() is not None

    def byte_heads_for_path(self, path: str) -> Iterator[_IndexRawRevisionHead]:
        from polylogue.storage.raw_retention import _IndexRawRevisionHead

        after = None
        while True:
            with closing(
                self._owner.require_connection().execute(
                    "SELECT logical_key,raw_id,source_revision,kind,frontier,generation,append_end "
                    "FROM frontier_heads WHERE comparison_path=? AND kind='byte' "
                    "AND (? IS NULL OR logical_key>?) ORDER BY logical_key LIMIT 256",
                    (path, after, after),
                )
            ) as rows:
                page = tuple(tuple(row) for row in rows)
            if not page:
                return
            for cells in page:
                yield _IndexRawRevisionHead(*cells)
                after = str(cells[0])

    def begin_cursor(self, source_path: str) -> bool:
        with closing(
            self._owner.require_connection().execute(
                "INSERT OR IGNORE INTO checked_cursor_paths VALUES(?)",
                (source_path,),
            )
        ) as result:
            return result.rowcount == 1

    def retain_proof(self, item: RawAuthorityFrontierItem) -> None:
        from polylogue.storage.raw_reconciler import _OBLIGATION_STATES, _canonical_json

        with closing(
            self._owner.require_connection().execute(
                "INSERT INTO frontier_proofs(plan_id,state,obligation,payload) VALUES(?,?,?,?)",
                (
                    item.plan_id,
                    item.state.value,
                    int(item.state in _OBLIGATION_STATES),
                    _canonical_json(item.to_dict()),
                ),
            )
        ):
            pass

    def proof_payloads(self, *, obligations_only: bool = False) -> Iterator[str]:
        after = 0
        while True:
            with closing(
                self._owner.require_connection().execute(
                    "SELECT ordinal,payload FROM frontier_proofs WHERE ordinal>? "
                    "AND (?=0 OR obligation=1) ORDER BY ordinal LIMIT 256",
                    (after, int(obligations_only)),
                )
            ) as rows:
                page = tuple((int(row[0]), str(row[1])) for row in rows)
            if not page:
                return
            for ordinal, payload in page:
                yield payload
                after = ordinal

    def proof_inventory_digest(self) -> str:
        digest = hashlib.sha256(b"[")
        first = True
        for payload in self.proof_payloads():
            if not first:
                digest.update(b",")
            digest.update(payload.encode())
            first = False
        digest.update(b"]")
        return digest.hexdigest()

    def contains_plan(self, plan_id: str) -> bool:
        with closing(
            self._owner.require_connection().execute(
                "SELECT 1 FROM frontier_proofs WHERE plan_id=? AND obligation=1",
                (plan_id,),
            )
        ) as rows:
            return rows.fetchone() is not None

    def close(self) -> None:
        self._owner.close()


@dataclass(slots=True)
class _PreparedFrontierInspectionFrame:
    """The one admitted creator's original proof and disposable work index."""

    seal: PreparedIndexMutation
    ops_owner: NativeSQLCustodyOwner
    selection: _FrontierSelection
    directory: tempfile.TemporaryDirectory[str]
    ops_identity: tuple[int, int]

    def close_payload(self) -> None:
        # A failed native close keeps the directory and terminal parent alive.
        # Never remove scratch files underneath an unsettled native child.
        failures: list[BaseException] = []
        for owner in (self.selection, self.ops_owner):
            try:
                owner.close()
            except BaseException as failure:
                failures.append(failure)
        if failures:
            raise BaseExceptionGroup("frontier native payload cleanup failed", failures)
        self.directory.cleanup()

    def close(self) -> None:
        self.seal.close()
        self.close_payload()


def _prepare_frontier_inspection_frame(
    archive_root: Path,
    *,
    input_demand: Callable[[int], None],
) -> _PreparedFrontierInspectionFrame:
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.archive_identity import resolve_active_index_path
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import _open_readonly_owner, owned_daemon_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    def initialize_ops_writer_profile() -> None:
        # The canonical writer may change DELETE mode to WAL. Complete that
        # physical profile initialization before pinning the original Ops read.
        # Publication still reserves and validates the same original operands.
        with owned_daemon_connection(archive_root / "ops.db", archive_root=archive_root):
            pass

    admit_stage_write("raw.frontier.ops.prepare", initialize_ops_writer_profile)
    directory = tempfile.TemporaryDirectory(prefix="polylogue-frontier-")
    seal = None
    ops_owner = None
    selection = None
    ops_descriptor = None
    try:
        seal = PreparedIndexMutation(
            resolve_active_index_path(archive_root),
            archive_root=archive_root,
            input_demand=input_demand,
        )
        ops_descriptor = os.open(archive_root / "ops.db", os.O_RDONLY | os.O_CLOEXEC)
        ops_stat = os.fstat(ops_descriptor)
        ops_identity = (ops_stat.st_dev, ops_stat.st_ino)
        ops_owner = _open_readonly_owner(
            archive_root / "ops.db",
            tier=ArchiveTier.OPS,
            terminal_parent=seal,
            lifetime_dependencies=(directory,),
            opened_main_fd=ops_descriptor,
        )
        ops_owner.retain_anchored_descriptor(ops_descriptor)
        ops_descriptor = None
        selection = _FrontierSelection(
            seal,
            Path(directory.name) / "frontier.sqlite",
            scratch_directory=directory,
        )
        return _PreparedFrontierInspectionFrame(seal, ops_owner, selection, directory, ops_identity)
    except BaseException as primary:
        failures = []
        for owner in (selection, ops_owner, seal):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as failure:
                    failures.append(failure)
        if ops_descriptor is not None:
            try:
                os.close(ops_descriptor)
            except BaseException as failure:
                failures.append(failure)
        if not failures:
            try:
                directory.cleanup()
            except BaseException as failure:
                failures.append(failure)
        if failures:
            raise BaseExceptionGroup(
                "frontier preparation and native cleanup failed", [primary, *failures]
            ) from primary
        raise


def populate_changed_frontier_selection(
    source: PreparedFrontierSource,
    selection: _FrontierSelection,
    selected_read: PreparedSessionSourceRead,
    state: FrontierJournalState,
    mark: FrontierInspectionMark,
) -> bool:
    """Expand invalidation inputs; return false when deletion requires a full path proof.

    Membership expansion stays in the canonical selected Source reader. This
    closure chooses which original operands to inspect and makes no new
    revision or acceptance decision.
    """
    from polylogue.core.compute_cancel import check_compute_cancelled

    for changed_raw_id in source.changed_raw_ids(state, mark):
        selection.add_raw(changed_raw_id)
    while (raw_id := selection.next_raw()) is not None:
        check_compute_cancelled()
        with source._seal.original_rows(
            "source",
            "SELECT source_path,canonical_source_path,predecessor_raw_id,baseline_raw_id "
            "FROM raw_sessions WHERE raw_id=?",
            (raw_id,),
        ) as rows:
            row = rows.fetchone()
        if row is None:
            # The journal intentionally carries a Raw identity, not a second
            # durable historical path. A removed row cannot certify which
            # cursor paths remain safe, so retain full inspection.
            return False
        for path in row[:2]:
            if path is not None:
                selection.add_path(str(path))
        for dependency in row[2:]:
            if dependency is not None:
                selection.add_raw(str(dependency))
        if not selection.membership_expanded(raw_id):
            component, keys = selected_read.expand_raw_membership_selection((raw_id,))
            for member in component:
                selection.add_raw(member)
                selection.mark_membership_expanded(member)
            for key in keys:
                selection.add_key(key)
        after = None
        while True:
            with source._seal.original_rows(
                "source",
                "SELECT raw_id FROM raw_sessions WHERE (predecessor_raw_id=? OR baseline_raw_id=?) "
                "AND (? IS NULL OR raw_id>?) ORDER BY raw_id LIMIT 256",
                (raw_id, raw_id, after, after),
            ) as rows:
                page = tuple(str(value[0]) for value in rows)
            if not page:
                break
            for dependent in page:
                selection.add_raw(dependent)
            after = page[-1]
        with source._seal.original_rows(
            "index",
            "SELECT logical_source_key FROM raw_revision_heads WHERE accepted_raw_id=? ORDER BY logical_source_key",
            (raw_id,),
        ) as rows:
            keys = tuple(str(value[0]) for value in rows)
        for key in keys:
            selection.add_key(key)
        selection.expanded(raw_id)
    return True


def _changed_cursor_paths(
    ops: sqlite3.Connection,
    state: FrontierJournalState,
    mark: FrontierInspectionMark,
) -> Iterator[str]:
    after = None
    while True:
        with closing(
            ops.execute(
                "SELECT DISTINCT source_path FROM raw_frontier_cursor_changes WHERE sequence>? AND sequence<=? "
                "AND (? IS NULL OR source_path>?) ORDER BY source_path LIMIT 256",
                (mark.cursor_watermark, state.cursor_high, after, after),
            )
        ) as rows:
            page = tuple(str(row[0]) for row in rows)
        if not page:
            return
        yield from page
        after = page[-1]


def populate_frontier_selection(
    source: PreparedFrontierSource,
    selection: _FrontierSelection,
    selected_read: PreparedSessionSourceRead,
    ops: sqlite3.Connection,
    state: FrontierJournalState,
    mark: FrontierInspectionMark | None,
) -> Literal["full", "delta", "current"]:
    mode = frontier_inspection_mode(state, mark)
    if mode == "current":
        return mode
    if mode == "delta":
        assert mark is not None
        for path in _changed_cursor_paths(ops, state, mark):
            selection.add_path(path)
            after = None
            while True:
                with source._seal.original_rows(
                    "source",
                    "SELECT raw_id FROM raw_sessions WHERE (source_path=? OR canonical_source_path=?) "
                    "AND (? IS NULL OR raw_id>?) ORDER BY raw_id LIMIT 256",
                    (path, path, after, after),
                ) as rows:
                    page = tuple(str(row[0]) for row in rows)
                if not page:
                    break
                for raw_id in page:
                    selection.add_raw(raw_id)
                after = page[-1]
        if populate_changed_frontier_selection(source, selection, selected_read, state, mark):
            return mode
        mode = "full"
    # The full bootstrap and unavailable/truncated-coverage route stream the
    # same original head/path inventories. No file count changes its outcome.
    for key in source.frontier_keys():
        selection.add_key(key)
    after = None
    while True:
        with closing(
            ops.execute(
                "SELECT source_path FROM ingest_cursor WHERE (? IS NULL OR source_path>?) "
                "ORDER BY source_path LIMIT 256",
                (after, after),
            )
        ) as rows:
            page = tuple(str(row[0]) for row in rows)
        if not page:
            break
        for path in page:
            selection.add_path(path)
        after = page[-1]
    return mode


def frontier_inspection_projection(
    conn: sqlite3.Connection,
    *,
    archive_root: Path,
    source_schema: str = "source_tier",
    index_schema: str = "main",
    ops_schema: str = "ops_tier",
) -> dict[str, object]:
    """Read the shared coverage reducer from one already pinned tier frame."""
    return frontier_inspection_projection_from_connections(
        conn,
        conn,
        conn,
        archive_root=archive_root,
        source_schema=source_schema,
        index_schema=index_schema,
        ops_schema=ops_schema,
    )


def frontier_inspection_projection_from_connections(
    source: sqlite3.Connection,
    index: sqlite3.Connection,
    ops: sqlite3.Connection,
    *,
    archive_root: Path,
    source_schema: str = "main",
    index_schema: str = "main",
    ops_schema: str = "main",
) -> dict[str, object]:
    """Read coverage from the status owner's same pinned tier frame.

    Missing, replaced, failed, or pruned evidence never becomes an empty
    healthy frontier. This performs only journal/identity reads, not a corpus
    inspection or a filesystem blob walk.
    """
    from polylogue.core.errors import SchemaRefusalError
    from polylogue.core.evidence import Measured, Unavailable
    from polylogue.storage.archive_identity import resolve_active_index_path
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
    from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.tier_access import capture_sqlite_read

    def read_frame() -> tuple[FrontierJournalState, FrontierInspectionMark | None]:
        state = read_frontier_journal_state(
            source=source,
            index=index,
            ops=ops,
            source_path=archive_root / "source.db",
            index_path=resolve_active_index_path(archive_root),
            ops_path=archive_root / "ops.db",
            source_schema=source_schema,
            index_schema=index_schema,
            ops_schema=ops_schema,
            source_triggers=declared_frontier_triggers(ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE]),
            index_triggers=declared_frontier_triggers(INDEX_DDL),
            cursor_triggers=declared_frontier_triggers(OPS_DDL),
        )
        return state, read_frontier_inspection_mark(ops, schema=ops_schema)

    try:
        frame = capture_sqlite_read(read_frame)
    except (OSError, ValueError, SchemaRefusalError) as failure:
        return {"available": False, "current": False, "healthy": False, "detail": str(failure)}
    if not isinstance(frame, Measured):
        detail = frame.detail if isinstance(frame, Unavailable) else None
        return {"available": False, "current": False, "healthy": False, "detail": detail or "sqlite_read_failed"}
    state, mark = frame.value
    if mark is None:
        return {
            "available": False,
            "current": False,
            "healthy": False,
            "detail": "accepted frontier has not been inspected",
        }
    # A blocked mark remains a truthful refusal, but only a healthy completed
    # mark certifies the fast unchanged path. Retryable failures never advance
    # that certificate.
    current = (
        mark.authority_identity == state.authority_identity
        and (mark.source_watermark, mark.index_watermark, mark.cursor_watermark)
        == (state.source_high, state.index_high, state.cursor_high)
        and all(
            floor <= watermark <= high
            for floor, watermark, high in (
                (state.source_floor, mark.source_watermark, state.source_high),
                (state.index_floor, mark.index_watermark, state.index_high),
                (state.cursor_floor, mark.cursor_watermark, state.cursor_high),
            )
        )
    )
    coverage: dict[str, object] = {
        "available": True,
        "current": current,
        "healthy": current and mark.state == "healthy",
        "state": mark.state,
    }
    if current:
        from polylogue.storage.raw_retention import FrontierInspectionFindings

        # The certificate's own categories. A document that does not decode
        # leaves the categories unknown; it never reads as healthy.
        try:
            if mark.findings_document is None:
                raise ValueError("frontier inspection recorded no findings")
            coverage["findings"] = FrontierInspectionFindings.from_document(mark.findings_document)
        except ValueError as failure:
            coverage["detail"] = f"frontier inspection findings unreadable: {failure}"
    return coverage


def publish_frontier_inspection_mark(
    seal: PreparedIndexMutation,
    permit: KnownTierMutationPermit | None,
    *,
    ops_observer: sqlite3.Connection,
    original_ops_version: int,
    original_ops_identity: tuple[int, int],
    state: FrontierJournalState,
    accepted_source_watermark: int,
    healthy: bool,
    observed_at_ms: int,
    detail: str,
) -> None:
    """Commit the prepared Source proof before its disposable coverage mark.

    The Ops writer reserves cursor publication before accepting Source, so a
    cursor cannot slip between the original proof and this mark. A Source
    success followed by a mark failure stays unmeasured and retryable.
    """
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL
    from polylogue.storage.sqlite.archive_tiers.revision_governance import publish_prepared_revision_source
    from polylogue.storage.sqlite.connection_profile import owned_daemon_connection
    from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError

    def publish() -> None:
        with owned_daemon_connection(seal.archive_root / "ops.db", archive_root=seal.archive_root) as writer, writer:
            with closing(writer.execute("BEGIN IMMEDIATE")):
                pass
            ops_stat = (seal.archive_root / "ops.db").stat()
            if (ops_stat.st_dev, ops_stat.st_ino) != original_ops_identity:
                raise ReferenceSealStaleError("frontier Ops namespace changed before original publication")
            with closing(ops_observer.execute("PRAGMA data_version")) as rows:
                current_version = int(rows.fetchone()[0])
            if current_version != original_ops_version:
                raise ReferenceSealStaleError("frontier cursor inputs changed before original publication")
            high, floor = _journal_coverage(
                writer,
                journal="raw_frontier_cursor_changes",
                control="raw_frontier_cursor_journal_control",
                expected_triggers=declared_frontier_triggers(OPS_DDL),
            )
            if high != state.cursor_high or floor != state.cursor_floor:
                raise ReferenceSealStaleError("frontier cursor journal changed before original publication")
            if permit is None:
                # An unchanged healthy mark needs reservation and currency,
                # but creates neither a Source write nor a fresh Ops ledger row.
                seal.validate_observers_current()
                return
            publish_prepared_revision_source(seal, permit)
            with closing(
                writer.execute(
                    "INSERT INTO raw_frontier_inspection(singleton,authority_identity,source_watermark,"
                    "index_watermark,cursor_watermark,state,inspected_at_ms,detail) VALUES(1,?,?,?,?,?,?,?) "
                    "ON CONFLICT(singleton) DO UPDATE SET authority_identity=excluded.authority_identity,"
                    "source_watermark=excluded.source_watermark,index_watermark=excluded.index_watermark,"
                    "cursor_watermark=excluded.cursor_watermark,state=excluded.state,"
                    "inspected_at_ms=excluded.inspected_at_ms,detail=excluded.detail",
                    (
                        state.authority_identity,
                        accepted_source_watermark,
                        state.index_high,
                        state.cursor_high,
                        "healthy" if healthy else "blocked",
                        observed_at_ms,
                        detail,
                    ),
                )
            ):
                pass

    admit_stage_write("raw.frontier.inspect", publish)


@dataclass(frozen=True, slots=True)
class FrontierInspectionOutcome:
    mode: Literal["full", "delta", "current"]
    healthy: bool
    pass_id: str | None
    accepted_head_checks: int
    blocking_head_checks: int
    broken_head_checks: int
    cursor_checks: int
    cursor_ahead_count: int
    cursor_gap_count: int
    missing_session_raw_count: int
    physical_dependencies_checked: bool = False


def _proof_item(payload: str) -> RawAuthorityFrontierItem:
    from polylogue.storage.raw_reconciler import RawAuthorityFrontierItem, RawAuthorityFrontierState

    fields = json.loads(payload)
    fields["state"] = RawAuthorityFrontierState(fields["state"])
    fields["input_raw_ids"] = tuple(fields["input_raw_ids"])
    return RawAuthorityFrontierItem(**fields)


def _resolve_selected_frontier_obligations(
    source: PreparedFrontierSource,
    selection: _FrontierSelection,
    *,
    mode: Literal["full", "delta", "current"],
    pass_id: str,
    observed_at_ms: int,
) -> None:
    keys = (None,) if mode == "full" else selection.keys()
    for key in keys:
        after = None
        while True:
            predicate = "" if key is None else " AND json_extract(expected_json,'$.logical_keys[0]')=?"
            parameters = (after, after) if key is None else (key, after, after)
            sql = (
                "SELECT blocker_id,json_extract(expected_json,'$.plan_id') FROM raw_authority_blockers "
                "WHERE resolved_at_ms IS NULL AND json_extract(expected_json,'$.authority_witness.schema')="
                "'polylogue.raw-authority-frontier-plan.v1'"
                + predicate
                + " AND (? IS NULL OR blocker_id>?) ORDER BY blocker_id LIMIT 256"
            )
            with source._seal.original_rows("source", sql, parameters) as rows:
                page = tuple((str(row[0]), str(row[1])) for row in rows)
            if not page:
                break
            for blocker_id, plan_id in page:
                if not selection.contains_plan(plan_id):
                    source.resolve_obligation_input(blocker_id, pass_id=pass_id, observed_at_ms=observed_at_ms)
                after = blocker_id


def _missing_session_raw_references(source: PreparedFrontierSource) -> int:
    """Check the full direct Index reference relation without retaining a list."""
    missing = 0
    after = None
    while True:
        with source._seal.original_rows(
            "index",
            "SELECT session_id,raw_id FROM sessions WHERE raw_id IS NOT NULL "
            "AND (? IS NULL OR session_id>?) ORDER BY session_id LIMIT 256",
            (after, after),
        ) as rows:
            page = tuple((str(row[0]), str(row[1])) for row in rows)
        if not page:
            return missing
        for session_id, raw_id in page:
            source._seal.before_index_input(
                "sessions",
                ("session_id", "raw_id"),
                "SELECT rowid FROM sessions WHERE session_id=?",
                (session_id,),
            )
            source._producer._load_artifact_inputs(
                "raw_sessions",
                "SELECT rowid FROM raw_sessions WHERE raw_id=?",
                (raw_id,),
            )
            with source._seal.source_rows("SELECT 1 FROM raw_sessions WHERE raw_id=?", (raw_id,)) as rows:
                missing += int(rows.fetchone() is None)
            after = session_id


def inspect_prepared_raw_authority_frontier(
    archive_root: Path,
    *,
    input_demand: Callable[[int], None],
    check_physical_dependencies: bool = False,
) -> FrontierInspectionOutcome:
    """Inspect and publish through the caller's supplied exclusive creator/bridge."""
    from polylogue.core.compute_cancel import check_compute_cancelled
    from polylogue.core.sql_settlement import retain_native_sql_lifetimes
    from polylogue.storage.archive_identity import resolve_active_index_path
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.index_generation import ActiveWriterLease
    from polylogue.storage.raw_reconciler import _OBLIGATION_STATES
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
    from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    frame = _prepare_frontier_inspection_frame(archive_root, input_demand=input_demand)
    exclusion = None
    lifetime_bound = False
    try:
        ops = frame.ops_owner.require_connection()
        with closing(ops.execute("BEGIN")):
            pass
        observed_at_ms = int(time.time() * 1000)
        with frame.seal.original_read_snapshot(input_demand=input_demand), frame.seal.source_producer():
            state = read_frontier_journal_state(
                source=frame.seal.observer("source"),
                index=frame.seal.observer("index"),
                ops=ops,
                source_path=archive_root / "source.db",
                index_path=resolve_active_index_path(archive_root),
                ops_path=archive_root / "ops.db",
                source_triggers=declared_frontier_triggers(ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE]),
                index_triggers=declared_frontier_triggers(INDEX_DDL),
                cursor_triggers=declared_frontier_triggers(OPS_DDL),
            )
            with closing(ops.execute("PRAGMA data_version")) as rows:
                ops_version = int(rows.fetchone()[0])
            mark = read_frontier_inspection_mark(ops)
            from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

            source = PreparedFrontierSource(frame.seal)
            selected_read = PreparedSessionSourceRead(frame.seal, blob_store=BlobStore(archive_root / "blob"))
            # Filesystem changes have no relational journal key. Explicit
            # maintenance checks the complete physical dependency inventory;
            # unchanged blobs still reuse their exact fingerprint receipts.
            mode = populate_frontier_selection(
                source,
                frame.selection,
                selected_read,
                ops,
                state,
                None if check_physical_dependencies else mark,
            )
            from polylogue.storage.raw_retention import (
                BrokenAppendHeadSample,
                CursorAheadSample,
                CursorAuthorityGapSample,
                FrontierInspectionFindings,
                cursor_ahead_sample,
                cursor_gap_sample,
            )

            head_count = blocking_count = broken_count = cursor_count = ahead_count = gap_count = 0
            ahead_comparisons = cursor_comparisons = comparable_cursor_count = deferred_count = missing_refs = 0
            broken_samples: list[BrokenAppendHeadSample] = []
            ahead_samples: list[CursorAheadSample] = []
            gap_samples: list[CursorAuthorityGapSample] = []
            if mode != "current":
                for item, chain in _iter_selected_frontier_proofs(
                    source,
                    frame.selection,
                    BlobStore(archive_root / "blob"),
                    observed_at_ms=observed_at_ms,
                ):
                    check_compute_cancelled()
                    frame.selection.retain_proof(item)
                    head_count += 1
                    blocking_count += int(item.state in _OBLIGATION_STATES)
                    if chain[0] != "healthy":
                        broken_count += 1
                        # A bounded sample for the operator; the count stays exact.
                        if len(broken_samples) < _FINDINGS_SAMPLE_LIMIT:
                            broken_samples.extend(cast(tuple[BrokenAppendHeadSample, ...], chain[3])[:1])
                for cursor, comparison, retained_path in _iter_selected_cursor_proofs(source, frame.selection, ops):
                    cursor_count += 1
                    # The findings carry the canonical comparable-cursor count, the same
                    # ``checked`` the direct comparison reports: a gap or deferred cursor
                    # is inspected but compares against no accepted byte head.
                    comparable_cursor_count += int(comparison.checked)
                    ahead_count += int(comparison.ahead_count > 0)
                    ahead_comparisons += comparison.ahead_count
                    cursor_comparisons += comparison.comparison_count
                    gap_count += int(comparison.gap)
                    deferred_count += int(comparison.deferred)
                    # Bounded samples for the operator; the counts stay exact.
                    if comparison.ahead_count and len(ahead_samples) < _FINDINGS_SAMPLE_LIMIT:
                        ahead_samples.append(cursor_ahead_sample(cursor.source_path, cursor, comparison))
                    if comparison.gap and len(gap_samples) < _FINDINGS_SAMPLE_LIMIT:
                        gap_samples.append(
                            cursor_gap_sample(cursor.source_path, cursor.byte_offset, retained=retained_path)
                        )
                if mode == "full":
                    missing_refs = _missing_session_raw_references(source)
                pass_id = "raw-authority-frontier-pass:" + frame.selection.proof_inventory_digest()
                for payload in frame.selection.proof_payloads(obligations_only=True):
                    source.publish_obligation_input(
                        _proof_item(payload), pass_id=pass_id, observed_at_ms=observed_at_ms
                    )
                _resolve_selected_frontier_obligations(
                    source,
                    frame.selection,
                    mode=mode,
                    pass_id=pass_id,
                    observed_at_ms=observed_at_ms,
                )
                frame.seal.source_allocation_dependencies("raw_existence_changes")
                with frame.seal.source_rows(
                    "SELECT seq FROM sqlite_sequence WHERE name='raw_existence_changes'"
                ) as rows:
                    sequence = rows.fetchone()
                    accepted_source_high = (
                        state.source_high if sequence is None else max(state.source_high, int(sequence[0]))
                    )
            else:
                pass_id = None
                accepted_source_high = state.source_high
        healthy = not (blocking_count or broken_count or ahead_count or gap_count or missing_refs)
        outcome = FrontierInspectionOutcome(
            mode,
            healthy,
            pass_id,
            head_count,
            blocking_count,
            broken_count,
            cursor_count,
            ahead_count,
            gap_count,
            missing_refs,
            check_physical_dependencies,
        )
        frame.seal.validate_observers_current()
        permit = None if mode == "current" else frame.seal.prepare_source_mutation()
        exclusion = ActiveWriterLease(archive_root)
        exclusion.acquire()
        frame.seal.retain_publication_lifetime(exclusion, frame.close_payload)
        lifetime_bound = True
        with retain_native_sql_lifetimes(frame.selection._owner, frame.ops_owner):
            publish_frontier_inspection_mark(
                frame.seal,
                permit,
                ops_observer=ops,
                original_ops_version=ops_version,
                original_ops_identity=frame.ops_identity,
                state=state,
                accepted_source_watermark=accepted_source_high,
                healthy=healthy,
                observed_at_ms=observed_at_ms,
                detail=FrontierInspectionFindings(
                    mode=mode,
                    head_checks=head_count,
                    blocking_heads=blocking_count,
                    broken_heads=broken_count,
                    broken_head_samples=tuple(broken_samples),
                    cursor_checks=comparable_cursor_count,
                    cursor_comparisons=cursor_comparisons,
                    cursor_ahead=ahead_count,
                    cursor_ahead_comparisons=ahead_comparisons,
                    cursor_ahead_samples=tuple(ahead_samples),
                    cursor_gaps=gap_count,
                    cursor_gap_samples=tuple(gap_samples),
                    cursor_deferred=deferred_count,
                    missing_session_raws=missing_refs,
                ).to_document(),
            )
    except BaseException as primary:
        _settle_frontier_inspection_frame(frame, exclusion, lifetime_bound=lifetime_bound, primary=primary)
        raise
    else:
        _settle_frontier_inspection_frame(frame, exclusion, lifetime_bound=lifetime_bound)
        return outcome


def _settle_frontier_inspection_frame(
    frame: _PreparedFrontierInspectionFrame,
    exclusion: ActiveWriterLease | None,
    *,
    lifetime_bound: bool,
    primary: BaseException | None = None,
) -> None:
    failures: list[BaseException] = []
    closers = [frame.close]
    if exclusion is not None and not lifetime_bound:
        closers.append(exclusion.close)
    for close in closers:
        try:
            close()
        except BaseException as failure:
            failures.append(failure)
    if failures:
        if primary is not None:
            failures.insert(0, primary)
        raise BaseExceptionGroup("frontier inspection and physical settlement failed", failures) from primary


def _prepared_blocker_replay_plan(source: PreparedFrontierSource, input_raw_ids: tuple[str, ...]) -> RawReplayPlan:
    """Read the canonical replay operands from this acknowledgement's original window."""
    from polylogue.storage.raw_authority import (
        _RAW_PLAN_CENSUS_SQL,
        _RAW_PLAN_HEAD_SQL,
        _RAW_PLAN_MEMBERSHIP_SQL,
        _RAW_PLAN_PARSER_CENSUS_SQL,
        _RAW_PLAN_SESSION_SQL,
        _RAW_PLAN_SOURCE_SQL,
        _raw_replay_logical_keys,
        _raw_replay_plan_from_rows,
    )

    raw_ids = tuple(sorted(dict.fromkeys(input_raw_ids)))
    if not raw_ids:
        raise ValueError("raw replay plan requires at least one input raw id")
    marks = ",".join("?" for _ in raw_ids)
    source_inputs: list[list[dict[str, object]]] = []
    for table, query in (
        ("raw_sessions", _RAW_PLAN_SOURCE_SQL),
        ("raw_session_memberships", _RAW_PLAN_MEMBERSHIP_SQL),
        ("raw_membership_census", _RAW_PLAN_CENSUS_SQL),
        ("raw_authority_parser_census", _RAW_PLAN_PARSER_CENSUS_SQL),
    ):
        source._producer._load_artifact_inputs(table, f"SELECT rowid FROM {table} WHERE raw_id IN ({marks})", raw_ids)
        with source._seal.original_rows("source", query.format(marks=marks, index_prefix=""), raw_ids) as rows:
            names = tuple(column[0] for column in rows.description or ())
            source_inputs.append([dict(zip(names, row, strict=True)) for row in rows])
    source_rows, membership_rows, census_rows, parser_census_rows = source_inputs
    if tuple(str(row["raw_id"]) for row in source_rows) != raw_ids:
        raise RuntimeError("raw replay plan input disappeared during census")
    logical_keys = _raw_replay_logical_keys(source_rows, membership_rows)
    head_rows: list[dict[str, object]] = []
    if logical_keys:
        key_marks = ",".join("?" for _ in logical_keys)
        source._seal.before_index_input(
            "raw_revision_heads",
            (
                "logical_source_key",
                "session_id",
                "accepted_raw_id",
                "accepted_source_revision",
                "accepted_content_hash",
                "accepted_frontier_kind",
                "accepted_frontier",
                "acquisition_generation",
                "append_end_offset",
            ),
            f"SELECT rowid FROM raw_revision_heads WHERE logical_source_key IN ({key_marks})",
            logical_keys,
        )
        with source._seal.original_rows(
            "index", _RAW_PLAN_HEAD_SQL.format(marks=key_marks, index_prefix=""), logical_keys
        ) as rows:
            names = tuple(column[0] for column in rows.description or ())
            head_rows = [dict(zip(names, row, strict=True)) for row in rows]
    source._seal.before_index_input(
        "sessions",
        ("raw_id", "content_hash", "message_count"),
        f"SELECT rowid FROM sessions WHERE raw_id IN ({marks})",
        raw_ids,
    )
    with source._seal.original_rows(
        "index", _RAW_PLAN_SESSION_SQL.format(marks=marks, index_prefix=""), raw_ids
    ) as rows:
        names = tuple(column[0] for column in rows.description or ())
        session_rows = [dict(zip(names, row, strict=True)) for row in rows]
    return _raw_replay_plan_from_rows(
        raw_ids,
        logical_keys,
        source_rows=source_rows,
        membership_rows=membership_rows,
        census_rows=census_rows,
        parser_census_rows=parser_census_rows,
        head_rows=head_rows,
        session_rows=session_rows,
    )


@dataclass(frozen=True, slots=True)
class PreparedFrontierAcknowledgement:
    """One original blocker acknowledgement held on its admitted creator."""

    archive_root: Path
    blocker_id: str
    resolution: str
    found: bool
    kind: str | None
    _frame: _PreparedFrontierInspectionFrame
    _permit: KnownTierMutationPermit | None
    _receipt: dict[str, object]
    _published: bool = False

    def publish(self) -> dict[str, object]:
        from polylogue.storage.sqlite.archive_tiers.revision_governance import publish_prepared_revision_source

        if self._published:
            return dict(self._receipt)
        if not self.found or self._permit is None:
            raise KeyError(self.blocker_id)
        self._frame.seal.validate_observers_current()
        publish_prepared_revision_source(self._frame.seal, self._permit)
        object.__setattr__(self, "_published", True)
        return dict(self._receipt)


@contextmanager
def prepared_frontier_blocker_acknowledgement(
    archive_root: Path,
    blocker_id: str,
    *,
    resolution: str,
    input_demand: Callable[[int], None],
) -> Iterator[PreparedFrontierAcknowledgement]:
    """Prepare, publish and settle only within the same caller-owned phase."""
    from polylogue.storage.index_generation import ActiveWriterLease
    from polylogue.storage.raw_authority import (
        _FRONTIER_WITNESS_SCHEMA,
        _blocker_kind,
        _raw_replay_plan_from_expected_json,
    )

    normalized_resolution = resolution.strip()
    if not normalized_resolution:
        raise ValueError("raw authority blocker resolution must be non-empty")
    frame = _prepare_frontier_inspection_frame(archive_root, input_demand=input_demand)
    exclusion = None
    lifetime_bound = False
    try:
        found = False
        kind = None
        summary: dict[str, object] = {}
        with frame.seal.original_read_snapshot(input_demand=input_demand), frame.seal.source_producer():
            source = PreparedFrontierSource(frame.seal)
            source._producer._load_artifact_inputs(
                "raw_authority_blockers",
                "SELECT rowid FROM raw_authority_blockers WHERE blocker_id=?",
                (blocker_id,),
            )
            with frame.seal.source_rows(
                "SELECT expected_json,observed_json FROM raw_authority_blockers "
                "WHERE blocker_id=? AND resolved_at_ms IS NULL",
                (blocker_id,),
            ) as rows:
                row = rows.fetchone()
            if row is not None:
                found = True
                stored_plan = _raw_replay_plan_from_expected_json(str(row[0]))
                witness_schema = str(stored_plan.authority_witness.get("schema", ""))
                kind = _blocker_kind(witness_schema=witness_schema)
                observed_plan = (
                    stored_plan
                    if witness_schema == _FRONTIER_WITNESS_SCHEMA
                    else _prepared_blocker_replay_plan(source, stored_plan.input_raw_ids)
                )
                now = int(time.time() * 1000)
                receipt = {
                    "schema": "polylogue.raw-authority-blocker-resolution.v1",
                    "blocker_id": blocker_id,
                    "superseded_plan_id": stored_plan.plan_id,
                    "current_plan": observed_plan.to_dict(),
                    "operator_resolution": normalized_resolution,
                    "resolved_at_ms": now,
                }
                key = frame.seal.retain_literal_scalar(blocker_id)
                sql, parameters, cells = source._statement_operands(
                    "UPDATE raw_authority_blockers SET resolved_at_ms=?,resolution=? "
                    "WHERE blocker_id=? AND resolved_at_ms IS NULL",
                    ("resolved_at_ms", "resolution", "blocker_id"),
                    (now, json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False), blocker_id),
                )
                with frame.seal.source_statement(
                    sql,
                    parameters,
                    table="raw_authority_blockers",
                    writable_targets=(("raw_authority_blockers", (key,)),),
                    prepared_cells=cells,
                ) as result:
                    if result.rowcount != 1:
                        raise ValueError("original frontier blocker changed during acknowledgement preparation")
                summary = {
                    "schema": "polylogue.raw-authority-blocker-resolution-summary.v1",
                    "blocker_id": blocker_id,
                    "superseded_plan_id": stored_plan.plan_id,
                    "current_plan": {
                        "plan_id": observed_plan.plan_id,
                        "input_digest": observed_plan.input_digest,
                        "input_raw_count": len(observed_plan.input_raw_ids),
                        "logical_key_count": len(observed_plan.logical_keys),
                    },
                    "operator_resolution": normalized_resolution,
                    "resolved_at_ms": now,
                }
        permit = frame.seal.prepare_source_mutation() if found else None
        exclusion = ActiveWriterLease(archive_root)
        exclusion.acquire()
        frame.seal.retain_publication_lifetime(exclusion, frame.close_payload)
        lifetime_bound = True
        yield PreparedFrontierAcknowledgement(
            archive_root.resolve(),
            blocker_id,
            normalized_resolution,
            found,
            kind,
            frame,
            permit,
            summary,
        )
    except BaseException as primary:
        _settle_frontier_inspection_frame(frame, exclusion, lifetime_bound=lifetime_bound, primary=primary)
        raise
    else:
        _settle_frontier_inspection_frame(frame, exclusion, lifetime_bound=lifetime_bound)


def prune_prepared_frontier_journals(archive_root: Path, *, input_demand: Callable[[int], None]) -> int:
    """Prune the intersection consumed by both original frontier consumers."""
    from polylogue.core.sql_settlement import retain_native_sql_lifetimes
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.archive_identity import resolve_active_index_path
    from polylogue.storage.frontier_existence import consumed_watermarks
    from polylogue.storage.index_generation import ActiveWriterLease
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
    from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL
    from polylogue.storage.sqlite.archive_tiers.revision_governance import publish_prepared_revision_source
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import owned_daemon_connection

    certificate = consumed_watermarks(archive_root)
    if certificate is None:
        return 0
    frame = _prepare_frontier_inspection_frame(archive_root, input_demand=input_demand)
    exclusion = None
    lifetime_bound = False
    try:
        ops = frame.ops_owner.require_connection()
        with closing(ops.execute("BEGIN")):
            pass
        with frame.seal.original_read_snapshot(input_demand=input_demand), frame.seal.source_producer():
            state = read_frontier_journal_state(
                source=frame.seal.observer("source"),
                index=frame.seal.observer("index"),
                ops=ops,
                source_path=archive_root / "source.db",
                index_path=resolve_active_index_path(archive_root),
                ops_path=archive_root / "ops.db",
                source_triggers=declared_frontier_triggers(ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE]),
                index_triggers=declared_frontier_triggers(INDEX_DDL),
                cursor_triggers=declared_frontier_triggers(OPS_DDL),
            )
            mark = read_frontier_inspection_mark(ops)
            if mark is None or mark.state != "healthy" or mark.authority_identity != state.authority_identity:
                source_count = 0
                source_watermark = index_watermark = cursor_watermark = 0
            else:
                source_watermark = min(certificate[0], mark.source_watermark)
                index_watermark = min(certificate[1], mark.index_watermark)
                cursor_watermark = mark.cursor_watermark
                if not (
                    state.source_floor <= source_watermark <= state.source_high
                    and state.index_floor <= index_watermark <= state.index_high
                    and state.cursor_floor <= cursor_watermark <= state.cursor_high
                ):
                    source_watermark = index_watermark = cursor_watermark = 0
                source_count = PreparedFrontierSource(frame.seal).prune_journal_input(source_watermark)
        frame.seal.validate_observers_current()
        permit = frame.seal.prepare_source_mutation() if source_count else None
        exclusion = ActiveWriterLease(archive_root)
        exclusion.acquire()
        frame.seal.retain_publication_lifetime(exclusion, frame.close_payload)
        lifetime_bound = True

        def publish() -> int:
            frame.seal.validate_observers_current()
            if permit is not None:
                publish_prepared_revision_source(frame.seal, permit)
            pruned = source_count
            for path, table, watermark in (
                (resolve_active_index_path(archive_root), "raw_existence_changes", index_watermark),
                (archive_root / "ops.db", "raw_frontier_cursor_changes", cursor_watermark),
            ):
                if not watermark:
                    continue
                with owned_daemon_connection(path, archive_root=archive_root) as writer, writer:
                    with closing(writer.execute("BEGIN IMMEDIATE")):
                        pass
                    with closing(writer.execute(f"DELETE FROM {table} WHERE sequence<=?", (watermark,))) as rows:
                        pruned += rows.rowcount
            return pruned

        with retain_native_sql_lifetimes(frame.selection._owner, frame.ops_owner):
            result: int = admit_stage_write("raw.frontier.journal.prune", publish)
    except BaseException as primary:
        _settle_frontier_inspection_frame(frame, exclusion, lifetime_bound=lifetime_bound, primary=primary)
        raise
    else:
        _settle_frontier_inspection_frame(frame, exclusion, lifetime_bound=lifetime_bound)
        return result


def read_frontier_coverage_for_archive(archive_root: Path) -> dict[str, object]:
    """Read coverage from one physically settled, pinned three-tier frame."""
    from polylogue.storage.archive_identity import resolve_active_index_path
    from polylogue.storage.sqlite.connection_profile import attach_readonly_database, open_readonly_connection

    with closing(open_readonly_connection(resolve_active_index_path(archive_root))) as conn:
        attach_readonly_database(conn, archive_root / "source.db", alias="source_tier")
        attach_readonly_database(conn, archive_root / "ops.db", alias="ops_tier")
        with closing(conn.execute("BEGIN")):
            pass
        _require_attached_tier_versions(conn)
        return frontier_inspection_projection(conn, archive_root=archive_root)


def _require_attached_tier_versions(conn: sqlite3.Connection) -> None:
    """Refuse an attached durable Source tier this runtime's schema cannot read.

    The Index reader validates its own tier on open; an attached tier carries
    no such check, so a newer or older Source would otherwise be read as if
    it held the current schema and report a misleading frontier.
    """
    from polylogue.core.errors import SchemaSkew
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import _schema_skew_remedy

    with closing(conn.execute("PRAGMA source_tier.user_version")) as cursor:
        found = int(cursor.fetchone()[0])
    expected = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
    if found != expected:
        raise SchemaSkew(
            tier=ArchiveTier.SOURCE.value,
            expected=expected,
            found=found,
            remedy=_schema_skew_remedy(ArchiveTier.SOURCE),
        )
