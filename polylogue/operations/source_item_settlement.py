"""Settle retained source items from the daemon's materialization receipt."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from polylogue.core.enums import IngestOutcome
from polylogue.operations.ingest_inputs import spool_connection
from polylogue.storage.sqlite.archive_tiers.source_items import (
    AcquisitionDisposition,
    transition_source_item,
)


def _matching_raw_members(
    source_conn: sqlite3.Connection,
    receipt: sqlite3.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    expected_count: int,
    check_stop: Callable[[], None] | None,
) -> tuple[bool, bool]:
    """Compare exact ordered raw/blob membership using bounded cursor state."""
    count = 0
    all_complete = True
    with (
        closing(
            receipt.execute(
                "SELECT raw_id, raw_blob_hash, complete FROM item_raws WHERE source_item_id=? ORDER BY raw_id",
                (source_item_id,),
            )
        ) as expected_rows,
        closing(
            source_conn.execute(
                "SELECT DISTINCT m.raw_id, m.raw_blob_hash, r.blob_hash "
                "FROM source_item_raw_members AS m LEFT JOIN raw_sessions AS r ON r.raw_id=m.raw_id "
                "WHERE m.source_generation_id=? AND m.source_item_id=? AND m.raw_id IS NOT NULL "
                "ORDER BY m.raw_id",
                (source_generation_id, source_item_id),
            )
        ) as actual_rows,
    ):
        while True:
            expected = expected_rows.fetchone()
            actual = actual_rows.fetchone()
            if expected is None or actual is None:
                return expected is None and actual is None and count == expected_count, all_complete
            if check_stop is not None and count % 128 == 0:
                check_stop()
            if (
                str(expected[0]) != str(actual[0])
                or bytes(expected[1]) != bytes(actual[1])
                or actual[2] is None
                or bytes(actual[1]) != bytes(actual[2])
            ):
                return False, False
            all_complete &= bool(expected[2])
            count += 1


def settle_materialized_source_items(
    source_conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    receipt_path: Path,
    observed_at_ms: int,
    check_stop: Callable[[], None] | None = None,
) -> int:
    """Advance only source items fully proved by the pinned materialization.

    The receipt is the read-side proof. This Source-writer callback joins it
    against the current generation, item, and exact raw/blob membership before
    making the durable transition. Incomplete and refused members stay in
    their existing typed state.
    """
    changed = 0
    with spool_connection(receipt_path, read_only=True) as receipt:
        generation = source_conn.execute(
            "SELECT item_count FROM source_generations WHERE source_generation_id=?",
            (source_generation_id,),
        ).fetchone()
        if generation is None:
            raise ValueError("materialization source generation is no longer current")
        cursor = -1
        while True:
            if check_stop is not None:
                check_stop()
            rows = receipt.execute(
                "SELECT ordinal, source_item_id, raw_count, retired_count, source_complete "
                "FROM items WHERE ordinal>? ORDER BY ordinal LIMIT 128",
                (cursor,),
            ).fetchall()
            if not rows:
                break
            for ordinal, item_id, raw_count, retired_count, complete in rows:
                cursor = int(ordinal)
                if int(retired_count) or int(raw_count) == 0:
                    continue
                members_match, raw_members_complete = _matching_raw_members(
                    source_conn,
                    receipt,
                    source_generation_id=source_generation_id,
                    source_item_id=str(item_id),
                    expected_count=int(raw_count),
                    check_stop=check_stop,
                )
                if not members_match:
                    continue
                logical = receipt.execute(
                    "SELECT COUNT(DISTINCT l.logical_key), COALESCE(MIN(l.complete), 0) "
                    "FROM item_raws AS ir JOIN raw_logicals AS rl ON rl.raw_id=ir.raw_id "
                    "JOIN logicals AS l ON l.logical_key=rl.logical_key WHERE ir.source_item_id=?",
                    (item_id,),
                ).fetchone()
                logical_count, logical_complete = int(logical[0]), bool(logical[1])
                item = source_conn.execute(
                    "SELECT disposition, revision, enumeration_fingerprint, enumerated_at_ms, "
                    "parser_fingerprint, addressing_mode "
                    "FROM source_items WHERE source_generation_id=? AND source_item_id=?",
                    (source_generation_id, item_id),
                ).fetchone()
                if item is None or item[0] != AcquisitionDisposition.PENDING.value or item[3] is None:
                    continue
                if item[2] is None:
                    continue
                validation = source_conn.execute(
                    "SELECT r.raw_id, r.validation_mode, c.parser_fingerprint "
                    "FROM source_item_raw_members AS m "
                    "JOIN raw_sessions AS r ON r.raw_id=m.raw_id "
                    "LEFT JOIN raw_membership_census AS c ON c.raw_id=r.raw_id "
                    "WHERE m.source_generation_id=? AND m.source_item_id=? "
                    "AND r.validation_status='failed' AND r.validation_mode='strict' "
                    "AND r.validated_at_ms IS NOT NULL AND r.validation_error IS NOT NULL "
                    "AND r.parse_error IS NULL ORDER BY r.raw_id LIMIT 1",
                    (source_generation_id, item_id),
                ).fetchone()
                if validation is not None:
                    transition_id = (
                        "validation-rejected:"
                        + hashlib.sha256(
                            "\0".join(
                                (
                                    source_generation_id,
                                    str(item_id),
                                    str(item[2]),
                                    str(item[5]),
                                    str(item[4] or ""),
                                    str(validation[1] or ""),
                                    str(validation[2] or item[4] or ""),
                                )
                            ).encode()
                        ).hexdigest()
                    )
                    transition_source_item(
                        source_conn,
                        source_generation_id=source_generation_id,
                        source_item_id=str(item_id),
                        request_id=transition_id,
                        disposition=AcquisitionDisposition.PENDING,
                        outcome_code=IngestOutcome.VALIDATION_REJECTED,
                        stage="validation",
                        retryable=False,
                        diagnostic="strict source validation rejected the acquired input",
                        evidence_ref=f"raw:{validation[0]}",
                        parser_fingerprint=str(validation[2]) if validation[2] else None,
                        observed_at_ms=observed_at_ms,
                        expected_revision=int(item[1]),
                        commit=False,
                    )
                    changed += 1
                    continue
                if not complete or not raw_members_complete:
                    continue
                if logical_count and logical_complete:
                    disposition = AcquisitionDisposition.ADMITTED
                    outcome = IngestOutcome.SUCCESS
                elif not logical_count:
                    typed = source_conn.execute(
                        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN "
                        "(r.parse_error IS NULL AND (a.parse_as_session=0 OR c.status='non_session')) "
                        "THEN 1 ELSE 0 END), 0) "
                        "FROM raw_sessions AS r LEFT JOIN raw_artifacts AS a ON a.raw_id=r.raw_id "
                        "LEFT JOIN raw_membership_census AS c ON c.raw_id=r.raw_id "
                        "WHERE r.raw_id IN (SELECT raw_id FROM source_item_raw_members "
                        "WHERE source_generation_id=? AND source_item_id=? AND raw_id IS NOT NULL)",
                        (source_generation_id, item_id),
                    ).fetchone()
                    if int(typed[0]) != int(raw_count) or int(typed[1]) != int(raw_count):
                        continue
                    disposition = AcquisitionDisposition.NON_SESSION
                    outcome = IngestOutcome.UNSUPPORTED_SHAPE
                else:
                    continue
                transition_id = (
                    "materialized:" + hashlib.sha256(f"{source_generation_id}\0{item_id}".encode()).hexdigest()
                )
                transition_source_item(
                    source_conn,
                    source_generation_id=source_generation_id,
                    source_item_id=str(item_id),
                    request_id=transition_id,
                    disposition=disposition,
                    outcome_code=outcome,
                    stage="materialization",
                    observed_at_ms=observed_at_ms,
                    expected_revision=int(item[1]),
                    commit=False,
                )
                changed += 1
    return changed


__all__ = ["settle_materialized_source_items"]
