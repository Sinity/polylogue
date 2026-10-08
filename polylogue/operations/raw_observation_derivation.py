"""Product seam for raw recovery through the derivation kernel."""

from __future__ import annotations

from builtins import BaseExceptionGroup
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.core.enums import ValidationMode
from polylogue.core.stage_admission import admit_stage_write
from polylogue.daemon.derivation import (
    DerivationFrame,
)
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.derived.raw import RAW_OBSERVATION_DOMAIN as _RAW_OBSERVATION_DOMAIN
from polylogue.storage.derived.raw import (
    RawObservationDerivation,
    RawObservationInspection,
    RawObservationScope,
    raw_observation_recipe_version,
)

if TYPE_CHECKING:
    from polylogue.storage.index_generation import IndexGeneration

RAW_OBSERVATION_DOMAIN = _RAW_OBSERVATION_DOMAIN


def make_raw_observation_derivation(
    archive_root: Path,
    *,
    compute_adapter: BoundedComputeAdapter,
    prepaid_blob_inputs: tuple[tuple[str, bytes, int], ...] = (),
    index_db_path: Path | None = None,
    owned_generation: IndexGeneration | None = None,
    validation_mode: ValidationMode = ValidationMode.ADVISORY,
) -> RawObservationDerivation:
    """Construct the storage-owned raw adapter from the operations boundary.

    Prepaid operands require actual acquisition in this still-admitted task.
    The original witness validates their raw/hash/size before CAS enrollment;
    inputs paid in an earlier released phase cannot be carried here.
    """
    from polylogue.sources.revision_backfill import prepare_retained_non_json_artifact

    return RawObservationDerivation(
        archive_root,
        prepare_non_json_artifact=prepare_retained_non_json_artifact,
        compute_adapter=compute_adapter,
        prepaid_blob_inputs=prepaid_blob_inputs,
        index_db_path=index_db_path,
        owned_generation=owned_generation,
        validation_mode=validation_mode,
    )


def raw_observation_output_session_ids(archive_root: Path, raw_id: str) -> tuple[str, ...]:
    """Read every active session output in the seed raw's replay component."""
    from polylogue.operations.operation_context import open_operation_read

    with open_operation_read(archive_root) as pinned:
        component_raw_ids, _logical_keys = pinned.archive.expand_raw_membership_selection([raw_id])
        if not component_raw_ids:
            return ()
        index = pinned.archive.index_connection
        if index is None:
            return ()
        rows = index.execute(
            f"""SELECT s.session_id
            FROM sessions AS s
            JOIN raw_revision_heads AS head
              ON head.session_id = s.session_id
             AND head.accepted_raw_id = s.raw_id
            WHERE head.accepted_raw_id IN ({",".join("?" for _ in component_raw_ids)})
            ORDER BY s.session_id""",
            component_raw_ids,
        ).fetchall()
        return tuple(str(row[0]) for row in rows)


def raw_observation_payload_bytes(archive_root: Path, raw_id: str) -> int:
    """The retained payload size a raw observation's parse holds, for compute admission.

    A read fault propagates: it is retryable, and admitting the parse as a
    zero-byte task would let it run beside a full byte reservation.
    """
    from contextlib import closing

    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    with closing(open_readonly_connection(archive_root / "source.db")) as conn:
        row = conn.execute("SELECT blob_size FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
    return int(row[0]) if row is not None else 0


def raw_observation_frame(
    archive_root: Path,
    *,
    raw_ids: Sequence[str] = (),
    index_db_path: Path | None = None,
    validation_mode: ValidationMode = ValidationMode.ADVISORY,
) -> DerivationFrame:
    index_path = index_db_path or ArchiveLocation.resolve(archive_root).active_index_path
    return DerivationFrame(
        archive_root=str(archive_root),
        source_revision=str(index_path.resolve()),
        recipe_versions={RAW_OBSERVATION_DOMAIN: raw_observation_recipe_version(validation_mode)},
        scope=RawObservationScope(raw_ids=tuple(raw_ids)),
    )


def raw_observation_inspection_frame(
    archive_root: Path,
    *,
    index_db_path: Path | None = None,
) -> DerivationFrame:
    """Bind read-only discovery to parser evidence without a validation policy."""
    index_path = index_db_path or ArchiveLocation.resolve(archive_root).active_index_path
    adapter = RawObservationInspection(archive_root, index_db_path=index_path)
    return DerivationFrame(
        archive_root=str(archive_root),
        source_revision=str(index_path.resolve()),
        recipe_versions={RAW_OBSERVATION_DOMAIN: adapter.recipe_version},
        scope=RawObservationScope(),
    )


def raw_observation_backlog_snapshot(
    archive_root: Path, *, limit: int, index_db_path: Path | None = None
) -> dict[str, object]:
    """Describe one bounded canonical raw-observation page for status surfaces.

    Status is observational and must not reintroduce a second all-raw
    materialization census.  The returned rows are therefore a page from the
    same derivation traversal fair intake owns, with its pending verdicts from
    the adapter's authoritative inspection.
    """
    if limit < 1:
        raise ValueError("limit must be positive")

    def unavailable(reason: str) -> dict[str, object]:
        return {
            "available": False,
            "reason": reason,
            "scan": "bounded_raw_observation_page",
            "candidate_count": 0,
            "total_blob_bytes": 0,
            "max_blob_bytes": 0,
            "top_raw_rows": [],
            "origin_summary": [],
            "source_path_summary": [],
            "page_limit": limit,
            "page_complete": True,
        }

    adapter = RawObservationInspection(archive_root, index_db_path=index_db_path)
    frame = raw_observation_inspection_frame(archive_root, index_db_path=index_db_path)
    from polylogue.sources.dispatch import is_stream_record_provider

    try:
        raw_ids, next_cursor = adapter.required_page(frame, cursor=None, limit=limit)
        states = adapter.inspect(frame, raw_ids)
    except FileNotFoundError as exc:
        return unavailable(str(exc))
    refusals = adapter.terminal_decode_refusals(raw_ids)
    pending_ids = tuple(raw_id for raw_id in raw_ids if states.get(raw_id) != "valid" and raw_id not in refusals)
    if not pending_ids:
        return {
            "available": True,
            "scan": "bounded_raw_observation_page",
            # This page can prove zero pending rows only when the traversal is
            # complete. An incomplete page is a lower bound, never a total.
            "candidate_count": 0,
            "total_blob_bytes": 0,
            "max_blob_bytes": 0,
            "top_raw_rows": [],
            "origin_summary": [],
            "source_path_summary": [],
            "page_limit": limit,
            "page_complete": next_cursor is None,
        }
    with adapter._read() as conn:
        selected = conn.execute(
            f"SELECT raw_id, origin, detected_provider, source_path, blob_size FROM raw_sessions "
            f"WHERE raw_id IN ({','.join('?' for _ in pending_ids)})",
            pending_ids,
        ).fetchall()
    rows: list[dict[str, Any]] = sorted(
        (
            {
                "raw_id": str(row["raw_id"]),
                "origin": str(row["origin"]),
                "source_path": str(row["source_path"] or ""),
                "blob_size": int(row["blob_size"] or 0),
                "oversized": int(row["blob_size"] or 0) > 64 * 1024 * 1024,
                # Stream safety is a provider-wire property: use the acquisition
                # provider evidence, never a reversed public origin token.
                "stream_safe": is_stream_record_provider(
                    str(row["source_path"] or ""), str(row["detected_provider"] or "")
                ),
            }
            for row in selected
        ),
        key=lambda row: (-int(row["blob_size"]), str(row["raw_id"])),
    )
    origin_summary: dict[str, dict[str, int | str]] = {}
    source_path_summary: dict[str, dict[str, int | str]] = {}
    for row in rows:
        origin = str(row["origin"])
        path = str(row["source_path"])
        for summary, key, name in ((origin_summary, origin, "origin"), (source_path_summary, path, "source_path")):
            entry = summary.setdefault(key, {name: key, "raw_count": 0, "total_blob_bytes": 0, "max_blob_bytes": 0})
            entry["raw_count"] = int(entry["raw_count"]) + 1
            entry["total_blob_bytes"] = int(entry["total_blob_bytes"]) + int(row["blob_size"])
            entry["max_blob_bytes"] = max(int(entry["max_blob_bytes"]), int(row["blob_size"]))
    return {
        "available": True,
        "scan": "bounded_raw_observation_page",
        "candidate_count": len(rows),
        "total_blob_bytes": sum(int(row["blob_size"]) for row in rows),
        "max_blob_bytes": max((int(row["blob_size"]) for row in rows), default=0),
        "top_raw_rows": rows,
        "origin_summary": sorted(origin_summary.values(), key=lambda row: str(row["origin"])),
        "source_path_summary": sorted(source_path_summary.values(), key=lambda row: str(row["source_path"])),
        "page_limit": limit,
        "page_complete": next_cursor is None,
    }


def publish_raw_observation_once(
    archive_root: Path,
    raw_id: str,
    *,
    retained_replacements: list[RawObservationReplacement],
    compute_adapter: BoundedComputeAdapter,
    prepaid_blob_inputs: tuple[tuple[str, bytes, int], ...] = (),
) -> bool:
    """Prepare one exact retained raw and publish its original carrier on this worker."""
    from polylogue.core.compute_cancel import check_compute_cancelled
    from polylogue.core.write_lease import coordinator_write_lease_active

    if coordinator_write_lease_active():
        raise RuntimeError("raw observation preparation requires the writer lease to be released")
    from polylogue.sources.live.cold_build import active_cold_build_generation

    cold_build = active_cold_build_generation(archive_root)
    owned_generation = None if cold_build is None else cold_build.generation
    index_path = None if owned_generation is None else Path(owned_generation.index_path)
    adapter = make_raw_observation_derivation(
        archive_root,
        index_db_path=index_path,
        owned_generation=owned_generation,
        compute_adapter=compute_adapter,
        prepaid_blob_inputs=prepaid_blob_inputs,
    )
    frame = raw_observation_frame(archive_root, raw_ids=(raw_id,), index_db_path=index_path)
    # A committed census, classification, byte restoration or deferred-child
    # parent publication is this raw's own phase, not a moved input: prepare
    # the next phase against it while the adapter reports committed progress,
    # exactly as the derivation kernel does.
    while True:
        replacement = adapter.compute(frame, raw_id)
        retained_replacements.append(replacement)

        def close(replacement: RawObservationReplacement = replacement) -> None:
            replacement.close()
            retained_replacements.remove(replacement)

        if replacement.prepared_key_refusals:
            # One exact raw has no healthy sibling key to publish around: its
            # refused member is this raw's outcome, never a silent deferral.
            close()
            raise replacement.prepared_key_refusals[0]
        try:
            check_compute_cancelled()
            result = admit_stage_write(
                "watcher.live_ingest.append.publish", partial(adapter.publish, frame, replacement)
            )
        except BaseException as primary:
            try:
                close()
            except BaseException as cleanup:
                raise BaseExceptionGroup("raw append publication and cleanup failed", [primary, cleanup]) from primary
            raise
        close()
        if result or not adapter.publication_advanced(replacement):
            return result


if TYPE_CHECKING:
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.storage.derived.raw import RawObservationReplacement
    from polylogue.storage.index_generation import IndexGeneration
