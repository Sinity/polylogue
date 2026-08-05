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
import tempfile
from collections import Counter
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
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
    identity: tuple[tuple[str, object], ...]
    operations: tuple[DifferenceOperation, ...] = ()
    columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.operations) != 1:
            raise ValueError("expected differences require exactly one operation")
        if not self.columns:
            raise ValueError("expected differences require a non-empty changed-column signature")
        if not self.identity:
            raise ValueError("expected differences require a bounded row identity")

    def matches(
        self,
        *,
        table: str,
        operation: DifferenceOperation,
        identity: tuple[tuple[str, object], ...],
        changed_columns: tuple[str, ...],
    ) -> bool:
        if self.table != table:
            return False
        if operation is not self.operations[0]:
            return False
        if identity != self.identity:
            return False
        return tuple(changed_columns) == self.columns


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
    missing_columns: tuple[tuple[str, tuple[str, ...]], ...]
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
            "missing_columns": [{"table": table, "columns": list(columns)} for table, columns in self.missing_columns],
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


@dataclass(frozen=True, slots=True)
class CanaryRunResult:
    """Evidence from one bounded inactive-generation canary run."""

    selection: CanarySelection
    comparison: CanaryDiffReport
    rebuild_receipt: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "selection": self.selection.to_dict(),
            "comparison": self.comparison.to_dict(),
            "rebuild_receipt": self.rebuild_receipt,
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


def run_reindex_canary(
    archive_root: Path,
    *,
    input_index: Path | None = None,
    sessions_per_origin: int = 100,
    pathology_session_ids: Iterable[str] = (),
    sample_session_ids: Iterable[str] = (),
    no_promote: bool,
) -> CanaryRunResult:
    """Replay a selected canary through the existing inactive rebuild route."""

    if not no_promote:
        raise CanarySelectionError("reindex canary requires --no-promote")
    from polylogue.config import resolve_archive_root
    from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
    from polylogue.storage.archive_identity import ArchiveLocation, TierFileIdentity

    root = Path(archive_root)
    if root.resolve() == resolve_archive_root().resolve():
        raise CanarySelectionError(
            "reindex canary refuses the configured live archive root; "
            "run it against an explicitly provisioned isolated canary archive"
        )
    current_index = _resolve_canary_input_index(root, input_index)
    location = ArchiveLocation.resolve(root)
    if input_index is not None:
        supplied = TierFileIdentity.resolve("index", Path(input_index))
        if not location.active_index.same_file(supplied):
            raise CanarySelectionError(
                "input index is not the configured archive active generation; "
                "explicit canary input index must be inside or bound to the selected archive root"
            )
    from polylogue.maintenance.archive_verification import REINDEX_CANARY_ACCEPTANCE_CHECKS
    from polylogue.maintenance.pathology_zoo import pathology_zoo_is_present, pathology_zoo_session_ids

    automatic_pathology_ids = (
        pathology_zoo_session_ids() if pathology_zoo_is_present(root, index_path_override=current_index) else ()
    )
    requested_pathology_ids = tuple(dict.fromkeys((*automatic_pathology_ids, *pathology_session_ids)))
    selection = select_canary_sessions(
        current_index,
        sessions_per_origin=sessions_per_origin,
        pathology_session_ids=requested_pathology_ids,
        sample_session_ids=sample_session_ids,
    )
    receipt = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            raw_ids=selection.selected_raw_ids,
            promote=False,
            candidate_acceptance_checks=REINDEX_CANARY_ACCEPTANCE_CHECKS,
        )
    )
    receipt_payload = receipt.to_dict()
    _validate_selection_evidence(receipt_payload, selection.selected_raw_ids)
    candidate_path = _validate_canary_candidate(
        root,
        current_index=current_index,
        selection=selection,
        receipt=receipt,
    )
    _validate_authoritative_rebuild_receipt(receipt_payload, candidate_path)
    comparison = compare_reindex_generations(
        current_index,
        candidate_path,
        session_ids=selection.selected_session_ids,
    )
    return CanaryRunResult(
        selection=selection,
        comparison=comparison,
        rebuild_receipt=receipt_payload,
    )


def _resolve_canary_input_index(archive_root: Path, input_index: Path | None) -> Path:
    """Resolve an explicit index only when it belongs to the selected archive."""
    from polylogue.storage.archive_identity import ArchiveLocation

    location = ArchiveLocation.resolve(archive_root)
    if input_index is None:
        return location.active_index_path

    candidate = Path(input_index)
    try:
        candidate_resolved = candidate.resolve()
        archive_resolved = Path(archive_root).resolve()
    except (OSError, RuntimeError) as exc:
        raise CanarySelectionError("explicit canary input index path cannot be resolved") from exc

    try:
        candidate_resolved.relative_to(archive_resolved)
    except ValueError:
        if candidate_resolved != location.active_index.resolved_path:
            raise CanarySelectionError(
                "input index is not the configured archive active generation; "
                "explicit canary input index must be inside or bound to the selected archive root"
            ) from None
    return candidate


def _validate_canary_candidate(
    archive_root: Path,
    *,
    current_index: Path,
    selection: CanarySelection,
    receipt: object,
) -> Path:
    """Prove the compared index is this run's own inactive generation."""
    from polylogue.storage.archive_identity import ArchiveLocation

    root = archive_root.resolve()
    receipt_root = getattr(receipt, "archive_root", None)
    if not isinstance(receipt_root, str) or Path(receipt_root).resolve() != root:
        raise CanarySelectionError("rebuild receipt belongs to a different archive root")
    if getattr(receipt, "status", None) != "replayed" or getattr(receipt, "materialized", None) is not True:
        raise CanarySelectionError("rebuild receipt is not a completed materialized replay")
    if getattr(receipt, "selected_raw_count", None) != len(selection.selected_raw_ids):
        raise CanarySelectionError("rebuild receipt selected a different raw-id set than the canary")

    generation = getattr(receipt, "generation", None)
    if not isinstance(generation, dict):
        raise CanarySelectionError("rebuild receipt did not identify an inactive candidate generation")
    if generation.get("archive_root") != str(root):
        raise CanarySelectionError("candidate generation belongs to a different archive root")
    if generation.get("state") != "inactive":
        raise CanarySelectionError("reindex canary candidate must remain an inactive generation")
    generation_id = generation.get("generation_id")
    owner_id = generation.get("owner_id")
    source_snapshot = generation.get("source_snapshot")
    candidate_value = generation.get("index_path")
    if not all(
        isinstance(value, str) and value for value in (generation_id, owner_id, source_snapshot, candidate_value)
    ):
        raise CanarySelectionError("rebuild receipt did not identify a complete inactive candidate generation")
    assert isinstance(candidate_value, str)

    location = ArchiveLocation.resolve(archive_root)
    anchor = location.active_pointer or location.configured_tier("index").configured_path
    expected_generation_root = anchor.parent / ".index-generations"
    candidate_path = Path(candidate_value)
    try:
        candidate_resolved = candidate_path.resolve(strict=True)
        current_resolved = Path(current_index).resolve(strict=True)
    except OSError as exc:
        raise CanarySelectionError("rebuild receipt candidate index is not readable") from exc
    if candidate_resolved == current_resolved:
        raise CanarySelectionError("rebuild receipt candidate index is the active index")
    if (
        candidate_resolved.name != "index.db"
        or candidate_resolved.parent.name != generation_id
        or candidate_resolved.parent.parent != expected_generation_root.resolve()
    ):
        raise CanarySelectionError("rebuild receipt candidate is outside this archive's generation root")
    return candidate_path


@dataclass(frozen=True, slots=True)
class CanaryDifferenceReview:
    """An explicit operator classification for one diff row."""

    table: str
    operation: DifferenceOperation
    identity: tuple[tuple[str, object], ...]
    changed_columns: tuple[str, ...]
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
            changed_columns=difference.changed_columns,
            classification=classification,
            reference=reference,
            rationale=rationale,
        )

    @property
    def key(self) -> tuple[str, DifferenceOperation, tuple[tuple[str, object], ...], tuple[str, ...]]:
        return self.table, self.operation, self.identity, self.changed_columns

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "operation": self.operation.value,
            "identity": dict(self.identity),
            "changed_columns": list(self.changed_columns),
            "classification": self.classification.value,
            "reference": self.reference,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class DurableCanaryReport:
    """The reviewed, persisted canary changelog."""

    selection: CanarySelection
    comparison: CanaryDiffReport
    rebuild_receipt: dict[str, object]
    reviews: tuple[CanaryDifferenceReview, ...]
    review_status: str
    comparison_fingerprint: str
    archive_provenance: dict[str, object]

    @property
    def unclassified_count(self) -> int:
        return self.comparison.unclassified_count

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 4,
            "selection": self.selection.to_dict(),
            "comparison": self.comparison.to_dict(),
            "rebuild_receipt": self.rebuild_receipt,
            "reviews": [review.to_dict() for review in self.reviews],
            "review_status": self.review_status,
            "comparison_fingerprint": self.comparison_fingerprint,
            "archive_provenance": self.archive_provenance,
        }


def write_canary_report(
    output_path: Path,
    *,
    selection: CanarySelection,
    comparison: CanaryDiffReport,
    rebuild_receipt: dict[str, object],
    reviews: Iterable[CanaryDifferenceReview],
    allow_unreviewed: bool = False,
) -> DurableCanaryReport:
    """Persist a fully reviewed report or an explicit durable unreviewed diff.

    Reviews must cover exactly the comparator's row identities and signatures.
    This prevents a report from becoming a durable green-light artifact while a diff was
    silently omitted from review. The write is atomic and touches only the
    requested report path, never either SQLite generation.
    """

    _validate_rebuild_receipt(
        rebuild_receipt,
        selected_raw_ids=selection.selected_raw_ids,
        candidate_index=comparison.candidate_index,
    )
    _validate_selection_binding(selection, comparison, rebuild_receipt)
    review_list = tuple(reviews)
    review_by_key: dict[
        tuple[str, DifferenceOperation, tuple[tuple[str, object], ...], tuple[str, ...]], CanaryDifferenceReview
    ] = {}
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
            difference.changed_columns,
        )
        for difference in comparison.differences
    }
    missing_keys = difference_keys.difference(review_by_key)
    extra_keys = set(review_by_key).difference(difference_keys)
    if (missing_keys and allow_unreviewed and not review_list and not extra_keys) or not comparison.differences:
        review_status = "unreviewed" if missing_keys else "reviewed"
    elif duplicate_keys or missing_keys or extra_keys:
        detail = [
            f"duplicate={len(duplicate_keys)}",
            f"missing={len(missing_keys)}",
            f"extra={len(extra_keys)}",
        ]
        raise UnclassifiedCanaryDiffError("canary report classification is incomplete (" + ", ".join(detail) + ")")
    else:
        review_status = "reviewed"

    archive_provenance = _capture_archive_provenance(comparison, rebuild_receipt)
    reviewed_differences = (
        comparison.differences
        if review_status == "unreviewed"
        else tuple(
            replace(
                difference,
                classification=review_by_key[
                    (difference.table, difference.operation, difference.identity, difference.changed_columns)
                ].classification,
                rationale=(
                    f"{review_by_key[(difference.table, difference.operation, difference.identity, difference.changed_columns)].reference}: "
                    f"{review_by_key[(difference.table, difference.operation, difference.identity, difference.changed_columns)].rationale}"
                ),
            )
            for difference in comparison.differences
        )
    )
    reviewed_comparison = replace(comparison, differences=reviewed_differences)
    durable = DurableCanaryReport(
        selection=selection,
        comparison=reviewed_comparison,
        rebuild_receipt=rebuild_receipt,
        reviews=review_list,
        review_status=review_status,
        comparison_fingerprint=_comparison_fingerprint(comparison),
        archive_provenance=archive_provenance,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(durable.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    except BaseException:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()
        raise
    return durable


def _validate_selection_binding(
    selection: CanarySelection, comparison: CanaryDiffReport, receipt: dict[str, object]
) -> None:
    if selection.index_path.resolve() != comparison.current_index.resolve():
        raise UnclassifiedCanaryDiffError("canary report selection index does not match the compared current index")
    if selection.selected_session_ids != comparison.session_ids:
        raise UnclassifiedCanaryDiffError("canary report selection sessions do not match the comparison")
    _validate_selection_evidence(receipt, selection.selected_raw_ids)


def _validate_selection_evidence(receipt: dict[str, object], selected_raw_ids: Iterable[str]) -> None:
    """Match report selection to the rebuild-owned candidate commitment."""

    from polylogue.maintenance.rebuild_index import rebuild_selection_evidence

    generation = receipt.get("generation")
    evidence = receipt.get("selection_evidence")
    if not isinstance(generation, dict) or not isinstance(evidence, dict):
        raise UnclassifiedCanaryDiffError("canary report has no authoritative rebuild selection evidence")
    required = (
        generation.get("generation_id"),
        generation.get("owner_id"),
        generation.get("index_path"),
        generation.get("source_snapshot"),
        receipt.get("archive_root"),
    )
    if not all(isinstance(value, str) and value for value in required):
        raise UnclassifiedCanaryDiffError("canary report has incomplete authoritative rebuild selection evidence")
    expected = rebuild_selection_evidence(
        tuple(selected_raw_ids),
        archive_root=Path(cast(str, receipt["archive_root"])),
        generation_id=cast(str, generation["generation_id"]),
        generation_owner_id=cast(str, generation["owner_id"]),
        candidate_index=Path(cast(str, generation["index_path"])),
        source_snapshot=cast(str, generation["source_snapshot"]),
    )
    if evidence != expected:
        raise UnclassifiedCanaryDiffError("canary report selection does not match the authoritative rebuild receipt")


def _comparison_fingerprint(comparison: CanaryDiffReport) -> str:
    """Hash comparison evidence independently of operator classification."""

    payload = {
        "current_index": str(comparison.current_index),
        "candidate_index": str(comparison.candidate_index),
        "session_ids": list(comparison.session_ids),
        "compared_tables": list(comparison.compared_tables),
        "missing_tables": list(comparison.missing_tables),
        "missing_columns": [
            {"table": table, "columns": list(columns)} for table, columns in comparison.missing_columns
        ],
        "differences": [
            {
                "table": difference.table,
                "operation": difference.operation.value,
                "identity": dict(difference.identity),
                "before": difference.before,
                "after": difference.after,
                "changed_columns": list(difference.changed_columns),
            }
            for difference in comparison.differences
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_rebuild_receipt(
    receipt: object,
    *,
    selected_raw_ids: Iterable[str],
    candidate_index: Path | str,
) -> None:
    if not isinstance(receipt, dict):
        raise UnclassifiedCanaryDiffError("canary report has no rebuild receipt")
    archive_root = receipt.get("archive_root")
    selected_raw_count = receipt.get("selected_raw_count")
    generation = receipt.get("generation")
    if (
        not isinstance(archive_root, str)
        or not archive_root
        or selected_raw_count != len(tuple(selected_raw_ids))
        or receipt.get("status") != "replayed"
        or receipt.get("materialized") is not True
        or not isinstance(generation, dict)
    ):
        raise UnclassifiedCanaryDiffError("canary report has invalid rebuild receipt")
    generation_archive_root = generation.get("archive_root")
    generation_index = generation.get("index_path")
    required_generation = (
        generation.get("generation_id"),
        generation.get("owner_id"),
        generation_archive_root,
        generation_index,
        generation.get("source_snapshot"),
    )
    if (
        generation.get("state") != "inactive"
        or not all(isinstance(value, str) and value for value in required_generation)
        or not isinstance(generation_archive_root, str)
        or not isinstance(generation_index, str)
    ):
        raise UnclassifiedCanaryDiffError("canary report has incomplete candidate generation provenance")
    if Path(archive_root).resolve() != Path(generation_archive_root).resolve():
        raise UnclassifiedCanaryDiffError("canary report rebuild receipt and candidate archive roots disagree")
    if Path(generation_index).resolve() != Path(candidate_index).resolve():
        raise UnclassifiedCanaryDiffError("canary report rebuild receipt does not identify the compared candidate")
    _validate_selection_evidence(receipt, selected_raw_ids)


def _validate_authoritative_rebuild_receipt(receipt: dict[str, object], candidate_index: Path) -> None:
    """Reject report receipts that differ from rebuild-owned candidate evidence."""

    path = candidate_index.parent / "rebuild-receipt.json"
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnclassifiedCanaryDiffError("archive-owned rebuild receipt is unreadable") from exc
    if not isinstance(stored, dict) or stored != receipt:
        raise UnclassifiedCanaryDiffError("archive-owned rebuild receipt does not match the report")


def _generation_root(location: object) -> Path:
    """Return the lifecycle directory anchored by this archive's active pointer."""

    from polylogue.storage.archive_identity import ArchiveLocation

    assert isinstance(location, ArchiveLocation)
    anchor = location.active_pointer or location.configured_tier("index").configured_path
    return anchor.parent / ".index-generations"


def _read_generation_metadata(generation_root: Path, generation_id: str) -> dict[str, object]:
    path = generation_root / generation_id / "generation.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnclassifiedCanaryDiffError("archive-owned generation metadata is unreadable") from exc
    if not isinstance(payload, dict):
        raise UnclassifiedCanaryDiffError("archive-owned generation metadata is invalid")
    return payload


def _generation_fields(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise UnclassifiedCanaryDiffError("archive-owned generation metadata is invalid")
    fields = {
        key: metadata.get(key)
        for key in ("generation_id", "owner_id", "archive_root", "index_path", "state", "source_snapshot")
    }
    if not all(isinstance(value, str) and value for value in fields.values()):
        raise UnclassifiedCanaryDiffError("archive-owned generation metadata is incomplete")
    return fields


def _index_evidence(path: Path) -> dict[str, object]:
    """Describe a local file identity without claiming it is a secret capability."""

    from polylogue.storage.archive_identity import TierFileIdentity

    identity = TierFileIdentity.resolve("index", path)
    if not identity.exists:
        raise UnclassifiedCanaryDiffError("archive-owned index is not readable")
    digest = sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise UnclassifiedCanaryDiffError("archive-owned index is not readable") from exc
    return {"file": identity.as_dict(), "content_sha256": digest.hexdigest()}


def _same_index_evidence(recorded: object, path: Path, *, label: str) -> None:
    if not isinstance(recorded, dict) or recorded != _index_evidence(path):
        raise UnclassifiedCanaryDiffError(f"archive-owned {label} index identity no longer matches the report")


def _active_generation_metadata(root: Path, location: object) -> dict[str, object]:
    from polylogue.storage.archive_identity import ArchiveLocation, TierFileIdentity

    assert isinstance(location, ArchiveLocation)
    active_identity = TierFileIdentity.resolve("index", location.active_index_path)
    matches: list[dict[str, object]] = []
    for metadata_path in _generation_root(location).glob("gen-*/generation.json"):
        try:
            metadata = _read_generation_metadata(_generation_root(location), metadata_path.parent.name)
            fields = _generation_fields(metadata)
            generation_index = TierFileIdentity.resolve("index", Path(cast(str, fields["index_path"])))
        except (OSError, json.JSONDecodeError):
            continue
        if fields["state"] == "active" and active_identity.same_file(generation_index):
            matches.append(metadata)
    if len(matches) != 1:
        raise UnclassifiedCanaryDiffError("archive-owned active generation metadata does not match the active pointer")
    active = matches[0]
    if Path(cast(str, _generation_fields(active)["archive_root"])).resolve() != root.resolve():
        raise UnclassifiedCanaryDiffError("archive-owned active generation belongs to another archive")
    return active


def _capture_archive_provenance(comparison: CanaryDiffReport, receipt: dict[str, object]) -> dict[str, object]:
    """Capture lifecycle records that make a local report archive-specific.

    These values are not a secret or a signature. They are deliberately
    re-read on load so approval follows the active pointer, generation record,
    source snapshot, and the exact inactive file that the rebuild created.
    """

    from polylogue.storage.archive_identity import ArchiveLocation, TierFileIdentity
    from polylogue.storage.index_generation import source_revision_snapshot

    archive_root = receipt.get("archive_root")
    generation = receipt.get("generation")
    if not isinstance(archive_root, str):
        raise UnclassifiedCanaryDiffError("canary report has invalid archive provenance root")
    root = Path(archive_root)
    location = ArchiveLocation.resolve(root)
    current_identity = TierFileIdentity.resolve("index", comparison.current_index)
    if not location.active_index.same_file(current_identity):
        raise UnclassifiedCanaryDiffError("archive-owned active index does not match the canary comparison")
    if not isinstance(generation, dict):
        raise UnclassifiedCanaryDiffError("canary report has invalid candidate generation provenance")
    candidate = _generation_fields(generation)
    candidate_metadata = _read_generation_metadata(_generation_root(location), cast(str, candidate["generation_id"]))
    if candidate_metadata != generation:
        raise UnclassifiedCanaryDiffError(
            "archive-owned candidate generation metadata does not match the rebuild receipt"
        )
    if candidate["state"] != "inactive":
        raise UnclassifiedCanaryDiffError("archive-owned candidate generation is not inactive")
    candidate_path = Path(cast(str, candidate["index_path"]))
    if not candidate_path.samefile(comparison.candidate_index):
        raise UnclassifiedCanaryDiffError("archive-owned candidate generation does not match the canary comparison")
    source_snapshot = source_revision_snapshot(root)
    if source_snapshot != candidate["source_snapshot"]:
        raise UnclassifiedCanaryDiffError("archive-owned source snapshot does not match the inactive candidate")
    return {
        "archive_root": str(root.resolve()),
        "active_pointer": str(location.active_pointer) if location.active_pointer is not None else None,
        "active_index": _index_evidence(location.active_index_path),
        "active_generation": _active_generation_metadata(root, location),
        "candidate_generation": candidate_metadata,
        "candidate_index": _index_evidence(candidate_path),
        "source_snapshot": source_snapshot,
    }


def _validate_archive_provenance(
    provenance: object,
    *,
    configured_archive_root: Path,
    current_index: Path,
    candidate_index: Path,
    receipt: dict[str, object],
) -> None:
    """Validate archive-owned evidence before opening report-provided indexes."""

    from polylogue.storage.archive_identity import ArchiveLocation, TierFileIdentity
    from polylogue.storage.index_generation import source_revision_snapshot

    if not isinstance(provenance, dict):
        raise UnclassifiedCanaryDiffError("canary report has no archive-owned provenance")
    archive_root = provenance.get("archive_root")
    if not isinstance(archive_root, str):
        raise UnclassifiedCanaryDiffError("canary report has invalid archive-owned provenance root")
    root = configured_archive_root.resolve()
    if Path(archive_root).resolve() != root:
        raise UnclassifiedCanaryDiffError("canary report belongs to a different configured archive root")
    receipt_root = receipt.get("archive_root")
    if not isinstance(receipt_root, str) or Path(receipt_root).resolve() != root:
        raise UnclassifiedCanaryDiffError("archive-owned provenance root does not match the rebuild receipt")
    location = ArchiveLocation.resolve(root)
    pointer = str(location.active_pointer) if location.active_pointer is not None else None
    if provenance.get("active_pointer") != pointer:
        raise UnclassifiedCanaryDiffError("archive-owned active pointer no longer matches the report")
    if provenance.get("active_generation") != _active_generation_metadata(root, location):
        raise UnclassifiedCanaryDiffError("archive-owned active generation metadata no longer matches the report")
    current_identity = TierFileIdentity.resolve("index", current_index)
    if not location.active_index.same_file(current_identity):
        raise UnclassifiedCanaryDiffError("archive-owned active index no longer matches the report")
    _same_index_evidence(provenance.get("active_index"), location.active_index_path, label="active")
    receipt_generation = receipt.get("generation")
    if not isinstance(receipt_generation, dict):
        raise UnclassifiedCanaryDiffError("canary report has invalid candidate generation provenance")
    receipt_fields = _generation_fields(receipt_generation)
    if provenance.get("candidate_generation") != receipt_generation:
        raise UnclassifiedCanaryDiffError("archive-owned candidate generation does not match the rebuild receipt")
    live_candidate = _read_generation_metadata(_generation_root(location), cast(str, receipt_fields["generation_id"]))
    live_fields = _generation_fields(live_candidate)
    if live_candidate != receipt_generation or live_fields["state"] != "inactive":
        raise UnclassifiedCanaryDiffError("archive-owned candidate generation metadata no longer matches the report")
    candidate_path = Path(cast(str, live_fields["index_path"]))
    generation_root = _generation_root(location).resolve()
    candidate_resolved = candidate_path.resolve(strict=True)
    expected_generation_path = generation_root / str(receipt_fields["generation_id"]) / "index.db"
    if candidate_resolved != expected_generation_path.resolve():
        raise UnclassifiedCanaryDiffError("archive-owned candidate generation path is not archive-owned")
    try:
        same_candidate = candidate_path.samefile(candidate_index)
    except OSError as exc:
        raise UnclassifiedCanaryDiffError("archive-owned candidate generation is not readable") from exc
    if not same_candidate:
        raise UnclassifiedCanaryDiffError("archive-owned candidate generation no longer matches the report")
    _same_index_evidence(provenance.get("candidate_index"), candidate_path, label="candidate")
    source_snapshot = source_revision_snapshot(root)
    if provenance.get("source_snapshot") != source_snapshot or live_fields["source_snapshot"] != source_snapshot:
        raise UnclassifiedCanaryDiffError("archive-owned source snapshot no longer matches the inactive candidate")


def load_canary_report(path: Path, *, archive_root: Path | None = None) -> dict[str, object]:
    """Read and structurally revalidate a durable report's review coverage."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UnclassifiedCanaryDiffError("canary report root must be an object")
    if payload.get("schema_version") != 4:
        raise UnclassifiedCanaryDiffError("canary report has no authoritative rebuild receipt schema")
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        raise UnclassifiedCanaryDiffError("canary report has no comparison object")
    summary = comparison.get("summary")
    if not isinstance(summary, dict):
        raise UnclassifiedCanaryDiffError("canary report has no comparison summary object")
    raw_differences = comparison.get("differences")
    if not isinstance(raw_differences, list):
        raise UnclassifiedCanaryDiffError("canary report has no differences list")
    differences = tuple(_difference_from_dict(item) for item in raw_differences)
    raw_reviews = payload.get("reviews")
    if not isinstance(raw_reviews, list):
        raise UnclassifiedCanaryDiffError("canary report has no reviews list")
    reviews = tuple(_review_from_dict(item) for item in raw_reviews)
    review_status = payload.get("review_status")
    if review_status not in {"reviewed", "unreviewed"}:
        raise UnclassifiedCanaryDiffError("canary report has invalid review status")
    comparison_fingerprint = payload.get("comparison_fingerprint")
    if not isinstance(comparison_fingerprint, str) or len(comparison_fingerprint) != 64:
        raise UnclassifiedCanaryDiffError("canary report has invalid comparison fingerprint")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise UnclassifiedCanaryDiffError("canary report has no selection object")
    selected_raw_ids = selection.get("selected_raw_ids")
    candidate_index = comparison.get("candidate_index")
    if not isinstance(selected_raw_ids, list) or not all(isinstance(value, str) for value in selected_raw_ids):
        raise UnclassifiedCanaryDiffError("canary report has invalid selected raw ids")
    if not isinstance(candidate_index, str):
        raise UnclassifiedCanaryDiffError("canary report has no candidate index")
    _validate_rebuild_receipt(
        payload.get("rebuild_receipt"),
        selected_raw_ids=cast(list[str], selected_raw_ids),
        candidate_index=candidate_index,
    )
    selection_sessions = selection.get("selected_session_ids")
    if not isinstance(selection_sessions, list) or not all(isinstance(value, str) for value in selection_sessions):
        raise UnclassifiedCanaryDiffError("canary report has invalid selected session ids")
    selection_index = selection.get("index_path")
    sessions_per_origin = selection.get("sessions_per_origin")
    if not isinstance(selection_index, str) or not isinstance(sessions_per_origin, int):
        raise UnclassifiedCanaryDiffError("canary report has invalid selection binding")
    persisted_selection = CanarySelection(
        index_path=Path(selection_index),
        sessions_per_origin=sessions_per_origin,
        selected_session_ids=tuple(cast(list[str], selection_sessions)),
        selected_raw_ids=tuple(cast(list[str], selected_raw_ids)),
        sampled_session_ids=(),
        pathology_session_ids=(),
        sample_session_ids=(),
        origin_counts=(),
    )
    current_index = comparison.get("current_index")
    compared_sessions = comparison.get("session_ids")
    compared_tables = comparison.get("compared_tables")
    missing_tables = comparison.get("missing_tables")
    raw_missing_columns = comparison.get("missing_columns")
    if (
        not isinstance(current_index, str)
        or not isinstance(compared_sessions, list)
        or not all(isinstance(value, str) for value in compared_sessions)
        or not isinstance(compared_tables, list)
        or not all(isinstance(value, str) for value in compared_tables)
        or not isinstance(missing_tables, list)
        or not all(isinstance(value, str) for value in missing_tables)
        or not isinstance(raw_missing_columns, list)
    ):
        raise UnclassifiedCanaryDiffError("canary report has invalid comparison selection")
    missing_columns: list[tuple[str, tuple[str, ...]]] = []
    for item in raw_missing_columns:
        if not isinstance(item, dict):
            raise UnclassifiedCanaryDiffError("canary report has invalid comparison schema evidence")
        table = item.get("table")
        columns = item.get("columns")
        if (
            not isinstance(table, str)
            or not isinstance(columns, list)
            or not all(isinstance(column, str) for column in columns)
        ):
            raise UnclassifiedCanaryDiffError("canary report has invalid comparison schema evidence")
        missing_columns.append((table, tuple(columns)))
    persisted_comparison = CanaryDiffReport(
        current_index=Path(current_index),
        candidate_index=Path(candidate_index),
        session_ids=tuple(cast(list[str], compared_sessions)),
        compared_tables=tuple(cast(list[str], compared_tables)),
        missing_tables=tuple(cast(list[str], missing_tables)),
        missing_columns=tuple(missing_columns),
        differences=differences,
    )
    _validate_selection_binding(
        persisted_selection, persisted_comparison, cast(dict[str, object], payload["rebuild_receipt"])
    )
    from polylogue.config import resolve_archive_root

    _validate_archive_provenance(
        payload.get("archive_provenance"),
        configured_archive_root=Path(archive_root) if archive_root is not None else resolve_archive_root(),
        current_index=persisted_comparison.current_index,
        candidate_index=persisted_comparison.candidate_index,
        receipt=cast(dict[str, object], payload["rebuild_receipt"]),
    )
    _validate_authoritative_rebuild_receipt(
        cast(dict[str, object], payload["rebuild_receipt"]), persisted_comparison.candidate_index
    )
    recomputed_comparison = compare_reindex_generations(
        persisted_comparison.current_index,
        persisted_comparison.candidate_index,
        session_ids=persisted_selection.selected_session_ids,
    )
    if comparison_fingerprint != _comparison_fingerprint(
        persisted_comparison
    ) or comparison_fingerprint != _comparison_fingerprint(recomputed_comparison):
        raise UnclassifiedCanaryDiffError("canary report comparison attestation does not match the recorded indexes")
    difference_keys = tuple(_difference_key(difference) for difference in differences)
    review_keys = tuple(review.key for review in reviews)
    if len(set(difference_keys)) != len(difference_keys) or len(set(review_keys)) != len(review_keys):
        raise UnclassifiedCanaryDiffError("canary report contains duplicate difference or review identities")
    difference_by_key = dict(zip(difference_keys, differences, strict=True))
    review_by_key = dict(zip(review_keys, reviews, strict=True))
    missing_keys = set(difference_by_key).difference(review_by_key)
    extra_keys = set(review_by_key).difference(difference_by_key)
    if review_status == "unreviewed":
        if reviews or not differences:
            raise UnclassifiedCanaryDiffError("canary report unreviewed state is invalid")
    elif missing_keys or extra_keys:
        raise UnclassifiedCanaryDiffError(
            f"canary report review coverage is incomplete (missing={len(missing_keys)}, extra={len(extra_keys)})"
        )
    if review_status == "reviewed":
        for key, difference in difference_by_key.items():
            review = review_by_key[key]
            if review.classification is not difference.classification:
                raise UnclassifiedCanaryDiffError("canary report review classification disagrees with its difference")
    comparison["summary"] = _summary_for_differences(differences)
    return cast(dict[str, object], payload)


def approve_canary_report(path: Path, *, archive_root: Path) -> dict[str, object]:
    """Approve evidence only. This cannot authorize or perform promotion."""

    payload = load_canary_report(path, archive_root=archive_root)
    if payload.get("review_status") != "reviewed":
        raise UnclassifiedCanaryDiffError("canary report is not approved: review is incomplete")
    comparison = payload.get("comparison")
    summary = comparison.get("summary") if isinstance(comparison, dict) else None
    if not isinstance(summary, dict) or summary.get("unexpected_count") != 0:
        raise UnclassifiedCanaryDiffError("canary report is not approved: unexpected differences remain")
    return payload


def _difference_key(
    difference: RowDifference,
) -> tuple[str, DifferenceOperation, tuple[tuple[str, object], ...], tuple[str, ...]]:
    return difference.table, difference.operation, difference.identity, difference.changed_columns


def _difference_from_dict(value: object) -> RowDifference:
    if not isinstance(value, dict):
        raise UnclassifiedCanaryDiffError("canary report difference must be an object")
    identity = value.get("identity")
    changed_columns = value.get("changed_columns")
    if (
        not isinstance(identity, dict)
        or not isinstance(changed_columns, list)
        or not all(isinstance(column, str) for column in changed_columns)
    ):
        raise UnclassifiedCanaryDiffError("canary report difference has invalid identity or changed columns")
    before = value.get("before")
    after = value.get("after")
    if before is not None and not isinstance(before, dict):
        raise UnclassifiedCanaryDiffError("canary report difference has invalid before row")
    if after is not None and not isinstance(after, dict):
        raise UnclassifiedCanaryDiffError("canary report difference has invalid after row")
    table = value.get("table")
    rationale = value.get("rationale")
    if not isinstance(table, str) or not isinstance(rationale, str):
        raise UnclassifiedCanaryDiffError("canary report difference has invalid table or rationale")
    try:
        operation = DifferenceOperation(value["operation"])
        classification = DifferenceClassification(value["classification"])
    except (KeyError, ValueError) as exc:
        raise UnclassifiedCanaryDiffError("canary report difference has invalid operation or classification") from exc
    return RowDifference(
        table=table,
        operation=operation,
        identity=tuple((str(key), item) for key, item in identity.items()),
        before=cast(dict[str, object] | None, before),
        after=cast(dict[str, object] | None, after),
        changed_columns=tuple(cast(list[str], changed_columns)),
        classification=classification,
        rationale=rationale,
    )


def _review_from_dict(value: object) -> CanaryDifferenceReview:
    if not isinstance(value, dict):
        raise UnclassifiedCanaryDiffError("canary report review must be an object")
    identity = value.get("identity")
    changed_columns = value.get("changed_columns")
    table = value.get("table")
    reference = value.get("reference")
    rationale = value.get("rationale")
    if (
        not isinstance(identity, dict)
        or not isinstance(changed_columns, list)
        or not changed_columns
        or not all(isinstance(column, str) for column in changed_columns)
        or not isinstance(table, str)
        or not isinstance(reference, str)
        or not isinstance(rationale, str)
        or not reference.strip()
        or not rationale.strip()
    ):
        raise UnclassifiedCanaryDiffError("every canary review needs a valid reference and rationale")
    try:
        operation = DifferenceOperation(value["operation"])
        classification = DifferenceClassification(value["classification"])
    except (KeyError, ValueError) as exc:
        raise UnclassifiedCanaryDiffError("canary report review has invalid operation or classification") from exc
    return CanaryDifferenceReview(
        table=table,
        operation=operation,
        identity=tuple((str(key), item) for key, item in identity.items()),
        changed_columns=tuple(cast(list[str], changed_columns)),
        classification=classification,
        reference=reference,
        rationale=rationale,
    )


def load_canary_review_manifest(path: Path) -> tuple[CanaryDifferenceReview, ...]:
    """Load the explicit per-difference review manifest accepted by the CLI."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("reviews"), list):
        raise UnclassifiedCanaryDiffError("canary review manifest must contain a reviews list")
    return tuple(_review_from_dict(item) for item in cast(list[object], payload["reviews"]))


def _summary_for_differences(differences: tuple[RowDifference, ...]) -> dict[str, object]:
    expected_count = sum(item.classification is DifferenceClassification.EXPECTED for item in differences)
    unexpected_count = sum(item.classification is DifferenceClassification.UNEXPECTED for item in differences)
    return {
        "difference_count": len(differences),
        "expected_count": expected_count,
        "unexpected_count": unexpected_count,
        "unclassified_count": 0,
        "counts_by_table": dict(sorted(Counter(item.table for item in differences).items())),
    }


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(str(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        schema_differences: list[RowDifference] = []
        missing_columns: list[tuple[str, tuple[str, ...]]] = []
        for table in missing_tables:
            current_present = table in current_tables
            current_columns = _table_columns(current, table) if current_present else ()
            candidate_columns = _table_columns(candidate, table) if table in candidate_tables else ()
            operation = DifferenceOperation.REMOVED if current_present else DifferenceOperation.ADDED
            changed_columns = tuple(sorted(set(current_columns).union(candidate_columns)))
            before: dict[str, object] | None = (
                {"table": table, "columns": list(current_columns)} if current_present else None
            )
            after: dict[str, object] | None = (
                {"table": table, "columns": list(candidate_columns)} if table in candidate_tables else None
            )
            schema_differences.append(
                _build_difference(
                    table=table,
                    operation=operation,
                    identity=(("__schema__", "table"),),
                    before=before,
                    after=after,
                    changed_columns=changed_columns,
                    expected=reviewed,
                )
            )
        for table in compared_tables:
            current_column_set = set(_table_columns(current, table))
            candidate_column_set = set(_table_columns(candidate, table))
            only_current = current_column_set - candidate_column_set
            only_candidate = candidate_column_set - current_column_set
            if only_current or only_candidate:
                missing_columns.append((table, tuple(sorted(only_current.union(only_candidate)))))
            for column in sorted(only_current):
                schema_differences.append(
                    _build_difference(
                        table=table,
                        operation=DifferenceOperation.REMOVED,
                        identity=(("__schema__", "column"), ("name", column)),
                        before={"table": table, "column": column},
                        after=None,
                        changed_columns=(column,),
                        expected=reviewed,
                    )
                )
            for column in sorted(only_candidate):
                schema_differences.append(
                    _build_difference(
                        table=table,
                        operation=DifferenceOperation.ADDED,
                        identity=(("__schema__", "column"), ("name", column)),
                        before=None,
                        after={"table": table, "column": column},
                        changed_columns=(column,),
                        expected=reviewed,
                    )
                )
        selected_sessions = _selected_session_ids(current, candidate, session_ids)
        differences = schema_differences
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
        missing_columns=tuple(missing_columns),
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
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    result: set[str] = set()
    for row in rows:
        table = str(row[0])
        if table in _EXCLUDED_TABLES or table.startswith("messages_fts") or table.startswith("blocks_command_trigram"):
            continue
        try:
            columns = _table_columns(connection, table)
        except sqlite3.OperationalError:
            # A canonical view whose backing table is itself absent is not a
            # second independent schema delta. The missing base relation is
            # still reported below without making the comparison unreadable.
            continue
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
    if table == "actions" and "tool_use_block_id" in columns:
        return ("tool_use_block_id",)
    rows = connection.execute(f"PRAGMA table_xinfo({_quote_identifier(table)})").fetchall()
    primary_key = [(int(row[5]), str(row[1])) for row in rows if int(row[5]) > 0 and int(row[6]) not in (1, 2)]
    if primary_key:
        return tuple(name for _position, name in sorted(primary_key))
    for preferred in ("tool_use_block_id", "session_id", "message_id", "block_id", "event_id", "policy_id"):
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
        differences.append(
            _build_difference(
                table=table,
                operation=operation,
                identity=tuple((column, key[index]) for index, column in enumerate(primary_key)),
                before=before,
                after=after,
                changed_columns=changed_columns,
                expected=expected,
            )
        )
    return differences


def _build_difference(
    *,
    table: str,
    operation: DifferenceOperation,
    identity: tuple[tuple[str, object], ...],
    before: dict[str, object] | None,
    after: dict[str, object] | None,
    changed_columns: tuple[str, ...],
    expected: tuple[ExpectedDifference, ...],
) -> RowDifference:
    matching = next(
        (
            item
            for item in expected
            if item.matches(table=table, operation=operation, identity=identity, changed_columns=changed_columns)
        ),
        None,
    )
    return RowDifference(
        table=table,
        operation=operation,
        identity=identity,
        before=before,
        after=after,
        changed_columns=changed_columns,
        classification=DifferenceClassification.EXPECTED
        if matching is not None
        else DifferenceClassification.UNEXPECTED,
        rationale=(
            f"{matching.bead_ref}: {matching.rationale}"
            if matching is not None
            else "no reviewed bead or delta declaration matched this difference"
        ),
    )


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
    "CanaryRunResult",
    "CanarySelection",
    "CanarySelectionError",
    "DurableCanaryReport",
    "DifferenceClassification",
    "DifferenceOperation",
    "ExpectedDifference",
    "RowDifference",
    "UnclassifiedCanaryDiffError",
    "approve_canary_report",
    "compare_reindex_generations",
    "load_canary_report",
    "run_reindex_canary",
    "select_canary_sessions",
    "write_canary_report",
]
