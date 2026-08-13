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


class CanaryAuthorityKind(StrEnum):
    """Structured authority attached to one reviewed difference."""

    BEAD = "bead"
    DELTA = "delta"
    SUCCESSOR = "successor"


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
    parser_fingerprints: tuple[tuple[str, str], ...] = ()
    lowering_fingerprint: str | None = None
    replay_routing_fingerprint: str | None = None
    materializer_fingerprint: str | None = None

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
            "parser_fingerprints": dict(self.parser_fingerprints),
            "lowering_fingerprint": self.lowering_fingerprint,
            "replay_routing_fingerprint": self.replay_routing_fingerprint,
            "materializer_fingerprint": self.materializer_fingerprint,
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
    source_root: Path | None = None,
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
        dict.fromkeys(sorted(raw_id for session_id in selected if (raw_id := records[session_id][1]) is not None))
    )
    if not selected_session_ids:
        raise CanarySelectionError("automatic canary selection produced zero sessions; refusing full-source replay")
    if not selected_raw_ids:
        raise CanarySelectionError("automatic canary selection produced zero raw ids; refusing full-source replay")
    origin_counts = Counter(records[session_id][0] for session_id in selected)
    from polylogue.sources.origin_specs import (
        lowering_fingerprint,
        materializer_fingerprint,
        parser_fingerprint_for_origin,
        replay_routing_fingerprint,
    )

    selected_origins = tuple(sorted(origin_counts))
    parser_fingerprints: tuple[tuple[str, str], ...] = ()
    lowering: str | None = None
    replay_routing: str | None = None
    materializer: str | None = None
    if (Path(source_root) if source_root is not None else path.parent).joinpath("source.db").exists():
        parser_fingerprints = tuple((origin, parser_fingerprint_for_origin(origin)) for origin in selected_origins)
        lowering = lowering_fingerprint()
        replay_routing = replay_routing_fingerprint()
        materializer = materializer_fingerprint()
    return CanarySelection(
        index_path=path,
        sessions_per_origin=sessions_per_origin,
        selected_session_ids=selected_session_ids,
        selected_raw_ids=selected_raw_ids,
        sampled_session_ids=tuple(sorted(sampled)),
        pathology_session_ids=tuple(sorted(pathology)),
        sample_session_ids=tuple(sorted(explicit_samples)),
        origin_counts=tuple(sorted(origin_counts.items())),
        parser_fingerprints=parser_fingerprints,
        lowering_fingerprint=lowering,
        replay_routing_fingerprint=replay_routing,
        materializer_fingerprint=materializer,
    )


def run_reindex_canary(
    archive_root: Path,
    *,
    input_index: Path | None = None,
    schema_inference_receipt_path: Path | None,
    message_owner_scope_backfill_receipt_path: Path | None = None,
    sessions_per_origin: int = 100,
    pathology_session_ids: Iterable[str] = (),
    sample_session_ids: Iterable[str] = (),
    no_promote: bool,
) -> CanaryRunResult:
    """Replay a selected canary through the existing inactive rebuild route.

    The schema-inference receipt is explicit because this partial rebuild must
    consume the same identity-bound source evidence as a full rebuild. The
    canary never falls back to ambient process configuration.
    """

    if schema_inference_receipt_path is None:
        raise CanarySelectionError("reindex canary requires an explicit schema-inference receipt path")
    if not no_promote:
        raise CanarySelectionError("reindex canary requires --no-promote")
    from polylogue.storage.archive_identity import ArchiveLocation, TierFileIdentity

    root = Path(archive_root)
    current_index = _resolve_canary_input_index(root, input_index)
    location = ArchiveLocation.resolve(root)
    if input_index is not None:
        supplied = TierFileIdentity.resolve("index", Path(input_index))
        if not location.active_index.same_file(supplied):
            raise CanarySelectionError(
                "input index is not the configured archive active generation; "
                "explicit canary input index must be inside or bound to the selected archive root"
            )
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
        source_root=root,
    )
    _validate_nonempty_selection(selection)
    from polylogue.daemon.bulk_rebuild import run_daemon_canary_rebuild
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION

    if message_owner_scope_backfill_receipt_path is None:
        receipt: object = run_daemon_canary_rebuild(
            archive_root=root,
            raw_ids=selection.selected_raw_ids,
            selected_session_ids=selection.selected_session_ids,
            index_schema_version=INDEX_SCHEMA_VERSION,
            schema_inference_receipt_path=schema_inference_receipt_path,
        )
    else:
        receipt = run_daemon_canary_rebuild(
            archive_root=root,
            raw_ids=selection.selected_raw_ids,
            selected_session_ids=selection.selected_session_ids,
            index_schema_version=INDEX_SCHEMA_VERSION,
            schema_inference_receipt_path=schema_inference_receipt_path,
            message_owner_scope_backfill_receipt_path=message_owner_scope_backfill_receipt_path,
        )
    try:
        try:
            active_after_rebuild = ArchiveLocation.resolve(root).active_index
            selected_active_index = TierFileIdentity.resolve("index", current_index)
            if not active_after_rebuild.same_file(selected_active_index):
                raise CanarySelectionError("canary active index changed during rebuild")
        except OSError as exc:
            raise CanarySelectionError("canary active index could not be revalidated after rebuild") from exc
        if isinstance(receipt, dict):
            receipt_payload = receipt
        else:
            to_dict = getattr(receipt, "to_dict", None)
            if not callable(to_dict):
                raise CanarySelectionError("daemon canary rebuild did not return a receipt object")
            receipt_payload = to_dict()
        if not isinstance(receipt_payload, dict):
            raise CanarySelectionError("daemon canary rebuild did not return a receipt object")
        _validate_selection_evidence(
            receipt_payload,
            selection.selected_raw_ids,
            selected_session_ids=selection.selected_session_ids,
            expected_parser_fingerprints=selection.parser_fingerprints,
            expected_lowering_fingerprint=selection.lowering_fingerprint,
            expected_replay_routing_fingerprint=selection.replay_routing_fingerprint,
            expected_materializer_fingerprint=selection.materializer_fingerprint,
        )
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
        if comparison.session_ids != selection.selected_session_ids:
            raise CanarySelectionError("canary comparator scope does not match the selected sessions")
        if not comparison.session_ids:
            raise CanarySelectionError("canary comparator scope is empty; refusing an unbounded comparison")
    except BaseException as exc:
        cleanup_errors = _discard_canary_candidate(root, receipt)
        _add_canary_cleanup_notes(exc, cleanup_errors)
        raise
    return CanaryRunResult(
        selection=selection,
        comparison=comparison,
        rebuild_receipt=receipt_payload,
    )


def _discard_canary_candidate(archive_root: Path, receipt: object) -> list[BaseException]:
    """Discard the inactive canary candidate after a post-rebuild failure."""
    from polylogue.daemon.bulk_rebuild import discard_daemon_canary_candidate

    errors: list[BaseException] = []
    generation = receipt.get("generation") if isinstance(receipt, dict) else getattr(receipt, "generation", None)
    if not isinstance(generation, dict):
        return [RuntimeError("canary receipt did not identify a candidate for cleanup")]
    generation_id = generation.get("generation_id")
    generation_owner_id = generation.get("owner_id")
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or not isinstance(generation_owner_id, str)
        or not generation_owner_id
    ):
        return [RuntimeError("canary receipt candidate generation identity is missing")]
    try:
        discard_daemon_canary_candidate(
            archive_root=archive_root,
            generation_id=generation_id,
            generation_owner_id=generation_owner_id,
        )
    except BaseException as exc:
        errors.append(exc)
    return errors


def _add_canary_cleanup_notes(primary: BaseException, errors: list[BaseException]) -> None:
    """Surface candidate cleanup failures without hiding the route failure."""
    if errors:
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        primary.add_note(f"canary candidate cleanup also failed: {detail}")


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

    def field(name: str) -> object:
        return receipt.get(name) if isinstance(receipt, dict) else getattr(receipt, name, None)

    receipt_root = field("archive_root")
    if not isinstance(receipt_root, str) or Path(receipt_root).resolve() != root:
        raise CanarySelectionError("rebuild receipt belongs to a different archive root")
    if field("status") != "replayed" or field("materialized") is not True:
        raise CanarySelectionError("rebuild receipt is not a completed materialized replay")
    if field("selected_raw_count") != len(selection.selected_raw_ids):
        raise CanarySelectionError("rebuild receipt selected a different raw-id set than the canary")

    generation = field("generation")
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


def _validate_nonempty_selection(selection: CanarySelection) -> None:
    """Keep the canary route from inheriting the rebuild engine's full-source default."""

    if not selection.selected_session_ids:
        raise CanarySelectionError("canary selection contains zero sessions; refusing full-source replay")
    if not selection.selected_raw_ids:
        raise CanarySelectionError("canary selection contains zero raw ids; refusing full-source replay")
    if len(set(selection.selected_session_ids)) != len(selection.selected_session_ids):
        raise CanarySelectionError("canary selection contains duplicate session ids")
    if len(set(selection.selected_raw_ids)) != len(selection.selected_raw_ids):
        raise CanarySelectionError("canary selection contains duplicate raw ids")


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
    authority_kind: CanaryAuthorityKind | None = None
    authority_id: str | None = None

    def __post_init__(self) -> None:
        kind = self.authority_kind
        authority_id = self.authority_id
        has_explicit_authority = kind is not None or authority_id is not None
        if (kind is None) != (authority_id is None):
            raise UnclassifiedCanaryDiffError("canary review authority must provide both kind and id")
        if kind is None or authority_id is None:
            prefix, separator, value = self.reference.partition(":")
            if separator and prefix in {item.value for item in CanaryAuthorityKind}:
                kind = CanaryAuthorityKind(prefix)
                authority_id = value
            elif self.classification is DifferenceClassification.EXPECTED:
                # Preserve compatibility for in-process callers. The durable
                # representation below is always structured.
                kind = CanaryAuthorityKind.BEAD
                authority_id = self.reference
            else:
                kind = CanaryAuthorityKind.SUCCESSOR
                authority_id = self.reference
        if not str(authority_id).strip() or any(character.isspace() for character in str(authority_id)):
            raise UnclassifiedCanaryDiffError("canary review authority must have a structured non-empty id")
        canonical_reference = f"{kind.value}:{authority_id}"
        if has_explicit_authority and self.reference != canonical_reference:
            raise UnclassifiedCanaryDiffError("canary review reference disagrees with its structured authority")
        if self.classification is DifferenceClassification.EXPECTED and kind not in {
            CanaryAuthorityKind.BEAD,
            CanaryAuthorityKind.DELTA,
        }:
            raise UnclassifiedCanaryDiffError("expected canary differences require a Bead or delta authority")
        if self.classification is DifferenceClassification.UNEXPECTED and kind is not CanaryAuthorityKind.SUCCESSOR:
            raise UnclassifiedCanaryDiffError("unexpected canary differences require a structured successor id")
        object.__setattr__(self, "authority_kind", kind)
        object.__setattr__(self, "authority_id", str(authority_id))
        object.__setattr__(self, "reference", canonical_reference)

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
        authority_kind = self.authority_kind
        assert authority_kind is not None
        return {
            "table": self.table,
            "operation": self.operation.value,
            "identity": dict(self.identity),
            "changed_columns": list(self.changed_columns),
            "classification": self.classification.value,
            "reference": self.reference,
            "authority": {"kind": authority_kind.value, "id": self.authority_id},
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
            "schema_version": 9,
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
    """Persist reviewed evidence, or one explicitly non-approvable discovery report.

    Reviews must cover exactly the comparator's row identities and signatures.
    This prevents a report from becoming a durable green-light artifact while a diff was
    silently omitted from review. The write is atomic and touches only the
    requested report path, never either SQLite generation.
    """

    _validate_rebuild_receipt(
        rebuild_receipt,
        selected_raw_ids=selection.selected_raw_ids,
        selected_session_ids=selection.selected_session_ids,
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
    _validate_expected_review_authorities(review_list)
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
    incomplete = bool(duplicate_keys or missing_keys or extra_keys)
    if incomplete and not (allow_unreviewed and not review_list):
        detail = [
            f"duplicate={len(duplicate_keys)}",
            f"missing={len(missing_keys)}",
            f"extra={len(extra_keys)}",
        ]
        raise UnclassifiedCanaryDiffError("canary report classification is incomplete (" + ", ".join(detail) + ")")
    review_status = "unreviewed" if incomplete else "reviewed"

    archive_provenance = _capture_archive_provenance(comparison, rebuild_receipt)
    if review_status == "reviewed":
        reviewed_differences = tuple(
            replace(
                difference,
                classification=review_by_key[
                    (difference.table, difference.operation, difference.identity, difference.changed_columns)
                ].classification,
                rationale=_reviewed_difference_rationale(
                    review_by_key[
                        (difference.table, difference.operation, difference.identity, difference.changed_columns)
                    ]
                ),
            )
            for difference in comparison.differences
        )
        reviewed_comparison = replace(comparison, differences=reviewed_differences)
    else:
        reviewed_comparison = comparison
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
    except BaseException as exc:
        if temporary is not None:
            try:
                temporary.unlink()
            except BaseException as cleanup_error:
                exc.add_note(
                    f"canary report temporary-file cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise
    return durable


def _validate_selection_binding(
    selection: CanarySelection,
    comparison: CanaryDiffReport,
    receipt: dict[str, object],
    *,
    archive_root: Path | None = None,
) -> None:
    _validate_nonempty_selection(selection)
    if selection.index_path.resolve() != comparison.current_index.resolve():
        raise UnclassifiedCanaryDiffError("canary report selection index does not match the compared current index")
    if not comparison.session_ids:
        raise UnclassifiedCanaryDiffError("canary report has an empty comparison scope")
    if selection.selected_session_ids != comparison.session_ids:
        raise UnclassifiedCanaryDiffError("canary report selection sessions do not match the comparison")
    _validate_selection_evidence(
        receipt,
        selection.selected_raw_ids,
        selected_session_ids=selection.selected_session_ids,
        archive_root=archive_root,
        expected_parser_fingerprints=selection.parser_fingerprints,
        expected_lowering_fingerprint=selection.lowering_fingerprint,
        expected_replay_routing_fingerprint=selection.replay_routing_fingerprint,
        expected_materializer_fingerprint=selection.materializer_fingerprint,
    )


def _validate_selection_evidence(
    receipt: dict[str, object],
    selected_raw_ids: Iterable[str],
    *,
    selected_session_ids: Iterable[str] = (),
    archive_root: Path | None = None,
    expected_parser_fingerprints: Iterable[tuple[str, str]] = (),
    expected_lowering_fingerprint: str | None = None,
    expected_replay_routing_fingerprint: str | None = None,
    expected_materializer_fingerprint: str | None = None,
) -> None:
    """Match report selection to the rebuild-owned candidate commitment."""

    from polylogue.maintenance.rebuild_index import rebuild_selection_evidence
    from polylogue.storage.sqlite.archive_tiers.revision_governance import FrozenSourceRemediationRequiredError

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
    receipt_archive_root = cast(str, receipt["archive_root"])
    evidence_root = Path(receipt_archive_root)
    if archive_root is not None:
        evidence_root = archive_root.resolve()
        if Path(receipt_archive_root).resolve() != evidence_root:
            raise UnclassifiedCanaryDiffError(
                "canary report rebuild receipt belongs to a different configured archive root"
            )
    try:
        expected = rebuild_selection_evidence(
            tuple(selected_raw_ids),
            archive_root=evidence_root,
            generation_id=cast(str, generation["generation_id"]),
            generation_owner_id=cast(str, generation["owner_id"]),
            candidate_index=Path(cast(str, generation["index_path"])),
            source_snapshot=cast(str, generation["source_snapshot"]),
            selected_session_ids=tuple(selected_session_ids),
        )
    except FrozenSourceRemediationRequiredError as exc:
        raise UnclassifiedCanaryDiffError(
            "canary report selection evidence cannot be recomputed after source remediation changed"
        ) from exc
    if evidence.get("selected_session_ids") != expected["selected_session_ids"]:
        raise UnclassifiedCanaryDiffError(
            "canary report selection sessions do not match the authoritative rebuild receipt"
        )
    if evidence != expected:
        raise UnclassifiedCanaryDiffError("canary report selection does not match the authoritative rebuild receipt")
    _validate_parser_binding(
        evidence,
        expected_parser_fingerprints=expected_parser_fingerprints,
        expected_lowering_fingerprint=expected_lowering_fingerprint,
        expected_replay_routing_fingerprint=expected_replay_routing_fingerprint,
        expected_materializer_fingerprint=expected_materializer_fingerprint,
    )
    _validate_canary_acceptance_evidence(receipt)
    _validate_live_replay_routing_fingerprint(expected_replay_routing_fingerprint)
    _validate_live_materializer_fingerprint(expected_materializer_fingerprint)
    _validate_live_parser_and_lowering_fingerprints(expected_parser_fingerprints, expected_lowering_fingerprint)


def _validate_canary_acceptance_evidence(receipt: dict[str, object]) -> None:
    """Require the running daemon's full canonical canary-profile attestation."""

    from polylogue.maintenance.archive_verification import (
        REINDEX_CANARY_ACCEPTANCE_CHECKS,
        REINDEX_CANARY_ACCEPTANCE_PROFILE,
    )

    acceptance = receipt.get("canary_acceptance")
    if not isinstance(acceptance, dict):
        raise UnclassifiedCanaryDiffError("canary report has no daemon acceptance-profile attestation")
    if acceptance.get("profile") != REINDEX_CANARY_ACCEPTANCE_PROFILE:
        raise UnclassifiedCanaryDiffError("canary report acceptance profile does not match the running daemon")
    results = acceptance.get("results")
    if not isinstance(results, list):
        raise UnclassifiedCanaryDiffError("canary report acceptance attestation has no per-check results")
    expected_names = list(REINDEX_CANARY_ACCEPTANCE_CHECKS)
    actual_names: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            raise UnclassifiedCanaryDiffError("canary report acceptance attestation has invalid per-check results")
        name = result.get("name")
        status = result.get("status")
        if not isinstance(name, str) or not isinstance(status, str):
            raise UnclassifiedCanaryDiffError("canary report acceptance attestation has invalid per-check results")
        actual_names.append(name)
        if status != "ok":
            raise UnclassifiedCanaryDiffError(f"canary report acceptance check {name} is not ok")
    if actual_names != expected_names:
        raise UnclassifiedCanaryDiffError("canary report acceptance attestation does not match the running profile")


def _validate_live_replay_routing_fingerprint(expected: str | None) -> None:
    """Reject a persisted canary when installed replay routing has changed."""

    if expected is None:
        return
    from polylogue.sources.origin_specs import replay_routing_fingerprint

    if replay_routing_fingerprint() != expected:
        raise UnclassifiedCanaryDiffError("canary replay routing fingerprint no longer matches the running code")


def _validate_live_materializer_fingerprint(expected: str | None) -> None:
    """Reject a persisted canary when installed materialization has changed."""

    if expected is None:
        return
    from polylogue.sources.origin_specs import materializer_fingerprint

    if materializer_fingerprint() != expected:
        raise UnclassifiedCanaryDiffError("canary materializer fingerprint no longer matches the running code")


def _validate_live_parser_and_lowering_fingerprints(
    expected_parsers: Iterable[tuple[str, str]], expected_lowering: str | None
) -> None:
    """Reject reports whose parser or lowering semantics differ from live code."""
    from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin

    if expected_lowering is not None and lowering_fingerprint() != expected_lowering:
        raise UnclassifiedCanaryDiffError("canary lowering fingerprint no longer matches the running code")
    for origin, fingerprint in expected_parsers:
        if parser_fingerprint_for_origin(origin) != fingerprint:
            raise UnclassifiedCanaryDiffError("canary parser fingerprints no longer match the running code")


def _validate_parser_binding(
    evidence: object,
    *,
    expected_parser_fingerprints: Iterable[tuple[str, str]],
    expected_lowering_fingerprint: str | None,
    expected_replay_routing_fingerprint: str | None = None,
    expected_materializer_fingerprint: str | None = None,
) -> None:
    """Keep parser semantics bound to the selected origin evidence."""
    expected_parsers = dict(expected_parser_fingerprints)
    if (
        not expected_parsers
        and expected_lowering_fingerprint is None
        and expected_replay_routing_fingerprint is None
        and expected_materializer_fingerprint is None
    ):
        return
    raw_session_evidence = _raw_session_evidence(evidence)
    if raw_session_evidence is None:
        raise UnclassifiedCanaryDiffError("canary selection has no parser-fingerprint evidence")
    parser_by_origin: dict[str, str] = {}
    lowering_values: set[str] = set()
    replay_routing_values: set[str] = set()
    materializer_values: set[str] = set()
    for item in raw_session_evidence:
        if not isinstance(item, dict):
            raise UnclassifiedCanaryDiffError("canary parser-fingerprint evidence is malformed")
        origin = item.get("origin")
        parser = item.get("parser_fingerprint")
        lowering = item.get("lowering_fingerprint")
        replay_routing = item.get("replay_routing_fingerprint")
        materializer = item.get("materializer_fingerprint")
        if not all(isinstance(value, str) and value for value in (origin, parser, lowering)):
            raise UnclassifiedCanaryDiffError("canary parser-fingerprint evidence is incomplete")
        if expected_replay_routing_fingerprint is not None and not isinstance(replay_routing, str):
            raise UnclassifiedCanaryDiffError("canary parser-fingerprint evidence is incomplete")
        if expected_materializer_fingerprint is not None and not isinstance(materializer, str):
            raise UnclassifiedCanaryDiffError("canary parser-fingerprint evidence is incomplete")
        parser_by_origin[str(origin)] = str(parser)
        lowering_values.add(str(lowering))
        if expected_replay_routing_fingerprint is not None:
            replay_routing_values.add(str(replay_routing))
        if expected_materializer_fingerprint is not None:
            materializer_values.add(str(materializer))
    if any(parser_by_origin.get(origin) != parser for origin, parser in expected_parsers.items()):
        raise UnclassifiedCanaryDiffError("canary parser fingerprints no longer match the selected origins")
    if expected_lowering_fingerprint is not None and lowering_values != {expected_lowering_fingerprint}:
        raise UnclassifiedCanaryDiffError("canary lowering fingerprint no longer matches the selected origins")
    if expected_replay_routing_fingerprint is not None and replay_routing_values != {
        expected_replay_routing_fingerprint
    }:
        raise UnclassifiedCanaryDiffError("canary replay routing fingerprint no longer matches the selected origins")
    if expected_materializer_fingerprint is not None and materializer_values != {expected_materializer_fingerprint}:
        raise UnclassifiedCanaryDiffError("canary materializer fingerprint no longer matches the selected origins")


def _parser_binding_from_evidence(
    evidence: object,
) -> tuple[tuple[tuple[str, str], ...], str | None, str | None, str | None]:
    raw_session_evidence = _raw_session_evidence(evidence)
    if raw_session_evidence is None:
        raise UnclassifiedCanaryDiffError("canary selection has no parser-fingerprint evidence")
    parsers: dict[str, str] = {}
    lowering: set[str] = set()
    replay_routing: set[str] = set()
    materializer: set[str] = set()
    for item in raw_session_evidence:
        if not isinstance(item, dict):
            raise UnclassifiedCanaryDiffError("canary parser-fingerprint evidence is malformed")
        origin = item.get("origin")
        parser = item.get("parser_fingerprint")
        value = item.get("lowering_fingerprint")
        routing = item.get("replay_routing_fingerprint")
        materialized = item.get("materializer_fingerprint")
        if not all(isinstance(entry, str) and entry for entry in (origin, parser, value, routing, materialized)):
            raise UnclassifiedCanaryDiffError("canary parser-fingerprint evidence is incomplete")
        parsers[str(origin)] = str(parser)
        lowering.add(str(value))
        replay_routing.add(str(routing))
        materializer.add(str(materialized))
    if len(lowering) != 1 or len(replay_routing) != 1 or len(materializer) != 1:
        raise UnclassifiedCanaryDiffError("canary selection has inconsistent parser routing fingerprints")
    return tuple(sorted(parsers.items())), next(iter(lowering)), next(iter(replay_routing)), next(iter(materializer))


def _raw_session_evidence(evidence: object) -> list[object] | None:
    if not isinstance(evidence, dict):
        return None
    direct = evidence.get("raw_session_evidence")
    if isinstance(direct, list):
        return direct
    closure = evidence.get("replay_closure")
    if isinstance(closure, dict) and isinstance(closure.get("raw_session_evidence"), list):
        return cast(list[object], closure["raw_session_evidence"])
    return None


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
    selected_session_ids: Iterable[str],
    candidate_index: Path | str,
    configured_archive_root: Path | None = None,
) -> None:
    requested_raw_ids = tuple(selected_raw_ids)
    if not isinstance(receipt, dict):
        raise UnclassifiedCanaryDiffError("canary report has no rebuild receipt")
    archive_root = receipt.get("archive_root")
    selected_raw_count = receipt.get("selected_raw_count")
    generation = receipt.get("generation")
    source_evidence_after = receipt.get("source_evidence_after")
    if (
        receipt.get("receipt_schema_version") != 4
        or not isinstance(archive_root, str)
        or not archive_root
        or not requested_raw_ids
        or selected_raw_count != len(requested_raw_ids)
        or receipt.get("status") != "replayed"
        or receipt.get("materialized") is not True
        or not isinstance(generation, dict)
        or not isinstance(source_evidence_after, str)
        or len(source_evidence_after) != 64
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
    if (
        configured_archive_root is not None
        and Path(cast(str, receipt["archive_root"])).resolve() != configured_archive_root.resolve()
    ):
        raise UnclassifiedCanaryDiffError(
            "canary report rebuild receipt belongs to a different configured archive root"
        )
    if Path(generation_index).resolve() != Path(candidate_index).resolve():
        raise UnclassifiedCanaryDiffError("canary report rebuild receipt does not identify the compared candidate")
    _validate_selection_evidence(
        receipt,
        requested_raw_ids,
        selected_session_ids=selected_session_ids,
        archive_root=configured_archive_root,
    )


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


def _verified_source_evidence(root: Path) -> str:
    """Recompute source evidence and normalize byte-verification failures."""

    from polylogue.storage.index_generation import rebuild_source_evidence_snapshot

    try:
        return rebuild_source_evidence_snapshot(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise UnclassifiedCanaryDiffError(
            f"archive-owned source evidence cannot verify referenced blob bytes: {exc}"
        ) from exc


def _capture_archive_provenance(comparison: CanaryDiffReport, receipt: dict[str, object]) -> dict[str, object]:
    """Capture lifecycle records that make a local report archive-specific.

    These values are not a secret or a signature. They are deliberately
    re-read on load so approval follows the active pointer, generation record,
    source snapshot, and the exact inactive file that the rebuild created.
    """

    from polylogue.storage.archive_identity import ArchiveLocation, TierFileIdentity

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
    source_snapshot = _verified_source_evidence(root)
    if source_snapshot != candidate["source_snapshot"]:
        raise UnclassifiedCanaryDiffError("archive-owned source snapshot does not match the inactive candidate")
    source_evidence_after = _verified_source_evidence(root)
    if source_evidence_after != receipt.get("source_evidence_after"):
        raise UnclassifiedCanaryDiffError("archive-owned source evidence does not match the rebuild receipt")
    parser_fingerprints, lowering_fingerprint, replay_routing_fingerprint, materializer_fingerprint = (
        _parser_binding_from_evidence(receipt.get("selection_evidence"))
    )
    return {
        "archive_root": str(root.resolve()),
        "active_pointer": str(location.active_pointer) if location.active_pointer is not None else None,
        "active_index": _index_evidence(location.active_index_path),
        "active_generation": _active_generation_metadata(root, location),
        "candidate_generation": candidate_metadata,
        "candidate_index": _index_evidence(candidate_path),
        "source_snapshot": source_snapshot,
        "source_evidence_after": source_evidence_after,
        "parser_fingerprints": dict(parser_fingerprints),
        "lowering_fingerprint": lowering_fingerprint,
        "replay_routing_fingerprint": replay_routing_fingerprint,
        "materializer_fingerprint": materializer_fingerprint,
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
    parser_fingerprints, lowering_fingerprint, replay_routing_fingerprint, materializer_fingerprint = (
        _parser_binding_from_evidence(receipt.get("selection_evidence"))
    )
    if provenance.get("parser_fingerprints") != dict(parser_fingerprints):
        raise UnclassifiedCanaryDiffError("archive-owned parser fingerprints no longer match the report")
    if provenance.get("lowering_fingerprint") != lowering_fingerprint:
        raise UnclassifiedCanaryDiffError("archive-owned lowering fingerprint no longer matches the report")
    if provenance.get("replay_routing_fingerprint") != replay_routing_fingerprint:
        raise UnclassifiedCanaryDiffError("archive-owned replay routing fingerprint no longer matches the report")
    if provenance.get("materializer_fingerprint") != materializer_fingerprint:
        raise UnclassifiedCanaryDiffError("archive-owned materializer fingerprint no longer matches the report")
    source_snapshot = _verified_source_evidence(root)
    if provenance.get("source_snapshot") != source_snapshot or live_fields["source_snapshot"] != source_snapshot:
        raise UnclassifiedCanaryDiffError("archive-owned source snapshot no longer matches the inactive candidate")
    source_evidence_after = _verified_source_evidence(root)
    if (
        provenance.get("source_evidence_after") != source_evidence_after
        or receipt.get("source_evidence_after") != source_evidence_after
    ):
        raise UnclassifiedCanaryDiffError("archive-owned source evidence no longer matches the rebuild receipt")


def _validate_report_archive_root(payload: dict[str, object], *, configured_archive_root: Path) -> Path:
    """Bind report and receipt roots before reading any archive evidence."""

    root = configured_archive_root.resolve()
    receipt = payload.get("rebuild_receipt")
    if not isinstance(receipt, dict):
        raise UnclassifiedCanaryDiffError("canary report has no rebuild receipt")
    receipt_root = receipt.get("archive_root")
    if not isinstance(receipt_root, str) or not receipt_root:
        raise UnclassifiedCanaryDiffError("canary report has invalid rebuild receipt archive root")
    provenance = payload.get("archive_provenance")
    if Path(receipt_root).resolve() != root:
        raise UnclassifiedCanaryDiffError(
            "canary report rebuild receipt belongs to a different configured archive root"
        )
    if isinstance(provenance, dict):
        report_root = provenance.get("archive_root")
        if isinstance(report_root, str) and report_root and Path(report_root).resolve() != root:
            raise UnclassifiedCanaryDiffError("canary report belongs to a different configured archive root")
    return root


def load_canary_report(path: Path, *, archive_root: Path | None = None) -> dict[str, object]:
    """Read and structurally revalidate a durable report's review coverage."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UnclassifiedCanaryDiffError("canary report root must be an object")
    if payload.get("schema_version") != 9:
        raise UnclassifiedCanaryDiffError("canary report has no authoritative rebuild receipt schema")
    configured_root = (
        _validate_report_archive_root(payload, configured_archive_root=Path(archive_root))
        if archive_root is not None
        else None
    )
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
    _validate_expected_review_authorities(reviews)
    review_status = payload.get("review_status")
    if review_status != "reviewed":
        raise UnclassifiedCanaryDiffError("canary report is not fully reviewed")
    comparison_fingerprint = payload.get("comparison_fingerprint")
    if not isinstance(comparison_fingerprint, str) or len(comparison_fingerprint) != 64:
        raise UnclassifiedCanaryDiffError("canary report has invalid comparison fingerprint")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise UnclassifiedCanaryDiffError("canary report has no selection object")
    selected_raw_ids = selection.get("selected_raw_ids")
    selection_sessions = selection.get("selected_session_ids")
    candidate_index = comparison.get("candidate_index")
    if (
        not isinstance(selected_raw_ids, list)
        or not selected_raw_ids
        or not all(isinstance(value, str) and value for value in selected_raw_ids)
    ):
        raise UnclassifiedCanaryDiffError("canary report has invalid selected raw ids")
    if (
        not isinstance(selection_sessions, list)
        or not selection_sessions
        or not all(isinstance(value, str) and value for value in selection_sessions)
    ):
        raise UnclassifiedCanaryDiffError("canary report has invalid selected session ids")
    if not isinstance(candidate_index, str):
        raise UnclassifiedCanaryDiffError("canary report has no candidate index")
    _validate_rebuild_receipt(
        payload.get("rebuild_receipt"),
        selected_raw_ids=cast(list[str], selected_raw_ids),
        selected_session_ids=cast(list[str], selection_sessions),
        candidate_index=candidate_index,
        configured_archive_root=configured_root,
    )
    selection_index = selection.get("index_path")
    sessions_per_origin = selection.get("sessions_per_origin")
    raw_parser_fingerprints = selection.get("parser_fingerprints")
    lowering_fingerprint = selection.get("lowering_fingerprint")
    replay_routing_fingerprint = selection.get("replay_routing_fingerprint")
    materializer_fingerprint = selection.get("materializer_fingerprint")
    if (
        not isinstance(selection_index, str)
        or not isinstance(sessions_per_origin, int)
        or not isinstance(raw_parser_fingerprints, dict)
        or not all(
            isinstance(origin, str) and isinstance(fingerprint, str)
            for origin, fingerprint in raw_parser_fingerprints.items()
        )
        or (lowering_fingerprint is not None and not isinstance(lowering_fingerprint, str))
        or (replay_routing_fingerprint is not None and not isinstance(replay_routing_fingerprint, str))
        or (materializer_fingerprint is not None and not isinstance(materializer_fingerprint, str))
    ):
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
        parser_fingerprints=tuple(sorted(cast(dict[str, str], raw_parser_fingerprints).items())),
        lowering_fingerprint=lowering_fingerprint,
        replay_routing_fingerprint=replay_routing_fingerprint,
        materializer_fingerprint=materializer_fingerprint,
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
        persisted_selection,
        persisted_comparison,
        cast(dict[str, object], payload["rebuild_receipt"]),
        archive_root=configured_root,
    )
    from polylogue.config import resolve_archive_root

    _validate_archive_provenance(
        payload.get("archive_provenance"),
        configured_archive_root=configured_root if configured_root is not None else resolve_archive_root(),
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
    if missing_keys or extra_keys:
        raise UnclassifiedCanaryDiffError(
            f"canary report review coverage is incomplete (missing={len(missing_keys)}, extra={len(extra_keys)})"
        )
    for key, difference in difference_by_key.items():
        review = review_by_key[key]
        if review.classification is not difference.classification:
            raise UnclassifiedCanaryDiffError("canary report review classification disagrees with its difference")
        if difference.rationale != _reviewed_difference_rationale(review):
            raise UnclassifiedCanaryDiffError(
                "canary report difference rationale disagrees with its structured authority"
            )
    comparison["summary"] = _summary_for_differences(differences)
    return cast(dict[str, object], payload)


def approve_canary_report(path: Path, *, archive_root: Path) -> dict[str, object]:
    """Approve evidence only under the archive's existing writer ownership."""

    from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation, assert_owns_archive_location
    from polylogue.storage.index_generation import RebuildLease

    root = Path(archive_root)
    location = ArchiveLocation.resolve(root)
    with RebuildLease(root):
        owned = OwnedArchiveLocation.acquire(location)
        try:
            assert_owns_archive_location(owned, ArchiveLocation.resolve(root))
            _validate_approved_canary_report(path, root)

            # The first load proves that the report was valid when approval
            # began. Re-run the complete report, source, candidate, and
            # comparison validation immediately before returning approval.
            assert_owns_archive_location(owned, ArchiveLocation.resolve(root))
            final_payload = _validate_approved_canary_report(path, root)
            assert_owns_archive_location(owned, ArchiveLocation.resolve(root))
            return final_payload
        finally:
            owned.release()


def approve_canary_report_under_daemon_ownership(path: Path, *, archive_root: Path) -> dict[str, object]:
    """Validate report approval while the daemon write coordinator owns the archive."""

    from polylogue.daemon.write_coordinator import daemon_write_lease_active

    if not daemon_write_lease_active():
        raise RuntimeError("canary report consumption requires the daemon write coordinator")
    root = Path(archive_root)
    _validate_approved_canary_report(path, root)
    return _validate_approved_canary_report(path, root)


def _validate_approved_canary_report(path: Path, archive_root: Path) -> dict[str, object]:
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
    authority = value.get("authority")
    rationale = value.get("rationale")
    if (
        not isinstance(identity, dict)
        or not isinstance(changed_columns, list)
        or not changed_columns
        or not all(isinstance(column, str) for column in changed_columns)
        or not isinstance(table, str)
        or not isinstance(reference, str)
        or not isinstance(authority, dict)
        or not isinstance(authority.get("kind"), str)
        or not isinstance(authority.get("id"), str)
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
    try:
        authority_kind = CanaryAuthorityKind(authority["kind"])
    except (KeyError, ValueError) as exc:
        raise UnclassifiedCanaryDiffError("canary report review has invalid structured authority") from exc
    return CanaryDifferenceReview(
        table=table,
        operation=operation,
        identity=tuple((str(key), item) for key, item in identity.items()),
        changed_columns=tuple(cast(list[str], changed_columns)),
        classification=classification,
        reference=reference,
        rationale=rationale,
        authority_kind=authority_kind,
        authority_id=authority["id"],
    )


def load_canary_review_manifest(path: Path) -> tuple[CanaryDifferenceReview, ...]:
    """Load the explicit per-difference review manifest accepted by the CLI."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("reviews"), list):
        raise UnclassifiedCanaryDiffError("canary review manifest must contain a reviews list")
    reviews = tuple(_review_from_dict(item) for item in cast(list[object], payload["reviews"]))
    _validate_expected_review_authorities(reviews)
    return reviews


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


def _reviewed_difference_rationale(review: CanaryDifferenceReview) -> str:
    """Return the only persisted rationale text a structured review may authorize."""
    return f"{review.reference}: {review.rationale}"


def _validate_expected_review_authorities(reviews: Iterable[CanaryDifferenceReview]) -> None:
    """Resolve every review authority from its independent canonical source."""
    from devtools.pr_scope import load_committed_bead_records
    from polylogue.storage.sqlite.lifecycle import INDEX_DELTA_DECLARATIONS

    expected_deltas = {
        review.authority_id
        for review in reviews
        if review.classification is DifferenceClassification.EXPECTED
        and review.authority_kind is CanaryAuthorityKind.DELTA
        and review.authority_id is not None
    }
    declared_deltas = {str(declaration.version) for declaration in INDEX_DELTA_DECLARATIONS}
    unknown_deltas = sorted(delta for delta in expected_deltas if delta not in declared_deltas)
    if unknown_deltas:
        detail = ", ".join(f"unknown index delta {delta}" for delta in unknown_deltas)
        raise UnclassifiedCanaryDiffError(f"expected canary authority is not declared in packaged evidence: {detail}")

    try:
        bead_records = load_committed_bead_records()
    except (OSError, ValueError) as exc:
        raise UnclassifiedCanaryDiffError(
            "cannot resolve canary review authority from committed Bead evidence"
        ) from exc

    expected_beads = {
        review.authority_id
        for review in reviews
        if review.classification is DifferenceClassification.EXPECTED
        and review.authority_kind is CanaryAuthorityKind.BEAD
        and review.authority_id is not None
    }
    unknown_beads = sorted(bead_id for bead_id in expected_beads if bead_id not in bead_records)
    if unknown_beads:
        detail = ", ".join(f"unknown Bead {bead_id}" for bead_id in unknown_beads)
        raise UnclassifiedCanaryDiffError(
            f"expected canary authority is not declared in committed Bead evidence: {detail}"
        )

    successors = {
        review.authority_id
        for review in reviews
        if review.classification is DifferenceClassification.UNEXPECTED
        and review.authority_kind is CanaryAuthorityKind.SUCCESSOR
        and review.authority_id is not None
    }
    unknown_successors = sorted(successor_id for successor_id in successors if successor_id not in bead_records)
    if unknown_successors:
        detail = ", ".join(f"unknown successor {successor_id}" for successor_id in unknown_successors)
        raise UnclassifiedCanaryDiffError(
            f"unexpected canary authority is not open in committed Bead evidence: {detail}"
        )
    closed_successors = sorted(
        successor_id for successor_id in successors if bead_records[successor_id].get("status") != "open"
    )
    if closed_successors:
        detail = ", ".join(f"closed successor {successor_id}" for successor_id in closed_successors)
        raise UnclassifiedCanaryDiffError(
            f"unexpected canary authority is not open in committed Bead evidence: {detail}"
        )


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
    "resolved_dst_session_id",
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
    scope_columns = (
        ("src_session_id",)
        if table == "session_links" and "src_session_id" in columns
        else tuple(column for column in _SESSION_SCOPE_COLUMNS if column in columns)
    )
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
    "CanaryAuthorityKind",
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
    "approve_canary_report_under_daemon_ownership",
    "compare_reindex_generations",
    "load_canary_report",
    "run_reindex_canary",
    "select_canary_sessions",
    "write_canary_report",
]
