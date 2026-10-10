"""Raw-observation derivation over retained bytes and logical membership.

The adapter owns discovery and output inspection. Publication uses the existing
revision-governance replay seam, which still owns durable arbitration and its
per-logical-key transactions. This is not an observation-wide atomic publisher.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import sqlite3
import sys
import tempfile
import time
import weakref
from builtins import BaseExceptionGroup
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from dataclasses import field as dataclasses_field
from functools import partial
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, BinaryIO, Literal, Protocol, TypeVar, cast

from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionKind,
    is_work_event_raw_id,
    parser_census_identity_measurement,
    raw_authority_parser_fingerprint,
)
from polylogue.archive.revision_replay import RevisionReplayPlan
from polylogue.archive.session_revision_membership import MembershipClassification
from polylogue.core.compute import (
    BoundedComputeAdapter,
    DaemonBackpressureError,
    DaemonOperationCancelled,
    SubmittedOperation,
)
from polylogue.core.compute_cancel import check_compute_cancelled, compute_cancel_requested
from polylogue.core.enums import Origin, Provider, ValidationMode
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
from polylogue.core.raw_failure_evidence import (
    RAW_FAILURE_DEFERRED_SUPPORT_STATUS,
    RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS,
    RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS,
    RAW_FAILURE_VALIDATION_FAILURE_KINDS,
    CohortMembershipRefusalError,
    RetainedRawDecodeRefusalError,
)
from polylogue.core.sql_settlement import retain_native_sql_lifetimes
from polylogue.logging import WARNING, emit
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.blob_store import (
    BlobStore,
    BlobVerificationCancelledError,
    PreparedBlob,
    blob_store_for_connection,
)
from polylogue.storage.raw_authority import (
    assess_raw_replay_materialization,
    iter_parser_census_logical_keys,
)
from polylogue.storage.source_blob_restoration import stage_blob_from_recorded_source
from polylogue.storage.sqlite.archive_tiers.source_write import PENDING_RAW_LOGICAL_SOURCE_PREFIX
from polylogue.storage.sqlite.connection_profile import attach_readonly_database, open_readonly_connection
from polylogue.storage.sqlite.queries.raw_state import raw_provider_origin_sql

if TYPE_CHECKING:
    from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.sources.revision_backfill import (
        PreparedMembershipReplay,
        PreparedRetainedAggregate,
        PreparedRetainedInput,
        PreparedRevisionReplayResult,
        RevisionCensusResult,
    )
    from polylogue.sources.sidecar_evidence import RetainedSidecarScope
    from polylogue.storage.index_generation import IndexGeneration
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead, PreparedSessionWrite
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    #: A Source phase committed in place: its receipt and the fields its tape changed.
    _InPlacePhase = tuple[tuple[Literal["census", "classification"], RevisionCensusResult], ...]

RAW_OBSERVATION_DOMAIN = "raw_observation"


def _record_retained_schema_drift(
    archive_root: Path,
    prepared_inputs: Mapping[str, PreparedRetainedInput] | None,
    *,
    index_db_path: Path,
    strict_refusals_only: bool = False,
) -> None:
    """Persist admitted retained-validation observations without gating writes."""
    if not prepared_inputs:
        return
    observations = []
    for retained in prepared_inputs.values():
        verdict = retained.validation_verdict
        if verdict is None or verdict.drift_observation is None:
            continue
        if strict_refusals_only and not verdict.strict_refusal:
            continue
        observations.append(verdict.drift_observation)
    if not observations:
        return
    try:
        from polylogue.schemas.drift_sentinel_sampling import record_schema_drift_observations_to_ops_sync

        record_schema_drift_observations_to_ops_sync(
            index_db_path,
            observations,
            archive_root=archive_root,
        )
    except Exception:
        from polylogue.logging import get_logger

        get_logger(__name__).debug("schema drift sampling failed after retained receipt", exc_info=True)


def raw_observation_recipe_version(validation_mode: ValidationMode = ValidationMode.ADVISORY) -> str:
    """Identify the parser and effective validation policy for raw replay."""
    parser_fingerprint = raw_authority_parser_fingerprint()
    digest = hashlib.sha256()
    digest.update(parser_fingerprint.encode("ascii"))
    digest.update(b"\0schema-validation\0")
    digest.update(validation_mode.value.encode("ascii"))
    return digest.hexdigest()


#: Origins whose enrichment reads session-scoped retained evidence. Both
#: origins map to exactly one provider, so the reverse lookup is not a guess.
_EVIDENCE_PROVIDER_BY_ORIGIN: Mapping[str, Provider] = {
    Origin.CLAUDE_CODE_SESSION.value: Provider.CLAUDE_CODE,
    Origin.CODEX_SESSION.value: Provider.CODEX,
}


def raw_replay_error_is_retryable(error: object, durable_retryable: bool = False) -> bool:
    """Recognize durable retry evidence and exact historical replay refusals."""
    if durable_retryable:
        return True
    if not isinstance(error, str):
        return False
    return (
        error == "OperationalError: database is locked"
        or error.startswith(("MembershipReplayConflictError:", "membership_replay_conflict:"))
        or (error.startswith("decode:") and "No such file or directory" in error)
        or error
        in {
            "RuntimeError: raw revision CAS rejected an older accepted frontier",
            "RuntimeError: membership replay cannot replace an unconvertible byte head",
        }
    )


class RawFrame(Protocol):
    @property
    def archive_root(self) -> str: ...

    @property
    def source_revision(self) -> str: ...

    @property
    def scope(self) -> object | None: ...

    def recipe_version(self, domain: str) -> str: ...


@dataclass(frozen=True, slots=True)
class RawObservationScope:
    raw_ids: tuple[str, ...] = ()


def _discard_staged_blobs(store: BlobStore, prepared: tuple[PreparedBlob, ...]) -> None:
    for item in prepared:
        store.discard_prepared(item)


class StagedBlobRestorations:
    """Exact source bytes staged for absent retained blobs, awaiting the writer.

    Compute stages them without a lease; ``publish`` reserves and publishes
    them under the archive writer. A replacement the kernel never hands to
    ``publish`` still has its staged files discarded when this owner is
    collected.
    """

    def __init__(self, store: BlobStore, staged: Sequence[tuple[str, PreparedBlob]]) -> None:
        self.store = store
        self.staged = tuple(staged)
        self._finalizer = weakref.finalize(
            self, _discard_staged_blobs, store, tuple(prepared for _raw_id, prepared in self.staged)
        )

    def published(self) -> None:
        """The writer moved the staged files into place; nothing is left to discard."""
        self._finalizer.detach()

    def discard(self) -> None:
        self._finalizer()


def _replacement_phase(replacement: RawObservationReplacement) -> Literal["census", "classification"]:
    """The Source phase a preparatory replacement would commit."""
    return "census" if replacement.needs_source_census else "classification"


@dataclass(slots=True)
class _PreparationCarry:
    """One preparation's seal and parsed artifacts, continued across its Source phases.

    After a census or classification commits in place, the seal rebases onto
    its own accepted tape and the next phase is prepared on it. An artifact is
    reused only under its unchanged descriptor key and while the enrichment
    and sidecar evidence it was parsed against is unchanged.
    """

    neutral_page: NeutralRawPreparation | None = None
    seal: PreparedIndexMutation | None = None
    raw_ids: tuple[str, ...] = ()
    scratch_owner: tempfile.TemporaryDirectory[str] | None = None
    artifacts: dict[tuple[object, ...], PreparedJsonl] = dataclasses_field(default_factory=dict)
    neutral_artifacts: dict[tuple[object, ...], PreparedJsonl] = dataclasses_field(default_factory=dict)
    attachment_refs_published: set[int] = dataclasses_field(default_factory=set)
    #: Each prepared raw's captured ZIP coordinate, read with its descriptor.
    zip_coordinates: dict[str, CapturedZipMemberCoordinate | None] = dataclasses_field(default_factory=dict)
    committed: list[tuple[Literal["census", "classification"], RevisionCensusResult]] = dataclasses_field(
        default_factory=list
    )

    def discard_payload(
        self,
        *,
        keep: RawObservationReplacement | None = None,
        preserve_neutral: bool = False,
    ) -> None:
        """Discard carried artifacts and scratch that ``keep`` does not itself close."""
        kept_artifacts = (
            set()
            if keep is None
            else {
                id(item.prepared_artifact)
                for item in (keep.prepared_inputs or {}).values()
                if item.prepared_artifact is not None
            }
        )
        failures: list[BaseException] = []
        try:
            _close_prepared_carriers(
                {},
                {},
                (
                    artifact
                    for artifact in (
                        *self.artifacts.values(),
                        *(() if preserve_neutral else self.neutral_artifacts.values()),
                    )
                    if id(artifact) not in kept_artifacts
                ),
            )
        except BaseException as failure:
            failures.append(failure)
        self.artifacts.clear()
        if not preserve_neutral:
            self.neutral_artifacts.clear()
        self.attachment_refs_published.clear()
        if keep is not None and keep.scratch_owner is self.scratch_owner and not preserve_neutral:
            self.scratch_owner = None
        if preserve_neutral and self.neutral_artifacts:
            if failures:
                raise BaseExceptionGroup("carried preparation cleanup failed", failures)
            return
        if self.scratch_owner is not None and not failures:
            try:
                _cleanup_scratch(self.scratch_owner)
            except BaseException as failure:
                failures.append(failure)
            self.scratch_owner = None
        if failures:
            raise BaseExceptionGroup("carried preparation cleanup failed", failures)


@dataclass(frozen=True, slots=True)
class _CapturedNeutralRaw:
    descriptor: tuple[Provider, str, str, RawRevisionKind, int]
    profile_identity: str | None
    fallback_timestamp: str | None
    native_id: str | None
    zip_coordinate: CapturedZipMemberCoordinate | None
    append_logical_key: str | None
    staged_blob: Path


@dataclass(frozen=True, slots=True)
class _NeutralParserOperand:
    descriptor: tuple[Provider, str, str, RawRevisionKind, int]
    profile_identity: str | None
    fallback_timestamp: str | None
    native_id: str | None
    zip_coordinate: CapturedZipMemberCoordinate | None
    append_logical_key: str | None
    sidecar_signature: tuple[object, ...] | None = None


@dataclass(slots=True)
class NeutralRawPreparation:
    """Closed capture files and exact parser operands for one admitted Raw page.

    Only the ordered owner mutates ``artifacts``. Workers receive fixed captures
    and resolver facts; none receives a Source reader, seal or publication right.
    The page remains alive until every parser and canonical consumer settles.
    """

    scratch_owner: tempfile.TemporaryDirectory[str]
    captures: Mapping[str, _CapturedNeutralRaw]
    operands: Mapping[str, _NeutralParserOperand]
    sidecar_scopes: Mapping[str, RetainedSidecarScope]
    artifacts: dict[tuple[object, ...], PreparedJsonl] = dataclasses_field(default_factory=dict)

    def parser_jobs(
        self, validation_mode: ValidationMode
    ) -> Iterator[tuple[tuple[object, ...], int, Callable[[], PreparedJsonl]]]:
        # Keep the existing first/second/head economy for long Codex chains.
        groups: dict[str, list[str]] = {}
        selected = set(self.captures)
        for raw_id, captured in self.captures.items():
            provider, _hash, path, kind, _size = captured.descriptor
            if provider is Provider.CODEX and kind.value in {"full", "unknown"}:
                groups.setdefault(path, []).append(raw_id)
        for ids in groups.values():
            ids.sort(key=lambda item: self.captures[item].descriptor[4])
            if len(ids) >= 4:
                selected.difference_update(ids[2:-1])
        # Sidecar/sibling inputs are conservatively charged to each parser.
        sidecar_bytes = sum(
            path.stat().st_size for path in Path(self.scratch_owner.name).glob("claude-sidecar-*/payload.bin")
        )
        for raw_id, captured in self.captures.items():
            if raw_id in selected:
                key = _neutral_artifact_key(raw_id, self.operands[raw_id], validation_mode)
                yield (
                    key,
                    captured.descriptor[4] + sidecar_bytes,
                    partial(
                        _parse_captured_neutral,
                        raw_id,
                        captured,
                        key,
                        self.sidecar_scopes,
                        Path(self.scratch_owner.name),
                    ),
                )

    def close(self) -> None:
        _close_prepared_carriers({}, {}, self.artifacts.values())
        self.artifacts.clear()
        _cleanup_scratch(self.scratch_owner)


def _parse_captured_neutral(
    raw_id: str,
    captured: _CapturedNeutralRaw,
    artifact_key: tuple[object, ...],
    sidecar_scopes: Mapping[str, RetainedSidecarScope],
    scratch: Path,
) -> PreparedJsonl:
    """Return sealed files after creator-owned SQLite and decoded state retire."""
    from polylogue.sources.dispatch import is_stream_record_provider
    from polylogue.sources.fallback_identity import fallback_session_id
    from polylogue.sources.live.batch_support import jsonl_parse_prefix_size_of_handle
    from polylogue.sources.prepared_jsonl import prepare_jsonl_blob
    from polylogue.sources.prepared_message_sink import discard_decoded_sessions_under
    from polylogue.sources.sidecar_evidence import CapturedSidecarResolver

    provider, blob_hash, source_path, kind, _size = captured.descriptor
    with captured.staged_blob.open("rb") as handle:
        parse_prefix_size = jsonl_parse_prefix_size_of_handle(handle)
    fallback_id = fallback_session_id(source_path, raw_id)
    if kind.value == "append":
        from polylogue.sources.revision_backfill import _append_session_native_id

        fallback_id = (
            _append_session_native_id(
                captured.append_logical_key, provider=provider, captured_native_id=captured.native_id
            )
            or fallback_id
        )
    try:
        return prepare_jsonl_blob(
            str(captured.staged_blob),
            source_path,
            provider.value,
            fallback_id,
            is_stream=is_stream_record_provider(source_path, provider),
            profile_identity=captured.profile_identity,
            shard_directory=str(scratch),
            attempt_directory=captured.staged_blob.parent,
            source_sha256=blob_hash,
            strict_jsonl_records=True,
            parse_prefix_size=parse_prefix_size,
            sidecar_resolver=CapturedSidecarResolver(sidecar_scopes),
            progress_identity=_neutral_identity_digest(("neutral-parser-work-v1", artifact_key)),
            captured_zip_coordinate=captured.zip_coordinate,
        )
    finally:
        discard_decoded_sessions_under(captured.staged_blob.parent)


def _neutral_artifact_key(
    raw_id: str,
    operand: _NeutralParserOperand,
    validation_mode: ValidationMode,
) -> tuple[object, ...]:
    """Identify parser output independently of the freshly bound publication cohort."""
    return (
        _neutral_parser_cache_identity(raw_id, operand),
        ("validation-mode", validation_mode.value),
    )


def _neutral_parser_cache_identity(raw_id: str, operand: _NeutralParserOperand) -> tuple[object, ...]:
    """Keep source kind exact for rebind proof while sharing equivalent full/unknown parses."""
    provider, blob_hash, source_path, kind, size = operand.descriptor
    coordinate = operand.zip_coordinate
    zip_identity: tuple[object, ...] | None = None
    if coordinate is not None:
        zip_identity = (
            "captured-zip-coordinate",
            coordinate.canonical_container,
            coordinate.declared_container,
            coordinate.member_name,
            coordinate.entry_ordinal,
            coordinate.split_index,
            ("enum", type(coordinate.addressing_mode).__qualname__, coordinate.addressing_mode.value),
            coordinate.container_blob_hash,
            coordinate.decoder_fingerprint,
            coordinate.profile_namespace,
        )
    return (
        ("raw-id", raw_id),
        ("provider-enum", type(provider).__module__, type(provider).__qualname__, provider.value),
        ("blob-sha256", blob_hash),
        ("source-path", source_path),
        ("append-revision", kind.value == "append"),
        ("blob-size", size),
        ("profile-identity", operand.profile_identity),
        ("fallback-timestamp", operand.fallback_timestamp),
        ("native-id", operand.native_id),
        ("zip-coordinate", zip_identity),
        ("append-logical-key", operand.append_logical_key),
        ("sidecar-signature", operand.sidecar_signature),
    )


def _neutral_identity_digest(recipe: tuple[object, ...]) -> str:
    """Hash the typed, JSON-safe identity without lossy text coercion."""
    encoded = json.dumps(recipe, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _neutral_parser_operand(
    read: PreparedSessionSourceRead,
    raw_id: str,
    *,
    descriptor: tuple[Provider, str, str, RawRevisionKind, int] | None = None,
) -> _NeutralParserOperand:
    """Read the exact typed values a detached parser and its enrichment consume."""
    descriptor = read.raw_revision_descriptor(raw_id) if descriptor is None else descriptor
    provider, _blob_hash, source_path, kind, _raw_size = descriptor
    scope_signature: tuple[object, ...] | None = None
    if provider is Provider.CLAUDE_CODE:
        scope = read.retained_sidecar_resolver().claude_code_scope(source_path)
        scope_signature = (scope.scope_key, scope.available, scope.witness)
    return _NeutralParserOperand(
        descriptor=descriptor,
        profile_identity=read.raw_profile_identity(raw_id),
        fallback_timestamp=read.raw_revision_file_mtime(raw_id),
        native_id=read.raw_native_id(raw_id) if kind.value == "append" else None,
        zip_coordinate=read.raw_captured_zip_coordinate(raw_id),
        append_logical_key=read.raw_append_logical_key(raw_id) if kind.value == "append" else None,
        sidecar_signature=scope_signature,
    )


def _neutral_jsonl_candidate(provider: Provider, source_path: str) -> bool:
    """Whether a retained JSONL raw can be detached and parsed before rebinding."""
    from polylogue.sources.dispatch import is_jsonl_source_path
    from polylogue.sources.origin_specs import path_declaration_refuses_session
    from polylogue.sources.sqlite_export import looks_like_logical_source_path

    return (
        provider in {Provider.CODEX, Provider.CLAUDE_CODE}
        and is_jsonl_source_path(source_path)
        and not path_declaration_refuses_session(provider, source_path)
        and not looks_like_logical_source_path(Path(source_path))
    )


class _CarryInvalidatedError(Exception):
    """The carried inputs changed; prepare again from a new seal."""

    def __init__(
        self,
        reason: Literal["selection_changed", "parser_operands_changed", "carried_membership_changed"],
    ) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class RawObservationReplacement:
    key: str
    # Kernel diagnostic for selected coordinates/recipe, not a Source-value
    # fingerprint. The original witness supplies actual publication authority.
    input_binding: str
    # ``ReplacementLike`` requires a payload; this adapter carries its inputs
    # in the typed fields below instead.
    payload: None
    raw_ids: tuple[str, ...]
    prepared_inputs: Mapping[str, PreparedRetainedInput] | None = None
    prepared_aggregates: Mapping[str, PreparedRetainedAggregate] | None = None
    prepared_writes: Mapping[tuple[str, str], PreparedSessionWrite] | None = None
    prepared_replay_adoption: Mapping[tuple[str, tuple[str, ...]], PreparedRevisionAdoption] | None = None
    prepared_source_classification: PreparedRevisionSourceClassification | None = None
    verified_blob_stats: Mapping[str, tuple[int, int, int, int, int]] | None = None
    planned_accepted_raw_ids: Mapping[str, tuple[str, ...]] | None = None
    prepared_revision_plans: Mapping[str, RevisionReplayPlan] | None = None
    prepared_byte_outcomes: Mapping[str, PreparedRevisionReplayOutcome] | None = None
    prepared_membership_plans: Mapping[str, PreparedMembershipReplay] | None = None
    prepared_replay_source: PreparedRetainedReplaySource | None = None
    needs_source_census: bool = False
    prepared_source_census: PreparedRevisionSourceCensus | None = None
    prepared_replay_schedule: ReplaySchedule | None = None
    prepared_logical_keys: tuple[str, ...] = ()
    prepared_membership_keys: tuple[str, ...] = ()
    prepared_byte_logical_keys: tuple[str, ...] = ()
    prepared_key_refusals: tuple[CohortMembershipRefusalError, ...] = ()
    #: Logical keys whose parent publishes earlier in this same unit; they
    #: publish nothing here and are re-prepared against that parent.
    prepared_lineage_deferrals: tuple[str, ...] = ()
    #: The retained raws of those deferred keys; the rest of the unit published.
    lineage_deferred_raw_ids: tuple[str, ...] = ()
    #: Retained raws outside this unit whose sessions a write claims as its
    #: absent parent; compute widens the unit to them and prepares again.
    lineage_parent_raw_ids: tuple[str, ...] = ()
    needs_source_classification: bool = False
    #: Source phases this preparation already committed in place, in order;
    #: publication reports them whatever its own outcome.
    committed_phase_receipts: tuple[tuple[Literal["census", "classification"], RevisionCensusResult], ...] = ()
    #: A refused in-place classification, reported by publication as its own.
    prepared_phase_failure: BaseException | None = None
    #: The preparation's carried artifacts and scratch; closing this carrier
    #: releases whatever its own payload does not already hold.
    carried_payload: _PreparationCarry | None = None
    scratch_directory: Path | None = None
    scratch_owner: tempfile.TemporaryDirectory[str] | None = None
    empty: bool = False
    already_valid: bool = False
    blob_restorations: StagedBlobRestorations | None = None
    reference_seal: PreparedIndexMutation | None = None
    # Wall time this compute spent in the provider parsers that sealed its
    # carriers. Publication reports it beside its own writer timings.
    provider_parse_seconds: float = 0.0

    def close(self) -> None:
        """Drain this prepared carrier when publication is cancelled or ends."""
        if self.reference_seal is not None and self.reference_seal.publication_lifetime_bound:
            self.reference_seal.close()
            return
        failures: list[BaseException] = []
        for close in (
            self._close_prepared_payload,
            *(() if self.reference_seal is None else (self.reference_seal.close,)),
        ):
            try:
                close()
            except BaseException as failure:
                failures.append(failure)
        if failures:
            raise BaseExceptionGroup("retained preparation cleanup failed", failures)

    def retire_unpublished(self) -> None:
        """Retire a preparation that is prepared again instead of published.

        Its seal settles its native SQL first and then the payload, as at
        publication: a thread-state projection holds native handles under
        the scratch tree the payload removes.
        """
        seal = self.reference_seal
        if seal is None or seal.publication_lifetime_bound:
            self.close()
            return
        seal.retain_preparation_payload(self._close_prepared_payload)
        seal.close()

    def _close_prepared_payload(self, *, preserve_scratch: bool = False) -> None:
        """Settle the carrier payload without recursing into its retained seal."""
        with retain_native_sql_lifetimes(*(() if self.scratch_owner is None else (self.scratch_owner,))):
            failures: list[BaseException] = []
            for close in (
                *(() if self.carried_payload is None else (partial(self.carried_payload.discard_payload, keep=self),)),
                partial(
                    _close_prepared_carriers,
                    self.prepared_writes or {},
                    self.prepared_membership_plans or {},
                    chain(
                        (
                            item.prepared_artifact
                            for item in (self.prepared_inputs or {}).values()
                            if item.prepared_artifact is not None
                        ),
                        (item.artifact for item in (self.prepared_aggregates or {}).values()),
                    ),
                ),
                *(() if self.blob_restorations is None else (self.blob_restorations.discard,)),
            ):
                try:
                    close()
                except BaseException as failure:
                    failures.append(failure)
            # Scratch files depend on every native preparation owner. Keep
            # them on any failed close for the original creator's retry.
            if not failures and self.scratch_owner is not None and not preserve_scratch:
                try:
                    _cleanup_scratch(self.scratch_owner)
                except BaseException as failure:
                    failures.append(failure)
            if failures:
                raise BaseExceptionGroup("retained preparation cleanup failed", failures)


def _session_id(session: ParsedSession) -> str:
    from polylogue.core.sources import origin_from_provider

    return f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}"


def _holds_several_sessions(artifact: PreparedJsonl | None) -> bool:
    if artifact is None:
        return False
    with closing(artifact.iter_sessions()) as sessions:
        return next(sessions, None) is not None and next(sessions, None) is not None


class RawObservationInspection:
    """Read retained Raw state without admitting computation or publication."""

    domain = RAW_OBSERVATION_DOMAIN
    prerequisites: tuple[str, ...] = ()

    @property
    def inspection_validation_mode(self) -> ValidationMode | None:
        """Use the selected policy when the caller is certifying readiness."""
        return self._inspection_validation_mode

    def __init__(
        self, archive_root: Path, *, index_db_path: Path | None = None, validation_mode: ValidationMode | None = None
    ) -> None:
        self.archive_root = archive_root
        self._index_db_path = index_db_path
        self._inspection_validation_mode = validation_mode

    @property
    def recipe_version(self) -> str:
        mode = self.inspection_validation_mode
        return raw_authority_parser_fingerprint() if mode is None else raw_observation_recipe_version(mode)

    @contextmanager
    def read_current(self) -> Iterator[sqlite3.Connection]:
        """Borrow one coherent Source and selected Index snapshot for inspection."""
        with self._read() as conn:
            yield conn

    def inspect_current(self, conn: sqlite3.Connection, key: str) -> str:
        """Inspect a raw in the snapshot supplied by read_current."""
        return self._inspect(conn, key)

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        source = self.archive_root / "source.db"
        conn = open_readonly_connection(source, timeout_class="background-read", validate_schema=False)
        try:
            conn.row_factory = sqlite3.Row
            index = self._index_db_path or ArchiveLocation.resolve(self.archive_root).active_index_path
            attach_readonly_database(conn, index, alias="index_tier")
            conn.execute("BEGIN")
            yield conn
        finally:
            conn.close()

    def _require_bootstrapped_source_tier(self, conn: sqlite3.Connection) -> None:
        """Refuse a present-but-unbootstrapped source tier with a typed reason.

        ``source.db`` existing as a file is not proof that the durable source
        tier was ever bootstrapped: a bare ``sqlite3.connect(...)`` creates the
        file with no schema at all. Every caller already handles
        ``FileNotFoundError`` as "this backlog is unavailable, and here is why",
        so an absent ``raw_sessions`` table reports through that same channel
        rather than escaping as a bare ``OperationalError``.
        """
        if (
            conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'raw_sessions'").fetchone()
            is None
        ):
            raise FileNotFoundError(
                f"durable source tier is not bootstrapped: {self.archive_root / 'source.db'} has no raw_sessions table"
            )

    def _current(self, frame: RawFrame) -> bool:
        return (
            Path(frame.archive_root).resolve() == self.archive_root.resolve()
            and frame.source_revision
            == str((self._index_db_path or ArchiveLocation.resolve(self.archive_root).active_index_path).resolve())
            and frame.recipe_version(self.domain) == self.recipe_version
        )

    def required_page(self, frame: RawFrame, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        if limit < 1:
            return (), None
        if not (self.archive_root / "source.db").exists():
            if (self.archive_root / "index.db").exists() or (self.archive_root / ".index-active-pointer").exists():
                raise FileNotFoundError(f"durable source tier is missing: {self.archive_root / 'source.db'}")
            return (), None
        scope = frame.scope if isinstance(frame.scope, RawObservationScope) else RawObservationScope()
        predicates = ["r.raw_id > ?"]
        parameters: list[object] = [cursor or ""]
        if scope.raw_ids:
            predicates.append(f"r.raw_id IN ({','.join('?' for _ in scope.raw_ids)})")
            parameters.extend(scope.raw_ids)
        with self._read() as conn:
            self._require_bootstrapped_source_tier(conn)
            rows = conn.execute(
                f"SELECT r.raw_id FROM raw_sessions r WHERE {' AND '.join(predicates)} ORDER BY r.raw_id LIMIT ?",
                (*parameters, limit),
            ).fetchall()
        keys = tuple(str(row[0]) for row in rows)
        return keys, keys[-1] if len(keys) == limit else None

    def excess_page(self, frame: RawFrame, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], None]:
        # Durable raws are retained. Excess logical identities are inspected
        # within their observation, never deleted by guessing an absent owner.
        return (), None

    def prerequisite_keys(self, frame: RawFrame, key: str) -> tuple[()]:
        return ()

    def quiet(self, frame: RawFrame, key: str) -> bool:
        return False

    def inspect(self, frame: RawFrame, keys: Sequence[str]) -> Mapping[str, str]:
        if not keys:
            # Nothing to inspect: opening the read connection here would demand
            # an existing source.db purely to answer the empty question, which
            # is how a probe of an archive with no source tier used to die on
            # "unable to open database file". ``source_paths`` already guards
            # the same way.
            return {}
        if not self._current(frame):
            return dict.fromkeys(keys, "stale")
        with self._read() as conn:
            # Results belong only to this page and its pinned Source/Index read.
            # Retain requested IDs, never the component's full material rows.
            replay_assessments: dict[str, bool | None] = dict.fromkeys(keys)
            return {key: self._inspect(conn, key, replay_assessments=replay_assessments) for key in keys}

    def source_paths(self, keys: Sequence[str]) -> Mapping[str, str]:
        if not keys:
            return {}
        with self._read() as conn:
            return dict(
                conn.execute(
                    f"SELECT raw_id, source_path FROM raw_sessions WHERE raw_id IN ({','.join('?' for _ in keys)})",
                    tuple(keys),
                )
            )

    @staticmethod
    def _terminal_revision_refusal(conn: sqlite3.Connection, key: str, parser_fingerprint: str | None) -> bool:
        classifier_superseded = (
            parser_fingerprint is not None and parser_fingerprint != raw_authority_parser_fingerprint()
        )
        unresolved = conn.execute(
            """SELECT decision FROM raw_session_memberships WHERE raw_id = ? AND decision IN ('ambiguous', 'deferred')
            UNION ALL SELECT decision FROM index_tier.raw_revision_applications
            WHERE raw_id = ? AND decision IN ('ambiguous', 'deferred')""",
            (key, key),
        ).fetchall()
        return bool(unresolved) and not classifier_superseded

    def _decode_refusal(self, conn: sqlite3.Connection, key: str) -> RetainedRawDecodeRefusalError | None:
        from polylogue.core.raw_failure_evidence import retained_raw_decode_refusal_from_row
        from polylogue.storage.sqlite.queries.raw_state import retained_raw_decode_refusal_sql

        row = conn.execute(
            retained_raw_decode_refusal_sql(),
            (key, raw_authority_parser_fingerprint(), *sorted(RAW_FAILURE_VALIDATION_FAILURE_KINDS)),
        ).fetchone()
        return retained_raw_decode_refusal_from_row(key, row)

    def terminal_decode_refusals(self, keys: Sequence[str]) -> Mapping[str, RetainedRawDecodeRefusalError]:
        """Read the same exact current receipt used by inspection and compute."""
        if not keys:
            return {}
        with self._read() as conn:
            return {key: refusal for key in keys if (refusal := self._decode_refusal(conn, key)) is not None}

    def _inspect(
        self, conn: sqlite3.Connection, key: str, *, replay_assessments: dict[str, bool | None] | None = None
    ) -> str:
        from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin

        raw = conn.execute(
            f"SELECT r.*, {raw_provider_origin_sql(table_alias='r')} AS effective_origin FROM raw_sessions r WHERE raw_id = ?",
            (key,),
        ).fetchone()
        if raw is None:
            return "missing"
        # Semantic refusals belong to the classifier that produced them. A
        # changed classifier must recensus retained bytes before accepting
        # either its ambiguity or its deferral as current evidence.
        census = conn.execute("SELECT * FROM raw_authority_parser_census WHERE raw_id = ?", (key,)).fetchone()
        parser_fingerprint = raw_authority_parser_fingerprint()
        if census is not None and census["parser_fingerprint"] != parser_fingerprint:
            return "stale"
        if self._decode_refusal(conn, key) is not None:
            # A retained-byte decode refusal is independent of schema policy;
            # changing ADVISORY/STRICT cannot make those same bytes decodable.
            # Compute must report the permanent refusal without parsing again.
            return "stale"
        validation_mode = self.inspection_validation_mode
        if validation_mode is not None and raw["validation_mode"] != validation_mode.value:
            # Captured native grammar and non-session classifications can
            # exclude JSON schema policy. Their current parser census and
            # exact session reachability are still required below.
            typed_schema_ineligible = (
                conn.execute(
                    "SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM raw_artifacts WHERE raw_id=? "
                    "AND schema_eligible IS NOT 0) "
                    "AND ((EXISTS (SELECT 1 FROM raw_artifacts WHERE raw_id=?) "
                    "AND EXISTS (SELECT 1 FROM raw_artifacts WHERE raw_id=? AND parse_as_session=1)) "
                    "OR EXISTS (SELECT 1 FROM raw_membership_census WHERE raw_id=? AND status='non_session' "
                    "AND member_count=0 AND parser_fingerprint=?))",
                    (key, key, key, key, parser_fingerprint),
                ).fetchone()
                is not None
            )
            if typed_schema_ineligible:
                artifact_exemption = conn.execute(
                    "SELECT 1 FROM raw_artifacts WHERE raw_id=? AND schema_eligible=0",
                    (key,),
                ).fetchone()
                if artifact_exemption is None:
                    from polylogue.sources.dispatch import is_jsonl_source_path
                    from polylogue.sources.live.batch_support import jsonl_parse_prefix_size_of_handle

                    typed_schema_ineligible = False
                    if is_jsonl_source_path(raw["source_path"]):
                        retained_path = BlobStore(self.archive_root / "blob").blob_path(bytes(raw["blob_hash"]).hex())
                        try:
                            with retained_path.open("rb") as retained_input:
                                typed_schema_ineligible = jsonl_parse_prefix_size_of_handle(retained_input) == 0
                        except FileNotFoundError:
                            # Exact-source restoration is a compute admission.
                            # A missing CAS file must reach that owner rather
                            # than fail while inspecting its empty census.
                            return "stale"
            if not (
                raw["validation_mode"] is None
                and census is not None
                and census["status"] == "complete"
                and typed_schema_ineligible
            ):
                return "stale"
        if (
            raw["parse_error"]
            and conn.execute(
                f"""SELECT 1 FROM raw_artifacts WHERE raw_id = ?
            AND (origin IS ? OR origin IS ?) AND source_path IS ? AND source_index IS ?
            AND artifact_kind IN ({",".join("?" for _ in RAW_FAILURE_VALIDATION_FAILURE_KINDS)}) LIMIT 1""",
                (
                    key,
                    raw["origin"],
                    raw["effective_origin"],
                    raw["source_path"],
                    raw["source_index"],
                    *sorted(RAW_FAILURE_VALIDATION_FAILURE_KINDS),
                ),
            ).fetchone()
            is not None
        ):
            # A malformed terminal carrier cannot acquire authority through a
            # generic validation marker or the older parse-error fast paths.
            return "stale"
        if self._terminal_revision_refusal(conn, key, census["parser_fingerprint"] if census else None):
            return "valid"
        strict_schema_refusal = (
            raw["validation_mode"] == ValidationMode.STRICT.value
            and raw["validation_status"] == "failed"
            and raw["validation_error"] is not None
            and raw["parse_error"] is None
            and raw["validated_at_ms"] is not None
            and (raw["parsed_at_ms"] is None or raw["validated_at_ms"] >= raw["parsed_at_ms"])
            and conn.execute(
                "SELECT 1 FROM raw_artifacts WHERE raw_id=? AND parse_as_session=1 AND schema_eligible=1 LIMIT 1",
                (key,),
            ).fetchone()
            is not None
        )
        if strict_schema_refusal:
            # STRICT is an explicit terminal refusal after validation. An
            # ADVISORY schema failure remains parsable and must still produce
            # its accepted session before inspection can settle it.
            return "valid"
        error = raw["parse_error"]
        coordinates = (key, raw["origin"], raw["effective_origin"], raw["source_path"], raw["source_index"])
        exact_coordinate = "raw_id = ? AND (origin IS ? OR origin IS ?) AND source_path IS ? AND source_index IS ?"
        if (
            error
            and conn.execute(
                f"SELECT 1 FROM raw_artifacts WHERE {exact_coordinate} "
                f"AND (artifact_kind, support_status) IN ({','.join('(?, ?)' for _ in RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS)}) LIMIT 1",
                (
                    *coordinates,
                    *(value for pair in RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS for value in pair),
                ),
            ).fetchone()
            is not None
        ):
            return "valid"
        if error and not raw_replay_error_is_retryable(error):
            retry = conn.execute(
                f"""SELECT 1 FROM raw_artifacts WHERE {exact_coordinate} AND support_status = ?
                AND artifact_kind IN ({",".join("?" for _ in RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS)}) LIMIT 1""",
                (
                    *coordinates,
                    RAW_FAILURE_DEFERRED_SUPPORT_STATUS,
                    *sorted(RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS),
                ),
            ).fetchone()
            if retry is None:
                return "valid"
        membership = conn.execute("SELECT * FROM raw_membership_census WHERE raw_id = ?", (key,)).fetchone()
        if census is None:
            return "missing"
        if census["parser_fingerprint"] != parser_fingerprint or census["status"] != "complete":
            return "stale"
        non_session = (
            conn.execute(
                "SELECT 1 FROM raw_artifacts WHERE raw_id = ? AND parse_as_session = 0 LIMIT 1",
                (key,),
            ).fetchone()
            is not None
        )
        parser_non_session = (
            membership is not None
            and membership["status"] == "non_session"
            and membership["parser_fingerprint"] == parser_fingerprint
        )
        if (
            non_session or (membership is not None and membership["status"] == "non_session")
        ) and not parser_non_session:
            # Typed raw-only taxonomy and its independent parser membership
            # receipt must describe the same current classification.
            return "stale"
        member_count = 0

        def member_keys(rows: sqlite3.Cursor) -> Iterator[object]:
            nonlocal member_count
            for row in rows:
                member_count += 1
                yield row[0]

        with (
            closing(
                conn.execute(
                    "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id=? ORDER BY logical_source_key",
                    (key,),
                )
            ) as members,
            parser_census_identity_measurement(
                raw_logical_key=raw["logical_source_key"],
                revision_kind=raw["revision_kind"],
                membership_logical_keys=member_keys(members),
                observed_logical_keys=iter_parser_census_logical_keys(census["logical_keys_json"]),
                observed_are_receipt=True,
            ) as measured,
        ):
            if not measured.complete(
                typed_non_session=non_session,
                parser_confirmed_non_session=membership is not None
                and membership["status"] == "non_session"
                and membership["parser_fingerprint"] == parser_fingerprint,
                byte_governed_fragment=raw["source_index"] < 0
                and membership is not None
                and membership["revision_authority"] == RawRevisionAuthority.BYTE_PROVEN.value,
            ):
                return "stale"
            if (
                membership is not None
                and membership["status"] == "complete"
                and membership["member_count"] != member_count
            ):
                return "stale"
            with closing(conn.execute("SELECT session_id FROM index_tier.sessions WHERE raw_id=?", (key,))) as owned:
                for row in owned:
                    with closing(
                        measured.connection.execute(
                            "SELECT 1 FROM census_identity WHERE kind=1 AND logical_key=?", (str(row[0]),)
                        )
                    ) as identity:
                        if identity.fetchone() is None:
                            return "excess"
            if not measured.observed_count or non_session or parser_non_session:
                # A non-session artifact (a hook carrier on its physical
                # chain) inherits its own durable chain key into the census
                # but owns no session output: there is no revision
                # application or head to verify, and demanding one left the
                # key "missing" on every pass.
                return "valid"
            with closing(measured.iter_durable_bindings()) as bindings:
                for logical_key, source_key in bindings:
                    decision = None
                    if source_key is not None:
                        with closing(
                            conn.execute(
                                "SELECT decision FROM raw_session_memberships WHERE raw_id=? AND logical_source_key=?",
                                (key, source_key),
                            )
                        ) as rows:
                            decision = rows.fetchone()
                    if decision is not None and decision[0] in {"ambiguous", "deferred"}:
                        continue
                    if decision is not None and decision[0] is None:
                        return "missing"
                    application = conn.execute(
                        """SELECT 1 FROM index_tier.raw_revision_applications
                        WHERE raw_id = ? AND logical_source_key = ?
                          AND decision IN ('selected_baseline', 'applied_append', 'superseded', 'reparse_reaffirmation')
                        LIMIT 1""",
                        (key, logical_key),
                    ).fetchone()
                    if application is None:
                        return "missing"
                    output = conn.execute(
                        """SELECT s.origin, s.parser_fingerprint, s.lowering_fingerprint, s.session_id, s.native_id,
                               b.evidence_key, accepted.source_path AS accepted_source_path
                        FROM index_tier.raw_revision_heads h JOIN index_tier.sessions s
                          ON s.session_id = h.session_id AND s.raw_id = h.accepted_raw_id
                         AND s.content_hash = h.accepted_content_hash
                        LEFT JOIN index_tier.session_enrichment_bindings b ON b.session_id = s.session_id
                        LEFT JOIN raw_sessions accepted ON accepted.raw_id = h.accepted_raw_id
                        WHERE h.logical_source_key = ?""",
                        (logical_key,),
                    ).fetchone()
                    if output is None:
                        return "missing"
                    if (
                        output["parser_fingerprint"] != parser_fingerprint_for_origin(Origin(output["origin"]))
                        or output["lowering_fingerprint"] != lowering_fingerprint()
                    ):
                        return "stale"
                    # The binding was written from the accepted head's raw; a
                    # superseded sibling from another directory reads different
                    # evidence and could never match it, so compare the head's own.
                    evidence_path = output["accepted_source_path"] or raw["source_path"]
                    if self._enrichment_evidence_moved(conn, evidence_path, output):
                        return "stale"
        # Every per-Raw census, policy, output and refusal check above still
        # runs. Only this snapshot's complete execution-component receipt is
        # shared with requested siblings that reach this same final boundary.
        check_compute_cancelled()
        if replay_assessments is not None and (assessed := replay_assessments.get(key)) is not None:
            return "valid" if assessed else "stale"
        from polylogue.storage.sqlite.archive_tiers.revision_governance import expand_raw_membership_selection_sync

        component, _logical_keys = expand_raw_membership_selection_sync(conn, [key])
        # Refused siblings have their own terminal authority; they did not
        # execute and cannot supply an execution receipt for accepted siblings.
        # A superseded classifier must still leave that sibling in repair debt.
        execution_component = tuple(
            str(row["raw_id"])
            for row in conn.execute(
                f"""SELECT r.raw_id, c.parser_fingerprint FROM raw_sessions r
                LEFT JOIN raw_authority_parser_census c ON c.raw_id = r.raw_id
                WHERE r.raw_id IN ({",".join("?" for _ in component)}) ORDER BY r.raw_id""",
                component,
            ).fetchall()
            if not self._terminal_revision_refusal(conn, str(row["raw_id"]), row["parser_fingerprint"])
        )
        exact, _problems = assess_raw_replay_materialization(
            conn,
            execution_component,
            index_db_path=self._index_db_path or ArchiveLocation.resolve(self.archive_root).active_index_path,
        )
        if replay_assessments is not None:
            for raw_id in execution_component:
                if raw_id in replay_assessments:
                    replay_assessments[raw_id] = exact
        return "valid" if exact else "stale"

    def _enrichment_evidence_moved(self, conn: sqlite3.Connection, source_path: object, output: sqlite3.Row) -> bool:
        """Whether the evidence this output was enriched from is no longer current.

        A session index, prompt history or thread-state export can be admitted
        after the session it describes, in any order. The writer binds each
        output to the evidence it read; here the evidence the archive holds now
        is compared with that binding. An absent binding cannot certify the
        output, so it is derived again on the retained route -- ordinary
        derivation from durable evidence, whatever order the bytes arrived in.
        """
        from polylogue.sources.revision_backfill import session_enrichment_evidence_key

        provider = _EVIDENCE_PROVIDER_BY_ORIGIN.get(str(output["origin"]))
        if provider is None or not isinstance(source_path, str):
            return False
        # ``session_enrichment_evidence_key`` (and ``read_thread_titles``
        # beneath it) queries ``work_evidence_*`` unqualified, expecting
        # ``index_conn``'s own ``main`` schema to be index.db. ``conn`` here
        # has source.db as ``main`` and the index attached only as
        # ``index_tier``, so passing it as ``index_conn`` resolved those
        # tables against the wrong schema (silently empty, not an error the
        # caller sees), and Codex evidence never registered as moved. Open a
        # real index connection for this specific read.
        index_path = ArchiveLocation.resolve(self.archive_root).active_index_path
        with closing(
            open_readonly_connection(index_path, timeout_class="background-read", validate_schema=False)
        ) as index_conn:
            current = session_enrichment_evidence_key(
                provider=provider,
                source_path=source_path,
                native_id=str(output["native_id"]),
                index_conn=index_conn,
                source_conn=conn,
                blob_root=self.archive_root / "blob",
            )
        return current is not None and output["evidence_key"] != current


def _publish_acquired_attachment_refs(
    seal: PreparedIndexMutation,
    prepared_inputs: Mapping[str, PreparedRetainedInput],
    *,
    blob_store: BlobStore,
) -> None:
    """Settle exact acquired claims before derived replay can fail or skip."""
    from polylogue.core.iterator_lifetime import settled_iterator
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        _PreparedSourceProducer,
        publish_prepared_revision_source,
    )
    from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceBlobRef, _write_source_blob_refs
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

    artifacts: dict[int, PreparedJsonl] = {}
    for retained in prepared_inputs.values():
        artifact = retained.prepared_artifact
        if artifact is not None and (artifact.error is not None or retained.parser_error is not None):
            raise ValueError("errored retained input cannot publish attachment claims")
        if artifact is not None and id(artifact) not in artifacts:
            with closing(artifact.iter_attachment_claims()) as claims:
                if next(claims, None) is not None:
                    artifacts[id(artifact)] = artifact
    if not artifacts:
        return
    has_refs = False
    with seal.original_read_snapshot(), seal.source_producer():
        reader = PreparedSessionSourceRead(seal, blob_store=blob_store)
        producer = _PreparedSourceProducer(seal)
        for artifact in artifacts.values():
            original_raw_id, original_input = next(
                (raw_id, item) for raw_id, item in prepared_inputs.items() if item.prepared_artifact is artifact
            )
            source_path = original_input.source_path
            with settled_iterator(
                artifact.iter_attachment_refs(
                    source_path=source_path,
                    acquired_at_ms=reader.raw_revision_observation_order(original_raw_id)[0],
                    source_read=reader,
                    before_input=seal.retain_prepared_blob_input,
                )
            ) as refs:
                for ref in refs:
                    # One exact published claim can prove several identical Raw
                    # carriers; read its reservation once before receipt consumption.
                    consumed = False
                    for raw_id, retained in prepared_inputs.items():
                        if retained.prepared_artifact is artifact:
                            if retained.blob_hash != artifact.blob_hash:
                                raise ValueError("acquired attachment carrier differs from its original Raw bytes")
                            current_ref = replace(
                                ref,
                                acquired_at_ms=reader.raw_revision_observation_order(raw_id)[0],
                                publication_receipt_id=None if consumed else ref.publication_receipt_id,
                            )

                            def current_refs(
                                current_ref: ArchiveSourceBlobRef = current_ref,
                            ) -> Iterator[ArchiveSourceBlobRef]:
                                yield current_ref

                            _write_source_blob_refs(producer, raw_id, current_refs)
                            consumed = True
                            has_refs = True
    if has_refs:
        permit = seal.prepare_source_mutation()
        admit_stage_write("retained-acquired-attachment-refs", partial(publish_prepared_revision_source, seal, permit))


class RawObservationDerivation(RawObservationInspection):
    """A paged raw adapter; no ops hint or backlog census certifies validity.

    Preparation can run without a writer lease. Existing synchronous recovery
    callers still hold their enclosing lease; their composition must move
    before that production route can claim lease-free computation.
    """

    domain = RAW_OBSERVATION_DOMAIN
    prerequisites: tuple[str, ...] = ()
    # The Source census and byte classification stage on one tape that
    # ``compute`` commits in place on the writer; preparation then continues
    # to the replay on the same seal, reusing each artifact whose descriptor
    # and enrichment evidence the commit left unchanged. A phase that cannot
    # commit in place, restored bytes and in-unit lineage deferral commit
    # through ``publish`` instead, and the kernel continues them within one
    # pass while :meth:`publication_advanced` reports committed progress.
    # Every advance is one that cannot repeat for the same state, so the
    # continuation is bounded by progress rather than a phase count.

    @property
    def inspection_validation_mode(self) -> ValidationMode:
        return self._validation_mode

    def __init__(
        self,
        archive_root: Path,
        *,
        prepare_non_json_artifact: RetainedArtifactPreparer | None = None,
        compute_adapter: BoundedComputeAdapter,
        prepaid_blob_inputs: tuple[tuple[str, bytes, int], ...] = (),
        index_db_path: Path | None = None,
        owned_generation: IndexGeneration | None = None,
        validation_mode: ValidationMode = ValidationMode.ADVISORY,
    ) -> None:
        super().__init__(archive_root, index_db_path=index_db_path)
        self._prepare_non_json_artifact = prepare_non_json_artifact
        self._compute_adapter = compute_adapter
        self._prepaid_blob_inputs = prepaid_blob_inputs
        self._index_db_path = index_db_path
        self._owned_generation = owned_generation
        self._validation_mode = validation_mode
        from polylogue.schemas.runtime_registry import SchemaRegistry

        self._schema_registry = SchemaRegistry()
        #: Replacements whose publication committed a prerequisite phase,
        #: consumed by :meth:`publication_advanced` on the same key.
        self._phase_committed: dict[int, str] = {}
        if owned_generation is not None:
            from polylogue.storage.sqlite.reference_seal import IndexMutationDestination

            destination = IndexMutationDestination.owned_inactive(owned_generation)
            if Path(owned_generation.archive_root).resolve(strict=True) != archive_root.resolve(strict=True):
                raise ValueError("retained replay generation belongs to another archive")
            if index_db_path is not None and index_db_path.resolve(strict=True) != destination.index_path:
                raise ValueError("retained replay Index differs from its owned generation")
            self._index_db_path = destination.index_path

    @property
    def recipe_version(self) -> str:
        """Bind retained preparation validity to its effective schema policy."""
        return raw_observation_recipe_version(self._validation_mode)

    @staticmethod
    def _lineage_deferrals(
        prepared_writes: Mapping[tuple[str, str], PreparedSessionWrite],
        selected_writes: Mapping[tuple[str, str], tuple[ParsedSession, PreparedJsonl]],
        *,
        write_keys: Mapping[str, tuple[str, str]],
        refused_keys: Collection[str],
    ) -> tuple[str, ...]:
        """Logical keys that must not publish together with their claimed parent.

        Each write was prepared against the Index as it stood before the unit,
        so a child whose parent is absent there expects no parent; if that
        parent publishes in the same unit, the child's write finds it and
        refuses as moved lineage. Such a child is deferred and re-prepared
        against the published parent.

        A key defers while any other write of this unit produces its parent,
        so a lineage chain (a fork of a fork) converges one generation per
        phase. A parent cycle has no such head: its least key publishes first,
        without its still-absent parent. A refused key publishes nothing, so
        it is never a parent here. Every pass publishes at least one key and
        the deferred keys re-prepare with fewer in-unit parents, so
        re-preparation terminates by progress, without a deferral budget.
        """
        from polylogue.core.sources import origin_from_provider

        publishing = {key: write_key for key, write_key in write_keys.items() if key not in refused_keys}
        key_by_session = {session_id: key for key, (_raw_id, session_id) in publishing.items()}
        parent_of: dict[str, str] = {}
        for logical_key, write_key in publishing.items():
            write = prepared_writes.get(write_key)
            selected = selected_writes.get(write_key)
            if write is None or selected is None or write.context.parent_session_id is not None:
                continue
            session = selected[0]
            claimed = write.context.hook_parent_native_id or session.parent_session_provider_id
            if not claimed:
                continue
            parent_session_id = f"{origin_from_provider(session.source_name).value}:{claimed.strip()}"
            parent_key = key_by_session.get(parent_session_id)
            if parent_session_id == write_key[1] or parent_key is None or parent_key == logical_key:
                continue
            parent_of[logical_key] = parent_key

        # One generation per phase: a key publishes only when no other write
        # of this unit produces its parent. A parent cycle has no such key, so
        # its least key anchors it and publishes without its still-absent
        # parent; every other member waits for the generation before it.
        heads = {key for key in publishing if key not in parent_of}
        for start in sorted(parent_of):
            path: list[str] = []
            node = start
            while node in parent_of and node not in path:
                path.append(node)
                node = parent_of[node]
            if node in path:
                heads.add(min(path[path.index(node) :]))
        return tuple(sorted(key for key in publishing if key not in heads))

    @staticmethod
    def _lineage_parent_raw_ids(
        reference_seal: PreparedIndexMutation,
        read: PreparedSessionSourceRead,
        selected_writes: Mapping[tuple[str, str], tuple[ParsedSession, PreparedJsonl]],
        *,
        unit_raw_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Retained raws of claimed parents that are absent from the Index and this unit.

        A parent shares its child's origin. It is found by its logical key once
        censused, and by its acquired native id before that. A parent whose
        retained bytes refuse to decode is left out: widening must not make a
        child fail on it.
        """
        from polylogue.core.sources import origin_from_provider

        produced = {session_id for _raw_id, session_id in selected_writes}
        claims: dict[str, tuple[str, str]] = {}
        for (_raw_id, session_id), (session, _artifact) in selected_writes.items():
            claimed = (session.parent_session_provider_id or "").strip()
            if not claimed:
                continue
            origin = origin_from_provider(session.source_name).value
            parent_session_id = f"{origin}:{claimed}"
            if parent_session_id == session_id or parent_session_id in produced:
                continue
            reference_seal.before_index_input(
                "sessions", ("session_id",), "SELECT rowid FROM sessions WHERE session_id=?", (parent_session_id,)
            )
            with closing(
                reference_seal.observer("index").execute(
                    "SELECT 1 FROM sessions WHERE session_id=?", (parent_session_id,)
                )
            ) as present:
                if present.fetchone() is not None:
                    continue
            claims.setdefault(parent_session_id, (origin, claimed))
        if not claims:
            return ()
        candidates: set[str] = set()
        with read.replay_representative_rows(sorted(claims)) as rows:
            for _logical_source_key, raw_id in rows:
                candidates.add(str(raw_id))
        for origin, native_id in sorted(set(claims.values())):
            candidates.update(read.raw_ids_for_native_session(origin, native_id))
        unit = set(unit_raw_ids)
        return tuple(
            raw_id
            for raw_id in sorted(candidates)
            if raw_id not in unit and read.raw_terminal_decode_refusal(raw_id) is None
        )

    def publication_advanced(self, replacement: RawObservationReplacement) -> bool:
        """Whether this replacement's publication committed a prerequisite phase.

        False from :meth:`publish` otherwise means a refusal or a moved input;
        only a committed restoration, census or classification is the key's
        own progress that a fresh preparation can continue.
        """
        return self._phase_committed.pop(id(replacement), None) == replacement.key

    #: The durable Source rows a census or classification phase may change.
    _CENSUS_STATE_TABLES = (
        "raw_sessions",
        "raw_artifacts",
        "raw_session_memberships",
        "raw_membership_census",
        "raw_authority_parser_census",
    )
    # Repeating a receipt's activity timestamp does not advance Source
    # interpretation. Acquisition coordinates and all semantic fields remain
    # part of this comparison; only these producer bookkeeping clocks differ.
    _CENSUS_ACTIVITY_COLUMNS = {
        "raw_sessions": frozenset({"parsed_at_ms", "validated_at_ms"}),
        "raw_membership_census": frozenset({"censused_at_ms"}),
        "raw_session_memberships": frozenset({"decided_at_ms"}),
    }

    def _census_state(self, raw_ids: Sequence[str]) -> tuple[tuple[object, ...], ...]:
        """Committed census state of ``raw_ids``; equal before and after means no progress."""
        from polylogue.storage.sqlite.connection_profile import readonly_connection_context

        selected = tuple(sorted(raw_ids))
        marks = ",".join("?" for _ in selected)
        state: list[tuple[object, ...]] = []
        with readonly_connection_context(self.archive_root / "source.db") as source:
            for table in self._CENSUS_STATE_TABLES:
                activity = self._CENSUS_ACTIVITY_COLUMNS.get(table, frozenset())
                columns = tuple(
                    str(row[1]) for row in source.execute(f"PRAGMA table_info({table})") if row[1] not in activity
                )
                with closing(
                    source.execute(
                        f"SELECT {','.join(columns)} FROM {table} WHERE raw_id IN ({marks}) ORDER BY rowid", selected
                    )
                ) as rows:
                    state.extend((table, *row) for row in rows)
        return tuple(state)

    @staticmethod
    def _blob_stat_identity(path: Path) -> tuple[int, int, int, int, int]:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @contextmanager
    def _preparation_archive(self) -> Iterator[ArchiveStore]:
        if self._owned_generation is None:
            from polylogue.operations.operation_context import open_operation_read

            with open_operation_read(self.archive_root) as pinned:
                yield pinned.archive
            return
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.storage.sqlite.reference_seal import IndexMutationDestination

        destination = IndexMutationDestination.owned_inactive(self._owned_generation)
        with ArchiveStore.open_owned_inactive_read(self._owned_generation) as archive:
            yield archive
            destination.validate()

    def _selection_diagnostic(self, raw_ids: tuple[str, ...]) -> str:
        digest = hashlib.sha256(self.recipe_version.encode())
        for raw_id in raw_ids:
            encoded = raw_id.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    @staticmethod
    def _source_replay_plans(
        reader: PreparedSessionSourceRead,
        logical_keys: tuple[str, ...],
    ) -> dict[str, RevisionReplayPlan]:
        """Capture byte plans on the parent's same original/selected Source view."""
        plans: dict[str, RevisionReplayPlan] = {}
        for logical_key in logical_keys:
            if reader.pending_raw_envelope_has_membership_authority(
                logical_key
            ) or reader.raw_membership_logical_raw_ids(logical_key):
                continue
            plan = reader.raw_revision_replay_plan(logical_key)
            if plan.applications:
                plans[logical_key] = plan
        return plans

    def compute(
        self,
        frame: RawFrame,
        key: str,
        *,
        replay_current: bool = False,
        select_retained_raw_ids: Callable[[PreparedSessionSourceRead], Sequence[str]] | None = None,
        neutral_page: NeutralRawPreparation | None = None,
    ) -> RawObservationReplacement:
        self._compute_adapter.require_current_creator()
        # Even an initially empty discovery holds exclusive byte admission
        # before its witness can hydrate durable reference proof inputs.
        self._compute_adapter.amend_current_input_demand(0)
        from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError

        # A child replayed before its parent stores the shared prefix whole and
        # is normalized again once the parent arrives. When a write claims a
        # parent retained in another component and absent from the Index, the
        # unit widens to that parent and prepares again; in-unit deferral then
        # publishes the parent first. The selection only grows, so this ends.
        widened: tuple[str, ...] = ()
        carry = _PreparationCarry(neutral_page=neutral_page)
        census_first: list[str] = []
        first = True
        retry_attempts = 0
        while True:
            selection = select_retained_raw_ids
            if widened:

                def selection(read: PreparedSessionSourceRead, extra: tuple[str, ...] = widened) -> Sequence[str]:
                    base = (key,) if select_retained_raw_ids is None else tuple(select_retained_raw_ids(read))
                    return (*base, *(raw_id for raw_id in extra if raw_id not in base))

            scope = frame.scope if isinstance(frame.scope, RawObservationScope) else RawObservationScope()
            if first and scope.raw_ids:
                selection = self._census_first_selection(scope.raw_ids, key, selection, census_first)
            first = False
            try:
                replacement = self._compute_once(
                    frame, key, replay_current=replay_current, selection=selection, carry=carry
                )
            except (_CarryInvalidatedError, ReferenceSealStaleError) as error:
                if isinstance(error, ReferenceSealStaleError) and not carry.neutral_artifacts:
                    raise
                retry_attempts += 1
                emit(
                    "storage.raw_observation.preparation_retry",
                    level=WARNING,
                    outcome="degraded",
                    reason=error.reason if isinstance(error, _CarryInvalidatedError) else "reference_seal_stale",
                    error_type=type(error).__name__,
                    phase="source_preparation",
                    productive_id=key,
                    raws=len(scope.raw_ids),
                    attempts=retry_attempts,
                )
                carry = _PreparationCarry(
                    neutral_page=carry.neutral_page,
                    scratch_owner=carry.scratch_owner,
                    neutral_artifacts=carry.neutral_artifacts,
                    committed=carry.committed,
                )
                continue
            try:
                outcome = self._after_preparation(frame, replacement, carry, census_first, widened)
            except BaseException as primary:
                failures: list[BaseException] = []
                for close in (replacement.close, partial(carry.discard_payload, keep=replacement)):
                    try:
                        close()
                    except BaseException as cleanup:
                        failures.append(cleanup)
                if failures:
                    raise BaseExceptionGroup(
                        "retained preparation and cleanup failed", [primary, *failures]
                    ) from primary
                raise
            if isinstance(outcome, RawObservationReplacement):
                return replace(outcome, carried_payload=carry)
            if outcome == "restart":
                carry.discard_payload(keep=replacement)
                carry = _PreparationCarry(committed=carry.committed)
                continue
            if outcome == "widen":
                carry = _PreparationCarry()
                widened = (
                    *widened,
                    *(raw_id for raw_id in replacement.lineage_parent_raw_ids if raw_id not in widened),
                )
                continue
            carry = outcome

    def _after_preparation(
        self,
        frame: RawFrame,
        replacement: RawObservationReplacement,
        carry: _PreparationCarry,
        census_first: list[str],
        widened: tuple[str, ...],
    ) -> RawObservationReplacement | _PreparationCarry | Literal["restart", "widen"]:
        """Decide what one preparation pass leads to within this compute."""
        if census_first:
            # Content decides these envelopes' identity: census all of them
            # before any of them replays, then prepare the seed's own unit.
            census_first.clear()
            committed = tuple(carry.committed)
            if replacement.needs_source_census:
                published = self._publish_phase_in_place(frame, replacement)
                if isinstance(published, BaseException):
                    return replace(replacement, committed_phase_receipts=committed, prepared_phase_failure=published)
                if published is not None:
                    # The next preparation reports these receipts with its own.
                    carry.committed.extend(published)
                    replacement.retire_unpublished()
                    return "restart"
            if replacement.needs_source_census or replacement.needs_source_classification:
                return replace(replacement, committed_phase_receipts=committed) if committed else replacement
            replacement.retire_unpublished()
            return "restart"
        if any(raw_id not in widened for raw_id in replacement.lineage_parent_raw_ids):
            replacement.retire_unpublished()
            carry.discard_payload(keep=replacement)
            return "widen"
        committed = tuple(carry.committed)
        if not (replacement.needs_source_census or replacement.needs_source_classification):
            return replace(replacement, committed_phase_receipts=committed) if committed else replacement
        if any(phase == _replacement_phase(replacement) for phase, _receipt in committed):
            # The phase this preparation committed in place is needed again:
            # it moved nothing the next phase depends on. Publication reports
            # the committed receipts and decides this phase's typed outcome.
            return replace(replacement, committed_phase_receipts=committed)
        continuation = self._continue_after_phase(frame, replacement, carry)
        if continuation is None:
            # A guard refused in-place publication: the existing publish path
            # receives this phase and revalidates it under the writer.
            return replace(replacement, committed_phase_receipts=committed) if committed else replacement
        if isinstance(continuation, BaseException):
            return replace(replacement, committed_phase_receipts=committed, prepared_phase_failure=continuation)
        return continuation

    def _census_first_selection(
        self,
        scope: tuple[str, ...],
        key: str,
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
        census_first: list[str],
    ) -> Callable[[PreparedSessionSourceRead], Sequence[str]]:
        """Add the scope's other uncensused identity-opaque envelopes to an opaque seed's census."""

        def select(read: PreparedSessionSourceRead) -> Sequence[str]:
            base = (key,) if selection is None else tuple(selection(read))
            if not read.uncensused_identity_opaque_raw_ids((key,)):
                return base
            opaque = read.uncensused_identity_opaque_raw_ids(scope)
            extra = tuple(raw_id for raw_id in opaque if raw_id not in base)
            census_first[:] = extra
            return (*base, *extra)

        return select

    def _continue_after_phase(
        self, frame: RawFrame, replacement: RawObservationReplacement, carry: _PreparationCarry
    ) -> _PreparationCarry | BaseException | None:
        """Commit a Source phase in place and keep its seal and artifacts for the next phase.

        Returns the carry to continue with, a refused classification to report,
        or None when publication must take the existing path.
        """
        seal = replacement.reference_seal
        if seal is None:
            return None
        inputs = replacement.prepared_inputs or {}
        before = self._artifact_dependency_digests(seal, inputs, carry.zip_coordinates)
        published = self._publish_phase_in_place(frame, replacement)
        if published is None or isinstance(published, BaseException):
            return published
        carry.committed.extend(published)
        carry.raw_ids = replacement.raw_ids
        after = self._artifact_dependency_digests(seal, inputs, carry.zip_coordinates)
        moved = {artifact_id for artifact_id, digest in before.items() if after.get(artifact_id) != digest}
        # A thread-state cohort completes its graph postimage once per
        # preparation against the Source it read; it is prepared again.
        moved.update(
            id(artifact) for artifact in carry.artifacts.values() if artifact.codex_state_kind == "thread_state"
        )
        if replacement.needs_source_census:
            # The census consumes these carriers' single-use material claims.
            # A later census must prepare fresh claims, even when its parser
            # and enrichment inputs have not changed.
            moved.update(
                id(artifact)
                for artifact in carry.artifacts.values()
                if artifact.codex_state_kind in {"goals", "memories"}
            )
        if moved:
            # The committed tape moved evidence these artifacts were parsed
            # against: this seal prepares them again.
            stale = {key: artifact for key, artifact in carry.artifacts.items() if id(artifact) in moved}
            for key in stale:
                del carry.artifacts[key]
            carry.attachment_refs_published.difference_update(moved)
            _close_prepared_carriers({}, {}, stale.values())
        return carry

    def _artifact_dependency_digests(
        self,
        seal: PreparedIndexMutation,
        inputs: Mapping[str, PreparedRetainedInput],
        zip_coordinates: Mapping[str, CapturedZipMemberCoordinate | None],
    ) -> dict[int, str]:
        """Digest the enrichment and sidecar evidence each prepared artifact depends on.

        This is the writer's own staleness criterion
        (``prepared_enrichment_dependency_state``), read from the seal's
        current committed Source and Index view.
        """
        from polylogue.sources.revision_backfill import enrichment_dependency_digest

        digests: dict[int, str] = {}
        if not inputs:
            return digests
        blob_store = BlobStore(self.archive_root / "blob")
        with seal.original_read_snapshot():
            index = seal.observer("index") if seal.has_tier_capability("index") else None
            for raw_id, item in inputs.items():
                artifact = item.prepared_artifact
                if artifact is None or id(artifact) in digests:
                    continue
                check_compute_cancelled()
                digests[id(artifact)] = enrichment_dependency_digest(
                    provider=item.provider,
                    source_path=item.source_path,
                    captured_zip_coordinate=zip_coordinates.get(raw_id),
                    provider_session_ids=(
                        artifact.iter_provider_session_ids() if artifact.sessions_path is not None else ()
                    ),
                    index_conn=index,
                    source_conn=seal.observer("source"),
                    blob_root=blob_store.root,
                    parser_sidecars=True,
                )
        return digests

    def _publish_phase_in_place(
        self, frame: RawFrame, replacement: RawObservationReplacement
    ) -> _InPlacePhase | BaseException | None:
        """Publish a census or classification on the writer without ending its seal.

        Every input guard of :meth:`publish` runs first; a refusing guard
        returns None and leaves the tape for that path.
        """
        from polylogue.core.stage_admission import admit_stage_write, stage_write_admission_bound
        from polylogue.core.write_lease import current_write_lease
        from polylogue.sources.revision_backfill import RevisionCensusResult
        from polylogue.storage.index_generation import ActiveWriterLease

        seal = replacement.reference_seal
        if seal is None or not (stage_write_admission_bound() or current_write_lease() is not None):
            return None

        def work() -> _InPlacePhase | BaseException | None:
            assert seal is not None
            lease = ActiveWriterLease(self.archive_root)
            lease.acquire()
            try:
                if not self._publication_inputs_current(frame, replacement, before_restoration=True):
                    return None
                if not self._publication_inputs_current(frame, replacement, before_restoration=False):
                    return None
                receipts: list[tuple[Literal["census", "classification"], RevisionCensusResult]] = []
                failures: list[BaseException] = []

                def receive(
                    phase: Literal["census", "classification", "replay"],
                    receipt: RevisionCensusResult | PreparedRevisionReplayResult,
                ) -> None:
                    if phase == "replay" or not isinstance(receipt, RevisionCensusResult):
                        raise RuntimeError("an in-place Source phase reported a replay receipt")
                    receipts.append((phase, receipt))

                if not self._apply_source_phase(
                    replacement, phase_receipt=receive, publication_failure=failures.append
                ):
                    return failures[0]
                return tuple(receipts)
            finally:
                lease.close()

        return admit_stage_write("raw_observation.prepared_source_phase", work)

    def _compute_once(
        self,
        frame: RawFrame,
        key: str,
        *,
        replay_current: bool,
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
        carry: _PreparationCarry,
    ) -> RawObservationReplacement:
        from polylogue.storage.sqlite.reference_seal import (
            IndexMutationDestination,
            PreparedIndexMutation,
            ReferenceSealStaleError,
        )

        first_source_binding = carry.seal is None
        if carry.seal is not None:
            seal = carry.seal
        else:
            index_path = self._index_db_path or ArchiveLocation.resolve(self.archive_root).active_index_path
            destination = (
                None
                if self._owned_generation is None
                else IndexMutationDestination.owned_inactive(self._owned_generation)
            )
            seal = PreparedIndexMutation(
                index_path,
                archive_root=self.archive_root,
                destination=destination,
                input_demand=self._compute_adapter.amend_current_input_demand,
            )
            carry.seal = seal
        replacement: RawObservationReplacement | None = None
        try:
            replacement = self._prepare_blob_restoration(key, selection=selection, seal=seal)
            if replacement is not None:
                replacement = replace(replacement, reference_seal=seal)
                seal.validate_observers_current()
                return replacement
            if first_source_binding:
                seal = self._prepare_neutral_jsonl_then_rebind(
                    key,
                    selection=selection,
                    seal=seal,
                    carry=carry,
                )
            replacement = replace(
                self._compute_prepared(
                    frame,
                    key,
                    replay_current=replay_current,
                    reference_seal=seal,
                    select_retained_raw_ids=selection,
                    carry=carry,
                ),
                reference_seal=seal,
            )
            seal.validate_observers_current()
            return replacement
        except BaseException as primary:
            preserve_neutral = isinstance(primary, (_CarryInvalidatedError, ReferenceSealStaleError)) and bool(
                carry.neutral_artifacts
            )
            if preserve_neutral and replacement is not None:
                # The bound artifact owns the scratch directory, while neutral
                # parser artifacts are the retry cache. Close its SQL owners
                # without letting it remove the shared directory.
                close_replacement = partial(replacement._close_prepared_payload, preserve_scratch=True)
                close_seal = () if replacement.reference_seal is None else (replacement.reference_seal.close,)
                close_replacement_lifetime = (close_replacement, *close_seal)
            else:
                close_replacement_lifetime = (replacement.close if replacement is not None else seal.close,)
            failures: list[BaseException] = []
            for close in (
                *close_replacement_lifetime,
                partial(carry.discard_payload, keep=replacement, preserve_neutral=preserve_neutral),
            ):
                try:
                    close()
                except BaseException as cleanup:
                    failures.append(cleanup)
            if failures:
                raise BaseExceptionGroup("retained preparation and cleanup failed", [primary, *failures]) from primary
            if preserve_neutral:
                carry.seal = None
            raise

    def _prepare_blob_restoration(
        self,
        key: str,
        *,
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
        seal: PreparedIndexMutation,
    ) -> RawObservationReplacement | None:
        """Admit exact retained-byte restoration before any detached capture."""
        from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

        with seal.original_read_snapshot(), seal.source_producer():
            read = PreparedSessionSourceRead(seal, blob_store=BlobStore(self.archive_root / "blob"))
            selected = (key,) if selection is None else tuple(selection(read))
            raw_ids, _logical_keys = read.expand_raw_membership_selection(selected)
            descriptors = {raw_id: read.raw_revision_descriptor(raw_id) for raw_id in raw_ids}
            with self._preparation_archive() as archive:
                restorations = self._stage_absent_blob_restorations(archive, raw_ids, descriptors)
        if restorations is None:
            return None
        return RawObservationReplacement(
            key, self._selection_diagnostic(raw_ids), None, raw_ids, blob_restorations=restorations
        )

    def capture_neutral_raws(
        self,
        raw_ids: Sequence[str],
        *,
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None = None,
    ) -> NeutralRawPreparation | None:
        """Stage a bounded selected page; publish nothing and return no reader."""
        from polylogue.storage.sqlite.reference_seal import IndexMutationDestination, PreparedIndexMutation

        self._compute_adapter.require_current_creator()
        destination = (
            None if self._owned_generation is None else IndexMutationDestination.owned_inactive(self._owned_generation)
        )
        seal = PreparedIndexMutation(
            self._index_db_path or ArchiveLocation.resolve(self.archive_root).active_index_path,
            archive_root=self.archive_root,
            destination=destination,
            input_demand=self._compute_adapter.amend_current_input_demand,
        )
        carry = _PreparationCarry()

        def select(read: PreparedSessionSourceRead) -> Sequence[str]:
            return (*raw_ids, *(selection(read) if selection is not None else ()))

        try:
            captured = self._capture_neutral_jsonl(raw_ids[0], selection=select, seal=seal, carry=carry)
            seal.close()
            return None if captured is None else captured[0]
        except BaseException as primary:
            failures: list[BaseException] = []
            for close in (seal.close, carry.discard_payload):
                try:
                    close()
                except BaseException as cleanup:
                    failures.append(cleanup)
            if failures:
                raise BaseExceptionGroup(
                    "neutral capture and physical retirement failed", [primary, *failures]
                ) from primary
            raise

    def _capture_neutral_jsonl(
        self,
        key: str,
        *,
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
        seal: PreparedIndexMutation,
        carry: _PreparationCarry,
    ) -> tuple[NeutralRawPreparation, tuple[tuple[str, ...], tuple[str, ...]]] | None:
        from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
        from polylogue.sources.sidecar_evidence import (
            RetainedSidecarFile,
            RetainedSidecarScope,
            SiblingTranscript,
            iter_jsonl_records,
        )
        from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

        blob_store = BlobStore(self.archive_root / "blob")
        captures: dict[str, _CapturedNeutralRaw] = {}
        sidecar_scope_by_raw: dict[str, RetainedSidecarScope] = {}
        captured_sidecar_scopes: dict[str, RetainedSidecarScope] = {}
        staged_sidecars: dict[tuple[str, str], Path] = {}
        with seal.original_read_snapshot(), seal.source_producer():
            read = PreparedSessionSourceRead(seal, blob_store=blob_store)
            selected = (key,) if selection is None else tuple(selection(read))
            raw_ids, logical_keys = read.expand_raw_membership_selection(selected)
            if not raw_ids:
                return None
            for raw_id in raw_ids:
                refusal = read.raw_terminal_decode_refusal(raw_id)
                if refusal is not None:
                    raise refusal
            descriptors = {raw_id: read.raw_revision_descriptor(raw_id) for raw_id in raw_ids}
            eligible_raw_ids: tuple[str, ...] = tuple(
                raw_id for raw_id in raw_ids if _neutral_jsonl_candidate(descriptors[raw_id][0], descriptors[raw_id][2])
            )
            if not eligible_raw_ids:
                return None
            operands: dict[str, _NeutralParserOperand] = {}
            for raw_id in eligible_raw_ids:
                operands[raw_id] = _neutral_parser_operand(read, raw_id, descriptor=descriptors[raw_id])

            if carry.scratch_owner is None:
                staging = blob_store._ensure_private_staging_root()
                carry.scratch_owner = tempfile.TemporaryDirectory(prefix=".raw-prepared-", dir=staging)
                from polylogue.sources.prepared_message_sink import discard_decoded_sessions_under

                weakref.finalize(carry.scratch_owner, discard_decoded_sessions_under, Path(carry.scratch_owner.name))
            scratch = Path(carry.scratch_owner.name)

            def capture_sidecar_payload(raw_id: str, blob_hash: str, expected_size: int) -> Path:
                identity = (raw_id, blob_hash)
                existing = staged_sidecars.get(identity)
                if existing is not None:
                    return existing
                directory = Path(tempfile.mkdtemp(prefix="claude-sidecar-", dir=scratch))
                target_path = directory / "payload.bin"
                digest = hashlib.sha256()
                copied = 0
                with (
                    read.open_sidecar_payload(raw_id, bytes.fromhex(blob_hash)) as source,
                    target_path.open("xb") as target,
                ):
                    while chunk := source.read(1024 * 1024):
                        check_compute_cancelled()
                        target.write(chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                if copied != expected_size or digest.hexdigest() != blob_hash:
                    raise RetainedPreparationRetryableError(
                        f"retained Claude sidecar changed while staging raw {raw_id}"
                    )
                staged_sidecars[identity] = target_path
                return target_path

            retained_sidecar_resolver = read.retained_sidecar_resolver()
            for raw_id in eligible_raw_ids:
                descriptor = operands[raw_id].descriptor
                provider, _hash, source_path, _kind, _size = descriptor
                if provider is not Provider.CLAUDE_CODE:
                    continue
                scope = retained_sidecar_resolver.claude_code_scope(source_path)
                sidecar_scope_by_raw[raw_id] = scope
                page = carry.neutral_page
                current_operand = dataclasses.replace(
                    operands[raw_id], sidecar_signature=(scope.scope_key, scope.available, scope.witness)
                )
                if (
                    page is not None
                    and raw_id in page.operands
                    and _neutral_parser_cache_identity(raw_id, page.operands[raw_id])
                    == _neutral_parser_cache_identity(raw_id, current_operand)
                    and page.captures[raw_id].staged_blob.is_file()
                ):
                    captured_sidecar_scopes[Path(source_path).as_posix()] = page.sidecar_scopes[
                        Path(source_path).as_posix()
                    ]
                    continue
                staged_files: list[RetainedSidecarFile] = []
                staged_siblings: list[SiblingTranscript] = []
                for retained_file in scope.files:
                    if retained_file.raw_id is None or retained_file.blob_hash is None:
                        raise RetainedPreparationRetryableError("retained Claude sidecar lacks a durable byte identity")
                    # The resolver already chose the exact durable raw and
                    # size. Read that raw's canonical descriptor under this
                    # same source witness to bind the staged bytes.
                    _provider, file_hash, file_path, _kind, file_size = read.raw_revision_descriptor(
                        retained_file.raw_id
                    )
                    if file_hash != retained_file.blob_hash or file_path != retained_file.source_path:
                        raise RetainedPreparationRetryableError("retained Claude sidecar descriptor changed")
                    staged_path = capture_sidecar_payload(retained_file.raw_id, file_hash, file_size)
                    staged_files.append(
                        RetainedSidecarFile(
                            filename=retained_file.filename,
                            byte_size=retained_file.byte_size,
                            file_mtime_ms=retained_file.file_mtime_ms,
                            read_text=partial(staged_path.read_text, encoding="utf-8", errors="replace"),
                            raw_id=retained_file.raw_id,
                            blob_hash=retained_file.blob_hash,
                            source_path=retained_file.source_path,
                        )
                    )
                for sibling in scope.siblings:
                    staged_records: list[Path] = []
                    for sibling_raw_id, sibling_hash in sibling.record_blobs:
                        _provider, actual_hash, _path, _kind, sibling_size = read.raw_revision_descriptor(
                            sibling_raw_id
                        )
                        if actual_hash != sibling_hash:
                            raise RetainedPreparationRetryableError("retained Claude sibling descriptor changed")
                        staged_records.append(capture_sidecar_payload(sibling_raw_id, actual_hash, sibling_size))

                    def open_records(paths: tuple[Path, ...] = tuple(staged_records)) -> Iterator[object]:
                        for record_path in paths:
                            check_compute_cancelled()
                            with record_path.open("rb") as handle:
                                yield from iter_jsonl_records(lambda: iter(handle))

                    staged_siblings.append(
                        SiblingTranscript(
                            coordinate=sibling.coordinate,
                            open_records=open_records,
                            record_blobs=sibling.record_blobs,
                            selection_witness=sibling.selection_witness,
                        )
                    )
                captured_sidecar_scopes[Path(source_path).as_posix()] = RetainedSidecarScope(
                    scope_key=scope.scope_key,
                    files=tuple(staged_files),
                    siblings=tuple(staged_siblings),
                    available=scope.available,
                    witness=scope.witness,
                )
            for raw_id in eligible_raw_ids:
                maybe_scope = sidecar_scope_by_raw.get(raw_id)
                signature = (
                    None if maybe_scope is None else (maybe_scope.scope_key, maybe_scope.available, maybe_scope.witness)
                )
                operands[raw_id] = dataclasses.replace(operands[raw_id], sidecar_signature=signature)
            for raw_id in eligible_raw_ids:
                operand = operands[raw_id]
                descriptor = operand.descriptor
                profile = operand.profile_identity
                fallback_timestamp = operand.fallback_timestamp
                native_id = operand.native_id
                zip_coordinate = operand.zip_coordinate
                append_logical_key = operand.append_logical_key
                provider, blob_hash, _source_path, _kind, raw_size = descriptor
                input_hash, input_size = seal.retain_original_blob_input(raw_id)
                if input_hash != bytes.fromhex(blob_hash) or input_size != raw_size:
                    raise RetainedPreparationRetryableError(
                        f"retained raw input identity changed before neutral preparation: {raw_id}"
                    )
                page = carry.neutral_page
                if (
                    page is not None
                    and raw_id in page.operands
                    and _neutral_parser_cache_identity(raw_id, page.operands[raw_id])
                    == _neutral_parser_cache_identity(raw_id, operand)
                    and page.captures[raw_id].staged_blob.is_file()
                ):
                    captures[raw_id] = dataclasses.replace(page.captures[raw_id], descriptor=operand.descriptor)
                    continue
                blob_path = read.raw_revision_blob_path(raw_id)
                if blob_path is None:
                    raise RetainedPreparationRetryableError(f"retained JSONL bytes are absent for raw {raw_id}")
                before = self._blob_stat_identity(blob_path)
                neutral_directory = Path(tempfile.mkdtemp(prefix="codex-neutral-", dir=scratch))
                staged_blob = neutral_directory / "input.jsonl"
                digest = hashlib.sha256()
                copied = 0
                with blob_path.open("rb") as source, staged_blob.open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        check_compute_cancelled()
                        target.write(chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                after = self._blob_stat_identity(blob_path)
                if before != after or copied != raw_size or digest.hexdigest() != blob_hash:
                    raise RetainedPreparationRetryableError(f"retained JSONL input changed while staging raw {raw_id}")
                captures[raw_id] = _CapturedNeutralRaw(
                    descriptor=descriptor,
                    profile_identity=profile,
                    fallback_timestamp=fallback_timestamp,
                    native_id=native_id,
                    zip_coordinate=zip_coordinate,
                    append_logical_key=append_logical_key,
                    staged_blob=staged_blob,
                )
            original_selection = (raw_ids, logical_keys)

        # Parsing can be long. Close every capture observer before it begins so
        # unrelated Source commits cannot stale the later publication witness.
        seal.close()
        assert carry.scratch_owner is not None
        return NeutralRawPreparation(
            carry.scratch_owner,
            MappingProxyType(captures),
            MappingProxyType(operands),
            MappingProxyType(captured_sidecar_scopes),
        ), original_selection

    def _prepare_neutral_jsonl_then_rebind(
        self,
        key: str,
        *,
        selection: Callable[[PreparedSessionSourceRead], Sequence[str]] | None,
        seal: PreparedIndexMutation,
        carry: _PreparationCarry,
    ) -> PreparedIndexMutation:
        """Parse selected Codex or Claude Code JSONL raws, then bind afresh.

        The private byte copies are made while the original Source witness is
        current. Parsing and retained-schema validation use only those copies.
        Claude Code's declared sidecars and sibling ownership transcripts are
        staged from their retained CAS rows and supplied through a resolver
        with no ambient filesystem fallback. A new witness proves the same
        selected raws and parser operands before enrichment and publication.
        """
        from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
        from polylogue.storage.sqlite.reference_seal import IndexMutationDestination, PreparedIndexMutation

        captured = self._capture_neutral_jsonl(key, selection=selection, seal=seal, carry=carry)
        if captured is None:
            return seal
        inputs, original_selection = captured
        captures, operands = inputs.captures, inputs.operands
        captured_sidecar_scopes = inputs.sidecar_scopes
        raw_ids, logical_keys = original_selection
        eligible_raw_ids = tuple(captures)
        blob_store = BlobStore(self.archive_root / "blob")
        from polylogue.schemas import validate_retained_document

        assert carry.scratch_owner is not None
        scratch = Path(carry.scratch_owner.name)
        neutral_keys: dict[str, tuple[object, ...]] = {}
        neutral_by_raw: dict[str, PreparedJsonl] = {}

        refreshed_neutral: dict[str, PreparedJsonl] = {}

        def prepare_neutral(raw_id: str) -> PreparedJsonl:
            if raw_id in refreshed_neutral:
                return refreshed_neutral[raw_id]
            cached = carry.neutral_artifacts.get(neutral_keys[raw_id])
            if cached is None and carry.neutral_page is not None:
                cached = carry.neutral_page.artifacts.get(neutral_keys[raw_id])
                if cached is not None:
                    cached = cached.borrow_sealed_files()
                    carry.neutral_artifacts[neutral_keys[raw_id]] = cached
            captured = captures[raw_id]
            provider, blob_hash, source_path, _kind, _raw_size = captured.descriptor
            staged_blob = captured.staged_blob
            neutral_directory = staged_blob.parent
            if cached is None:
                neutral = _parse_captured_neutral(
                    raw_id, captured, neutral_keys[raw_id], captured_sidecar_scopes, scratch
                )
                # Transfer ownership before validation or checkpoint work can fail.
                carry.neutral_artifacts[neutral_keys[raw_id]] = neutral
            else:
                # Parser bytes survive rebinding; current schema evidence does not.
                neutral = cached
            if neutral.error is None and neutral.resolved_provider is not None and neutral.parsed_prefix_size != 0:
                from polylogue.sources.revision_backfill import _retained_validation_input

                prefix = neutral.parsed_prefix_size if self._validation_mode is not ValidationMode.OFF else None
                try:
                    with _retained_validation_input(staged_blob, prefix) as (validation_path, accepted_prefix_size):
                        verdict = validate_retained_document(
                            neutral.resolved_provider,
                            validation_path,
                            mode=self._validation_mode,
                            raw_id=raw_id,
                            revision_sha256=blob_hash,
                            evidence_id=raw_id,
                            source_path=source_path,
                            jsonl=True,
                            accepted_prefix_size=accepted_prefix_size,
                            captured_zip_coordinate=captured.zip_coordinate,
                            registry=self._schema_registry,
                            signature_directory=neutral_directory,
                        )
                except Exception as error:
                    from polylogue.sources.prepared_jsonl import classify_decode_failure

                    decode_failure = classify_decode_failure(error)
                    if decode_failure is None:
                        raise
                    neutral = dataclasses.replace(
                        neutral, error=f"{type(error).__name__}: {error}", decode_failure=decode_failure
                    )
                else:
                    neutral = dataclasses.replace(neutral, validation_verdict=verdict)
                carry.neutral_artifacts[neutral_keys[raw_id]] = neutral
            refreshed_neutral[raw_id] = neutral
            return neutral

        for raw_id in eligible_raw_ids:
            artifact_key = _neutral_artifact_key(
                raw_id,
                operands[raw_id],
                self._validation_mode,
            )
            neutral_keys[raw_id] = artifact_key

        reusable_keys = set(neutral_keys.values())
        obsolete_keys = set(carry.neutral_artifacts) - reusable_keys
        if obsolete_keys:
            obsolete = [carry.neutral_artifacts.pop(artifact_key) for artifact_key in obsolete_keys]
            _close_prepared_carriers({}, {}, obsolete)

        # Preserve the exact three canonical parses and typed one-pass prefix
        # reducer for simple, strictly growing Codex message chains. The helper
        # reads only these private staged copies through CapturedCodexRead.
        groups: dict[str, list[str]] = {}
        for raw_id in eligible_raw_ids:
            descriptor = captures[raw_id].descriptor
            if descriptor[0] is Provider.CODEX and descriptor[3].value in {"full", "unknown"}:
                groups.setdefault(descriptor[2], []).append(raw_id)

        class CapturedCodexRead:
            def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]:
                return captures[raw_id].descriptor

            def raw_profile_identity(self, raw_id: str) -> str | None:
                return captures[raw_id].profile_identity

            def raw_revision_file_mtime(self, raw_id: str) -> str | None:
                return captures[raw_id].fallback_timestamp

            @contextmanager
            def open_raw_revision_material(
                self, raw_id: str
            ) -> Iterator[tuple[Provider, BinaryIO, str, RawRevisionKind]]:
                descriptor = captures[raw_id].descriptor
                staged_blob = captures[raw_id].staged_blob
                with staged_blob.open("rb") as payload:
                    yield descriptor[0], payload, descriptor[2], descriptor[3]

        captured_read = CapturedCodexRead()
        from polylogue.archive.artifact_taxonomy import ArtifactStreamClassification
        from polylogue.sources.prepared_codex_checkpoints import (
            CodexCheckpointArtifactOptions,
            CodexCheckpointDisposition,
            prepare_codex_prefix_checkpoints,
        )

        for group_ids in groups.values():
            group_ids.sort(key=lambda raw_id: captures[raw_id].descriptor[4])
            if len(group_ids) < 4:
                continue
            first_id, second_id, head_id = group_ids[0], group_ids[1], group_ids[-1]
            first, second, head = (prepare_neutral(raw_id) for raw_id in (first_id, second_id, head_id))
            for endpoint_id, endpoint in zip((first_id, second_id, head_id), (first, second, head), strict=True):
                neutral_by_raw[endpoint_id] = endpoint
                carry.neutral_artifacts[neutral_keys[endpoint_id]] = endpoint
            if any(artifact.error is not None or artifact.deferred for artifact in (first, second, head)):
                continue
            head_classification = head.stream_classification()
            if not isinstance(head_classification, ArtifactStreamClassification):
                continue
            checkpoint_options_by_raw: dict[str, CodexCheckpointArtifactOptions] = {}

            def checkpoint_options(
                raw_id: str,
                record_count: int,
                taxonomy: ArtifactStreamClassification = head_classification,
                options_by_raw: dict[str, CodexCheckpointArtifactOptions] = checkpoint_options_by_raw,
            ) -> CodexCheckpointArtifactOptions:
                existing = options_by_raw.get(raw_id)
                if existing is not None:
                    return existing
                captured = captures[raw_id]
                descriptor = captured.descriptor
                profile = captured.profile_identity
                fallback_timestamp = captured.fallback_timestamp
                _provider, _blob_hash, source_path, _kind, raw_size = descriptor
                options = CodexCheckpointArtifactOptions(
                    source_path=source_path,
                    fallback_timestamp=fallback_timestamp,
                    classification=dataclasses.replace(taxonomy, record_count=record_count),
                    parsed_prefix_size=raw_size,
                    captured_profile_key=profile,
                    artifact_directory=Path(tempfile.mkdtemp(prefix="codex-interior-", dir=scratch)),
                )
                options_by_raw[raw_id] = options
                return options

            with prepare_codex_prefix_checkpoints(
                captured_read,
                group_ids,
                head_artifact=head,
                artifact_directory=Path(tempfile.mkdtemp(prefix="codex-validation-", dir=scratch)),
                validation_mode=self._validation_mode,
                publication_publisher=None,
                publication_source_read=None,
                prepare_sessions=lambda _raw_id, sessions: sessions,
                artifact_options=checkpoint_options,
                schema_registry=self._schema_registry,
            ) as checkpoint:
                if checkpoint.disposition is CodexCheckpointDisposition.READY:
                    for checkpoint_raw_id, checkpoint_artifact in checkpoint.iter_artifacts():
                        artifact_key = neutral_keys[checkpoint_raw_id]
                        previous = carry.neutral_artifacts.get(artifact_key)
                        neutral_by_raw[checkpoint_raw_id] = checkpoint_artifact
                        carry.neutral_artifacts[artifact_key] = checkpoint_artifact
                        if previous is not None and previous is not checkpoint_artifact:
                            _close_prepared_carriers({}, {}, (previous,))

        for raw_id in eligible_raw_ids:
            neutral = neutral_by_raw.get(raw_id)
            if neutral is None:
                neutral = prepare_neutral(raw_id)
                neutral_by_raw[raw_id] = neutral
            carry.neutral_artifacts[neutral_keys[raw_id]] = neutral

        destination = (
            None if self._owned_generation is None else IndexMutationDestination.owned_inactive(self._owned_generation)
        )
        fresh = PreparedIndexMutation(
            self._index_db_path or ArchiveLocation.resolve(self.archive_root).active_index_path,
            archive_root=self.archive_root,
            destination=destination,
            input_demand=self._compute_adapter.amend_current_input_demand,
        )
        try:
            with fresh.original_read_snapshot(), fresh.source_producer():
                fresh_read = PreparedSessionSourceRead(fresh, blob_store=blob_store)
                fresh_selected = (key,) if selection is None else tuple(selection(fresh_read))
                fresh_raw_ids, fresh_logical_keys = fresh_read.expand_raw_membership_selection(fresh_selected)
                fresh_descriptors = {raw_id: fresh_read.raw_revision_descriptor(raw_id) for raw_id in fresh_raw_ids}
                fresh_operands: dict[str, _NeutralParserOperand] = {}
                fresh_eligible_raw_ids = tuple(
                    raw_id
                    for raw_id in fresh_raw_ids
                    if _neutral_jsonl_candidate(fresh_descriptors[raw_id][0], fresh_descriptors[raw_id][2])
                )
                if fresh_raw_ids == raw_ids and fresh_eligible_raw_ids == eligible_raw_ids:
                    fresh_sidecar_resolver = fresh_read.retained_sidecar_resolver()
                    for raw_id in fresh_eligible_raw_ids:
                        descriptor = fresh_descriptors[raw_id]
                        fresh_scope_signature: tuple[object, ...] | None = None
                        if descriptor[0] is Provider.CLAUDE_CODE:
                            fresh_scope = fresh_sidecar_resolver.claude_code_scope(descriptor[2])
                            fresh_scope_signature = (
                                fresh_scope.scope_key,
                                fresh_scope.available,
                                fresh_scope.witness,
                            )
                        fresh_operands[raw_id] = _NeutralParserOperand(
                            descriptor=descriptor,
                            profile_identity=fresh_read.raw_profile_identity(raw_id),
                            fallback_timestamp=fresh_read.raw_revision_file_mtime(raw_id),
                            native_id=fresh_read.raw_native_id(raw_id) if descriptor[3].value == "append" else None,
                            zip_coordinate=fresh_read.raw_captured_zip_coordinate(raw_id),
                            append_logical_key=(
                                fresh_read.raw_append_logical_key(raw_id) if descriptor[3].value == "append" else None
                            ),
                            sidecar_signature=fresh_scope_signature,
                        )
                if (fresh_raw_ids, fresh_logical_keys) != original_selection:
                    raise _CarryInvalidatedError("selection_changed")
                if fresh_eligible_raw_ids != eligible_raw_ids or fresh_operands != operands:
                    raise _CarryInvalidatedError("parser_operands_changed")

                from polylogue.core.timestamp_authority import normalize_session_timestamps
                from polylogue.sources.prepared_jsonl import PreparedSessionSequence, _finalize_prepared_cohort
                from polylogue.sources.revision_backfill import iter_enriched_sessions_from_retained_read
                from polylogue.storage.blob_publication import ArchiveBlobPublisher

                for raw_id in eligible_raw_ids:
                    captured_raw = captures[raw_id]
                    descriptor = captured_raw.descriptor
                    fallback_timestamp = captured_raw.fallback_timestamp
                    zip_coordinate = captured_raw.zip_coordinate
                    provider, _blob_hash, source_path, _kind, _raw_size = descriptor
                    neutral = carry.neutral_artifacts[neutral_keys[raw_id]]
                    if neutral.error is not None or neutral.deferred:
                        bound = neutral
                    else:

                        def provider_session_ids(artifact: PreparedJsonl = neutral) -> Iterator[str]:
                            return artifact.session_sequence().iter_provider_session_ids()

                        def finalize(
                            session_sequence: PreparedSessionSequence,
                            *,
                            retained_raw_id: str = raw_id,
                            retained_provider: Provider = provider,
                            retained_path: str = source_path,
                            retained_fallback: str | None = fallback_timestamp,
                            retained_zip: CapturedZipMemberCoordinate | None = zip_coordinate,
                            retained_provider_ids: Callable[[], Iterator[str]] = partial(provider_session_ids, neutral),
                        ) -> Iterable[ParsedSession]:
                            return iter_enriched_sessions_from_retained_read(
                                evidence_reader=fresh_read,
                                provider=retained_provider,
                                source_path=retained_path,
                                sessions=session_sequence,
                                captured_zip_coordinate=retained_zip,
                                provider_session_ids=retained_provider_ids(),
                                normalize_session=lambda session: normalize_session_timestamps(
                                    session, fallback_timestamp=retained_fallback
                                ),
                            )

                        bound = _finalize_prepared_cohort(
                            neutral,
                            finalize,
                            artifact_directory=Path(tempfile.mkdtemp(prefix="codex-bound-", dir=scratch)),
                            publication_publisher=ArchiveBlobPublisher(
                                self.archive_root / "source.db", blob_store.root
                            ),
                            publication_source_read=fresh_read,
                            preparation_dependency=None,
                            preserve_parser_stage=False,
                        )
                        bound = dataclasses.replace(bound, validation_verdict=neutral.validation_verdict)
                    carry.artifacts[neutral_keys[raw_id]] = bound
                carry.raw_ids = raw_ids
                carry.seal = fresh
                return fresh
        except BaseException:
            fresh.close()
            raise

    def _compute_prepared(
        self,
        frame: RawFrame,
        key: str,
        *,
        replay_current: bool,
        reference_seal: PreparedIndexMutation,
        select_retained_raw_ids: Callable[[PreparedSessionSourceRead], Sequence[str]] | None = None,
        carry: _PreparationCarry | None = None,
    ) -> RawObservationReplacement:
        carry = _PreparationCarry(seal=reference_seal) if carry is None else carry
        from polylogue.core.prepared_file import VerificationCancelledError
        from polylogue.sources.dispatch import is_jsonl_source_path
        from polylogue.sources.revision_backfill import (
            PreparedRetainedInput,
            RetainedPreparationRetryableError,
            prepare_retained_jsonl_artifact,
        )
        from polylogue.sources.sqlite_export import looks_like_logical_source_path

        # One component replay settles every member. The kernel classified the
        # page before it began publishing, so a sibling can still arrive here
        # with that old ``stale`` verdict after an earlier member has made the
        # shared authoritative output current. Re-inspect at the compute
        # boundary before parsing retained bytes again. The publication method
        # repeats this check, so a later race remains pending rather than being
        # certified from this observation alone.

        with self._preparation_archive() as archive:
            from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

            with (
                reference_seal.original_read_snapshot(
                    input_demand=self._compute_adapter.amend_current_input_demand,
                    prepaid_blob_inputs=self._prepaid_blob_inputs,
                ),
                reference_seal.source_producer(),
            ):
                selection_read = PreparedSessionSourceRead(
                    reference_seal,
                    blob_store=BlobStore(self.archive_root / "blob"),
                )
                selected = (key,) if select_retained_raw_ids is None else tuple(select_retained_raw_ids(selection_read))
                raw_ids, logical_keys = selection_read.expand_raw_membership_selection(selected)
                if carry.raw_ids and carry.raw_ids != raw_ids:
                    # The committed phase changed this unit's membership; its
                    # carried artifacts describe another unit.
                    raise _CarryInvalidatedError("carried_membership_changed")
                for selected_raw_id in raw_ids:
                    refusal = selection_read.raw_terminal_decode_refusal(selected_raw_id)
                    if refusal is not None:
                        raise refusal
                descriptors = {raw_id: selection_read.raw_revision_descriptor(raw_id) for raw_id in raw_ids}
                neutral_artifact_keys: dict[str, tuple[object, ...]] = {}
                neutral_operands: dict[str, _NeutralParserOperand] = {}
                neutral_raw_ids = tuple(
                    raw_id
                    for raw_id in raw_ids
                    if _neutral_jsonl_candidate(descriptors[raw_id][0], descriptors[raw_id][2])
                )
                if neutral_raw_ids:
                    neutral_operands = {
                        raw_id: _neutral_parser_operand(selection_read, raw_id) for raw_id in neutral_raw_ids
                    }
                    neutral_artifact_keys = {
                        raw_id: _neutral_artifact_key(raw_id, neutral_operands[raw_id], self._validation_mode)
                        for raw_id in neutral_raw_ids
                    }
            if (
                select_retained_raw_ids is None
                and not replay_current
                and self.inspect(frame, (key,)).get(key) == "valid"
            ):
                return RawObservationReplacement(
                    key,
                    "",
                    None,
                    (),
                    already_valid=True,
                )
            binding = self._selection_diagnostic(raw_ids)
            process_prepared = bool(descriptors)
            if process_prepared:
                from polylogue.sources.prepared_merge import (
                    prepare_retained_cohort_artifact,
                    prepared_cohort_source_hash,
                )
                from polylogue.sources.revision_backfill import (
                    PreparedRetainedAggregate,
                    prepare_membership_replay,
                    prepare_retained_replay_source,
                )
                from polylogue.storage.sqlite.archive_tiers.revision_governance import (
                    membership_head_input_from_revision_head,
                    prepare_membership_head_plan,
                    prepare_membership_head_plan_from_inputs,
                    prepared_parser_census_is_current,
                )
                from polylogue.storage.sqlite.archive_tiers.write import (
                    prepare_session_write,
                    prepared_session_rows_from_shard,
                )

                with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                    census_read = PreparedSessionSourceRead(
                        reference_seal,
                        blob_store=BlobStore(self.archive_root / "blob"),
                    )
                    from polylogue.sources.origin_specs import path_declaration_refuses_session

                    complete_census: set[str] = set()
                    for raw_id in raw_ids:
                        if not prepared_parser_census_is_current(reference_seal, raw_id):
                            continue
                        provider, _blob_hash, source_path, _kind, _raw_size = descriptors[raw_id]
                        declared_non_session = path_declaration_refuses_session(provider, source_path)
                        # A raw-only declaration needs its independent current
                        # membership receipt even when taxonomy excludes schema
                        # validation. Native session grammars remain exempt.
                        if declared_non_session and not census_read.raw_parser_confirmed_non_session(raw_id):
                            continue
                        validation_mode = census_read.raw_validation_mode(raw_id)
                        if validation_mode == self._validation_mode.value:
                            complete_census.add(raw_id)
                            continue
                        if validation_mode is not None:
                            continue
                        schema_eligible = census_read.raw_schema_eligible(raw_id)
                        admitted_no_records = False
                        if is_jsonl_source_path(source_path) and census_read.raw_parser_confirmed_non_session(raw_id):
                            from polylogue.sources.live.batch_support import jsonl_parse_prefix_size_of_handle

                            with census_read.open_raw_revision_material(raw_id) as (_, census_material, _, _):
                                admitted_no_records = jsonl_parse_prefix_size_of_handle(census_material) == 0
                        if not schema_eligible or declared_non_session or admitted_no_records:
                            complete_census.add(raw_id)
                # Every retained raw needs its actual parser authority before
                # replay can select a session. A singleton can still refine an
                # opaque acquisition identity or prove a non-session artifact.
                needs_source_census = len(complete_census) != len(raw_ids)
                prepared_source_classification = None
                if not needs_source_census:
                    from polylogue.sources.revision_backfill import prepare_revision_source_classification

                    prepared_source_classification = prepare_revision_source_classification(
                        reference_seal,
                        selection_read,
                        selected_raw_ids=raw_ids,
                        logical_keys=logical_keys,
                        payload_store=BlobStore(self.archive_root / "blob"),
                    )
                    if prepared_source_classification is not None:
                        return RawObservationReplacement(
                            key,
                            binding,
                            None,
                            raw_ids,
                            prepared_source_classification=prepared_source_classification,
                            needs_source_classification=True,
                        )
                with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                    prepared_revision_plans = self._source_replay_plans(
                        selection_read, selection_read.raw_revision_rebuild_logical_keys(raw_ids)
                    )
                    planned_accepted_raw_ids = {
                        key: plan.accepted_raw_ids for key, plan in prepared_revision_plans.items()
                    }
                material_store = BlobStore(self.archive_root / "blob")
                if carry.scratch_owner is not None:
                    scratch_owner = carry.scratch_owner
                else:
                    material_staging = material_store._ensure_private_staging_root()
                    scratch_owner = tempfile.TemporaryDirectory(prefix=".raw-prepared-", dir=material_staging)
                    # Whatever path removes the scratch tree (publication, an
                    # exception, or the directory's own finalizer when publication
                    # is bypassed), the decodes cached from it go with it.
                    from polylogue.sources.prepared_message_sink import discard_decoded_sessions_under

                    weakref.finalize(scratch_owner, discard_decoded_sessions_under, Path(scratch_owner.name))
                    carry.scratch_owner = scratch_owner
                scratch = Path(scratch_owner.name)
                with retain_native_sql_lifetimes(scratch_owner):
                    prepared: dict[str, PreparedRetainedInput] = {}
                    aggregates: dict[str, PreparedRetainedAggregate] = {}
                    prepared_writes: dict[tuple[str, str], PreparedSessionWrite] = {}
                    prepared_replay_adoption: dict[tuple[str, tuple[str, ...]], PreparedRevisionAdoption] = {}
                    membership_plans: dict[str, PreparedMembershipReplay] = {}
                    prepared_byte_outcomes: dict[str, PreparedRevisionReplayOutcome] = {}
                    prepared_lineage_deferrals: tuple[str, ...] = ()
                    lineage_deferred_raw_ids: set[str] = set()
                    prepared_replay_source: PreparedRetainedReplaySource | None = None
                    verified_blob_stats: dict[str, tuple[int, int, int, int, int]] = {}
                    prepared_source_census: PreparedRevisionSourceCensus | None = None
                    prepared_replay_schedule: ReplaySchedule | None = None
                    prepared_logical_keys: tuple[str, ...] = ()
                    prepared_membership_keys: tuple[str, ...] = ()
                    prepared_byte_logical_keys: tuple[str, ...] = ()
                    prepared_key_refusals: dict[str, CohortMembershipRefusalError] = {}
                    # A continued preparation keeps the artifacts parsed under
                    # this seal; each is reused only under its unchanged key.
                    prepared_artifacts = carry.artifacts
                    provider_parse_seconds = 0.0
                    try:
                        # A Codex JSONL revision chain can share one proven
                        # plain-message prefix. Prepare the three canonical
                        # witnesses independently, then derive only its
                        # interior artifacts from the typed per-prefix reducer.
                        from polylogue.archive.artifact_taxonomy import ArtifactStreamClassification
                        from polylogue.archive.revision_authority import RawRevisionKind
                        from polylogue.core.timestamp_authority import normalize_session_timestamps
                        from polylogue.sources.prepared_codex_checkpoints import (
                            CodexCheckpointArtifactOptions,
                            CodexCheckpointDisposition,
                            prepare_codex_prefix_checkpoints,
                        )

                        codex_groups: dict[str, list[str]] = {}
                        for candidate_raw_id in raw_ids:
                            candidate_provider, _hash, candidate_path, candidate_kind, _size = descriptors[
                                candidate_raw_id
                            ]
                            if (
                                candidate_raw_id in neutral_operands
                                and candidate_provider is Provider.CODEX
                                and candidate_kind in {RawRevisionKind.FULL, RawRevisionKind.UNKNOWN}
                                and is_jsonl_source_path(candidate_path)
                                and not looks_like_logical_source_path(Path(candidate_path))
                            ):
                                codex_groups.setdefault(candidate_path, []).append(candidate_raw_id)

                        def checkpoint_key(raw_id: str) -> tuple[object, ...]:
                            neutral_key = neutral_artifact_keys.get(raw_id)
                            if neutral_key is not None:
                                return neutral_key
                            provider, blob_hash, path, kind, _size = descriptors[raw_id]
                            with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                native_id = read.raw_native_id(raw_id) if kind.value == "append" else None
                                fallback = read.raw_revision_file_mtime(raw_id)
                                profile = read.raw_profile_identity(raw_id)
                                carry.zip_coordinates[raw_id] = read.raw_captured_zip_coordinate(raw_id)
                            return (
                                raw_id,
                                provider,
                                blob_hash,
                                path,
                                kind.value == "append",
                                native_id,
                                fallback,
                                profile,
                                self._validation_mode,
                            )

                        def prepare_checkpoint_endpoint(raw_id: str) -> PreparedJsonl:
                            nonlocal provider_parse_seconds
                            _provider, blob_hash, path, _kind, raw_size = descriptors[raw_id]
                            with reference_seal.original_read_snapshot():
                                input_hash, input_size = reference_seal.retain_original_blob_input(raw_id)
                                if input_hash != bytes.fromhex(blob_hash) or input_size != raw_size:
                                    raise RetainedPreparationRetryableError(
                                        f"retained checkpoint input identity changed for raw {raw_id}"
                                    )
                            endpoint_blob_store = BlobStore(self.archive_root / "blob")
                            if not endpoint_blob_store.verify(blob_hash, stop=compute_cancel_requested):
                                raise RetainedPreparationRetryableError(
                                    f"retained checkpoint blob changed for raw {raw_id}"
                                )
                            artifact_key = checkpoint_key(raw_id)
                            artifact = prepared_artifacts.get(artifact_key)
                            if artifact is None:
                                with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                    read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                    parse_started = time.perf_counter()
                                    artifact = prepare_retained_jsonl_artifact(
                                        read,
                                        raw_id,
                                        directory=Path(tempfile.mkdtemp(prefix="codex-endpoint-", dir=scratch)),
                                        validation_mode=self._validation_mode,
                                        schema_registry=self._schema_registry,
                                    )
                                    provider_parse_seconds += time.perf_counter() - parse_started
                                prepared_artifacts[artifact_key] = artifact
                            if artifact.error is None and artifact.blob_hash != blob_hash:
                                raise RetainedPreparationRetryableError(
                                    f"retained checkpoint endpoint hash changed for raw {raw_id}"
                                )
                            if artifact.error is None:
                                artifact.verify_files(full=True, stop=compute_cancel_requested)
                            return artifact

                        for candidate_ids in codex_groups.values():
                            if len(candidate_ids) < 4:
                                continue
                            candidate_ids.sort(key=lambda raw_id: descriptors[raw_id][4])
                            try:
                                first = prepare_checkpoint_endpoint(candidate_ids[0])
                                smallest_match = prepare_checkpoint_endpoint(candidate_ids[1])
                                head = prepare_checkpoint_endpoint(candidate_ids[-1])
                                if (
                                    first.error is not None
                                    or smallest_match.error is not None
                                    or head.error is not None
                                ):
                                    continue
                                head_classification = head.stream_classification()
                                if not isinstance(head_classification, ArtifactStreamClassification):
                                    continue

                                checkpoint_keys: dict[str, tuple[object, ...]] = {
                                    candidate_raw_id: checkpoint_key(candidate_raw_id)
                                    for candidate_raw_id in candidate_ids
                                }
                                if all(
                                    prepared_artifacts.get(checkpoint_keys[candidate_raw_id]) is not None
                                    for candidate_raw_id in candidate_ids
                                ):
                                    # A prior Source phase already proved this
                                    # exact cohort and cached every per-raw
                                    # verdict. _continue_after_phase retains
                                    # those artifacts only while their source
                                    # dependencies remain current.
                                    continue
                                checkpoint_options_by_raw: dict[str, CodexCheckpointArtifactOptions] = {}

                                def checkpoint_options(
                                    raw_id: str,
                                    record_count: int,
                                    head_taxonomy: ArtifactStreamClassification = head_classification,
                                    options_by_raw: dict[
                                        str, CodexCheckpointArtifactOptions
                                    ] = checkpoint_options_by_raw,
                                ) -> CodexCheckpointArtifactOptions:
                                    existing = options_by_raw.get(raw_id)
                                    if existing is not None:
                                        return existing
                                    _provider, _blob_hash, _path, _kind, raw_size = descriptors[raw_id]
                                    operand = neutral_operands.get(raw_id)
                                    if operand is None:
                                        with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                            read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                            operand = _neutral_parser_operand(read, raw_id)
                                    profile = operand.profile_identity
                                    fallback_timestamp = operand.fallback_timestamp
                                    if profile is not None and not isinstance(profile, str):
                                        raise TypeError("checkpoint profile identity must be text or absent")
                                    if fallback_timestamp is not None and not isinstance(fallback_timestamp, str):
                                        raise TypeError("checkpoint fallback timestamp must be text or absent")
                                    options = CodexCheckpointArtifactOptions(
                                        classification=dataclasses.replace(head_taxonomy, record_count=record_count),
                                        parsed_prefix_size=raw_size,
                                        captured_profile_key=profile,
                                        artifact_directory=Path(
                                            tempfile.mkdtemp(prefix="codex-interior-", dir=scratch)
                                        ),
                                        source_path=_path,
                                        fallback_timestamp=fallback_timestamp,
                                    )
                                    options_by_raw[raw_id] = options
                                    return options

                                def checkpoint_sessions(
                                    raw_id: str, sessions: Iterable[ParsedSession]
                                ) -> Iterable[ParsedSession]:
                                    _provider, _blob_hash, source_path, _kind, _size = descriptors[raw_id]
                                    with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                        read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                        captured_zip = read.raw_captured_zip_coordinate(raw_id)
                                        fallback = read.raw_revision_file_mtime(raw_id)
                                        from polylogue.sources.revision_backfill import (
                                            iter_enriched_sessions_from_retained_read,
                                        )

                                        session_rows = tuple(sessions)
                                        return tuple(
                                            iter_enriched_sessions_from_retained_read(
                                                read,
                                                Provider.CODEX,
                                                source_path,
                                                session_rows,
                                                captured_zip_coordinate=captured_zip,
                                                provider_session_ids=tuple(
                                                    session.provider_session_id for session in session_rows
                                                ),
                                                normalize_session=lambda session: normalize_session_timestamps(
                                                    session, fallback_timestamp=fallback
                                                ),
                                            )
                                        )

                                with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                    retained_read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                    checkpoint = prepare_codex_prefix_checkpoints(
                                        retained_read,
                                        candidate_ids,
                                        head_artifact=head,
                                        artifact_directory=Path(
                                            tempfile.mkdtemp(prefix="codex-validation-", dir=scratch)
                                        ),
                                        validation_mode=self._validation_mode,
                                        publication_publisher=None,
                                        publication_source_read=None,
                                        prepare_sessions=checkpoint_sessions,
                                        artifact_options=checkpoint_options,
                                        schema_registry=self._schema_registry,
                                    )
                                with checkpoint:
                                    if checkpoint.disposition is not CodexCheckpointDisposition.READY:
                                        continue
                                    for checkpoint_raw_id, checkpoint_artifact in checkpoint.iter_artifacts():
                                        prepared_artifacts[checkpoint_keys[checkpoint_raw_id]] = checkpoint_artifact
                            except (DaemonOperationCancelled, DaemonBackpressureError):
                                raise
                            except (OSError, ValueError, RetainedPreparationRetryableError):
                                # An unproved chain uses the ordinary complete
                                # parser path below; no partial checkpoint is
                                # allowed to become source evidence.
                                continue

                        for raw_id in raw_ids:
                            provider, blob_hash, path, kind, size = descriptors[raw_id]
                            blob_store = BlobStore(self.archive_root / "blob")
                            # Verification consumes the same canonical CAS
                            # input as parsing, so enroll it before the first
                            # byte read rather than charging after verification.
                            with reference_seal.original_read_snapshot():
                                input_hash, input_size = reference_seal.retain_original_blob_input(raw_id)
                                if input_hash != bytes.fromhex(blob_hash) or input_size != size:
                                    raise RetainedPreparationRetryableError(
                                        f"retained raw input identity changed before verification: {raw_id}"
                                    )
                            blob_path = blob_store.blob_path(blob_hash)
                            try:
                                before = self._blob_stat_identity(blob_path)
                            except OSError as exc:
                                raise RetainedPreparationRetryableError(
                                    f"retained raw blob disappeared: {raw_id}"
                                ) from exc
                            if not blob_store.verify(blob_hash, stop=compute_cancel_requested):
                                raise RetainedPreparationRetryableError(f"retained raw blob changed: {raw_id}")

                            with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                retained_read = PreparedSessionSourceRead(reference_seal, blob_store=blob_store)
                                native_id = retained_read.raw_native_id(raw_id) if kind.value == "append" else None
                                fallback_timestamp = retained_read.raw_revision_file_mtime(raw_id)
                                profile_identity = retained_read.raw_profile_identity(raw_id)
                                append_logical_key = (
                                    retained_read.raw_append_logical_key(raw_id) if kind.value == "append" else None
                                )
                                carry.zip_coordinates[raw_id] = retained_read.raw_captured_zip_coordinate(raw_id)
                                sidecar_signature = None
                                if provider is Provider.CLAUDE_CODE:
                                    scope = retained_read.retained_sidecar_resolver().claude_code_scope(path)
                                    sidecar_signature = (scope.scope_key, scope.available, scope.witness)
                            # Everything a worker reads to parse and enrich these
                            # bytes. Parsing distinguishes only an append revision,
                            # so a census that types an unknown revision as full
                            # keeps the artifact.
                            artifact_key: tuple[object, ...] = (
                                raw_id,
                                provider,
                                blob_hash,
                                path,
                                kind.value == "append",
                                native_id,
                                fallback_timestamp,
                                profile_identity,
                                self._validation_mode,
                            )
                            if provider is Provider.CLAUDE_CODE:
                                artifact_key = (*artifact_key, sidecar_signature)
                            # Ordinary JSON carriers can share equivalent parser
                            # inputs. Codex state owns a per-raw mutable projection,
                            # and UNKNOWN can resolve to that route only after decode.
                            share_json = (
                                (is_jsonl_source_path(path) or Path(path).suffix.lower() == ".json")
                                and not looks_like_logical_source_path(blob_path)
                                and provider not in {Provider.CODEX, Provider.UNKNOWN}
                            )
                            if share_json:
                                from polylogue.sources.fallback_identity import fallback_session_id

                                operand = _NeutralParserOperand(
                                    descriptor=descriptors[raw_id],
                                    profile_identity=profile_identity,
                                    fallback_timestamp=fallback_timestamp,
                                    native_id=native_id,
                                    zip_coordinate=carry.zip_coordinates[raw_id],
                                    append_logical_key=append_logical_key,
                                    sidecar_signature=sidecar_signature,
                                )
                                artifact_key = (
                                    ("fallback-source-id", fallback_session_id(path, raw_id)),
                                    *_neutral_parser_cache_identity(raw_id, operand)[1:],
                                    ("validation-mode", self._validation_mode.value),
                                )
                            if raw_id in neutral_artifact_keys:
                                artifact_key = neutral_artifact_keys[raw_id]
                            artifact = prepared_artifacts.get(artifact_key)
                            reused_artifact = artifact is not None
                            if artifact is None:
                                try:
                                    is_json = (
                                        is_jsonl_source_path(path) or Path(path).suffix.lower() == ".json"
                                    ) and not looks_like_logical_source_path(blob_path)
                                    worker = (
                                        prepare_retained_jsonl_artifact if is_json else self._prepare_non_json_artifact
                                    )
                                    if worker is None:
                                        raise RetainedPreparationRetryableError(
                                            "non-JSON retained preparation requires an operations worker"
                                        )
                                    # This carrier stays under the already admitted Raw
                                    # reservation through publication and physical close.
                                    check_compute_cancelled()

                                    with (
                                        reference_seal.original_read_snapshot(),
                                        reference_seal.source_producer(),
                                    ):
                                        retained_read = PreparedSessionSourceRead(
                                            reference_seal,
                                            blob_store=blob_store,
                                        )
                                        parse_started = time.perf_counter()
                                        # Each artifact owns its directory, so a stale
                                        # one is discarded without its siblings.
                                        artifact = worker(
                                            retained_read,
                                            raw_id,
                                            directory=Path(tempfile.mkdtemp(prefix="artifact-", dir=scratch)),
                                            validation_mode=self._validation_mode,
                                            schema_registry=self._schema_registry,
                                        )
                                        provider_parse_seconds += time.perf_counter() - parse_started
                                    # Own the returned files before cancellation or
                                    # seal verification can refuse their publication.
                                    prepared_artifacts[artifact_key] = artifact
                                    check_compute_cancelled()
                                except DaemonOperationCancelled:
                                    raise
                                except DaemonBackpressureError as exc:
                                    raise RetainedPreparationRetryableError(
                                        f"retained compute unavailable while preparing raw {raw_id}"
                                    ) from exc
                            if not blob_store.verify(blob_hash, stop=compute_cancel_requested):
                                raise RetainedPreparationRetryableError(f"retained raw blob changed: {raw_id}")
                            try:
                                after = self._blob_stat_identity(blob_path)
                            except OSError as exc:
                                raise RetainedPreparationRetryableError(
                                    f"retained raw blob disappeared: {raw_id}"
                                ) from exc
                            if before != after:
                                raise RetainedPreparationRetryableError(f"retained raw blob changed: {raw_id}")
                            verified_blob_stats[raw_id] = after
                            if artifact.deferred:
                                raise RetainedPreparationRetryableError(
                                    f"retained preparation deferred for raw {raw_id}: {artifact.error}"
                                )
                            if artifact.error is None and artifact.blob_hash != blob_hash:
                                raise RetainedPreparationRetryableError(
                                    f"retained preparation hash changed for raw {raw_id}"
                                )
                            if artifact.error is None:
                                try:
                                    artifact.verify_files(full=not reused_artifact, stop=compute_cancel_requested)
                                except (OSError, ValueError) as exc:
                                    raise RetainedPreparationRetryableError(
                                        f"retained preparation seal changed for raw {raw_id}"
                                    ) from exc
                            # One cache entry owns the physical carrier. Each raw
                            # still owes current schema evidence at its own coordinate.
                            verdict = artifact.validation_verdict
                            if reused_artifact and artifact.validation_verdict is not None and share_json:
                                from polylogue.schemas import validate_retained_document
                                from polylogue.sources.revision_backfill import _retained_validation_input

                                prefix = (
                                    artifact.parsed_prefix_size
                                    if is_jsonl_source_path(path) and self._validation_mode is not ValidationMode.OFF
                                    else None
                                )
                                with _retained_validation_input(blob_path, prefix) as (
                                    validation_path,
                                    accepted_prefix_size,
                                ):
                                    verdict = validate_retained_document(
                                        artifact.resolved_provider or provider,
                                        validation_path,
                                        mode=self._validation_mode,
                                        raw_id=raw_id,
                                        revision_sha256=blob_hash,
                                        evidence_id=raw_id,
                                        source_path=path,
                                        jsonl=is_jsonl_source_path(path),
                                        accepted_prefix_size=accepted_prefix_size,
                                        captured_zip_coordinate=carry.zip_coordinates[raw_id],
                                        registry=self._schema_registry,
                                        signature_directory=scratch,
                                    )
                            else:
                                prepared_artifacts[artifact_key] = artifact
                            prepared[raw_id] = PreparedRetainedInput(
                                raw_id,
                                provider,
                                blob_hash,
                                path,
                                kind,
                                size,
                                native_id,
                                raw_authority_parser_fingerprint(),
                                fallback_timestamp,
                                verified_blob_stat=after,
                                validation_verdict=verdict,
                                parser_error=artifact.error,
                                parser_decode_failure=artifact.decode_failure,
                                missing_profile_identity=artifact.missing_profile_identity,
                                retained_zip_membership_unproved=artifact.retained_zip_membership_unproved,
                                unsupported_shape=artifact.unsupported_shape,
                                captured_profile_key=profile_identity,
                                prepared_artifact=artifact if artifact.error is None else None,
                            )
                        # Each publisher commits/exposes its first Source unit on
                        # this creator before the later census/session tape exists.
                        for artifact in prepared_artifacts.values():
                            if artifact.error is None:
                                artifact.publish_blobs(reference_seal=reference_seal)
                        # A claim is consumed once: a continued preparation
                        # publishes only the artifacts it has not yet published.
                        published_refs = carry.attachment_refs_published
                        _publish_acquired_attachment_refs(
                            reference_seal,
                            {
                                raw_id: item
                                for raw_id, item in prepared.items()
                                if item.prepared_artifact is None or id(item.prepared_artifact) not in published_refs
                            },
                            blob_store=material_store,
                        )
                        published_refs.update(
                            id(item.prepared_artifact)
                            for item in prepared.values()
                            if item.prepared_artifact is not None
                        )
                        from polylogue.sources.prepared_jsonl import complete_thread_projection_cohort

                        with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                            state_read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                            for retained_raw_id, retained_input in prepared.items():
                                state_artifact = retained_input.prepared_artifact
                                if state_artifact is None or state_artifact.codex_state_kind != "thread_state":
                                    continue
                                state_observed_at, state_order = state_read.raw_revision_observation_order(
                                    retained_raw_id
                                )
                                state_artifact.prepare_thread_projection(
                                    reference_seal,
                                    source_read=state_read,
                                    raw_id=retained_raw_id,
                                    blob_hash=retained_input.blob_hash,
                                    observed_at_ms=state_observed_at,
                                    observation_order=state_order,
                                    source_path=retained_input.source_path,
                                )
                            complete_thread_projection_cohort(
                                (
                                    item.prepared_artifact
                                    for item in prepared.values()
                                    if item.prepared_artifact is not None
                                ),
                                reference_seal,
                                source_read=state_read,
                            )
                        if not needs_source_census:
                            from polylogue.sources.revision_backfill import (
                                _prepared_retained_outcome,
                                prepared_session_for_revision_key,
                            )

                            with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                member_read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                for logical_key, accepted_raw_ids in planned_accepted_raw_ids.items():
                                    if logical_key.startswith(PENDING_RAW_LOGICAL_SOURCE_PREFIX):
                                        continue
                                    for member_raw_id in accepted_raw_ids:
                                        try:
                                            original_output = _prepared_retained_outcome(
                                                member_read, member_raw_id, prepared, stop=compute_cancel_requested
                                            )
                                            if isinstance(original_output, Exception):
                                                raise CohortMembershipRefusalError(
                                                    logical_key,
                                                    member_raw_id,
                                                    f"selector member did not parse: {original_output}",
                                                ) from original_output
                                            prepared_session_for_revision_key(
                                                original_output[0], raw_id=member_raw_id, logical_source_key=logical_key
                                            )
                                        except CohortMembershipRefusalError as refusal:
                                            prepared_key_refusals[logical_key] = refusal
                                            break
                        # Schema STRICT refusal is independent of parsing and
                        # census authority. Keep the actual parsed membership,
                        # but do not derive an Index replacement from a chain
                        # whose accepted revision failed its validation policy.
                        for logical_key, accepted_raw_ids in planned_accepted_raw_ids.items():
                            if logical_key in prepared_key_refusals:
                                continue
                            refused_raw_id = next(
                                (
                                    raw_id
                                    for raw_id in accepted_raw_ids
                                    if (item := prepared.get(raw_id)) is not None
                                    and item.prepared_artifact is not None
                                    and item.validation_verdict is not None
                                    and item.validation_verdict.strict_refusal
                                ),
                                None,
                            )
                            if refused_raw_id is not None:
                                refused_input = prepared.get(refused_raw_id)
                                if refused_input is None or refused_input.prepared_artifact is None:
                                    raise AssertionError("strictly refused raw lost its prepared artifact")
                                verdict = refused_input.validation_verdict
                                if verdict is None:
                                    raise AssertionError("strictly refused raw lost its validation verdict")
                                detail = (
                                    verdict.first_diagnostic
                                    if verdict.first_diagnostic
                                    else "strict schema validation refused the retained revision"
                                )
                                prepared_key_refusals[logical_key] = CohortMembershipRefusalError(
                                    logical_key,
                                    refused_raw_id,
                                    f"retained schema validation refused revision: {detail}",
                                )
                        for logical_key, accepted_raw_ids in (
                            planned_accepted_raw_ids.items() if not needs_source_census else ()
                        ):
                            if logical_key in prepared_key_refusals or len(accepted_raw_ids) < 2:
                                continue
                            ordered: list[tuple[str, PreparedJsonl]] = []
                            for raw_id in accepted_raw_ids:
                                ordered_input = prepared.get(raw_id)
                                retained_artifact = (
                                    ordered_input.prepared_artifact if ordered_input is not None else None
                                )
                                if retained_artifact is not None:
                                    ordered.append((raw_id, retained_artifact))
                            if len(ordered) != len(accepted_raw_ids):
                                continue
                            try:
                                check_compute_cancelled()
                                aggregate = prepare_retained_cohort_artifact(ordered, scratch)
                                check_compute_cancelled()
                            except DaemonOperationCancelled:
                                raise
                            except DaemonBackpressureError as exc:
                                raise RetainedPreparationRetryableError(
                                    f"retained cohort compute unavailable while preparing {logical_key}"
                                ) from exc
                            if aggregate.deferred or aggregate.error is not None:
                                raise RetainedPreparationRetryableError(
                                    f"retained cohort preparation deferred for {logical_key}: {aggregate.error}"
                                )
                            if aggregate.blob_hash != prepared_cohort_source_hash(ordered):
                                raise RetainedPreparationRetryableError(
                                    f"retained cohort source dependency changed for {logical_key}"
                                )
                            try:
                                aggregate.verify_files(full=True, stop=compute_cancel_requested)
                            except (OSError, ValueError) as exc:
                                raise RetainedPreparationRetryableError(
                                    f"retained cohort preparation seal changed for {logical_key}"
                                ) from exc
                            aggregates[logical_key] = PreparedRetainedAggregate(accepted_raw_ids, aggregate)
                        if not needs_source_census:
                            # A pending-raw envelope whose bytes hold several
                            # sessions has not yet been censused into per-session
                            # memberships; the census, not a one-session write,
                            # settles it.
                            needs_source_census = any(
                                logical_key.startswith(PENDING_RAW_LOGICAL_SOURCE_PREFIX)
                                and _holds_several_sessions(prepared[accepted_raw_ids[-1]].prepared_artifact)
                                for logical_key, accepted_raw_ids in planned_accepted_raw_ids.items()
                                if accepted_raw_ids
                            )
                        if needs_source_census:
                            from polylogue.sources.revision_backfill import (
                                PreparedRevisionSourceCensus,
                                RevisionCensusResult,
                                prepare_revision_source_census,
                            )

                            with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                census_read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                census_state = prepare_revision_source_census(
                                    reference_seal,
                                    census_read,
                                    selected_raw_ids=raw_ids,
                                    prepared_inputs=prepared,
                                )
                                census_raw_ids, census_keys = census_read.expand_raw_membership_selection(raw_ids)
                            # Byte classification reads the census staged above,
                            # so one tape publishes both phases.
                            from polylogue.sources.revision_backfill import stage_revision_source_classification

                            classified, classification_stats = stage_revision_source_classification(
                                reference_seal,
                                PreparedSessionSourceRead(reference_seal, blob_store=material_store),
                                logical_keys=census_keys,
                                payload_store=material_store,
                            )
                            prepared_source_census = PreparedRevisionSourceCensus(
                                reference_seal.prepare_source_mutation(),
                                RevisionCensusResult(
                                    census_state.scanned,
                                    census_state.classified,
                                    census_state.quarantined,
                                    census_raw_ids,
                                    census_keys,
                                ),
                                classification_result=(
                                    RevisionCensusResult(0, 0, 0, census_raw_ids, census_keys) if classified else None
                                ),
                                classification_blob_stats=classification_stats if classified else (),
                            )
                        if not needs_source_census:
                            from polylogue.sources.revision_backfill import (
                                PreparedRevisionSourceCensus,
                                RevisionCensusResult,
                                prepare_revision_source_membership_conversion,
                            )

                            with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                conversion_read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                converted = prepare_revision_source_membership_conversion(
                                    reference_seal,
                                    conversion_read,
                                    logical_keys=tuple(key for key in logical_keys if key not in prepared_key_refusals),
                                    prepared_inputs=prepared,
                                    prepared_replay_plans=planned_accepted_raw_ids,
                                )
                                conversion_raw_ids: tuple[str, ...] = ()
                                conversion_keys: tuple[str, ...] = ()
                                if converted:
                                    conversion_raw_ids, conversion_keys = (
                                        conversion_read.expand_raw_membership_selection(raw_ids)
                                    )
                            if converted:
                                prepared_source_census = PreparedRevisionSourceCensus(
                                    reference_seal.prepare_source_mutation(),
                                    RevisionCensusResult(0, converted, 0, conversion_raw_ids, conversion_keys),
                                )
                                needs_source_census = True
                        if not needs_source_census and archive.index_connection is not None:
                            # Keyed by (raw, session): one raw may carry several sessions.
                            from polylogue.sources.revision_backfill import prepared_session_for_revision_key

                            selected_writes: dict[tuple[str, str], tuple[ParsedSession, PreparedJsonl]] = {}
                            for logical_key, accepted_raw_ids in planned_accepted_raw_ids.items():
                                if logical_key in prepared_key_refusals or not accepted_raw_ids:
                                    continue
                                tip_raw_id = accepted_raw_ids[-1]
                                selected_artifact = (
                                    aggregates[logical_key].artifact
                                    if len(accepted_raw_ids) > 1 and logical_key in aggregates
                                    else prepared[tip_raw_id].prepared_artifact
                                )
                                if selected_artifact is None:
                                    # The accepted tip's retained bytes did not parse, so
                                    # the key has no session to write (acquisition already
                                    # settled such bytes as a typed non-session admission).
                                    # Refuse the key here so no later byte phase expects an
                                    # adoption for it while the healthy keys publish.
                                    tip_input = prepared.get(tip_raw_id)
                                    tip_error = tip_input.parser_error if tip_input is not None else None
                                    prepared_key_refusals[logical_key] = CohortMembershipRefusalError(
                                        logical_key,
                                        tip_raw_id,
                                        f"retained raw did not parse: {tip_error or 'no prepared artifact'}",
                                    )
                                    continue
                                try:
                                    selected_session = prepared_session_for_revision_key(
                                        selected_artifact.session_sequence(),
                                        raw_id=tip_raw_id,
                                        logical_source_key=logical_key,
                                    )
                                except CohortMembershipRefusalError as refusal:
                                    prepared_key_refusals[logical_key] = refusal
                                    continue
                                selected_writes[(tip_raw_id, _session_id(selected_session))] = (
                                    selected_session,
                                    selected_artifact,
                                )
                                if is_work_event_raw_id(logical_key):
                                    continue
                                with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                    adoption_read = PreparedSessionSourceRead(reference_seal, blob_store=material_store)
                                    prepared_replay_adoption[(logical_key, accepted_raw_ids)] = (
                                        adoption_read.prepare_raw_revision_replay_adoption(
                                            [selected_session],
                                            logical_source_key=logical_key,
                                            raw_ids=accepted_raw_ids,
                                        )
                                    )
                            with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                lineage_parent_raw_ids = self._lineage_parent_raw_ids(
                                    reference_seal,
                                    PreparedSessionSourceRead(reference_seal, blob_store=material_store),
                                    selected_writes,
                                    unit_raw_ids=raw_ids,
                                )
                            if lineage_parent_raw_ids:
                                # The widened unit prepares everything again; this
                                # carrier only drains what was prepared so far.
                                return RawObservationReplacement(
                                    key,
                                    binding,
                                    None,
                                    raw_ids,
                                    prepared_inputs=prepared,
                                    prepared_aggregates=aggregates,
                                    lineage_parent_raw_ids=lineage_parent_raw_ids,
                                    scratch_directory=scratch,
                                    scratch_owner=scratch_owner,
                                )
                            from polylogue.storage.sqlite.archive_tiers.revision_governance import (
                                prepared_raw_revision_file_mtime,
                            )

                            with (
                                reference_seal.original_read_snapshot(
                                    input_demand=self._compute_adapter.amend_current_input_demand,
                                ),
                                reference_seal.source_producer(),
                            ):
                                source_read = PreparedSessionSourceRead(
                                    reference_seal,
                                    blob_store=blob_store_for_connection(reference_seal.observer("source")),
                                )
                                from polylogue.sources.revision_backfill import (
                                    _lineage_aware_replay_schedule,
                                    _PreparedReplayInputs,
                                )

                                prepared_logical_keys = source_read.raw_revision_rebuild_logical_keys(raw_ids)
                                _expanded_raw_ids, expanded_membership_keys = (
                                    source_read.expand_raw_membership_selection(raw_ids)
                                )
                                # Event envelopes have singleton byte authority and
                                # append to an existing target; they never replace
                                # transcript membership or its accepted head.
                                prepared_membership_keys = tuple(
                                    key for key in expanded_membership_keys if not is_work_event_raw_id(key)
                                )
                                # Membership-keyed sessions are ordered with the
                                # byte-typed ones, so a parent replays before its
                                # children whichever authority keys either.
                                prepared_replay_schedule = _lineage_aware_replay_schedule(
                                    {
                                        logical_key
                                        for logical_key in (*prepared_logical_keys, *prepared_membership_keys)
                                        if not is_work_event_raw_id(logical_key)
                                    },
                                    source_read,
                                    _PreparedReplayInputs(prepared),
                                )
                                prepared_byte_logical_keys = tuple(
                                    logical_key
                                    for logical_key in prepared_replay_schedule.order
                                    if not (
                                        source_read.pending_raw_envelope_has_membership_authority(logical_key)
                                        or source_read.raw_membership_logical_raw_ids(logical_key)
                                    )
                                )
                                for (tip_raw_id, session_id), (session, artifact) in selected_writes.items():
                                    _raise_if_compute_cancelled(f"raw {tip_raw_id}")
                                    if artifact.shard_path is None:
                                        raise RetainedPreparationRetryableError(
                                            f"retained replay shard is absent for raw {tip_raw_id}"
                                        )
                                    event_only = is_work_event_raw_id(tip_raw_id)
                                    prepared_writes[(tip_raw_id, session_id)] = prepare_session_write(
                                        reference_seal.observer("index"),
                                        session,
                                        merge_append=event_only,
                                        fallback_timestamp=prepared_raw_revision_file_mtime(reference_seal, tip_raw_id),
                                        source_read=source_read,
                                        raw_id=tip_raw_id,
                                        force_replace=not event_only,
                                        prepared_rows=(
                                            None
                                            if event_only
                                            else prepared_session_rows_from_shard(artifact.shard_path, session_id)
                                        ),
                                        before_input=reference_seal.before_index_input,
                                    )
                                from polylogue.storage.sqlite.archive_tiers.revision_governance import (
                                    prepare_revision_replay_outcome,
                                )

                                for logical_key in prepared_byte_logical_keys:
                                    if logical_key in prepared_key_refusals:
                                        continue
                                    byte_plan = prepared_revision_plans[logical_key]
                                    if not byte_plan.accepted_raw_ids:
                                        continue
                                    byte_adoption = prepared_replay_adoption[(logical_key, byte_plan.accepted_raw_ids)]
                                    if byte_adoption.session_id is None:
                                        raise RetainedPreparationRetryableError(
                                            "byte replay has no prepared adoption identity"
                                        )
                                    byte_tip = byte_plan.accepted_raw_ids[-1]
                                    byte_session, _byte_artifact = selected_writes[(byte_tip, byte_adoption.session_id)]
                                    byte_write = prepared_writes[(byte_tip, byte_adoption.session_id)]
                                    prepared_byte_outcomes[logical_key] = prepare_revision_replay_outcome(
                                        reference_seal,
                                        source_read,
                                        byte_plan,
                                        byte_adoption,
                                        aggregate_session=byte_session,
                                        aggregate_content_hash=byte_write.rows.session_content_hash,
                                        prepared_write=byte_write,
                                    )
                            for logical_key in logical_keys:
                                if logical_key in prepared_key_refusals or is_work_event_raw_id(logical_key):
                                    continue
                                with (
                                    reference_seal.original_read_snapshot(),
                                    reference_seal.source_producer(),
                                ):
                                    membership_read = PreparedSessionSourceRead(
                                        reference_seal,
                                        blob_store=material_store,
                                    )
                                    if planned_accepted_raw_ids.get(logical_key) and not (
                                        membership_read.membership_key_has_pending_envelope_member(logical_key)
                                        or membership_read.raw_membership_logical_raw_ids(logical_key)
                                    ):
                                        continue
                                    preceding_byte = prepared_byte_outcomes.get(logical_key)
                                    effective_head_raw_id = (
                                        (
                                            None
                                            if preceding_byte.effective_head is None
                                            else str(preceding_byte.effective_head[1])
                                        )
                                        if preceding_byte is not None
                                        else membership_read.raw_revision_head_raw_id(logical_key)
                                    )
                                    try:
                                        membership_plan = prepare_membership_replay(
                                            membership_read,
                                            logical_key,
                                            prepared,
                                            head_raw_id=effective_head_raw_id,
                                            stop=compute_cancel_requested,
                                        )
                                    except CohortMembershipRefusalError as refusal:
                                        prepared_key_refusals[logical_key] = refusal
                                        continue
                                    if preceding_byte is None:
                                        head_plan = prepare_membership_head_plan(
                                            reference_seal.observer("index"),
                                            membership_read,
                                            logical_key,
                                            membership_plan.classification,
                                            before_input=reference_seal.before_index_input,
                                        )
                                    else:
                                        head_plan = prepare_membership_head_plan_from_inputs(
                                            membership_read,
                                            logical_key,
                                            membership_plan.classification,
                                            existing_head=membership_head_input_from_revision_head(
                                                preceding_byte.effective_head
                                            ),
                                            persisted_session=preceding_byte.effective_session_revision,
                                        )
                                    membership_plan = replace(membership_plan, head_plan=head_plan)
                                    if head_plan.conflict is not None:
                                        # A refused cohort publishes nothing this pass;
                                        # its members carry the retryable evidence.
                                        membership_plan = replace(
                                            membership_plan, classification=MembershipClassification((), (), ())
                                        )
                                    accepted_members = membership_plan.classification.accepted_raw_ids
                                    refused_member = next(
                                        (
                                            raw_id
                                            for raw_id in accepted_members
                                            if (item := prepared.get(raw_id)) is not None
                                            and item.prepared_artifact is not None
                                            and item.validation_verdict is not None
                                            and item.validation_verdict.strict_refusal
                                        ),
                                        None,
                                    )
                                    if refused_member is not None:
                                        refused_input = prepared.get(refused_member)
                                        if refused_input is None or refused_input.prepared_artifact is None:
                                            raise AssertionError("strictly refused member lost its prepared artifact")
                                        verdict = refused_input.validation_verdict
                                        if verdict is None:
                                            raise AssertionError("strictly refused member lost its validation verdict")
                                        detail = (
                                            verdict.first_diagnostic
                                            if verdict.first_diagnostic
                                            else "strict schema validation refused the retained revision"
                                        )
                                        prepared_key_refusals[logical_key] = CohortMembershipRefusalError(
                                            logical_key,
                                            refused_member,
                                            f"retained schema validation refused revision: {detail}",
                                        )
                                        continue
                                    if accepted_members:
                                        prepared_replay_adoption[(logical_key, accepted_members)] = (
                                            membership_read.prepare_raw_revision_replay_adoption(
                                                [membership_plan.sessions[raw_id] for raw_id in accepted_members],
                                                logical_source_key=logical_key,
                                                raw_ids=accepted_members,
                                            )
                                        )
                                membership_plans[logical_key] = membership_plan
                                if (
                                    not membership_plan.classification.accepted_raw_ids
                                    or head_plan.yield_to_head_raw_id is not None
                                ):
                                    continue
                                accepted_raw_id = membership_plan.classification.accepted_raw_ids[-1]
                                accepted_session = membership_plan.sessions[accepted_raw_id]
                                membership_artifact = prepared[accepted_raw_id].prepared_artifact
                                if membership_artifact is not None:
                                    selected_writes[(accepted_raw_id, _session_id(accepted_session))] = (
                                        accepted_session,
                                        membership_artifact,
                                    )
                                    with reference_seal.original_read_snapshot(), reference_seal.source_producer():
                                        membership_source = PreparedSessionSourceRead(
                                            reference_seal, blob_store=material_store
                                        )
                                        if membership_artifact.shard_path is None:
                                            raise RetainedPreparationRetryableError("membership replay shard is absent")
                                        member_session_id = _session_id(accepted_session)
                                        prepared_writes[(accepted_raw_id, member_session_id)] = prepare_session_write(
                                            reference_seal.observer("index"),
                                            accepted_session,
                                            merge_append=False,
                                            fallback_timestamp=prepared_raw_revision_file_mtime(
                                                reference_seal, accepted_raw_id
                                            ),
                                            source_read=membership_source,
                                            raw_id=accepted_raw_id,
                                            force_replace=True,
                                            prepared_rows=prepared_session_rows_from_shard(
                                                membership_artifact.shard_path, member_session_id
                                            ),
                                            before_input=reference_seal.before_index_input,
                                        )
                            # A child prepared against an Index without its parent
                            # expects no parent; if that parent publishes earlier in
                            # this same unit, the child's write would find it and
                            # refuse as moved lineage. Defer the child: the rest
                            # publishes, and the child is re-prepared in this same
                            # pass against its published parent.
                            write_keys = {
                                **{
                                    logical_key: (
                                        prepared_revision_plans[logical_key].accepted_raw_ids[-1],
                                        str(
                                            prepared_replay_adoption[
                                                (logical_key, prepared_revision_plans[logical_key].accepted_raw_ids)
                                            ].session_id
                                        ),
                                    )
                                    for logical_key in prepared_byte_outcomes
                                },
                                **{
                                    logical_key: (
                                        plan.classification.accepted_raw_ids[-1],
                                        _session_id(plan.sessions[plan.classification.accepted_raw_ids[-1]]),
                                    )
                                    for logical_key, plan in membership_plans.items()
                                    if plan.classification.accepted_raw_ids
                                },
                            }
                            prepared_lineage_deferrals = self._lineage_deferrals(
                                prepared_writes,
                                selected_writes,
                                write_keys=write_keys,
                                refused_keys=prepared_key_refusals,
                            )
                            for deferred_key in prepared_lineage_deferrals:
                                prepared_byte_outcomes.pop(deferred_key, None)
                                deferred_plan = membership_plans.pop(deferred_key, None)
                                lineage_deferred_raw_ids.update(
                                    deferred_plan.candidate_raw_ids
                                    if deferred_plan is not None
                                    else prepared_revision_plans[deferred_key].accepted_raw_ids
                                )
                            # A later membership refusal supersedes any earlier
                            # prepared byte outcome for that same original key.
                            # The Source acknowledgement and writer consume the
                            # same remaining outcomes, including shared Raw IDs.
                            for refused_key in prepared_key_refusals:
                                prepared_byte_outcomes.pop(refused_key, None)

                            def prepare_marker_write(raw_id: str, session: ParsedSession) -> PreparedSessionWrite:
                                retained = prepared.get(raw_id)
                                artifact = retained.prepared_artifact if retained is not None else None
                                if artifact is None or artifact.shard_path is None:
                                    raise RetainedPreparationRetryableError(
                                        f"accepted marker request has no owned prepared shard for {raw_id}"
                                    )
                                session_id = _session_id(session)
                                marker_source = PreparedSessionSourceRead(
                                    reference_seal,
                                    blob_store=material_store,
                                )
                                return prepare_session_write(
                                    reference_seal.observer("index"),
                                    session,
                                    merge_append=False,
                                    fallback_timestamp=prepared_raw_revision_file_mtime(reference_seal, raw_id),
                                    source_read=marker_source,
                                    raw_id=raw_id,
                                    force_replace=True,
                                    prepared_rows=prepared_session_rows_from_shard(
                                        artifact.shard_path,
                                        session_id,
                                    ),
                                    before_input=reference_seal.before_index_input,
                                )

                            prepared_replay_source = prepare_retained_replay_source(
                                reference_seal,
                                prepared_inputs=prepared,
                                byte_outcomes=prepared_byte_outcomes,
                                membership_plans=membership_plans,
                                adoptions=prepared_replay_adoption,
                                prepared_writes=prepared_writes,
                                marker_write_factory=prepare_marker_write,
                            )
                            membership_plans = dict(prepared_replay_source.membership_plans)
                    except BaseException as primary:
                        failure = (
                            DaemonOperationCancelled("retained preparation cancelled during verification")
                            if isinstance(primary, (BlobVerificationCancelledError, VerificationCancelledError))
                            else primary
                        )

                        from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError

                        preserve_neutral = isinstance(
                            primary, (_CarryInvalidatedError, ReferenceSealStaleError)
                        ) and bool(carry.neutral_artifacts)
                        failed_artifacts = (
                            *prepared_artifacts.values(),
                            *(() if preserve_neutral else carry.neutral_artifacts.values()),
                            *(item.artifact for item in aggregates.values()),
                        )
                        # The original seal retains failed native drains. Transfer
                        # these physical owners before the outer carry can close them.
                        prepared_artifacts.clear()
                        if not preserve_neutral:
                            carry.neutral_artifacts.clear()
                            carry.scratch_owner = None

                        def close_failed_payload() -> None:
                            _close_prepared_carriers(prepared_writes, membership_plans, failed_artifacts)
                            if not preserve_neutral:
                                _cleanup_scratch(scratch_owner)

                        # The original seal releases settled SQL dependencies
                        # before invoking this payload, and retains it on any
                        # physical close failure for the original owner's retry.
                        reference_seal.retain_preparation_payload(close_failed_payload)
                        if failure is not primary:
                            raise failure from primary
                        raise
                    return RawObservationReplacement(
                        key,
                        binding,
                        None,
                        raw_ids,
                        prepared_inputs=prepared,
                        prepared_aggregates=aggregates,
                        prepared_writes=prepared_writes,
                        prepared_replay_adoption=prepared_replay_adoption,
                        prepared_source_classification=prepared_source_classification,
                        verified_blob_stats=verified_blob_stats,
                        planned_accepted_raw_ids=planned_accepted_raw_ids,
                        prepared_revision_plans=prepared_revision_plans,
                        prepared_byte_outcomes=prepared_byte_outcomes,
                        prepared_membership_plans=membership_plans,
                        prepared_replay_source=prepared_replay_source,
                        needs_source_census=needs_source_census,
                        prepared_source_census=prepared_source_census,
                        prepared_replay_schedule=prepared_replay_schedule,
                        prepared_logical_keys=prepared_logical_keys,
                        prepared_membership_keys=prepared_membership_keys,
                        prepared_byte_logical_keys=prepared_byte_logical_keys,
                        prepared_key_refusals=tuple(prepared_key_refusals.values()),
                        prepared_lineage_deferrals=prepared_lineage_deferrals,
                        lineage_deferred_raw_ids=tuple(sorted(lineage_deferred_raw_ids)),
                        scratch_directory=scratch,
                        scratch_owner=scratch_owner,
                        provider_parse_seconds=provider_parse_seconds,
                    )
            # An empty raw selection has no parse payload; the canonical replay
            # owner remains responsible for its final authority verdict.
            return RawObservationReplacement(key, binding, None, raw_ids)

    def _stage_absent_blob_restorations(
        self,
        archive: ArchiveStore,
        raw_ids: tuple[str, ...],
        descriptors: Mapping[str, tuple[Provider, str, str, object, int]],
    ) -> StagedBlobRestorations | None:
        """Stage exact source bytes for every retained blob the component lacks.

        Preparation cannot read an absent blob, and retrying it cannot make
        the bytes reappear. When a recorded source window of the raw -- a
        direct file window or its ZIP member -- still holds bytes whose
        SHA-256 and size equal the raw's, those bytes are staged here and
        published by the writer in ``publish``; the next pass then prepares
        over present bytes. Otherwise the retryable refusal names why
        (``source_missing``, ``hash_mismatch``, ``container_member_rejected``,
        ``inexact_payload`` ...) instead of reporting a vanished file; it
        stays retryable because the source or a restored backup can still
        bring the bytes back.
        """
        from polylogue.sources.revision_backfill import RetainedPreparationRetryableError

        blob_store = BlobStore(self.archive_root / "blob")
        staged: list[tuple[str, PreparedBlob]] = []
        staged_hashes: set[str] = set()
        try:
            for raw_id in raw_ids:
                _provider, blob_hash, path, _kind, _size = descriptors[raw_id]
                if blob_hash in staged_hashes:
                    continue
                try:
                    self._blob_stat_identity(blob_store.blob_path(blob_hash))
                    continue
                except FileNotFoundError:
                    pass
                except OSError:
                    # Present but unreadable: preparation reports it as a
                    # retryable disappearance, not as lost bytes.
                    continue
                prepared, reason = stage_blob_from_recorded_source(
                    archive.source_connection,
                    self.archive_root,
                    blob_store,
                    raw_id,
                    blob_hash=blob_hash,
                    source_path=path,
                    stop=compute_cancel_requested,
                )
                if prepared is None:
                    raise RetainedPreparationRetryableError(
                        f"retained raw blob absent and not restorable from its source ({reason}): {raw_id}"
                    )
                staged.append((raw_id, prepared))
                staged_hashes.add(blob_hash)
        except BlobVerificationCancelledError as exc:
            _discard_staged_blobs(blob_store, tuple(prepared for _raw_id, prepared in staged))
            raise DaemonOperationCancelled("retained blob restoration cancelled") from exc
        except BaseException:
            _discard_staged_blobs(blob_store, tuple(prepared for _raw_id, prepared in staged))
            raise
        return StagedBlobRestorations(blob_store, staged) if staged else None

    def _publish_blob_restorations(self, restorations: StagedBlobRestorations) -> None:
        """Reserve and publish staged restorations, then consume their receipts."""
        from polylogue.storage.blob_publication import ArchiveBlobPublisher, consume_restored_raw_blob_receipts

        source_db = self.archive_root / "source.db"
        publisher = ArchiveBlobPublisher(source_db, restorations.store.root, store=restorations.store)
        raw_by_hash = {prepared.hash_hex: raw_id for raw_id, prepared in restorations.staged}
        for _raw_id, prepared in restorations.staged:
            publisher.queue_prepared(prepared)
        receipts = publisher.flush()
        restorations.published()
        consume_restored_raw_blob_receipts(
            source_db, tuple((receipt, raw_by_hash[receipt.blob_hash]) for receipt in receipts)
        )

    def _publication_inputs_current(
        self, frame: RawFrame, replacement: RawObservationReplacement, *, before_restoration: bool
    ) -> bool:
        """Revalidate a prepared unit's inputs under the writer, before any exposure."""
        from polylogue.storage.raw_retention import raw_frontier_blocked_raw_ids

        assert replacement.reference_seal is not None
        if before_restoration:
            if not self._current(frame):
                return False
            # This gate also covers early Blob restoration, whose branch
            # does not reach a Source or Index permit. It must precede any
            # exposure, and checks the same original observers after actual
            # writer admission instead of copying a second value digest.
            replacement.reference_seal.validate_observers_current()
            frontier_refusal = raw_frontier_blocked_raw_ids(self.archive_root, replacement.raw_ids)
            selected_paths = set(self.source_paths(replacement.raw_ids).values())
            return frontier_refusal.unattributed_reason is None and not selected_paths.intersection(
                frontier_refusal.source_paths
            )
        with self._preparation_archive() as archive:
            raw_ids, _keys = archive.expand_raw_membership_selection(list(replacement.raw_ids))
            if raw_ids != replacement.raw_ids:
                return False
            for raw_id in raw_ids:
                _provider, blob_hash, _path, _kind, _size = archive.raw_revision_descriptor(raw_id)
                blob_store = BlobStore(self.archive_root / "blob")
                if replacement.prepared_inputs is not None:
                    expected_stat = (replacement.verified_blob_stats or {}).get(raw_id)
                    try:
                        current_stat = self._blob_stat_identity(blob_store.blob_path(blob_hash))
                    except OSError:
                        return False
                    if expected_stat is None or current_stat != expected_stat:
                        return False
                elif not replacement.needs_source_classification:
                    return False
        for prepared in (replacement.prepared_inputs or {}).values():
            if prepared.prepared_artifact is not None:
                try:
                    prepared.prepared_artifact.verify_files(full=False)
                except (OSError, ValueError):
                    return False
        for aggregate in (replacement.prepared_aggregates or {}).values():
            try:
                aggregate.artifact.verify_files(full=False)
            except (OSError, ValueError):
                return False
        return True

    def _apply_source_phase(
        self,
        replacement: RawObservationReplacement,
        *,
        phase_receipt: Callable[
            [Literal["census", "classification", "replay"], RevisionCensusResult | PreparedRevisionReplayResult], None
        ]
        | None,
        publication_failure: Callable[[BaseException], None] | None,
    ) -> bool:
        """Commit a prepared census or classification tape; False reports a refused classification."""
        from polylogue.sources.revision_backfill import (
            RetainedPreparationNoProgressError,
            RetainedPreparationRetryableError,
            apply_prepared_revision_census,
            apply_prepared_revision_classification,
        )
        from polylogue.storage.sqlite.archive_tiers.revision_governance import PreparedRawClassificationStaleError

        assert replacement.reference_seal is not None
        phase: Literal["census", "classification"]
        receipts: list[tuple[Literal["census", "classification"], RevisionCensusResult]] = []
        before = self._census_state(replacement.raw_ids)
        if replacement.needs_source_census:
            if replacement.prepared_source_census is None:
                raise RetainedPreparationRetryableError("retained census lacks its original prepared Source tape")
            phase = "census"
            try:
                receipt = apply_prepared_revision_census(
                    replacement.reference_seal,
                    replacement.prepared_source_census,
                    payload_store=BlobStore(self.archive_root / "blob"),
                )
            except PreparedRawClassificationStaleError as failure:
                if publication_failure is not None:
                    publication_failure(failure)
                return False
            receipts.append((phase, receipt))
            classification = replacement.prepared_source_census.classification_result
            if classification is not None:
                receipts.append(("classification", classification))
        else:
            if replacement.prepared_source_classification is None:
                raise RetainedPreparationRetryableError("retained classification lacks its original Source tape")
            phase = "classification"
            try:
                receipt = apply_prepared_revision_classification(
                    replacement.reference_seal,
                    replacement.prepared_source_classification,
                    payload_store=BlobStore(self.archive_root / "blob"),
                )
            except (PreparedRawClassificationStaleError, RetainedPreparationRetryableError) as failure:
                if publication_failure is not None:
                    publication_failure(failure)
                return False
            receipts.append((phase, receipt))
        if self._census_state(replacement.raw_ids) == before:
            raise RetainedPreparationNoProgressError(
                f"retained {phase} left its durable inputs unchanged: {replacement.key}"
            )
        # A terminal STRICT schema refusal is durable as soon as its Source
        # census/classification receipt commits. That route returns before the
        # later Index replay hook, so persist its best-effort drift signal here.
        _record_retained_schema_drift(
            self.archive_root,
            replacement.prepared_inputs,
            index_db_path=Path(self.archive_root / "index.db"),
            strict_refusals_only=True,
        )
        if phase_receipt is not None:
            for committed_phase, committed_receipt in receipts:
                phase_receipt(committed_phase, committed_receipt)
        refusal = next(iter(self.terminal_decode_refusals(replacement.raw_ids).values()), None)
        if refusal is not None:
            raise refusal
        return True

    def publish(
        self,
        frame: RawFrame,
        replacement: RawObservationReplacement,
        *,
        phase_receipt: Callable[
            [Literal["census", "classification", "replay"], RevisionCensusResult | PreparedRevisionReplayResult], None
        ]
        | None = None,
        publication_failure: Callable[[BaseException], None] | None = None,
    ) -> bool:
        if phase_receipt is not None:
            for phase, receipt in replacement.committed_phase_receipts:
                phase_receipt(phase, receipt)
        if replacement.prepared_phase_failure is not None:
            replacement.close()
            if publication_failure is not None:
                publication_failure(replacement.prepared_phase_failure)
            published = False
        else:
            published = self._publish_prepared(
                frame, replacement, phase_receipt=phase_receipt, publication_failure=publication_failure
            )
        if not published and replacement.committed_phase_receipts:
            # The Source phases committed in place are this key's progress.
            self._phase_committed[id(replacement)] = replacement.key
        return published

    def _publish_prepared(
        self,
        frame: RawFrame,
        replacement: RawObservationReplacement,
        *,
        phase_receipt: Callable[
            [Literal["census", "classification", "replay"], RevisionCensusResult | PreparedRevisionReplayResult], None
        ]
        | None,
        publication_failure: Callable[[BaseException], None] | None,
    ) -> bool:
        from polylogue.sources.revision_backfill import (
            RetainedPreparationNoProgressError,
            RetainedPreparationRetryableError,
            apply_prepared_revision_replay,
        )
        from polylogue.storage.index_generation import ActiveWriterLease

        if replacement.already_valid:
            try:
                return self._current(frame) and self.inspect(frame, (replacement.key,)).get(replacement.key) == "valid"
            finally:
                replacement.close()

        with retain_native_sql_lifetimes(*(() if replacement.scratch_owner is None else (replacement.scratch_owner,))):
            lease = ActiveWriterLease(self.archive_root)
            lifetime_bound = False
            try:
                lease.acquire()
                if replacement.reference_seal is None:
                    raise RetainedPreparationRetryableError("retained publication lacks its original reference seal")
                replacement.reference_seal.retain_publication_lifetime(lease, replacement._close_prepared_payload)
                lifetime_bound = True
                if not self._publication_inputs_current(frame, replacement, before_restoration=True):
                    return False
                if replacement.blob_restorations is not None:
                    self._publish_blob_restorations(replacement.blob_restorations)
                    # The restored bytes are prepared by the next phase, which
                    # now finds them present; this publication certifies no output.
                    self._phase_committed[id(replacement)] = replacement.key
                    return False
                if not self._publication_inputs_current(frame, replacement, before_restoration=False):
                    return False
                if replacement.needs_source_census or replacement.needs_source_classification:
                    if self._apply_source_phase(
                        replacement, phase_receipt=phase_receipt, publication_failure=publication_failure
                    ):
                        self._phase_committed[id(replacement)] = replacement.key
                    return False
                if (
                    replacement.prepared_inputs is None
                    or replacement.prepared_aggregates is None
                    or replacement.prepared_writes is None
                    or replacement.planned_accepted_raw_ids is None
                    or replacement.prepared_revision_plans is None
                    or replacement.prepared_byte_outcomes is None
                    or replacement.prepared_membership_plans is None
                    or replacement.prepared_replay_source is None
                    or replacement.prepared_replay_adoption is None
                    or replacement.prepared_replay_schedule is None
                ):
                    return False
                try:
                    replay_receipt = apply_prepared_revision_replay(
                        self.archive_root,
                        reference_seal=replacement.reference_seal,
                        active_index_path=Path(frame.source_revision),
                        selected_raw_ids=list(replacement.raw_ids),
                        prepared_inputs=replacement.prepared_inputs,
                        prepared_aggregates=replacement.prepared_aggregates,
                        prepared_writes=replacement.prepared_writes,
                        prepared_replay_adoption=replacement.prepared_replay_adoption,
                        prepared_replay_plans=replacement.prepared_revision_plans,
                        prepared_byte_outcomes=replacement.prepared_byte_outcomes,
                        prepared_membership_plans=replacement.prepared_membership_plans,
                        prepared_replay_source=replacement.prepared_replay_source,
                        prepared_replay_schedule=replacement.prepared_replay_schedule,
                        prepared_logical_keys=replacement.prepared_logical_keys,
                        prepared_membership_keys=replacement.prepared_membership_keys,
                        prepared_byte_logical_keys=replacement.prepared_byte_logical_keys,
                        prepared_key_refusals=replacement.prepared_key_refusals,
                        prepared_lineage_deferrals=replacement.prepared_lineage_deferrals,
                        bulk_fts=True,
                        # One publication is one component of a pass: each
                        # replayed session proves its own FTS rows, and the
                        # archive-wide audit belongs to the daemon's
                        # fts_readiness_binding stage after the burst.
                        exact_fts_audit=False,
                    )
                except RetainedPreparationRetryableError as failure:
                    if publication_failure is not None:
                        publication_failure(failure)
                    return False
                _record_retained_schema_drift(
                    self.archive_root,
                    replacement.prepared_inputs,
                    index_db_path=Path(frame.source_revision),
                )
                replay_receipt.stage_timings_s["provider_parse"] = (
                    replay_receipt.stage_timings_s.get("provider_parse", 0.0) + replacement.provider_parse_seconds
                )
                if phase_receipt is not None:
                    phase_receipt("replay", replay_receipt)
                if replay_receipt.changed_session_ids:
                    # Searches answered before this publication are cached by
                    # epoch; a changed session must never be served stale.
                    from polylogue.storage.search.cache import invalidate_search_cache

                    invalidate_search_cache()
                refusal = next(iter(self.terminal_decode_refusals(replacement.raw_ids).values()), None)
                if refusal is not None:
                    raise refusal
                if replacement.prepared_lineage_deferrals:
                    # The parents published; their deferred children are
                    # prepared against them by the next phase of this pass.
                    self._phase_committed[id(replacement)] = replacement.key
                    return False
                published_nothing = not (
                    replay_receipt.replayed_logical_sources
                    or replay_receipt.written_session_ids
                    or replay_receipt.writer_changed_raw_ids
                    or replay_receipt.membership_refusals
                )
                if published_nothing and self.inspect(frame, (replacement.key,)).get(replacement.key) != "valid":
                    # The replay applied nothing and left this key unsettled:
                    # its cohort has no accepted chain (an append over a
                    # baseline whose authority is only asserted). Preparing
                    # again reads the same Source state, so this is no
                    # progress, never a publication.
                    raise RetainedPreparationNoProgressError(f"retained replay published nothing for {replacement.key}")
                return True
            finally:
                if lifetime_bound:
                    primary = sys.exception()
                    try:
                        replacement.close()
                    except BaseException as cleanup:
                        if primary is not None:
                            raise BaseExceptionGroup(
                                "retained publication and cleanup failed", [primary, cleanup]
                            ) from primary
                        raise
                else:
                    primary = sys.exception()
                    failures: list[BaseException] = []
                    for close in (replacement.close, lease.close):
                        try:
                            close()
                        except BaseException as failure:
                            failures.append(failure)
                    if failures:
                        if primary is not None:
                            failures.insert(0, primary)
                        raise BaseExceptionGroup("retained publication admission cleanup failed", failures)


def _close_prepared_carriers(
    writes: Mapping[tuple[str, str], PreparedSessionWrite],
    plans: Mapping[str, PreparedMembershipReplay],
    artifacts: Iterable[PreparedJsonl] = (),
) -> None:
    """Attempt each owner close before deleting their containing scratch tree."""
    failures: list[BaseException] = []
    carriers: Iterator[SQLCustodyOwner] = chain(writes.values(), plans.values())
    for carrier in carriers:
        try:
            carrier.close()
        except BaseException as exc:
            failures.append(exc)
    seen: set[int] = set()
    for artifact in artifacts:
        if id(artifact) in seen:
            continue
        seen.add(id(artifact))
        try:
            artifact.discard()
        except BaseException as exc:
            failures.append(exc)
    if failures:
        raise BaseExceptionGroup("retained replay carrier cleanup failed", failures)


def _cleanup_scratch(scratch_owner: tempfile.TemporaryDirectory[str]) -> None:
    """Remove a replay scratch tree and the decodes cached from its carriers."""
    from polylogue.sources.prepared_message_sink import discard_decoded_sessions_under
    from polylogue.storage.sqlite.connection_profile import (
        NativeConnectionSettlementError,
        retained_native_sql_owners_for_lifetime,
    )

    pending = retained_native_sql_owners_for_lifetime(scratch_owner)
    if pending:
        raise NativeConnectionSettlementError(pending[0], RuntimeError("scratch cleanup requires native drain"))
    discard_decoded_sessions_under(Path(scratch_owner.name))
    scratch_owner.cleanup()


T = TypeVar("T")

#: Seconds without a result before a retained preparation is reported
#: stalled. A report, not a deadline (polylogue-slc55): terminating the pool
#: at a fixed time discarded a large raw's preparation and retried it from
#: scratch, so it never finished.
_RETAINED_PREPARATION_STALL_REPORT_SECONDS = 600.0


_STILL_RUNNING = object()


def _raise_if_compute_cancelled(subject: str) -> None:
    """Stop an owned retained-preparation phase without surrendering cleanup."""
    if compute_cancel_requested():
        raise DaemonOperationCancelled(f"retained preparation cancelled for {subject}")


#: How often a wait checks its caller's cancellation.
_RETAINED_PREPARATION_CANCEL_POLL_SECONDS = 1.0


def _await_reporting_stalls(operation: SubmittedOperation[T], *, subject: str) -> T:
    """Wait for ``future``; report every window it runs without finishing.

    A retained preparation has no deadline, so the compute owner's
    cancellation is cooperative. A cancelled caller drains its own submitted
    work before scratch cleanup and raises a retryable refusal, retaining the
    raw for a later pass. It never closes the shared adapter.
    """
    from polylogue.sources.revision_backfill import RetainedPreparationRetryableError

    future = operation.future
    window = _RETAINED_PREPARATION_STALL_REPORT_SECONDS
    step = min(window, _RETAINED_PREPARATION_CANCEL_POLL_SECONDS)

    def refuse_if_cancelled() -> None:
        if compute_cancel_requested() or operation.cancellation.cancelled:
            operation.cancellation.cancel()
            # Running threads retain scratch ownership until they settle.
            # Cooperative checkpoints observe the compute owner's event.
            with contextlib.suppress(BaseException):
                future.result()
            raise RetainedPreparationRetryableError(f"retained preparation cancelled for {subject}")

    waited = 0.0
    unreported = 0.0
    while True:
        # Checked before each wait and before a finished result is accepted,
        # so a run of fast preparations cannot carry a cancelled pass on.
        refuse_if_cancelled()
        result: object
        try:
            result = future.result(timeout=step)
        except TimeoutError:
            # A future that finished between the wait expiring and this
            # check is re-read for its value, or for the worker's exception.
            result = future.result() if future.done() else _STILL_RUNNING
        if result is not _STILL_RUNNING:
            refuse_if_cancelled()
            return cast(T, result)
        waited += step
        unreported += step
        if unreported >= window:
            unreported = 0.0
            emit(
                "storage.raw_observation.preparation_stalled",
                level=WARNING,
                outcome="degraded",
                reason="no_result_in_window",
                subject_kind=subject.split(" ", 1)[0],
                wait_ms=round(waited * 1000),
            )


if TYPE_CHECKING:
    from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
    from polylogue.core.sql_settlement import SQLCustodyOwner
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.sources.revision_backfill import (
        PreparedMembershipReplay,
        PreparedRetainedAggregate,
        PreparedRetainedInput,
        PreparedRetainedReplaySource,
        PreparedRevisionSourceCensus,
        PreparedRevisionSourceClassification,
        ReplaySchedule,
        RetainedArtifactPreparer,
    )
    from polylogue.storage.index_generation import IndexGeneration
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        PreparedRevisionAdoption,
        PreparedRevisionReplayOutcome,
    )
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead, PreparedSessionWrite
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
