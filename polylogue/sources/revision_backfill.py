"""Prepare retained revisions and publish captured authority decisions."""

from __future__ import annotations

import contextvars
import dataclasses
import hashlib
import json
import os
import pickle
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from builtins import BaseExceptionGroup, ExceptionGroup
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, closing, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from io import BytesIO
from itertools import chain, groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Final, Literal, Protocol, cast

import ijson

from polylogue import logging as _polylogue_logging
from polylogue.archive.artifact_taxonomy import (
    ArtifactClassification,
    ArtifactKind,
    ArtifactStreamClassification,
    classify_artifact_stream,
    declared_evidence_classification,
)
from polylogue.archive.ingest_flags import (
    COMPACT_BROWSER_CAPTURE_INGEST_FLAG,
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
)
from polylogue.archive.revision_authority import (
    BYTE_AUTHORITY_CENSUS_DETAIL,
    HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
    SUPERSEDED_IDENTITY_GOVERNANCE_DETAIL,
    RawRevisionAuthority,
    RawRevisionKind,
    canonical_authority_logical_key,
    is_work_event_raw_id,
    parser_census_identity_measurement,
)
from polylogue.archive.revision_replay import RevisionReplayPlan
from polylogue.archive.session_revision_membership import (
    MembershipClassification,
    MembershipDecision,
    MembershipRevision,
    classify_membership_revisions,
)
from polylogue.core.binary_signatures import looks_like_sqlite_bytes
from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import check_compute_cancelled, compute_cancel_requested
from polylogue.core.enums import ArtifactSupportStatus, PolylogueStrEnum, Provider, ValidationMode
from polylogue.core.json import JSONValue
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
from polylogue.core.raw_failure_evidence import (
    CohortMembershipRefusalError,
    MissingProfileIdentityError,
    RawFailureEvidenceKind,
    RetainedZipMembershipUnprovedError,
)
from polylogue.core.sources import origin_from_provider
from polylogue.core.timestamp_authority import normalize_session_timestamps
from polylogue.pipeline.ids import SessionRevisionProjection, session_revision_projection
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.sources.artifact_observations import record_session_artifact_observation
from polylogue.sources.assembly import SidecarData
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import (
    BUNDLE_PROVIDERS,
    admit_parsed_sessions_for_publication,
    detect_provider_from_stream_evidence,
    is_jsonl_source_path,
    is_stream_record_provider,
    parse_payload,
    parse_stream_payload,
)
from polylogue.sources.fallback_identity import fallback_session_id
from polylogue.sources.live.batch_support import (
    jsonl_complete_prefix,
    jsonl_parse_input_of_handle,
    jsonl_parse_prefix_size,
    jsonl_parse_prefix_size_of_handle,
)
from polylogue.sources.origin_specs import path_declaration_refuses_session
from polylogue.sources.parsers import antigravity, codex_state, hermes_state, hermes_verification
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.pickle_spool import PickleSpool
from polylogue.sources.prepared_jsonl import (
    DecodeFailure,
    PreparedDecodeError,
    PreparedJsonl,
    PreparedSessionSequence,
    _iter_prefix_lines,
    classify_decode_failure,
    prepare_jsonl_blob,
    terminal_decode_evidence,
)
from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteMessageStore
from polylogue.sources.retained_sqlite import collect_sqlite_sessions, iter_sqlite_sessions
from polylogue.sources.sidecar_evidence import SidecarResolver
from polylogue.sources.sqlite_export import looks_like_logical_source_bytes, looks_like_logical_source_path
from polylogue.sources.sqlite_snapshot import (
    is_declared_logical_export,
    is_sqlite_page_image,
    is_undeclared_logical_export,
)
from polylogue.storage.artifacts.inspection import artifact_observation_id
from polylogue.storage.blob_publication import BlobPublicationSourceRead
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.raw_authority import (
    iter_parser_census_logical_keys,
    raw_authority_parser_fingerprint,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.revision_governance import (
    ArchiveRawParsedWriteResult,
    MembershipHeadPlan,
    PreparedRawClassificationStaleError,
    PreparedRevisionAdoption,
    PreparedRevisionReplayOutcome,
    _raw_parse_failure_state,
    _raw_parse_success_state,
    _record_raw_failure_evidence,
    record_current_parser_source_census,
    record_prepared_membership_census_receipt,
)
from polylogue.storage.sqlite.archive_tiers.source_write import (
    PENDING_RAW_LOGICAL_SOURCE_PREFIX,
    ArchiveSourceArtifact,
    SourceArtifactProducer,
    SourceRawStateProducer,
    _apply_source_raw_state_update,
    _upsert_raw_artifact,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    ConnectionSessionSourceRead,
    PreparedSessionSourceRead,
    PreparedSessionWrite,
    PreparedSessionWriteRefusedError,
    prepare_session_shard,
)
from polylogue.storage.sqlite.connection_profile import (
    StaleContinuationError,
    read_frame,
)
from polylogue.storage.sqlite.reference_seal import ReferenceSealError
from polylogue.storage.sqlite.session_shard import discard_session_shard

_LOGGER = _polylogue_logging.get_logger(__name__)


def _resolved_retained_provider(payload: BinaryIO, source_path: str) -> tuple[Provider, str]:
    """Classify exactly the retained input accepted by its parser boundary."""
    jsonl = is_jsonl_source_path(source_path)
    scope = jsonl_parse_input_of_handle(payload, check_stop=check_compute_cancelled) if jsonl else nullcontext(payload)
    try:
        with scope as accepted:
            if jsonl:
                start = accepted.tell()
                first = accepted.read(1)
                accepted.seek(start)
                if not first:
                    return Provider.UNKNOWN, "no accepted complete JSONL records"
            provider, evidence = detect_provider_from_stream_evidence(
                accepted,
                check_stop=check_compute_cancelled,
            )
    except (ijson.JSONError, UnicodeError, json.JSONDecodeError) as error:
        kind = DecodeFailure.JSONL_RECORD if jsonl else DecodeFailure.DOCUMENT
        raise PreparedDecodeError(kind, str(error)) from error
    return (Provider.UNKNOWN if provider is None else Provider(provider)), evidence


def _resolved_retained_member_provider(
    evidence_reader: RetainedRawRead, raw_id: str, payload: BinaryIO, source_path: str
) -> tuple[Provider, str]:
    """Resolve an UNKNOWN retained raw by its own shape, then by its ZIP container.

    Source-only acquisition retains every ZIP member as UNKNOWN. A member's own
    records name its provider when they can; a sibling that no detector
    claims (a GDPR export's ``message_feedback.json``) takes the container
    hint, exactly as retained enumeration of the same physical ZIP assigns it
    through ``zip_member_admission`` (polylogue-hs3y).
    """
    provider, evidence = _resolved_retained_provider(payload, source_path)
    if provider is not Provider.UNKNOWN:
        return provider, evidence
    coordinate = evidence_reader.raw_captured_zip_coordinate(raw_id)
    if coordinate is None:
        return provider, evidence
    from polylogue.sources.source_acquisition_components import zip_member_admission

    with evidence_reader.open_raw_container_material(raw_id) as physical:
        if physical is None:
            raise RetainedPreparationRetryableError(f"retained ZIP container bytes are absent for raw {raw_id}")
        with zipfile.ZipFile(physical) as archive:
            entries = archive.infolist()
            if coordinate.entry_ordinal >= len(entries) or (
                entries[coordinate.entry_ordinal].filename != coordinate.member_name
            ):
                raise RetainedZipMembershipUnprovedError("retained ZIP member differs from its container receipt")
            with zip_member_admission(
                archive,
                Path(coordinate.declared_container),
                entries,
                Provider.UNKNOWN,
                container_blob_hash=coordinate.container_blob_hash,
            ) as admission:
                container_provider = admission.entry_provider_hint(
                    entries[coordinate.entry_ordinal], entry_ordinal=coordinate.entry_ordinal
                )
    if container_provider is Provider.UNKNOWN:
        return provider, evidence
    return container_provider, f"zip container member admission: {container_provider.value}"


def _expand_frozen_revision_link_selection(archive_root: Path, raw_ids: Sequence[str]) -> tuple[str, ...]:
    """Include every predecessor and baseline needed to validate selected APPEND authority."""
    expanded = set(raw_ids)
    pending = set(raw_ids)
    with read_frame(
        archive_root / "source.db", tier=ArchiveTier.SOURCE, timeout_class="background-read"
    ) as source_frame:
        source_conn = source_frame.connection
        while pending:
            current = tuple(sorted(pending))
            pending.clear()
            for offset in range(0, len(current), 500):
                chunk = current[offset : offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = source_conn.execute(
                    f"""
                    SELECT predecessor_raw_id, baseline_raw_id
                    FROM raw_sessions WHERE raw_id IN ({placeholders})
                    """,
                    chunk,
                )
                for predecessor_raw_id, baseline_raw_id in rows:
                    for linked_raw_id in (predecessor_raw_id, baseline_raw_id):
                        if linked_raw_id is not None and str(linked_raw_id) not in expanded:
                            expanded.add(str(linked_raw_id))
                            pending.add(str(linked_raw_id))
    return tuple(sorted(expanded))


def _browser_snapshot_fidelity(ingest_flags: Sequence[str]) -> Literal["dom", "native"] | None:
    """Derive membership-classification browser fidelity from parser ingest flags.

    ``session_revision_membership.classify_membership_revisions`` special-cases
    dom-fallback vs. native browser captures (and, since polylogue-z1c6, a
    genuine non-browser-capture revision outranking any browser capture) --
    but only when ``MembershipRevision.browser_snapshot_fidelity`` is actually
    populated. A plain provider export carries neither flag and is
    ``None`` (not a browser capture at all).
    """
    flags = set(ingest_flags)
    if NATIVE_BROWSER_CAPTURE_INGEST_FLAG in flags or COMPACT_BROWSER_CAPTURE_INGEST_FLAG in flags:
        return "native"
    if DOM_FALLBACK_INGEST_FLAG in flags:
        return "dom"
    return None


@dataclass(frozen=True, slots=True)
class PreparedRevisionReplayResult:
    """Authority and index effects of one prepared publication window."""

    scanned: int
    classified_full: int
    replayed_logical_sources: int
    quarantined: int
    adoption_deferred: int = 0
    # Publication timings contain source census and writer effects. Parsing
    # belongs to Raw.compute and is not inferred from a publication timer.
    stage_timings_s: dict[str, float] = field(default_factory=dict, compare=False)
    stage_counts: dict[str, int] = field(default_factory=dict, compare=False)
    written_session_ids: tuple[str, ...] = ()
    changed_session_ids: tuple[str, ...] = ()
    # Original writer effects are orthogonal to transcript hash changes,
    # especially for idempotent event-only Raw publications.
    writer_changed_raw_ids: tuple[str, ...] = ()
    written_message_count: int = 0
    written_counts: dict[str, int] = field(default_factory=dict, compare=False)
    session_outputs: tuple[tuple[str, bytes | None, bytes | None, int], ...] = ()
    membership_refusals: tuple[tuple[str, str, MembershipDecision], ...] = ()


@dataclass(frozen=True, slots=True)
class RetainedRawRetryableFailure:
    """One raw whose retained preparation failed retryably within a replay page.

    ``error`` is the original failure, kept as evidence. The raw's bytes stay
    retained and unpublished, so its next pass (the live cursor's retry, or
    fair intake) prepares it again.
    """

    raw_id: str
    error: Exception


@dataclass(frozen=True, slots=True)
class RetainedReplayOutcome:
    """A retained replay page's receipts and its raws' retryable failures.

    A failing raw does not stop its page: the receipts its siblings published
    are returned beside the failures, so a caller can account for both.
    """

    receipts: tuple[PreparedRevisionReplayResult, ...] = ()
    failures: tuple[RetainedRawRetryableFailure, ...] = ()

    def require_complete(self) -> tuple[PreparedRevisionReplayResult, ...]:
        """The receipts, or the page's failures raised for a caller that admits no partial page."""
        if len(self.failures) == 1:
            raise self.failures[0].error
        if self.failures:
            raise ExceptionGroup(
                "retained replay raws failed preparation", [failure.error for failure in self.failures]
            )
        return self.receipts


_REPLAY_ENRICHMENT_DEGRADATIONS: contextvars.ContextVar[Counter[str] | None] = contextvars.ContextVar(
    "replay_enrichment_degradations", default=None
)
_REPLAY_ENRICHMENT_DEGRADATIONS_LOCK = threading.Lock()


def _capture_replay_enrichment_degradations(
    function: Callable[..., PreparedRevisionReplayResult],
) -> Callable[..., PreparedRevisionReplayResult]:
    """Give one historical backfill and its workers a private degradation ledger."""

    @wraps(function)
    def wrapped(*args: object, **kwargs: object) -> PreparedRevisionReplayResult:
        counts: Counter[str] = Counter()
        token = _REPLAY_ENRICHMENT_DEGRADATIONS.set(counts)
        try:
            result = function(*args, **kwargs)
            result.stage_counts.update(
                {f"replay_enrichment_degraded.{reason}": count for reason, count in counts.items()}
            )
            return result
        finally:
            _REPLAY_ENRICHMENT_DEGRADATIONS.reset(token)

    return wrapped


@dataclass(frozen=True, slots=True)
class RevisionCensusResult:
    scanned: int
    classified_full: int
    quarantined: int
    input_raw_ids: tuple[str, ...]
    logical_keys: tuple[str, ...]


@dataclass(slots=True)
class _RevisionCensusState:
    scanned: int
    classified: int
    quarantined: int
    censused: set[str]
    membership_candidates: dict[str, set[str]]
    transient_non_session_raw_ids: set[str]


class RetainedPreparationRetryableError(RuntimeError):
    """A supplied retained parse cannot be trusted; retry without quarantining bytes."""


class RetainedPreparationNoProgressError(RuntimeError):
    """A preparatory Source phase committed without changing its durable inputs.

    Preparing again would read the same state and commit the same phase, so
    this is a terminal outcome for the component, never a reason to retry.
    """

    code = "retained_phase_no_progress"


class UnsupportedRetainedJsonShapeError(ValueError):
    """Bounded retained detection found no supported provider for textual JSON."""


@dataclass(frozen=True, slots=True)
class RetainedParseFailure:
    """One retained parse failure, carried across a worker boundary with its decode kind.

    A decode exception does not survive the boundary with its type (a
    ``JsonlDecodeError`` cannot be rebuilt from its message), so the carrier
    names the decode kind and the consumer rebuilds a typed exception.
    """

    detail: str
    decode_failure: DecodeFailure | None = None
    missing_profile_identity: bool = False
    retained_zip_membership_unproved: bool = False

    @classmethod
    def of(cls, error: BaseException) -> RetainedParseFailure:
        return cls(
            str(error),
            classify_decode_failure(error),
            isinstance(error, MissingProfileIdentityError),
            isinstance(error, RetainedZipMembershipUnprovedError),
        )

    def as_exception(self) -> Exception:
        if self.retained_zip_membership_unproved:
            return RetainedZipMembershipUnprovedError(self.detail)
        if self.missing_profile_identity:
            return MissingProfileIdentityError(self.detail)
        return retained_parse_exception(self.detail, self.decode_failure)


def retained_parse_exception(detail: str, decode_failure: DecodeFailure | None) -> Exception:
    """The typed exception a carried retained parse failure stands for."""
    if decode_failure is not None:
        return PreparedDecodeError(decode_failure, detail)
    return RuntimeError(detail)


def _retained_jsonl_records(payload: bytes, source_name: str, source_path: str) -> list[JSONValue]:
    """Decode retained bytes as live intake decodes the same capture.

    Every complete JSONL record must decode (``fail_on_decode_error``); only
    an unterminated tail -- an append in progress -- is left out, by the same
    rule the live path worker applies (``jsonl_parse_prefix_size``).
    """
    if is_jsonl_source_path(source_path):
        prefix_size = jsonl_parse_prefix_size(jsonl_complete_prefix(payload), len(payload))
        if prefix_size is not None:
            payload = payload[:prefix_size]
    return list(_iter_json_stream(BytesIO(payload), source_name, fail_on_decode_error=True))


def _retained_jsonl_stream(payload: BinaryIO, source_name: str, source_path: str) -> Iterable[JSONValue]:
    """Stream retained records under :func:`_retained_jsonl_records`' rule."""
    if not is_jsonl_source_path(source_path):
        return _iter_json_stream(payload, source_name, fail_on_decode_error=True)
    prefix_size = jsonl_parse_prefix_size_of_handle(payload)
    record_input = _iter_prefix_lines(payload, prefix_size) if prefix_size is not None else payload
    return _iter_json_stream(record_input, source_name, fail_on_decode_error=True)


@dataclass(slots=True)
class PreparedRetainedInput:
    raw_id: str
    provider: Provider
    blob_hash: str
    source_path: str
    revision_kind: RawRevisionKind
    payload_bytes: int
    native_id: str | None
    parser_fingerprint: str
    fallback_timestamp: str | None
    verified_blob_stat: tuple[int, int, int, int, int]
    validation_verdict: RetainedValidationVerdict | None
    parser_error: str | None = None
    #: Which decode boundary refused the bytes when ``parser_error`` is a
    #: decode refusal; the census turns that into a terminal outcome.
    parser_decode_failure: DecodeFailure | None = None
    missing_profile_identity: bool = False
    retained_zip_membership_unproved: bool = False
    unsupported_shape: bool = False
    captured_profile_key: str | None = None
    enriched: bool = False
    # The sealed disk-backed carrier is the retained worker publication boundary.
    prepared_artifact: PreparedJsonl | None = None


@dataclass(frozen=True, slots=True)
class PreparedRetainedAggregate:
    """One worker-sealed composition for an exact ordered byte revision chain."""

    raw_ids: tuple[str, ...]
    artifact: PreparedJsonl


def _enrichment_evidence_digest(value: object) -> str:
    """Hash retained assembly inputs without making a second byte-sized copy."""
    digest = hashlib.sha256()

    class DigestWriter:
        def write(self, data: bytes) -> int:
            digest.update(data)
            return len(data)

    from polylogue.sources.retained_title_index import RetainedTitleIndex, title_evidence_digest

    if isinstance(value, dict) and "retained_state_titles" in value:
        titles = value["retained_state_titles"]
        if isinstance(titles, RetainedTitleIndex):
            title_digest = titles.evidence_digest()
        else:
            title_digest = title_evidence_digest(
                sorted(titles.items(), key=lambda row: row[0].encode("utf-8", "surrogatepass"))
            )
        value = {**value, "retained_state_titles": ("retained-title-evidence:v1", title_digest)}

    from polylogue.sources.parsers.chatgpt_sidecars import ChatGPTAssetIndex

    if isinstance(value, dict) and isinstance(value.get("chatgpt_asset_index"), ChatGPTAssetIndex):
        from polylogue.sources.parsers.chatgpt_sidecars import _AssetBlobs

        index = value["chatgpt_asset_index"]
        value = {**value, "chatgpt_asset_index": index.evidence_digest()}
        assets = value.get("chatgpt_asset_blobs")
        if isinstance(assets, _AssetBlobs):
            # A retained supplement can own the missing acquired-member map
            # while an acquisition-carried naming index remains authoritative.
            # Bind each actual artifact without serializing SQL handles.
            value["chatgpt_asset_blobs"] = assets.index.evidence_digest()
    pickle.dump(value, DigestWriter(), protocol=pickle.HIGHEST_PROTOCOL)
    return digest.hexdigest()


def _owned_enrichment_evidence_digest(value: SidecarData) -> str:
    """Digest operation-owned sidecars and settle their artifact lifetime."""
    from polylogue.sources.assembly import close_sidecar_data

    try:
        return _enrichment_evidence_digest(value)
    finally:
        close_sidecar_data(value)


def _retained_parser_sidecar_digest(source_conn: sqlite3.Connection, *, provider: Provider, source_path: str) -> str:
    """Bind a stream parser's retained sibling/tool-result population."""
    digest = hashlib.sha256()
    if provider is not Provider.CLAUDE_CODE:
        return digest.hexdigest()
    from polylogue.sources.live.tool_result_sidecars import resolve_tool_results_dir

    directory = resolve_tool_results_dir(source_path)
    if directory is None:
        return digest.hexdigest()
    path = Path(source_path)
    session_dir = path.parent.parent if path.parent.name == "subagents" else path.parent / path.stem
    root_path = session_dir.parent / f"{session_dir.name}.jsonl"
    tool_prefix = f"{directory.as_posix()}/"
    sibling_prefix = f"{(session_dir / 'subagents').as_posix()}/"
    # The sibling resolver distinguishes only an append revision; a census
    # that types an unknown revision as full does not move this evidence.
    for row in source_conn.execute(
        "SELECT source_path, raw_id, hex(blob_hash), blob_size, file_mtime_ms, "
        "revision_kind = 'append', acquired_at_ms FROM raw_sessions "
        "WHERE source_path = ? OR (source_path >= ? AND source_path < ?) "
        "OR (source_path >= ? AND source_path < ?) ORDER BY source_path, raw_id",
        (
            root_path.as_posix(),
            tool_prefix,
            tool_prefix + "\uffff",
            sibling_prefix,
            sibling_prefix + "\uffff",
        ),
    ):
        encoded = repr(tuple(row)).encode("utf-8", "surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _retained_dependency_digest(assembly_digest: str | None, parser_sidecars_digest: str) -> str:
    return hashlib.sha256(f"{assembly_digest or ''}:{parser_sidecars_digest}".encode("ascii")).hexdigest()


def enrichment_dependency_digest(
    *,
    provider: Provider,
    source_path: str,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
    provider_session_ids: Iterable[str],
    index_conn: sqlite3.Connection | None,
    source_conn: sqlite3.Connection | None,
    blob_root: Path | None,
    parser_sidecars: bool,
) -> str:
    """Digest every archive input a prepared session's enrichment depends on.

    The preparing worker and the writer compute this from their own read
    views; a mismatch means the evidence moved between preparation and
    publication and the prepared interpretation is stale. ``parser_sidecars``
    also binds retained tool-result siblings, which only a retained parse
    reads; a live parse reads them from the source tree.
    ``provider_session_ids`` is consumed only by a provider whose evidence is
    keyed by session (``_replay_enrichment_reads_index``).
    """
    # Every provider digests its (possibly empty) evidence, as the retained
    # seals do: gating this on an assembly spec made the writer's value for a
    # provider without one differ from the sealed value, so every prepared
    # artifact of that provider read as stale and never published.
    assembly_digest = _owned_enrichment_evidence_digest(
        _retained_enrichment_sidecar_data(
            provider=provider,
            sessions=(),
            provider_session_ids=provider_session_ids,
            evidence_reader=ConnectionRetainedEnrichmentRead(index_conn, source_conn, blob_root),
            source_path=source_path,
            captured_zip_coordinate=captured_zip_coordinate,
        )
    )
    parser_digest = (
        _retained_parser_sidecar_digest(source_conn, provider=provider, source_path=source_path)
        if parser_sidecars and source_conn is not None
        else ""
    )
    return _retained_dependency_digest(assembly_digest, parser_digest)


#: Providers whose enrichment reads session-scoped retained evidence that can
#: arrive after the session it describes: a Claude Code project's
#: ``sessions-index.json`` and the install's prompt history, and Codex's
#: session index, history and projected thread-state titles. Export bundles
#: (ChatGPT asset maps) arrive with the export they describe.
_SESSION_EVIDENCE_PROVIDERS = frozenset({Provider.CLAUDE_CODE, Provider.CODEX})


def provider_binds_enrichment(provider: Provider) -> bool:
    """Whether sessions of ``provider`` carry a late-arriving enrichment binding."""
    return provider in _SESSION_EVIDENCE_PROVIDERS


def _evidence_json(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    raise TypeError(f"unencodable enrichment evidence: {type(value).__name__}")


def enrichment_evidence_key(provider: Provider, sidecar_data: SidecarData, native_id: str) -> str | None:
    """Digest the evidence ``enrich_session`` reads for exactly one session.

    The projection is this session's own index entry and prompt-history rows
    (Claude Code), or its thread name, history title and state titles (Codex),
    so an unrelated session's evidence moving never marks this one. Canonical
    JSON, not pickle, so a worker process and the writer compute equal keys
    for equal evidence. ``None``: the provider has no late-arriving evidence.
    """
    if provider not in _SESSION_EVIDENCE_PROVIDERS or not native_id:
        return None
    projection: tuple[object, ...]
    if provider is Provider.CLAUDE_CODE:
        projection = (
            provider.value,
            sidecar_data.get("session_index", {}).get(native_id),
            sidecar_data.get("history_paste_index", {}).get(native_id),
        )
    else:
        projection = (
            provider.value,
            *(
                cast("Mapping[str, str]", sidecar_data.get(name) or {}).get(native_id)
                for name in ("thread_names", "history_titles", "state_titles", "retained_state_titles")
            ),
        )
    encoded = json.dumps(projection, default=_evidence_json, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8", "surrogatepass")).hexdigest()


def stamp_enrichment_evidence(provider: Provider, sidecar_data: SidecarData, session: ParsedSession) -> ParsedSession:
    """Carry the key of the evidence ``session`` was just enriched from."""
    key = enrichment_evidence_key(provider, sidecar_data, session.provider_session_id.strip())
    if key is None:
        return session
    return session.model_copy(update={"enrichment_evidence_key": key})


def session_enrichment_evidence_key(
    *,
    provider: Provider,
    source_path: str | None,
    native_id: str,
    index_conn: sqlite3.Connection | None,
    source_conn: sqlite3.Connection | None,
    blob_root: Path | None,
) -> str | None:
    """The key of the evidence the archive holds now for one stored session.

    Resolved exactly as retained replay resolves its enrichment evidence, so
    it equals the key a session enriched from that evidence carries.
    """
    return session_enrichment_evidence_key_from_reader(
        provider=provider,
        source_path=source_path,
        native_id=native_id,
        evidence_reader=ConnectionRetainedEnrichmentRead(index_conn, source_conn, blob_root),
    )


def record_session_enrichment_binding(
    index_conn: sqlite3.Connection,
    *,
    session_id: str,
    carried_key: str | None,
    current_key: str | None,
) -> None:
    """Bind a just-published session to the evidence it was enriched from.

    Called by the writer after the session row is written, in the same
    transaction. The session carries the key of the evidence its enrichment
    read; it is bound only when that is still the archive's evidence. A
    session enriched before its evidence arrived (or moved) stays unbound, so
    inspection re-derives it on the retained route instead of certifying it.
    """
    if current_key is None or carried_key != current_key:
        return
    index_conn.execute(
        """INSERT INTO session_enrichment_bindings (session_id, evidence_key) VALUES (?, ?)
        ON CONFLICT(session_id) DO UPDATE SET evidence_key = excluded.evidence_key""",
        (session_id, current_key),
    )


def prepared_enrichment_dependency_state(
    archive: Any,
    artifact: PreparedJsonl,
    *,
    provider: Provider,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
    source_path: str,
    sessions: Iterable[ParsedSession],
    parser_sidecars: bool,
) -> str | None:
    """Return why a prepared artifact's enrichment is stale, or ``None``.

    The writer publishes into ``archive``; the artifact is current only when
    it was enriched against that same index and the same evidence.
    """
    if artifact.enrichment_index_path != str(Path(archive.index_db_path).resolve()):
        return "index dependency changed"
    if artifact.enrichment_digest is None:
        return None
    current = enrichment_dependency_digest(
        provider=provider,
        source_path=source_path,
        captured_zip_coordinate=captured_zip_coordinate,
        provider_session_ids=(session.provider_session_id for session in sessions if session.provider_session_id),
        index_conn=archive.index_connection,
        source_conn=archive._ensure_source_conn(),
        blob_root=Path(archive.archive_root) / "blob",
        parser_sidecars=parser_sidecars,
    )
    return None if current == artifact.enrichment_digest else "enrichment evidence changed"


def prepare_retained_jsonl_artifact(
    evidence_reader: RetainedSessionRead,
    raw_id: str,
    *,
    directory: Path,
    allow_generic_object_alias: bool = False,
    validation_mode: ValidationMode = ValidationMode.ADVISORY,
    schema_registry: SchemaRegistry | None = None,
) -> PreparedJsonl:
    """Seal JSON sessions using this creator's actual selected Source inputs.

    The admitted parent owns the original witness/read window through
    publication and physical discard. This producer never opens another
    Source or Index generation and never substitutes a digest for currency.
    """
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    provider, blob_hash, source_path, kind, _size = evidence_reader.raw_revision_descriptor(raw_id)
    if not (
        is_jsonl_source_path(source_path)
        or Path(source_path).suffix.lower() == ".json"
        or path_declaration_refuses_session(provider, source_path)
        or allow_generic_object_alias
    ):
        raise RetainedPreparationRetryableError(f"retained JSON worker cannot parse {raw_id}")
    blob_path = evidence_reader.raw_revision_blob_path(raw_id)
    if blob_path is None:
        raise RetainedPreparationRetryableError(f"retained JSON bytes are absent for raw {raw_id}")
    if provider is Provider.UNKNOWN:
        try:
            with evidence_reader.open_raw_revision_material(raw_id) as (_provider, payload, _path, _kind):
                provider, _evidence = _resolved_retained_member_provider(evidence_reader, raw_id, payload, source_path)
        except PreparedDecodeError as exc:
            return PreparedJsonl(
                blob_hash,
                None,
                None,
                f"{type(exc).__name__}: {exc}",
                decode_failure=exc.kind,
            )
        if provider is Provider.UNKNOWN:
            refusal = UnsupportedRetainedJsonShapeError(
                f"retained UNKNOWN provider has no recognized complete input shape: {source_path}"
            )
            return PreparedJsonl(None, None, None, f"{type(refusal).__name__}: {refusal}", unsupported_shape=True)
    fallback_id = fallback_session_id(source_path, raw_id)
    if kind is RawRevisionKind.APPEND:
        fallback_id = (
            _append_session_native_id(
                evidence_reader.raw_append_logical_key(raw_id),
                provider=provider,
                captured_native_id=evidence_reader.raw_native_id(raw_id),
            )
            or fallback_id
        )
    profile_identity = evidence_reader.raw_profile_identity(raw_id)
    captured_zip_coordinate = evidence_reader.raw_captured_zip_coordinate(raw_id)
    fallback_timestamp = evidence_reader.raw_revision_file_mtime(raw_id)
    if provider is Provider.HERMES and profile_identity is None:
        return PreparedJsonl(
            blob_hash,
            None,
            None,
            "retained Hermes input has no captured profile identity receipt",
            missing_profile_identity=True,
        )
    from polylogue.sources.assembly import close_sidecar_data, get_assembly_spec

    sidecar_data_cache: SidecarData = {}
    sidecar_data_loaded = False
    prepare_per_session = (
        not is_stream_record_provider(source_path, provider)
        and provider in BUNDLE_PROVIDERS
        and Path(source_path).name.lower().endswith(".json")
    )

    def finalize(sessions: PreparedSessionSequence) -> Iterator[ParsedSession]:
        return iter_enriched_sessions_from_retained_read(
            evidence_reader=evidence_reader,
            provider=provider,
            sessions=sessions,
            source_path=source_path,
            captured_zip_coordinate=captured_zip_coordinate,
            provider_session_ids=sessions.iter_provider_session_ids(),
            normalize_session=lambda session: normalize_session_timestamps(
                session, fallback_timestamp=fallback_timestamp
            ),
        )

    def prepare_bundle_session(session: ParsedSession) -> ParsedSession:
        nonlocal sidecar_data_loaded
        check_compute_cancelled()
        normalized = normalize_session_timestamps(session, fallback_timestamp=fallback_timestamp)
        spec = get_assembly_spec(provider)
        if spec is None:
            return normalized
        if not sidecar_data_loaded:
            sidecar_data_cache.update(
                _retained_enrichment_sidecar_data(
                    provider=provider,
                    sessions=(),
                    evidence_reader=evidence_reader,
                    source_path=source_path,
                    captured_zip_coordinate=captured_zip_coordinate,
                )
            )
            sidecar_data_loaded = True
        return spec.enrich_session(normalized, sidecar_data_cache)

    try:
        parse_prefix_size: int | None = None
        if is_jsonl_source_path(source_path) and not path_declaration_refuses_session(provider, source_path):
            with evidence_reader.open_raw_revision_material(raw_id) as (_provider, payload, _path, _kind):
                try:
                    parse_prefix_size = jsonl_parse_prefix_size_of_handle(payload)
                except Exception as exc:
                    decode_failure = classify_decode_failure(exc)
                    if decode_failure is None:
                        raise
                    return PreparedJsonl(
                        blob_hash,
                        None,
                        None,
                        f"{type(exc).__name__}: {exc}",
                        decode_failure=decode_failure,
                    )
            if _size == 0 and _is_declared_provider_session_stream(provider, source_path):
                return PreparedJsonl(
                    blob_hash,
                    None,
                    None,
                    "zero-byte provider session stream contains no decodable session record",
                    decode_failure=DecodeFailure.JSONL_RECORD,
                )
        artifact = prepare_jsonl_blob(
            str(blob_path),
            source_path,
            provider.value,
            fallback_id,
            is_stream=is_stream_record_provider(source_path, provider),
            profile_identity=profile_identity,
            shard_directory=str(directory),
            publication_publisher=ArchiveBlobPublisher(
                evidence_reader.archive_root / "source.db",
                evidence_reader.archive_root / "blob",
            ),
            publication_source_read=evidence_reader,
            strict_jsonl_records=True,
            parse_prefix_size=parse_prefix_size,
            source_sha256=blob_hash,
            sidecar_resolver=evidence_reader.retained_sidecar_resolver(),
            prepare_session=prepare_bundle_session if prepare_per_session else None,
            prepare_sessions=None if prepare_per_session else finalize,
            captured_zip_coordinate=captured_zip_coordinate,
        )
    except (OSError, sqlite3.OperationalError) as exc:
        raise RetainedPreparationRetryableError(f"retained JSON evidence read failed for raw {raw_id}") from exc
    finally:
        primary = sys.exception()
        try:
            close_sidecar_data(sidecar_data_cache)
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup(
                    "retained JSON preparation and sidecar cleanup failed", [primary, cleanup]
                ) from None
            raise
    if artifact.blob_hash is not None and artifact.blob_hash != blob_hash:
        blob_refusal = RetainedPreparationRetryableError(f"retained JSON blob changed for raw {raw_id}")
        try:
            artifact.discard()
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "retained JSON refusal and artifact cleanup failed", [blob_refusal, cleanup]
            ) from None
        raise blob_refusal
    return _attach_retained_validation_verdict(
        artifact,
        provider=provider,
        blob_hash=blob_hash,
        blob_path=blob_path,
        source_path=source_path,
        raw_id=raw_id,
        directory=directory,
        validation_mode=validation_mode,
        captured_zip_coordinate=captured_zip_coordinate,
        jsonl=is_jsonl_source_path(source_path),
        schema_registry=schema_registry,
    )


@contextmanager
def _retained_validation_input(
    blob_path: Path,
    prefix_size: int | None,
    directory: Path,
) -> Iterator[Path]:
    """Expose exactly the parsed JSONL frontier to the spill-backed validator."""
    if prefix_size is None or prefix_size == blob_path.stat().st_size:
        yield blob_path
        return
    if prefix_size < 0:
        raise RetainedPreparationRetryableError("retained JSONL parser returned a negative prefix")
    fd, raw_path = tempfile.mkstemp(prefix="validation-", suffix=".jsonl", dir=directory)
    path = Path(raw_path)
    try:
        remaining = prefix_size
        with os.fdopen(fd, "wb") as target, blob_path.open("rb") as source:
            while remaining:
                check_compute_cancelled()
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RetainedPreparationRetryableError("retained JSONL parser prefix exceeds source bytes")
                target.write(chunk)
                remaining -= len(chunk)
        yield path
    finally:
        path.unlink(missing_ok=True)


def _attach_retained_validation_verdict(
    artifact: PreparedJsonl,
    *,
    provider: Provider,
    blob_hash: str,
    blob_path: Path,
    source_path: str,
    raw_id: str,
    directory: Path,
    validation_mode: ValidationMode,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
    jsonl: bool,
    schema_registry: SchemaRegistry | None,
) -> PreparedJsonl:
    """Bind schema evidence to the exact retained bytes the parser consumed."""
    if (
        artifact.error is None
        and artifact.resolved_provider is not None
        and not path_declaration_refuses_session(provider, source_path)
        and not (jsonl and artifact.parsed_prefix_size == 0)
    ):
        from polylogue.schemas import validate_retained_document

        validation_prefix = artifact.parsed_prefix_size if jsonl and validation_mode is not ValidationMode.OFF else None
        try:
            with _retained_validation_input(blob_path, validation_prefix, directory) as validation_path:
                verdict = validate_retained_document(
                    artifact.resolved_provider,
                    validation_path,
                    mode=validation_mode,
                    raw_id=raw_id,
                    revision_sha256=blob_hash,
                    evidence_id=raw_id,
                    source_path=source_path,
                    jsonl=jsonl,
                    captured_zip_coordinate=captured_zip_coordinate,
                    registry=schema_registry,
                )
            return dataclasses.replace(artifact, validation_verdict=verdict)
        except BaseException as primary:
            try:
                artifact.discard()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "retained validation and artifact cleanup failed", [primary, cleanup]
                ) from None
            raise
    return artifact


def _is_declared_provider_session_stream(provider: Provider, source_path: str) -> bool:
    """Whether an empty provider JSONL path should be decoded as a session stream."""
    if provider not in {Provider.CODEX, Provider.CLAUDE_CODE}:
        return False
    # Claude's configured history file is raw-only intake metadata. Other
    # provider JSONL paths that are not explicitly excluded by OriginSpec are
    # session decode inputs, including neutral and exported path spellings.
    return not (provider is Provider.CLAUDE_CODE and Path(source_path).name.lower() == "history.jsonl")


def prepare_retained_non_json_artifact(
    evidence_reader: RetainedSessionRead,
    raw_id: str,
    *,
    directory: Path,
    validation_mode: ValidationMode = ValidationMode.ADVISORY,
    schema_registry: SchemaRegistry | None = None,
) -> PreparedJsonl:
    """Seal non-JSON sessions through the same original retained read owner."""
    from polylogue.core.compute import DaemonBackpressureError
    from polylogue.core.prepared_file import VerificationCancelledError
    from polylogue.pipeline.ids import session_content_hash
    from polylogue.sources.prepared_jsonl import (
        PreparedJsonl,
        _prepare_attachment_publications,
        _prepare_codex_state_blob,
        _prepare_sidecar_publications,
        _write_artifact,
        record_prepared_classification,
    )
    from polylogue.sources.prepared_message_sink import SqliteMessageStore
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.blob_store import BlobStore, BlobVerificationCancelledError
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    provider, blob_hash, source_path, _kind, _size = evidence_reader.raw_revision_descriptor(raw_id)
    # Some providers arrive under neutral or mislabeled filenames. A simple
    # top-level message envelope has an existing ijson-to-SQLite route in
    # prepare_jsonl_blob; route by that complete shape before the collecting
    # non-JSON replay below. The probe validates through EOF and leaves the
    # retained blob untouched.
    if provider in {Provider.DRIVE, Provider.GEMINI} and not (
        is_jsonl_source_path(source_path) or Path(source_path).suffix.lower() == ".json"
    ):
        from polylogue.sources.decoder_json import generic_message_object_envelope

        blob_path = evidence_reader.raw_revision_blob_path(raw_id)
        if blob_path is not None:
            with blob_path.open("rb") as handle:
                generic_envelope = generic_message_object_envelope(handle)
            if generic_envelope is not None:
                return prepare_retained_jsonl_artifact(
                    evidence_reader,
                    raw_id,
                    directory=directory,
                    allow_generic_object_alias=True,
                    validation_mode=validation_mode,
                    schema_registry=schema_registry,
                )
    if path_declaration_refuses_session(provider, source_path):
        # A raw-only member (an export's binary asset) is evidence whatever its
        # suffix: the sealed preparation records its path classification
        # without decoding the bytes, exactly as for a JSON-suffixed member.
        return prepare_retained_jsonl_artifact(
            evidence_reader,
            raw_id,
            directory=directory,
            validation_mode=validation_mode,
            schema_registry=schema_registry,
        )
    publisher = ArchiveBlobPublisher(
        evidence_reader.archive_root / "source.db",
        evidence_reader.archive_root / "blob",
    )
    sessions_path = Path(directory) / f"prepared-{uuid.uuid4().hex}.db"
    shard_path: Path | None = None
    store: SqliteMessageStore | None = None
    sealed = False
    parsed = False
    try:
        state_descriptor = _retained_codex_state_descriptor(evidence_reader, raw_id)
        if not BlobStore(evidence_reader.archive_root / "blob").verify(blob_hash, stop=compute_cancel_requested):
            raise RetainedPreparationRetryableError(f"retained blob changed for raw {raw_id}")
        if state_descriptor is not None:
            parsed = True
            artifact = _prepare_codex_state_blob(
                state_descriptor[0],
                Path(directory),
                state_kind=state_descriptor[2],
                source_hash=blob_hash,
                semantic_source_path=source_path,
                enrichment_digest=None,
                enrichment_index_path=None,
                publication_publisher=publisher,
                captured_profile_key=evidence_reader.raw_profile_identity(raw_id),
            )
            sealed = True
            return artifact
        # Retained SQLite sessions are parsed while their output store owns
        # every sink, array, and accounting stream through publication.
        sqlite_path = BlobStore(evidence_reader.archive_root / "blob").blob_path(blob_hash)
        from polylogue.sources.sqlite_export import looks_like_logical_source_path

        if (
            provider in {Provider.HERMES, Provider.ANTIGRAVITY}
            and looks_like_logical_source_path(sqlite_path)
            and not path_declaration_refuses_session(provider, source_path)
        ):
            from polylogue.storage.sqlite.archive_tiers.write import append_session_to_shard
            from polylogue.storage.sqlite.session_shard import SessionShardBuilder

            store = SqliteMessageStore(sessions_path)
            with closing(SessionShardBuilder(Path(directory) / f"shard-{uuid.uuid4().hex}.db")) as builder:

                def prepared_sessions() -> Generator[ParsedSession, None, None]:
                    assert store is not None
                    with closing(
                        _iter_sqlite_path(
                            provider,
                            sqlite_path,
                            source_path,
                            store,
                            fallback_id=fallback_session_id(source_path, raw_id),
                            profile_identity=evidence_reader.raw_profile_identity(raw_id),
                        )
                    ) as source_sessions:
                        for session in source_sessions:
                            check_compute_cancelled()
                            session = normalize_session_timestamps(
                                session, fallback_timestamp=evidence_reader.raw_revision_file_mtime(raw_id)
                            )
                            with closing(
                                iter_enriched_sessions_from_retained_read(
                                    evidence_reader,
                                    provider,
                                    source_path,
                                    [session],
                                    captured_zip_coordinate=evidence_reader.raw_captured_zip_coordinate(raw_id),
                                )
                            ) as enriched_sessions:
                                for enriched in enriched_sessions:
                                    enriched.content_hash = session_content_hash(enriched)
                                    append_session_to_shard(builder, enriched)
                                    yield enriched

                with closing(prepared_sessions()) as selected_sessions:
                    _write_artifact(
                        store, blob_hash, selected_sessions, enrichment_digest=None, enrichment_index_path=None
                    )
                shard_path = builder.seal().path
            parsed = True
            # Hermes' SQLite export is parsed through a deterministic JSON
            # marker document. Validate that same marker projection (the
            # schema-eligible parser input), while binding its verdict to the
            # content-addressed SQLite revision that produced it.
            from polylogue.archive.raw_payload.decode import build_raw_payload_envelope

            envelope = None
            if provider is Provider.HERMES:
                envelope = build_raw_payload_envelope(
                    sqlite_path,
                    source_path=source_path,
                    fallback_provider=provider,
                    sqlite_immutable=True,
                )
                classification = envelope.artifact
            else:
                # Antigravity's SQLite parser validates its native table and
                # step shapes directly; there is no JSON document to resolve
                # against the retained schema registry.
                classification = ArtifactClassification(
                    provider=provider,
                    kind=ArtifactKind.SESSION_DOCUMENT,
                    parse_as_session=True,
                    schema_eligible=False,
                    default_priority=120,
                    reason="Antigravity trajectory SQLite parser",
                )
            if classification is not None:
                record_prepared_classification(
                    store.conn,
                    ArtifactStreamClassification(classification, False, 1),
                )
            verdict = None
            if envelope is not None and classification.schema_eligible:
                marker_path = Path(directory) / f"validation-marker-{uuid.uuid4().hex}.json"
                try:
                    marker_path.write_text(json.dumps(envelope.payload, ensure_ascii=False), encoding="utf-8")
                    from polylogue.schemas import validate_retained_document

                    verdict = validate_retained_document(
                        envelope.provider,
                        marker_path,
                        mode=validation_mode,
                        raw_id=raw_id,
                        revision_sha256=blob_hash,
                        evidence_id=raw_id,
                        source_path=source_path,
                        jsonl=False,
                        captured_zip_coordinate=evidence_reader.raw_captured_zip_coordinate(raw_id),
                        registry=schema_registry,
                    )
                finally:
                    marker_path.unlink(missing_ok=True)
            _prepare_attachment_publications(store, publisher, Path(directory))
            _prepare_sidecar_publications(store, publisher, Path(directory))
            store.close()
            store = None
            artifact = PreparedJsonl.seal(
                blob_hash,
                sessions_path,
                shard_path,
                enrichment_digest=None,
                enrichment_index_path=None,
                resolved_provider=provider,
                publication_publisher=publisher,
                captured_profile_key=evidence_reader.raw_profile_identity(raw_id),
            )
            if verdict is not None:
                artifact = dataclasses.replace(artifact, validation_verdict=verdict)
            sealed = True
            return artifact
        # A declared raw-only evidence path is terminal by its declaration:
        # its bytes are retained, never decoded as a session grammar.
        declared = declared_evidence_classification(source_path, provider=provider)
        sessions = (
            []
            if declared is not None
            else enrich_sessions_from_retained_read(
                evidence_reader,
                provider=provider,
                source_path=source_path,
                sessions=parse_retained_raw_sessions(evidence_reader, raw_id),
                captured_zip_coordinate=evidence_reader.raw_captured_zip_coordinate(raw_id),
            )
        )
        resolved_provider = Provider.from_string(sessions[0].source_name) if sessions else provider
        if not sessions and resolved_provider is Provider.UNKNOWN:
            with evidence_reader.open_raw_revision_material(raw_id) as (_provider, payload, _path, _kind):
                resolved_provider, _evidence = _resolved_retained_member_provider(
                    evidence_reader, raw_id, payload, source_path
                )
        parsed = True
        store = SqliteMessageStore(sessions_path)
        for session in sessions:
            session.content_hash = session_content_hash(session)
        shard_path = prepare_session_shard(Path(directory), sessions).path
        _write_artifact(
            store,
            blob_hash,
            sessions,
            enrichment_digest=None,
            enrichment_index_path=None,
        )
        if declared is not None:
            record_prepared_classification(store.conn, ArtifactStreamClassification(declared, True, 0))
        _prepare_attachment_publications(store, publisher, Path(directory))
        _prepare_sidecar_publications(store, publisher, Path(directory))
        store.close()
        store = None
        artifact = PreparedJsonl.seal(
            blob_hash,
            sessions_path,
            shard_path,
            enrichment_digest=None,
            enrichment_index_path=None,
            resolved_provider=resolved_provider,
            publication_publisher=publisher,
            captured_profile_key=evidence_reader.raw_profile_identity(raw_id),
        )
        artifact = _attach_retained_validation_verdict(
            artifact,
            provider=provider,
            blob_hash=blob_hash,
            blob_path=sqlite_path,
            source_path=source_path,
            raw_id=raw_id,
            directory=Path(directory),
            validation_mode=validation_mode,
            captured_zip_coordinate=evidence_reader.raw_captured_zip_coordinate(raw_id),
            jsonl=False,
            schema_registry=schema_registry,
        )
        sealed = True
        return artifact
    except (BlobVerificationCancelledError, VerificationCancelledError) as exc:
        raise DaemonOperationCancelled("retained byte verification cancelled") from exc
    except RetainedPreparationRetryableError:
        raise
    except (OSError, sqlite3.OperationalError, MemoryError) as exc:
        raise RetainedPreparationRetryableError(f"retained worker could not prepare raw {raw_id}") from exc
    except (
        BaseExceptionGroup,
        DaemonOperationCancelled,
        DaemonBackpressureError,
        ReferenceSealError,
        NativeConnectionSettlementError,
    ):
        raise
    except Exception as exc:
        if parsed:
            raise RetainedPreparationRetryableError(f"retained worker artifact failed for raw {raw_id}") from exc
        return PreparedJsonl(
            blob_hash,
            None,
            None,
            f"{type(exc).__name__}: {exc}",
            decode_failure=classify_decode_failure(exc),
            missing_profile_identity=isinstance(exc, MissingProfileIdentityError),
            retained_zip_membership_unproved=isinstance(exc, RetainedZipMembershipUnprovedError),
            unsupported_shape=isinstance(exc, UnsupportedRetainedJsonShapeError),
        )
    finally:
        if not sealed:
            primary = sys.exception()
            failures: list[BaseException] = []
            # Preparation has queued claims but has not admitted a flush.
            # Settle the artifact's actual SQL owner before retiring its
            # private claim files. A failed close retains both families.
            if store is not None:
                try:
                    store.close()
                except BaseException as cleanup:
                    failures.append(cleanup)
            if not failures:
                try:
                    publisher.discard_pending()
                except BaseException as cleanup:
                    failures.append(cleanup)
            if failures:
                if primary is not None:
                    failures.insert(0, primary)
                raise BaseExceptionGroup(
                    "retained artifact preparation and physical cleanup failed", failures
                ) from None
            sessions_path.unlink(missing_ok=True)
            if shard_path is not None:
                discard_session_shard(shard_path)


def _prepared_retained_outcome(
    archive: RetainedRawRead,
    raw_id: str,
    prepared_inputs: Mapping[str, PreparedRetainedInput],
    *,
    stop: Callable[[], bool] | None = None,
) -> tuple[Sequence[ParsedSession], int, RawRevisionKind] | Exception:
    prepared = prepared_inputs.get(raw_id)
    if prepared is None:
        raise RetainedPreparationRetryableError(f"prepared retained input is missing for raw {raw_id}")
    provider, blob_hash, source_path, kind, size = archive.raw_revision_descriptor(raw_id)
    native_id = archive.raw_native_id(raw_id) if kind is RawRevisionKind.APPEND else None
    fallback_timestamp = archive.raw_revision_file_mtime(raw_id)
    mismatched = (
        ("raw_id", prepared.raw_id, raw_id),
        ("blob_hash", prepared.blob_hash, blob_hash),
        ("source_path", prepared.source_path, source_path),
        ("payload_bytes", prepared.payload_bytes, size),
        ("native_id", prepared.native_id, native_id),
        ("profile_identity", prepared.captured_profile_key, archive.raw_profile_identity(raw_id)),
        ("parser_fingerprint", prepared.parser_fingerprint, raw_authority_parser_fingerprint()),
        ("fallback_timestamp", prepared.fallback_timestamp, fallback_timestamp),
    )
    changed = [name for name, expected, actual in mismatched if expected != actual]
    if prepared.provider != provider and not (
        prepared.provider is Provider.UNKNOWN
        and prepared.prepared_artifact is not None
        and prepared.prepared_artifact.resolved_provider is provider
    ):
        changed.append("provider")
    if prepared.revision_kind != kind and not (
        prepared.revision_kind is RawRevisionKind.UNKNOWN and kind is RawRevisionKind.FULL
    ):
        changed.append("revision_kind")
    if changed:
        raise RetainedPreparationRetryableError(
            f"prepared retained descriptor changed for raw {raw_id}: {', '.join(changed)}"
        )
    from polylogue.storage.blob_store import BlobStore

    # The producer verified the full bytes before and after preparation. Writer
    # publication compares that exact physical input without rehashing it.
    try:
        stat = BlobStore(Path(archive.archive_root) / "blob").blob_path(blob_hash).stat()
    except OSError as exc:
        raise RetainedPreparationRetryableError(f"prepared retained blob unavailable for raw {raw_id}") from exc
    current_stat = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if current_stat != prepared.verified_blob_stat:
        raise RetainedPreparationRetryableError(f"prepared retained blob changed for raw {raw_id}")
    sessions = _sealed_retained_sessions(prepared, stop=stop)
    if isinstance(sessions, Exception):
        return sessions
    return sessions, size, kind


class AntigravityTrajectoryDriftError(RuntimeError):
    """The live ``.pb`` trajectory no longer matches the retained blob.

    Antigravity replay cannot decode retained bytes on its own (protobuf
    decoding needs a live language-server client), so it re-reads the file.
    When that file has drifted, replaying it would bind current content to an
    older revision's ``raw_id``. This refusal counts as a replay degradation
    exactly the way the other replay ``RuntimeError``s do.
    """


def uncensused_historical_revision_raw_ids(
    archive_root: Path,
    raw_ids: list[str],
) -> tuple[str, ...]:
    """Return inputs whose current parser identity has not been persisted.

    The dedicated receipt proves that the parser whose current executable
    semantics fingerprint is stored actually observed every relevant raw.
    Any fingerprint change makes a former receipt stale, so no growing list of
    manually-known revisions can accidentally keep a changed parser authority
    current.

    Current-fingerprint receipts also have to prove the durable authority
    shape. Receipts with another fingerprint, or an empty key list when
    membership rows establish a canonical identity, are selected for
    recomputation instead of remaining permanently blocked by readiness.
    """
    if not raw_ids:
        return ()
    current_fingerprint = raw_authority_parser_fingerprint()
    with read_frame(
        archive_root / "source.db", tier=ArchiveTier.SOURCE, timeout_class="background-read"
    ) as source_frame:
        conn = source_frame.connection
        uncensused: list[str] = []
        for offset in range(0, len(raw_ids), 500):
            raw_id_chunk = raw_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in raw_id_chunk)
            rows = conn.execute(
                f"""
                SELECT r.raw_id
                FROM raw_sessions AS r
                LEFT JOIN raw_authority_parser_census AS c ON c.raw_id = r.raw_id
                WHERE r.raw_id IN ({placeholders})
                  AND NOT COALESCE(
                      c.parser_fingerprint = ?
                      AND c.status = 'complete'
                      AND c.detail LIKE 'parser-observed:%',
                      0
                  )
                ORDER BY r.raw_id
                """,
                [*raw_id_chunk, current_fingerprint],
            )
            uncensused.extend(str(row[0]) for row in rows)
            with closing(
                conn.execute(
                    f"""
                SELECT r.raw_id, c.logical_keys_json, r.logical_source_key, r.revision_kind,
                       EXISTS(SELECT 1 FROM raw_artifacts AS a
                              WHERE a.raw_id = r.raw_id AND a.parse_as_session = 0),
                       EXISTS(SELECT 1 FROM raw_membership_census AS mc
                              WHERE mc.raw_id = r.raw_id
                                AND mc.parser_fingerprint = ?
                                AND mc.status = 'non_session'),
                       EXISTS(SELECT 1 FROM raw_membership_census AS mc
                              WHERE mc.raw_id = r.raw_id
                                AND r.source_index < 0
                                AND mc.parser_fingerprint = ?
                                AND mc.status = 'failed'
                                AND mc.revision_authority = ?),
                       m.logical_source_key
                FROM raw_sessions AS r
                JOIN raw_authority_parser_census AS c ON c.raw_id = r.raw_id
                LEFT JOIN raw_session_memberships AS m ON m.raw_id = r.raw_id
                WHERE r.raw_id IN ({placeholders})
                  AND c.parser_fingerprint = ?
                  AND c.status = 'complete'
                  AND c.detail LIKE 'parser-observed:%'
                ORDER BY r.raw_id, m.logical_source_key
                """,
                    (
                        raw_authority_parser_fingerprint(),
                        raw_authority_parser_fingerprint(),
                        RawRevisionAuthority.BYTE_PROVEN.value,
                        *raw_id_chunk,
                        raw_authority_parser_fingerprint(),
                    ),
                )
            ) as receipt_rows:
                for raw_id, raw_rows in groupby(receipt_rows, key=lambda row: str(row[0])):
                    check_compute_cancelled()
                    first = next(raw_rows)
                    (
                        _raw_id,
                        logical_keys_json,
                        typed_key,
                        revision_kind,
                        typed_non_session,
                        parser_confirmed_non_session,
                        byte_governed_fragment,
                        _membership_key,
                    ) = first
                    with parser_census_identity_measurement(
                        raw_logical_key=typed_key,
                        revision_kind=revision_kind,
                        membership_logical_keys=(row[7] for row in chain((first,), raw_rows)),
                        observed_logical_keys=iter_parser_census_logical_keys(logical_keys_json),
                        observed_are_receipt=True,
                        check_stop=check_compute_cancelled,
                    ) as measured:
                        if not measured.complete(
                            typed_non_session=bool(typed_non_session),
                            parser_confirmed_non_session=bool(parser_confirmed_non_session),
                            byte_governed_fragment=bool(byte_governed_fragment),
                        ):
                            uncensused.append(raw_id)
        if not source_frame.revalidate():
            raise StaleContinuationError("source archive changed during parser source census")
    return tuple(sorted(set(uncensused)))


def apply_prepared_revision_census(
    seal: PreparedIndexMutation,
    prepared: PreparedRevisionSourceCensus,
    *,
    payload_store: BlobStore,
) -> RevisionCensusResult:
    """Publish the same off-writer census tape before returning its receipt."""
    from polylogue.storage.sqlite.archive_tiers.revision_governance import publish_prepared_revision_source

    _require_classification_inputs_current(prepared.classification_blob_stats, payload_store)
    publish_prepared_revision_source(seal, prepared.permit)
    return prepared.result


def _require_classification_inputs_current(
    blob_stats: Sequence[tuple[str, tuple[int, int, int, int, int]]], payload_store: BlobStore
) -> None:
    from polylogue.storage.sqlite.archive_tiers.revision_governance import _blob_stat_identity

    for blob_hash, identity in blob_stats:
        check_compute_cancelled()
        try:
            current = _blob_stat_identity(payload_store.blob_path(blob_hash))
        except OSError as failure:
            raise PreparedRawClassificationStaleError(
                "retained classification input disappeared before publication"
            ) from failure
        if current != identity:
            raise PreparedRawClassificationStaleError("retained classification input changed before publication")


class ReplayTopologyState(PolylogueStrEnum):
    """Why one logical key sits where it does in a rebuild's replay schedule.

    Every key in a rebuild carries exactly one state, so a schedule can be
    checked for topology -- nothing skipped, no parent fabricated -- without
    re-deriving the lineage graph from the archive.
    """

    ROOT = "root"
    """No parent claim at all; replays first, in lexicographic order."""

    DESCENDANT = "descendant"
    """Parent claim resolved to another key in this rebuild; replays after it."""

    ALIAS = "alias"
    """A second spelling -- provider form against public-origin form -- of
    another key in this rebuild; replays immediately after that spelling."""

    SELF_PARENT = "self_parent"
    """Claims its own identity as its parent; replays as a root."""

    UNRESOLVED_PARENT = "unresolved_parent"
    """Claims a parent that is missing, external, or in another rebuild
    batch; replays as a root with the edge left unresolved."""

    CYCLE = "cycle"
    """Sits on a parent cycle, so no member can precede all the others. The
    component replays after the roots, entered at its lexicographically
    smallest member."""

    SOLE = "sole"
    """The rebuild holds this key alone. There is nothing to order, so no
    parent claim was read -- reading one would parse a raw the census may
    have proven superseded without parsing."""


@dataclass(frozen=True, slots=True)
class ReplaySchedule:
    """One rebuild's replay order together with the topology that produced it.

    ``parent_of`` carries the edge resolved INSIDE this rebuild, so ``None``
    covers both "claimed nothing" and "claimed something absent";
    ``topology`` tells those apart.
    """

    order: tuple[str, ...]
    topology: Mapping[str, ReplayTopologyState]
    parent_of: Mapping[str, str | None]


#: Representative-raw lookups run in chunks of this many keys, so a full
#: rebuild's tens of thousands of logical keys never become one statement's
#: parameter list.
_REPLAY_KEY_QUERY_CHUNK: Final = 500


def _canonical_replay_key(logical_key: str) -> str | None:
    """Public-origin form of ``logical_key``, or ``None`` when it has no
    identity to canonicalize (a ``pending-raw:`` envelope, a legacy prefix).
    Such a key still schedules -- it simply resolves no lineage edge."""
    try:
        return canonical_authority_logical_key(logical_key)
    except ValueError:
        return None


def _replay_representative_raw_ids(
    sorted_keys: list[str],
    evidence_reader: RetainedReplayScheduleRead,
) -> dict[str, str]:
    """Choose canonical receipt-ranked representatives without reopening Source."""
    representative: dict[str, str] = {}
    for start in range(0, len(sorted_keys), _REPLAY_KEY_QUERY_CHUNK):
        chunk = sorted_keys[start : start + _REPLAY_KEY_QUERY_CHUNK]
        with evidence_reader.replay_representative_rows(chunk) as rows:
            for logical_source_key, raw_id in rows:
                check_compute_cancelled()
                representative.setdefault(str(logical_source_key), str(raw_id))
    return representative


def _replay_parent_claims(
    sorted_keys: list[str],
    evidence_reader: RetainedReplayScheduleRead,
    spill: _PreparedReplayInputs,
) -> dict[str, str | None]:
    """Read each key's claimed parent logical key from its representative raw.

    The claim comes from the parsed session whose OWN logical key is the key
    being resolved. A raw can carry many sessions, so reading the first
    one's claim would attribute a parent to a session that never made it;
    a key whose representative raw yields no matching session claims nothing.
    """
    representative = _replay_representative_raw_ids(sorted_keys, evidence_reader)
    keys_by_raw: dict[str, list[str]] = {}
    for key in sorted_keys:
        raw_id = representative.get(key)
        if raw_id is not None:
            keys_by_raw.setdefault(raw_id, []).append(key)

    claims: dict[str, str | None] = dict.fromkeys(sorted_keys, None)
    for raw_id, keys in keys_by_raw.items():
        try:
            sessions, _payload_bytes = spill.for_raw(raw_id)
        except DaemonOperationCancelled:
            raise
        except Exception:
            # Lineage ordering is a scheduling optimization only -- any
            # failure here degrades to "claims nothing", never to a replay
            # or adoption failure.
            continue
        claim_by_canonical: dict[str, str] = {}
        for session in sessions:
            parent_provider_id = session.parent_session_provider_id
            if not parent_provider_id:
                continue
            origin = origin_from_provider(session.source_name)
            own_key = _canonical_replay_key(f"{origin.value}:{session.provider_session_id}")
            if own_key is None:
                continue
            # A raw that repeats one key across sessions keeps the first
            # claim in parse order, which is fixed for a given raw.
            claim_by_canonical.setdefault(own_key, f"{origin.value}:{parent_provider_id}")
        for key in keys:
            canonical_key = _canonical_replay_key(key)
            if canonical_key is not None:
                claims[key] = claim_by_canonical.get(canonical_key)
    return claims


def _replay_cycle_members(sorted_keys: list[str], edge: Mapping[str, str | None]) -> set[str]:
    """Return every key sitting ON a parent cycle.

    Each key has at most one parent edge, so a cycle is exactly a chain that
    re-enters a key still on the current walk. Keys that merely descend from
    a cycle are not members: they have a parent that precedes them.
    """
    members: set[str] = set()
    settled: set[str] = set()
    for start in sorted_keys:
        if start in settled:
            continue
        walk: list[str] = []
        position: dict[str, int] = {}
        node: str | None = start
        while node is not None and node not in settled:
            if node in position:
                members.update(walk[position[node] :])
                break
            position[node] = len(walk)
            walk.append(node)
            node = edge.get(node)
        settled.update(walk)
    return members


def _lineage_aware_replay_schedule(
    logical_keys: set[str],
    evidence_reader: RetainedReplayScheduleRead,
    spill: _PreparedReplayInputs,
) -> ReplaySchedule:
    """Order one rebuild's byte-typed logical keys so a parent's cohort
    replays before any of its children's (polylogue-5q2u).

    Replaying a child before its parent forces ``_resolve_session_graph`` to
    store the child's shared prefix WHOLE, then re-walk and normalize it
    (delete the duplicate prefix rows, remap ``session_events`` refs, delete
    prefix-scoped dependents) once the parent finally arrives -- the
    #2467 deferred-tail path, O(orphaned_children * shared_prefix_size) real
    row-mutation work. Lexicographic order has zero relationship to
    parent/child lineage, so it triggers this expensive path roughly as often
    as not during a cold/full rebuild. Visiting roots first, and each child
    only after its parent, minimizes how often it triggers.

    This is deliberately scheduling-only: it must never change WHAT gets
    replayed or adopted, only the order this module's own replay loop visits
    logical keys in. Every key in ``logical_keys`` appears exactly once in
    ``order`` whatever its topology, and every key carries a
    ``ReplayTopologyState`` saying which rule placed it, so a caller can
    check a schedule without re-deriving the graph. The order is a function
    of the archive alone: ties break lexicographically at every step, and
    each component is entered at its lexicographically smallest member.
    """
    sorted_keys = sorted(logical_keys)
    if len(sorted_keys) <= 1:
        return ReplaySchedule(
            order=tuple(sorted_keys),
            topology=dict.fromkeys(sorted_keys, ReplayTopologyState.SOLE),
            parent_of=dict.fromkeys(sorted_keys, None),
        )

    claims = _replay_parent_claims(sorted_keys, evidence_reader, spill)

    # One spelling represents each identity; the others are its aliases.
    representative_spelling: dict[str, str] = {}
    for key in sorted_keys:
        canonical_key = _canonical_replay_key(key)
        if canonical_key is not None:
            representative_spelling.setdefault(canonical_key, key)

    topology: dict[str, ReplayTopologyState] = {}
    edge: dict[str, str | None] = {}
    for key in sorted_keys:
        canonical_key = _canonical_replay_key(key)
        if canonical_key is not None and representative_spelling[canonical_key] != key:
            edge[key] = representative_spelling[canonical_key]
            topology[key] = ReplayTopologyState.ALIAS
            continue
        claim = claims[key]
        canonical_claim = None if claim is None else _canonical_replay_key(claim)
        target = None if canonical_claim is None else representative_spelling.get(canonical_claim)
        if claim is None:
            edge[key] = None
            topology[key] = ReplayTopologyState.ROOT
        elif target is None:
            edge[key] = None
            topology[key] = ReplayTopologyState.UNRESOLVED_PARENT
        elif target == key:
            edge[key] = None
            topology[key] = ReplayTopologyState.SELF_PARENT
        else:
            edge[key] = target
            topology[key] = ReplayTopologyState.DESCENDANT

    cycle_members = _replay_cycle_members(sorted_keys, edge)
    for key in cycle_members:
        topology[key] = ReplayTopologyState.CYCLE

    children: dict[str, list[str]] = {}
    for key in sorted_keys:
        target = edge[key]
        if target is not None:
            children.setdefault(target, []).append(key)

    order: list[str] = []
    seen: set[str] = set()

    def visit(start: str) -> None:
        # Iterative: a resume chain is as deep as the archive is old.
        stack = [start]
        while stack:
            key = stack.pop()
            if key in seen:
                continue
            seen.add(key)
            order.append(key)
            stack.extend(reversed(children.get(key, ())))

    # Roots first, then each remaining component entered at its
    # lexicographically smallest cycle member. Every key carries at most one
    # parent edge, so its chain ends at a root or closes a cycle: these two
    # passes together reach every key exactly once.
    for key in sorted_keys:
        if edge[key] is None:
            visit(key)
    for key in sorted_keys:
        if key in cycle_members:
            visit(key)

    return ReplaySchedule(order=tuple(order), topology=topology, parent_of=edge)


def _validated_prepared_aggregate(
    logical_key: str,
    accepted_raw_ids: tuple[str, ...],
    *,
    prepared_inputs: Mapping[str, PreparedRetainedInput],
    prepared_aggregates: Mapping[str, PreparedRetainedAggregate] | None,
) -> tuple[ParsedSession, Path]:
    """Revalidate an off-writer chain composition against accepted raw order."""
    from polylogue.sources.prepared_merge import prepared_cohort_source_hash

    aggregate = (prepared_aggregates or {}).get(logical_key)
    if aggregate is None or aggregate.raw_ids != accepted_raw_ids:
        raise RetainedPreparationRetryableError(f"prepared aggregate order changed for logical source {logical_key}")
    ordered: list[tuple[str, PreparedJsonl]] = []
    for raw_id in accepted_raw_ids:
        prepared = prepared_inputs.get(raw_id)
        artifact = prepared.prepared_artifact if prepared is not None else None
        if artifact is None or artifact.blob_hash is None:
            raise RetainedPreparationRetryableError(f"prepared aggregate lost raw dependency {raw_id}")
        ordered.append((raw_id, artifact))
    artifact = aggregate.artifact
    if artifact.error is not None or artifact.blob_hash != prepared_cohort_source_hash(ordered):
        raise RetainedPreparationRetryableError(f"prepared aggregate source dependency changed for {logical_key}")
    if artifact.shard_path is None:
        raise RetainedPreparationRetryableError(f"prepared aggregate shard is absent for {logical_key}")
    try:
        sessions = artifact.session_sequence()
    except DaemonOperationCancelled:
        raise
    except Exception as exc:
        raise RetainedPreparationRetryableError(f"prepared aggregate seal is invalid for {logical_key}") from exc
    if len(sessions) != 1:
        raise RetainedPreparationRetryableError(f"prepared aggregate does not contain one session for {logical_key}")
    return sessions[0], artifact.shard_path


@dataclass(frozen=True, slots=True)
class PreparedMembershipReplay:
    """Captured membership decisions and immutable comparison artifacts."""

    candidate_raw_ids: tuple[str, ...]
    head_raw_id: str | None
    sessions: Mapping[str, ParsedSession]
    projections: Mapping[str, SessionRevisionProjection]
    classification: MembershipClassification
    head_plan: MembershipHeadPlan | None = None
    source_decisions: Mapping[str, MembershipDecision] | None = None
    decided_at_ms: int | None = None

    def close(self) -> None:
        failures: list[BaseException] = []
        for projection in self.projections.values():
            try:
                projection.close()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise BaseExceptionGroup("prepared membership cleanup failed", failures)


def prepared_session_for_logical_key(
    sessions: Sequence[ParsedSession], *, raw_id: str, logical_source_key: str
) -> ParsedSession:
    """Select exactly one original parser output for the admitted member key."""
    selected: ParsedSession | None = None
    parsed_count = 0
    match_count = 0
    for session in sessions:
        check_compute_cancelled()
        parsed_count += 1
        if f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}" != logical_source_key:
            continue
        match_count += 1
        if selected is None:
            selected = session
    if selected is None:
        raise CohortMembershipRefusalError(
            logical_source_key,
            raw_id,
            f"selector member parsed {parsed_count} session(s), none for this logical key",
        )
    if match_count != 1:
        raise CohortMembershipRefusalError(
            logical_source_key,
            raw_id,
            f"selector member parsed {match_count} sessions for this logical key, not one",
        )
    return selected


def prepared_session_for_revision_key(
    sessions: Sequence[ParsedSession], *, raw_id: str, logical_source_key: str
) -> ParsedSession:
    """Select a retained transcript member or its validated singleton event output."""
    if not is_work_event_raw_id(raw_id):
        return prepared_session_for_logical_key(sessions, raw_id=raw_id, logical_source_key=logical_source_key)
    # The retained event parser has validated the original durable envelope.
    # Its authority key identifies that envelope; its session identifies the
    # existing destination, so transcript membership equality does not apply.
    if logical_source_key != raw_id or len(sessions) != 1:
        raise CohortMembershipRefusalError(
            logical_source_key, raw_id, "work event lost its singleton envelope authority"
        )
    session = sessions[0]
    if session.messages or len(session.session_events) != 1:
        raise CohortMembershipRefusalError(
            logical_source_key, raw_id, "work event output is not one event without transcript messages"
        )
    return session


def prepare_membership_replay(
    archive: RetainedMembershipRead,
    logical_key: str,
    prepared_inputs: Mapping[str, PreparedRetainedInput],
    *,
    head_raw_id: str | None,
    stop: Callable[[], bool] | None = None,
) -> PreparedMembershipReplay:
    """Capture membership comparison on the retained read-only snapshot."""

    candidate_raw_ids = set(archive.raw_membership_rebuild_raw_ids(logical_key))
    # Rebuild replay also carries its current-pass `membership_candidates` for
    # quarantined/unknown raws; the persisted membership rows are the exact
    # read-only equivalent for the selected prepared component. The ordinary
    # rebuild selector intentionally filters these out until classification.
    candidate_raw_ids.update(
        raw_id for raw_id in archive.raw_membership_logical_raw_ids(logical_key) if raw_id in prepared_inputs
    )
    # The accepted head is comparison evidence under any authority. Without
    # it, a cohort cannot tell a member that adds content from one the head
    # already contains, and yielding to a byte head would record both as
    # superseded. Its own binding stays as it is: a head without a membership
    # row receives no Source decision.
    if head_raw_id is not None:
        candidate_raw_ids.add(head_raw_id)
    member_sessions: dict[str, ParsedSession] = {}
    revisions: list[MembershipRevision] = []
    projections: dict[str, SessionRevisionProjection] = {}
    try:
        for raw_id in sorted(candidate_raw_ids):
            if raw_id not in prepared_inputs:
                raise RetainedPreparationRetryableError(
                    f"prepared membership candidate is absent for {logical_key}: {raw_id}"
                )
            outcome = _prepared_retained_outcome(archive, raw_id, prepared_inputs, stop=stop)
            if isinstance(outcome, Exception):
                raise CohortMembershipRefusalError(
                    logical_key, raw_id, f"selector member did not parse: {outcome}"
                ) from outcome
            sessions, _size, _kind = outcome
            session = prepared_session_for_logical_key(sessions, raw_id=raw_id, logical_source_key=logical_key)
            member_sessions[raw_id] = session
            projections[raw_id] = session_revision_projection(session)
            revisions.append(
                MembershipRevision(
                    raw_id,
                    projections[raw_id],
                    # Only the producer's own time orders browser snapshots. An
                    # acquisition fallback (the capture file's mtime) is not
                    # provider authority: the receiver rewrites one spool file
                    # per capture, so its mtime would order any two captures.
                    session.updated_at if session.updated_at_provenance != "fallback" else None,
                    browser_snapshot_fidelity=_browser_snapshot_fidelity(session.ingest_flags),
                    # Declared capture order: the latest retained observation
                    # of these bytes, never their evidence volume.
                    capture_order=archive.raw_revision_observation_order(raw_id),
                    # Only declared provider IDs are identity evidence; an
                    # id-less message must not stand in as a shared ``None``
                    # member that makes unrelated snapshots look preserved.
                    provider_message_ids=(
                        session.messages.provider_message_ids(include_none=False)
                        if isinstance(session.messages, SqliteMessageSink)
                        else frozenset(
                            message.provider_message_id for message in session.messages if message.provider_message_id
                        )
                    ),
                    provider_attachment_ids=frozenset(
                        attachment.provider_attachment_id for attachment in session.attachments
                    ),
                )
            )
        classification = classify_membership_revisions(revisions, existing_accepted_raw_id=head_raw_id)
    except BaseException as primary:
        failures: list[BaseException] = [primary]
        for projection in projections.values():
            try:
                projection.close()
            except BaseException as cleanup:
                failures.append(cleanup)
        if len(failures) > 1:
            raise BaseExceptionGroup("membership preparation and physical cleanup failed", failures) from None
        raise
    return PreparedMembershipReplay(
        tuple(sorted(candidate_raw_ids)),
        head_raw_id,
        member_sessions,
        projections,
        classification,
    )


def _prepared_write_for(
    prepared_writes: Mapping[tuple[str, str], PreparedSessionWrite] | None,
    raw_id: str,
    session: ParsedSession,
) -> PreparedSessionWrite | None:
    """Select a prepared write by raw and session: one raw may carry several."""
    if not prepared_writes:
        return None
    return prepared_writes.get(
        (raw_id, f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}")
    )


def _raw_has_pending_envelope(archive: ArchiveStore, raw_id: str) -> bool:
    row = (
        archive._ensure_source_conn()
        .execute("SELECT logical_source_key FROM raw_sessions WHERE raw_id = ?", (raw_id,))
        .fetchone()
    )
    return row is not None and str(row[0] or "").startswith(PENDING_RAW_LOGICAL_SOURCE_PREFIX)


def _require_prepared_cross_acquisition_write(
    archive: ArchiveStore,
    session: ParsedSession,
    *,
    accepted_raw_id: str,
    prepared_write: PreparedSessionWrite | None,
    prepared_inputs: Mapping[str, PreparedRetainedInput] | None,
) -> None:
    if prepared_inputs is None:
        return
    session_id = f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}"
    row = archive._conn.execute("SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row is not None and row[0] is not None and row[0] != accepted_raw_id and prepared_write is None:
        raise RetainedPreparationRetryableError(f"prepared cross-acquisition write is missing for {session_id}")


@_capture_replay_enrichment_degradations
def apply_prepared_revision_replay(
    archive_root: Path,
    *,
    reference_seal: PreparedIndexMutation,
    active_index_path: Path,
    selected_raw_ids: list[str],
    prepared_inputs: Mapping[str, PreparedRetainedInput],
    prepared_aggregates: Mapping[str, PreparedRetainedAggregate],
    prepared_writes: Mapping[tuple[str, str], PreparedSessionWrite],
    prepared_replay_adoption: Mapping[tuple[str, tuple[str, ...]], PreparedRevisionAdoption],
    prepared_replay_plans: Mapping[str, RevisionReplayPlan],
    prepared_byte_outcomes: Mapping[str, PreparedRevisionReplayOutcome],
    prepared_membership_plans: Mapping[str, PreparedMembershipReplay],
    prepared_replay_source: PreparedRetainedReplaySource,
    prepared_replay_schedule: ReplaySchedule,
    prepared_logical_keys: tuple[str, ...],
    prepared_membership_keys: tuple[str, ...],
    prepared_byte_logical_keys: tuple[str, ...],
    prepared_key_refusals: tuple[CohortMembershipRefusalError, ...],
    prepared_lineage_deferrals: tuple[str, ...] = (),
    bulk_fts: bool = True,
    exact_fts_audit: bool = False,
) -> PreparedRevisionReplayResult:
    """Publish one retained component from its captured worker-sealed inputs.

    Parsing, byte classification and row lowering belong to canonical retained
    preparation. A changed plan or unavailable carrier refuses this attempt;
    publication never submits compute or falls back to inline parsing.
    """
    from polylogue.sources.codex_state_projection import THREAD_STATE_KIND
    from polylogue.storage.blob_publication import ConnectionBlobPublicationRead

    def attachment_view(raw_id: str, session: ParsedSession) -> Mapping[object, tuple[bytes | None, int, str]]:
        artifact = prepared_replay_source.attachment_artifacts.get(raw_id)
        if artifact is None:
            raise RetainedPreparationRetryableError(f"publication has no original attachment carrier for {raw_id}")
        return artifact.resident_attachment_blobs(
            raw_id=raw_id,
            source_read=publication_read,
            session_id=str(make_session_id(session.source_name, session.provider_session_id)),
        )

    publication_started = time.perf_counter()
    adoption_deferred = 0
    quarantined = 0
    stage_timings: dict[str, float] = {}
    stage_counts: dict[str, int] = {}
    written_session_ids: dict[str, None] = {}
    changed_session_ids: dict[str, None] = {}
    writer_changed_raw_ids: dict[str, None] = {}
    written_message_count = 0
    written_counts: dict[str, int] = {}
    session_outputs: dict[str, tuple[str, bytes | None, bytes | None, int]] = {}
    membership_refusals: list[tuple[str, str, MembershipDecision]] = []

    refused_publication_raw_ids: set[str] = set()

    def record_write_result(result: ArchiveRawParsedWriteResult) -> None:
        nonlocal written_message_count
        if result.publication_refused:
            # Deterministic writer precedence (e.g. a weaker DOM capture after
            # native evidence) refuses this output; it is settled, not retryable.
            refused_publication_raw_ids.add(result.raw_id)
        written_session_ids[result.session_id] = None
        if result.content_changed:
            writer_changed_raw_ids[result.raw_id] = None
        if result.session_id not in prepared_replay_source.original_index_outputs:
            raise ReferenceSealError("replay output lacks its original Index content witness")
        before = prepared_replay_source.original_index_outputs[result.session_id]
        with connection_cursor(
            archive._conn, "SELECT content_hash,message_count FROM sessions WHERE session_id=?", (result.session_id,)
        ) as rows:
            after = rows.fetchone()
        if after is not None:
            session_outputs[result.session_id] = (
                result.session_id,
                None if before is None else before[0],
                None if after[0] is None else bytes(after[0]),
                int(after[1]),
            )
        # Membership publication can replace an existing Index projection
        # while retaining the same Raw claim. Its acquisition-change flag is
        # not an Index content measurement; compare the original pinned
        # projection with the actual row produced by this writer.
        if after is not None and (before is None or before[0] != session_outputs[result.session_id][2]):
            changed_session_ids[result.session_id] = None
        elif result.content_changed:
            # Session-hash-excluded rows (an appended agent work event keeps
            # every session-owned field) still change the session's output.
            changed_session_ids[result.session_id] = None
        written_message_count += result.counts.get("messages", 0)
        for key, count in result.counts.items():
            written_counts[key] = written_counts.get(key, 0) + count

    # Deferred children publish nothing in this unit; like refused keys they
    # carry no prepared outcome and no Source acknowledgement here.
    refused_keys = {refusal.logical_source_key for refusal in prepared_key_refusals} | set(prepared_lineage_deferrals)
    logical_keys: set[str] = set()
    spill = _PreparedReplayInputs(prepared_inputs)
    with _prepared_replay_archive(archive_root, reference_seal, active_index_path) as archive:
        # Only this validated, never-published destination has a mandatory
        # readiness pass that rebuilds deferred materializations before readers.
        bulk_build = archive.owns_inactive_generation
        publication_read = ConnectionBlobPublicationRead(archive._ensure_source_conn())
        with archive.index_mutation_scope(prepared_seal=reference_seal):
            # Source census has its own original prepared unit. Its counts
            # belong to that receipt; this Index publication returns only replay.
            logical_keys.update(prepared_logical_keys)
            membership_keys = set(prepared_membership_keys)
            replayed = 0
            byte_replayed_keys: set[str] = set()
            settled_byte_keys: set[str] = set()
            actual_terminal_raw_ids: set[str] = set()
            work_event_keys = sorted(key for key in logical_keys if is_work_event_raw_id(key))
            if any(is_work_event_raw_id(key) for key in membership_keys):
                raise RetainedPreparationRetryableError(
                    "prepared work-event envelope acquired transcript membership publication"
                )
            replay_schedule = prepared_replay_schedule
            if set(replay_schedule.order) != (logical_keys | membership_keys).difference(work_event_keys):
                raise RetainedPreparationRetryableError(
                    "prepared retained replay schedule has another selected key set"
                )
            # Thread-state Source receipts have already been prepared/published
            # on this witness. The companion graph/link carrier is mandatory for
            # a normal Index route and never rehydrates original writer payloads.
            projected_state = 0
            for selected_raw_id in selected_raw_ids:
                retained_input = prepared_inputs.get(selected_raw_id)
                artifact = retained_input.prepared_artifact if retained_input is not None else None
                if artifact is None or artifact.codex_state_kind != THREAD_STATE_KIND:
                    continue
                index_connection = archive.index_connection
                if index_connection is None:
                    if reference_seal.has_tier_capability("index"):
                        raise RetainedPreparationRetryableError(
                            "prepared thread projection lost its admitted Index destination"
                        )
                    continue
                projected_state += int(artifact.apply_thread_projection(reference_seal, index_connection))
            stage_counts["state_projection"] = projected_state

            ordered_logical_keys = list(prepared_byte_logical_keys)
            if ordered_logical_keys != [key for key in replay_schedule.order if key in prepared_byte_logical_keys]:
                raise RetainedPreparationRetryableError("prepared byte replay keys have another canonical order")

            def adoption_is_current(logical_key: str, accepted_raw_ids: tuple[str, ...]) -> bool:
                evidence = prepared_replay_adoption.get((logical_key, accepted_raw_ids))
                if evidence is None:
                    raise RetainedPreparationRetryableError(
                        f"prepared replay adoption evidence is missing for {logical_key}"
                    )
                return evidence.adoptable

            def selected_byte_plan(logical_key: str) -> RevisionReplayPlan:
                plan = prepared_replay_plans.get(logical_key)
                if plan is None or plan.logical_source_key != logical_key:
                    raise RetainedPreparationRetryableError(f"original prepared byte plan is absent for {logical_key}")
                return plan

            for logical_key in ordered_logical_keys:
                if logical_key in refused_keys:
                    continue
                plan = selected_byte_plan(logical_key)
                if not plan.accepted_raw_ids:
                    # Full-only non-prefix conversion belongs to the original
                    # Source preparation unit, never this admitted Index writer.
                    membership_keys.add(logical_key)
                    continue
                parsed_by_raw_id: dict[str, ParsedSession] = {}
                retained_bytes = 0
                for raw_id in plan.accepted_raw_ids:
                    spill_started = time.perf_counter()
                    sessions, payload_bytes = spill.for_raw(raw_id)
                    stage_timings["spill_load"] = stage_timings.get("spill_load", 0.0) + (
                        time.perf_counter() - spill_started
                    )
                    parsed_by_raw_id[raw_id] = prepared_session_for_logical_key(
                        sessions, raw_id=raw_id, logical_source_key=logical_key
                    )
                    retained_bytes += payload_bytes
                prepared_aggregate_session: ParsedSession | None = None
                if len(plan.accepted_raw_ids) > 1:
                    prepared_aggregate_session, _prepared_aggregate_path = _validated_prepared_aggregate(
                        logical_key,
                        tuple(plan.accepted_raw_ids),
                        prepared_inputs=prepared_inputs,
                        prepared_aggregates=prepared_aggregates,
                    )
                adoptable_started = time.perf_counter()
                adoptable = adoption_is_current(logical_key, plan.accepted_raw_ids)
                stage_timings["replay.adoptable_check"] = stage_timings.get("replay.adoptable_check", 0.0) + (
                    time.perf_counter() - adoptable_started
                )
                if not adoptable:
                    archive.defer_raw_revision_adoption(prepared_replay_adoption[(logical_key, plan.accepted_raw_ids)])
                    adoption_deferred += len(plan.accepted_raw_ids)
                    settled_byte_keys.add(logical_key)
                    continue
                if prepared_byte_outcomes[logical_key].suppressed:
                    adoption_deferred += len(plan.accepted_raw_ids)
                    settled_byte_keys.add(logical_key)
                    continue
                try:
                    tip_raw_id = plan.accepted_raw_ids[-1]
                    prepared_write = _required_prepared_write_for(
                        prepared_writes, tip_raw_id, prepared_aggregate_session or parsed_by_raw_id[tip_raw_id]
                    )
                    composed_session = prepared_aggregate_session or parsed_by_raw_id[tip_raw_id]
                    if prepared_aggregate_session is not None:
                        from polylogue.sources.prepared_merge import aggregate_resident_attachment_blobs

                        aggregate_attachment_view = aggregate_resident_attachment_blobs(
                            prepared_aggregates[logical_key].artifact,
                            source_read=publication_read,
                            session_id=str(
                                make_session_id(composed_session.source_name, composed_session.provider_session_id)
                            ),
                            original_artifacts=prepared_replay_source.attachment_artifacts,
                            accepted_raw_ids=plan.accepted_raw_ids,
                        )
                    else:
                        aggregate_attachment_view = attachment_view(tip_raw_id, composed_session)
                    try:
                        _session_id, applied_raw_ids = archive.apply_raw_revision_replay(
                            plan,
                            parsed_by_raw_id,
                            prepared_outcome=prepared_byte_outcomes[logical_key],
                            acquired_at_ms=0,
                            stage_timings_s=stage_timings,
                            manage_transaction=False,
                            bulk_fts=bulk_fts,
                            bulk_build=bulk_build,
                            fresh_build=False,
                            fresh_build_batch=None,
                            prepared_aggregate_rows=prepared_write.rows,
                            prepared_aggregate_session=composed_session,
                            preacquired_aggregate_attachment_blobs=aggregate_attachment_view,
                            prepared_required_raw_ids=frozenset({tip_raw_id}),
                            prepared_write=prepared_write,
                            write_result=record_write_result,
                            preacquired_attachment_blobs_by_raw_id={
                                raw_id: attachment_view(raw_id, parsed_by_raw_id[raw_id])
                                for raw_id in plan.accepted_raw_ids
                            },
                        )
                        from polylogue.storage.sqlite.archive_tiers.revision_governance import (
                            revision_replay_terminal_raw_ids,
                        )

                        if not applied_raw_ids and tip_raw_id in refused_publication_raw_ids:
                            # Writer precedence refused the output: the Source
                            # acknowledgements stand as this key's terminal
                            # outcome and the stored session keeps its head.
                            actual_terminal_raw_ids.update(revision_replay_terminal_raw_ids(plan))
                            settled_byte_keys.add(logical_key)
                            continue
                        if applied_raw_ids != revision_replay_terminal_raw_ids(plan):
                            raise RetainedPreparationRetryableError(
                                "byte publication disagrees with its original Source acknowledgements"
                            )
                        actual_terminal_raw_ids.update(applied_raw_ids)
                    except PreparedSessionWriteRefusedError as exc:
                        raise RetainedPreparationRetryableError(
                            f"prepared byte replay dependency changed for {logical_key}"
                        ) from exc
                except sqlite3.IntegrityError as exc:
                    raise sqlite3.IntegrityError(
                        f"apply_prepared_revision_replay: byte-proven replay failed for logical_key={plan.logical_source_key!r}: {exc}"
                    ) from exc
                replayed += 1
                byte_replayed_keys.add(logical_key)
            from polylogue.storage.sqlite.archive_tiers.revision_governance import apply_prepared_membership_index

            for logical_key in (key for key in replay_schedule.order if key in membership_keys):
                if logical_key in refused_keys:
                    continue
                membership_plan = prepared_membership_plans.get(logical_key)
                if membership_plan is None:
                    if logical_key in byte_replayed_keys or logical_key in settled_byte_keys:
                        continue
                    raise RetainedPreparationRetryableError(
                        f"original prepared membership outcome is absent for {logical_key}"
                    )
                head_plan = membership_plan.head_plan
                if (
                    head_plan is None
                    or membership_plan.source_decisions is None
                    or membership_plan.decided_at_ms is None
                ):
                    raise RetainedPreparationRetryableError(
                        f"original prepared membership acknowledgements are absent for {logical_key}"
                    )
                classification = membership_plan.classification
                quarantined += len(classification.ambiguous_raw_ids)
                accepted_members = classification.accepted_raw_ids
                if (
                    accepted_members
                    and head_plan.yield_to_head_raw_id is None
                    and not adoption_is_current(logical_key, accepted_members)
                ):
                    archive.defer_raw_revision_adoption(prepared_replay_adoption[(logical_key, accepted_members)])
                    adoption_deferred += len(accepted_members)
                    continue
                if (
                    accepted_members
                    and head_plan.yield_to_head_raw_id is None
                    and head_plan.conflict is None
                    and membership_plan.source_decisions.get(accepted_members[-1]) is MembershipDecision.DEFERRED
                ):
                    # The original Source preparation found the session
                    # suppressed and acknowledged its accepted members as
                    # deferred without retaining a carrier. The Index writes
                    # nothing for it, exactly as a suppressed byte outcome.
                    adoption_deferred += len(accepted_members)
                    continue
                attachments = (
                    {}
                    if not accepted_members
                    else attachment_view(accepted_members[-1], membership_plan.sessions[accepted_members[-1]])
                )
                membership_prepared_write = None
                if accepted_members and head_plan.yield_to_head_raw_id is None:
                    membership_prepared_write = _required_prepared_write_for(
                        prepared_writes,
                        accepted_members[-1],
                        membership_plan.sessions[accepted_members[-1]],
                    )
                session_id, actual_decisions = apply_prepared_membership_index(
                    archive,
                    logical_key,
                    classification,
                    membership_plan.sessions,
                    membership_plan.projections,
                    head_plan,
                    decided_at_ms=membership_plan.decided_at_ms,
                    preacquired_attachment_blobs=attachments,
                    stage_timings_s=stage_timings,
                    bulk_fts=bulk_fts,
                    bulk_build=bulk_build,
                    prepared_write=membership_prepared_write,
                    write_result=record_write_result,
                )
                if actual_decisions != membership_plan.source_decisions:
                    raise RetainedPreparationRetryableError(
                        f"membership publication disagrees with original Source acknowledgements for {logical_key}"
                    )
                membership_refusals.extend(
                    (logical_key, raw_id, decision)
                    for raw_id, decision in actual_decisions.items()
                    if decision is MembershipDecision.AMBIGUOUS
                )
                if (
                    accepted_members
                    and head_plan.yield_to_head_raw_id is None
                    and actual_decisions[accepted_members[-1]] is MembershipDecision.APPLIED
                ):
                    replayed += 1
            for logical_key in work_event_keys:
                if logical_key not in prepared_replay_source.terminal_raw_ids:
                    adoption_deferred += 1
                    continue
                plan = selected_byte_plan(logical_key)
                if plan.accepted_raw_ids != (logical_key,):
                    message = f"retained work event {logical_key} lost its singleton byte authority"
                    raise RuntimeError(message)
                event_sessions, _event_bytes = spill.for_raw(logical_key)
                event_session = prepared_session_for_revision_key(
                    event_sessions, raw_id=logical_key, logical_source_key=logical_key
                )
                event_session_id = str(make_session_id(event_session.source_name, event_session.provider_session_id))
                with connection_cursor(
                    archive._conn,
                    "SELECT 1 FROM sessions WHERE session_id = ?",
                    (event_session_id,),
                ) as rows:
                    event_session_exists = rows.fetchone() is not None
                if not event_session_exists:
                    _LOGGER.warning("work_event_session_absent: raw_id=%s session_id=%s", logical_key, event_session_id)
                    adoption_deferred += 1
                    continue
                from polylogue.storage.sqlite.archive_tiers.revision_governance import _index_parsed_for_retained_raw

                event_result = _index_parsed_for_retained_raw(
                    archive,
                    event_session,
                    raw_id=logical_key,
                    source_index=-1,
                    stage_timings_s=stage_timings,
                    stage_timing_prefix="replay.work_event",
                    manage_transaction=False,
                    bulk_fts=bulk_fts,
                    bulk_build=bulk_build,
                    preacquired_attachment_blobs={},
                    finalize_raw_parse=False,
                    prepared_required=True,
                    prepared_write=_required_prepared_write_for(prepared_writes, logical_key, event_session),
                )
                record_write_result(event_result)
                replayed += 1
                byte_replayed_keys.add(logical_key)
                actual_terminal_raw_ids.add(logical_key)
            if actual_terminal_raw_ids != prepared_replay_source.terminal_raw_ids:
                raise RetainedPreparationRetryableError(
                    "ordered Index outcomes disagree with original Source parse acknowledgements"
                )
            if replayed and not adoption_deferred and exact_fts_audit:
                from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync

                fts_snapshot = fts_invariant_snapshot_sync(archive._conn)
                if not fts_snapshot.messages.ready:
                    raise RuntimeError("retained replay found messages_fts out of sync")
            if stage_timings:
                stage_timings["total"] = time.perf_counter() - publication_started
                _LOGGER.info(
                    "backfill stage timings: %s",
                    " ".join(
                        (f"{key}={value:.1f}s" for key, value in sorted(stage_timings.items(), key=lambda kv: -kv[1]))
                    ),
                )
        from polylogue.storage.sqlite.archive_tiers.revision_governance import publish_prepared_revision_source

        # An owned inactive generation reconstructs Index from frozen Source.
        # Its terminal commit retires this seal and grants no Source continuation.
        if reference_seal.destination is None or reference_seal.destination.kind != "owned_inactive":
            publish_prepared_revision_source(reference_seal, prepared_replay_source.permit)
    return PreparedRevisionReplayResult(
        0,
        0,
        replayed,
        quarantined,
        adoption_deferred,
        stage_timings_s=stage_timings,
        stage_counts=stage_counts,
        written_session_ids=tuple(written_session_ids),
        changed_session_ids=tuple(changed_session_ids),
        writer_changed_raw_ids=tuple(writer_changed_raw_ids),
        written_message_count=written_message_count,
        written_counts=written_counts,
        session_outputs=tuple(session_outputs.values()),
        membership_refusals=tuple(membership_refusals),
    )


def _enrich_retained_parse_outcome(
    archive: ArchiveStore,
    raw_id: str,
    *,
    descriptor: tuple[Provider, str, str, RawRevisionKind, int, str | None, str | None],
    outcome: tuple[list[ParsedSession], int, RawRevisionKind] | Exception,
) -> tuple[list[ParsedSession], int, RawRevisionKind] | Exception:
    """Enrich one decoded raw -- the per-raw body of enrichment.

    Enrichment never reads across raws: each outcome is normalized against its
    own ``raw_id``'s retained mtime and assembled against its own descriptor's
    source path. That independence is what lets the streaming consumer enrich
    a raw at the moment it is read instead of holding a whole page to enrich
    it in one pass. :func:`_enrich_retained_parse_results` is this function
    applied over an already-materialized dict.
    """
    if isinstance(outcome, Exception):
        return outcome
    # Unit-level parser/dedupe probes deliberately pass tiny protocol fakes;
    # enrichment is an ArchiveStore production concern.
    if not isinstance(archive, ArchiveStore):
        return outcome
    provider, _blob_hash, descriptor_source_path, _descriptor_kind, _size, _native_id, profile_identity = descriptor
    sessions, payload_bytes, kind = outcome
    source_conn = archive._ensure_source_conn()
    sessions = _normalize_retained_parse_sessions(source_conn, raw_id, sessions)
    if sessions:
        provider = Provider.from_string(sessions[0].source_name)
    return (
        _replay_safe_enrich_sessions(
            provider=provider,
            sessions=sessions,
            index_conn=archive.index_connection,
            source_conn=source_conn,
            blob_root=Path(archive.archive_root) / "blob",
            source_path=descriptor_source_path,
            captured_zip_coordinate=archive.raw_captured_zip_coordinate(raw_id),
        ),
        payload_bytes,
        kind,
    )


def _normalize_retained_parse_sessions(
    source_conn: sqlite3.Connection,
    raw_id: str,
    sessions: list[ParsedSession],
) -> list[ParsedSession]:
    """Bind decoded timestamps to the captured retained file-mtime fallback."""
    row = source_conn.execute(
        "SELECT file_mtime_ms FROM raw_sessions WHERE raw_id = ?",
        (raw_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown raw revision {raw_id}")
    fallback_timestamp = datetime.fromtimestamp(int(row[0]) / 1000, UTC).isoformat() if row[0] is not None else None
    return [normalize_session_timestamps(session, fallback_timestamp=fallback_timestamp) for session in sessions]


#: Counted enrichment degradations, keyed by reason. ``_replay_safe_enrich_sessions``
#: is the single enrichment entry point for every replay decode path; when a
#: caller cannot supply the evidence handles the provider's ladder needs, the
#: shortfall is counted HERE rather than silently falling through to a
#: parsed-content heuristic. Replay determinism is then auditable: a nonzero
#: count means some raw's title/assembly came from content, not durable
#: evidence, and the receipt says so.
def _count_enrichment_degradation(reason: str) -> None:
    counts = _REPLAY_ENRICHMENT_DEGRADATIONS.get()
    if counts is None:
        return
    with _REPLAY_ENRICHMENT_DEGRADATIONS_LOCK:
        counts[reason] += 1


def replay_enrichment_degradations() -> dict[str, int]:
    """Snapshot the counted enrichment degradations (test/receipt surface)."""
    counts = _REPLAY_ENRICHMENT_DEGRADATIONS.get()
    if counts is None:
        return {}
    with _REPLAY_ENRICHMENT_DEGRADATIONS_LOCK:
        return dict(counts)


def reset_replay_enrichment_degradations() -> None:
    counts = _REPLAY_ENRICHMENT_DEGRADATIONS.get()
    if counts is not None:
        with _REPLAY_ENRICHMENT_DEGRADATIONS_LOCK:
            counts.clear()


class RetainedSessionEnricher:
    """Apply retained assembly evidence to one session at a time.

    Live intake, the writer's inline fallback and retained replay must publish
    the same interpretation of the same bytes. Retained replay enriches every
    parsed session from durable archive evidence (Codex thread titles, Claude
    Code session index/history, ChatGPT asset maps); a route that skipped it
    stored the native id as the title and a different content hash, and the
    raw owner then accepted that output as current. Bundle exports share one
    source-scoped evidence snapshot; other providers resolve evidence for the
    session being enriched.
    """

    __slots__ = (
        "_blob_root",
        "_bundle",
        "_cached",
        "_frames",
        "_index_conn",
        "_keeps_session_ids",
        "_provider",
        "_session_ids",
        "_source_conn",
        "_source_path",
        "_captured_zip_coordinate",
    )

    def __init__(
        self,
        provider: Provider,
        *,
        source_path: str,
        captured_zip_coordinate: CapturedZipMemberCoordinate | None,
        index_conn: sqlite3.Connection | None,
        source_conn: sqlite3.Connection | None,
        blob_root: Path | None,
        frames: _RebindingEvidenceFrames | None = None,
    ) -> None:
        self._provider = provider
        self._frames = frames
        self._source_path = source_path
        self._captured_zip_coordinate = captured_zip_coordinate
        self._index_conn = index_conn
        self._source_conn = source_conn
        self._blob_root = blob_root
        self._bundle = provider in BUNDLE_PROVIDERS and Path(source_path).name.lower().endswith(".json")
        self._cached: SidecarData | None = None
        # Only a provider whose evidence is keyed by session needs the ids;
        # a bundle of any other provider keeps none of them.
        self._keeps_session_ids = _replay_enrichment_reads_index(provider)
        self._session_ids: PickleSpool[str] = PickleSpool()

    def _bind(self, *, enriched: bool) -> None:
        """Take the current evidence frames, rebinding them when they near expiry."""
        if self._frames is None:
            return
        connections = self._frames.current(self._digest_from_bound if enriched else None)
        self._index_conn = connections.get(ArchiveTier.INDEX)
        self._source_conn = connections.get(ArchiveTier.SOURCE)

    def _digest_from_bound(self) -> str:
        connections = self._frames.connections if self._frames is not None else {}
        if self._frames is not None:
            self._index_conn = connections.get(ArchiveTier.INDEX)
            self._source_conn = connections.get(ArchiveTier.SOURCE)
        return self._compute_digest()

    def dependency_digest(self) -> str:
        """The evidence every session enriched so far depends on.

        Sealed into a prepared artifact, then recomputed by the writer
        (``prepared_enrichment_dependency_state``) before it publishes.
        """
        self._bind(enriched=True)
        return self._compute_digest()

    def _compute_digest(self) -> str:
        return enrichment_dependency_digest(
            provider=self._provider,
            source_path=self._source_path,
            captured_zip_coordinate=self._captured_zip_coordinate,
            provider_session_ids=self._session_ids,
            index_conn=self._index_conn,
            source_conn=self._source_conn,
            blob_root=self._blob_root,
            parser_sidecars=False,
        )

    def __call__(self, session: ParsedSession) -> ParsedSession:
        from polylogue.sources.assembly import get_assembly_spec

        self._bind(enriched=bool(self._session_ids) or self._cached is not None)
        if self._keeps_session_ids and session.provider_session_id:
            self._session_ids.append(session.provider_session_id)
        spec = get_assembly_spec(self._provider)
        if spec is None:
            return session
        if not self._bundle:
            return _replay_safe_enrich_sessions(
                provider=self._provider,
                sessions=[session],
                index_conn=self._index_conn,
                source_conn=self._source_conn,
                blob_root=self._blob_root,
                source_path=self._source_path,
                captured_zip_coordinate=self._captured_zip_coordinate,
            )[0]
        if self._cached is None:
            self._cached = _retained_enrichment_sidecar_data(
                provider=self._provider,
                sessions=(),
                evidence_reader=ConnectionRetainedEnrichmentRead(self._index_conn, self._source_conn, self._blob_root),
                source_path=self._source_path,
                captured_zip_coordinate=self._captured_zip_coordinate,
            )
        return stamp_enrichment_evidence(self._provider, self._cached, spec.enrich_session(session, self._cached))

    def close(self) -> None:
        from polylogue.sources.assembly import close_sidecar_data

        failures: list[BaseException] = []
        if self._cached is not None:
            try:
                close_sidecar_data(self._cached)
            except BaseException as error:
                failures.append(error)
            else:
                self._cached = None
        try:
            self._session_ids.close()
        except BaseException as error:
            failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("retained enrichment and ID tape close failed", failures)

    def enrich_all(self, sessions: Sequence[ParsedSession]) -> list[ParsedSession]:
        return [self(session) for session in sessions]


class EnrichmentEvidenceMovedError(RuntimeError):
    """Retained evidence changed while one preparation was enriching its sessions.

    Retryable: a fresh preparation reads one consistent view again.
    """


class _RebindingEvidenceFrames:
    """Short read frames for a worker's enrichment, rebound before they expire.

    A preparation can outlast any read frame's declared bound (a very large
    bundle), so the frames are not held across the whole parse: each is
    opened when enrichment first reads it and replaced once it has lived half
    its bound. Every session enriched so far must read the same evidence in
    the replacement as in the frame it was enriched from, or the sealed
    digest -- computed from the last frame -- would certify evidence some
    sessions never saw; a moved view refuses the preparation instead.
    """

    def __init__(self, *, source_db_path: str | Path, index_db_path: str | Path) -> None:
        self._paths = ((ArchiveTier.SOURCE, Path(source_db_path)), (ArchiveTier.INDEX, Path(index_db_path)))
        self._stack: ExitStack | None = None
        self._opened_at = 0.0
        self.connections: dict[ArchiveTier, sqlite3.Connection | None] = {}

    def _open(self) -> None:
        stack = ExitStack()
        connections: dict[ArchiveTier, sqlite3.Connection | None] = {}
        try:
            for tier, path in self._paths:
                if not path.exists():
                    connections[tier] = None
                    continue
                frame = stack.enter_context(read_frame(path, tier=tier, timeout_class="background-read"))
                frame.connection.execute("BEGIN")
                connections[tier] = frame.connection
        except BaseException:
            stack.close()
            raise
        self._stack, self.connections, self._opened_at = stack, connections, time.monotonic()

    def current(self, digest: Callable[[], str] | None) -> dict[ArchiveTier, sqlite3.Connection | None]:
        """The live frames, rebound when half their bound has passed.

        ``digest`` computes the enrichment dependency digest from
        ``self.connections``; ``None`` when nothing has been enriched yet.
        """
        if self._stack is None:
            self._open()
        elif time.monotonic() - self._opened_at >= _EVIDENCE_FRAME_REBIND_S:
            before = digest() if digest is not None else None
            self.close()
            self._open()
            if before is not None and digest is not None and digest() != before:
                raise EnrichmentEvidenceMovedError("retained enrichment evidence moved during preparation")
        return self.connections

    def close(self) -> None:
        if self._stack is not None:
            stack, self._stack = self._stack, None
            self.connections = {}
            stack.close()


#: Half the background read frame bound: a frame is replaced well before it expires.
_EVIDENCE_FRAME_REBIND_S = 150.0


@contextmanager
def open_retained_session_enricher(
    provider: Provider,
    *,
    source_path: str,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
    source_db_path: str | Path,
    index_db_path: str | Path,
    blob_root: str | Path,
) -> Iterator[RetainedSessionEnricher]:
    """Enrichment for a worker that has no archive handle, over rebinding read frames.

    An absent tier is ordinary absence (a source-only or not yet bootstrapped
    archive): enrichment then counts the degradation and applies only the
    parsed-content fallbacks, exactly as retained replay does without it.
    """
    frames = _RebindingEvidenceFrames(source_db_path=source_db_path, index_db_path=index_db_path)
    enricher = RetainedSessionEnricher(
        provider,
        source_path=source_path,
        captured_zip_coordinate=captured_zip_coordinate,
        index_conn=None,
        source_conn=None,
        blob_root=Path(blob_root),
        frames=frames,
    )
    try:
        yield enricher
    finally:
        try:
            enricher.close()
        finally:
            frames.close()


def _replay_enrichment_reads_index(provider: Provider) -> bool:
    """Codex retained-state titles require the captured Index dependency."""
    return provider is Provider.CODEX


def _replay_safe_enrich_sessions(
    *,
    provider: Provider,
    sessions: list[ParsedSession],
    index_conn: sqlite3.Connection | None,
    source_conn: sqlite3.Connection | None,
    blob_root: Path | None,
    source_path: str | None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
    evidence_observer: Callable[[object], None] | None = None,
) -> list[ParsedSession]:
    """Enrich one retained parse without consulting ambient source files.

    Codex title resolution's step 3b reads the projected ``threads.title``
    rows, recomputed from the retained state export, so a reindex resolves the
    same curated titles a live ingest does instead of baking in the
    content-heuristic first-prompt fallback. Without an index connection the
    bundle stays empty and only the parsed-content fallbacks apply.

    polylogue-sqy57: every handle is a REQUIRED keyword, with no default. The
    cache-miss decode paths (the spill prefetcher's reparse and
    ``_PreparedReplayInputs.for_raw``'s inline reparse) previously omitted them
    and silently produced heuristic titles, making replay output depend on
    cache state instead of durable evidence. A caller that genuinely has no
    handle passes ``None`` and the shortfall is counted by
    ``replay_enrichment_degradations()``; it is never silent.
    """
    from polylogue.sources.assembly import get_assembly_spec

    spec = get_assembly_spec(provider)
    if spec is None:
        return sessions
    sidecar_data = _retained_enrichment_sidecar_data(
        provider=provider,
        sessions=sessions,
        evidence_reader=ConnectionRetainedEnrichmentRead(index_conn, source_conn, blob_root),
        source_path=source_path,
        captured_zip_coordinate=captured_zip_coordinate,
    )
    if evidence_observer is not None:
        evidence_observer(sidecar_data)
    from polylogue.sources.assembly import close_sidecar_data

    try:
        return [
            stamp_enrichment_evidence(provider, sidecar_data, spec.enrich_session(session, sidecar_data))
            for session in sessions
        ]

    finally:
        close_sidecar_data(sidecar_data)


def _retained_enrichment_sidecar_data(
    *,
    provider: Provider,
    sessions: Sequence[ParsedSession],
    evidence_reader: RetainedEnrichmentRead,
    source_path: str | None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
    provider_session_ids: Iterable[str] | None = None,
) -> SidecarData:
    """Read one canonical evidence bundle through the caller's finite host."""
    sidecar_data = cast("SidecarData", {})
    if _replay_enrichment_reads_index(provider):
        thread_ids = (
            provider_session_ids
            if provider_session_ids is not None
            else (session.provider_session_id for session in sessions if session.provider_session_id)
        )
        titles = evidence_reader.retained_state_titles(thread_ids, source_path)
        if titles is None:
            _count_enrichment_degradation("codex_titles_without_index_conn")
        elif titles:
            sidecar_data = cast("SidecarData", {"retained_state_titles": titles})
        else:
            from polylogue.sources.retained_title_index import RetainedTitleIndex

            if isinstance(titles, RetainedTitleIndex):
                titles.close()
    from polylogue.sources.assembly import close_sidecar_data

    retained: SidecarData | None = None
    try:
        retained = (
            None
            if not source_path
            else evidence_reader.retained_assembly_evidence(
                sidecar_data,
                provider=provider,
                source_path=source_path,
                captured_zip_coordinate=captured_zip_coordinate,
            )
        )
        if retained is None:
            _count_enrichment_degradation("retained_assembly_without_source_evidence")
            return sidecar_data
        close_sidecar_data(sidecar_data, borrowed=retained)
        return retained
    except BaseException as primary:
        failures: list[BaseException] = [primary]
        for owned, borrowed in ((sidecar_data, retained), (retained, None)):
            if owned is None:
                continue
            try:
                close_sidecar_data(owned, borrowed=borrowed)
            except BaseException as cleanup:
                failures.append(cleanup)
        if len(failures) > 1:
            raise BaseExceptionGroup("retained evidence selection and cleanup failed", failures) from None
        raise


def parse_retained_raw_sessions(archive: RetainedRawRead, raw_id: str) -> list[ParsedSession]:
    """Parse retained raw evidence without eagerly loading stream records.

    Raw-revision replay is shared by historical repair and the live full and
    append routes.  Keeping the provider-shape decision here prevents a
    seemingly harmless live replay helper from reintroducing ``read_all()``
    for Codex/Claude JSONL evidence.
    """
    provider, blob_hash, source_path, kind, _payload_size = archive.raw_revision_descriptor(raw_id)
    if (
        not is_work_event_raw_id(raw_id)
        and declared_evidence_classification(source_path, provider=provider) is not None
    ):
        # A raw-only declaration is terminal even when the retained payload is
        # empty. Do not send zero bytes through a provider JSON decoder.
        return []
    profile_identity = archive.raw_profile_identity(raw_id)
    fallback_timestamp = archive.raw_revision_file_mtime(raw_id)
    sidecar_resolver = archive.retained_sidecar_resolver()

    # Work events have their own durable envelope.  They are not provider
    # transcript records, so replay them before dispatching to provider parsers.
    # Replay returns the event alone; ``write_parsed_session_to_archive``
    # recognizes the work-event raw and writes it event-only, keeping the
    # stored session header.
    if is_work_event_raw_id(raw_id):
        _provider, payload, _path, _kind = archive.raw_revision_material(raw_id)
        try:
            envelope = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid retained work-event envelope for {raw_id}") from exc
        if not isinstance(envelope, dict) or envelope.get("_polylogue_work_event") != 1:
            raise ValueError(f"unrecognized retained work-event envelope for {raw_id}")
        from polylogue.sources.parsers.base import ParsedSession, ParsedSessionEvent

        try:
            event_provider = Provider(str(envelope["provider"]))
            native_id = str(envelope["native_session_id"])
            event_type = str(envelope["event_type"])
            event_payload = envelope["payload"]
            if not isinstance(event_payload, dict):
                raise TypeError("payload must be an object")
            event = ParsedSessionEvent(
                event_type=event_type,
                timestamp=envelope.get("timestamp"),
                payload=event_payload,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed retained work-event envelope for {raw_id}") from exc
        return [
            ParsedSession(
                source_name=event_provider,
                provider_session_id=native_id,
                messages=[],
                session_events=[event],
            )
        ]

    def normalize_replay(sessions: list[ParsedSession]) -> list[ParsedSession]:
        return [normalize_session_timestamps(session, fallback_timestamp=fallback_timestamp) for session in sessions]

    # SQLite revisions are already immutable blob files. Open them directly:
    # materializing the entire database as ``bytes`` before SQLite opens it
    # duplicates every page in Python memory.
    if provider in {Provider.HERMES, Provider.ANTIGRAVITY}:
        from polylogue.sources.sqlite_export import looks_like_logical_source_path

        sqlite_path = archive.raw_revision_blob_path(raw_id)
        if sqlite_path is not None and looks_like_logical_source_path(sqlite_path):
            return normalize_replay(
                _parse_sqlite_path(
                    provider,
                    sqlite_path,
                    source_path,
                    fallback_id=fallback_session_id(source_path, raw_id),
                    profile_identity=profile_identity,
                )
            )

    fallback_id_override = (
        _append_session_native_id(
            archive.raw_append_logical_key(raw_id),
            provider=provider,
            captured_native_id=archive.raw_native_id(raw_id),
        )
        if kind is RawRevisionKind.APPEND
        else None
    )
    if provider is Provider.UNKNOWN:
        # Source-only acquisition deliberately retains unknown ZIP members
        # without decoding them.  Recovery is the first lawful point to
        # inspect the durable bytes and resolve their parser, before deciding
        # whether their filename is a stream route.
        with archive.open_raw_revision_material(raw_id) as (
            _stream_provider,
            stream_payload,
            _stream_path,
            _stream_kind,
        ):
            provider, _evidence = _resolved_retained_member_provider(archive, raw_id, stream_payload, source_path)
        if is_stream_record_provider(source_path, str(provider)):
            with archive.open_raw_revision_material(raw_id) as (
                _stream_provider,
                stream_payload,
                stream_path,
                _stream_kind,
            ):
                return normalize_replay(
                    _parse_stream(
                        provider,
                        stream_payload,
                        stream_path,
                        fallback_id_override=fallback_id_override,
                        archive_root=archive.archive_root,
                        sidecar_resolver=sidecar_resolver,
                        profile_identity=profile_identity,
                    )
                )
        if provider is Provider.UNKNOWN:
            raise UnsupportedRetainedJsonShapeError(f"retained input has no recognized provider shape: {source_path}")
        _provider, eager_payload, _source_path, _eager_kind = archive.raw_revision_material(raw_id)
        payload_path = archive.raw_revision_blob_path(raw_id) if provider is Provider.HERMES else None
        return normalize_replay(
            _parse_one(
                provider,
                eager_payload,
                source_path,
                payload_path=payload_path,
                archive_root=archive.archive_root,
                sidecar_resolver=sidecar_resolver,
                profile_identity=profile_identity,
                fallback_id_override=fallback_id_override,
            )
        )
    if is_stream_record_provider(source_path, str(provider)):
        with archive.open_raw_revision_material(raw_id) as (stream_provider, stream_payload, stream_path, _stream_kind):
            return normalize_replay(
                _parse_stream(
                    stream_provider,
                    stream_payload,
                    stream_path,
                    fallback_id_override=fallback_id_override,
                    archive_root=archive.archive_root,
                    sidecar_resolver=sidecar_resolver,
                    profile_identity=profile_identity,
                )
            )
    _provider, eager_payload, _source_path, _eager_kind = archive.raw_revision_material(raw_id)
    payload_path = archive.raw_revision_blob_path(raw_id) if provider is Provider.HERMES else None
    return normalize_replay(
        _parse_one(
            provider,
            eager_payload,
            source_path,
            payload_path=payload_path,
            archive_root=archive.archive_root,
            sidecar_resolver=sidecar_resolver,
            profile_identity=profile_identity,
            fallback_id_override=fallback_id_override,
        )
    )


def _retained_codex_state_descriptor(archive: RetainedRawRead, raw_id: str) -> tuple[Path, str, str, str] | None:
    """Identify one immutable retained Codex state export without mutating it."""
    provider, blob_hash, source_path, _kind, _payload_size = archive.raw_revision_descriptor(raw_id)
    if provider is not Provider.CODEX:
        return None
    state_path = archive.raw_revision_blob_path(raw_id)
    if state_path is None:
        return None
    if not is_declared_logical_export(state_path, source_path):
        return None
    state_kind = codex_state.classify_codex_sqlite_path(state_path, immutable=True)
    if state_kind not in codex_state.IN_SCOPE_KINDS:
        return None
    return state_path, source_path, state_kind, blob_hash


class _PreparedReplayInputs:
    """Lookup view over the existing complete worker-sealed carrier."""

    def __init__(self, prepared_inputs: Mapping[str, PreparedRetainedInput]) -> None:
        self._prepared_inputs = prepared_inputs

    def for_raw(self, raw_id: str) -> tuple[Sequence[ParsedSession], int]:
        prepared = self._prepared_inputs.get(raw_id)
        if prepared is None:
            raise RetainedPreparationRetryableError(f"prepared replay carrier is missing for raw {raw_id}")
        sessions = _sealed_retained_sessions(prepared)
        if isinstance(sessions, Exception):
            raise RetainedPreparationRetryableError(f"parser-refused raw {raw_id} reached replay") from sessions
        return sessions, prepared.payload_bytes


LEGACY_PAGE_IMAGE_CENSUS_DETAIL = (
    "retained legacy SQLite page image; no current parser reads this material and it is "
    "not a logical export, so it produces no session"
)

#: Declared shapes whose bytes are one session's own record stream.
_NATIVE_SESSION_STREAM_KINDS = frozenset({ArtifactKind.SESSION_RECORD_STREAM, ArtifactKind.COORDINATOR_SESSION_STREAM})


def _retained_page_image_raw(archive: RetainedRawRead, raw_id: str) -> bool:
    """Return whether this raw's retained bytes are a SQLite page image.

    Conservation accounting has to distinguish "parsed to nothing" from
    "deliberately retained material the current parser cannot read", so the
    terminal receipt names which one happened (polylogue-qjscw).
    """
    try:
        _provider, blob_hash, _source_path, _kind, _size = archive.raw_revision_descriptor(raw_id)
    except DaemonOperationCancelled:
        raise
    except Exception:
        return False
    blob_path = archive.raw_revision_blob_path(raw_id)
    return blob_path is not None and is_sqlite_page_image(blob_path)


def _persist_terminal_non_session_artifact(
    producer: SourceRawOutcomeProducer,
    raw_id: str,
    *,
    provider: Provider,
    observed_at_ms: int,
    source_path: str,
    source_index: int,
    stream_classification: ArtifactStreamClassification | None,
    manage_transaction: bool,
) -> bool:
    """Persist the complete classification bound to the sealed retained input."""
    if stream_classification is None or not stream_classification.proved_non_session:
        return False
    classification = stream_classification.classification
    if classification.provider is not provider:
        raise RetainedPreparationRetryableError("prepared artifact classification provider changed")
    origin = origin_from_provider(provider)
    _upsert_raw_artifact(
        producer,
        raw_id,
        ArchiveSourceArtifact(
            artifact_id=artifact_observation_id(
                source_name=origin.value,
                source_path=source_path,
                source_index=source_index,
            ),
            origin=origin,
            source_path=source_path,
            source_index=source_index,
            artifact_kind=classification.cohort,
            classification_reason=classification.reason,
            parse_as_session=False,
            schema_eligible=classification.schema_eligible,
            first_observed_at_ms=observed_at_ms,
            last_observed_at_ms=observed_at_ms,
        ),
        manage_transaction=manage_transaction,
    )
    _apply_source_raw_state_update(
        producer,
        raw_id,
        state=_raw_parse_success_state(provider),
        manage_transaction=manage_transaction,
    )
    return True


def _persist_strict_session_refusal_artifact(
    producer: SourceRawOutcomeProducer,
    raw_id: str,
    *,
    provider: Provider,
    source_path: str,
    source_index: int,
    observed_at_ms: int,
    stream_classification: ArtifactStreamClassification | None,
    manage_transaction: bool,
) -> bool:
    """Retain positive session evidence when strict schema validation refuses it."""
    if (
        stream_classification is None
        or not stream_classification.classification.parse_as_session
        or stream_classification.classification.provider is not provider
    ):
        return False
    classification = stream_classification.classification
    origin = origin_from_provider(provider)
    _upsert_raw_artifact(
        producer,
        raw_id,
        ArchiveSourceArtifact(
            artifact_id=artifact_observation_id(
                source_name=origin.value,
                source_path=source_path,
                source_index=source_index,
            ),
            origin=origin,
            source_path=source_path,
            source_index=source_index,
            artifact_kind=classification.cohort,
            classification_reason=f"{classification.reason}; strict schema validation refused",
            support_status=ArtifactSupportStatus.UNSUPPORTED_PARSEABLE,
            parse_as_session=True,
            schema_eligible=True,
            first_observed_at_ms=observed_at_ms,
            last_observed_at_ms=observed_at_ms,
        ),
        manage_transaction=manage_transaction,
    )
    return True


def _persist_codex_state_artifact(
    producer: SourceRawOutcomeProducer,
    raw_id: str,
    *,
    source_path: str,
    source_index: int,
    observed_at_ms: int,
    state_kind: str,
    manage_transaction: bool,
) -> None:
    """Record a validated Codex state database as typed non-session evidence."""
    _upsert_raw_artifact(
        producer,
        raw_id,
        ArchiveSourceArtifact(
            artifact_id=artifact_observation_id(
                source_name=Provider.CODEX.value,
                source_path=source_path,
                source_index=source_index,
            ),
            origin=origin_from_provider(Provider.CODEX),
            source_path=source_path,
            source_index=source_index,
            artifact_kind=ArtifactKind.BINARY_DATABASE.value,
            classification_reason=f"validated Codex state database ({state_kind})",
            support_status=ArtifactSupportStatus.RECOGNIZED_UNPARSED,
            parse_as_session=False,
            schema_eligible=False,
            first_observed_at_ms=observed_at_ms,
            last_observed_at_ms=observed_at_ms,
        ),
        manage_transaction=manage_transaction,
    )


def _persist_legacy_page_image_artifact(
    producer: SourceRawOutcomeProducer,
    raw_id: str,
    *,
    provider: Provider,
    source_path: str,
    source_index: int,
    observed_at_ms: int,
    manage_transaction: bool,
) -> None:
    """Type an obsolete SQLite page image as recognized non-session evidence."""
    origin = origin_from_provider(provider)
    _upsert_raw_artifact(
        producer,
        raw_id,
        ArchiveSourceArtifact(
            artifact_id=artifact_observation_id(
                source_name=origin.value,
                source_path=source_path,
                source_index=source_index,
            ),
            origin=origin,
            source_path=source_path,
            source_index=source_index,
            artifact_kind=ArtifactKind.BINARY_DATABASE.value,
            classification_reason="legacy SQLite page image",
            support_status=ArtifactSupportStatus.RECOGNIZED_UNPARSED,
            parse_as_session=False,
            schema_eligible=False,
            first_observed_at_ms=observed_at_ms,
            last_observed_at_ms=observed_at_ms,
        ),
        manage_transaction=manage_transaction,
    )


def _parse_one(
    provider: Provider,
    payload: bytes,
    source_path: str,
    *,
    profile_identity: str | None = None,
    payload_path: Path | None = None,
    archive_root: Path | None = None,
    sidecar_resolver: SidecarResolver | None,
    fallback_id_override: str | None = None,
) -> list[ParsedSession]:
    return _parse_one_raw(
        provider,
        payload,
        source_path,
        payload_path=payload_path,
        archive_root=archive_root,
        sidecar_resolver=sidecar_resolver,
        profile_identity=profile_identity,
        fallback_id_override=fallback_id_override,
    )


def _parse_one_raw(
    provider: Provider,
    payload: bytes,
    source_path: str,
    *,
    profile_identity: str | None = None,
    payload_path: Path | None = None,
    archive_root: Path | None = None,
    sidecar_resolver: SidecarResolver | None,
    fallback_id_override: str | None = None,
) -> list[ParsedSession]:
    if provider is Provider.HERMES and profile_identity is None:
        raise MissingProfileIdentityError("retained Hermes input has no captured profile identity receipt")

    if provider is Provider.ANTIGRAVITY and Path(source_path).suffix.lower() == ".pb":
        trajectory_path = Path(source_path)
        root = trajectory_path.parent.parent
        if trajectory_path.parent.name != "conversations" or not trajectory_path.is_file():
            raise RuntimeError(
                f"Antigravity raw replay requires its original conversations/<cascade_id>.pb trajectory: {source_path}"
            )
        # Antigravity decoding needs a live language-server client, so this
        # replay re-derives from the file on disk rather than from the
        # retained bytes. Antigravity rewrites conversations/<cascade_id>.pb
        # in place, so an unchecked replay would return CURRENT content under
        # an OLDER revision's raw_id -- rebuildable state derived from mutable
        # source instead of from the archived bytes. Existence and session
        # count do not detect that; only the bytes do. A drifted file is a
        # typed refusal, never a silent substitution.
        try:
            live_bytes = trajectory_path.read_bytes()
        except OSError as exc:
            raise AntigravityTrajectoryDriftError(
                f"Antigravity trajectory is unreadable for replay: {source_path}: {exc}"
            ) from exc
        if hashlib.sha256(live_bytes).digest() != hashlib.sha256(payload).digest():
            raise AntigravityTrajectoryDriftError(
                "Antigravity trajectory on disk no longer matches the retained blob for this revision; "
                f"refusing to replay current content under the retained raw id: {source_path}"
            )
        cascade_id = trajectory_path.stem
        sessions = list(antigravity.iter_language_server_exports(root, only_cascade_ids=frozenset({cascade_id})))
        if len(sessions) != 1 or sessions[0].provider_session_id != cascade_id:
            raise RuntimeError(
                "Antigravity raw replay did not reproduce exactly one requested trajectory "
                f"{cascade_id!r} from {source_path}"
            )
        return admit_parsed_sessions_for_publication(sessions, provider=provider, source_path=source_path)
    source_name = Path(source_path).name
    fallback_id = fallback_id_override or fallback_session_id(source_path, source_path)
    if provider is Provider.HERMES and looks_like_logical_source_bytes(payload):
        with _sqlite_payload_path(payload, payload_path, archive_root) as sqlite_path:
            if not (
                is_declared_logical_export(sqlite_path, source_path)
                or is_undeclared_logical_export(sqlite_path, source_path)
            ):
                # A page image is refused: it cannot be proven against the
                # database it was copied from. A well-framed export acquired
                # under a noncanonical filename carries its own scope header
                # and stays replayable (polylogue-qjscw).
                raise RuntimeError(f"retained Hermes SQLite material is not a logical export: {source_path}")
            if hermes_state.looks_like_state_db_path(sqlite_path, immutable=True):
                return admit_parsed_sessions_for_publication(
                    collect_sqlite_sessions(
                        provider,
                        sqlite_path,
                        fallback_id=fallback_id,
                        profile_identity=profile_identity,
                    ),
                    provider=provider,
                    source_path=source_path,
                )
            if hermes_verification.looks_like_verification_evidence_db_path(sqlite_path, immutable=True):
                return admit_parsed_sessions_for_publication(
                    hermes_verification.parse_verification_evidence_db(
                        sqlite_path,
                        fallback_id=fallback_id,
                        profile_identity=profile_identity,
                        immutable=True,
                    ),
                    provider=provider,
                    source_path=source_path,
                )
    if provider is Provider.ANTIGRAVITY and looks_like_logical_source_bytes(payload):
        with _sqlite_payload_path(payload, payload_path, archive_root) as sqlite_path:
            # Antigravity declares no database member, so
            # ``is_declared_logical_export`` can never hold here. Without an
            # undeclared-export gate a legacy PAGE IMAGE would be parsed into
            # sessions as current authority -- rebuildable state derived from
            # a snapshot that re-snapshots on every commit (polylogue-qjscw).
            if is_sqlite_page_image(sqlite_path):
                return []
            if antigravity.looks_like_trajectory_db_path(sqlite_path, immutable=True):
                return admit_parsed_sessions_for_publication(
                    collect_sqlite_sessions(provider, sqlite_path, fallback_id=fallback_id),
                    provider=provider,
                    source_path=source_path,
                )
    if looks_like_sqlite_bytes(payload):
        # polylogue-qjscw: a retained SQLite PAGE IMAGE reaching this point has
        # no current parser -- every provider that can replay a database has
        # already claimed its export above. Feeding these bytes to the JSON
        # stream parser raises, and on the frozen-candidate route that single
        # exception is promoted to FrozenSourceRemediationRequiredError, which
        # ends the WHOLE rebuild rather than failing one raw. The archive
        # deliberately retained this material, so it becomes terminal
        # non-session evidence instead: no session, no abort, and a receipt
        # that still names what the material was.
        return []
    from polylogue.sources.live.batch_support import jsonl_parse_input_of_handle

    with BytesIO(payload) as input_handle, ExitStack() as input_lifetime:
        input_is_jsonl = is_jsonl_source_path(source_path)
        classified_input = (
            input_lifetime.enter_context(jsonl_parse_input_of_handle(input_handle, check_stop=check_compute_cancelled))
            if input_is_jsonl
            else input_handle
        )
        classification = classify_artifact_stream(
            classified_input,
            provider=provider,
            source_path=source_path,
            wire_format="jsonl" if input_is_jsonl else "json",
            check_stop=check_compute_cancelled,
        )
    if classification.proved_non_session:
        return []
    if is_stream_record_provider(source_path, str(provider)):
        records = _retained_jsonl_records(payload, source_name, source_path)
        return parse_stream_payload(
            provider,
            records,
            fallback_id,
            source_path=source_path,
            profile_identity=profile_identity,
            sidecar_resolver=sidecar_resolver,
        )
    records = _retained_jsonl_records(payload, source_name, source_path)
    return parse_payload(
        provider,
        records,
        fallback_id,
        source_path=source_path,
        profile_identity=profile_identity,
        sidecar_resolver=sidecar_resolver,
    )


def _iter_sqlite_path(
    provider: Provider,
    path: Path,
    source_path: str,
    store: SqliteMessageStore,
    *,
    fallback_id: str,
    profile_identity: str | None,
) -> Generator[ParsedSession, None, None]:
    """Replay logical SQLite material through the preparation owner's sinks."""
    if provider is Provider.HERMES:
        if profile_identity is None:
            raise MissingProfileIdentityError("retained Hermes input has no captured profile identity receipt")
        if not (is_declared_logical_export(path, source_path) or is_undeclared_logical_export(path, source_path)):
            raise RuntimeError(f"retained Hermes SQLite material is not a logical export: {source_path}")
        if hermes_state.looks_like_state_db_path(path, immutable=True):
            sessions = iter_sqlite_sessions(
                provider, path, store, fallback_id=fallback_id, profile_identity=profile_identity
            )
        elif hermes_verification.looks_like_verification_evidence_db_path(path, immutable=True):
            sessions = (
                session
                for session in hermes_verification.parse_verification_evidence_db(
                    path,
                    fallback_id=fallback_id,
                    profile_identity=profile_identity,
                    immutable=True,
                )
            )
        else:
            return
    elif provider is Provider.ANTIGRAVITY:
        if is_sqlite_page_image(path) or not antigravity.looks_like_trajectory_db_path(path, immutable=True):
            return
        sessions = iter_sqlite_sessions(provider, path, store, fallback_id=fallback_id)
    else:
        raise ValueError(f"SQLite replay is not supported for {provider}")
    with closing(sessions) as selected_sessions:
        for session in selected_sessions:
            check_compute_cancelled()
            if admit_parsed_sessions_for_publication([session], provider=provider, source_path=source_path):
                yield session


def _parse_sqlite_path(
    provider: Provider,
    path: Path,
    source_path: str,
    *,
    fallback_id: str,
    profile_identity: str | None,
) -> list[ParsedSession]:
    """Parse a retained SQLite export from its immutable blob path."""
    if provider is Provider.HERMES:
        if profile_identity is None:
            raise MissingProfileIdentityError("retained Hermes input has no captured profile identity receipt")
        if not (is_declared_logical_export(path, source_path) or is_undeclared_logical_export(path, source_path)):
            raise RuntimeError(f"retained Hermes SQLite material is not a logical export: {source_path}")
        if hermes_state.looks_like_state_db_path(path, immutable=True):
            sessions = collect_sqlite_sessions(
                provider,
                path,
                fallback_id=fallback_id,
                profile_identity=profile_identity,
            )
        elif hermes_verification.looks_like_verification_evidence_db_path(path, immutable=True):
            sessions = hermes_verification.parse_verification_evidence_db(
                path,
                fallback_id=fallback_id,
                profile_identity=profile_identity,
                immutable=True,
            )
        else:
            sessions = []
    elif provider is Provider.ANTIGRAVITY:
        if is_sqlite_page_image(path):
            return []
        sessions = (
            collect_sqlite_sessions(provider, path, fallback_id=fallback_id)
            if antigravity.looks_like_trajectory_db_path(path, immutable=True)
            else []
        )
    else:
        raise ValueError(f"SQLite replay is not supported for {provider}")
    return admit_parsed_sessions_for_publication(sessions, provider=provider, source_path=source_path)


@contextmanager
def _sqlite_payload_path(
    payload: bytes,
    payload_path: Path | None,
    archive_root: Path | None,
) -> Iterator[Path]:
    """Yield a real filesystem path for SQLite-shaped raw revision bytes.

    ``sqlite3.connect`` cannot open in-memory bytes. Prefer the already-
    materialized blob path (no copy); only spill to a bounded temp file when
    no real path is available (e.g. the blob is not yet flushed to disk).
    """
    if payload_path is not None:
        yield payload_path
        return
    scratch_dir = archive_root if archive_root is not None else Path(tempfile.gettempdir())
    fd, name = tempfile.mkstemp(prefix=".revision-sqlite-spill-", suffix=".sqlite", dir=scratch_dir)
    os.close(fd)
    temp_path = Path(name)
    try:
        temp_path.write_bytes(payload)
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


def _parse_stream(
    provider: Provider,
    payload: BinaryIO,
    source_path: str,
    *,
    profile_identity: str | None = None,
    fallback_id_override: str | None = None,
    archive_root: Path | None = None,
    sidecar_resolver: SidecarResolver | None,
) -> list[ParsedSession]:
    return _parse_stream_raw(
        provider,
        payload,
        source_path,
        fallback_id_override=fallback_id_override,
        archive_root=archive_root,
        sidecar_resolver=sidecar_resolver,
        profile_identity=profile_identity,
    )


def _parse_stream_raw(
    provider: Provider,
    payload: BinaryIO,
    source_path: str,
    *,
    profile_identity: str | None = None,
    fallback_id_override: str | None = None,
    archive_root: Path | None = None,
    sidecar_resolver: SidecarResolver | None,
) -> list[ParsedSession]:
    if provider is Provider.HERMES and profile_identity is None:
        raise MissingProfileIdentityError("retained Hermes input has no captured profile identity receipt")

    source_name = Path(source_path).name
    fallback_id = fallback_id_override or fallback_session_id(source_path, source_path)
    stream = _retained_jsonl_stream(payload, source_name, source_path)
    return parse_stream_payload(
        provider,
        stream,
        fallback_id,
        source_path=source_path,
        profile_identity=profile_identity,
        sidecar_resolver=sidecar_resolver,
    )


__all__ = [
    "raw_authority_parser_fingerprint",
    "RetainedSessionEnricher",
    "PreparedRevisionReplayResult",
    "RetainedRawRetryableFailure",
    "RetainedReplayOutcome",
    "RevisionCensusResult",
    "apply_prepared_revision_replay",
    "apply_prepared_revision_census",
    "enrich_sessions_from_retained_read",
    "open_retained_session_enricher",
    "uncensused_historical_revision_raw_ids",
    "parse_retained_raw_sessions",
]


def session_enrichment_evidence_key_from_reader(
    *,
    provider: Provider,
    source_path: str | None,
    native_id: str,
    evidence_reader: RetainedEnrichmentRead,
) -> str | None:
    """Resolve the same current key through actual ordinary or prepared inputs."""
    if provider not in _SESSION_EVIDENCE_PROVIDERS or not source_path or not native_id:
        return None
    data = _retained_enrichment_sidecar_data(
        provider=provider,
        sessions=(),
        provider_session_ids=[native_id],
        evidence_reader=evidence_reader,
        source_path=source_path,
        captured_zip_coordinate=None,
    )
    key = enrichment_evidence_key(provider, data, native_id)
    if provider is Provider.CLAUDE_CODE:
        hook_key = evidence_reader.hook_tool_response_evidence_digest(
            origin=origin_from_provider(provider).value,
            session_native_ids=tuple(dict.fromkeys((native_id, native_id.split(":", 1)[0]))),
        )
        if hook_key is not None:
            encoded = json.dumps((provider.value, key, "hook_tool_responses", hook_key), separators=(",", ":"))
            key = hashlib.sha256(encoded.encode("utf-8", "surrogatepass")).hexdigest()
    return key


def _append_session_native_id(
    logical_key: str | None,
    *,
    provider: Provider,
    captured_native_id: str | None,
) -> str | None:
    if logical_key is None:
        return captured_native_id
    try:
        key = canonical_authority_logical_key(logical_key)
    except ValueError:
        return captured_native_id
    origin, _separator, session_native_id = key.partition(":")
    return session_native_id if origin == origin_from_provider(provider).value else captured_native_id


def _sealed_retained_sessions(
    prepared: PreparedRetainedInput,
    *,
    stop: Callable[[], bool] | None = None,
) -> Sequence[ParsedSession] | Exception:
    """Read only this creator's actual sealed parser carrier.

    Source descriptor/currency validation belongs to the original parent
    before admission. Publication consumes the retained carrier without
    rehydrating Source or reopening an enrichment generation.
    """
    if prepared.parser_error is not None:
        if prepared.retained_zip_membership_unproved:
            return RetainedZipMembershipUnprovedError(prepared.parser_error)
        if prepared.missing_profile_identity:
            return MissingProfileIdentityError(prepared.parser_error)
        if prepared.unsupported_shape:
            return UnsupportedRetainedJsonShapeError(prepared.parser_error)
        return retained_parse_exception(prepared.parser_error, prepared.parser_decode_failure)
    artifact = prepared.prepared_artifact
    if artifact is None:
        raise RetainedPreparationRetryableError(f"prepared retained carrier is missing for raw {prepared.raw_id}")
    if (
        artifact.blob_hash != prepared.blob_hash
        or artifact.error is not None
        or artifact.captured_profile_key != prepared.captured_profile_key
    ):
        raise RetainedPreparationRetryableError(f"prepared retained artifact changed for raw {prepared.raw_id}")
    if artifact.enrichment_index_path is not None or artifact.enrichment_digest is not None:
        raise RetainedPreparationRetryableError(
            f"retained artifact carries an unrelated enrichment owner: {prepared.raw_id}"
        )
    try:
        artifact.verify_files(full=False, stop=stop)
        return artifact.session_sequence()
    except DaemonOperationCancelled:
        raise
    except (OSError, ValueError) as exc:
        raise RetainedPreparationRetryableError(
            f"prepared retained artifact unavailable for raw {prepared.raw_id}"
        ) from exc


def _superseded_full_revision_identity(
    seal: PreparedIndexMutation,
    evidence_reader: PreparedSessionSourceRead,
    raw_id: str,
    revision_kind: RawRevisionKind,
    sessions: Sequence[ParsedSession],
) -> bool:
    """Whether a quarantined full revision is typed under an identity it no longer parses to.

    Byte-proven revisions keep their byte authority; only an unproven full
    revision may move, and only when the current parser derives exactly one
    identity that differs from the stored one.
    """
    from polylogue.storage.sqlite.archive_tiers.revision_governance import prepared_raw_typed_logical_key

    if revision_kind is not RawRevisionKind.FULL or len(sessions) != 1:
        return False
    if evidence_reader.raw_revision_authority(raw_id) == RawRevisionAuthority.BYTE_PROVEN.value:
        return False
    stored = prepared_raw_typed_logical_key(seal, raw_id)
    if stored is None:
        return False
    parsed = canonical_authority_logical_key(f"{sessions[0].source_name.value}:{sessions[0].provider_session_id}")
    try:
        return canonical_authority_logical_key(stored) != parsed
    except ValueError:
        return True


def prepare_revision_source_census(
    seal: PreparedIndexMutation,
    evidence_reader: PreparedSessionSourceRead,
    *,
    selected_raw_ids: Sequence[str],
    prepared_inputs: Mapping[str, PreparedRetainedInput],
) -> _RevisionCensusState:
    """Prepare the canonical parser census before admitting its Source writer.

    The parent owns the original read window and selected Source phase. Every
    observation uses that same state, including earlier staged refinements.
    Blob placement precedes this phase; this body never commits or opens an
    ArchiveStore, and the parent publishes its original captured tape.
    """
    from polylogue.sources.codex_state_evidence import prepare_codex_state_source_terminal
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        _PreparedSourceProducer,
        prepare_raw_state_update,
        prepared_parser_census_is_current,
        refine_prepared_raw_origin,
        replace_raw_membership_census,
    )

    state = _RevisionCensusState(0, 0, 0, set(), {}, set())
    producer = _PreparedSourceProducer(seal)

    def stage_current_parser_followup(raw_id: str, source_index: int) -> bool:
        """Finish Source obligations not represented by a current parser receipt."""
        if source_index >= 0 and _retained_page_image_raw(evidence_reader, raw_id):
            provider, _blob_hash, source_path, revision_kind, _size = evidence_reader.raw_revision_descriptor(raw_id)
            observed_at_ms = evidence_reader.raw_revision_observation_order(raw_id)[0]
            _persist_legacy_page_image_artifact(
                producer,
                raw_id,
                provider=provider,
                source_path=source_path,
                source_index=source_index,
                observed_at_ms=observed_at_ms,
                manage_transaction=False,
            )
            replace_raw_membership_census(
                seal,
                raw_id,
                [],
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=0,
                detail=LEGACY_PAGE_IMAGE_CENSUS_DETAIL,
                retire_full_revision_governance=revision_kind is RawRevisionKind.FULL,
                revision_authority=None,
            )
            return True
        prepared = prepared_inputs.get(raw_id)
        artifact = prepared.prepared_artifact if prepared is not None else None
        schema_validation_required = evidence_reader.raw_schema_eligible(raw_id)
        stream = artifact.stream_classification() if artifact is not None else None
        if (
            source_index >= 0
            and stream is not None
            and stream.classification.parse_as_session
            and not stream.classification.schema_eligible
            and artifact is not None
            and prepared is not None
            and prepared.validation_verdict is None
            and schema_validation_required
        ):
            # An old parser receipt omitted the native grammar's captured
            # schema policy. Its retained SQLite format proves the native
            # grammar; coarse JSON taxonomy cannot supply this exemption.
            # Publish the actual prepared outcome and taxonomy together.
            native_path = evidence_reader.raw_revision_blob_path(raw_id)
            if native_path is not None and looks_like_logical_source_path(native_path):
                apply_outcome(raw_id, source_index)
                return True
        # A current non-session parser census alone is insufficient: an
        # eligible structured document may legitimately yield no sessions.
        # The provider's exact raw-only path declaration is the independent
        # evidence that this revision does not require session validation.
        if schema_validation_required:
            provider, _blob_hash, source_path, _revision_kind, _size = evidence_reader.raw_revision_descriptor(raw_id)
            schema_validation_required = not path_declaration_refuses_session(provider, source_path)
        staged = False
        if not schema_validation_required and not evidence_reader.raw_parser_confirmed_non_session(raw_id):
            # The current parser authority already proves this typed raw-only
            # input. Its independent membership receipt may still name the
            # previous parser; finish that prerequisite before replay.
            byte_proven = evidence_reader.raw_revision_authority(raw_id) == RawRevisionAuthority.BYTE_PROVEN.value
            record_prepared_membership_census_receipt(
                seal,
                raw_id,
                parser_fingerprint=raw_authority_parser_fingerprint(),
                status="non_session",
                member_count=0,
                censused_at_ms=0,
                detail="current parser authority confirms typed non-session input",
                revision_authority=RawRevisionAuthority.BYTE_PROVEN if byte_proven else None,
            )
            staged = True
        if artifact is None:
            if schema_validation_required:
                raise RetainedPreparationRetryableError(
                    f"current retained parser receipt lacks captured validation evidence for raw {raw_id}"
                )
            return staged
        if artifact.codex_state_kind is not None:
            # Append fragments have no stable artifact-observation coordinate.
            # Their parser receipt is byte-governed; never mint an artifact at -1.
            if source_index < 0:
                return False
            _provider, _blob_hash, source_path, _revision_kind, _size = evidence_reader.raw_revision_descriptor(raw_id)
            acquired_at_ms = evidence_reader.raw_revision_observation_order(raw_id)[0]
            _persist_codex_state_artifact(
                producer,
                raw_id,
                source_path=source_path,
                source_index=source_index,
                observed_at_ms=acquired_at_ms,
                state_kind=artifact.codex_state_kind,
                manage_transaction=False,
            )
            prepare_codex_state_source_terminal(
                seal,
                raw_id,
                prepared_state=artifact,
                state_kind=artifact.codex_state_kind,
                source_path=source_path,
                acquired_at_ms=acquired_at_ms,
                censused_at_ms=0,
                source_read=evidence_reader,
            )
            state.transient_non_session_raw_ids.add(raw_id)
            return True

        verdict = prepared.validation_verdict if prepared is not None else None
        if verdict is None:
            if schema_validation_required:
                raise RetainedPreparationRetryableError(
                    f"current retained parser receipt lacks captured validation evidence for raw {raw_id}"
                )
        else:
            blob_hash = evidence_reader.raw_revision_descriptor(raw_id)[1]
            if verdict.raw_id != raw_id or verdict.revision_sha256 != blob_hash:
                raise RetainedPreparationRetryableError(f"retained validation evidence changed for {raw_id}")
            if evidence_reader.raw_validation_mode(raw_id) != verdict.mode:
                prepare_raw_state_update(
                    seal,
                    raw_id,
                    state=RawSessionStateUpdate(
                        validation_status=verdict.status,
                        validation_error=verdict.first_diagnostic,
                        validation_drift_count=verdict.drift_count,
                        validation_provider=artifact.resolved_provider,
                        validation_mode=verdict.mode,
                    ),
                )
                staged = True
        return staged

    def apply_outcome(raw_id: str, source_index: int) -> None:
        check_compute_cancelled()
        state.scanned += 1
        state.censused.add(raw_id)
        if source_index < 0:
            stage_current_parser_followup(raw_id, source_index)
            if evidence_reader.raw_has_membership_authority(raw_id):
                record_current_parser_source_census(seal, raw_id)
            else:
                replace_raw_membership_census(
                    seal,
                    raw_id,
                    None,
                    parser_fingerprint=raw_authority_parser_fingerprint(),
                    censused_at_ms=0,
                    detail=BYTE_AUTHORITY_CENSUS_DETAIL,
                    revision_authority=RawRevisionAuthority.BYTE_PROVEN,
                )
            state.quarantined += 1
            return
        outcome = _prepared_retained_outcome(evidence_reader, raw_id, prepared_inputs)
        provider, _hash, source_path, revision_kind, _size = evidence_reader.raw_revision_descriptor(raw_id)
        observed_at_ms = evidence_reader.raw_revision_observation_order(raw_id)[0]
        if isinstance(outcome, Exception):
            if _persist_terminal_raw_refusal(
                producer,
                raw_id,
                outcome,
                provider=provider,
                source_path=source_path,
                source_index=source_index,
                observed_at_ms=observed_at_ms,
                manage_transaction=False,
            ):
                # Some legacy page images reach a parser-specific terminal
                # refusal (for example, a Hermes database without profile
                # identity) before the ordinary empty-result path. Preserve
                # both facts: the refusal explains the parser outcome, while
                # the binary artifact prevents the same SQLite bytes from
                # being reconsidered as session input on the next census.
                if _retained_page_image_raw(evidence_reader, raw_id):
                    _persist_legacy_page_image_artifact(
                        producer,
                        raw_id,
                        provider=provider,
                        source_path=source_path,
                        source_index=source_index,
                        observed_at_ms=observed_at_ms,
                        manage_transaction=False,
                    )
                    replace_raw_membership_census(
                        seal,
                        raw_id,
                        [],
                        parser_fingerprint=raw_authority_parser_fingerprint(),
                        censused_at_ms=0,
                        detail=LEGACY_PAGE_IMAGE_CENSUS_DETAIL,
                        retire_full_revision_governance=revision_kind is RawRevisionKind.FULL,
                        revision_authority=None,
                    )
                record_current_parser_source_census(seal, raw_id)
            else:
                # Any other retained parser failure (an unrecognized shape, a
                # parser exception) is this parser's settled answer for these
                # immutable bytes: parsing them again can only fail the same
                # way. It admits no session, so it settles as a non-session
                # census with typed terminal evidence; a failed census would
                # be re-censused on every pass without progress. A changed
                # parser fingerprint re-censuses the raw.
                _record_raw_failure_evidence(
                    producer,
                    raw_id,
                    provider=provider,
                    source_path=source_path,
                    source_index=source_index,
                    acquired_at_ms=observed_at_ms,
                    kind=(
                        RawFailureEvidenceKind.TERMINAL_UNKNOWN_EXPORT_NO_SESSION
                        if provider is Provider.UNKNOWN
                        else RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE
                    ),
                    manage_transaction=False,
                )
                _apply_source_raw_state_update(
                    producer,
                    raw_id,
                    state=_raw_parse_failure_state(provider, outcome),
                    manage_transaction=False,
                )
                replace_raw_membership_census(
                    seal,
                    raw_id,
                    [],
                    parser_fingerprint=raw_authority_parser_fingerprint(),
                    censused_at_ms=0,
                    detail=str(outcome),
                    retire_full_revision_governance=revision_kind is RawRevisionKind.FULL,
                    revision_authority=None,
                )
            state.quarantined += 1
            return
        sessions, _payload_bytes, _parsed_kind = outcome
        prepared = prepared_inputs.get(raw_id)
        artifact = prepared.prepared_artifact if prepared is not None else None
        verdict = prepared.validation_verdict if prepared is not None else None
        stream = artifact.stream_classification() if artifact is not None else None
        if verdict is not None and artifact is not None:
            if (
                verdict.raw_id != raw_id
                or verdict.revision_sha256 != evidence_reader.raw_revision_descriptor(raw_id)[1]
            ):
                raise RetainedPreparationRetryableError(f"retained validation evidence changed for {raw_id}")
            prepare_raw_state_update(
                seal,
                raw_id,
                state=RawSessionStateUpdate(
                    validation_status=verdict.status,
                    validation_error=verdict.first_diagnostic,
                    validation_drift_count=verdict.drift_count,
                    validation_provider=artifact.resolved_provider,
                    validation_mode=verdict.mode,
                ),
            )
        if artifact is not None and artifact.codex_state_kind is not None:
            _persist_codex_state_artifact(
                producer,
                raw_id,
                source_path=source_path,
                source_index=source_index,
                observed_at_ms=observed_at_ms,
                state_kind=artifact.codex_state_kind,
                manage_transaction=False,
            )
            prepare_codex_state_source_terminal(
                seal,
                raw_id,
                prepared_state=artifact,
                state_kind=artifact.codex_state_kind,
                source_path=source_path,
                acquired_at_ms=observed_at_ms,
                censused_at_ms=0,
                source_read=evidence_reader,
            )
            # The companion Index projection remains on its actual prepared
            # owner. A Source receipt alone cannot certify that projection.
            state.transient_non_session_raw_ids.add(raw_id)
            return
        if sessions:
            parsed_provider = Provider.from_string(sessions[0].source_name)
            strict_refusal = verdict is not None and verdict.strict_refusal
            artifact_observed = False
            if strict_refusal and artifact is not None:
                artifact_observed = _persist_strict_session_refusal_artifact(
                    producer,
                    raw_id,
                    provider=parsed_provider,
                    source_path=source_path,
                    source_index=source_index,
                    observed_at_ms=observed_at_ms,
                    stream_classification=artifact.stream_classification(),
                    manage_transaction=False,
                )
            if not artifact_observed:
                record_session_artifact_observation(
                    producer,
                    raw_id=raw_id,
                    provider=parsed_provider,
                    source_path=source_path,
                    source_index=source_index,
                    observed_at_ms=observed_at_ms,
                    manage_transaction=False,
                    captured_classification=(
                        dataclasses.replace(stream.classification, schema_eligible=True)
                        if stream is not None and verdict is not None
                        else stream.classification
                        if stream is not None
                        else None
                    ),
                )
            if provider is Provider.UNKNOWN:
                prepare_raw_state_update(seal, raw_id, state=RawSessionStateUpdate(payload_provider=parsed_provider))
            refine_prepared_raw_origin(seal, raw_id, origin_from_provider(parsed_provider))
        else:
            if artifact is None or artifact.resolved_provider is None:
                raise RetainedPreparationRetryableError(f"empty retained outcome has no captured provider: {raw_id}")
            resolved_provider = artifact.resolved_provider
            if provider is Provider.UNKNOWN and resolved_provider is not Provider.UNKNOWN:
                prepare_raw_state_update(seal, raw_id, state=RawSessionStateUpdate(payload_provider=resolved_provider))
            if evidence_reader.raw_captured_zip_coordinate(raw_id) is not None:
                # A ZIP member's provider is its export's, which acquisition of
                # the decoded container would have stamped as its origin; the
                # placeholder converges to it. A loose non-session input keeps
                # the placeholder and records only its detected provider.
                refine_prepared_raw_origin(seal, raw_id, origin_from_provider(resolved_provider))
            stream_classification = artifact.stream_classification()
            if (
                stream_classification is not None
                and stream_classification.classification.parse_as_session
                and not stream_classification.classification.schema_eligible
                and verdict is None
            ):
                native_path = evidence_reader.raw_revision_blob_path(raw_id)
                if native_path is not None and looks_like_logical_source_path(native_path):
                    # An empty native parse still owes its captured grammar's
                    # schema policy; its independent non-session census stays
                    # on the ordinary receipt path below.
                    record_session_artifact_observation(
                        producer,
                        raw_id=raw_id,
                        provider=resolved_provider,
                        source_path=source_path,
                        source_index=source_index,
                        observed_at_ms=observed_at_ms,
                        manage_transaction=False,
                        captured_classification=stream_classification.classification,
                    )
            if _retained_page_image_raw(evidence_reader, raw_id):
                _persist_legacy_page_image_artifact(
                    producer,
                    raw_id,
                    provider=resolved_provider,
                    source_path=source_path,
                    source_index=source_index,
                    observed_at_ms=observed_at_ms,
                    manage_transaction=False,
                )
                terminalized = True
            else:
                terminalized = _persist_terminal_non_session_artifact(
                    producer,
                    raw_id,
                    provider=resolved_provider,
                    observed_at_ms=observed_at_ms,
                    source_path=source_path,
                    source_index=source_index,
                    stream_classification=stream_classification,
                    manage_transaction=False,
                )
            # A hook-event carrier is a physical append chain: its full
            # baseline keeps that binding for the tails grown onto it, so its
            # census never retires it to membership governance.
            carrier = (
                terminalized
                and stream_classification is not None
                and stream_classification.classification.kind is ArtifactKind.HOOK_EVENT_CARRIER
            )
            if resolved_provider is not Provider.UNKNOWN:
                byte_proven = evidence_reader.raw_revision_authority(raw_id) == RawRevisionAuthority.BYTE_PROVEN.value
                if revision_kind in {RawRevisionKind.FULL, RawRevisionKind.APPEND} and (byte_proven or carrier):
                    if terminalized:
                        record_prepared_membership_census_receipt(
                            seal,
                            raw_id,
                            parser_fingerprint=raw_authority_parser_fingerprint(),
                            status="non_session",
                            member_count=0,
                            censused_at_ms=0,
                            detail="",
                            revision_authority=RawRevisionAuthority.BYTE_PROVEN if byte_proven else None,
                        )
                    record_current_parser_source_census(seal, raw_id, parser_sessions=sessions)
                    if not terminalized:
                        prepare_raw_state_update(seal, raw_id, state=_raw_parse_success_state(resolved_provider))
                    return
                replace_raw_membership_census(
                    seal,
                    raw_id,
                    [],
                    parser_fingerprint=raw_authority_parser_fingerprint(),
                    censused_at_ms=0,
                    detail=LEGACY_PAGE_IMAGE_CENSUS_DETAIL if _retained_page_image_raw(evidence_reader, raw_id) else "",
                    retire_full_revision_governance=revision_kind is RawRevisionKind.FULL,
                    revision_authority=None,
                )
                if not terminalized:
                    prepare_raw_state_update(seal, raw_id, state=_raw_parse_success_state(resolved_provider))
                return
        state.classified += int(len(sessions) == 1)
        pending = evidence_reader.raw_has_pending_envelope(raw_id)
        # A grouped acquisition retains native_id=NULL even when its parser
        # currently yields one session; its original membership governs that
        # session. A native singleton learns its own key through the census.
        # A complete session record stream (a Codex rollout, a Claude Code
        # transcript) is one session's own byte stream by its declared shape,
        # whatever native_id the acquisition recorded: live intake acquires it
        # before parsing, and its appends need that byte-revision chain. A
        # Claude Code transcript under ``projects/<proj>/<uuid>.jsonl`` is
        # declared as the coordinator session stream.
        stream = artifact.stream_classification() if artifact is not None else None
        native_stream = stream is not None and stream.classification.kind in _NATIVE_SESSION_STREAM_KINDS
        grouped = evidence_reader.raw_native_id(raw_id) is None and not native_stream
        if (
            len(sessions) == 1
            and not grouped
            and (revision_kind is RawRevisionKind.UNKNOWN or pending)
            and not evidence_reader.raw_has_membership_authority(raw_id)
        ):
            record_current_parser_source_census(seal, raw_id, parser_sessions=sessions)
        elif revision_kind is RawRevisionKind.UNKNOWN or (
            pending
            and (grouped or len(sessions) > 1 or evidence_reader.raw_has_membership_governed_pending_envelope(raw_id))
        ):
            replace_raw_membership_census(
                seal,
                raw_id,
                sessions,
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=0,
                revision_authority=None,
            )
            for session in sessions:
                logical_key = canonical_authority_logical_key(
                    f"{session.source_name.value}:{session.provider_session_id}"
                )
                state.membership_candidates.setdefault(logical_key, set()).add(raw_id)
        elif _superseded_full_revision_identity(seal, evidence_reader, raw_id, revision_kind, sessions):
            # The stored key came from a parser that no longer derives it. A
            # receipt here could never agree with that key, so every pass would
            # re-census it unchanged. Retire it to membership governance under
            # the identity the current parser derives, beside any raw already
            # typed under that identity, so the cohort is arbitrated jointly.
            replace_raw_membership_census(
                seal,
                raw_id,
                sessions,
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=0,
                detail=SUPERSEDED_IDENTITY_GOVERNANCE_DETAIL,
                retire_full_revision_governance=True,
                revision_authority=RawRevisionAuthority.QUARANTINED,
            )
            for session in sessions:
                logical_key = canonical_authority_logical_key(
                    f"{session.source_name.value}:{session.provider_session_id}"
                )
                state.membership_candidates.setdefault(logical_key, set()).add(raw_id)
        else:
            record_current_parser_source_census(seal, raw_id, parser_sessions=sessions)

    selection = tuple(selected_raw_ids)
    while True:
        rows = evidence_reader.raw_membership_census_rows(selection)
        for raw_id, source_index, terminal_non_session, _rowid in sorted(
            rows,
            key=lambda row: evidence_reader.raw_revision_observation_order(row[0])[1],
        ):
            check_compute_cancelled()
            if raw_id in state.censused:
                continue
            if prepared_parser_census_is_current(seal, raw_id):
                # A current parser receipt may still lack the independent
                # validation policy or typed non-session Source result needed
                # before replay. Complete those captured obligations without
                # re-parsing and risking a conflicting authority update.
                stage_current_parser_followup(raw_id, source_index)
                state.censused.add(raw_id)
                continue
            prepared = prepared_inputs.get(raw_id)
            artifact = prepared.prepared_artifact if prepared is not None else None
            # A captured Codex state still needs its real Source material and
            # companion projection even if its earlier artifact is terminal.
            if terminal_non_session and (artifact is None or artifact.codex_state_kind is None):
                state.scanned += 1
                state.censused.add(raw_id)
                record_current_parser_source_census(seal, raw_id)
            else:
                apply_outcome(raw_id, source_index)
        expanded, _keys = evidence_reader.expand_raw_membership_selection(selection)
        if set(expanded) == set(selection):
            break
        if set(expanded).difference(prepared_inputs):
            # This census can reveal a previously unselected logical sibling.
            # Publish only this creator's prepared inputs; the resident owner's
            # next pass enrolls the expanded cohort on a fresh original frame.
            break
        selection = expanded
    return state


def prepare_revision_source_membership_conversion(
    seal: PreparedIndexMutation,
    evidence_reader: PreparedSessionSourceRead,
    *,
    logical_keys: Sequence[str],
    prepared_inputs: Mapping[str, PreparedRetainedInput],
    prepared_replay_plans: Mapping[str, tuple[str, ...]],
) -> int:
    """Census undecided full-only observations for canonical member comparison.

    Retain the original Raw governance, including already-proved byte claims.
    The parser census adds member evidence; the existing membership/head
    classifier still decides admission against the original accepted head.
    """
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        prepared_convertible_full_revision_raw_ids,
        replace_raw_membership_census,
    )

    converted = 0
    for logical_key in sorted(logical_keys):
        check_compute_cancelled()
        if is_work_event_raw_id(logical_key):
            continue
        if evidence_reader.pending_raw_envelope_has_membership_authority(logical_key):
            continue
        original_members = evidence_reader.raw_membership_logical_raw_ids(logical_key)
        if logical_key not in prepared_replay_plans and not original_members:
            raise RetainedPreparationRetryableError(f"prepared byte selection is missing for {logical_key}")
        cohort_raw_ids = prepared_convertible_full_revision_raw_ids(seal, logical_key)
        if prepared_replay_plans.get(logical_key) and not any(
            evidence_reader.raw_revision_authority(raw_id) == RawRevisionAuthority.QUARANTINED.value
            for raw_id in cohort_raw_ids
        ):
            continue
        for raw_id in cohort_raw_ids:
            if evidence_reader.raw_has_membership_authority(raw_id):
                continue
            check_compute_cancelled()
            outcome = _prepared_retained_outcome(evidence_reader, raw_id, prepared_inputs)
            if isinstance(outcome, Exception):
                raise RetainedPreparationRetryableError(
                    f"non-prefix membership input refused for {raw_id}"
                ) from outcome
            sessions, _payload_bytes, _kind = outcome
            prepared_artifact = prepared_inputs[raw_id].prepared_artifact if raw_id in prepared_inputs else None
            stream = prepared_artifact.stream_classification() if prepared_artifact is not None else None
            if not sessions and stream is not None and stream.proved_non_session:
                # Proven non-session evidence (a hook-event carrier) has no
                # membership to convert; its byte revision chain governs it.
                continue
            if len(sessions) != 1:
                raise RetainedPreparationRetryableError(f"full revision {raw_id} no longer parses to one session")
            replace_raw_membership_census(
                seal,
                raw_id,
                sessions,
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=0,
                detail=HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
                retire_full_revision_governance=False,
                revision_authority=RawRevisionAuthority.QUARANTINED,
            )
            converted += 1
    return converted


@dataclass(frozen=True, slots=True)
class PreparedRevisionSourceCensus:
    """The original Source tape and detached census outcome for one component.

    A census may carry the byte classification staged on top of it in the
    same tape; ``classification_blob_stats`` are its inputs, which must be
    unchanged at publication.
    """

    permit: KnownTierMutationPermit
    result: RevisionCensusResult
    classification_result: RevisionCensusResult | None = None
    classification_blob_stats: tuple[tuple[str, tuple[int, int, int, int, int]], ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedRevisionSourceClassification:
    permit: KnownTierMutationPermit
    result: RevisionCensusResult
    blob_stats: tuple[tuple[str, tuple[int, int, int, int, int]], ...]


def prepare_revision_source_classification(
    seal: PreparedIndexMutation,
    evidence_reader: PreparedSessionSourceRead,
    *,
    selected_raw_ids: Sequence[str],
    logical_keys: Sequence[str],
    payload_store: BlobStore,
) -> PreparedRevisionSourceClassification | None:
    """Prepare byte authority on the same selected Source state as the census."""
    changed, blob_stats = stage_revision_source_classification(
        seal, evidence_reader, logical_keys=logical_keys, payload_store=payload_store
    )
    if not changed:
        return None
    return PreparedRevisionSourceClassification(
        seal.prepare_source_mutation(),
        RevisionCensusResult(0, 0, 0, tuple(selected_raw_ids), tuple(logical_keys)),
        blob_stats,
    )


def stage_revision_source_classification(
    seal: PreparedIndexMutation,
    evidence_reader: PreparedSessionSourceRead,
    *,
    logical_keys: Sequence[str],
    payload_store: BlobStore,
) -> tuple[bool, tuple[tuple[str, tuple[int, int, int, int, int]], ...]]:
    """Stage byte authority on the seal's selected Source state, census included.

    Returns whether any cohort changed and the byte identities the
    classification read, which publication requires unchanged.
    """
    from polylogue.storage.sqlite.archive_tiers.revision_governance import prepare_raw_revision_byte_classification

    changed = False
    blob_stats: dict[str, tuple[int, int, int, int, int]] = {}
    with seal.original_read_snapshot(), seal.source_producer():
        for logical_key in sorted(logical_keys):
            check_compute_cancelled()
            if evidence_reader.pending_raw_envelope_has_membership_authority(
                logical_key
            ) or evidence_reader.raw_membership_logical_raw_ids(logical_key):
                continue
            key_changed, key_stats = prepare_raw_revision_byte_classification(
                seal, logical_key, payload_store=payload_store
            )
            changed = changed or key_changed
            for blob_hash, identity in key_stats:
                previous = blob_stats.get(blob_hash)
                if previous is not None and previous != identity:
                    raise PreparedRawClassificationStaleError("classification input changed between logical cohorts")
                blob_stats[blob_hash] = identity
    return changed, tuple(blob_stats.items())


def apply_prepared_revision_classification(
    seal: PreparedIndexMutation,
    prepared: PreparedRevisionSourceClassification,
    *,
    payload_store: BlobStore,
) -> RevisionCensusResult:
    """Validate the original byte seals, then apply its canonical Source tape."""
    from polylogue.storage.sqlite.archive_tiers.revision_governance import publish_prepared_revision_source

    _require_classification_inputs_current(prepared.blob_stats, payload_store)
    publish_prepared_revision_source(seal, prepared.permit)
    return prepared.result


class RetainedReplayScheduleRead(Protocol):
    """The canonical representative-raw selector on the parent's Source view."""

    def replay_representative_rows(self, keys: Sequence[str]) -> AbstractContextManager[sqlite3.Cursor]: ...


@dataclass(frozen=True, slots=True)
class PreparedRetainedReplaySource:
    permit: KnownTierMutationPermit
    attachment_artifacts: Mapping[str, PreparedJsonl]
    membership_plans: Mapping[str, PreparedMembershipReplay]
    terminal_raw_ids: frozenset[str]
    original_index_outputs: Mapping[str, tuple[bytes | None, int] | None]


def _accepted_marker_request_session_binding(session: object) -> dict[str, object]:
    """Build the small, complete session identity record used by marker replay."""
    from polylogue.pipeline.ids import session_content_hash
    from polylogue.sources.parsers.base_models import ParsedSession

    if not isinstance(session, ParsedSession):
        raise TypeError("accepted marker request requires a parsed session")
    content_hash = str(session_content_hash(session))
    if session.content_hash is not None and session.content_hash != content_hash:
        raise RetainedPreparationRetryableError("accepted marker session hash changed during preparation")
    binding = session.model_dump(
        mode="json",
        exclude={"messages", "session_events", "attachments", "unit_accounting"},
    )
    binding["content_hash"] = content_hash
    binding["session_id"] = str(make_session_id(session.source_name, session.provider_session_id))
    accounting = session.unit_accounting
    binding["unit_accounting_digest"] = None if accounting is None else accounting.stable_binding_digest()
    return binding


def _prepared_accepted_marker_sessions(
    *,
    raw_id: str,
    artifact: PreparedJsonl,
    selected_session_ids: set[str],
    prepared_writes: Mapping[tuple[str, str], PreparedSessionWrite],
    marker_write_factory: Callable[[str, ParsedSession], PreparedSessionWrite],
) -> Generator[tuple[str, PreparedSessionWrite, tuple[object, ...]], None, None]:
    """Stream canonical marker writes, closing each owned temporary after use."""
    from polylogue.sources.parsers.base_models import ParsedSession as ParsedSessionModel

    found_binding_digests: dict[str, str] = {}
    with closing(artifact.iter_sessions()) as parsed_sessions:
        for session in parsed_sessions:
            if not isinstance(session, ParsedSessionModel):
                raise RetainedPreparationRetryableError(
                    f"accepted marker request has an invalid retained session for {raw_id}"
                )
            binding = _accepted_marker_request_session_binding(session)
            session_id = binding["session_id"]
            if not isinstance(session_id, str) or session_id not in selected_session_ids:
                continue
            binding_digest = hashlib.sha256(
                json.dumps(binding, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            prior_binding_digest = found_binding_digests.get(session_id)
            if prior_binding_digest is not None:
                if prior_binding_digest != binding_digest:
                    raise RetainedPreparationRetryableError(
                        f"accepted marker request has conflicting parsed session {raw_id}:{session_id}"
                    )
                continue
            found_binding_digests[session_id] = binding_digest
            prepared = prepared_writes.get((raw_id, session_id))
            if prepared is None:
                prepared = marker_write_factory(raw_id, session)
                try:
                    yield session_id, prepared, ()
                finally:
                    prepared.close()
            else:
                yield session_id, prepared, ()
    missing = selected_session_ids - found_binding_digests.keys()
    if missing:
        missing_session_id = min(missing)
        raise RetainedPreparationRetryableError(
            f"accepted marker request lost selected parsed session {raw_id}:{missing_session_id}"
        )


def prepare_retained_replay_source(
    seal: PreparedIndexMutation,
    *,
    prepared_inputs: Mapping[str, PreparedRetainedInput],
    byte_outcomes: Mapping[str, PreparedRevisionReplayOutcome],
    membership_plans: Mapping[str, PreparedMembershipReplay],
    adoptions: Mapping[tuple[str, tuple[str, ...]], PreparedRevisionAdoption],
    prepared_writes: Mapping[tuple[str, str], PreparedSessionWrite],
    marker_write_factory: Callable[[str, ParsedSession], PreparedSessionWrite],
) -> PreparedRetainedReplaySource:
    """Stage exact acknowledgements for the original ordered Index outcomes."""
    from polylogue.markers.preparation import marker_recipe_fingerprint
    from polylogue.storage.accepted_marker_producer import (
        accepted_marker_input_is_durable,
        prepare_accepted_marker_carrier,
        stage_accepted_marker_carrier,
    )
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        membership_decisions_for_head_plan,
        prepare_membership_classification_source,
        prepare_raw_parse_success,
        revision_replay_terminal_raw_ids,
    )
    from polylogue.storage.sqlite.archive_tiers.session_suppression import _reader_suppresses

    attachment_artifacts: dict[str, PreparedJsonl] = {}
    selected_membership = dict(membership_plans)
    terminal_raw_ids: set[str] = set()
    produced_session_ids: set[str] = set()
    original_index_outputs: dict[str, tuple[bytes | None, int] | None] = {}
    decided_at_ms = int(datetime.now(UTC).timestamp() * 1000)
    marker_recipe = marker_recipe_fingerprint()
    marker_sessions_by_raw: dict[str, set[str]] = {}
    with seal.original_read_snapshot(), seal.source_producer():

        def request_sessions_for(raw_id: str) -> Callable[[], Iterable[Mapping[str, object]]]:
            retained = prepared_inputs[raw_id]
            artifact = retained.prepared_artifact
            if artifact is None:
                raise RetainedPreparationRetryableError(
                    f"accepted marker request has no canonical retained parse for {raw_id}"
                )

            def sessions() -> Iterator[Mapping[str, object]]:
                from contextlib import closing

                from polylogue.sources.parsers.base_models import ParsedSession

                with closing(artifact.iter_sessions()) as parsed_sessions:
                    for session in parsed_sessions:
                        if not isinstance(session, ParsedSession):
                            raise RetainedPreparationRetryableError(
                                f"accepted marker request has an invalid retained session for {raw_id}"
                            )
                        binding = _accepted_marker_request_session_binding(session)
                        # Content and accounting are both bound without
                        # serializing their disk-backed or spilled arrays.
                        yield binding

            return sessions

        def accepted_marker_facts(raw_id: str) -> dict[str, str]:
            retained = prepared_inputs[raw_id]
            return {
                "blob_hash": retained.blob_hash,
                "provider": retained.provider.value,
                "revision_kind": retained.revision_kind.value,
                "source_path": retained.source_path,
                "parser_fingerprint": retained.parser_fingerprint,
                "marker_recipe": marker_recipe,
            }

        def retain_accepted_marker_session(raw_id: str, session_id: str) -> None:
            marker_sessions_by_raw.setdefault(raw_id, set()).add(session_id)

        def stage_accepted_marker_history(raw_id: str, selected_session_ids: set[str]) -> None:
            verdict = prepared_inputs[raw_id].validation_verdict
            if verdict is not None and verdict.strict_refusal:
                return
            request_sessions = request_sessions_for(raw_id)
            facts = accepted_marker_facts(raw_id)
            if accepted_marker_input_is_durable(
                seal,
                raw_id=raw_id,
                request_facts=facts,
                request_sessions=request_sessions,
            ):
                return
            artifact = prepared_inputs[raw_id].prepared_artifact
            if artifact is None:
                raise RetainedPreparationRetryableError(
                    f"accepted marker request has no canonical retained parse for {raw_id}"
                )

            with closing(
                _prepared_accepted_marker_sessions(
                    raw_id=raw_id,
                    artifact=artifact,
                    selected_session_ids=selected_session_ids,
                    prepared_writes=prepared_writes,
                    marker_write_factory=marker_write_factory,
                )
            ) as marker_sessions:
                carrier = prepare_accepted_marker_carrier(
                    raw_id=raw_id,
                    request_facts=facts,
                    request_sessions=request_sessions,
                    prepared_sessions=marker_sessions,
                )
                try:
                    stage_accepted_marker_carrier(seal, carrier)
                finally:
                    carrier.close()

        def retain_attachment_carrier(raw_id: str) -> None:
            if raw_id in attachment_artifacts:
                return
            retained = prepared_inputs[raw_id]
            artifact = retained.prepared_artifact
            if artifact is None:
                raise RetainedPreparationRetryableError(f"attachment preparation has no actual carrier for {raw_id}")
            # The captured artifact survives preparation. Its lazy Source
            # lookup belongs to the actual publishing writer, not this read
            # window; a multi-session Raw must select its own session there.
            attachment_artifacts[raw_id] = artifact

        for logical_key, outcome in byte_outcomes.items():
            adoption = adoptions[(logical_key, outcome.plan.accepted_raw_ids)]
            if not adoption.adoptable or outcome.suppressed:
                continue
            if adoption.session_id is None:
                raise RetainedPreparationRetryableError("accepted byte acknowledgement has no session identity")
            for raw_id in outcome.plan.accepted_raw_ids:
                retain_accepted_marker_session(raw_id, adoption.session_id)
                retain_attachment_carrier(raw_id)
            for raw_id in revision_replay_terminal_raw_ids(outcome.plan):
                prepare_raw_parse_success(seal, raw_id, provider=prepared_inputs[raw_id].provider)
                terminal_raw_ids.add(raw_id)
            produced_session_ids.add(adoption.session_id)

        # A refused cohort's failure state goes last, so another key's
        # incomplete-cohort correction of a shared raw cannot clear it.
        for logical_key, plan in sorted(
            membership_plans.items(),
            key=lambda item: item[1].head_plan is not None and item[1].head_plan.conflict is not None,
        ):
            if plan.head_plan is None:
                raise RetainedPreparationRetryableError("membership acknowledgement has no original head decision")
            accepted = plan.classification.accepted_raw_ids
            skipped = False
            if accepted and plan.head_plan.yield_to_head_raw_id is None:
                adoption = adoptions[(logical_key, accepted)]
                session = plan.sessions[accepted[-1]]
                session_id = str(make_session_id(session.source_name, session.provider_session_id))
                skipped = not adoption.adoptable or _reader_suppresses(seal.observer("user"), session_id)
                if not skipped:
                    for raw_id in accepted:
                        retain_accepted_marker_session(
                            raw_id,
                            str(
                                make_session_id(
                                    plan.sessions[raw_id].source_name, plan.sessions[raw_id].provider_session_id
                                )
                            ),
                        )
                    retain_attachment_carrier(accepted[-1])
                    produced_session_ids.add(session_id)
            decisions = membership_decisions_for_head_plan(plan.classification, plan.head_plan, suppressed=skipped)
            prepare_membership_classification_source(
                seal,
                logical_key,
                plan.classification,
                decisions=decisions,
                decided_at_ms=decided_at_ms,
                projections=plan.projections,
                head_plan=plan.head_plan,
            )
            selected_membership[logical_key] = dataclasses.replace(
                plan,
                source_decisions=decisions,
                decided_at_ms=decided_at_ms,
            )
        for raw_id, session_ids in sorted(marker_sessions_by_raw.items()):
            stage_accepted_marker_history(raw_id, session_ids)
        for raw_id, session_id in prepared_writes:
            if not is_work_event_raw_id(raw_id):
                continue
            seal.before_index_input(
                "sessions",
                ("session_id",),
                "SELECT rowid FROM sessions WHERE session_id=?",
                (session_id,),
            )
            with seal.original_rows("index", "SELECT 1 FROM sessions WHERE session_id=?", (session_id,)) as rows:
                existing_session = rows.fetchone() is not None
            if not (existing_session or session_id in produced_session_ids) or _reader_suppresses(
                seal.observer("user"), session_id
            ):
                continue
            prepare_raw_parse_success(seal, raw_id, provider=prepared_inputs[raw_id].provider)
            terminal_raw_ids.add(raw_id)
        output_ids = {session_id for _raw_id, session_id in prepared_writes} | produced_session_ids
        for plan in membership_plans.values():
            if plan.head_plan is not None and plan.head_plan.existing_head is not None:
                output_ids.add(str(plan.head_plan.existing_head[0]))
        for session_id in sorted(output_ids):
            seal.before_index_input(
                "sessions",
                ("content_hash", "message_count"),
                "SELECT rowid FROM sessions WHERE session_id=?",
                (session_id,),
            )
            with seal.original_rows(
                "index", "SELECT content_hash,message_count FROM sessions WHERE session_id=?", (session_id,)
            ) as rows:
                row = rows.fetchone()
            original_index_outputs[session_id] = (
                None if row is None else (None if row[0] is None else bytes(row[0]), int(row[1]))
            )
    return PreparedRetainedReplaySource(
        seal.prepare_source_mutation(),
        attachment_artifacts,
        selected_membership,
        frozenset(terminal_raw_ids),
        original_index_outputs,
    )


def _required_prepared_write_for(
    prepared_writes: Mapping[tuple[str, str], PreparedSessionWrite],
    raw_id: str,
    session: ParsedSession,
) -> PreparedSessionWrite:
    """Require the original canonical carrier for every selected session write."""
    session_id = str(make_session_id(session.source_name, session.provider_session_id))
    prepared = prepared_writes.get((raw_id, session_id))
    if prepared is None:
        raise RetainedPreparationRetryableError(f"prepared retained session write is missing for {raw_id}:{session_id}")
    return prepared


@contextmanager
def _prepared_replay_archive(
    archive_root: Path,
    reference_seal: PreparedIndexMutation,
    selected_index_path: Path,
) -> Iterator[ArchiveStore]:
    """Open only the actual Index destination carried by retained preparation."""
    if selected_index_path.resolve(strict=True) != reference_seal.index_path:
        raise RetainedPreparationRetryableError("retained replay Index differs from its original preparation")
    destination = reference_seal.destination
    if destination is None:
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            if archive.index_db_path.resolve(strict=True) != reference_seal.index_path:
                raise RetainedPreparationRetryableError("retained replay active Index moved after preparation")
            yield archive
        return
    destination.validate()
    generation = destination.generation
    if destination.kind != "owned_inactive" or generation is None:
        raise RetainedPreparationRetryableError("retained replay lacks its actual owned generation")
    from polylogue.sources.live.cold_build import active_cold_build_generation

    cold_build = active_cold_build_generation(archive_root)
    if cold_build is not None and cold_build.generation == generation:
        with cold_build.open_writer() as archive:
            yield archive
            destination.validate()
        return
    with ArchiveStore.open_owned_inactive_generation(
        Path(generation.index_path).parent,
        generation_id=generation.generation_id,
        owner_id=generation.owner_id,
        # Preparation sealed the current layout, whether deferred or restored
        # by an interrupted readiness pass. Reopening must perform no index DDL.
        preserve_secondary_index_layout=True,
    ) as archive:
        yield archive
        destination.validate()


def enrich_sessions_from_retained_read(
    evidence_reader: RetainedEnrichmentRead,
    provider: Provider,
    source_path: str | None,
    sessions: Sequence[ParsedSession],
    *,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
    evidence_observer: Callable[[object], None] | None = None,
) -> list[ParsedSession]:
    """Collect the shared canonical one-bundle enrichment for explicit readers."""
    with closing(
        iter_enriched_sessions_from_retained_read(
            evidence_reader,
            provider,
            source_path,
            sessions,
            captured_zip_coordinate=captured_zip_coordinate,
            evidence_observer=evidence_observer,
        )
    ) as enriched:
        return list(enriched)


def iter_enriched_sessions_from_retained_read(
    evidence_reader: RetainedEnrichmentRead,
    provider: Provider,
    source_path: str | None,
    sessions: Sequence[ParsedSession],
    *,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
    evidence_observer: Callable[[object], None] | None = None,
    provider_session_ids: Iterable[str] | None = None,
    normalize_session: Callable[[ParsedSession], ParsedSession] | None = None,
) -> Generator[ParsedSession, None, None]:
    """Apply actual provider assembly to the same original/selected evidence."""
    from polylogue.sources.assembly import close_sidecar_data, get_assembly_spec

    if provider is Provider.UNKNOWN and sessions:
        provider = sessions[0].source_name
    spec = get_assembly_spec(provider)

    def recover_hook_results(session: ParsedSession) -> ParsedSession:
        if provider is not Provider.CLAUDE_CODE:
            return session
        from polylogue.sources.live.hook_tool_response import (
            recover_persisted_tool_results,
            unresolved_persisted_truncations,
        )

        truncations = unresolved_persisted_truncations(session)
        native_id = str(session.provider_session_id or "")
        if not native_id:
            return session
        session_native_ids = tuple(dict.fromkeys((native_id, native_id.split(":", 1)[0])))
        origin = origin_from_provider(provider).value
        recovered = session
        if truncations:
            responses = evidence_reader.hook_tool_responses(
                origin=origin,
                # Claude journals subagent calls under the parent native id.
                session_native_ids=session_native_ids,
                tool_use_ids=(item.tool_use_id for item in truncations),
            )
            recovered = recover_persisted_tool_results(session, responses=responses)
        hook_key = evidence_reader.hook_tool_response_evidence_digest(
            origin=origin, session_native_ids=session_native_ids
        )
        if hook_key is None:
            return recovered
        encoded = json.dumps(
            (provider.value, recovered.enrichment_evidence_key, "hook_tool_responses", hook_key),
            separators=(",", ":"),
        )
        combined_key = hashlib.sha256(encoded.encode("utf-8", "surrogatepass")).hexdigest()
        return recovered.model_copy(update={"enrichment_evidence_key": combined_key})

    if spec is None:
        for session in sessions:
            check_compute_cancelled()
            prepared = normalize_session(session) if normalize_session is not None else session
            yield recover_hook_results(prepared)
        return
    sidecar_data = _retained_enrichment_sidecar_data(
        provider=provider,
        sessions=sessions,
        evidence_reader=evidence_reader,
        source_path=source_path,
        captured_zip_coordinate=captured_zip_coordinate,
        provider_session_ids=provider_session_ids,
    )
    try:
        if evidence_observer is not None:
            evidence_observer(sidecar_data)
        for session in sessions:
            check_compute_cancelled()
            if normalize_session is not None:
                session = normalize_session(session)
            if (
                provider is Provider.CLAUDE_CODE
                and isinstance(session.messages, SqliteMessageSink)
                and sidecar_data.get("history_paste_index", {}).get(session.provider_session_id)
            ):
                # `_finalize_prepared_cohort` restores the original sealed
                # message sink without its parser writer. Keep this cloned
                # sink's owner alive across the yield: the downstream hash and
                # `PreparedJsonl.from_sessions` copy both consume it before
                # requesting the next enriched session.
                from polylogue.sources.assembly_claude_code import ClaudeCodeAssemblySpec
                from polylogue.sources.prepared_message_sink import SqliteMessageStore

                if not isinstance(spec, ClaudeCodeAssemblySpec):
                    raise RuntimeError("Claude Code provider resolved to a different assembly spec")
                scratch_root = Path("/realm/tmp/work")
                with (
                    tempfile.TemporaryDirectory(
                        prefix="polylogue-history-paste-output-",
                        dir=scratch_root if scratch_root.is_dir() else None,
                    ) as scratch_dir,
                    closing(SqliteMessageStore(Path(scratch_dir) / "messages.sqlite")) as output_store,
                ):
                    enriched = spec.enrich_session(
                        session,
                        sidecar_data,
                        message_sink_factory=output_store.new_sink,
                    )
                    yield recover_hook_results(stamp_enrichment_evidence(provider, sidecar_data, enriched))
            else:
                enriched = spec.enrich_session(session, sidecar_data)
                yield recover_hook_results(stamp_enrichment_evidence(provider, sidecar_data, enriched))
    finally:
        primary = sys.exception()
        try:
            close_sidecar_data(sidecar_data)
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup("retained enrichment and sidecar cleanup failed", [primary, cleanup]) from None
            raise


class RetainedEnrichmentRead(Protocol):
    """Provider metadata read through the current preparation's actual owner."""

    def retained_state_titles(self, thread_ids: Iterable[str], source_path: str | None) -> Mapping[str, str] | None: ...

    def hook_tool_responses(
        self,
        *,
        origin: str,
        session_native_ids: Iterable[str],
        tool_use_ids: Iterable[str],
    ) -> Mapping[str, Any]: ...

    def hook_tool_response_evidence_digest(self, *, origin: str, session_native_ids: Iterable[str]) -> str | None: ...

    def retained_assembly_evidence(
        self,
        sidecar_data: SidecarData,
        *,
        provider: Provider,
        source_path: str,
        captured_zip_coordinate: CapturedZipMemberCoordinate | None,
    ) -> SidecarData | None: ...


class ConnectionRetainedEnrichmentRead:
    """Borrow explicit existing ordinary evidence capabilities without reopening."""

    def __init__(
        self,
        index_connection: sqlite3.Connection | None,
        source_connection: sqlite3.Connection | None,
        blob_root: Path | None,
    ) -> None:
        self._index = index_connection
        self._source = source_connection
        self._blob_root = blob_root

    def retained_state_titles(self, thread_ids: Iterable[str], source_path: str | None) -> Mapping[str, str] | None:
        if self._index is None:
            return None
        from polylogue.sources.codex_state_projection import iter_thread_title_candidates
        from polylogue.sources.retained_title_index import RetainedTitleIndex

        with closing(iter_thread_title_candidates(self._index, thread_ids=thread_ids, source_path=source_path)) as rows:
            return RetainedTitleIndex(rows)

    def hook_tool_responses(
        self,
        *,
        origin: str,
        session_native_ids: Iterable[str],
        tool_use_ids: Iterable[str],
    ) -> Mapping[str, Any]:
        if self._source is None:
            return {}
        from polylogue.sources.live.hook_tool_response import read_hook_tool_responses

        return read_hook_tool_responses(
            self._source,
            origin=origin,
            session_native_ids=tuple(session_native_ids),
            tool_use_ids=tuple(tool_use_ids),
        )

    def hook_tool_response_evidence_digest(self, *, origin: str, session_native_ids: Iterable[str]) -> str | None:
        if self._source is None:
            return None
        from polylogue.sources.live.hook_tool_response import read_hook_tool_response_evidence_digest

        return read_hook_tool_response_evidence_digest(
            self._source,
            origin=origin,
            session_native_ids=tuple(session_native_ids),
        )

    def retained_assembly_evidence(
        self,
        sidecar_data: SidecarData,
        *,
        provider: Provider,
        source_path: str,
        captured_zip_coordinate: CapturedZipMemberCoordinate | None,
    ) -> SidecarData | None:
        if self._source is None or self._blob_root is None:
            return None
        from polylogue.sources.retained_assembly import with_retained_assembly_evidence
        from polylogue.storage.blob_store import BlobStore

        return with_retained_assembly_evidence(
            sidecar_data,
            provider=provider,
            source_read=ConnectionSessionSourceRead(self._source),
            blob_store=BlobStore(self._blob_root),
            source_path=source_path,
            captured_zip_coordinate=captured_zip_coordinate,
        )


class RetainedRawRead(Protocol):
    """The canonical parser's finite retained Source and Blob read capability."""

    @property
    def archive_root(self) -> Path: ...

    def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]: ...

    def raw_profile_identity(self, raw_id: str) -> str | None: ...

    def raw_captured_zip_coordinate(self, raw_id: str) -> CapturedZipMemberCoordinate | None: ...

    def raw_revision_file_mtime(self, raw_id: str) -> str | None: ...

    def raw_native_id(self, raw_id: str) -> str | None: ...

    def raw_append_logical_key(self, raw_id: str) -> str | None: ...

    def open_raw_revision_material(
        self,
        raw_id: str,
    ) -> AbstractContextManager[tuple[Provider, BinaryIO, str, RawRevisionKind]]: ...

    def raw_revision_material(self, raw_id: str) -> tuple[Provider, bytes, str, RawRevisionKind]: ...

    def open_raw_container_material(self, raw_id: str) -> AbstractContextManager[BinaryIO | None]: ...

    def retained_sidecar_resolver(self) -> SidecarResolver: ...

    def raw_revision_blob_path(self, raw_id: str) -> Path | None: ...


class RetainedMembershipRead(RetainedRawRead, Protocol):
    """Exact selected membership and original accepted-head inputs."""

    def raw_membership_rebuild_raw_ids(self, logical_source_key: str) -> tuple[str, ...]: ...

    def raw_membership_logical_raw_ids(self, logical_source_key: str) -> tuple[str, ...]: ...

    def raw_revision_head_raw_id(self, logical_source_key: str) -> str | None: ...

    def raw_revision_authority(self, raw_id: str) -> str | None: ...

    def raw_revision_observation_order(self, raw_id: str) -> tuple[int, int]: ...


class RetainedSessionRead(RetainedRawRead, RetainedEnrichmentRead, BlobPublicationSourceRead, Protocol):
    """The parser and enrichment inputs required by a prepared cohort callback."""


class RetainedArtifactPreparer(Protocol):
    """Seal a retained artifact while its parent owns the original read window."""

    def __call__(
        self,
        evidence_reader: RetainedSessionRead,
        raw_id: str,
        *,
        directory: Path,
        validation_mode: ValidationMode,
        schema_registry: SchemaRegistry,
    ) -> PreparedJsonl: ...


def _persist_terminal_raw_refusal(
    producer: SourceRawOutcomeProducer,
    raw_id: str,
    error: Exception,
    *,
    provider: Provider,
    source_path: str,
    source_index: int,
    observed_at_ms: int,
    manage_transaction: bool,
) -> bool:
    """Persist one typed retained refusal through the canonical Source hosts.

    The parent retains the original parser receipt and transaction boundary.
    This body classifies the refusal and writes its artifact/raw state on
    either actual ordinary Source or the same selected preparation tape.
    """
    evidence = (
        RawFailureEvidenceKind.TERMINAL_RETAINED_ZIP_MEMBERSHIP_UNPROVED
        if isinstance(error, RetainedZipMembershipUnprovedError)
        else RawFailureEvidenceKind.TERMINAL_MISSING_PROFILE_IDENTITY
        if isinstance(error, MissingProfileIdentityError)
        else terminal_decode_evidence(error, provider=provider)
    )
    if evidence is None:
        return False
    _record_raw_failure_evidence(
        producer,
        raw_id,
        provider=provider,
        source_path=source_path,
        source_index=source_index,
        acquired_at_ms=observed_at_ms,
        kind=evidence,
        manage_transaction=manage_transaction,
    )
    _apply_source_raw_state_update(
        producer,
        raw_id,
        state=_raw_parse_failure_state(provider, error),
        manage_transaction=manage_transaction,
    )
    return True


class SourceRawOutcomeProducer(SourceArtifactProducer, SourceRawStateProducer, Protocol):
    """The existing artifact and raw-state producers for one classified input."""


if TYPE_CHECKING:
    from polylogue.schemas.retained_validation import RetainedValidationVerdict
    from polylogue.schemas.runtime_registry import SchemaRegistry
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
    from polylogue.storage.sqlite.reference_seal import KnownTierMutationPermit, PreparedIndexMutation
