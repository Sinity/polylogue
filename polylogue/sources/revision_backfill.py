"""Prepare retained revisions and publish captured authority decisions."""

from __future__ import annotations

import contextvars
import dataclasses
import hashlib
import json
import os
import pickle
import sqlite3
import tempfile
import threading
import time
import uuid
from builtins import BaseExceptionGroup
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial, wraps
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Final, Literal, cast

import ijson

from polylogue import logging as _polylogue_logging
from polylogue.archive.artifact_taxonomy import ArtifactStreamClassification, classify_artifact_stream
from polylogue.archive.ingest_flags import (
    COMPACT_BROWSER_CAPTURE_INGEST_FLAG,
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
)
from polylogue.archive.revision_authority import (
    BYTE_AUTHORITY_CENSUS_DETAIL,
    HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    canonical_authority_logical_key,
    durable_authority_logical_keys,
    is_work_event_raw_id,
    parser_census_is_complete,
    raw_receipt_order_sql,
)
from polylogue.archive.revision_replay import RevisionReplayPlan
from polylogue.archive.session_revision_membership import (
    MembershipClassification,
    MembershipRevision,
    classify_membership_revisions,
)
from polylogue.core.binary_signatures import looks_like_sqlite_bytes
from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import PolylogueStrEnum, Provider
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.json import JSONValue
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
from polylogue.core.raw_failure_evidence import (
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
from polylogue.sources.codex_state_evidence import record_codex_state_snapshot_terminal
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import (
    BUNDLE_PROVIDERS,
    detect_provider_from_stream_evidence,
    is_jsonl_source_path,
    is_stream_record_provider,
    parse_payload,
    parse_stream_payload,
    require_positive_conversational_evidence,
)
from polylogue.sources.live.batch_support import (
    jsonl_complete_prefix,
    jsonl_parse_input_of_handle,
    jsonl_parse_prefix_size,
    jsonl_parse_prefix_size_of_handle,
)
from polylogue.sources.parsers import antigravity, codex_state, hermes_state, hermes_verification
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.prepared_jsonl import (
    DecodeFailure,
    PreparedDecodeError,
    PreparedJsonl,
    _iter_prefix_lines,
    classify_decode_failure,
    prepare_jsonl_blob,
    terminal_decode_evidence,
)
from polylogue.sources.prepared_message_sink import SqliteMessageSink
from polylogue.sources.sidecar_evidence import SidecarResolver
from polylogue.sources.sqlite_export import looks_like_logical_source_bytes
from polylogue.sources.sqlite_snapshot import (
    is_declared_logical_export,
    is_sqlite_page_image,
    is_undeclared_logical_export,
)
from polylogue.storage.artifacts.inspection import artifact_observation_id
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.raw_authority import (
    parser_census_logical_keys,
    raw_authority_parser_fingerprint,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.revision_governance import (
    FrozenSourceRemediationRequiredError,
    PreparedRawRevisionClassification,
    _raw_parse_failure_state,
    _raw_parse_success_state,
    apply_prepared_raw_revision_classification,
    membership_key_has_pending_envelope_member,
    pending_raw_envelope_has_membership_authority,
    raw_has_membership_governed_pending_envelope,
    record_current_parser_source_census,
    record_raw_failure_evidence,
)
from polylogue.storage.sqlite.archive_tiers.source_write import (
    PENDING_RAW_LOGICAL_SOURCE_PREFIX,
    ArchiveSourceArtifact,
    apply_source_raw_state_update,
    read_raw_captured_zip_coordinate,
    upsert_raw_artifact,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    PreparedRows,
    PreparedSessionWrite,
    PreparedSessionWriteRefusedError,
    prepare_session_shard,
)
from polylogue.storage.sqlite.archive_tiers.write_shard import ShardRefusedError, discard_session_shard
from polylogue.storage.sqlite.connection_profile import (
    ReadContinuation,
    ReadFrame,
    ReadFrameExpiredError,
    StaleContinuationError,
    read_frame,
)

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
class _CurrentParserReceiptShape:
    recorded_keys_json: object
    typed_key: object
    revision_kind: object
    typed_non_session: bool
    parser_confirmed_non_session: bool
    byte_governed_fragment: bool
    membership_keys: list[object] = field(default_factory=list)


@dataclass(slots=True)
class _RevisionCensusState:
    scanned: int
    classified: int
    quarantined: int
    censused: set[str]
    membership_candidates: dict[str, set[str]]
    provisional_full_raw_ids: dict[str, set[str]]
    transient_non_session_raw_ids: set[str]
    #: Reverse of ``provisional_full_raw_ids``: a chain head or probe is looked
    #: up once per deferred member, so a scan of every key would be quadratic
    #: in the number of independently growing sources.
    provisional_key_by_raw_id: dict[str, str] = field(default_factory=dict)

    def bind_provisional_full_raw(self, logical_key: str, raw_id: str) -> None:
        self.provisional_full_raw_ids.setdefault(logical_key, set()).add(raw_id)
        self.provisional_key_by_raw_id[raw_id] = logical_key

    def provisional_logical_key(self, raw_id: str) -> str | None:
        return self.provisional_key_by_raw_id.get(raw_id)


class RetainedPreparationRetryableError(RuntimeError):
    """A supplied retained parse cannot be trusted; retry without quarantining bytes."""


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
    return _iter_json_stream(record_input, source_name, fail_on_decode_error=True)  # type: ignore[arg-type]


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
    parser_error: str | None = None
    #: Which decode boundary refused the bytes when ``parser_error`` is a
    #: decode refusal; the census turns that into a terminal outcome.
    parser_decode_failure: DecodeFailure | None = None
    missing_profile_identity: bool = False
    retained_zip_membership_unproved: bool = False
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
    for row in source_conn.execute(
        "SELECT source_path, raw_id, hex(blob_hash), blob_size, file_mtime_ms, "
        "revision_kind, acquired_at_ms FROM raw_sessions "
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
            index_conn=index_conn,
            source_conn=source_conn,
            blob_root=blob_root,
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
    if provider not in _SESSION_EVIDENCE_PROVIDERS or not source_path or not native_id:
        return None
    data = _retained_enrichment_sidecar_data(
        provider=provider,
        sessions=(),
        provider_session_ids=[native_id],
        index_conn=index_conn,
        source_conn=source_conn,
        blob_root=blob_root,
        source_path=source_path,
        captured_zip_coordinate=None,
    )
    return enrichment_evidence_key(provider, data, native_id)


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
    raw_id: str,
    provider_token: str,
    blob_hash: str,
    source_path: str,
    kind_token: str,
    native_id: str | None,
    blob_root: str,
    source_db_path: str,
    index_db_path: str,
    directory: str,
    fallback_timestamp: str | None,
) -> PreparedJsonl:
    """Prepare a retained JSON or JSONL session view on a read-only snapshot."""
    from polylogue.sources.live.sidecar_resolution import RetainedSidecarResolver
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.blob_store import BlobStore

    provider = Provider(provider_token)
    if not (is_jsonl_source_path(source_path) or Path(source_path).suffix.lower() == ".json"):
        raise RetainedPreparationRetryableError(f"retained JSON worker cannot parse {raw_id}")
    blob_path = BlobStore(Path(blob_root)).blob_path(blob_hash)
    if provider is Provider.UNKNOWN:
        try:
            with blob_path.open("rb") as payload:
                provider, _evidence = _resolved_retained_provider(payload, source_path)
        except OSError as exc:
            raise RetainedPreparationRetryableError(f"retained JSON evidence read failed for raw {raw_id}") from exc
        if provider is Provider.UNKNOWN:
            refusal = UnsupportedRetainedJsonShapeError(
                f"retained UNKNOWN provider has no recognized complete input shape: {source_path}"
            )
            return PreparedJsonl(
                None,
                None,
                None,
                f"{type(refusal).__name__}: {refusal}",
            )
    kind = RawRevisionKind(kind_token)
    fallback_id = (native_id or Path(source_path).stem) if kind is RawRevisionKind.APPEND else Path(source_path).stem
    try:
        with (
            read_frame(source_db_path, tier=ArchiveTier.SOURCE, timeout_class="background-read") as source_frame,
            read_frame(index_db_path, tier=ArchiveTier.INDEX, timeout_class="background-read") as index_frame,
            ExitStack() as sidecar_lifetimes,
        ):
            source_conn = source_frame.connection
            index_conn = index_frame.connection
            source_conn.execute("BEGIN")
            index_conn.execute("BEGIN")
            from polylogue.storage.sqlite.archive_tiers.source_write import read_raw_profile_identity

            profile_identity = read_raw_profile_identity(source_conn, raw_id)
            captured_zip_coordinate = read_raw_captured_zip_coordinate(source_conn, raw_id)
            if provider is Provider.HERMES and profile_identity is None:
                return PreparedJsonl(
                    blob_hash,
                    None,
                    None,
                    "retained Hermes input has no captured profile identity receipt",
                    missing_profile_identity=True,
                )
            evidence_digest: str | None = None

            def capture_evidence(value: object) -> None:
                nonlocal evidence_digest
                evidence_digest = _enrichment_evidence_digest(value)

            sidecar_data_cache: SidecarData = {}
            from polylogue.sources.assembly import close_sidecar_data

            sidecar_lifetimes.callback(close_sidecar_data, sidecar_data_cache)
            sidecar_data_loaded = False
            from polylogue.sources.assembly import get_assembly_spec

            prepare_per_session = (
                not is_stream_record_provider(source_path, provider)
                and provider in BUNDLE_PROVIDERS
                and Path(source_path).name.lower().endswith(".json")
            )

            # Bundle providers have source-scoped assembly evidence. Codex's
            # title enrichment needs the cohort's session IDs and is not a
            # bundle provider, so it remains on the cohort callback below.
            def finalize(sessions: list[ParsedSession]) -> list[ParsedSession]:
                # ``prepare_jsonl_blob`` admits every session before it calls
                # a finalizer, so this sees only admitted sessions.
                normalized = [
                    normalize_session_timestamps(session, fallback_timestamp=fallback_timestamp) for session in sessions
                ]
                return _replay_safe_enrich_sessions(
                    provider=provider,
                    sessions=normalized,
                    index_conn=index_conn,
                    source_conn=source_conn,
                    blob_root=Path(blob_root),
                    source_path=source_path,
                    captured_zip_coordinate=captured_zip_coordinate,
                    evidence_observer=capture_evidence,
                )

            def prepare_bundle_session(session: ParsedSession) -> ParsedSession:
                nonlocal sidecar_data_loaded
                normalized = normalize_session_timestamps(session, fallback_timestamp=fallback_timestamp)
                spec = get_assembly_spec(provider)
                if spec is None:
                    return cast(ParsedSession, normalized)
                if not sidecar_data_loaded:
                    sidecar_data_cache.update(
                        _retained_enrichment_sidecar_data(
                            provider=provider,
                            sessions=(),
                            index_conn=index_conn,
                            source_conn=source_conn,
                            blob_root=Path(blob_root),
                            source_path=source_path,
                            captured_zip_coordinate=captured_zip_coordinate,
                        )
                    )
                    capture_evidence(sidecar_data_cache)
                    sidecar_data_loaded = True
                assert sidecar_data_loaded
                return spec.enrich_session(normalized, sidecar_data_cache)

            parse_prefix_size: int | None = None
            if is_jsonl_source_path(source_path):
                with blob_path.open("rb") as tail_handle:
                    parse_prefix_size = jsonl_parse_prefix_size_of_handle(tail_handle)
            artifact = prepare_jsonl_blob(
                str(blob_path),
                source_path,
                provider.value,
                fallback_id,
                is_stream=is_stream_record_provider(source_path, provider),
                profile_identity=profile_identity,
                shard_directory=directory,
                publication_publisher=ArchiveBlobPublisher(Path(source_db_path), Path(blob_root)),
                # Live intake refuses a complete JSONL record that does not
                # decode; replay of the same bytes must refuse it too.
                strict_jsonl_records=True,
                parse_prefix_size=parse_prefix_size,
                sidecar_resolver=RetainedSidecarResolver(
                    Path(blob_root).parent,
                    blob_root=Path(blob_root),
                    source_conn=source_conn,
                ),
                prepare_session=prepare_bundle_session if prepare_per_session else None,
                prepare_sessions=None if prepare_per_session else finalize,
                # The publisher recomputes this digest from the retained
                # evidence for every artifact, so a pass that enriched nothing
                # (no assembly spec, or no admitted session) must bind the
                # same evidence value rather than an absent one.
                preparation_dependency=lambda: (
                    _retained_dependency_digest(
                        evidence_digest
                        if evidence_digest is not None
                        else _owned_enrichment_evidence_digest(
                            _retained_enrichment_sidecar_data(
                                provider=provider,
                                sessions=(),
                                index_conn=index_conn,
                                source_conn=source_conn,
                                blob_root=Path(blob_root),
                                source_path=source_path,
                                captured_zip_coordinate=captured_zip_coordinate,
                            )
                        ),
                        _retained_parser_sidecar_digest(source_conn, provider=provider, source_path=source_path),
                    ),
                    str(Path(index_db_path).resolve()),
                ),
            )
    except (OSError, sqlite3.OperationalError) as exc:
        raise RetainedPreparationRetryableError(f"retained JSON evidence read failed for raw {raw_id}") from exc
    if artifact.blob_hash is not None and artifact.blob_hash != blob_hash:
        artifact.discard()
        raise RetainedPreparationRetryableError(f"retained JSON blob changed for raw {raw_id}")
    return artifact


def prepare_retained_non_json_artifact(
    archive: ArchiveStore,
    raw_id: str,
    provider_token: str,
    blob_hash: str,
    source_path: str,
    kind_token: str,
    native_id: str | None,
    blob_root: str,
    source_db_path: str,
    index_db_path: str,
    directory: str,
    fallback_timestamp: str | None,
) -> PreparedJsonl:
    """Seal a non-JSON retained parse inside the isolated preparation worker."""
    from polylogue.pipeline.ids import session_content_hash
    from polylogue.sources.prepared_jsonl import (
        PreparedJsonl,
        _prepare_attachment_publications,
        _prepare_codex_state_blob,
        _prepare_sidecar_publications,
        _write_artifact,
    )
    from polylogue.sources.prepared_message_sink import SqliteMessageStore
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.blob_store import BlobStore

    publisher = ArchiveBlobPublisher(Path(source_db_path), Path(blob_root))
    archive_root = Path(source_db_path).parent
    if archive_root / "blob" != Path(blob_root):
        raise RetainedPreparationRetryableError(f"retained archive binding changed for raw {raw_id}")
    sessions_path = Path(directory) / f"prepared-{uuid.uuid4().hex}.db"
    shard_path: Path | None = None
    store: SqliteMessageStore | None = None
    sealed = False
    parsed = False
    try:
        provider, current_hash, current_path, kind, _size = archive.raw_revision_descriptor(raw_id)
        if (
            archive.index_db_path.resolve() != Path(index_db_path).resolve()
            or provider != Provider(provider_token)
            or current_hash != blob_hash
            or current_path != source_path
            or kind != RawRevisionKind(kind_token)
            or (archive.raw_native_id(raw_id) if kind is RawRevisionKind.APPEND else None) != native_id
            or archive.raw_revision_file_mtime(raw_id) != fallback_timestamp
        ):
            raise RetainedPreparationRetryableError(f"retained descriptor changed for raw {raw_id}")
        state_descriptor = _retained_codex_state_descriptor(archive, raw_id)
        sessions = parse_retained_raw_sessions(archive, raw_id)
        outcome = _enrich_retained_parse_outcome(
            archive,
            raw_id,
            descriptor=(provider, blob_hash, source_path, kind, _size, native_id, archive.raw_profile_identity(raw_id)),
            outcome=(sessions, _size, kind),
        )
        if isinstance(outcome, Exception):
            raise outcome
        sessions = outcome[0]
        resolved_provider = Provider.from_string(sessions[0].source_name) if sessions else provider
        if not sessions and resolved_provider is Provider.UNKNOWN:
            with archive.open_raw_revision_material(raw_id) as (_provider, payload, _path, _kind):
                resolved_provider, _evidence = _resolved_retained_provider(payload, source_path)
        evidence = _retained_enrichment_sidecar_data(
            provider=resolved_provider,
            sessions=sessions,
            index_conn=archive.index_connection,
            source_conn=archive._ensure_source_conn(),
            blob_root=Path(blob_root),
            source_path=source_path,
            captured_zip_coordinate=archive.raw_captured_zip_coordinate(raw_id),
        )
        dependency = _retained_dependency_digest(
            _owned_enrichment_evidence_digest(evidence),
            _retained_parser_sidecar_digest(
                archive._ensure_source_conn(), provider=resolved_provider, source_path=source_path
            ),
        )
        parsed = True
        if not BlobStore(Path(blob_root)).verify(blob_hash):
            raise RetainedPreparationRetryableError(f"retained blob changed for raw {raw_id}")
        if state_descriptor is not None:
            artifact = _prepare_codex_state_blob(
                state_descriptor[0],
                Path(directory),
                state_kind=state_descriptor[2],
                source_hash=blob_hash,
                semantic_source_path=source_path,
                enrichment_digest=dependency,
                enrichment_index_path=str(Path(index_db_path).resolve()),
                publication_publisher=publisher,
            )
            sealed = True
            return artifact
        store = SqliteMessageStore(sessions_path)
        for session in sessions:
            session.content_hash = session_content_hash(session)
        shard_path = prepare_session_shard(Path(directory), sessions).path
        _write_artifact(
            store,
            blob_hash,
            sessions,
            enrichment_digest=dependency,
            enrichment_index_path=str(Path(index_db_path).resolve()),
        )
        _prepare_attachment_publications(store, publisher, Path(directory))
        _prepare_sidecar_publications(store, publisher, Path(directory))
        store.close()
        store = None
        artifact = PreparedJsonl.seal(
            blob_hash,
            sessions_path,
            shard_path,
            enrichment_digest=dependency,
            enrichment_index_path=str(Path(index_db_path).resolve()),
            resolved_provider=resolved_provider,
            publication_publisher=publisher,
            captured_profile_key=archive.raw_profile_identity(raw_id),
        )
        sealed = True
        return artifact
    except RetainedPreparationRetryableError:
        raise
    except (OSError, sqlite3.OperationalError, MemoryError) as exc:
        raise RetainedPreparationRetryableError(f"retained worker could not prepare raw {raw_id}") from exc
    except DaemonOperationCancelled:
        raise
    except Exception as exc:
        if parsed:
            raise RetainedPreparationRetryableError(f"retained worker artifact failed for raw {raw_id}") from exc
        return PreparedJsonl(
            blob_hash,
            None,
            None,
            f"{type(exc).__name__}: {exc}"[:500],
            missing_profile_identity=isinstance(exc, MissingProfileIdentityError),
            retained_zip_membership_unproved=isinstance(exc, RetainedZipMembershipUnprovedError),
        )
    finally:
        if store is not None:
            store.close()
        if not sealed:
            sessions_path.unlink(missing_ok=True)
            if shard_path is not None:
                discard_session_shard(shard_path)


def _prepared_retained_outcome(
    archive: ArchiveStore,
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
    if prepared.parser_error is not None:
        if prepared.retained_zip_membership_unproved:
            return RetainedZipMembershipUnprovedError(prepared.parser_error)
        if prepared.missing_profile_identity:
            return MissingProfileIdentityError(prepared.parser_error)
        return retained_parse_exception(prepared.parser_error, prepared.parser_decode_failure)
    if prepared.prepared_artifact is not None:
        artifact = prepared.prepared_artifact
        if (
            artifact.blob_hash != blob_hash
            or artifact.error is not None
            or artifact.captured_profile_key != prepared.captured_profile_key
        ):
            raise RetainedPreparationRetryableError(f"prepared retained artifact changed for raw {raw_id}")
        if artifact.enrichment_index_path != str(archive.index_db_path.resolve()):
            raise RetainedPreparationRetryableError(f"prepared retained index dependency changed for raw {raw_id}")
        try:
            sessions = artifact.session_sequence()
        except DaemonOperationCancelled:
            raise
        except Exception as exc:
            raise RetainedPreparationRetryableError(f"prepared retained artifact unavailable for raw {raw_id}") from exc
        # The worker enriched and sealed the dependency under the provider it
        # resolved from these verified bytes; an UNKNOWN descriptor never
        # names the evidence that was read.
        stale = prepared_enrichment_dependency_state(
            archive,
            artifact,
            provider=artifact.resolved_provider or provider,
            source_path=source_path,
            captured_zip_coordinate=archive.raw_captured_zip_coordinate(raw_id),
            sessions=sessions,
            parser_sidecars=True,
        )
        if stale is not None:
            raise RetainedPreparationRetryableError(f"prepared retained {stale} for raw {raw_id}")
        return sessions, size, kind
    raise RetainedPreparationRetryableError(f"prepared retained carrier is missing for raw {raw_id}")


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
            current_receipt_shapes: dict[str, _CurrentParserReceiptShape] = {}
            for (
                raw_id_value,
                logical_keys_json,
                typed_key,
                revision_kind,
                typed_non_session,
                parser_confirmed_non_session,
                byte_governed_fragment,
                membership_key,
            ) in conn.execute(
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
            ):
                raw_id = str(raw_id_value)
                shape = current_receipt_shapes.setdefault(
                    raw_id,
                    _CurrentParserReceiptShape(
                        logical_keys_json,
                        typed_key,
                        revision_kind,
                        bool(typed_non_session),
                        bool(parser_confirmed_non_session),
                        bool(byte_governed_fragment),
                    ),
                )
                if membership_key is not None:
                    shape.membership_keys.append(membership_key)
            for raw_id, shape in current_receipt_shapes.items():
                durable_keys = durable_authority_logical_keys(
                    raw_logical_key=shape.typed_key,
                    revision_kind=shape.revision_kind,
                    membership_logical_keys=shape.membership_keys,
                )
                if not parser_census_is_complete(
                    recorded_keys=parser_census_logical_keys(shape.recorded_keys_json),
                    durable_keys=durable_keys,
                    typed_non_session=shape.typed_non_session,
                    parser_confirmed_non_session=shape.parser_confirmed_non_session,
                    byte_governed_fragment=shape.byte_governed_fragment,
                ):
                    uncensused.append(raw_id)
    return tuple(sorted(set(uncensused)))


def _census_historical_revision_evidence(
    archive: ArchiveStore,
    *,
    selected_raw_ids: list[str] | None,
    prepared_inputs: Mapping[str, PreparedRetainedInput] | None = None,
) -> _RevisionCensusState:
    """Apply captured parser outcomes to source authority in fixed raw order."""
    state = _RevisionCensusState(0, 0, 0, set(), {}, {}, set())

    def commit_unit() -> None:
        archive.commit()

    def apply_outcome(
        raw_id: str,
        source_index: int,
        outcomes: Mapping[str, tuple[Sequence[ParsedSession], int, RawRevisionKind] | Exception],
    ) -> None:
        state.scanned += 1
        state.censused.add(raw_id)
        if source_index < 0:
            source_conn = archive._ensure_source_conn()
            has_membership_authority = source_conn.execute(
                "SELECT 1 FROM raw_session_memberships WHERE raw_id = ? LIMIT 1", (raw_id,)
            ).fetchone()
            if has_membership_authority is not None:
                record_current_parser_source_census(source_conn, raw_id)
            else:
                archive.replace_raw_membership_census(
                    raw_id,
                    None,
                    parser_fingerprint=raw_authority_parser_fingerprint(),
                    censused_at_ms=0,
                    detail=BYTE_AUTHORITY_CENSUS_DETAIL,
                    manage_transaction=True,
                    revision_authority=RawRevisionAuthority.BYTE_PROVEN,
                )
            state.quarantined += 1
            commit_unit()
            return
        outcome = outcomes[raw_id]
        if isinstance(outcome, Exception) and _settle_terminal_raw_refusal(
            archive, raw_id, outcome, source_index=source_index, manage_transaction=True
        ):
            state.quarantined += 1
            commit_unit()
            return
        if isinstance(outcome, Exception):
            archive.replace_raw_membership_census(
                raw_id,
                None,
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=0,
                detail=str(outcome),
                manage_transaction=True,
                revision_authority=None,
            )
            state.quarantined += 1
            commit_unit()
            return
        sessions, payload_bytes, revision_kind = outcome
        stored_provider, _blob_hash, _source_path, _stored_kind, _stored_size = archive.raw_revision_descriptor(raw_id)
        if sessions:
            parsed_provider = Provider.from_string(sessions[0].source_name)
            record_session_artifact_observation(
                archive,
                raw_id=raw_id,
                provider=parsed_provider,
                source_path=_source_path,
                source_index=source_index,
                observed_at_ms=archive.raw_revision_observed_at_ms(raw_id),
                manage_transaction=True,
            )
        if stored_provider is Provider.UNKNOWN and sessions:
            apply_source_raw_state_update(
                archive._ensure_source_conn(),
                raw_id,
                state=RawSessionStateUpdate(payload_provider=Provider.from_string(sessions[0].source_name)),
                manage_transaction=True,
            )
        if not sessions:
            prepared = prepared_inputs.get(raw_id) if prepared_inputs is not None else None
            artifact = prepared.prepared_artifact if prepared is not None else None
            if artifact is None or artifact.resolved_provider is None:
                raise RetainedPreparationRetryableError(f"empty retained outcome has no captured provider: {raw_id}")
            provider = artifact.resolved_provider
            with archive._ensure_source_conn():
                if stored_provider is Provider.UNKNOWN and provider is not Provider.UNKNOWN:
                    apply_source_raw_state_update(
                        archive._ensure_source_conn(),
                        raw_id,
                        state=RawSessionStateUpdate(payload_provider=provider),
                        manage_transaction=False,
                    )
                terminalized = _persist_terminal_non_session_artifact(
                    archive,
                    raw_id,
                    provider=provider,
                    source_path=_source_path,
                    source_index=source_index,
                    stream_classification=artifact.stream_classification(),
                    manage_transaction=False,
                )
                if provider is not Provider.UNKNOWN:
                    archive.replace_raw_membership_census(
                        raw_id,
                        [],
                        parser_fingerprint=raw_authority_parser_fingerprint(),
                        censused_at_ms=0,
                        detail=LEGACY_PAGE_IMAGE_CENSUS_DETAIL if _retained_page_image_raw(archive, raw_id) else "",
                        retire_full_revision_governance=revision_kind is RawRevisionKind.FULL,
                        manage_transaction=False,
                        revision_authority=None,
                    )
                    if not terminalized:
                        apply_source_raw_state_update(
                            archive._ensure_source_conn(),
                            raw_id,
                            state=_raw_parse_success_state(provider),
                            manage_transaction=False,
                        )
            if provider is not Provider.UNKNOWN:
                commit_unit()
                return
        state.classified += int(len(sessions) == 1)
        if len(sessions) == 1 and revision_kind is RawRevisionKind.UNKNOWN:
            session = sessions[0]
            logical_key = f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}"
            archive.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    logical_source_key=logical_key,
                    kind=RawRevisionKind.FULL,
                    source_revision=raw_id,
                    acquisition_generation=0,
                    authority=RawRevisionAuthority.QUARANTINED,
                ),
                manage_transaction=True,
            )
            record_current_parser_source_census(archive._ensure_source_conn(), raw_id, parser_sessions=sessions)
            state.bind_provisional_full_raw(logical_key, raw_id)
            commit_unit()
        elif revision_kind is RawRevisionKind.UNKNOWN or (
            _raw_has_pending_envelope(archive, raw_id)
            and (
                len(sessions) > 1 or raw_has_membership_governed_pending_envelope(archive._ensure_source_conn(), raw_id)
            )
        ):
            archive.replace_raw_membership_census(
                raw_id,
                sessions,
                parser_fingerprint=raw_authority_parser_fingerprint(),
                censused_at_ms=0,
                manage_transaction=True,
                revision_authority=None,
            )
            for session in sessions:
                logical_key = f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}"
                state.membership_candidates.setdefault(logical_key, set()).add(raw_id)
            commit_unit()
        else:
            record_current_parser_source_census(archive._ensure_source_conn(), raw_id, parser_sessions=sessions)
            commit_unit()

    census_selections: tuple[tuple[str, ...] | None, ...]
    if selected_raw_ids is None:
        census_selections = (None,)
    else:
        census_selections = archive.raw_membership_selection_components(selected_raw_ids)
    try:
        for initial_selection in census_selections:
            census_selection = initial_selection
            while True:
                rows = archive.raw_membership_census_rows(census_selection)
                for raw_id, _source_index, _terminal_non_session, _raw_rowid in sorted(
                    rows, key=lambda row: archive.raw_revision_observation_order(row[0])[1]
                ):
                    if raw_id in state.censused or not _replay_retained_codex_state_evidence(
                        archive, raw_id, prepared_inputs
                    ):
                        continue
                    state.scanned += 1
                    state.censused.add(raw_id)
                    state.transient_non_session_raw_ids.add(raw_id)
                    commit_unit()
                terminal_raw_ids = {
                    raw_id for raw_id, _source_index, terminal_non_session, _raw_rowid in rows if terminal_non_session
                }
                for raw_id in terminal_raw_ids - state.censused:
                    state.scanned += 1
                    state.censused.add(raw_id)
                    record_current_parser_source_census(archive._ensure_source_conn(), raw_id)
                    commit_unit()
                pending_rows = [
                    (raw_id, source_index)
                    for raw_id, source_index, terminal_non_session, _raw_rowid in rows
                    if raw_id not in state.censused and (not terminal_non_session)
                ]
                parseable_raw_ids = [raw_id for raw_id, source_index in pending_rows if source_index >= 0]
                if parseable_raw_ids and prepared_inputs is None:
                    raise RetainedPreparationRetryableError("source census requires retained worker preparation")
                for raw_id, source_index in pending_rows:
                    check_compute_cancelled()
                    outcomes = (
                        {raw_id: _prepared_retained_outcome(archive, raw_id, prepared_inputs)}
                        if source_index >= 0 and prepared_inputs is not None
                        else {}
                    )
                    apply_outcome(raw_id, source_index, outcomes)
                if census_selection is None:
                    break
                expanded, _keys = archive.expand_raw_membership_selection(list(census_selection))
                if set(expanded) == set(census_selection):
                    break
                census_selection = expanded
    except BaseException:
        raise
    return state


def require_current_parser_source_census(
    archive_root: Path,
    *,
    selected_raw_ids: Sequence[str] | None = None,
    transient_non_session_raw_ids: Set[str] = frozenset(),
) -> dict[str, tuple[str, ...]]:
    """Require phase-2 parser receipts before allocating an index candidate."""
    stale_raw_ids: list[str] = []
    recorded_logical_keys: dict[str, tuple[str, ...]] = {}
    selections: tuple[tuple[str, ...] | None, ...] | None
    if selected_raw_ids is None:
        selections = None
    else:
        selections = tuple(
            tuple(selected_raw_ids[offset : offset + 500]) for offset in range(0, len(selected_raw_ids), 500)
        )
    with read_frame(
        archive_root / "source.db", tier=ArchiveTier.SOURCE, timeout_class="background-read"
    ) as source_frame:
        source_frontier_rowid: int | None = None
        if selections is None:
            source_frontier_rowid = _current_source_rowid_frontier(source_frame)
        for selection in _current_source_raw_id_selections(
            source_frame, selections, frontier_rowid=source_frontier_rowid
        ):
            where = f"WHERE r.raw_id IN ({','.join('?' for _ in selection)})"
            rows = _read_current_source_census_page(
                source_frame,
                selection,
                source_frontier_rowid,
                f"""
                SELECT r.raw_id, c.parser_fingerprint, c.status, c.logical_keys_json
                FROM raw_sessions AS r
                LEFT JOIN raw_authority_parser_census AS c ON c.raw_id = r.raw_id
                {where}
                ORDER BY r.raw_id
                """,
                selection,
            )
            for raw_id_value, fingerprint, status, logical_keys_json in rows:
                raw_id = str(raw_id_value)
                if raw_id in transient_non_session_raw_ids:
                    recorded_logical_keys[raw_id] = ()
                    continue
                if fingerprint != raw_authority_parser_fingerprint() or status != "complete":
                    stale_raw_ids.append(raw_id)
                    continue
                normalized_keys = parser_census_logical_keys(logical_keys_json)
                if normalized_keys is None:
                    stale_raw_ids.append(raw_id)
                    continue
                recorded_logical_keys[raw_id] = normalized_keys
        _require_current_source_census_frame(source_frame)
    if stale_raw_ids:
        sample = ", ".join(stale_raw_ids[:5])
        raise FrozenSourceRemediationRequiredError(
            "inactive candidate requires a complete current-parser source census; "
            f"{len(stale_raw_ids)} raw(s) are stale or incomplete (sample: {sample})"
        )

    durable_bindings: dict[str, tuple[object, object, list[object], bool, bool, bool]] = {
        raw_id: (None, RawRevisionKind.UNKNOWN.value, [], False, False, False) for raw_id in recorded_logical_keys
    }
    with read_frame(
        archive_root / "source.db", tier=ArchiveTier.SOURCE, timeout_class="background-read"
    ) as source_frame:
        for selection in _current_source_raw_id_selections(
            source_frame, selections, frontier_rowid=source_frontier_rowid
        ):
            where = f"WHERE r.raw_id IN ({','.join('?' for _ in selection)})"
            params = (
                raw_authority_parser_fingerprint(),
                raw_authority_parser_fingerprint(),
                RawRevisionAuthority.BYTE_PROVEN.value,
                *selection,
            )
            rows = _read_current_source_census_page(
                source_frame,
                selection,
                source_frontier_rowid,
                f"""
                SELECT r.raw_id, r.logical_source_key, r.revision_kind, r.source_index, m.logical_source_key,
                       EXISTS(SELECT 1 FROM raw_artifacts AS a WHERE a.raw_id = r.raw_id AND a.parse_as_session = 0),
                       EXISTS(
                           SELECT 1 FROM raw_membership_census AS mc
                           WHERE mc.raw_id = r.raw_id
                             AND mc.parser_fingerprint = ?
                             AND mc.status = 'non_session'
                       ),
                       EXISTS(
                           SELECT 1 FROM raw_membership_census AS mc
                           WHERE mc.raw_id = r.raw_id
                             AND r.source_index < 0
                             AND mc.parser_fingerprint = ?
                             AND mc.status = 'failed'
                             AND mc.revision_authority = ?
                       )
                FROM raw_sessions AS r
                LEFT JOIN raw_session_memberships AS m ON m.raw_id = r.raw_id
                {where}
                ORDER BY r.raw_id, m.logical_source_key
                """,
                params,
            )
            for (
                raw_id_value,
                typed_key,
                revision_kind,
                _source_index,
                membership_key,
                typed_non_session,
                parser_confirmed_non_session,
                byte_governed_fragment,
            ) in rows:
                raw_id = str(raw_id_value)
                typed_non_session = bool(typed_non_session) or raw_id in transient_non_session_raw_ids
                (
                    existing_typed,
                    existing_kind,
                    memberships,
                    existing_non_session,
                    existing_parser_confirmed_non_session,
                    existing_byte_governed_fragment,
                ) = durable_bindings.get(
                    raw_id,
                    (
                        typed_key,
                        revision_kind,
                        [],
                        bool(typed_non_session),
                        bool(parser_confirmed_non_session),
                        bool(byte_governed_fragment),
                    ),
                )
                if membership_key is not None:
                    memberships.append(membership_key)
                durable_bindings[raw_id] = (
                    typed_key if existing_typed is None else existing_typed,
                    revision_kind if existing_kind == RawRevisionKind.UNKNOWN.value else existing_kind,
                    memberships,
                    bool(typed_non_session) or existing_non_session,
                    bool(parser_confirmed_non_session) or existing_parser_confirmed_non_session,
                    bool(byte_governed_fragment) or existing_byte_governed_fragment,
                )

    invalid_durable_bindings: set[str] = set()
    durable_logical_keys: dict[str, tuple[str, ...]] = {}
    for (
        raw_id,
        (
            typed_key,
            revision_kind,
            membership_keys,
            typed_non_session,
            parser_confirmed_non_session,
            byte_governed_fragment,
        ),
    ) in durable_bindings.items():
        durable_keys = durable_authority_logical_keys(
            raw_logical_key=typed_key,
            revision_kind=revision_kind,
            membership_logical_keys=membership_keys,
        )
        if durable_keys is None or not parser_census_is_complete(
            recorded_keys=recorded_logical_keys.get(raw_id),
            durable_keys=durable_keys,
            typed_non_session=typed_non_session,
            parser_confirmed_non_session=parser_confirmed_non_session,
            byte_governed_fragment=byte_governed_fragment,
        ):
            invalid_durable_bindings.add(raw_id)
        else:
            durable_logical_keys[raw_id] = durable_keys

    authority_binding_drift = sorted(
        invalid_durable_bindings
        | {
            raw_id
            for raw_id, census_keys in recorded_logical_keys.items()
            if durable_logical_keys.get(raw_id, ()) != census_keys
        }
    )
    if authority_binding_drift:
        sample = ", ".join(authority_binding_drift[:5])
        raise FrozenSourceRemediationRequiredError(
            "inactive candidate current-parser census differs from frozen durable authority bindings; "
            f"{len(authority_binding_drift)} raw(s) require source remediation (sample: {sample})"
        )

    authority_rows: dict[str, tuple[str | None, str, str, int, str | None, str | None]] = {}
    with read_frame(
        archive_root / "source.db", tier=ArchiveTier.SOURCE, timeout_class="background-read"
    ) as source_frame:
        for selection in _current_source_raw_id_selections(
            source_frame, selections, frontier_rowid=source_frontier_rowid
        ):
            where = f"WHERE raw_id IN ({','.join('?' for _ in selection)})"
            rows = _read_current_source_census_page(
                source_frame,
                selection,
                source_frontier_rowid,
                f"""
                SELECT raw_id, logical_source_key, revision_kind, revision_authority,
                       source_index, predecessor_raw_id, baseline_raw_id
                FROM raw_sessions {where}
                ORDER BY raw_id
                """,
                selection,
            )
            for raw_id_value, logical_key, revision_kind, authority, source_index, predecessor_id, baseline_id in rows:
                authority_rows[str(raw_id_value)] = (
                    str(logical_key) if logical_key is not None else None,
                    str(revision_kind),
                    str(authority),
                    int(source_index),
                    str(predecessor_id) if predecessor_id is not None else None,
                    str(baseline_id) if baseline_id is not None else None,
                )

    append_identity_drift: set[str] = set()
    for raw_id, (
        append_key,
        revision_kind,
        authority,
        _source_index,
        predecessor_id,
        baseline_id,
    ) in authority_rows.items():
        if revision_kind != RawRevisionKind.APPEND.value:
            continue
        predecessor = authority_rows.get(predecessor_id or "")
        baseline = authority_rows.get(baseline_id or "")
        if (
            append_key is None
            or authority != RawRevisionAuthority.BYTE_PROVEN.value
            or predecessor_id is None
            or baseline_id is None
            or predecessor_id == raw_id
            or baseline_id == raw_id
            or predecessor is None
            or baseline is None
            or predecessor[2] != RawRevisionAuthority.BYTE_PROVEN.value
            or baseline[1] != RawRevisionKind.FULL.value
            or baseline[2] != RawRevisionAuthority.BYTE_PROVEN.value
            or baseline[3] < 0
        ):
            append_identity_drift.add(raw_id)
            continue
        try:
            canonical_append_key = canonical_authority_logical_key(append_key)
            canonical_predecessor_key = canonical_authority_logical_key(predecessor[0] or "")
            canonical_baseline_key = canonical_authority_logical_key(baseline[0] or "")
        except ValueError:
            append_identity_drift.add(raw_id)
            continue
        if {canonical_predecessor_key, canonical_baseline_key} != {canonical_append_key}:
            append_identity_drift.add(raw_id)
            continue

        seen = {raw_id}
        cursor_id = predecessor_id
        while True:
            if cursor_id in seen:
                append_identity_drift.add(raw_id)
                break
            seen.add(cursor_id)
            cursor = authority_rows.get(cursor_id)
            if cursor is None:
                append_identity_drift.add(raw_id)
                break
            if cursor[1] == RawRevisionKind.FULL.value:
                if cursor_id != baseline_id:
                    append_identity_drift.add(raw_id)
                break
            if cursor[1] != RawRevisionKind.APPEND.value or cursor[4] is None:
                append_identity_drift.add(raw_id)
                break
            try:
                if canonical_authority_logical_key(cursor[0] or "") != canonical_append_key:
                    append_identity_drift.add(raw_id)
                    break
            except ValueError:
                append_identity_drift.add(raw_id)
                break
            cursor_id = cursor[4]
    if append_identity_drift:
        sample = ", ".join(sorted(append_identity_drift)[:5])
        raise FrozenSourceRemediationRequiredError(
            "inactive candidate typed continuation identity differs from linked byte authority; "
            f"{len(append_identity_drift)} raw(s) require source remediation (sample: {sample})"
        )

    unresolved_raw_ids: list[str] = []
    with read_frame(
        archive_root / "source.db", tier=ArchiveTier.SOURCE, timeout_class="background-read"
    ) as source_frame:
        for selection in _current_source_raw_id_selections(
            source_frame, selections, frontier_rowid=source_frontier_rowid
        ):
            authority_where = f"AND r.raw_id IN ({','.join('?' for _ in selection)})"
            authority_params: tuple[object, ...] = selection
            unresolved_raw_ids.extend(
                str(row[0])
                for row in _read_current_source_census_page(
                    source_frame,
                    selection,
                    source_frontier_rowid,
                    f"""
                    SELECT DISTINCT r.raw_id
                    FROM raw_sessions AS r
                    LEFT JOIN raw_membership_census AS c ON c.raw_id = r.raw_id
                    LEFT JOIN raw_session_memberships AS m ON m.raw_id = r.raw_id
                    WHERE r.revision_authority = 'quarantined'
                      {authority_where}
                      AND NOT EXISTS (
                          SELECT 1 FROM raw_artifacts AS a
                          WHERE a.raw_id = r.raw_id AND a.parse_as_session = 0
                      )
                      AND (
                          c.raw_id IS NULL OR c.status NOT IN ('complete', 'non_session')
                          OR (
                              c.status = 'complete'
                              AND (
                                  m.raw_id IS NULL OR m.decision IS NULL
                                  OR m.decision IN ('ambiguous', 'deferred')
                              )
                          )
                      )
                    ORDER BY r.raw_id
                    """,
                    authority_params,
                )
                if str(row[0]) not in transient_non_session_raw_ids
            )
    if unresolved_raw_ids:
        sample = ", ".join(unresolved_raw_ids[:5])
        raise FrozenSourceRemediationRequiredError(
            "inactive candidate requires complete frozen source authority; "
            f"{len(unresolved_raw_ids)} raw(s) remain quarantined or undecided (sample: {sample})"
        )
    return recorded_logical_keys


_CURRENT_SOURCE_CENSUS_PAGE_SIZE = 500
_CURRENT_SOURCE_CENSUS_PAGE_RETRIES = 1


def _current_source_rowid_frontier(source_frame: ReadFrame) -> int:
    """Capture the committed source row frontier, retrying if its frame expires."""
    sql = "SELECT COALESCE(MAX(rowid), 0) FROM raw_sessions"
    for attempt in range(_CURRENT_SOURCE_CENSUS_PAGE_RETRIES + 1):
        try:
            rows = tuple(source_frame.stream(sql))
            return int(rows[0][0]) if rows else 0
        except ReadFrameExpiredError:
            if attempt >= _CURRENT_SOURCE_CENSUS_PAGE_RETRIES:
                raise
            _require_current_source_census_frame(source_frame)
            source_frame.rebind()
    raise AssertionError("bounded source frontier retry loop fell through")


def _current_source_selection_continuation(
    source_frame: ReadFrame,
    selection: tuple[str, ...],
    frontier_rowid: int | None,
) -> ReadContinuation:
    """Anchor one page by its source endpoint or its immutable input selection."""
    last_raw_id = selection[-1]
    if frontier_rowid is None:
        continuation = ReadContinuation(
            position=last_raw_id,
            anchor_sql="SELECT ?",
            anchor_params=(last_raw_id,),
        )
    else:
        continuation = ReadContinuation(
            position=last_raw_id,
            anchor_sql="SELECT raw_id FROM raw_sessions WHERE raw_id = ? AND rowid <= ?",
            anchor_params=(last_raw_id, frontier_rowid),
        )
    return source_frame.bind(continuation)


def _require_current_source_census_frame(source_frame: ReadFrame) -> None:
    """Refuse a mixed census if the source changed since this frame opened."""
    if not source_frame.revalidate():
        raise StaleContinuationError(
            "source archive changed during parser source census; retry against one unchanged generation"
        )


def _resume_current_source_census(
    source_frame: ReadFrame,
    continuation: ReadContinuation,
) -> ReadContinuation:
    if source_frame.expired:
        _require_current_source_census_frame(source_frame)
    return source_frame.resume(continuation)


def _read_current_source_census_page(
    source_frame: ReadFrame,
    selection: tuple[str, ...],
    frontier_rowid: int | None,
    sql: str,
    parameters: Sequence[object],
) -> tuple[sqlite3.Row, ...]:
    """Read a whole selection before changing result state, retrying one expiry."""
    continuation = _current_source_selection_continuation(source_frame, selection, frontier_rowid)
    for attempt in range(_CURRENT_SOURCE_CENSUS_PAGE_RETRIES + 1):
        try:
            return tuple(source_frame.stream(sql, parameters))
        except ReadFrameExpiredError:
            if attempt >= _CURRENT_SOURCE_CENSUS_PAGE_RETRIES:
                raise
            _require_current_source_census_frame(source_frame)
            _resume_current_source_census(source_frame, continuation)
    raise AssertionError("bounded source census page retry loop fell through")


def _current_source_raw_id_selections(
    source_frame: ReadFrame,
    selections: tuple[tuple[str, ...] | None, ...] | None,
    *,
    frontier_rowid: int | None,
) -> Iterator[tuple[str, ...]]:
    """Yield bounded raw-id selections, rebinding between pages as needed."""
    if selections is not None:
        continuation: ReadContinuation | None = None
        for selection in selections:
            if not selection:
                continue
            if continuation is not None:
                _resume_current_source_census(source_frame, continuation)
            continuation = _current_source_selection_continuation(source_frame, selection, frontier_rowid)
            _resume_current_source_census(source_frame, continuation)
            yield selection
        return

    if frontier_rowid is None:
        raise ValueError("archive-wide source census requires its initial rowid frontier")
    continuation = None
    after_raw_id: str | None = None
    while True:
        if continuation is not None:
            _resume_current_source_census(source_frame, continuation)
            after_raw_id = str(continuation.position)
        params: tuple[object, ...] = (
            (frontier_rowid, _CURRENT_SOURCE_CENSUS_PAGE_SIZE)
            if after_raw_id is None
            else (frontier_rowid, after_raw_id, _CURRENT_SOURCE_CENSUS_PAGE_SIZE)
        )
        where = "WHERE rowid <= ?" if after_raw_id is None else "WHERE rowid <= ? AND raw_id > ?"
        page_sql = f"SELECT raw_id FROM raw_sessions {where} ORDER BY raw_id LIMIT ?"
        page_anchor = continuation or source_frame.bind(
            ReadContinuation(position=frontier_rowid, anchor_sql="SELECT ?", anchor_params=(frontier_rowid,))
        )
        for attempt in range(_CURRENT_SOURCE_CENSUS_PAGE_RETRIES + 1):
            try:
                raw_ids = tuple(str(row[0]) for row in source_frame.stream(page_sql, params))
                break
            except ReadFrameExpiredError:
                if attempt >= _CURRENT_SOURCE_CENSUS_PAGE_RETRIES:
                    raise
                _require_current_source_census_frame(source_frame)
                _resume_current_source_census(source_frame, page_anchor)
        else:
            raise AssertionError("bounded source ID page retry loop fell through")
        if not raw_ids:
            return
        after_raw_id = raw_ids[-1]
        continuation = _current_source_selection_continuation(source_frame, raw_ids, frontier_rowid)
        # A page's ID scan can itself spend time near the frame limit. Renew
        # before handing its selection to the caller, which starts the page's
        # authority query on the same frame.
        _resume_current_source_census(source_frame, continuation)
        yield raw_ids


def apply_prepared_revision_census(
    archive_root: Path,
    *,
    active_index_path: Path,
    selected_raw_ids: list[str],
    prepared_inputs: Mapping[str, PreparedRetainedInput] | None = None,
    classification_proofs: Mapping[str, PreparedRawRevisionClassification] | None = None,
) -> RevisionCensusResult:
    """Apply the canonical retained census or captured byte classification.

    Sealed parser inputs are mandatory for a census. A classification-only
    pass requires already current source receipts and applies captured SQL
    decisions without parsing, scanning blobs or constructing writer rows.
    """
    if prepared_inputs is None and classification_proofs is None:
        raise RetainedPreparationRetryableError("source publication requires captured preparation")
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        if prepared_inputs is not None:
            state = _census_historical_revision_evidence(
                archive,
                selected_raw_ids=selected_raw_ids,
                prepared_inputs=prepared_inputs,
            )
        else:
            if uncensused_historical_revision_raw_ids(archive_root, selected_raw_ids):
                raise RetainedPreparationRetryableError("byte classification requires current parser receipts")
            state = _RevisionCensusState(0, 0, 0, set(), {}, {}, set())
        archive.commit()
        expanded, logical_keys = archive.expand_raw_membership_selection(selected_raw_ids)
        if classification_proofs is not None:
            if set(classification_proofs) - set(logical_keys):
                raise RetainedPreparationRetryableError("prepared byte classification key changed after census")
            # The byte scan and its source binding were prepared on a pinned
            # read-only snapshot. Apply only its SQL decisions here, after the
            # ordinary parser census has committed; any changed dependency
            # refuses the proof before source authority moves.
            source_conn = archive._ensure_source_conn()
            for logical_key in sorted(logical_keys):
                proof = classification_proofs.get(logical_key)
                # The census may have just moved a multi-session raw to
                # membership governance; its pending envelope is then no byte
                # chain to prove.
                if proof is not None and not pending_raw_envelope_has_membership_authority(source_conn, logical_key):
                    apply_prepared_raw_revision_classification(archive, proof)
            archive.commit()
    return RevisionCensusResult(
        state.scanned,
        state.classified,
        state.quarantined,
        expanded,
        logical_keys,
    )


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


def _replay_representative_raw_ids(sorted_keys: list[str], archive_root: Path) -> dict[str, str]:
    """Pick one raw per logical key to read that key's parent claim from.

    Newest acquisition wins and ``raw_id`` breaks ties, so a cohort whose
    raws claim different parents resolves to the same claim on every run.
    """
    representative: dict[str, str] = {}
    with read_frame(
        archive_root / "source.db", tier=ArchiveTier.SOURCE, timeout_class="background-read"
    ) as source_frame:
        conn = source_frame.connection
        for start in range(0, len(sorted_keys), _REPLAY_KEY_QUERY_CHUNK):
            chunk = sorted_keys[start : start + _REPLAY_KEY_QUERY_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT logical_source_key, raw_id
                FROM raw_sessions
                WHERE logical_source_key IN ({placeholders})
                ORDER BY logical_source_key, {raw_receipt_order_sql("raw_sessions")} DESC, raw_id ASC
                """,
                chunk,
            )
            for logical_source_key, raw_id in rows:
                representative.setdefault(str(logical_source_key), str(raw_id))
    return representative


def _replay_parent_claims(
    sorted_keys: list[str],
    archive: ArchiveStore,
    spill: _PreparedReplayInputs,
    archive_root: Path,
) -> dict[str, str | None]:
    """Read each key's claimed parent logical key from its representative raw.

    The claim comes from the parsed session whose OWN logical key is the key
    being resolved. A raw can carry many sessions, so reading the first
    one's claim would attribute a parent to a session that never made it;
    a key whose representative raw yields no matching session claims nothing.
    """
    representative = _replay_representative_raw_ids(sorted_keys, archive_root)
    keys_by_raw: dict[str, list[str]] = {}
    for key in sorted_keys:
        raw_id = representative.get(key)
        if raw_id is not None:
            keys_by_raw.setdefault(raw_id, []).append(key)

    claims: dict[str, str | None] = dict.fromkeys(sorted_keys, None)
    for raw_id, keys in keys_by_raw.items():
        try:
            sessions, _payload_bytes = spill.for_raw(archive, raw_id)
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
    archive: ArchiveStore,
    spill: _PreparedReplayInputs,
    archive_root: Path,
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

    claims = _replay_parent_claims(sorted_keys, archive, spill, archive_root)

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


def _required_shard_prepared_rows(
    raw_id: str,
    session: ParsedSession,
    bindings: Mapping[str, PreparedRows],
) -> dict[str, PreparedRows]:
    """Bind exactly the session the frozen replay is about to full-replace."""
    session_id = archive_session_id(origin_from_provider(session.source_name).value, session.provider_session_id)
    try:
        return {raw_id: bindings[session_id]}
    except KeyError as exc:
        raise ShardRefusedError(f"sealed shard has no rows for replay session {session_id}") from exc


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

    def close(self) -> None:
        failures: list[BaseException] = []
        for projection in self.projections.values():
            try:
                projection.close()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise BaseExceptionGroup("prepared membership cleanup failed", failures)


def prepare_membership_replay(
    archive: ArchiveStore,
    logical_key: str,
    prepared_inputs: Mapping[str, PreparedRetainedInput],
    *,
    stop: Callable[[], bool] | None = None,
) -> PreparedMembershipReplay:
    """Capture membership comparison on the retained read-only snapshot."""

    candidate_raw_ids = set(archive.raw_membership_rebuild_raw_ids(logical_key))
    # Rebuild replay also carries its current-pass `membership_candidates` for
    # quarantined/unknown raws; the persisted membership rows are the exact
    # read-only equivalent for the selected prepared component. The ordinary
    # rebuild selector intentionally filters these out until classification.
    candidate_raw_ids.update(
        str(row[0])
        for row in archive.source_connection.execute(
            "SELECT raw_id FROM raw_session_memberships WHERE logical_source_key = ? ORDER BY raw_id",
            (logical_key,),
        )
        if str(row[0]) in prepared_inputs
    )
    head_raw_id = archive.raw_revision_head_raw_id(logical_key)
    if head_raw_id is not None and archive._raw_revision_authority(head_raw_id) == "quarantined":
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
                raise RetainedPreparationRetryableError(
                    f"prepared membership candidate has parser failure for {logical_key}: {raw_id}"
                ) from outcome
            sessions, _size, _kind = outcome
            for session in sessions:
                session_key = f"{origin_from_provider(session.source_name).value}:{session.provider_session_id}"
                if session_key != logical_key:
                    continue
                member_sessions[raw_id] = session
                projections[raw_id] = session_revision_projection(session)
                revisions.append(
                    MembershipRevision(
                        raw_id,
                        projections[raw_id],
                        session.updated_at,
                        browser_snapshot_fidelity=_browser_snapshot_fidelity(session.ingest_flags),
                        provider_message_ids=(
                            session.messages.provider_message_ids(include_none=True)
                            if isinstance(session.messages, SqliteMessageSink)
                            else frozenset(message.provider_message_id for message in session.messages)
                        ),
                        provider_attachment_ids=frozenset(
                            attachment.provider_attachment_id for attachment in session.attachments
                        ),
                    )
                )
        classification = classify_membership_revisions(revisions, existing_accepted_raw_id=head_raw_id)
    except BaseException as primary:
        for projection in projections.values():
            try:
                projection.close()
            except BaseException as cleanup:
                primary.add_note(f"membership projection cleanup failed: {cleanup!r}")
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
    active_index_path: Path,
    selected_raw_ids: list[str],
    prepared_inputs: Mapping[str, PreparedRetainedInput],
    prepared_aggregates: Mapping[str, PreparedRetainedAggregate],
    prepared_writes: Mapping[tuple[str, str], PreparedSessionWrite],
    prepared_replay_plans: Mapping[str, tuple[str, ...]],
    prepared_membership_plans: Mapping[str, PreparedMembershipReplay],
    bulk_fts: bool = True,
    exact_fts_audit: bool = False,
) -> PreparedRevisionReplayResult:
    """Publish one retained component from its captured worker-sealed inputs.

    Parsing, byte classification and row lowering belong to canonical retained
    preparation. A changed plan or unavailable carrier refuses this attempt;
    publication never submits compute or falls back to inline parsing.
    """
    adoption_deferred = 0
    quarantined = 0
    stage_timings: dict[str, float] = {}
    stage_counts: dict[str, int] = {}
    logical_keys: set[str] = set()
    spill = _PreparedReplayInputs(prepared_inputs)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        census_started = time.perf_counter()
        census = _census_historical_revision_evidence(
            archive, selected_raw_ids=selected_raw_ids, prepared_inputs=prepared_inputs
        )
        stage_timings["census"] = time.perf_counter() - census_started
        receipt_started = time.perf_counter()
        censused_raw_ids, _censused_keys = archive.expand_raw_membership_selection(selected_raw_ids)
        archive.commit()
        stage_timings["census_receipt"] = time.perf_counter() - receipt_started
        membership_candidates = census.membership_candidates
        provisional_full_raw_ids = census.provisional_full_raw_ids
        selected_keys = archive.raw_revision_rebuild_logical_keys(selected_raw_ids)
        logical_keys.update(selected_keys)
        _selected_membership_raws, selected_membership_keys = archive.expand_raw_membership_selection(selected_raw_ids)
        membership_keys = set(selected_membership_keys)
        replayed = 0
        byte_replayed_keys: set[str] = set()
        work_event_keys = sorted(key for key in logical_keys if is_work_event_raw_id(key))
        replay_schedule = _lineage_aware_replay_schedule(
            logical_keys.difference(work_event_keys), archive, spill, archive_root
        )
        source_conn = archive._ensure_source_conn()

        def attachment_preparation(raw_id: str, session: ParsedSession):
            prepared = prepared_inputs.get(raw_id)
            if prepared is None or prepared.prepared_artifact is None:
                raise RetainedPreparationRetryableError(f"sealed attachment carrier is absent for {raw_id}")
            artifact = prepared.prepared_artifact
            return artifact.attachment_blobs(
                source_connection=source_conn,
                session_id=str(make_session_id(session.source_name, session.provider_session_id)),
            ), partial(
                artifact.iter_attachment_refs,
                source_path=prepared.source_path,
                acquired_at_ms=0,
                source_connection=source_conn,
            )

        ordered_logical_keys = [
            logical_key
            for logical_key in replay_schedule.order
            if not pending_raw_envelope_has_membership_authority(source_conn, logical_key)
        ]

        def classify_byte_cohort(logical_key: str) -> RevisionReplayPlan:
            classify_started = time.perf_counter()
            plan = archive.raw_revision_replay_plan(logical_key)
            if plan.accepted_raw_ids != prepared_replay_plans.get(logical_key, ()):
                raise RetainedPreparationRetryableError(f"prepared byte authority plan changed for {logical_key}")
            stage_timings["replay.classify_cohort"] = stage_timings.get("replay.classify_cohort", 0.0) + (
                time.perf_counter() - classify_started
            )
            return plan

        for logical_key in ordered_logical_keys:
            plan = classify_byte_cohort(logical_key)
            if not plan.accepted_raw_ids:
                convertible = archive.convertible_full_revision_raw_ids(logical_key)
                for raw_id in convertible:
                    spill_started = time.perf_counter()
                    sessions, _payload_bytes = spill.for_raw(archive, raw_id)
                    stage_timings["spill_load"] = stage_timings.get("spill_load", 0.0) + (
                        time.perf_counter() - spill_started
                    )
                    if len(sessions) != 1:
                        raise RuntimeError(f"full revision {raw_id} no longer parses to one session")
                    archive.replace_raw_membership_census(
                        raw_id,
                        sessions,
                        parser_fingerprint=raw_authority_parser_fingerprint(),
                        censused_at_ms=0,
                        detail=HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
                        retire_full_revision_governance=True,
                        revision_authority=RawRevisionAuthority.QUARANTINED,
                    )
                    fresh_session = sessions[0]
                    fresh_key = (
                        f"{origin_from_provider(fresh_session.source_name).value}:{fresh_session.provider_session_id}"
                    )
                    membership_candidates.setdefault(fresh_key, set()).add(raw_id)
                    membership_keys.add(fresh_key)
                membership_keys.add(logical_key)
                continue
            parsed_by_raw_id: dict[str, ParsedSession] = {}
            retained_bytes = 0
            for raw_id in plan.accepted_raw_ids:
                spill_started = time.perf_counter()
                sessions, payload_bytes = spill.for_raw(archive, raw_id)
                stage_timings["spill_load"] = stage_timings.get("spill_load", 0.0) + (
                    time.perf_counter() - spill_started
                )
                if len(sessions) != 1:
                    raise RuntimeError(f"classified raw revision {raw_id} no longer parses to one session")
                parsed_by_raw_id[raw_id] = sessions[0]
                retained_bytes += payload_bytes
            prepared_aggregate_session: ParsedSession | None = None
            prepared_aggregate_path: Path | None = None
            if len(plan.accepted_raw_ids) > 1:
                prepared_aggregate_session, prepared_aggregate_path = _validated_prepared_aggregate(
                    logical_key,
                    tuple(plan.accepted_raw_ids),
                    prepared_inputs=prepared_inputs,
                    prepared_aggregates=prepared_aggregates,
                )
            accepted_sessions = [parsed_by_raw_id[raw_id] for raw_id in plan.accepted_raw_ids]
            adoptable_started = time.perf_counter()
            adoptable = archive.raw_revision_replay_adoptable(
                [prepared_aggregate_session] if prepared_aggregate_session is not None else accepted_sessions
            )
            stage_timings["replay.adoptable_check"] = stage_timings.get("replay.adoptable_check", 0.0) + (
                time.perf_counter() - adoptable_started
            )
            if not adoptable:
                archive.defer_raw_revision_adoption(
                    plan.logical_source_key,
                    plan.accepted_raw_ids,
                    [prepared_aggregate_session] if prepared_aggregate_session is not None else accepted_sessions,
                )
                provisional_raw_ids = provisional_full_raw_ids.get(logical_key, set())
                plan_raw_ids = {application.raw_id for application in plan.applications}
                if plan_raw_ids and plan_raw_ids <= provisional_raw_ids:
                    archive.release_provisional_full_revisions(sorted(plan_raw_ids))
                adoption_deferred += len(plan.accepted_raw_ids)
                continue
            try:
                tip_raw_id = plan.accepted_raw_ids[-1]
                prepared_write = _prepared_write_for(
                    prepared_writes, tip_raw_id, prepared_aggregate_session or parsed_by_raw_id[tip_raw_id]
                )
                _require_prepared_cross_acquisition_write(
                    archive,
                    prepared_aggregate_session or parsed_by_raw_id[tip_raw_id],
                    accepted_raw_id=tip_raw_id,
                    prepared_write=prepared_write,
                    prepared_inputs=prepared_inputs,
                )
                attachment_views = {
                    raw_id: attachment_preparation(raw_id, parsed_by_raw_id[raw_id]) for raw_id in plan.accepted_raw_ids
                }
                composed_session = prepared_aggregate_session or parsed_by_raw_id[tip_raw_id]
                shard_path = prepared_aggregate_path or _prepared_shard_path(prepared_inputs, tip_raw_id)
                try:
                    with archive.attached_session_shard(shard_path, required=True) as bindings:
                        prepared = _required_shard_prepared_rows(tip_raw_id, composed_session, bindings)
                        archive.apply_raw_revision_replay(
                            plan,
                            parsed_by_raw_id,
                            acquired_at_ms=0,
                            stage_timings_s=stage_timings,
                            manage_transaction=True,
                            bulk_fts=bulk_fts,
                            bulk_build=False,
                            fresh_build=False,
                            fresh_build_batch=None,
                            prepared_aggregate_rows=prepared[tip_raw_id],
                            prepared_aggregate_session=composed_session,
                            prepared_required_raw_ids=frozenset({tip_raw_id}),
                            prepared_write=prepared_write,
                            preacquired_attachment_blobs_by_raw_id={
                                raw_id: view[0] for raw_id, view in attachment_views.items()
                            },
                            preacquired_attachment_refs_by_raw_id={
                                raw_id: view[1] for raw_id, view in attachment_views.items()
                            },
                        )
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
        for logical_key in sorted(membership_keys):
            if logical_key in byte_replayed_keys and (
                not membership_key_has_pending_envelope_member(source_conn, logical_key)
            ):
                continue
            member_sessions: dict[str, ParsedSession] = {}
            projections = {}
            retained_bytes = 0
            candidates_started = time.perf_counter()
            candidate_raw_ids = set(archive.raw_membership_rebuild_raw_ids(logical_key))
            candidate_raw_ids.update(membership_candidates.get(logical_key, ()))
            stage_timings["membership.candidates"] = stage_timings.get("membership.candidates", 0.0) + (
                time.perf_counter() - candidates_started
            )
            head_raw_id = archive.raw_revision_head_raw_id(logical_key)
            if head_raw_id is not None and archive._raw_revision_authority(head_raw_id) == "quarantined":
                candidate_raw_ids.add(head_raw_id)
            membership_plan = prepared_membership_plans.get(logical_key)
            if (
                membership_plan is None
                or membership_plan.candidate_raw_ids != tuple(sorted(candidate_raw_ids))
                or membership_plan.head_raw_id != head_raw_id
            ):
                raise RetainedPreparationRetryableError(f"prepared membership authority changed for {logical_key}")
            member_sessions = dict(membership_plan.sessions)
            projections = dict(membership_plan.projections)
            classification = membership_plan.classification
            if classification.ambiguous_raw_ids:
                quarantined += len(classification.ambiguous_raw_ids)
            accepted_sessions = [member_sessions[raw_id] for raw_id in classification.accepted_raw_ids]
            if accepted_sessions and (not archive.raw_revision_replay_adoptable(accepted_sessions)):
                archive.defer_raw_revision_adoption(logical_key, classification.accepted_raw_ids, accepted_sessions)
                adoption_deferred += len(classification.accepted_raw_ids)
                continue
            try:
                if classification.accepted_raw_ids:
                    accepted_raw_id = classification.accepted_raw_ids[-1]
                    _require_prepared_cross_acquisition_write(
                        archive,
                        member_sessions[accepted_raw_id],
                        accepted_raw_id=accepted_raw_id,
                        prepared_write=_prepared_write_for(
                            prepared_writes, accepted_raw_id, member_sessions[accepted_raw_id]
                        ),
                        prepared_inputs=prepared_inputs,
                    )
                if not classification.accepted_raw_ids:
                    archive.apply_raw_membership_classification(
                        logical_key,
                        classification,
                        member_sessions,
                        projections,
                        acquired_at_ms=0,
                        stage_timings_s=stage_timings,
                        manage_transaction=True,
                        bulk_fts=bulk_fts,
                        bulk_build=False,
                        fresh_build=False,
                        fresh_build_batch=None,
                        prepared_write=_prepared_write_for(
                            prepared_writes,
                            classification.accepted_raw_ids[-1],
                            member_sessions[classification.accepted_raw_ids[-1]],
                        )
                        if classification.accepted_raw_ids
                        else None,
                    )
                else:
                    accepted_raw_id = classification.accepted_raw_ids[-1]
                    accepted_session = member_sessions[accepted_raw_id]
                    attachment_blobs, attachment_refs = attachment_preparation(accepted_raw_id, accepted_session)
                    try:
                        with archive.attached_session_shard(
                            _prepared_shard_path(prepared_inputs, accepted_raw_id), required=True
                        ) as bindings:
                            prepared = _required_shard_prepared_rows(accepted_raw_id, accepted_session, bindings)
                            archive.apply_raw_membership_classification(
                                logical_key,
                                classification,
                                member_sessions,
                                projections,
                                acquired_at_ms=0,
                                stage_timings_s=stage_timings,
                                manage_transaction=True,
                                bulk_fts=bulk_fts,
                                bulk_build=False,
                                fresh_build=False,
                                fresh_build_batch=None,
                                prepared_by_raw_id=prepared,
                                prepared_required_raw_ids=frozenset({accepted_raw_id}),
                                prepared_write=_prepared_write_for(prepared_writes, accepted_raw_id, accepted_session),
                                preacquired_attachment_blobs=attachment_blobs,
                                preacquired_attachment_refs=attachment_refs,
                            )
                    except PreparedSessionWriteRefusedError as exc:
                        raise RetainedPreparationRetryableError(
                            f"prepared membership replay dependency changed for {logical_key}"
                        ) from exc
            except sqlite3.IntegrityError as exc:
                raise sqlite3.IntegrityError(
                    f"apply_prepared_revision_replay: membership replay failed for logical_key={logical_key!r}: {exc}"
                ) from exc
            if classification.accepted_raw_ids:
                replayed += 1
        for logical_key in work_event_keys:
            plan = classify_byte_cohort(logical_key)
            if plan.accepted_raw_ids != (logical_key,):
                message = f"retained work event {logical_key} lost its singleton byte authority"
                raise RuntimeError(message)
            (event_session,), _event_bytes = spill.for_raw(archive, logical_key)
            event_session_id = str(make_session_id(event_session.source_name, event_session.provider_session_id))
            if (
                archive._conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (event_session_id,)).fetchone()
                is None
            ):
                _LOGGER.warning("work_event_session_absent: raw_id=%s session_id=%s", logical_key, event_session_id)
                adoption_deferred += 1
                continue
            archive._index_parsed_for_retained_raw(
                event_session,
                raw_id=logical_key,
                source_index=-1,
                stage_timings_s=stage_timings,
                stage_timing_prefix="replay.work_event",
                manage_transaction=True,
                preacquired_attachment_blobs={},
                finalize_raw_parse=True,
            )
            replayed += 1
            byte_replayed_keys.add(logical_key)
        if replayed and (not adoption_deferred):
            if exact_fts_audit:
                from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync

                fts_snapshot = fts_invariant_snapshot_sync(archive._conn)
                if not fts_snapshot.messages.ready:
                    raise RuntimeError("retained replay found messages_fts out of sync")
            archive.commit()
        if stage_timings:
            stage_timings["total"] = time.perf_counter() - census_started
            _LOGGER.info(
                "backfill stage timings: %s",
                " ".join(
                    (f"{key}={value:.1f}s" for key, value in sorted(stage_timings.items(), key=lambda kv: -kv[1]))
                ),
            )
    return PreparedRevisionReplayResult(
        census.scanned,
        census.classified,
        replayed,
        census.quarantined + quarantined,
        adoption_deferred,
        stage_timings_s=stage_timings,
        stage_counts=stage_counts,
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
        self._session_ids: list[str] = []

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
                index_conn=self._index_conn,
                source_conn=self._source_conn,
                blob_root=self._blob_root,
                source_path=self._source_path,
                captured_zip_coordinate=self._captured_zip_coordinate,
            )
        return stamp_enrichment_evidence(self._provider, self._cached, spec.enrich_session(session, self._cached))

    def close(self) -> None:
        from polylogue.sources.assembly import close_sidecar_data

        if self._cached is not None:
            close_sidecar_data(self._cached)
            self._cached = None

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


def enrich_sessions_from_archive(
    archive: Any,
    provider: Provider,
    source_path: str,
    sessions: Sequence[ParsedSession],
    *,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
) -> list[ParsedSession]:
    """Enrich a writer-side parse from the archive's own retained evidence.

    An ``UNKNOWN`` acquisition provider resolves to the parser's, as retained
    replay does before it enriches, so both routes find the same assembly.
    """
    if provider is Provider.UNKNOWN and sessions:
        provider = sessions[0].source_name
    enricher = RetainedSessionEnricher(
        provider,
        source_path=source_path,
        captured_zip_coordinate=captured_zip_coordinate,
        index_conn=archive.index_connection,
        source_conn=archive.source_connection,
        blob_root=Path(archive.archive_root) / "blob",
    )

    try:
        return enricher.enrich_all(sessions)
    finally:
        enricher.close()


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
        index_conn=index_conn,
        source_conn=source_conn,
        blob_root=blob_root,
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
    index_conn: sqlite3.Connection | None,
    source_conn: sqlite3.Connection | None,
    blob_root: Path | None,
    source_path: str | None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
    provider_session_ids: Iterable[str] | None = None,
) -> SidecarData:
    """Read the exact retained assembly evidence used by enrichment.

    ``provider_session_ids`` replaces ``sessions`` when the caller holds only
    the identities (the evidence depends on nothing else of a session).
    """

    sidecar_data = cast("SidecarData", {})
    reads_index = _replay_enrichment_reads_index(provider)
    if reads_index and index_conn is None:
        _count_enrichment_degradation("codex_titles_without_index_conn")
    if reads_index and index_conn is not None:
        from polylogue.sources.codex_state_projection import read_thread_titles

        thread_ids = (
            list(provider_session_ids)
            if provider_session_ids is not None
            else [session.provider_session_id for session in sessions if session.provider_session_id]
        )
        titles = read_thread_titles(index_conn, thread_ids=thread_ids, source_path=source_path)
        if titles:
            sidecar_data = cast("SidecarData", {"retained_state_titles": titles})
    if source_conn is None or blob_root is None or not source_path:
        _count_enrichment_degradation("retained_assembly_without_source_evidence")
    if source_conn is not None and blob_root is not None and source_path:
        # polylogue-ximhz: Claude Code index/history and ChatGPT asset maps
        # are retained source artifacts. Replay resolves them from the source
        # tier and the blob store -- never from a file beside the original
        # source path, which may no longer exist.
        from polylogue.sources.retained_assembly import with_retained_assembly_evidence
        from polylogue.storage.blob_store import BlobStore

        sidecar_data = with_retained_assembly_evidence(
            sidecar_data,
            provider=provider,
            source_conn=source_conn,
            blob_store=BlobStore(blob_root),
            source_path=source_path,
            captured_zip_coordinate=captured_zip_coordinate,
        )
    return sidecar_data


def parse_retained_raw_sessions(archive: ArchiveStore, raw_id: str) -> list[ParsedSession]:
    """Parse retained raw evidence without eagerly loading stream records.

    Raw-revision replay is shared by historical repair and the live full and
    append routes.  Keeping the provider-shape decision here prevents a
    seemingly harmless live replay helper from reintroducing ``read_all()``
    for Codex/Claude JSONL evidence.
    """
    provider, blob_hash, source_path, kind, _payload_size = archive.raw_revision_descriptor(raw_id)
    profile_identity = archive.raw_profile_identity(raw_id)
    fallback_timestamp = archive.raw_revision_file_mtime(raw_id)

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

    # polylogue-u19l: an append-kind raw's own record stream may carry no
    # self-describing identity of its own (a Codex append delta has no
    # session_meta record) -- recover the identity hint recorded at write
    # time (``sources/live/batch.py``'s ``_append_payload_for_provider`` /
    # ``write_raw_payload``'s ``native_id``) and use it as the parser's
    # fallback_id instead of the bare filename stem. Historical rows
    # (written before this) have no recorded native_id and fall through to
    # the unchanged stem-based fallback -- their stored bytes still carry
    # the synthetic session_meta line that made this unnecessary for them.
    fallback_id_override = archive.raw_native_id(raw_id) if kind is RawRevisionKind.APPEND else None
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
            provider, _evidence = _resolved_retained_provider(stream_payload, source_path)
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
                        profile_identity=profile_identity,
                    )
                )
        if provider is Provider.UNKNOWN:
            raise UnsupportedRetainedJsonShapeError(f"retained input has no recognized provider shape: {source_path}")
        _provider, eager_payload, _source_path, _eager_kind = archive.raw_revision_material(raw_id)
        payload_path = archive.blob_path_for_hash(blob_hash) if provider is Provider.HERMES else None
        return normalize_replay(
            _parse_one(
                provider,
                eager_payload,
                source_path,
                payload_path=payload_path,
                archive_root=archive.archive_root,
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
                    profile_identity=profile_identity,
                )
            )
    _provider, eager_payload, _source_path, _eager_kind = archive.raw_revision_material(raw_id)
    payload_path = archive.blob_path_for_hash(blob_hash) if provider is Provider.HERMES else None
    return normalize_replay(
        _parse_one(
            provider,
            eager_payload,
            source_path,
            payload_path=payload_path,
            archive_root=archive.archive_root,
            profile_identity=profile_identity,
            fallback_id_override=fallback_id_override,
        )
    )


def _retained_codex_state_descriptor(archive: ArchiveStore, raw_id: str) -> tuple[Path, str, str, str] | None:
    """Identify one immutable retained Codex state export without mutating it."""
    provider, blob_hash, source_path, _kind, _payload_size = archive.raw_revision_descriptor(raw_id)
    if provider is not Provider.CODEX:
        return None
    state_path = archive.blob_path_for_hash(blob_hash)
    if state_path is None:
        return None
    if not is_declared_logical_export(state_path, source_path):
        return None
    state_kind = codex_state.classify_codex_sqlite_path(state_path, immutable=True)
    if state_kind not in codex_state.IN_SCOPE_KINDS:
        return None
    return state_path, source_path, state_kind, blob_hash


def _replay_retained_codex_state_evidence(
    archive: ArchiveStore,
    raw_id: str,
    prepared_inputs: Mapping[str, PreparedRetainedInput] | None,
) -> bool:
    """Publish worker-captured state evidence without reading the export."""
    prepared = prepared_inputs.get(raw_id) if prepared_inputs is not None else None
    artifact = prepared.prepared_artifact if prepared is not None else None
    if artifact is None or artifact.codex_state_kind is None:
        return False
    # Descriptor, stat and enrichment dependencies are validated before any
    # source or index mutation, just like session-bearing retained inputs.
    _prepared_retained_outcome(archive, raw_id, prepared_inputs)
    record_codex_state_snapshot_terminal(
        archive,
        raw_id,
        prepared_state=artifact,
        state_kind=artifact.codex_state_kind,
        source_path=prepared.source_path,
        acquired_at_ms=archive.raw_revision_observed_at_ms(raw_id),
        censused_at_ms=0,
        blob_hash=prepared.blob_hash,
    )
    return True


class _PreparedReplayInputs:
    """Lookup view over the existing complete worker-sealed carrier."""

    def __init__(self, prepared_inputs: Mapping[str, PreparedRetainedInput]) -> None:
        self._prepared_inputs = prepared_inputs

    def for_raw(self, archive: ArchiveStore, raw_id: str) -> tuple[Sequence[ParsedSession], int]:
        outcome = _prepared_retained_outcome(archive, raw_id, self._prepared_inputs)
        if isinstance(outcome, Exception):
            raise RetainedPreparationRetryableError(f"parser-refused raw {raw_id} reached replay") from outcome
        sessions, payload_bytes, _kind = outcome
        return sessions, payload_bytes


def _prepared_shard_path(prepared_inputs: Mapping[str, PreparedRetainedInput], raw_id: str) -> Path:
    prepared = prepared_inputs.get(raw_id)
    artifact = prepared.prepared_artifact if prepared is not None else None
    if artifact is None or artifact.shard_path is None:
        raise ShardRefusedError(f"retained replay has no sealed shard for raw {raw_id}")
    return artifact.shard_path


LEGACY_PAGE_IMAGE_CENSUS_DETAIL = (
    "retained legacy SQLite page image; no current parser reads this material and it is "
    "not a logical export, so it produces no session"
)


def _retained_page_image_raw(archive: ArchiveStore, raw_id: str) -> bool:
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
    blob_path = archive.blob_path_for_hash(blob_hash)
    return blob_path is not None and is_sqlite_page_image(blob_path)


def _settle_terminal_raw_refusal(
    archive: ArchiveStore,
    raw_id: str,
    error: Exception,
    *,
    source_index: int,
    manage_transaction: bool,
) -> bool:
    """Record a typed retained refusal beside its unchanged source bytes.

    Returns ``False`` when ``error`` is not a decode refusal that earns
    terminal evidence (``terminal_decode_evidence``). Otherwise the raw
    carries ``terminal_corrupt_input`` (or the unknown-provider decode
    kind), is marked parse-failed, and gets a complete current-parser
    receipt with no identity. The failure artifact is not session-parsable,
    so ``raw_membership_census_rows`` reports the raw as terminal and later
    passes settle it without parsing it again.
    """
    provider, _blob_hash, source_path, _kind, _size = archive.raw_revision_descriptor(raw_id)
    evidence = (
        RawFailureEvidenceKind.TERMINAL_RETAINED_ZIP_MEMBERSHIP_UNPROVED
        if isinstance(error, RetainedZipMembershipUnprovedError)
        else RawFailureEvidenceKind.TERMINAL_MISSING_PROFILE_IDENTITY
        if isinstance(error, MissingProfileIdentityError)
        else terminal_decode_evidence(error, provider=provider)
    )
    if evidence is None:
        return False
    conn = archive._ensure_source_conn()
    with conn if manage_transaction else nullcontext():
        record_raw_failure_evidence(
            archive,
            raw_id,
            provider=provider,
            source_path=source_path,
            source_index=source_index,
            acquired_at_ms=archive.raw_revision_observed_at_ms(raw_id),
            kind=evidence,
            manage_transaction=False,
        )
        apply_source_raw_state_update(
            conn,
            raw_id,
            state=_raw_parse_failure_state(provider, error),
            manage_transaction=False,
        )
        record_current_parser_source_census(conn, raw_id)
    return True


def _persist_terminal_non_session_artifact(
    archive: ArchiveStore,
    raw_id: str,
    *,
    provider: Provider,
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
    observed_at_ms = archive.raw_revision_observed_at_ms(raw_id)
    upsert_raw_artifact(
        archive._ensure_source_conn(),
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
    apply_source_raw_state_update(
        archive._ensure_source_conn(),
        raw_id,
        state=_raw_parse_success_state(provider),
        manage_transaction=manage_transaction,
    )
    return True


def _parse_one(
    provider: Provider,
    payload: bytes,
    source_path: str,
    *,
    profile_identity: str | None = None,
    payload_path: Path | None = None,
    archive_root: Path | None = None,
    fallback_id_override: str | None = None,
) -> list[ParsedSession]:
    return require_positive_conversational_evidence(
        _parse_one_raw(
            provider,
            payload,
            source_path,
            payload_path=payload_path,
            archive_root=archive_root,
            profile_identity=profile_identity,
            fallback_id_override=fallback_id_override,
        ),
        provider=provider,
        source_path=source_path,
    )


def _parse_one_raw(
    provider: Provider,
    payload: bytes,
    source_path: str,
    *,
    profile_identity: str | None = None,
    payload_path: Path | None = None,
    archive_root: Path | None = None,
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
        return sessions
    source_name = Path(source_path).name
    fallback_id = fallback_id_override or Path(source_path).stem
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
                return hermes_state.parse_state_db(
                    sqlite_path,
                    fallback_id=fallback_id,
                    profile_identity=profile_identity,
                    immutable=True,
                )
            if hermes_verification.looks_like_verification_evidence_db_path(sqlite_path, immutable=True):
                return hermes_verification.parse_verification_evidence_db(
                    sqlite_path,
                    fallback_id=fallback_id,
                    profile_identity=profile_identity,
                    immutable=True,
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
                return list(antigravity.parse_trajectory_db(sqlite_path, fallback_id=fallback_id, immutable=True))
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
    sidecar_resolver = _retained_sidecar_resolver(archive_root)
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


def _retained_sidecar_resolver(archive_root: Path | None) -> SidecarResolver | None:
    """Resolve overflowed tool outputs from retained bytes during replay.

    polylogue-cq1ql: replay reads a retained blob, not the file it came from,
    so the sidecar join must read the archive too -- ``None`` (no archive root
    in scope) keeps the routing default, which is acquisition-time filesystem
    resolution. Every replay entry point that already carries an archive root
    passes it, so a session replays to the same full text, outcome and
    ownership after the original tree is gone.
    """
    if archive_root is None:
        return None
    from polylogue.sources.live.sidecar_resolution import RetainedSidecarResolver

    return RetainedSidecarResolver(archive_root)


def _parse_stream(
    provider: Provider,
    payload: BinaryIO,
    source_path: str,
    *,
    profile_identity: str | None = None,
    fallback_id_override: str | None = None,
    archive_root: Path | None = None,
) -> list[ParsedSession]:
    # polylogue-9ykn: see ``_parse_one``'s comment -- the same positive-
    # conversational-evidence gate applies to the streaming replay path.
    return require_positive_conversational_evidence(
        _parse_stream_raw(
            provider,
            payload,
            source_path,
            fallback_id_override=fallback_id_override,
            archive_root=archive_root,
            profile_identity=profile_identity,
        ),
        provider=provider,
        source_path=source_path,
    )


def _parse_stream_raw(
    provider: Provider,
    payload: BinaryIO,
    source_path: str,
    *,
    profile_identity: str | None = None,
    fallback_id_override: str | None = None,
    archive_root: Path | None = None,
) -> list[ParsedSession]:
    if provider is Provider.HERMES and profile_identity is None:
        raise MissingProfileIdentityError("retained Hermes input has no captured profile identity receipt")

    source_name = Path(source_path).name
    fallback_id = fallback_id_override or Path(source_path).stem
    stream = _retained_jsonl_stream(payload, source_name, source_path)
    return parse_stream_payload(
        provider,
        stream,
        fallback_id,
        source_path=source_path,
        profile_identity=profile_identity,
        sidecar_resolver=_retained_sidecar_resolver(archive_root),
    )


__all__ = [
    "raw_authority_parser_fingerprint",
    "RetainedSessionEnricher",
    "PreparedRevisionReplayResult",
    "RevisionCensusResult",
    "apply_prepared_revision_replay",
    "apply_prepared_revision_census",
    "enrich_sessions_from_archive",
    "open_retained_session_enricher",
    "require_current_parser_source_census",
    "uncensused_historical_revision_raw_ids",
    "parse_retained_raw_sessions",
]
