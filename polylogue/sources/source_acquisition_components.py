"""Typed local-file and ZIP acquisition components."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pickle
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TypeAlias, cast

import ijson

from polylogue.archive.artifact_taxonomy import classify_artifact
from polylogue.archive.zip_admission import ZipAdmission
from polylogue.config import Source
from polylogue.core.content_identity import (
    ContentIdentityRefusal,
    payload_content_identity,
    stream_payload_content_identity,
)
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue, is_json_value, normalize_json_decimal
from polylogue.core.json import dumps_bytes as json_dumps_bytes
from polylogue.core.metrics import read_current_rss_mb, read_peak_rss_self_mb
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate, MemberAddressingMode, split_zip_member_text
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.storage.sqlite.archive_tiers.source_items import CapturedSourceInputIdentity

from . import decoders as _decoders
from .acquisition_boundary import (
    bind_stream,
    capture_bound_path,
    capture_bound_stream,
    drain_bound,
    open_bound_member,
    refuse_declared_foreign,
    release_refused_capture,
)
from .decoder_zip import (
    ZIP_JSON_SUFFIXES,
    declared_artifact_provider,
    is_declared_artifact_path,
    provider_detection_path,
)
from .decoders import _zip_entry_provider_hint
from .dispatch import (
    GROUP_PROVIDERS,
    detect_provider,
    detect_provider_from_raw_stream_evidence,
    detect_provider_from_stream_evidence,
    is_jsonl_source_path,
)
from .parsers.base import RawSessionData
from .parsers.hermes_identity import declares_profile_identity
from .pickle_spool import PickleSpool
from .source_staging import SourceInputBinding, bind_source_input
from .sqlite_snapshot import is_sqlite_path, snapshot_sqlite_to_blob

_HEARTBEAT_INTERVAL_S = 5.0
_REVISION_CHUNK_BYTES = 1024 * 1024
AcquisitionObservation: TypeAlias = JSONDocument
ObservationCallback: TypeAlias = Callable[[AcquisitionObservation], None]
StatusCallback: TypeAlias = Callable[[str], None]
CursorState: TypeAlias = CursorStatePayload


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Exact identity of an input already retained in the blob store."""

    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SourceReadContext:
    """Common acquisition dependencies for one local source artifact."""

    source: Source
    path: Path
    file_mtime: str | None
    provider_hint: Provider
    blob_store: BlobStore
    observation_callback: ObservationCallback | None = None
    status_callback: StatusCallback | None = None
    retained_blob: ArtifactIdentity | None = None
    captured_input_identity: CapturedSourceInputIdentity | None = None


@dataclass(frozen=True, slots=True)
class ZipEntryReadContext:
    """Acquisition dependencies for one ZIP member."""

    source: Source
    zip_path: Path
    entry: zipfile.ZipInfo
    file_mtime: str | None
    provider_hint: Provider
    blob_store: BlobStore
    observation_callback: ObservationCallback | None = None
    status_callback: StatusCallback | None = None
    # The origin the archive's location binds; ``None`` for an import-inbox
    # export, whose members classify.
    bound_provider: Provider | None = None
    captured_input_identity: CapturedSourceInputIdentity | None = None
    entry_ordinal: int | None = None
    container_blob_hash: str | None = None
    decoder_fingerprint: str | None = None

    @property
    def source_path(self) -> str:
        return f"{self.zip_path}:{self.entry.filename}"


@dataclass(frozen=True, slots=True)
class DetectedEntryPayload:
    """JSON stream payload paired with provider detection timing."""

    provider: Provider
    payload: JSONValue
    detect_provider_ms: float


@dataclass(frozen=True, slots=True)
class SerializedSplitPayload:
    """A serialized acquisition unit ready for blob persistence.

    ``addressing_mode`` is the unit's address kind, not a summary of
    ``source_index``: a whole-member document has no element index, and a
    member that yields exactly one element still yields it as an element.
    """

    provider: Provider
    payload_bytes: bytes
    source_index: int | None
    addressing_mode: MemberAddressingMode = MemberAddressingMode.ELEMENT_OF_CONTAINER
    content_identity: str | None = None


@dataclass(slots=True)
class SplitPayloadBuffer:
    """Buffer ZIP payloads until an entry proves it contains multiple sessions.

    An element whose content identity is refused keeps its source index and
    is recorded in ``refusals``; the elements after it are still emitted.
    """

    _pending: list[tuple[Provider, bytes]] = field(default_factory=list)
    _next_source_index: int = 0
    did_split: bool = False
    refusals: list[ContentIdentityRefusal] = field(default_factory=list)

    @property
    def pending_index(self) -> int:
        return self._next_source_index + len(self._pending)

    def element(self, provider: Provider, payload_bytes: bytes, index: int) -> SerializedSplitPayload | None:
        """One split element with its identity, or ``None`` when that identity is refused."""
        try:
            identity = payload_content_identity(payload_bytes)
        except ContentIdentityRefusal as refusal:
            self.refusals.append(refusal)
            return None
        return SerializedSplitPayload(
            provider=provider,
            payload_bytes=payload_bytes,
            source_index=index,
            addressing_mode=MemberAddressingMode.ELEMENT_OF_CONTAINER,
            content_identity=identity,
        )

    def add_grouped(self, provider: Provider, payload_bytes: bytes) -> SerializedSplitPayload | None:
        """A grouped-provider element observed after the split; it takes the next index, emitted or refused."""
        index = self._next_source_index
        self._next_source_index += 1
        return self.element(provider, payload_bytes, index)

    def add(self, provider: Provider, payload_bytes: bytes) -> tuple[SerializedSplitPayload, ...]:
        if self.did_split:
            payload = self.element(provider, payload_bytes, self._next_source_index)
            self._next_source_index += 1
            return () if payload is None else (payload,)

        self._pending.append((provider, payload_bytes))
        if len(self._pending) < 2:
            return ()

        self.did_split = True
        emitted = tuple(
            payload
            for index, (pending_provider, pending_payload_bytes) in enumerate(
                self._pending,
                start=self._next_source_index,
            )
            if (payload := self.element(pending_provider, pending_payload_bytes, index)) is not None
        )
        self._next_source_index += len(self._pending)
        self._pending.clear()
        return emitted


@dataclass(slots=True)
class _ZipEntrySplitState:
    did_split: bool = False
    detected_provider: Provider = Provider.UNKNOWN


def _artifact_payload(value: object) -> JSONValue:
    normalized = normalize_json_decimal(value)
    return normalized if is_json_value(normalized) else {}


def _heartbeat_label(source_path: str) -> str:
    split = split_zip_member_text(source_path)
    if split is None:
        return Path(source_path).name or source_path
    base_path, zip_entry = split
    return f"{Path(base_path).name}:{zip_entry}"


def make_status_heartbeat(
    status_callback: StatusCallback | None,
    *,
    source_name: str,
    source_path: str,
) -> Callable[[], None] | None:
    if status_callback is None:
        return None

    label = _heartbeat_label(source_path)
    last_emit = 0.0

    def emit() -> None:
        nonlocal last_emit
        now = time.monotonic()
        if last_emit and now - last_emit < _HEARTBEAT_INTERVAL_S:
            return
        last_emit = now
        status_callback(f"Scanning [{source_name}] reading {label}")

    return emit


def observe_acquisition(
    observation_callback: ObservationCallback | None,
    *,
    phase: str,
    source_path: str,
    provider_hint: Provider,
    blob_size: int,
    source_index: int | None = None,
    **extra: object,
) -> None:
    if observation_callback is None:
        return
    current_rss_mb = read_current_rss_mb()
    peak_rss_self_mb = read_peak_rss_self_mb()
    if current_rss_mb is None and peak_rss_self_mb is None:
        return
    payload: AcquisitionObservation = {
        "phase": phase,
        "source_path": source_path,
        "provider_hint": str(provider_hint),
        "blob_size": blob_size,
        "blob_mb": round(blob_size / (1024 * 1024), 3),
        "source_index": source_index,
        "current_rss_mb": current_rss_mb,
        "peak_rss_self_mb": peak_rss_self_mb,
    }
    for key, value in extra.items():
        if is_json_value(value):
            payload[key] = value
    observation_callback(payload)


def raw_data_record(
    *,
    source_path: str,
    canonical_source_path: str | None = None,
    captured_profile_key: str | None = None,
    captured_profile_source_path: str | None = None,
    captured_file_observation: tuple[int, int, int, int, int] | None = None,
    file_mtime: str | None,
    provider_hint: Provider,
    blob_hash: str,
    blob_size: int,
    source_index: int | None = None,
    blob_publication_receipt_id: str | None = None,
    addressing_mode: MemberAddressingMode | None = None,
    content_identity: str | None = None,
) -> RawSessionData:
    return RawSessionData(
        raw_bytes=b"",
        source_path=source_path,
        canonical_source_path=canonical_source_path,
        captured_profile_key=captured_profile_key,
        captured_profile_source_path=captured_profile_source_path,
        captured_file_observation=captured_file_observation,
        source_index=source_index,
        file_mtime=file_mtime,
        provider_hint=provider_hint,
        blob_hash=blob_hash,
        blob_size=blob_size,
        blob_publication_receipt_id=blob_publication_receipt_id,
        addressing_mode=addressing_mode,
        content_identity=content_identity,
    )


def iter_entry_payloads(
    handle: IO[bytes],
    *,
    stream_name: str,
    provider_hint: Provider,
    bound_provider: Provider | None = None,
) -> Iterable[DetectedEntryPayload]:
    """Yield payloads from a streamed JSON/JSONL document with provider hints.

    ``bound_provider`` names the origin the document's location binds; the
    handle reads through the acquisition boundary, which refuses a record of
    another origin before it is decoded here.
    """
    handle = bind_stream(handle, stream_name, bound_provider)
    current_provider = provider_hint
    for payload in _decoders._iter_json_stream(handle, stream_name):
        normalized_payload = _artifact_payload(payload)
        detect_start = time.perf_counter()
        detected_provider = detect_provider(normalized_payload)
        detect_provider_ms = (time.perf_counter() - detect_start) * 1000.0
        provider = detected_provider or current_provider
        if detected_provider is not None and detected_provider is not Provider.UNKNOWN:
            current_provider = detected_provider
        yield DetectedEntryPayload(provider, normalized_payload, detect_provider_ms)


def make_split_entry_raw_data(
    *,
    blob_store: BlobStore,
    split_payload: SerializedSplitPayload,
    source_path: str,
    file_mtime: str | None,
) -> RawSessionData:
    """Persist a split payload to the blob store and return raw metadata."""
    blob_hash, blob_size = blob_store.write_from_bytes(split_payload.payload_bytes)
    from polylogue.storage.blob_publication import publication_receipt_id

    identity = split_payload.content_identity
    if identity is None:
        identity = payload_content_identity(split_payload.payload_bytes)
    return raw_data_record(
        source_path=source_path,
        file_mtime=file_mtime,
        provider_hint=split_payload.provider,
        blob_hash=blob_hash,
        blob_size=blob_size,
        source_index=split_payload.source_index,
        blob_publication_receipt_id=publication_receipt_id(blob_store, blob_hash),
        addressing_mode=split_payload.addressing_mode,
        content_identity=identity,
    )


def read_plain_source_file(context: SourceReadContext) -> RawSessionData:
    if is_sqlite_path(context.path) and context.retained_blob is None:
        with bind_source_input(context.path) as binding:
            return _read_plain_source_file(context, binding)
    return _read_plain_source_file(context, None)


def _read_plain_source_file(context: SourceReadContext, binding: SourceInputBinding | None) -> RawSessionData:
    """Stream one non-ZIP source file into the blob store.

    This is the real per-file production acquisition entry point for the
    daemon watcher and retained ingest pipeline (every non-ZIP file a
    ``WatchSource`` accepts passes through here). It emits one structured
    ``file_acquisition_decision`` log record (see
    ``polylogue.sources.live.acquisition_log``) per file, carrying the
    detected origin (or ``UNRECOGNIZED``), the specific detector evidence
    that decided it, and elapsed detect-stage time -- the per-file
    observability trail that was previously missing entirely.
    """
    from polylogue.sources.origin_specs import path_declaration_refuses_session

    # Deferred import: ``polylogue.sources.live`` (package ``__init__``) pulls
    # in ``batch.py``, which imports this module at module level -- a
    # module-level import here would be circular (same hazard documented on
    # ``dispatch._join_claude_code_sidecars``).
    from .live.acquisition_log import AcquisitionStageTimings, log_file_acquisition_decision

    stage_timings = AcquisitionStageTimings()
    sqlite_path = is_sqlite_path(context.path)
    canonical_source_path: str | None = None
    captured_profile_key: str | None = None
    captured_profile_source_path: str | None = None
    captured_file_observation: tuple[int, int, int, int, int] | None = None
    original_source_path = binding.source_path if binding is not None and binding.staged else None
    if context.retained_blob is not None and context.captured_input_identity is not None:
        receipt = context.captured_input_identity
        if receipt.semantic_source_path != str(context.path):
            raise ValueError("retained input coordinate does not match its captured identity")
        canonical_source_path = receipt.canonical_source_path
        captured_profile_key = receipt.profile_key
        captured_profile_source_path = receipt.profile_source_path
    if (
        context.retained_blob is None
        and (context.provider_hint is Provider.HERMES or original_source_path is not None)
        and sqlite_path
    ):
        heartbeat = make_status_heartbeat(
            context.status_callback,
            source_name=context.source.name,
            source_path=str(context.path),
        )
        # A declared database of another origin (Codex ``state_5.sqlite`` under
        # the broad Hermes root) is refused by declaration before snapshotting.
        refuse_declared_foreign(
            (binding.source_path if binding is not None else context.path).name, context.provider_hint
        )
        with stage_timings.stage("detect"):
            snapshot = snapshot_sqlite_to_blob(
                context.path, context.blob_store, heartbeat=heartbeat, source_binding=binding
            )
            original_source_path = snapshot.source_path
            canonical_source_path = str(snapshot.identity_path)
            captured_profile_key = snapshot.captured_profile_key
            captured_profile_source_path = str(snapshot.captured_profile_source_path)
            blob_hash, blob_size = snapshot.blob_hash, snapshot.blob_size
            publication_id = snapshot.blob_publication_receipt_id
            detected_provider = Provider.HERMES
            detection_evidence = "sqlite_snapshot.snapshot_sqlite_to_blob (Hermes sqlite state/sidecar)"
    else:
        if context.retained_blob is None:
            capture = capture_bound_path(
                context.blob_store,
                context.path,
                context.provider_hint,
                source_binding=binding,
                heartbeat=make_status_heartbeat(
                    context.status_callback,
                    source_name=context.source.name,
                    source_path=str(context.path),
                ),
            )
            blob_hash, blob_size = capture.blob_hash, capture.blob_size
            canonical_source_path = capture.canonical_source_path
            captured_profile_key = capture.captured_profile_key
            captured_profile_source_path = capture.captured_profile_source_path
            captured_file_observation = capture.file_observation
        else:
            # The path is acquisition identity only after acceptance. Every
            # byte read below comes from the retained content-addressed input,
            # which was retained unbound and so passes the boundary here.
            blob_hash, blob_size = context.retained_blob.sha256, context.retained_blob.size_bytes
            with bind_stream(context.blob_store.open(blob_hash), str(context.path), context.provider_hint) as stream:
                drain_bound(stream)
        with stage_timings.stage("detect"):
            if path_declaration_refuses_session(context.provider_hint, context.path):
                # Declared raw-only evidence (a prompt log, a sidecar) is
                # classified by location; its shape is never consulted.
                detected_provider = context.provider_hint
                detection_evidence = "declared raw-only artifact rule (location)"
            else:
                with context.blob_store.open(blob_hash) as detection_input:
                    detected_provider, detection_evidence = detect_provider_from_raw_stream_evidence(
                        detection_input,
                        context.path.name,
                        context.provider_hint,
                        truncated_tail_ok=is_jsonl_source_path(str(context.path)),
                    )
            if detected_provider is Provider.UNKNOWN and context.source.name == "browser-capture":
                detected_provider = _stream_browser_capture_provider(context.blob_store, blob_hash)
                detection_evidence = "browser_capture provider recovered from spool metadata"
        from polylogue.storage.blob_publication import publication_receipt_id

        publication_id = publication_receipt_id(context.blob_store, blob_hash)
    # The returned record is not an admission: the raw row and its
    # ``raw_payload`` receipt, committed by the caller's source-tier write,
    # are the durable accepted outcome, and a restart re-acquires exactly
    # what that commit does not hold.
    observe_acquisition(
        context.observation_callback,
        phase="source-file-streamed",
        source_path=str(context.path),
        provider_hint=detected_provider,
        blob_size=blob_size,
        blob_publication_receipt_id=publication_id,
    )
    file_mtime_epoch: float | None = None
    if context.retained_blob is None:
        try:
            file_mtime_epoch = context.path.stat().st_mtime
        except OSError:
            file_mtime_epoch = None
    log_file_acquisition_decision(
        path=context.path,
        size=blob_size,
        mtime=file_mtime_epoch,
        source_name=context.source.name,
        origin=str(detected_provider) if detected_provider is not Provider.UNKNOWN else None,
        evidence=detection_evidence,
        stage_timings=stage_timings,
    )
    if context.retained_blob is not None and detected_provider is Provider.HERMES and captured_profile_key is None:
        from polylogue.core.raw_failure_evidence import MissingProfileIdentityError

        raise MissingProfileIdentityError("retained Hermes input has no captured profile identity receipt")
    return raw_data_record(
        source_path=str(original_source_path or context.path),
        canonical_source_path=canonical_source_path,
        captured_profile_key=(captured_profile_key if declares_profile_identity(detected_provider) else None),
        captured_profile_source_path=(
            captured_profile_source_path if declares_profile_identity(detected_provider) else None
        ),
        captured_file_observation=captured_file_observation,
        file_mtime=context.file_mtime,
        provider_hint=detected_provider,
        blob_hash=blob_hash,
        blob_size=blob_size,
        blob_publication_receipt_id=publication_id,
    )


def _stream_browser_capture_provider(blob_store: BlobStore, blob_hash: str) -> Provider:
    """Read the typed envelope provider without materializing a large capture.

    Native browser captures can place a multi-megabyte ``raw_provider_payload``
    before ``session.provider``.  Prefix detection therefore cannot prove the
    provider even though the complete retained artifact can.  Parse the scalar
    event stream until both the envelope kind and nested provider are known;
    ijson keeps this bounded regardless of the native payload size.
    """
    capture_kind: str | None = None
    provider: Provider | None = None
    try:
        with blob_store.open(blob_hash) as handle:
            for prefix, event, value in ijson.parse(handle):
                if event != "string":
                    continue
                if prefix == "polylogue_capture_kind":
                    capture_kind = str(value)
                elif prefix == "session.provider":
                    provider = Provider.from_string(str(value))
                if capture_kind is not None and provider is not None:
                    break
    except ijson.JSONError:
        return Provider.UNKNOWN
    if capture_kind != "browser_llm_session" or provider is None or provider is Provider.UNKNOWN:
        return Provider.UNKNOWN
    return provider


def _stream_preserved_zip_entry(
    zf: zipfile.ZipFile,
    context: ZipEntryReadContext,
    *,
    provider_hint: Provider,
) -> RawSessionData:
    return stream_preserved_zip_entry_raw_data(
        zf,
        context,
        provider_hint=provider_hint,
    )


def stream_preserved_zip_entry_raw_data(
    zf: zipfile.ZipFile,
    context: ZipEntryReadContext,
    *,
    provider_hint: Provider,
) -> RawSessionData:
    """Durably stream one admitted ZIP member without decoding its content.

    The caller remains responsible for applying :class:`_ZipEntryValidator`
    before this function. Complete entry streaming verifies decompression and
    CRC while avoiding provider detection, JSON decoding and artifact
    classification on a source-tier-only acquisition route.
    """
    with open_bound_member(zf, context.entry, context.bound_provider) as handle:
        blob_hash, blob_size = capture_bound_stream(
            context.blob_store,
            handle,
            heartbeat=make_status_heartbeat(
                context.status_callback,
                source_name=context.source.name,
                source_path=context.source_path,
            ),
        )
    from polylogue.storage.blob_publication import publication_receipt_id

    publication_id = publication_receipt_id(context.blob_store, blob_hash)
    # Derive structural identity from the published bytes. The identity
    # streams, so this re-read holds one window, never the whole member.
    try:
        with context.blob_store.open(blob_hash) as stored_handle:
            content_identity = stream_payload_content_identity(stored_handle)
    except ContentIdentityRefusal:
        # The refused member has no raw record, so a queued publication of
        # its bytes would be reserved with nothing to reference it.
        release_refused_capture(context.blob_store, blob_hash, publication_id)
        raise
    observe_acquisition(
        context.observation_callback,
        phase="zip-entry-streamed",
        source_path=context.source_path,
        provider_hint=provider_hint,
        blob_size=blob_size,
        blob_publication_receipt_id=publication_id,
    )
    return _captured_zip_record(
        raw_data_record(
            source_path=context.source_path,
            file_mtime=context.file_mtime,
            provider_hint=provider_hint,
            blob_hash=blob_hash,
            blob_size=blob_size,
            # A preserved member addresses the document itself. Its physical
            # entry ordinal is independent of element addressing on replay.
            source_index=None,
            blob_publication_receipt_id=publication_id,
            addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
            content_identity=content_identity,
        ),
        context,
    )


def _iter_zip_entry_split_payloads(
    zf: zipfile.ZipFile,
    context: ZipEntryReadContext,
    state: _ZipEntrySplitState,
) -> Iterable[SerializedSplitPayload]:
    """Yield the split payloads produced by the production ZIP acquisition."""
    entry_provider_hint = _zip_entry_provider_hint(context.entry.filename, context.provider_hint)
    state.detected_provider = entry_provider_hint
    split_buffer = SplitPayloadBuffer()
    with open_bound_member(zf, context.entry, context.bound_provider) as handle:
        for detected in iter_entry_payloads(
            handle,
            stream_name=context.entry.filename,
            provider_hint=entry_provider_hint,
            bound_provider=context.bound_provider,
        ):
            # Never downgrade the ZIP-level hint to UNKNOWN. A GDPR export ships
            # non-conversation siblings (message_feedback.json, user.json, ...)
            # that legitimately detect as UNKNOWN on their own contents, but the
            # archive they arrived in is unambiguous. Overwriting the sniffed hint
            # with a per-payload UNKNOWN is what stamped those members
            # `unknown-export` (polylogue-hs3y).
            if detected.provider is not Provider.UNKNOWN:
                state.detected_provider = detected.provider
            if detected.provider in GROUP_PROVIDERS:
                # A grouped-provider record can occur after ordinary session
                # records in a mixed export.  Breaking here used to discard
                # the grouped record and every later record, while the caller
                # still considered the ZIP successfully scanned.  Preserve
                # this observed record as a durable raw item; the already-
                # emitted split siblings remain valid and must not be rolled
                # back.
                if split_buffer.did_split:
                    grouped = split_buffer.add_grouped(detected.provider, json_dumps_bytes(detected.payload))
                    if grouped is not None:
                        yield grouped
                    continue
                break
            classify_start = time.perf_counter()
            artifact = classify_artifact(
                detected.payload,
                provider=detected.provider,
                source_path=context.source_path,
            )
            classify_ms = (time.perf_counter() - classify_start) * 1000.0
            if not artifact.parse_as_session:
                continue
            pending_index = split_buffer.pending_index
            serialize_start = time.perf_counter()
            payload_bytes = json_dumps_bytes(detected.payload)
            serialize_ms = (time.perf_counter() - serialize_start) * 1000.0
            observe_acquisition(
                context.observation_callback,
                phase="zip-entry-split-payload-serialized",
                source_path=context.source_path,
                provider_hint=detected.provider,
                blob_size=len(payload_bytes),
                source_index=pending_index,
                artifact_kind=str(artifact.kind),
                detect_provider_ms=round(detected.detect_provider_ms, 3),
                classify_ms=round(classify_ms, 3),
                serialize_ms=round(serialize_ms, 3),
            )
            yield from split_buffer.add(detected.provider, payload_bytes)
    if split_buffer.refusals:
        # Every other element is already emitted; the refused ones are the
        # member's recorded gap.
        raise split_buffer.refusals[0]


def _whole_member_provider(context: ZipEntryReadContext) -> Provider | None:
    """The member's provider when acquisition preserves it whole without splitting."""
    from polylogue.sources.origin_specs import path_declaration_refuses_session

    entry_provider_hint = _zip_entry_provider_hint(context.entry.filename, context.provider_hint)
    if entry_provider_hint in GROUP_PROVIDERS or path_declaration_refuses_session(
        entry_provider_hint, context.entry.filename
    ):
        return entry_provider_hint
    return None


@contextlib.contextmanager
def open_replayed_zip_unit(
    zf: zipfile.ZipFile,
    context: ZipEntryReadContext,
    *,
    source_index: int | None,
) -> Iterator[IO[bytes]]:
    """Reopen the exact bytes of one unit ZIP acquisition retains.

    A whole member (``source_index`` ``None``) streams from the archive; a
    split element is re-split and yields only that element's serialized
    bytes, which the splitter already bounds. The caller verifies the bytes
    against the recorded blob identity.
    """
    if source_index is None:
        with open_bound_member(zf, context.entry, context.bound_provider) as handle:
            yield handle
        return
    state = _ZipEntrySplitState()
    for payload in _iter_zip_entry_split_payloads(zf, context, state):
        if payload.source_index == source_index:
            yield io.BytesIO(payload.payload_bytes)
            return
    raise LookupError(f"ZIP member {context.entry.filename!r} no longer yields element {source_index}")


@dataclass(frozen=True, slots=True)
class ReplayedZipRevision:
    """The identities of one payload unit ZIP acquisition retains.

    ``revision`` is the SHA-256 of the unit's bytes and ``content_identity``
    the value identity acquisition records for it. A replayed unit carries
    only these digests, never its bytes: a preserved member can be several
    gigabytes, and a verification pass keeps every unit it replayed.
    """

    source_index: int | None
    revision: str
    size_bytes: int
    addressing_mode: MemberAddressingMode
    content_identity: str


class _HashingZipEntry:
    """A ZIP entry the identity pass can seek, hashed as its frontier advances.

    The identity pass reads the member through this, so the revision is the
    SHA-256 of the same one decompression, and no member-sized scratch copy
    is made. Seeking back reopens the entry (re-decompression, not storage);
    bytes before the hashed frontier are never hashed twice.
    """

    def __init__(
        self,
        zf: zipfile.ZipFile,
        entry: zipfile.ZipInfo,
        location: Provider | None,
        checkpoint: Callable[[], None] | None,
    ) -> None:
        self._zf = zf
        self._entry = entry
        self._location = location
        self._checkpoint = checkpoint
        self._stack = contextlib.ExitStack()
        # Read through the boundary, as acquisition reads the member, so a
        # member live intake refuses is refused here too.
        self._handle = self._stack.enter_context(open_bound_member(zf, entry, location))
        self._position = 0
        self.digest = hashlib.sha256()
        #: Bytes hashed so far: the member's size once drained.
        self.hashed = 0

    def read(self, size: int = -1) -> bytes:
        data = self._handle.read(size if size >= 0 else _REVISION_CHUNK_BYTES)
        start = self._position
        self._position += len(data)
        if self._position > self.hashed:
            self.digest.update(data[self.hashed - start :])
            self.hashed = self._position
        return data

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = 0) -> int:
        target = offset if whence == 0 else self._position + offset
        if whence not in (0, 1) or target < 0:
            raise ValueError("a ZIP entry reader seeks only to a known position")
        if target < self._position:
            self._stack.close()
            self._stack = contextlib.ExitStack()
            self._handle = self._stack.enter_context(open_bound_member(self._zf, self._entry, self._location))
            self._position = 0
        while self._position < target:
            if self._checkpoint is not None:
                self._checkpoint()
            if not self.read(min(_REVISION_CHUNK_BYTES, target - self._position)):
                break
        return self._position

    def drain(self) -> None:
        """Hash whatever the identity pass did not read."""
        self.seek(self.hashed)
        while True:
            if self._checkpoint is not None:
                self._checkpoint()
            if not self.read(_REVISION_CHUNK_BYTES):
                return

    def close(self) -> None:
        self._stack.close()


def _stream_member_revision(
    zf: zipfile.ZipFile,
    entry: zipfile.ZipInfo,
    location: Provider | None,
    checkpoint: Callable[[], None] | None,
) -> ReplayedZipRevision:
    # Acquisition refuses a member whose content identity cannot be stored,
    # so the replay does too: the identity streams from the entry itself,
    # which is hashed for the revision as it is read.
    reader = _HashingZipEntry(zf, entry, location, checkpoint)
    try:
        identity = stream_payload_content_identity(cast(IO[bytes], reader), checkpoint=checkpoint)
        reader.drain()
    finally:
        reader.close()
    return ReplayedZipRevision(
        None, reader.digest.hexdigest(), reader.hashed, MemberAddressingMode.WHOLE_MEMBER, identity
    )


def replay_zip_entry_acquisition_revisions(
    zf: zipfile.ZipFile,
    context: ZipEntryReadContext,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> Iterable[ReplayedZipRevision]:
    """Replay the identities of the units ZIP acquisition would retain.

    A member is one preserved unit unless the production splitter recognizes
    at least two session payloads; then each split carries the splitter's
    source index. A whole member is hashed in chunks, as acquisition streams
    it, instead of being read into memory. A refused split element is raised
    after its validated siblings were yielded, as acquisition records it.
    """
    if _whole_member_provider(context) is not None:
        yield _stream_member_revision(zf, context.entry, context.bound_provider, checkpoint)
        return
    state = _ZipEntrySplitState()
    for payload in _iter_zip_entry_split_payloads(zf, context, state):
        state.did_split = True
        identity = payload.content_identity
        if identity is None:
            identity = payload_content_identity(payload.payload_bytes)
        yield ReplayedZipRevision(
            payload.source_index,
            hashlib.sha256(payload.payload_bytes).hexdigest(),
            len(payload.payload_bytes),
            payload.addressing_mode,
            identity,
        )
    if not state.did_split:
        yield _stream_member_revision(zf, context.entry, context.bound_provider, checkpoint)


def _zip_entry_detected_provider(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> Provider | None:
    """Classify an exact entry after complete JSON and CRC validation."""
    if not info.filename.lower().endswith(ZIP_JSON_SUFFIXES) or not info.file_size:
        return None
    try:
        with _decoders.open_zip_entry(zf, info) as handle:
            try:
                provider, _evidence = detect_provider_from_stream_evidence(handle)
            except (ijson.JSONError, UnicodeError, ValueError):
                provider = None
            # Negative syntax outcomes also need a complete CRC read before
            # the acquisition owner can reuse their inspected member result.
            while handle.read(_REVISION_CHUNK_BYTES):
                pass
            return provider
    except (ijson.JSONError, UnicodeError, ValueError):
        # A malformed JSON member has no provider proof. Its acquisition
        # still owns the visible decode disposition; storage/CRC faults are
        # not silently converted into a healthy negative shape observation.
        return None


def sniff_zip_provider(zf: zipfile.ZipFile, entries: Iterable[zipfile.ZipInfo]) -> Provider | None:
    """Find the strict heaviest complete JSON-member claim, preserving ties."""
    weights: dict[Provider, int] = {}
    for info in entries:
        detected = _zip_entry_detected_provider(zf, info)
        if detected is not None and detected is not Provider.UNKNOWN:
            weights[detected] = weights.get(detected, 0) + max(1, int(info.file_size))
    return _dominant_zip_provider(weights)


def _dominant_zip_provider(weights: dict[Provider, int]) -> Provider | None:
    if not weights:
        return None
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0].value))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


@dataclass(frozen=True, slots=True)
class ZipMemberAdmission:
    """Admission and complete detector outcomes for one captured container.

    The private indexed spool belongs to this acquisition context. Its key is
    the exact container SHA, declared source coordinate, enumeration closure
    and central-directory ordinal. It never certifies a mutable source path.
    """

    provider_hint: Provider
    allowed_path: Callable[[str], bool] | None
    bound_provider: Provider | None
    capture_key: tuple[str, str, str]
    entries: list[zipfile.ZipInfo]
    detected_members: PickleSpool[Provider | None] | None

    def entry_provider_hint(self, info: zipfile.ZipInfo, *, entry_ordinal: int) -> Provider:
        if entry_ordinal < 0 or entry_ordinal >= len(self.entries) or self.entries[entry_ordinal] is not info:
            raise ValueError("ZIP member differs from its captured central-directory ordinal")
        if self.bound_provider is not None:
            return self.bound_provider
        # A container winner never erases a known minority provider. Declared
        # non-session assets keep their own owner without a JSON shape claim.
        declared = declared_artifact_provider(info.filename)
        if not provider_detection_path(info.filename):
            return declared or self.provider_hint
        if info.filename.lower().endswith(ZIP_JSON_SUFFIXES):
            if self.detected_members is None:
                raise ValueError("unbound ZIP admission lacks its completed member inspection")
            detected = next(self.detected_members.iter_from(entry_ordinal))
            return detected or self.provider_hint
        return declared or self.provider_hint


@contextlib.contextmanager
def zip_member_admission(
    zf: zipfile.ZipFile,
    zip_path: Path,
    central_directory: list[zipfile.ZipInfo],
    fallback_provider: Provider,
    *,
    container_blob_hash: str,
) -> Iterator[ZipMemberAdmission]:
    """Inspect once, then reuse exact member outcomes until acquisition exits."""
    from polylogue.sources.dispatch import bound_location_provider

    bound = bound_location_provider(fallback_provider)
    capture_key = (container_blob_hash, str(zip_path), zip_acquisition_fingerprint(fallback_provider))
    if bound is not None:
        yield ZipMemberAdmission(fallback_provider, None, bound, capture_key, central_directory, None)
        return
    detected_members: PickleSpool[Provider | None] = PickleSpool(indexed=True)
    try:
        weights: dict[Provider, int] = {}
        validator = ZipAdmission(zip_path=zip_path)
        for info in central_directory:
            safe = next(iter(validator.filter_entries((info,), allowed_suffixes=ZIP_JSON_SUFFIXES)), None) is not None
            detected = (
                _zip_entry_detected_provider(zf, info) if safe and provider_detection_path(info.filename) else None
            )
            detected_members.append(detected)
            if detected is not None and detected is not Provider.UNKNOWN:
                weights[detected] = weights.get(detected, 0) + max(1, int(info.file_size))
        yield ZipMemberAdmission(
            _dominant_zip_provider(weights) or Provider.UNKNOWN,
            is_declared_artifact_path,
            None,
            capture_key,
            central_directory,
            detected_members,
        )
    finally:
        detected_members.close()


def zip_acquisition_fingerprint(provider: Provider, *, preserved_only: bool = False) -> str:
    """Bind the actual decoder closure and its declared origin admission input."""
    from .dispatch import bound_location_provider
    from .origin_specs import retained_enumeration_fingerprint

    location_provider = bound_location_provider(provider)
    return hashlib.sha256(
        json.dumps(
            (
                retained_enumeration_fingerprint(),
                location_provider.value if location_provider is not None else None,
                preserved_only,
            ),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _captured_zip_record(data: RawSessionData, context: ZipEntryReadContext) -> RawSessionData:
    """Carry exact member evidence from the accepted physical container."""
    receipt = context.captured_input_identity
    if data.captured_zip_coordinate is not None:
        return data
    if receipt is None:
        return data
    if data.addressing_mode is None:
        raise ValueError("captured ZIP member lacks its addressing mode")
    coordinate = captured_zip_member_coordinate(
        receipt,
        entry_name=context.entry.filename,
        entry_ordinal=context.entry_ordinal,
        split_index=data.source_index or 0,
        addressing_mode=data.addressing_mode,
        container_blob_hash=context.container_blob_hash,
        decoder_fingerprint=context.decoder_fingerprint,
    )
    if coordinate is None:
        return data
    profile = receipt.member_profile_identity(context.entry.filename)
    namespace, profile_path = profile if profile is not None else (None, None)
    from polylogue.core.provider_identity import captured_hermes_profile_key

    return data.model_copy(
        update={
            "source_path": coordinate.declared_member,
            "source_index": coordinate.source_index,
            "canonical_source_path": coordinate.canonical_member,
            "captured_zip_coordinate": coordinate,
            "captured_profile_key": None if namespace is None else captured_hermes_profile_key(namespace),
            "captured_profile_source_path": None if profile_path is None else str(profile_path),
        }
    )


def captured_zip_member_coordinate(
    captured_identity: CapturedSourceInputIdentity | None,
    *,
    entry_name: str,
    entry_ordinal: int | None,
    split_index: int,
    addressing_mode: MemberAddressingMode,
    container_blob_hash: str | None,
    decoder_fingerprint: str | None,
) -> CapturedZipMemberCoordinate | None:
    """Build a member coordinate only from the accepted typed Source witness."""
    if captured_identity is None:
        return None
    if entry_ordinal is None:
        raise ValueError("captured ZIP member lacks its central-directory ordinal")
    if container_blob_hash is None or decoder_fingerprint is None:
        raise ValueError("captured ZIP member lacks its accepted container or decoder identity")
    profile = captured_identity.member_profile_identity(entry_name)
    namespace = None if profile is None else profile[0]
    return CapturedZipMemberCoordinate(
        captured_identity.canonical_source_path,
        captured_identity.semantic_source_path,
        entry_name,
        entry_ordinal,
        split_index,
        addressing_mode,
        container_blob_hash,
        decoder_fingerprint,
        None if namespace is None else str(namespace),
    )


def iter_zip_entry_raw_data(
    zf: zipfile.ZipFile,
    context: ZipEntryReadContext,
) -> Iterable[RawSessionData]:
    """Decode one member and preserve its accepted container receipt."""
    for data in _iter_zip_entry_raw_data(zf, context):
        yield _captured_zip_record(data, context)


def _iter_zip_entry_raw_data(
    zf: zipfile.ZipFile,
    context: ZipEntryReadContext,
) -> Iterable[RawSessionData]:
    """Yield raw records for one ZIP entry, splitting multi-session payloads."""
    # A ``raw-only`` member is evidence, not a session document: preserve its
    # exact bytes rather than decoding it as a JSON payload to split. Export
    # assets are arbitrary binary (polylogue-ximhz), so the split route's
    # UTF-8 decode would fail the whole archive read, not just the member.
    whole_member_provider = _whole_member_provider(context)
    if whole_member_provider is not None:
        yield _stream_preserved_zip_entry(zf, context, provider_hint=whole_member_provider)
        return

    state = _ZipEntrySplitState()
    # Nothing leaves this admission unit before complete syntax/CRC validation.
    # Metadata is spooled privately; failed/cancelled acquisition releases every
    # publication prefix without retaining all split records in memory.
    identity_refusal: ContentIdentityRefusal | None = None
    with tempfile.TemporaryFile(mode="w+b", prefix="polylogue-zip-splits-") as splits:
        try:
            try:
                for split_payload in _iter_zip_entry_split_payloads(zf, context, state):
                    state.did_split = True
                    split = make_split_entry_raw_data(
                        blob_store=context.blob_store,
                        split_payload=split_payload,
                        source_path=context.source_path,
                        file_mtime=context.file_mtime,
                    )
                    position = splits.tell()
                    try:
                        pickle.dump(split, splits, protocol=pickle.HIGHEST_PROTOCOL)
                    except BaseException:
                        splits.seek(position)
                        splits.truncate()
                        if split.blob_hash is not None:
                            release_refused_capture(
                                context.blob_store, split.blob_hash, split.blob_publication_receipt_id
                            )
                        raise
            except ContentIdentityRefusal as exc:
                # Identity refusal follows complete member validation. Keep
                # admissible siblings and report the refused element afterward.
                identity_refusal = exc
        except BaseException:
            splits.seek(0)
            while True:
                try:
                    split = pickle.load(splits)
                except EOFError:
                    break
                if split.blob_hash is not None:
                    release_refused_capture(context.blob_store, split.blob_hash, split.blob_publication_receipt_id)
            raise
        splits.seek(0)
        while True:
            try:
                split = pickle.load(splits)
            except EOFError:
                break
            yield split
    if identity_refusal is not None:
        raise identity_refusal

    if state.did_split:
        return

    # Preserve original ZIP entry bytes when the entry is metadata or a
    # single session document.
    yield _stream_preserved_zip_entry(zf, context, provider_hint=state.detected_provider)


__all__ = [
    "AcquisitionObservation",
    "ArtifactIdentity",
    "CursorState",
    "DetectedEntryPayload",
    "ObservationCallback",
    "SerializedSplitPayload",
    "SourceReadContext",
    "SplitPayloadBuffer",
    "StatusCallback",
    "ZipEntryReadContext",
    "ZipMemberAdmission",
    "iter_entry_payloads",
    "open_replayed_zip_unit",
    "replay_zip_entry_acquisition_revisions",
    "ReplayedZipRevision",
    "iter_zip_entry_raw_data",
    "sniff_zip_provider",
    "make_status_heartbeat",
    "observe_acquisition",
    "captured_zip_member_coordinate",
    "raw_data_record",
    "read_plain_source_file",
    "stream_preserved_zip_entry_raw_data",
    "zip_member_admission",
]
