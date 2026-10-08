"""Browser-capture envelope facts folded from retained bytes.

A capture is admitted from a staged file of any size. One streamed pass
validates the envelope and folds every fact spool admission decides from --
identity, turn count, deduplication fingerprint, attachment identities and
carriers, accepted message identities, native-payload shape -- so memory is
proportional to individual turns or attachments in the ordinary fold.
The ordinary fold does not materialize ``raw_provider_payload`` (a provider
transcript as large as the capture itself) or open-ended ``provider_meta``:
each contributes a structural digest (plus, for the raw payload, the shape of
its root fields). Native state-only turns additionally need canonical parser
evidence. Preparation can supply its existing scratch-backed witness; an
independent replay derives it from the original native model, whose token,
record and session allocations remain an explicit memory limitation.
"""

from __future__ import annotations

import binascii
import errno
import fcntl
import hashlib
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, BinaryIO, Literal

import ijson
from pydantic import ValidationError

from polylogue.browser_capture.capture_decode import (
    _build,
    _Event,
    _json_events,
    _skip,
    iter_carrier_bytes,
)
from polylogue.browser_capture.models import (
    BrowserCaptureAttachment,
    BrowserCaptureEnvelope,
    BrowserCaptureIdentityObservation,
    BrowserCaptureProvenance,
    BrowserCaptureSession,
    BrowserCaptureTurn,
    _CanonicalNativeTurnWitness,
    envelope_has_native_provider_payload,
)
from polylogue.core.enums import Provider
from polylogue.core.json import dumps_bytes

#: Bytes read from the request body per call while staging a capture. A pacing
#: bound on one read, not a bound on the capture.
CAPTURE_READ_CHUNK_BYTES = 1024 * 1024

#: Spool subdirectory holding captures being received. Same filesystem as the
#: published artifacts, so publication is one ``os.replace``.
STAGING_DIRNAME = ".staging"

# Backfill scheduling identifies the observer that acquired a snapshot rather
# than a semantic property of the provider session.  Keep it out of the spool
# deduplication fingerprint, while retaining any future semantic backfill
# metadata a provider might add.
_BACKFILL_OBSERVER_ATTRIBUTION_KEYS = frozenset({"job_id", "queue_id", "instance_id"})

# These fields describe which retained raw revision produced a prepared asset
# and what the receiver acquired for it. They remain in the serialized
# envelope and its content fingerprint; they are not intrinsic attachment
# identity when two snapshots of the same provider session are compared.
_ATTACHMENT_RAW_COORDINATE_KEYS = frozenset({"native_attachment_ordinal", "native_turn_ordinal", "native_raw_position"})

_DEDUP_DOMAIN = b"polylogue-browser-capture-dedup/v2\x00"

_ENVELOPE_FIELDS = frozenset(BrowserCaptureEnvelope.model_fields) - {
    "session",
    "raw_provider_payload",
    "provider_meta",
    "provenance",
}
_SESSION_FIELDS = frozenset(BrowserCaptureSession.model_fields) - {"turns", "attachments", "provider_meta"}
_PROVENANCE_FIELDS = frozenset(BrowserCaptureProvenance.model_fields) - {"provider_meta"}


class CaptureEnvelopeError(ValueError):
    """A staged capture is not an admissible envelope.

    ``reason`` is ``invalid_json`` when the bytes are not one JSON document and
    ``invalid_payload`` when the document is not a valid capture envelope.
    """

    def __init__(self, reason: Literal["invalid_json", "invalid_payload"], detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


class CaptureBodyIncompleteError(ValueError):
    """The request ended before its declared ``Content-Length``."""


class _NativeWitnessRequiredError(CaptureEnvelopeError):
    def __init__(self) -> None:
        super().__init__("invalid_payload", "state-only turn requires canonical native content")


class SpoolStorageExhaustedError(RuntimeError):
    """The spool filesystem cannot hold an incoming capture.

    Raised before any body byte is written: the declared length is reserved on
    disk first, so the only refusal is the physical one, and it is retryable.
    """

    def __init__(self, requested_bytes: int, available_bytes: int | None) -> None:
        super().__init__(f"spool storage cannot hold {requested_bytes} bytes (available: {available_bytes})")
        self.requested_bytes = requested_bytes
        self.available_bytes = available_bytes


class StagedCapture:
    """A received capture body on disk, with the digest of its exact bytes.

    The staging file stays ``flock``-ed until :meth:`discard`, so a receiver
    starting beside this one (:func:`reap_stale_staging`) can tell a live
    upload from one a crashed process abandoned.
    """

    __slots__ = ("_lock_fd", "path", "sha256", "size_bytes")

    def __init__(self, path: Path, size_bytes: int, sha256: str, lock_fd: int | None = None) -> None:
        self.path = path
        self.size_bytes = size_bytes
        self.sha256 = sha256
        self._lock_fd = lock_fd

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None


def stage_retained_capture(
    handle: BinaryIO, artifact_path: Path, *, size_bytes: int, sha256: str, spool_root: Path
) -> StagedCapture:
    """Borrow an immutable rooted artifact for ordinary spool publication.

    Admission moves only this staging link. The registry's original artifact
    and its custody roots survive publication, cancellation and response loss.
    The shared inode lock excludes stale-stage and unrooted-artifact reaping.
    """
    directory = spool_root / STAGING_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
    placeholder_fd, name = tempfile.mkstemp(prefix=_STAGING_PREFIX, suffix=_STAGING_SUFFIX, dir=directory)
    path = Path(name)
    lock_fd = None
    try:
        fcntl.flock(placeholder_fd, fcntl.LOCK_EX)
        path.unlink()
        os.link(artifact_path, path)
        lock_fd = os.open(path, os.O_RDONLY)
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        expected = os.fstat(handle.fileno())
        actual = os.fstat(lock_fd)
        if (actual.st_dev, actual.st_ino, actual.st_size) != (expected.st_dev, expected.st_ino, size_bytes):
            raise CaptureEnvelopeError("invalid_payload", "retained capture artifact changed")
        with os.fdopen(os.dup(lock_fd), "rb") as linked:
            if hashlib.file_digest(linked, "sha256").hexdigest() != sha256:
                raise CaptureEnvelopeError("invalid_payload", "retained capture artifact digest mismatch")
        return StagedCapture(path, size_bytes, sha256, lock_fd)
    except BaseException:
        path.unlink(missing_ok=True)
        if lock_fd is not None:
            os.close(lock_fd)
        raise
    finally:
        os.close(placeholder_fd)


def _available_bytes(directory: Path) -> int:
    stats = os.statvfs(directory)
    return stats.f_bavail * stats.f_frsize


def _reserve(fd: int, directory: Path, length: int) -> None:
    """Reserve ``length`` bytes for the staging file before writing any.

    ``posix_fallocate`` allocates the blocks, so concurrent uploads -- in this
    process or another sharing the spool -- cannot both be admitted into the
    same free space. Where the filesystem cannot allocate, free space is
    compared instead. A length no file offset can represent, or one past the
    filesystem's largest file, is the same physical refusal.
    """
    if length <= 0:
        return
    fallocate = getattr(os, "posix_fallocate", None)
    if fallocate is not None:
        try:
            fallocate(fd, 0, length)
            return
        except OverflowError as exc:
            raise SpoolStorageExhaustedError(length, _available_bytes(directory)) from exc
        except OSError as exc:
            if is_storage_exhausted(exc):
                raise SpoolStorageExhaustedError(length, _available_bytes(directory)) from exc
            if exc.errno not in {errno.EOPNOTSUPP, errno.EINVAL, errno.ENOSYS}:
                raise
    available = _available_bytes(directory)
    if length > available:
        raise SpoolStorageExhaustedError(length, available)


def _locked_staging_file(staging: Path) -> tuple[int, Path]:
    """Create and lock a staging file that no concurrent reaper has removed."""
    while True:
        fd, name = tempfile.mkstemp(dir=staging, prefix=_STAGING_PREFIX, suffix=_STAGING_SUFFIX)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            if os.stat(name).st_ino == os.fstat(fd).st_ino:
                return fd, Path(name)
        except FileNotFoundError:
            pass
        os.close(fd)


_STAGING_PREFIX = ".capture-"
_STAGING_SUFFIX = ".tmp"


def stage_capture_body(read: Callable[[int], bytes], length: int, *, spool_root: Path) -> StagedCapture:
    """Copy exactly ``length`` body bytes into a staging file in the spool.

    The declared length is reserved on disk before the first read, so a body
    the filesystem cannot hold is refused with
    :class:`SpoolStorageExhaustedError` without consuming space. Reads at
    most :data:`CAPTURE_READ_CHUNK_BYTES` per call and hashes while writing,
    so no body is held in memory. The staged file is fsynced and stays locked
    until the caller publishes it by ``os.replace`` or discards it.
    """

    def chunks() -> Iterator[bytes]:
        remaining = length
        while remaining > 0:
            chunk = read(min(CAPTURE_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise CaptureBodyIncompleteError(f"request body ended {remaining} bytes before its declared length")
            remaining -= len(chunk)
            yield chunk

    return stage_capture_chunks(chunks(), spool_root=spool_root, durable=True, reserved_length=length)


def stage_capture_chunks(
    chunks: Iterator[bytes], *, spool_root: Path, durable: bool = True, reserved_length: int | None = None
) -> StagedCapture:
    """Seal generated artifact bytes through the same locked staging owner.

    A received File reserves its known length before reading. A generated
    prefix/final artifact has no declared length: actual filesystem exhaustion
    remains a typed refusal, and no arbitrary body limit is substituted.
    Publication bytes are fsynced before their artifact owner adopts them.
    Transient JSON cells and responses are flushed for their immediate reader;
    their durable authority is the committed registry, not this scratch file.
    The caller's chunk/token allocation remains its own memory contract.
    """
    staging = spool_root / STAGING_DIRNAME
    staging.mkdir(parents=True, exist_ok=True)
    fd, path = _locked_staging_file(staging)
    digest = hashlib.sha256()
    size = 0
    try:
        if reserved_length is not None:
            _reserve(fd, staging, reserved_length)
        with os.fdopen(os.dup(fd), "wb") as handle:
            for chunk in chunks:
                size += len(chunk)
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
    except BaseException as exc:
        path.unlink(missing_ok=True)
        os.close(fd)
        if isinstance(exc, OSError) and is_storage_exhausted(exc):
            raise SpoolStorageExhaustedError(size, None) from exc
        raise
    return StagedCapture(path=path, size_bytes=size, sha256=digest.hexdigest(), lock_fd=fd)


def reap_stale_staging(spool_root: Path) -> int:
    """Remove staging files no live upload holds; return how many.

    A receiver that died mid-upload leaves its staging file behind, and it is
    invisible to the spool quota. Every live upload holds its file's lock, so
    a file whose lock can be taken belongs to no one.
    """
    staging = spool_root / STAGING_DIRNAME
    if not staging.is_dir():
        return 0
    reaped = 0
    for path in staging.glob(f"{_STAGING_PREFIX}*{_STAGING_SUFFIX}"):
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            path.unlink(missing_ok=True)
            reaped += 1
        finally:
            os.close(fd)
    return reaped


def is_storage_exhausted(exc: OSError) -> bool:
    """Whether a staging failure is the spool's physical limit, not a fault.

    Space or quota exhaustion, or a body past the filesystem's largest file.
    """
    return exc.errno in {errno.ENOSPC, errno.EFBIG, getattr(errno, "EDQUOT", errno.ENOSPC)}


@dataclass(frozen=True, slots=True)
class AttachmentFact:
    """What convergence compares about one attachment, without its bytes.

    ``identity`` digests the stable observed provider descriptor;
    ``size_bytes`` is compatible evidence (unknown may be enriched);
    ``carrier`` digests the decoded ``content_base64`` bytes; ``inline_carrier``
    and ``data_carrier`` digest the corresponding fallback fields. A digest is
    ``None`` when that carrier is absent; its validity flag is false when the
    field is present but malformed.
    ``scope`` (``session`` or ``turn:<provider_turn_id>``) and ``attachment_id``
    locate it: attachment IDs need not be unique, and a turn inserted before
    another must not pair its attachments with the other turn's.
    """

    identity: bytes
    size_bytes: int | None
    carrier: bytes | None
    carrier_valid: bool
    scope: str
    attachment_id: str
    message_provider_id: str | None
    #: Digest of the ``inline_base64``/``data`` bytes an attachment already
    #: carries (``None`` when neither is present; an empty string is present).
    inline_carrier: bytes | None = None
    inline_valid: bool = True
    data_carrier: bytes | None = None
    data_valid: bool = True

    @property
    def effective_carrier(self) -> tuple[bytes | None, bool]:
        """The preferred present carrier and its validity."""
        for digest, valid in (
            (self.carrier, self.carrier_valid),
            (self.inline_carrier, self.inline_valid),
            (self.data_carrier, self.data_valid),
        ):
            if digest is not None:
                return digest, valid
        return None, True

    @property
    def carrier_evidence(self) -> tuple[tuple[bytes, bool], ...]:
        """Every present encoded byte carrier, preserving malformed evidence."""
        return tuple(
            (digest, valid)
            for digest, valid in (
                (self.carrier, self.carrier_valid),
                (self.inline_carrier, self.inline_valid),
                (self.data_carrier, self.data_valid),
            )
            if digest is not None
        )

    @property
    def has_invalid_or_conflicting_carrier(self) -> bool:
        evidence = self.carrier_evidence
        return any(not valid for _, valid in evidence) or len({digest for digest, _ in evidence}) > 1


@dataclass(frozen=True, slots=True)
class CaptureSummary:
    """Every fact spool admission reads from one capture envelope.

    ``head`` is the validated envelope with no turns, attachments, raw
    provider payload or ``provider_meta``: every turn is validated as it
    streams past and then folded into the other fields, so admission holds
    no turn content while it summarizes a second (resident) capture.
    ``provenance_meta_digest`` digests ``provenance.provider_meta``; the
    envelope and session metadata are folded into both fingerprints.

    ``dedup_content_hash`` fingerprints capture content independently of
    observation-specific provenance: an extension instance and its capture
    time identify *who saw* a snapshot, not a different session revision, so
    two concurrently running instances converge on one spool artifact while
    each poster's attribution is still echoed. ``carrierless_fingerprint``
    keeps the ordered stable attachment descriptors while omitting byte
    carriers, acquisition outcomes and raw-revision attachment coordinates.
    """

    head: BrowserCaptureEnvelope
    turn_count: int
    dedup_content_hash: str
    carrierless_fingerprint: bytes
    attachments: tuple[AttachmentFact, ...]
    turn_identities: tuple[tuple[str, Literal["native", "unknown"]], ...]
    has_native_provider_payload: bool
    provenance_meta_digest: bytes

    @property
    def provider(self) -> Provider:
        return self.head.session.provider

    @property
    def provider_session_id(self) -> str:
        return self.head.session.provider_session_id

    @property
    def capture_id(self) -> str | None:
        return self.head.capture_id


def carrier_digest(value: str) -> bytes | None:
    """SHA-256 of a carrier's decoded bytes; ``None`` when malformed."""
    digest = hashlib.sha256()
    try:
        for chunk in iter_carrier_bytes(value):
            digest.update(chunk)
    except (ValueError, binascii.Error):
        return None
    return digest.digest()


def _attachment_fact(attachment: BrowserCaptureAttachment, *, scope: str) -> AttachmentFact:
    identity = dumps_bytes(_attachment_comparison_payload(attachment), sort_keys=True)

    def digest(value: str | None) -> tuple[bytes | None, bool]:
        if value is None:
            return None, True
        decoded = carrier_digest(value)
        return (decoded if decoded is not None else hashlib.sha256(b"").digest()), decoded is not None

    carrier, valid = digest(attachment.content_base64)
    inline_carrier, inline_valid = digest(attachment.inline_base64)
    data_carrier, data_valid = digest(attachment.data)
    return AttachmentFact(
        identity=hashlib.sha256(identity).digest(),
        size_bytes=attachment.size_bytes,
        carrier=carrier,
        carrier_valid=valid,
        scope=scope,
        attachment_id=attachment.provider_attachment_id,
        message_provider_id=attachment.message_provider_id,
        inline_carrier=inline_carrier,
        inline_valid=inline_valid,
        data_carrier=data_carrier,
        data_valid=data_valid,
    )


def _attachment_comparison_payload(attachment: BrowserCaptureAttachment) -> dict[str, object]:
    """Return stable observed descriptors, separate from size and acquired bytes.

    Size is retained as compatible evidence: absence may be enriched, while
    two known unequal sizes conflict. Inline and content carriers are compared
    by decoded bytes in ``AttachmentFact``. The explicit acquisition and
    raw-coordinate fields stay in the envelope. They do not change identity
    across retained revisions when a nonempty provider message owner is
    present; id-less attachments retain those ordinal constraints.
    """
    stable_owner = bool(attachment.message_provider_id)
    provider_meta = {
        key: value
        for key, value in attachment.provider_meta.items()
        if key not in {"asset_acquisition", "content_sha256"}
        and (not stable_owner or key not in _ATTACHMENT_RAW_COORDINATE_KEYS)
    }
    return {
        "provider_attachment_id": attachment.provider_attachment_id,
        "message_provider_id": attachment.message_provider_id,
        "attachment_kind": attachment.attachment_kind,
        "name": attachment.name,
        "mime_type": attachment.mime_type,
        "url": attachment.url,
        "extracted_content": attachment.extracted_content,
        "provider_meta": provider_meta,
    }


def _attachment_comparison_dump(attachments: list[BrowserCaptureAttachment]) -> list[dict[str, object]]:
    """Keep attachment occurrences and order while folding stable descriptors."""
    return [_attachment_comparison_payload(attachment) for attachment in attachments]


def _framed(digest: hashlib._Hash, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


@dataclass
class _ItemFold:
    """Ordered fold of one streamed list (turns or session attachments)."""

    count: int = 0
    digest: hashlib._Hash = field(default_factory=hashlib.sha256)
    carrierless: hashlib._Hash = field(default_factory=hashlib.sha256)
    attachments: list[AttachmentFact] = field(default_factory=list)
    identities: list[tuple[str, BrowserCaptureIdentityObservation]] = field(default_factory=list)
    native_witness: _CanonicalNativeTurnWitness | None = None

    def add_turn(self, turn: BrowserCaptureTurn) -> None:
        dump = turn.model_dump(mode="json", exclude_none=True)
        _framed(self.digest, dumps_bytes(dump, sort_keys=True))
        dump["attachments"] = _attachment_comparison_dump(turn.attachments)
        _framed(self.carrierless, dumps_bytes(dump, sort_keys=True))
        scope = f"turn:{turn.provider_turn_id}"
        self.attachments.extend(_attachment_fact(attachment, scope=scope) for attachment in turn.attachments)
        if turn.provider_turn_id and turn.identity_observation is not None:
            self.identities.append((turn.provider_turn_id, turn.identity_observation))
        self.count += 1

    def add_raw_turn(self, item: object) -> None:
        if self.native_witness is None and isinstance(item, Mapping):
            text = item.get("text")
            if (
                (text is None or isinstance(text, str) and not text.strip())
                and not item.get("blocks")
                and not item.get("attachments")
            ):
                raise _NativeWitnessRequiredError()
        self.add_turn(BrowserCaptureTurn.model_validate(item, context={"canonical_native_turns": self.native_witness}))

    def add_raw_attachment(self, item: object) -> None:
        self.add_attachment(BrowserCaptureAttachment.model_validate(item))

    def add_attachment(self, attachment: BrowserCaptureAttachment) -> None:
        dump = attachment.model_dump(mode="json", exclude_none=True)
        _framed(self.digest, dumps_bytes(dump, sort_keys=True))
        _framed(self.carrierless, dumps_bytes(_attachment_comparison_payload(attachment), sort_keys=True))
        self.attachments.append(_attachment_fact(attachment, scope="session"))
        self.count += 1


@dataclass
class _SessionFold:
    head: dict[str, object] = field(default_factory=dict)
    meta_digest: bytes = b""
    turns: _ItemFold = field(default_factory=_ItemFold)
    attachments: _ItemFold = field(default_factory=_ItemFold)


@dataclass
class _RawFold:
    """``raw_provider_payload`` as a structural digest plus its root shape."""

    digest: bytes = b"none"
    shape: dict[str, object] | list[dict[str, object]] | None = None


class _MapFrame:
    __slots__ = ("key", "members")

    def __init__(self) -> None:
        self.members: dict[str, bytes] = {}
        self.key = ""


def _object_digest(members: dict[str, bytes]) -> bytes:
    digest = hashlib.sha256(b"o")
    for key in sorted(members):
        _framed(digest, key.encode("utf-8", "surrogatepass"))
        digest.update(members[key])
    return digest.digest()


def _scalar_digest(value: object) -> bytes:
    """Digest one scalar's canonical encoding.

    A string is hashed as its JSON encoding piecewise (quote, escaped body,
    quote), so a large string is not copied into one more encoded buffer.
    """
    if not isinstance(value, str):
        return hashlib.sha256(b"s" + dumps_bytes(value)).digest()
    digest = hashlib.sha256(b"s")
    encoded = dumps_bytes(value[:0])
    digest.update(encoded[:1])
    for start in range(0, len(value), _SCALAR_DIGEST_CHUNK_CHARS):
        digest.update(dumps_bytes(value[start : start + _SCALAR_DIGEST_CHUNK_CHARS])[1:-1])
    digest.update(encoded[-1:])
    return digest.digest()


_SCALAR_DIGEST_CHUNK_CHARS = 1 << 20


def _structural_digest(events: Iterator[_Event], event: str, value: object) -> bytes:
    """Digest one JSON value without materializing it.

    Objects hash their members sorted by key (last duplicate wins, as the
    decoder does), arrays hash their items in order, scalars hash their
    canonical encoding. Memory is bounded by nesting depth and member count,
    never by string or array size.
    """
    stack: list[_MapFrame | hashlib._Hash] = []
    while True:
        result: bytes | None = None
        if event == "start_map":
            stack.append(_MapFrame())
        elif event == "start_array":
            stack.append(hashlib.sha256(b"a"))
        elif event == "map_key":
            frame = stack[-1]
            assert isinstance(frame, _MapFrame)
            frame.key = str(value)
        elif event == "end_map":
            frame = stack.pop()
            assert isinstance(frame, _MapFrame)
            result = _object_digest(frame.members)
        elif event == "end_array":
            frame = stack.pop()
            assert not isinstance(frame, _MapFrame)
            result = frame.digest()
        else:
            result = _scalar_digest(value)
        if result is not None:
            if not stack:
                return result
            parent = stack[-1]
            if isinstance(parent, _MapFrame):
                parent.members[parent.key] = result
            else:
                parent.update(result)
        event, value = next(events)


def _read_raw_payload(events: Iterator[_Event], event: str, value: object) -> _RawFold:
    if event == "null":
        return _RawFold()
    if event == "start_array":
        digest = hashlib.sha256(b"a")
        populated = False
        while True:
            event, value = next(events)
            if event == "end_array":
                break
            if event != "start_map":
                raise CaptureEnvelopeError("invalid_payload", "native record payload must contain JSON objects")
            populated = True
            digest.update(_structural_digest(events, event, value))
        if not populated:
            raise CaptureEnvelopeError("invalid_payload", "native record payload must not be empty")
        # Admission needs the root shape, not another copy of the transcript.
        return _RawFold(digest=digest.digest(), shape=[{}])
    if event != "start_map":
        # The envelope coerces a non-object payload to ``{}``.
        _skip(events, event)
        return _RawFold(digest=_object_digest({}), shape={})
    members: dict[str, bytes] = {}
    shape: dict[str, object] = {}
    while True:
        event, value = next(events)
        if event == "end_map":
            break
        key = str(value)
        event, value = next(events)
        # Native-payload detection reads only root keys, their container
        # kinds, and the bridge-projection marker. Any other scalar is kept as
        # ``None`` so a huge string is not retained beside its digest.
        if event == "start_map":
            shape[key] = {}
        elif event == "start_array":
            shape[key] = []
        else:
            shape[key] = (
                "chatgpt-native-compact-v1"
                if key == "polylogue_bridge_projection" and value == "chatgpt-native-compact-v1"
                else None
            )
        members[key] = _structural_digest(events, event, value)
    return _RawFold(digest=_object_digest(members), shape=shape)


def _read_meta(events: Iterator[_Event], event: str, value: object, *, semantic: bool) -> bytes:
    """Digest one ``provider_meta`` object without materializing it.

    The models coerce a non-object to ``{}``, so it digests as one. With
    ``semantic``, backfill-observer attribution is left out -- it identifies
    who acquired a snapshot, not a property of the session -- and a
    ``backfill`` object left empty by that is dropped.
    """
    if event != "start_map":
        _skip(events, event)
        return _object_digest({})
    members: dict[str, bytes] = {}
    while True:
        event, value = next(events)
        if event == "end_map":
            return _object_digest(members)
        key = str(value)
        event, value = next(events)
        if not (semantic and key == "backfill" and event == "start_map"):
            members[key] = _structural_digest(events, event, value)
            continue
        backfill: dict[str, bytes] = {}
        while True:
            event, value = next(events)
            if event == "end_map":
                break
            backfill_key = str(value)
            event, value = next(events)
            if backfill_key in _BACKFILL_OBSERVER_ATTRIBUTION_KEYS:
                _skip(events, event)
            else:
                backfill[backfill_key] = _structural_digest(events, event, value)
        if backfill:
            members[key] = _object_digest(backfill)
        else:
            members.pop(key, None)


def _read_provenance(events: Iterator[_Event], event: str, value: object) -> tuple[object, bytes]:
    """Build provenance except its ``provider_meta``, which is digested."""
    meta = _object_digest({})
    if event != "start_map":
        return _build(events, event, value), meta
    provenance: dict[str, object] = {}
    while True:
        event, value = next(events)
        if event == "end_map":
            return provenance, meta
        key = str(value)
        event, value = next(events)
        if key == "provider_meta":
            meta = _read_meta(events, event, value, semantic=False)
        elif key in _PROVENANCE_FIELDS:
            provenance[key] = _build(events, event, value)
        else:
            _skip(events, event)


def _read_list(
    events: Iterator[_Event],
    event: str,
    name: str,
    add: Callable[[object], None],
) -> None:
    if event != "start_array":
        raise CaptureEnvelopeError("invalid_payload", f"session.{name} must be an array")
    while True:
        event, value = next(events)
        if event == "end_array":
            return
        add(_build(events, event, value))


def _read_session(
    events: Iterator[_Event], event: str, native_witness: _CanonicalNativeTurnWitness | None = None
) -> _SessionFold:
    if event != "start_map":
        raise CaptureEnvelopeError("invalid_payload", "session must be an object")
    session = _SessionFold()
    while True:
        event, value = next(events)
        if event == "end_map":
            return session
        key = str(value)
        event, value = next(events)
        if key == "turns":
            session.turns = _ItemFold(native_witness=native_witness)
            _read_list(events, event, key, session.turns.add_raw_turn)
        elif key == "attachments":
            session.attachments = _ItemFold()
            _read_list(events, event, key, session.attachments.add_raw_attachment)
        elif key == "provider_meta":
            session.meta_digest = _read_meta(events, event, value, semantic=True)
        elif key in _SESSION_FIELDS:
            session.head[key] = _build(events, event, value)
        else:
            _skip(events, event)


def _summarize_capture_stream(
    handle: IO[bytes], *, native_witness: _CanonicalNativeTurnWitness | None = None
) -> CaptureSummary:
    """Validate and fold one capture envelope in a single streamed pass."""
    events: Iterator[_Event] = _json_events(handle)
    root: dict[str, object] = {}
    session: _SessionFold | None = None
    raw = _RawFold()
    meta_digest = _object_digest({})
    provenance_meta_digest = _object_digest({})
    try:
        event, value = next(events)
        if event != "start_map":
            raise CaptureEnvelopeError("invalid_payload", "capture envelope must be a JSON object")
        while True:
            event, value = next(events)
            if event == "end_map":
                break
            key = str(value)
            event, value = next(events)
            if key == "session":
                session = _read_session(events, event, native_witness)
            elif key == "raw_provider_payload":
                raw = _read_raw_payload(events, event, value)
            elif key == "provider_meta":
                meta_digest = _read_meta(events, event, value, semantic=True)
            elif key == "provenance":
                root[key], provenance_meta_digest = _read_provenance(events, event, value)
            elif key in _ENVELOPE_FIELDS:
                root[key] = _build(events, event, value)
            else:
                _skip(events, event)
        for _ in events:
            raise CaptureEnvelopeError("invalid_json", "content after the envelope")
    except ijson.JSONError as exc:
        raise CaptureEnvelopeError("invalid_json", str(exc)) from exc
    except StopIteration as exc:
        raise CaptureEnvelopeError("invalid_json", "truncated envelope") from exc
    except (CaptureEnvelopeError, ValidationError, RecursionError, UnicodeError, TypeError) as exc:
        # A document that is not JSON is reported as such even when its
        # prefix already failed envelope validation, as a whole-document
        # decode would report it.
        _require_well_formed(events)
        if isinstance(exc, CaptureEnvelopeError):
            raise
        raise CaptureEnvelopeError("invalid_payload", repr(exc)) from exc
    return _summary(root, session, raw, meta_digest=meta_digest, provenance_meta_digest=provenance_meta_digest)


def _native_witness_from_stream(handle: IO[bytes]) -> _CanonicalNativeTurnWitness:
    """Parse this retained raw revision, without materializing typed turns.

    Native preparation supplies its existing scratch-backed witness directly.
    Independent spool replay uses the ordinary native model route here, whose
    provider/session allocation remains an explicit memory limitation.
    """
    from polylogue.browser_capture.identity import legacy_browser_capture_native_id
    from polylogue.sources.parsers.browser_capture import parse_native_payload

    events = _json_events(handle)
    event, _ = next(events)
    if event != "start_map":
        raise CaptureEnvelopeError("invalid_payload", "capture envelope must be an object")
    provider = Provider.UNKNOWN
    native_id = None
    raw = None
    for event, value in events:
        if event == "end_map":
            break
        key = str(value)
        event, value = next(events)
        if key == "raw_provider_payload":
            raw = _build(events, event, value)
        elif key == "session" and event == "start_map":
            while True:
                event, value = next(events)
                if event == "end_map":
                    break
                field_name = str(value)
                event, value = next(events)
                if field_name == "provider" and event == "string":
                    provider = Provider.from_string(str(value))
                elif field_name == "provider_session_id" and event == "string":
                    native_id = str(value)
                else:
                    _skip(events, event)
        else:
            _skip(events, event)
    if not native_id:
        raise CaptureEnvelopeError("invalid_payload", "state-only turn requires a native conversation")
    parsed = parse_native_payload(provider, raw, legacy_browser_capture_native_id(provider, native_id) or native_id)
    return _CanonicalNativeTurnWitness(parsed.messages)


def summarize_capture_stream(
    handle: IO[bytes], *, native_witness: _CanonicalNativeTurnWitness | None = None
) -> CaptureSummary:
    """Fold a capture, deriving native evidence only for actual state-only turns."""
    if not handle.seekable():
        # A replayable file is necessary only to inspect original native bytes
        # after a state-only turn is found. Never reject a valid stream solely
        # because its source does not implement seek.
        with tempfile.TemporaryFile(mode="w+b") as retained:
            while part := handle.read(CAPTURE_READ_CHUNK_BYTES):
                retained.write(part)
            retained.seek(0)
            return summarize_capture_stream(retained, native_witness=native_witness)
    start = handle.tell()
    try:
        return _summarize_capture_stream(handle, native_witness=native_witness)
    except _NativeWitnessRequiredError:
        if native_witness is not None:
            raise
    handle.seek(start)
    try:
        witness = _native_witness_from_stream(handle)
    except (ValueError, StopIteration, ijson.JSONError) as exc:
        raise CaptureEnvelopeError("invalid_payload", str(exc)) from exc
    handle.seek(start)
    return _summarize_capture_stream(handle, native_witness=witness)


def _require_well_formed(events: Iterator[_Event]) -> None:
    try:
        for _ in events:
            pass
    except ijson.JSONError as exc:
        raise CaptureEnvelopeError("invalid_json", str(exc)) from exc


#: Stands in for the turns a session already validated one by one, so the
#: session-level rule (at least one turn) is checked without holding any.
_PLACEHOLDER_TURN = BrowserCaptureTurn.model_validate({"provider_turn_id": "placeholder", "role": "user", "text": "-"})


def _summary(
    root: dict[str, object],
    session: _SessionFold | None,
    raw: _RawFold,
    *,
    meta_digest: bytes,
    provenance_meta_digest: bytes,
) -> CaptureSummary:
    head_input: dict[str, object] = dict(root)
    head_input["raw_provider_payload"] = raw.shape
    if session is not None:
        # Each turn was validated as it streamed past; the session's own
        # rule is only that it has one, so a placeholder stands in for them
        # and is dropped once the session validates.
        head_input["session"] = {
            **session.head,
            "turns": [_PLACEHOLDER_TURN] * min(session.turns.count, 1),
            "attachments": [],
        }
    try:
        head = BrowserCaptureEnvelope.model_validate(head_input)
    except ValidationError as exc:
        raise CaptureEnvelopeError("invalid_payload", str(exc)) from exc
    assert session is not None  # a validated head has a session
    head = head.model_copy(update={"session": head.session.model_copy(update={"turns": []})})
    session_dump = head.session.model_dump(mode="json", exclude_none=True, exclude={"turns", "attachments"})
    session_dump["provider_meta"] = (session.meta_digest or _object_digest({})).hex()
    head_bytes = dumps_bytes(
        {
            "polylogue_capture_kind": head.polylogue_capture_kind,
            "schema_version": head.schema_version,
            "session": session_dump,
            "provider_meta": meta_digest.hex(),
        },
        sort_keys=True,
    )

    def fingerprint(turns: bytes, attachments: bytes) -> hashlib._Hash:
        digest = hashlib.sha256(_DEDUP_DOMAIN)
        _framed(digest, head_bytes)
        digest.update(attachments)
        digest.update(turns)
        _framed(digest, raw.digest)
        return digest

    dedup = fingerprint(session.turns.digest.digest(), session.attachments.digest.digest()).hexdigest()
    carrierless = fingerprint(session.turns.carrierless.digest(), session.attachments.carrierless.digest()).digest()
    native = envelope_has_native_provider_payload(head.model_copy(update={"raw_provider_payload": raw.shape}))
    return CaptureSummary(
        head=head,
        turn_count=session.turns.count,
        dedup_content_hash=dedup,
        carrierless_fingerprint=carrierless,
        attachments=(*session.attachments.attachments, *session.turns.attachments),
        turn_identities=_turn_fidelities(head, session.turns.identities),
        has_native_provider_payload=native,
        provenance_meta_digest=provenance_meta_digest,
    )


def _turn_fidelities(
    head: BrowserCaptureEnvelope,
    observations: list[tuple[str, BrowserCaptureIdentityObservation]],
) -> tuple[tuple[str, Literal["native", "unknown"]], ...]:
    """A turn is native only when its observation names this capture's own identity.

    A self-declared ``fidelity="native"`` whose origin, conversation, message
    or adapter version disagrees with the envelope is not provider-native evidence.
    """
    from polylogue.browser_capture.identity import canonical_origin

    expected_origin = canonical_origin(head.session.provider)
    return tuple(
        (
            turn_id,
            "native"
            if observation.fidelity == "native"
            and observation.origin == expected_origin
            and observation.provider_conversation_id == head.session.provider_session_id
            and observation.provider_message_id == turn_id
            and observation.adapter_version == head.provenance.adapter_version
            else "unknown",
        )
        for turn_id, observation in observations
    )


def summarize_capture_file(path: Path, *, native_witness: _CanonicalNativeTurnWitness | None = None) -> CaptureSummary:
    with path.open("rb") as handle:
        return summarize_capture_stream(handle, native_witness=native_witness)


def read_capture_state_fields(path: Path) -> tuple[object, object]:
    """Stream ``capture_id`` and ``session.updated_at`` from a spooled artifact.

    Raises ``ValueError`` when the artifact is not a JSON object whose
    ``session`` (when present) is an object.
    """
    capture_id: object = None
    updated_at: object = None
    with path.open("rb") as handle:
        events: Iterator[_Event] = _json_events(handle)
        try:
            event, value = next(events)
            if event != "start_map":
                raise ValueError("capture artifact is not a JSON object")
            while True:
                event, value = next(events)
                if event == "end_map":
                    break
                key = str(value)
                event, value = next(events)
                if key == "capture_id":
                    capture_id = _build(events, event, value)
                elif key == "session":
                    if event != "start_map":
                        raise ValueError("capture artifact session is not an object")
                    while True:
                        event, value = next(events)
                        if event == "end_map":
                            break
                        session_key = str(value)
                        event, value = next(events)
                        if session_key == "updated_at":
                            updated_at = _build(events, event, value)
                        else:
                            _skip(events, event)
                else:
                    _skip(events, event)
            for _ in events:
                raise ValueError("content after the capture artifact")
        except ijson.JSONError as exc:
            raise ValueError(str(exc)) from exc
        except StopIteration as exc:
            raise ValueError("truncated capture artifact") from exc
    return capture_id, updated_at


__all__ = [
    "CAPTURE_READ_CHUNK_BYTES",
    "STAGING_DIRNAME",
    "AttachmentFact",
    "CaptureBodyIncompleteError",
    "CaptureEnvelopeError",
    "CaptureSummary",
    "SpoolStorageExhaustedError",
    "StagedCapture",
    "carrier_digest",
    "is_storage_exhausted",
    "read_capture_state_fields",
    "reap_stale_staging",
    "stage_capture_body",
    "stage_capture_chunks",
    "stage_retained_capture",
    "summarize_capture_file",
    "summarize_capture_stream",
]
