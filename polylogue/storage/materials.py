"""Durable, provider-neutral admission of linked materials.

Materials are observations, not sessions.  The record is committed even when
acquisition fails, while successfully obtained bytes are published to the
existing content-addressed store before the observation is committed.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import mimetypes
import os
import socket
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, Protocol

import ijson

from polylogue.core.prepared_file import PreparedFileSeal
from polylogue.storage.blob_publication import (
    ArchiveBlobPublisher,
    BlobPublicationSourceRead,
    ConnectionBlobPublicationRead,
    PreparedBlobPublicationClaim,
    _prepared_claim_from_record,
    _prepared_claim_record,
    consume_blob_publication_receipt,
)
from polylogue.storage.blob_store import BlobStore, PreparedBlob, blob_store_for_connection
from polylogue.storage.io_phase_metrics import connection_cursor

MaterialState = Literal[
    "claimed",
    "acquired",
    "duplicate",
    "partial",
    "unavailable",
    "expired",
    "access_denied",
    "malformed",
    "superseded",
]
MaterialPrivacy = Literal["private", "restricted", "public", "synthetic"]
MaterialRelation = Literal["refers_to", "acquired_from", "supports", "affected"]
MaterialFetchState = Literal["unavailable", "expired", "access_denied", "malformed", "partial"]


@dataclass(frozen=True, slots=True)
class MaterialObservation:
    material_id: str
    referrer_ref: str
    source_uri: str
    acquisition_state: MaterialState
    diagnostic: str
    retryable: bool
    blob_hash: str | None
    byte_size: int | None
    media_type: str | None
    media_charset: str | None
    filename: str | None
    extraction_manifest: dict[str, object]
    custody: Literal["claimed", "retained", "verified", "released"]
    privacy_classification: MaterialPrivacy
    acquired_at_ms: int
    created_at_ms: int


@dataclass(frozen=True, slots=True)
class MaterialEvidenceLink:
    evidence_ref: str
    relation: MaterialRelation
    authority: Literal["provider", "operator", "repository", "inferred", "unknown"]
    confidence: float
    observed_at_ms: int
    source_diagnostic: str


@dataclass(frozen=True, slots=True)
class MaterialPage:
    """A stable keyset page of material observations."""

    items: tuple[MaterialObservation, ...]
    next_cursor: tuple[int, str] | None


def _material_id(source_uri: str, referrer_ref: str, payload: bytes | None) -> str:
    digest = hashlib.sha256()
    digest.update(source_uri.encode("utf-8"))
    digest.update(b"\0")
    digest.update(referrer_ref.encode("utf-8"))
    digest.update(b"\0")
    if payload is not None:
        digest.update(payload)
    return "material:" + digest.hexdigest()


_JSON_MEDIA_TYPES = frozenset({"application/json", "text/json"})
_NDJSON_MEDIA_TYPES = frozenset({"application/ndjson", "application/x-ndjson"})
_JSON_SCALAR_TYPE_NAMES = {"string": "str", "boolean": "bool", "null": "NoneType"}


def _json_document_type(payload: bytes) -> str:
    return _json_document_type_of_stream(BytesIO(payload))


def _json_document_type_of_stream(source: BinaryIO) -> str:
    """Validate one complete JSON document and name its top-level type.

    The whole document is validated as a stream: a prefix of a large document
    is not a JSON value, and a budgeted slice would turn valid retained bytes
    into a false ``malformed`` verdict.
    """
    top_level: str | None = None
    for event, value in ijson.basic_parse(source, use_float=True):
        if top_level is None:
            if event == "start_map":
                top_level = "dict"
            elif event == "start_array":
                top_level = "list"
            elif event == "number":
                top_level = type(value).__name__
            else:
                top_level = _JSON_SCALAR_TYPE_NAMES.get(event, event)
    if top_level is None:
        raise ijson.IncompleteJSONError("empty JSON document")
    return top_level


def extraction_manifest(payload: bytes, media_type: str | None) -> dict[str, object]:
    """Describe retained bytes through the same streaming extractor."""
    return _extraction_manifest_stream(BytesIO(payload), len(payload), media_type)


def _extraction_manifest_stream(source: BinaryIO, size: int, media_type: str | None) -> dict[str, object]:
    """Describe retained bytes without copying unbounded content into metadata."""
    manifest: dict[str, object] = {"bytes": size, "extractor": "materials-v1"}
    kind = (media_type or "").lower()
    if kind in _JSON_MEDIA_TYPES or kind in _NDJSON_MEDIA_TYPES:
        try:
            if kind in _NDJSON_MEDIA_TYPES:
                # NDJSON is a record stream, not one JSON value.
                record_count = 0
                for line in source:
                    if line.strip():
                        json.loads(line.decode("utf-8"))
                        record_count += 1
                manifest["json_type"] = "ndjson"
                manifest["record_count"] = record_count
            else:
                manifest["json_type"] = _json_document_type_of_stream(source)
        except (UnicodeDecodeError, json.JSONDecodeError, ijson.JSONError) as exc:
            manifest["diagnostic"] = f"json extraction failed: {type(exc).__name__}: {exc}"
    elif kind in {"application/zip", "application/x-zip-compressed"}:
        try:
            with zipfile.ZipFile(source) as archive:
                # Entry names stay in the retained CAS bytes, not in this
                # queryable summary: one legal ZIP name can dwarf the manifest.
                entries = archive.infolist()
                manifest["entry_count"] = len(entries)
                manifest["uncompressed_bytes"] = sum(info.file_size for info in entries)
        except (OSError, zipfile.BadZipFile) as exc:
            manifest["diagnostic"] = f"zip extraction failed: {type(exc).__name__}: {exc}"
    elif kind.startswith("text/") or not kind:
        # Keep the manifest safe to expose through query surfaces. Raw text is
        # available from the CAS blob; it must not be copied into indexable
        # metadata or synthetic/public fixtures by default.
        manifest["text"] = {"available": True, "encoding": media_type or "unknown"}
    return manifest


class MaterialDestinationRefusedError(Exception):
    """Permanent refusal: the destination is not an admissible acquisition target.

    This is a policy outcome, not a transport failure. It never becomes
    retryable convergence debt; callers record it as a permanent observation.
    """

    def __init__(self, diagnostic: str) -> None:
        super().__init__(diagnostic)
        self.diagnostic = diagnostic


_MAX_REDIRECT_HOPS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def _address_is_refused(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Reject anything that is not a globally routable unicast destination."""
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return bool(
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _resolve_addresses(host: str, port: int) -> list[str]:
    """Resolve one host to literal addresses; the sole DNS seam for acquisition."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    ordered: list[str] = []
    for info in infos:
        literal = str(info[4][0])
        if literal not in ordered:
            ordered.append(literal)
    return ordered


def _vet_destination(url: str) -> tuple[str, int, str]:
    """Return (host, port, pinned address) or refuse permanently.

    Every resolved address must pass; a DNS-rebinding answer that mixes a
    public and a private address is refused rather than partially trusted.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise MaterialDestinationRefusedError(
            f"unsupported material URI scheme or missing host: {parsed.scheme or '<none>'}"
        )
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = _resolve_addresses(host, port)
    if not addresses:
        raise MaterialDestinationRefusedError(f"destination {host!r} resolved to no addresses")
    for literal in addresses:
        candidate = ipaddress.ip_address(literal)
        if _address_is_refused(candidate):
            raise MaterialDestinationRefusedError(
                f"destination policy refused non-public address {literal} for host {host!r}"
            )
    return host, port, addresses[0]


def _pinned_connection_factory(address: str, is_https: bool) -> Callable[..., http.client.HTTPConnection]:
    """Build a connection factory that dials the vetted address, not the name again.

    Only the socket target is pinned: ``self.host`` keeps the original name, so
    the ``Host`` header and the TLS SNI value stay correct while a second DNS
    lookup can no longer substitute a private address (rebinding).
    """

    def _create_connection(destination: tuple[str, int], *args: object, **kwargs: object) -> socket.socket:
        return socket.create_connection((address, destination[1]), *args, **kwargs)  # type: ignore[arg-type]

    def factory(host: str, **kwargs: Any) -> http.client.HTTPConnection:
        connection: http.client.HTTPConnection = (
            http.client.HTTPSConnection(host, **kwargs) if is_https else http.client.HTTPConnection(host, **kwargs)
        )
        connection._create_connection = _create_connection  # type: ignore[attr-defined]
        return connection

    return factory


class _NoRedirectErrorProcessor(urllib.request.HTTPErrorProcessor):
    """Hand 3xx responses back to the caller so every hop is re-vetted."""

    def http_response(self, request: urllib.request.Request, response: Any) -> Any:
        code: Any = getattr(response, "status", None) or getattr(response, "code", 200)
        if int(code or 200) in _REDIRECT_STATUSES:
            return response
        return super().http_response(request, response)

    https_response = http_response


def _open_url(url: str, address: str, timeout: float) -> Any:
    """Open one hop against the vetted address without following redirects."""
    is_https = urllib.parse.urlparse(url).scheme == "https"
    factory = _pinned_connection_factory(address, is_https)

    base: Any = urllib.request.HTTPSHandler if is_https else urllib.request.HTTPHandler

    class _Handler(base):
        def _open(self, req: urllib.request.Request) -> Any:
            return self.do_open(factory, req)

        http_open = _open
        https_open = _open

    opener = urllib.request.build_opener(_NoRedirectErrorProcessor, _Handler)
    return opener.open(url, timeout=timeout)


def _acquire_response(url: str, timeout: float) -> tuple[Any, str]:
    """Follow redirects manually, re-vetting the destination at every hop."""
    current = url
    for _ in range(_MAX_REDIRECT_HOPS + 1):
        _host, _port, address = _vet_destination(current)
        response = _open_url(current, address, timeout)
        status = int(getattr(response, "status", None) or getattr(response, "code", 200) or 200)
        if status not in _REDIRECT_STATUSES:
            return response, current
        location = response.headers.get("Location")
        close = getattr(response, "close", None)
        if callable(close):
            close()
        if not location:
            raise MaterialDestinationRefusedError(f"redirect from {current} carried no Location header")
        current = urllib.parse.urljoin(current, location)
    raise MaterialDestinationRefusedError(f"redirect chain exceeded {_MAX_REDIRECT_HOPS} hops from {url}")


def _declared_content_length(response: object) -> int | None:
    """Return a response's declared body length, or ``None`` if it declared none.

    Only a well-formed non-negative ``Content-Length`` is a completion
    statement. A missing, repeated-and-inconsistent, or unparseable header
    states nothing, and this returns ``None`` rather than substituting a guess.
    """
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    # Transfer coding determines HTTP framing when present; a stale length
    # must not downgrade a completely decoded response to partial evidence.
    if getattr(response, "chunked", False) or (hasattr(headers, "get") and headers.get("Transfer-Encoding")):
        return None
    get_all = getattr(headers, "get_all", None)
    values = get_all("Content-Length") if callable(get_all) else None
    if values is None:
        raw = headers.get("Content-Length") if hasattr(headers, "get") else None
        values = [] if raw is None else [raw]
    declared: set[int] = set()
    for value in values:
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        if parsed < 0:
            return None
        declared.add(parsed)
    if len(declared) != 1:
        return None
    return declared.pop()


@dataclass(frozen=True, slots=True, init=False)
class PreparedMaterial:
    """One off-writer material preparation, including its private-file seal."""

    material_id: str
    source_uri: str
    referrer_ref: str
    state: MaterialState
    infer_duplicate: bool
    diagnostic: str
    retryable: bool
    media_type: str | None
    media_charset: str | None
    filename: str | None
    privacy_classification: MaterialPrivacy
    manifest_json: str
    blob_root: Path
    blob: PreparedBlob | None
    seal: PreparedFileSeal | None
    publisher: ArchiveBlobPublisher
    publication_claim: PreparedBlobPublicationClaim | None

    def discard(self) -> None:
        if self.blob is not None:
            BlobStore(self.blob_root).discard_prepared(self.blob)


def _material_preparation(
    material_id: str,
    source_uri: str,
    referrer_ref: str,
    state: MaterialState,
    infer_duplicate: bool,
    diagnostic: str,
    retryable: bool,
    media_type: str | None,
    media_charset: str | None,
    filename: str | None,
    privacy_classification: MaterialPrivacy,
    manifest_json: str,
    blob_root: Path,
    blob: PreparedBlob | None,
    seal: PreparedFileSeal | None,
    publisher: ArchiveBlobPublisher,
    publication_claim: PreparedBlobPublicationClaim | None,
) -> PreparedMaterial:
    from dataclasses import fields

    prepared = object.__new__(PreparedMaterial)
    values = (
        material_id,
        source_uri,
        referrer_ref,
        state,
        infer_duplicate,
        diagnostic,
        retryable,
        media_type,
        media_charset,
        filename,
        privacy_classification,
        manifest_json,
        blob_root,
        blob,
        seal,
        publisher,
        publication_claim,
    )
    for field, value in zip(fields(PreparedMaterial), values, strict=True):
        object.__setattr__(prepared, field.name, value)
    return prepared


def _prepared_material_record(prepared: PreparedMaterial) -> str:
    """Encode preparation inside the canonical sealed artifact."""
    from dataclasses import asdict, fields

    record = {
        field.name: getattr(prepared, field.name)
        for field in fields(prepared)
        if field.name not in {"publisher", "publication_claim", "blob", "seal", "blob_root"}
    }
    record["blob_root"] = str(prepared.blob_root)
    record["blob"] = (
        {
            "hash_hex": prepared.blob.hash_hex,
            "size_bytes": prepared.blob.size_bytes,
            "temporary_path": str(prepared.blob.temporary_path),
        }
        if prepared.blob is not None
        else None
    )
    record["seal"] = asdict(prepared.seal) if prepared.seal is not None else None
    record["publication_claim"] = (
        _prepared_claim_record(prepared.publication_claim) if prepared.publication_claim is not None else None
    )
    return json.dumps(record, sort_keys=True)


def _prepared_material_from_record(encoded: str, publisher: ArchiveBlobPublisher) -> PreparedMaterial:
    """Restore a claim only from an already-verified canonical artifact row."""
    record = json.loads(encoded)
    record["blob_root"] = Path(record["blob_root"])
    if record["blob_root"] != publisher.root.resolve():
        raise ValueError("sealed material belongs to another publisher root")
    if record["blob"] is not None:
        record["blob"]["temporary_path"] = Path(record["blob"]["temporary_path"])
        record["blob"] = PreparedBlob(**record["blob"])
    if record["seal"] is not None:
        record["seal"] = PreparedFileSeal(**record["seal"])
    encoded_claim = record.pop("publication_claim")
    claim = _prepared_claim_from_record(encoded_claim, publisher) if encoded_claim is not None else None
    if claim is not None and (
        record["blob"] is None
        or record["seal"] != claim.seal
        or record["blob"].hash_hex != claim.receipt.blob_hash
        or record["blob"].size_bytes != claim.receipt.size_bytes
        or Path(os.path.abspath(record["blob"].temporary_path)) != claim.prepared_path
    ):
        raise ValueError("sealed material disagrees with its captured publication claim")
    return _material_preparation(**record, publisher=publisher, publication_claim=claim)


def prepare_material(
    *,
    blob_store: ArchiveBlobPublisher,
    staging_directory: Path | None = None,
    source_uri: str,
    referrer_ref: str,
    payload: bytes | BinaryIO | None = None,
    media_type: str | None = None,
    media_charset: str | None = None,
    filename: str | None = None,
    state: MaterialState | None = None,
    diagnostic: str = "",
    retryable: bool = False,
    privacy_classification: MaterialPrivacy = "private",
) -> PreparedMaterial:
    """Prepare bytes, extraction and identity before archive writer admission."""
    if not source_uri.strip() or not referrer_ref.strip():
        raise ValueError("source_uri and referrer_ref are required")
    if privacy_classification == "synthetic" and payload is not None:
        raise ValueError("synthetic materials cannot carry arbitrary raw bytes")
    material_state: MaterialState = state or ("acquired" if payload is not None else "claimed")
    prepared_blob = None
    seal = None
    claim = None
    if payload is not None:
        from polylogue.core.compute_cancel import check_compute_cancelled

        class CheckedInput:
            def read(self, size: int = -1) -> bytes:
                check_compute_cancelled()
                return stream.read(size)

        stream = BytesIO(payload) if isinstance(payload, bytes) else payload
        media_type = media_type or mimetypes.guess_type(filename or "")[0]
        prepared_blob = blob_store.prepare_from_fileobj(CheckedInput(), staging_directory=staging_directory)
        try:
            digest = hashlib.sha256()
            digest.update(source_uri.encode("utf-8"))
            digest.update(b"\0")
            digest.update(referrer_ref.encode("utf-8"))
            digest.update(b"\0")
            with prepared_blob.temporary_path.open("rb") as captured:
                while chunk := captured.read(1024 * 1024):
                    check_compute_cancelled()
                    digest.update(chunk)
                material_id = "material:" + digest.hexdigest()
                captured.seek(0)
                manifest = _extraction_manifest_stream(captured, prepared_blob.size_bytes, media_type)
            if manifest.get("diagnostic") and material_state == "acquired":
                material_state = "malformed"
                if not diagnostic:
                    diagnostic = str(manifest["diagnostic"])
            claim = blob_store.prepare_claim(prepared_blob)
            seal = claim.seal
        except BaseException as primary:
            try:
                blob_store.discard_prepared(prepared_blob)
            except BaseException as cleanup:
                primary.add_note(f"material preparation cleanup failed: {cleanup!r}")
            raise
    else:
        material_id = _material_id(source_uri, referrer_ref, None)
        manifest = {"bytes": None, "extractor": "materials-v1"}
    return _material_preparation(
        material_id,
        source_uri,
        referrer_ref,
        material_state,
        state is None,
        diagnostic,
        retryable,
        media_type,
        media_charset,
        filename,
        privacy_classification,
        json.dumps(manifest, sort_keys=True),
        blob_store.root.resolve(),
        prepared_blob,
        seal,
        blob_store,
        claim,
    )


def publish_prepared_materials(
    materials: Iterable[PreparedMaterial], *, reference_seal: PreparedIndexMutation | None = None
) -> None:
    """Publish closed bounded material pages through the canonical page bodies."""
    from polylogue.core.compute_cancel import check_compute_cancelled

    page: list[PreparedMaterial] = []

    def flush_page() -> None:
        if not page:
            return
        publisher = prepare_material_publication_page(page, reference_seal=reference_seal)
        flush_material_publication_page(page, publisher, reference_seal=reference_seal)
        page.clear()

    for material in materials:
        check_compute_cancelled()
        if material.blob is None:
            continue
        if page and page[0].publisher is not material.publisher:
            flush_page()
        page.append(material)
        if len(page) == 256:
            flush_page()
    flush_page()


def admit_material(
    conn: sqlite3.Connection,
    *,
    prepared: PreparedMaterial,
    observed_at_ms: int,
    supersedes_material_id: str | None = None,
    commit: bool = True,
) -> MaterialObservation:
    """Apply a sealed preparation through the existing archive publisher."""
    if prepared.publisher.root.resolve() != prepared.blob_root:
        raise ValueError("material preparation belongs to another archive")
    source_uri, referrer_ref = prepared.source_uri, prepared.referrer_ref
    material_id = prepared.material_id
    material_state = prepared.state
    diagnostic, retryable = prepared.diagnostic, prepared.retryable
    media_type, media_charset, filename = prepared.media_type, prepared.media_charset, prepared.filename
    privacy_classification = prepared.privacy_classification
    if prepared.blob is not None:
        claim = prepared.publication_claim
        if claim is None:
            raise ValueError("material has no captured publication claim")
        blob_hash, byte_size = prepared.publisher.validate_published_claim(
            ConnectionBlobPublicationRead(conn), claim, source_path=prepared.source_uri
        )
        custody = "retained"
    else:
        if prepared.seal is not None or prepared.publication_claim is not None:
            raise ValueError("material claim has publication proof without bytes")
        blob_hash, byte_size, custody = None, None, "claimed"
    if blob_hash is not None:
        duplicate = (
            conn.execute(
                "SELECT 1 FROM material_observations WHERE blob_hash = ? AND material_id != ? LIMIT 1",
                (bytes.fromhex(blob_hash), material_id),
            ).fetchone()
            is not None
        )
        if duplicate and prepared.infer_duplicate:
            material_state = "duplicate"
    now = observed_at_ms
    conn.execute(
        """INSERT INTO material_observations
        (material_id, referrer_ref, source_uri, acquisition_state, diagnostic,
         retryable, supersedes_material_id, blob_hash, byte_size, media_type,
         media_charset, filename, extraction_manifest_json, custody,
         privacy_classification, acquired_at_ms, created_at_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(material_id) DO UPDATE SET
          acquisition_state=excluded.acquisition_state, diagnostic=excluded.diagnostic,
          retryable=excluded.retryable, blob_hash=excluded.blob_hash,
          byte_size=excluded.byte_size, extraction_manifest_json=excluded.extraction_manifest_json,
          custody=excluded.custody, acquired_at_ms=excluded.acquired_at_ms""",
        (
            material_id,
            referrer_ref,
            source_uri,
            material_state,
            diagnostic,
            int(retryable),
            supersedes_material_id,
            bytes.fromhex(blob_hash) if blob_hash else None,
            byte_size,
            media_type,
            media_charset,
            filename,
            prepared.manifest_json,
            custody,
            privacy_classification,
            observed_at_ms,
            now,
        ),
    )
    if prepared.publication_claim is not None:
        consume_blob_publication_receipt(
            conn,
            prepared.publication_claim.receipt.publication_id,
            bytes.fromhex(prepared.publication_claim.receipt.blob_hash),
        )
    # A readmission keeps the stored identity metadata and creation time, so
    # report the row that persisted rather than this call's arguments.
    observation = get_material(conn, material_id)
    if observation is None:
        raise RuntimeError(f"material admission did not persist {material_id}")
    if commit:
        conn.commit()
    return observation


def prepare_material_acquisition(
    *,
    source_uri: str,
    referrer_ref: str,
    blob_store: ArchiveBlobPublisher,
    media_type: str | None = None,
    media_charset: str | None = None,
    filename: str | None = None,
    privacy_classification: MaterialPrivacy = "private",
    timeout_seconds: float = 20.0,
) -> PreparedMaterial:
    """Acquire a URL while retaining a durable claim for every outcome.

    The response streams into archive-owned private staging. Redirects
    are followed manually so that every hop is re-vetted by the destination
    policy, and the final URL is recorded in the diagnostic when it differs
    from the admitted source URI. Transport failures remain retryable material
    observations; a destination the policy refuses is a permanent
    ``access_denied`` observation with no bytes retained.
    """
    if not source_uri.strip() or not referrer_ref.strip():
        raise ValueError("source_uri and referrer_ref are required")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        try:
            parsed = urllib.parse.urlparse(source_uri)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"unsupported material URI scheme or missing host: {parsed.scheme or '<none>'}")
            opened, final_uri = _acquire_response(source_uri, timeout_seconds)
        except (ValueError, http.client.InvalidURL) as exc:
            # URL parsing/connection construction (including redirect targets)
            # can fail before any response exists. Preserve the failed claim.
            return prepare_material(
                blob_store=blob_store,
                source_uri=source_uri,
                referrer_ref=referrer_ref,
                filename=filename,
                state="malformed",
                diagnostic=f"invalid material URI: {type(exc).__name__}: {exc}",
                retryable=False,
                privacy_classification=privacy_classification,
            )
        with opened as response:
            response_media_type = response.headers.get_content_type()
            response_charset = response.headers.get_content_charset()
            diagnostic = "" if final_uri == source_uri else f"redirected to {final_uri}"
            prepared = prepare_material(
                blob_store=blob_store,
                source_uri=source_uri,
                referrer_ref=referrer_ref,
                payload=response,
                media_type=media_type or response_media_type,
                media_charset=media_charset or response_charset,
                filename=filename,
                diagnostic=diagnostic,
                privacy_classification=privacy_classification,
            )
            declared_length = _declared_content_length(response)
            assert prepared.blob is not None
            if declared_length is not None and prepared.blob.size_bytes < declared_length:
                from dataclasses import fields

                short = f"response declared {declared_length} bytes but the connection closed after {prepared.blob.size_bytes}"
                if diagnostic:
                    short += f"; {diagnostic}"
                values = {field.name: getattr(prepared, field.name) for field in fields(prepared)}
                values.update(state="partial", infer_duplicate=False, diagnostic=short, retryable=True)
                return _material_preparation(**values)
            return prepared
    except MaterialDestinationRefusedError as exc:
        return prepare_material(
            blob_store=blob_store,
            source_uri=source_uri,
            referrer_ref=referrer_ref,
            filename=filename,
            state="access_denied",
            diagnostic=exc.diagnostic,
            retryable=False,
            privacy_classification=privacy_classification,
        )
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        state: MaterialFetchState = (
            "expired" if status in {404, 410} else "access_denied" if status in {401, 403} else "unavailable"
        )
        return prepare_material(
            blob_store=blob_store,
            source_uri=source_uri,
            referrer_ref=referrer_ref,
            filename=filename,
            state=state,
            diagnostic=(
                f"HTTP {status} {exc.reason}"
                + (f"; redirected to {exc.geturl()}" if exc.geturl() and exc.geturl() != source_uri else "")
            ),
            retryable=state == "unavailable",
            privacy_classification=privacy_classification,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return prepare_material(
            blob_store=blob_store,
            source_uri=source_uri,
            referrer_ref=referrer_ref,
            filename=filename,
            state="unavailable",
            diagnostic=f"acquisition failed: {type(exc).__name__}: {exc}",
            retryable=True,
            privacy_classification=privacy_classification,
        )


def prepare_material_file(
    *,
    path: str | Path,
    referrer_ref: str,
    blob_store: ArchiveBlobPublisher,
    media_type: str | None = None,
    privacy_classification: MaterialPrivacy = "private",
) -> PreparedMaterial:
    """Admit a pasted/downloaded local file through the same material route."""
    file_path = Path(path).absolute()
    source_uri = file_path.as_uri()
    try:
        with file_path.open("rb") as source:
            return prepare_material(
                blob_store=blob_store,
                source_uri=source_uri,
                referrer_ref=referrer_ref,
                payload=source,
                filename=file_path.name,
                media_type=media_type,
                privacy_classification=privacy_classification,
            )
    except PermissionError as exc:
        return prepare_material(
            blob_store=blob_store,
            source_uri=source_uri,
            referrer_ref=referrer_ref,
            filename=file_path.name,
            state="access_denied",
            diagnostic=f"file access denied: {exc}",
            retryable=False,
            privacy_classification=privacy_classification,
        )
    except FileNotFoundError as exc:
        return prepare_material(
            blob_store=blob_store,
            source_uri=source_uri,
            referrer_ref=referrer_ref,
            filename=file_path.name,
            state="unavailable",
            diagnostic=f"file unavailable: {exc}",
            retryable=True,
            privacy_classification=privacy_classification,
        )


def link_material(
    conn: sqlite3.Connection,
    material_id: str,
    evidence_ref: str,
    *,
    relation: MaterialRelation,
    authority: Literal["provider", "operator", "repository", "inferred", "unknown"] = "unknown",
    confidence: float = 1.0,
    observed_at_ms: int,
    source_diagnostic: str = "",
    commit: bool = True,
) -> None:
    if not material_id.strip() or not evidence_ref.strip():
        raise ValueError("material_id and evidence_ref are required")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if conn.execute("SELECT 1 FROM material_observations WHERE material_id = ?", (material_id,)).fetchone() is None:
        raise KeyError(material_id)
    conn.execute(
        """INSERT INTO material_evidence_links
      (material_id, evidence_ref, relation, authority, confidence, observed_at_ms, source_diagnostic)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(material_id, evidence_ref, relation) DO UPDATE SET
        authority=excluded.authority, confidence=excluded.confidence,
        observed_at_ms=excluded.observed_at_ms, source_diagnostic=excluded.source_diagnostic""",
        (material_id, evidence_ref, relation, authority, confidence, observed_at_ms, source_diagnostic),
    )
    if commit:
        conn.commit()


def get_material(conn: sqlite3.Connection, material_id: str) -> MaterialObservation | None:
    cursor = conn.execute("SELECT * FROM material_observations WHERE material_id = ?", (material_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [column[0] for column in cursor.description or ()]
    row = dict(zip(columns, row, strict=True))
    return MaterialObservation(
        material_id=row["material_id"],
        referrer_ref=row["referrer_ref"],
        source_uri=row["source_uri"],
        acquisition_state=row["acquisition_state"],
        diagnostic=row["diagnostic"],
        retryable=bool(row["retryable"]),
        blob_hash=bytes(row["blob_hash"]).hex() if row["blob_hash"] is not None else None,
        byte_size=row["byte_size"],
        media_type=row["media_type"],
        media_charset=row["media_charset"],
        filename=row["filename"],
        extraction_manifest=json.loads(row["extraction_manifest_json"]),
        custody=row["custody"],
        privacy_classification=row["privacy_classification"],
        acquired_at_ms=row["acquired_at_ms"],
        created_at_ms=row["created_at_ms"],
    )


def read_material(conn: sqlite3.Connection, material_id: str, *, blob_store: BlobStore | None = None) -> bytes:
    """Read retained bytes for one material, preserving claim-only absence."""
    observation = get_material(conn, material_id)
    if observation is None:
        raise KeyError(material_id)
    if observation.blob_hash is None:
        raise FileNotFoundError(f"material {material_id!r} has no retained bytes")
    # Read from the same archive the observation was committed in; see
    # ``blob_store_for_connection``.
    return (blob_store or blob_store_for_connection(conn)).read_all(observation.blob_hash)


def list_materials(conn: sqlite3.Connection, *, evidence_ref: str | None = None) -> list[MaterialObservation]:
    """List truthful observations, optionally through a direct evidence link."""
    if evidence_ref is None:
        rows = conn.execute("SELECT * FROM material_observations ORDER BY created_at_ms, material_id").fetchall()
    else:
        rows = conn.execute(
            "SELECT m.* FROM material_observations m WHERE EXISTS ("
            "SELECT 1 FROM material_evidence_links l "
            "WHERE l.material_id = m.material_id AND l.evidence_ref = ?) "
            "ORDER BY m.created_at_ms, m.material_id",
            (evidence_ref,),
        ).fetchall()
    observations: list[MaterialObservation] = []
    for row in rows:
        # The source-tier query API accepts both the default tuple row factory
        # and sqlite3.Row connections used by the archive runtime.
        material_id = row[0] if not isinstance(row, sqlite3.Row) else row["material_id"]
        observation = get_material(conn, material_id)
        if observation is not None:
            observations.append(observation)
    return observations


def list_materials_page(
    conn: sqlite3.Connection,
    *,
    evidence_ref: str | None = None,
    after: tuple[int, str] | None = None,
    limit: int = 256,
) -> MaterialPage:
    """Page observations in creation order, including Codex text chunks."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    where = []
    params: list[object] = []
    if evidence_ref is not None:
        where.append(
            "EXISTS (SELECT 1 FROM material_evidence_links l WHERE l.material_id = m.material_id AND l.evidence_ref = ?)"
        )
        params.append(evidence_ref)
    if after is not None:
        where.append("(m.created_at_ms, m.material_id) > (?, ?)")
        params.extend(after)
    predicate = " WHERE " + " AND ".join(where) if where else ""
    rows = conn.execute(
        "SELECT m.material_id, m.created_at_ms FROM material_observations m"
        + predicate
        + " ORDER BY m.created_at_ms, m.material_id LIMIT ?",
        (*params, limit + 1),
    ).fetchall()
    ids = rows[:limit]
    items = tuple(observation for row in ids if (observation := get_material(conn, str(row[0]))) is not None)
    cursor = (int(ids[-1][1]), str(ids[-1][0])) if len(rows) > limit else None
    return MaterialPage(items=items, next_cursor=cursor)


def list_material_links(conn: sqlite3.Connection, material_id: str) -> list[MaterialEvidenceLink]:
    """Return direct provenance/effect edges for one retained or claimed material."""
    rows = conn.execute(
        """SELECT evidence_ref, relation, authority, confidence, observed_at_ms, source_diagnostic
           FROM material_evidence_links
           WHERE material_id = ?
           ORDER BY observed_at_ms, evidence_ref, relation""",
        (material_id,),
    ).fetchall()
    return [
        MaterialEvidenceLink(
            evidence_ref=row[0],
            relation=row[1],
            authority=row[2],
            confidence=float(row[3]),
            observed_at_ms=int(row[4]),
            source_diagnostic=row[5],
        )
        for row in rows
    ]


__all__ = [
    "MaterialDestinationRefusedError",
    "MaterialObservation",
    "MaterialPage",
    "PreparedMaterial",
    "prepare_material",
    "publish_prepared_materials",
    "admit_material",
    "prepare_material_file",
    "prepare_material_acquisition",
    "extraction_manifest",
    "get_material",
    "link_material",
    "list_material_links",
    "list_materials",
    "list_materials_page",
    "read_material",
    "MaterialEvidenceLink",
]


def prepare_material_publication_page(
    page: Sequence[PreparedMaterial],
    *,
    reference_seal: PreparedIndexMutation | None,
) -> ArchiveBlobPublisher:
    """Queue this closed canonical page and prepare its original reservation."""
    if not page:
        raise ValueError("material publication page is empty")
    publisher = page[0].publisher
    for material in page:
        if material.publisher is not publisher or material.blob is None or material.publication_claim is None:
            raise ValueError("material page differs from its captured publisher or publication claim")
        publisher.queue_prepared(material.blob, claim=material.publication_claim)
    if reference_seal is not None:
        from polylogue.core.write_lease import current_write_lease

        if current_write_lease() is not None:
            raise RuntimeError("prepared material reservations require lease-free preparation")
        publisher.prepare_flush(reference_seal=reference_seal)
    return publisher


def flush_material_publication_page(
    page: Sequence[PreparedMaterial],
    publisher: ArchiveBlobPublisher,
    *,
    reference_seal: PreparedIndexMutation | None,
) -> None:
    """Flush the same pending batch and preserve its typed placement outcome."""
    from polylogue.storage.blob_publication import require_published

    if any(material.publisher is not publisher for material in page):
        raise ValueError("material continuation requires its original publisher")
    if reference_seal is None:
        publisher.flush()
    else:
        from polylogue.core.stage_admission import admit_stage_write
        from polylogue.core.write_lease import current_write_lease

        if current_write_lease() is not None:
            raise RuntimeError("prepared material reservations require lease-free preparation")
        admit_stage_write(
            "prepared-material-blob-reservations", partial(publisher.flush, reference_seal=reference_seal)
        )
    try:
        for material in page:
            assert material.blob is not None
            require_published(publisher, material.blob.hash_hex, source_path=material.source_uri)
    finally:
        for material in page:
            assert material.publication_claim is not None
            publisher.forget_completed_claim(material.publication_claim)


class MaterialSourceProducer(Protocol):
    """Finite canonical material operations on ordinary or selected Source state."""

    def material_publication_read(self, publisher: ArchiveBlobPublisher) -> BlobPublicationSourceRead: ...

    def material_literal(self, value: object) -> tuple[str, tuple[object, ...]]: ...
    def material_rows(self, material_id: str) -> AbstractContextManager[sqlite3.Cursor]: ...
    def material_duplicate_rows(self, blob_hash: bytes, material_id: str) -> AbstractContextManager[sqlite3.Cursor]: ...
    def material_previous_rows(
        self,
        referrer_ref: str,
        source_uri: str,
    ) -> AbstractContextManager[sqlite3.Cursor]: ...
    def material_supersede_write(
        self,
        material_id: str,
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]: ...
    def material_write(
        self,
        material_id: str,
        supersedes_material_id: str | None,
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]: ...
    def material_link_write(
        self,
        key: tuple[str, str, str],
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]: ...
    def consume_material_receipt(self, claim: PreparedBlobPublicationClaim) -> None: ...


_MATERIAL_ROWS_SQL = "SELECT * FROM material_observations WHERE material_id = ?"


_MATERIAL_DUPLICATE_SQL = "SELECT 1 FROM material_observations WHERE blob_hash = ? AND material_id != ? LIMIT 1"


_MATERIAL_PREVIOUS_SQL = (
    "SELECT m.material_id FROM material_evidence_links AS l "
    "JOIN material_observations AS m USING(material_id) "
    "WHERE l.evidence_ref = ? AND l.relation = 'refers_to' "
    "AND m.source_uri = ? AND m.acquisition_state != 'superseded' "
    "ORDER BY m.created_at_ms DESC, m.material_id DESC LIMIT 1"
)


@dataclass(frozen=True, slots=True)
class ConnectionMaterialSourceProducer:
    connection: sqlite3.Connection

    def material_publication_read(self, publisher: ArchiveBlobPublisher) -> BlobPublicationSourceRead:
        return ConnectionBlobPublicationRead(self.connection)

    def material_literal(self, value: object) -> tuple[str, tuple[object, ...]]:
        return "?", (value,)

    def material_rows(self, material_id: str) -> AbstractContextManager[sqlite3.Cursor]:
        return connection_cursor(self.connection, _MATERIAL_ROWS_SQL, (material_id,))

    def material_duplicate_rows(self, blob_hash: bytes, material_id: str) -> AbstractContextManager[sqlite3.Cursor]:
        return connection_cursor(self.connection, _MATERIAL_DUPLICATE_SQL, (blob_hash, material_id))

    def material_previous_rows(
        self,
        referrer_ref: str,
        source_uri: str,
    ) -> AbstractContextManager[sqlite3.Cursor]:
        return connection_cursor(self.connection, _MATERIAL_PREVIOUS_SQL, (referrer_ref, source_uri))

    def material_supersede_write(
        self,
        material_id: str,
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]:
        return connection_cursor(self.connection, sql, parameters)

    def material_write(
        self,
        material_id: str,
        supersedes_material_id: str | None,
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]:
        return connection_cursor(self.connection, sql, parameters)

    def material_link_write(
        self,
        key: tuple[str, str, str],
        sql: str,
        parameters: tuple[object, ...],
    ) -> AbstractContextManager[sqlite3.Cursor]:
        return connection_cursor(self.connection, sql, parameters)

    def consume_material_receipt(self, claim: PreparedBlobPublicationClaim) -> None:
        consume_blob_publication_receipt(
            self.connection,
            claim.receipt.publication_id,
            bytes.fromhex(claim.receipt.blob_hash),
        )


def _admit_material(
    producer: MaterialSourceProducer,
    *,
    prepared: PreparedMaterial,
    observed_at_ms: int,
    supersedes_material_id: str | None = None,
) -> MaterialObservation:
    """Apply a sealed preparation through the existing archive publisher."""
    if prepared.publisher.root.resolve() != prepared.blob_root:
        raise ValueError("material preparation belongs to another archive")
    source_uri, referrer_ref = prepared.source_uri, prepared.referrer_ref
    material_id = prepared.material_id
    material_state = prepared.state
    diagnostic, retryable = prepared.diagnostic, prepared.retryable
    media_type, media_charset, filename = prepared.media_type, prepared.media_charset, prepared.filename
    privacy_classification = prepared.privacy_classification
    if prepared.blob is not None:
        claim = prepared.publication_claim
        if claim is None:
            raise ValueError("material has no captured publication claim")
        blob_hash, byte_size = prepared.publisher.validate_published_claim(
            producer.material_publication_read(prepared.publisher), claim, source_path=prepared.source_uri
        )
        custody = "retained"
    else:
        if prepared.seal is not None or prepared.publication_claim is not None:
            raise ValueError("material claim has publication proof without bytes")
        blob_hash, byte_size, custody = None, None, "claimed"
    if blob_hash is not None:
        with producer.material_duplicate_rows(bytes.fromhex(blob_hash), material_id) as rows:
            duplicate = rows.fetchone() is not None
        if duplicate and prepared.infer_duplicate:
            material_state = "duplicate"
    now = observed_at_ms
    values = (
        None,
        material_id,
        referrer_ref,
        source_uri,
        material_state,
        diagnostic,
        int(retryable),
        supersedes_material_id,
        bytes.fromhex(blob_hash) if blob_hash else None,
        byte_size,
        media_type,
        media_charset,
        filename,
        prepared.manifest_json,
        custody,
        privacy_classification,
        observed_at_ms,
        now,
    )
    expressions: list[str] = []
    parameters: list[object] = []
    for position, value in enumerate(values):
        expression, operands = ("?", (None,)) if position == 0 else producer.material_literal(value)
        expressions.append(expression)
        parameters.extend(operands)
    sql = """INSERT INTO material_observations
        (rowid, material_id, referrer_ref, source_uri, acquisition_state, diagnostic,
         retryable, supersedes_material_id, blob_hash, byte_size, media_type,
         media_charset, filename, extraction_manifest_json, custody,
         privacy_classification, acquired_at_ms, created_at_ms)
        VALUES ({values})
        ON CONFLICT(material_id) DO UPDATE SET
          acquisition_state=excluded.acquisition_state, diagnostic=excluded.diagnostic,
          retryable=excluded.retryable, blob_hash=excluded.blob_hash,
          byte_size=excluded.byte_size, extraction_manifest_json=excluded.extraction_manifest_json,
          custody=excluded.custody, acquired_at_ms=excluded.acquired_at_ms""".format(values=", ".join(expressions))
    with producer.material_write(material_id, supersedes_material_id, sql, tuple(parameters)):
        pass
    if prepared.publication_claim is not None:
        producer.consume_material_receipt(prepared.publication_claim)

    # A readmission keeps the stored identity metadata and creation time, so
    # report the row that persisted rather than this call's arguments.
    with producer.material_rows(material_id) as cursor:
        observation = _material_from_cursor(cursor)
    if observation is None:
        raise RuntimeError(f"material admission did not persist {material_id}")
    return observation


def _supersede_material(producer: MaterialSourceProducer, material_id: str) -> None:
    state_sql, state_operands = producer.material_literal("superseded")
    key_sql, key_operands = producer.material_literal(material_id)
    with producer.material_supersede_write(
        material_id,
        f"UPDATE material_observations SET acquisition_state={state_sql} WHERE material_id={key_sql}",
        (*state_operands, *key_operands),
    ):
        pass


def _link_material(
    producer: MaterialSourceProducer,
    material_id: str,
    evidence_ref: str,
    *,
    relation: MaterialRelation,
    authority: Literal["provider", "operator", "repository", "inferred", "unknown"] = "unknown",
    confidence: float = 1.0,
    observed_at_ms: int,
    source_diagnostic: str = "",
) -> None:
    if not material_id.strip() or not evidence_ref.strip():
        raise ValueError("material_id and evidence_ref are required")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    with producer.material_rows(material_id) as rows:
        if rows.fetchone() is None:
            raise KeyError(material_id)
    values = (None, material_id, evidence_ref, relation, authority, confidence, observed_at_ms, source_diagnostic)
    expressions: list[str] = []
    parameters: list[object] = []
    for position, value in enumerate(values):
        expression, operands = ("?", (None,)) if position == 0 else producer.material_literal(value)
        expressions.append(expression)
        parameters.extend(operands)
    sql = """INSERT INTO material_evidence_links
      (rowid, material_id, evidence_ref, relation, authority, confidence, observed_at_ms, source_diagnostic)
      VALUES ({values})
      ON CONFLICT(material_id, evidence_ref, relation) DO UPDATE SET
        authority=excluded.authority, confidence=excluded.confidence,
        observed_at_ms=excluded.observed_at_ms, source_diagnostic=excluded.source_diagnostic""".format(
        values=", ".join(expressions)
    )
    with producer.material_link_write((material_id, evidence_ref, relation), sql, tuple(parameters)):
        pass


def _material_from_cursor(cursor: sqlite3.Cursor) -> MaterialObservation | None:
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [column[0] for column in cursor.description or ()]
    row = dict(zip(columns, row, strict=True))
    return MaterialObservation(
        material_id=row["material_id"],
        referrer_ref=row["referrer_ref"],
        source_uri=row["source_uri"],
        acquisition_state=row["acquisition_state"],
        diagnostic=row["diagnostic"],
        retryable=bool(row["retryable"]),
        blob_hash=bytes(row["blob_hash"]).hex() if row["blob_hash"] is not None else None,
        byte_size=row["byte_size"],
        media_type=row["media_type"],
        media_charset=row["media_charset"],
        filename=row["filename"],
        extraction_manifest=json.loads(row["extraction_manifest_json"]),
        custody=row["custody"],
        privacy_classification=row["privacy_classification"],
        acquired_at_ms=row["acquired_at_ms"],
        created_at_ms=row["created_at_ms"],
    )


if TYPE_CHECKING:
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
