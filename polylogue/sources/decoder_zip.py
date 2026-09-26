"""ZIP validation and extraction helpers for source ingestion."""

from __future__ import annotations

import zipfile
from collections.abc import Callable, Collection, Iterable
from pathlib import Path

from polylogue.archive.artifact_taxonomy import ArtifactClassification, classify_artifact_path
from polylogue.archive.zip_admission import (
    _ZIP_READ_CHUNK_SIZE,
    MAX_AGGREGATE_UNCOMPRESSED_SIZE,
    MAX_COMPRESSION_RATIO,
    MAX_UNCOMPRESSED_SIZE,
    ZIP_JSON_SUFFIXES,
    ZipAdmission,
    ZipBombError,
    open_bounded_zip_entry,
)
from polylogue.core.content_identity import STRUCTURAL_IDENTITY_MAX_BYTES, bounded_payload_content_identity
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDecodeError
from polylogue.core.json import loads as json_loads
from polylogue.core.raw_coordinates import MemberAddressingMode
from polylogue.logging import WARNING, emit, get_logger
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload

from .assembly import SidecarData
from .cursor import _record_cursor_failure
from .parsers.base import ParsedSession, RawSessionData

logger = get_logger(__name__)

# A classification probe asks only whether a member's decoded content overrides
# a non-session path rule. It needs the head of a conversation document, not the
# archival ceiling: reusing ``MAX_UNCOMPRESSED_SIZE`` here turns one ``read()``
# into a 10 GiB allocation for a member a 1000:1 compression ratio lets a ~10 MB
# download declare. Real provider conversation documents sit far below this.
ZIP_PROBE_MAX_BYTES = 256 * 1024 * 1024


def is_declared_artifact_path(source_path: str) -> bool:
    """Return whether any provider declaration owns this archive path."""
    return any(
        provider is not Provider.UNKNOWN and artifact_rule_for_path(provider, source_path) is not None
        for provider in Provider
    )


def declared_artifact_provider(source_path: str) -> Provider | None:
    """Return the single provider whose declaration owns this archive path.

    ``None`` when no declaration owns it, or when more than one family claims
    the same path -- an ambiguous claim is not evidence of a provider.
    """
    owners = {
        provider
        for provider in Provider
        if provider is not Provider.UNKNOWN and artifact_rule_for_path(provider, source_path) is not None
    }
    if len(owners) != 1:
        return None
    return next(iter(owners))


def provider_detection_path(source_path: str) -> bool:
    """Exclude declaration-owned non-session evidence from provider sniffing."""
    rules = [
        rule
        for provider in Provider
        if provider is not Provider.UNKNOWN
        if (rule := artifact_rule_for_path(provider, source_path)) is not None
    ]
    # An explicit raw-only declaration must keep an overflow sidecar out of
    # sniffing even if another provider has a broad session-path pattern.
    return not rules or all(rule.parse_policy == "session" for rule in rules)


class ZipEntryValidator:
    """Validate ZIP entries for security and relevance."""

    __slots__ = ("_cursor_state", "_zip_path", "_admission", "_provider")

    def __init__(
        self,
        provider_hint: str | Provider,
        *,
        cursor_state: CursorStatePayload | None,
        zip_path: Path,
    ) -> None:
        self._provider = Provider.from_string(provider_hint)
        self._cursor_state = cursor_state
        self._zip_path = zip_path
        self._admission = ZipAdmission(zip_path=zip_path)

    def filter_entries(
        self,
        entries: list[zipfile.ZipInfo],
        *,
        allowed_suffixes: Collection[str] | None = None,
        allowed_path: Callable[[str], bool] | None = None,
        on_rejected: Callable[[zipfile.ZipInfo, str], None] | None = None,
        on_unselected: Callable[[zipfile.ZipInfo, str], None] | None = None,
    ) -> Iterable[zipfile.ZipInfo]:
        """Yield safe, relevant entries and record failures in cursor state.

        ``allowed_suffixes`` selects which member kinds a caller needs, while
        this validator remains the sole owner of the security checks. The
        yielded object is the exact central-directory ``ZipInfo`` that was
        admitted. Callers must pass it through to ``open_bounded_zip_entry``;
        reopening by filename can select a different duplicate member.

        ``on_rejected`` lets read-only surfaces report the same admission
        decisions without duplicating the security checks. ``on_unselected``
        is the separate relevance channel: a member neither matching a
        requested suffix nor owned by an artifact declaration. It is not a
        cursor failure -- an ordinary export ships members this filter is not
        asking for -- but a caller that must account for every physical member
        needs to see it instead of having it silently disappear.
        """

        def reject(info: zipfile.ZipInfo, reason: str) -> None:
            _record_cursor_failure(
                self._cursor_state,
                f"{self._zip_path}:{info.filename}",
                reason.capitalize() if reason.startswith("aggregate") else reason,
            )
            if on_rejected is not None:
                on_rejected(info, reason)

        infer_declared_paths = allowed_suffixes is None and allowed_path is None
        if allowed_suffixes is None:
            allowed_suffixes = ZIP_JSON_SUFFIXES
        if infer_declared_paths:

            def allowed_path(name: str) -> bool:
                return artifact_rule_for_path(self._provider, name) is not None

        yield from self._admission.filter_entries(
            entries,
            allowed_suffixes=allowed_suffixes,
            allowed_path=allowed_path,
            on_rejected=reject,
            on_unselected=on_unselected,
        )


def zip_entry_session_artifact(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    provider: Provider,
) -> ArtifactClassification | None:
    """Decode a member before applying a terminal artifact path rule."""
    from polylogue.archive.raw_payload.decode import (
        JSONL_RECORD_INSPECTION_BYTES,
        scan_jsonl_session_artifact,
    )

    lower_name = info.filename.lower()
    if lower_name.endswith((".jsonl", ".jsonl.txt", ".ndjson")):
        with open_bounded_zip_entry(zf, info) as handle:
            scan = scan_jsonl_session_artifact(
                handle,
                provider=provider,
                max_record_bytes=JSONL_RECORD_INSPECTION_BYTES,
            )
        if scan.artifact is None:
            return None
        if scan.sample:
            return scan.artifact
        # Unresolved evidence, not positive evidence. ``scan`` reaches here
        # only through its oversized-record retention branch: no record was
        # small enough to inspect, so it synthesised a parse-as-session
        # classification for the streaming-parser providers. That retention
        # rule is for a raw whose *only* classifier is a weak path heuristic;
        # this caller asks a narrower question -- may decoded content override
        # an OriginSpec-declared terminal artifact rule -- and an inspection
        # that read nothing has not answered it. Overriding here reclassifies
        # the member as a session, which the positive-conversational-evidence
        # refusal (polylogue-9ykn) then drops entirely, losing the bytes the
        # artifact rule would have retained. Same posture as the ZIP probe
        # ceiling above: the path rule stands and the skipped inspection is
        # named rather than silently reclassifying the member.
        emit(
            "sources.zip.artifact_probe_unbounded",
            level=WARNING,
            outcome="degraded",
            reason="record_size_exceeded",
            entry=info.filename,
            declared_bytes=info.file_size,
            probe_ceiling_bytes=JSONL_RECORD_INSPECTION_BYTES,
        )
        return None
    if not lower_name.endswith(".json"):
        return None
    try:
        # This is a classification probe, not archival retention, so it gets
        # its own small ceiling rather than reusing the 10 GiB per-member
        # archival cap as an in-memory limit. Admission allows a 1000:1
        # compression ratio, so a ~10 MB crafted member could otherwise make
        # this single ``read()`` allocate 10 GiB before any parse.
        with open_bounded_zip_entry(zf, info, max_bytes=ZIP_PROBE_MAX_BYTES) as handle:
            payload = json_loads(handle.read())
    except JSONDecodeError:
        return None
    except ZipBombError:
        # Not a silent reclassification: the path rule stands, and the event
        # names the member whose content evidence was never examined. The
        # structured event is the whole report -- a parallel prose log would
        # duplicate it and add a `legacy-prose-logging` match.
        emit(
            "sources.zip.artifact_probe_unbounded",
            level=WARNING,
            outcome="degraded",
            reason="probe_size_exceeded",
            entry=info.filename,
            declared_bytes=info.file_size,
            probe_ceiling_bytes=ZIP_PROBE_MAX_BYTES,
        )
        return None
    # Deliberately omit source_path. The caller is asking whether decoded
    # content can override a non-session path rule, so reapplying that rule
    # here would make the evidence check circular.
    from polylogue.archive.artifact_taxonomy import classify_artifact

    artifact = classify_artifact(payload, provider=provider)
    return artifact if artifact.parse_as_session else None


def zip_entry_provider_hint(entry_name: str, fallback_provider: str | Provider) -> Provider:
    del entry_name
    return Provider.from_string(fallback_provider)


def process_zip(
    zip_path: Path,
    *,
    provider_hint: Provider,
    should_group: bool,
    file_mtime: str | None,
    capture_raw: bool,
    cursor_state: CursorStatePayload | None,
    blob_root: Path | None = None,
    blob_store: BlobStore | None = None,
    sidecar_data: SidecarData | None = None,
) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
    """Process a ZIP file, yielding sessions from its entries.

    ``sidecar_data`` (bd polylogue-8ac0) threads the source-scan-level
    provider assembly sidecars (e.g. ChatGPT's ``chatgpt_asset_index``/
    ``chatgpt_asset_blobs``, discovered once per source by
    ``_setup_source_walk`` before any entry is parsed) into every entry's
    ``_ParseContext`` so ``_SessionEmitter.emit``'s ``enrich_session`` hook
    actually fires for ZIP-bundle sources. Without it, every entry got an
    empty sidecar mapping and provider-assembly enrichment silently never ran
    for ZIP-shaped sources -- the common shape for a GDPR/Takeout export.
    """
    del should_group

    from polylogue.paths import blob_store_root
    from polylogue.storage.blob_publication import flush_blob_publications, publication_receipt_id

    from .cursor import _ParseContext
    from .dispatch import GROUP_PROVIDERS
    from .emitter import _SessionEmitter

    resolved_sidecar_data: SidecarData = sidecar_data if sidecar_data is not None else {}

    store = blob_store or BlobStore(blob_root or blob_store_root())

    validator = ZipEntryValidator(
        provider_hint,
        cursor_state=cursor_state,
        zip_path=zip_path,
    )

    with zipfile.ZipFile(zip_path) as zf:
        for info in validator.filter_entries(zf.infolist()):
            name = info.filename
            entry_provider_hint = zip_entry_provider_hint(name, provider_hint)
            path_classification = classify_artifact_path(name, provider=entry_provider_hint)
            session_artifact: ArtifactClassification | None = None
            if path_classification is not None and not path_classification.parse_as_session:
                session_artifact = zip_entry_session_artifact(zf, info, provider=entry_provider_hint)
                if session_artifact is None:
                    continue
            entry_should_group = entry_provider_hint in GROUP_PROVIDERS
            ctx = _ParseContext(
                provider_hint=entry_provider_hint,
                should_group=entry_should_group,
                source_path_str=f"{zip_path}:{name}",
                fallback_id=zip_path.stem,
                file_mtime=file_mtime,
                capture_raw=capture_raw,
                sidecar_data=resolved_sidecar_data,
            )
            emitter = _SessionEmitter(ctx)
            precomputed_raw: RawSessionData | None = None
            try:
                if capture_raw and entry_should_group:
                    # ``open_bounded_zip_entry`` enforces a hard real-byte
                    # ceiling during decompression, independent of the
                    # entry's (forgeable) declared header sizes.
                    with open_bounded_zip_entry(zf, info) as handle:
                        blob_hash, blob_size = store.write_from_fileobj(handle)
                    with store.open(blob_hash) as stored_handle:
                        content_identity, identity_skipped = bounded_payload_content_identity(
                            stored_handle, size=blob_size, byte_digest=blob_hash
                        )
                    if identity_skipped is not None:
                        emit(
                            "sources.zip.structural_identity_skipped",
                            level=WARNING,
                            outcome="degraded",
                            reason=identity_skipped,
                            entry=name,
                            blob_bytes=blob_size,
                            identity_ceiling_bytes=STRUCTURAL_IDENTITY_MAX_BYTES,
                        )
                    receipt_id = publication_receipt_id(store, blob_hash)
                    flush_blob_publications(store)
                    precomputed_raw = RawSessionData(
                        raw_bytes=b"",
                        source_path=f"{zip_path}:{name}",
                        source_index=None,
                        addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
                        content_identity=content_identity,
                        content_identity_skipped_reason=identity_skipped,
                        file_mtime=file_mtime,
                        provider_hint=entry_provider_hint,
                        blob_hash=blob_hash,
                        blob_size=blob_size,
                        blob_publication_receipt_id=receipt_id,
                    )
                with open_bounded_zip_entry(zf, info) as handle:
                    yield from emitter.emit(
                        handle,
                        name,
                        precomputed_raw=precomputed_raw,
                        session_artifact=session_artifact,
                    )
            except ZipBombError as exc:
                logger.warning(
                    "Skipping ZIP entry %s in %s: %s",
                    name,
                    zip_path,
                    exc,
                )
                _record_cursor_failure(
                    cursor_state,
                    f"{zip_path}:{name}",
                    str(exc),
                )
                continue


__all__ = [
    "_ZIP_READ_CHUNK_SIZE",
    "MAX_AGGREGATE_UNCOMPRESSED_SIZE",
    "MAX_COMPRESSION_RATIO",
    "MAX_UNCOMPRESSED_SIZE",
    "ZIP_JSON_SUFFIXES",
    "ZipBombError",
    "ZipEntryValidator",
    "open_bounded_zip_entry",
    "process_zip",
    "zip_entry_session_artifact",
    "zip_entry_provider_hint",
]
