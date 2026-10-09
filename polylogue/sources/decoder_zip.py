"""ZIP validation and extraction helpers for source ingestion."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Callable, Collection, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from polylogue.archive.artifact_taxonomy import ArtifactClassification, ArtifactKind, classify_artifact_path
from polylogue.archive.zip_admission import (
    _ZIP_READ_CHUNK_SIZE,
    ZIP_JSON_SUFFIXES,
    ZipAdmission,
    open_zip_entry,
)
from polylogue.core.content_identity import ContentIdentityRefusal, stream_payload_content_identity
from polylogue.core.enums import Provider
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate, MemberAddressingMode
from polylogue.logging import WARNING, emit, get_logger
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload

from .assembly import SidecarData
from .cursor import _record_cursor_failure
from .parsers.base import ParsedSession, RawSessionData

if TYPE_CHECKING:
    from .prepared_jsonl import PreparedJsonl
    from .source_staging import SourceInputBinding

logger = get_logger(__name__)


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
        entries: Iterable[zipfile.ZipInfo],
        *,
        allowed_suffixes: Collection[str] | None = None,
        allowed_path: Callable[[str], bool] | None = None,
        on_unselected: Callable[[zipfile.ZipInfo, str], None] | None = None,
    ) -> Iterable[zipfile.ZipInfo]:
        """Yield safe, relevant entries and record failures in cursor state.

        ``allowed_suffixes`` selects which member kinds a caller needs, while
        this validator remains the sole owner of the security checks. The
        yielded object is the exact central-directory ``ZipInfo`` that was
        admitted. Callers must pass it through to ``open_zip_entry``;
        reopening by filename can select a different duplicate member.

        ``on_unselected`` reports relevance decisions. Actual read, decode,
        CRC and physical storage failures are reported by their consumers.
        """

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
            on_unselected=on_unselected,
        )


@contextmanager
def prepare_zip_entry(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    provider: Provider,
    source_path: str,
    profile_identity: str | None = None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None,
) -> Iterator[PreparedJsonl]:
    """Prepare the complete exact member with the existing streamed parser owner."""
    from .dispatch import is_stream_record_provider
    from .prepared_jsonl import prepare_jsonl_blob

    with TemporaryDirectory(prefix="polylogue-zip-entry-") as directory:
        root = Path(directory)
        member = root / "input"
        digest = hashlib.sha256()
        with open_zip_entry(zf, info) as source, member.open("wb") as destination:
            while chunk := source.read(_ZIP_READ_CHUNK_SIZE):
                destination.write(chunk)
                digest.update(chunk)
        artifact = prepare_jsonl_blob(
            str(member),
            source_path,
            provider.value,
            Path(info.filename).stem,
            is_stream=is_stream_record_provider(info.filename, provider),
            profile_identity=profile_identity,
            shard_directory=str(root),
            strict_jsonl_records=True,
            source_sha256=digest.hexdigest(),
            captured_zip_coordinate=captured_zip_coordinate,
        )
        try:
            yield artifact
        finally:
            artifact.discard()


def zip_entry_session_artifact(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    provider: Provider,
    profile_identity: str | None = None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None,
) -> ArtifactClassification | None:
    """Override a weak path rule only with complete positive parsed evidence."""
    from .dispatch import is_stream_record_provider
    from .prepared_jsonl import PreparedJsonl

    if not info.filename.lower().endswith(ZIP_JSON_SUFFIXES):
        return None
    with prepare_zip_entry(
        zf,
        info,
        provider=provider,
        source_path=info.filename,
        profile_identity=profile_identity,
        captured_zip_coordinate=captured_zip_coordinate,
    ) as prepared:
        assert isinstance(prepared, PreparedJsonl)
        if prepared.error is not None or prepared.deferred or prepared.blob_hash is None:
            return None
        found = False
        for _session in prepared.iter_sessions():
            found = True
    if not found:
        return None
    stream = is_stream_record_provider(info.filename, provider)
    return ArtifactClassification(
        provider=provider,
        kind=ArtifactKind.SESSION_RECORD_STREAM if stream else ArtifactKind.SESSION_DOCUMENT,
        parse_as_session=True,
        schema_eligible=True,
        default_priority=120,
        reason="complete member has positive conversational evidence",
    )


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
    source_binding: SourceInputBinding | None = None,
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
    from polylogue.storage.blob_publication import flush_blob_publications, publication_receipt_id, require_published
    from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError

    from .acquisition_boundary import (
        capture_bound_stream,
        open_bound_container,
        open_bound_member,
        release_captures_on_refusal,
        release_refused_capture,
    )
    from .cursor import _ParseContext
    from .dispatch import GROUP_PROVIDERS, ForeignOriginContentError, bound_location_provider
    from .emitter import _SessionEmitter
    from .origin_specs import path_declaration_refuses_session

    resolved_sidecar_data: SidecarData = sidecar_data if sidecar_data is not None else {}

    store = blob_store or BlobStore(blob_root or blob_store_root())

    validator = ZipEntryValidator(
        provider_hint,
        cursor_state=cursor_state,
        zip_path=zip_path,
    )

    from polylogue.config import Source
    from polylogue.core.provider_identity import captured_hermes_profile_key

    from .parsers.hermes_identity import CapturedHermesProfile
    from .source_acquisition_components import (
        ZipEntryReadContext,
        _captured_zip_record,
        zip_acquisition_fingerprint,
        zip_member_admission,
    )
    from .source_staging import bind_source_input

    with ExitStack() as custody:
        binding = source_binding or custody.enter_context(bind_source_input(zip_path))
        physical = custody.enter_context(open_bound_container(store, binding))
        zf = custody.enter_context(zipfile.ZipFile(physical.stream))
        entries = zf.infolist()
        admission = custody.enter_context(
            zip_member_admission(zf, zip_path, entries, provider_hint, container_blob_hash=physical.blob_hash)
        )

        def member_skipped(entry: zipfile.ZipInfo, reason: str) -> None:
            # Every member the parse route does not turn into sessions gets
            # the same typed disposition the acquisition route records.
            emit(
                "sources.zip.member_skipped",
                outcome="skipped",
                reason=reason,
                source_path=f"{zip_path}:{entry.filename}",
            )

        for entry_ordinal, info in enumerate(entries):
            if (
                next(
                    iter(
                        validator.filter_entries(
                            (info,), allowed_path=admission.allowed_path, on_unselected=member_skipped
                        )
                    ),
                    None,
                )
                is None
            ):
                continue
            name = info.filename
            entry_provider_hint = admission.entry_provider_hint(info, entry_ordinal=entry_ordinal)
            member_context = ZipEntryReadContext(
                source=Source(name=provider_hint.value, path=zip_path),
                zip_path=zip_path,
                entry=info,
                file_mtime=file_mtime,
                provider_hint=entry_provider_hint,
                blob_store=store,
                bound_provider=bound_location_provider(provider_hint),
                captured_input_identity=binding.captured_identity,
                container_blob_hash=physical.blob_hash,
                decoder_fingerprint=zip_acquisition_fingerprint(provider_hint),
                entry_ordinal=entry_ordinal,
            )
            namespace = binding.captured_identity.member_profile_identity(name)
            profile = (
                None
                if namespace is None
                else CapturedHermesProfile(
                    namespace[0],
                    captured_hermes_profile_key(namespace[0]),
                    namespace[1],
                )
            )
            path_classification = classify_artifact_path(name, provider=entry_provider_hint)
            session_artifact: ArtifactClassification | None = None
            if path_classification is not None and not path_classification.parse_as_session:
                if path_declaration_refuses_session(entry_provider_hint, name):
                    # Declared raw-only evidence is never probed for sessions.
                    member_skipped(info, "declared raw-only artifact")
                    continue
                from .source_acquisition_components import captured_zip_member_coordinate

                session_coordinate = captured_zip_member_coordinate(
                    member_context.captured_input_identity,
                    entry_name=name,
                    entry_ordinal=entry_ordinal,
                    split_index=0,
                    addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
                    container_blob_hash=member_context.container_blob_hash,
                    decoder_fingerprint=member_context.decoder_fingerprint,
                )
                session_artifact = zip_entry_session_artifact(
                    zf,
                    info,
                    provider=entry_provider_hint,
                    captured_zip_coordinate=session_coordinate,
                )
                if session_artifact is None:
                    member_skipped(info, "non-session path without positive session evidence")
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
                # The archive's own location binds; an inbox export (UNKNOWN)
                # classifies its members.
                bound_provider=bound_location_provider(provider_hint),
            )
            emitter = _SessionEmitter(ctx)
            precomputed_raw: RawSessionData | None = None
            try:
                if capture_raw and entry_should_group:
                    # The complete member boundary refuses foreign records
                    # before publishing grouped bytes.
                    with open_bound_member(zf, info, ctx.bound_provider, profile_identity=profile) as handle:
                        blob_hash, blob_size = capture_bound_stream(store, handle)
                    try:
                        with store.open(blob_hash) as stored_handle:
                            content_identity = stream_payload_content_identity(stored_handle)
                    except ContentIdentityRefusal:
                        # No raw record will reference the refused member, so
                        # its queued publication must not be reserved later.
                        release_refused_capture(store, blob_hash, publication_receipt_id(store, blob_hash))
                        raise
                    receipt_id = publication_receipt_id(store, blob_hash)
                    flush_blob_publications(store)
                    require_published(store, blob_hash, source_path=f"{zip_path}:{name}")
                    precomputed_raw = RawSessionData(
                        raw_bytes=b"",
                        source_path=f"{zip_path}:{name}",
                        source_index=None,
                        addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
                        content_identity=content_identity,
                        file_mtime=file_mtime,
                        provider_hint=entry_provider_hint,
                        blob_hash=blob_hash,
                        blob_size=blob_size,
                        blob_publication_receipt_id=receipt_id,
                    )
                with release_captures_on_refusal(store) as captures:
                    if precomputed_raw is not None and precomputed_raw.blob_hash is not None:
                        captures.append((precomputed_raw.blob_hash, precomputed_raw.blob_publication_receipt_id))
                    emitted_sessions = 0
                    with open_bound_member(zf, info, ctx.bound_provider, profile_identity=profile) as handle:
                        for raw, session in emitter.emit(
                            handle,
                            name,
                            precomputed_raw=precomputed_raw,
                            session_artifact=session_artifact,
                        ):
                            if raw is not None:
                                if raw.addressing_mode is None:
                                    raw = raw.model_copy(update={"addressing_mode": MemberAddressingMode.WHOLE_MEMBER})
                                raw = _captured_zip_record(raw, member_context)
                            emitted_sessions += 1
                            yield raw, session
                    if not emitted_sessions:
                        member_skipped(info, "member parsed to no sessions")
            except ContentExcisedError as exc:
                # An excised member is skipped; the archive's other members
                # still ingest. Not a cursor failure: nothing to retry.
                emit(
                    "sources.zip.member_content_excised",
                    outcome="skipped",
                    reason="content_excised",
                    entry=name,
                    blob_hash=exc.blob_hash.hex(),
                )
                continue
            except ContentIdentityRefusal as exc:
                # A refused member is a recorded gap; the rest of the ZIP
                # is still acquired.
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
            except ForeignOriginContentError as exc:
                # A refused member is recorded on its own; admissible siblings
                # in the same archive are still parsed.
                emit(
                    "sources.acquisition.foreign_origin_refused",
                    level=WARNING,
                    outcome="refused",
                    source_path=str(f"{zip_path}:{name}"),
                    reason=f"{exc.code}: {exc}",
                )
                _record_cursor_failure(cursor_state, f"{zip_path}:{name}", f"{exc.code}: {exc}")
                continue


__all__ = [
    "_ZIP_READ_CHUNK_SIZE",
    "ZIP_JSON_SUFFIXES",
    "ZipEntryValidator",
    "open_zip_entry",
    "process_zip",
    "zip_entry_session_artifact",
    "zip_entry_provider_hint",
]
