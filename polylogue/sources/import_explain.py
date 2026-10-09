"""Import explain payload construction over the existing parser stack."""

from __future__ import annotations

import json
import os
import sqlite3
import zipfile
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from polylogue.archive.artifact_taxonomy import ArtifactClassification, classify_artifact, classify_artifact_path
from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.evidence import Measured, Unavailable
from polylogue.core.json import JSONValue
from polylogue.core.provider_identity import captured_hermes_profile_key
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate, MemberAddressingMode
from polylogue.core.sources import origin_from_provider
from polylogue.sources.acquisition_boundary import open_bound_container
from polylogue.sources.decoder_zip import (
    ZipEntryValidator,
    open_zip_entry,
    prepare_zip_entry,
    zip_entry_session_artifact,
)
from polylogue.sources.decoders import _decode_json_bytes, _iter_json_stream
from polylogue.sources.dispatch import (
    GROUP_PROVIDERS,
    detect_provider,
    is_jsonl_source_path,
    is_stream_record_provider,
    parse_payload,
    parse_stream_payload,
)
from polylogue.sources.parsers import hermes_spans, hermes_state
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.source_acquisition_components import (
    captured_zip_member_coordinate,
    zip_acquisition_fingerprint,
    zip_member_admission,
)
from polylogue.sources.source_staging import bind_source_input
from polylogue.sources.source_walk import _resolve_source_paths
from polylogue.sources.sqlite_inspection import SQLiteInspection, inspect_sqlite_source
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.source_write import read_capture_mode_resolution
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import read_frame
from polylogue.storage.tier_access import capture_sqlite_read
from polylogue.surfaces.payloads import (
    ImportDetectorEvidencePayload,
    ImportExplainEntryPayload,
    ImportExplainPayload,
    ImportFidelityCapabilityPayload,
    ImportFidelityDeclarationPayload,
    ImportProducedRowsPayload,
    ImportSkippedRowPayload,
)


def explain_import_path(
    path: Path,
    *,
    source_name: str = "unknown",
    limit: int = 100,
    checkpoint: Callable[[], None] = lambda: None,
) -> ImportExplainPayload:
    """Return a bounded import explanation for a file or directory.

    This is intentionally non-mutating: it reads local bytes, runs the same
    detector/parser path used by import, and reports what would be produced
    without staging daemon work or writing raw blobs.
    """

    checkpoint()
    resolved = path.expanduser().resolve()
    entries: list[ImportExplainEntryPayload] = []
    skipped: list[ImportSkippedRowPayload] = []
    caveats: list[str] = []

    if not resolved.exists():
        skipped.append(ImportSkippedRowPayload(reason="path does not exist", source_path=str(resolved)))
        return _envelope(resolved, entries=entries, skipped=skipped, caveats=caveats)

    for candidate in _candidate_paths(resolved, source_name=source_name):
        checkpoint()
        if len(entries) >= limit:
            caveats.append(f"entry limit {limit} reached; remaining files omitted")
            break
        entry = _explain_file(candidate, provider_hint=Provider.from_string(source_name))
        entries.append(entry)
        skipped.extend(entry.skipped)

    if not entries and not skipped:
        skipped.append(ImportSkippedRowPayload(reason="no supported import files found", source_path=str(resolved)))

    return _envelope(resolved, entries=entries, skipped=skipped, caveats=caveats)


def explain_import_archive(
    archive_root: Path,
    *,
    raw_ref: str | None = None,
    source_path: str | None = None,
    limit: int = 100,
    redact_paths: bool = True,
) -> ImportExplainPayload:
    """Return an import explanation from already-archived source/index evidence."""

    if raw_ref is None and source_path is None:
        raise ValueError("raw_ref or source_path is required for archive import explain")

    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    query_label = _archive_query_label(raw_ref=raw_ref, source_path=source_path, redact_paths=redact_paths)
    entries: list[ImportExplainEntryPayload] = []
    skipped: list[ImportSkippedRowPayload] = []
    caveats: list[str] = ["archive-backed explanation; raw bytes are omitted."]

    if not source_db.exists():
        skipped.append(ImportSkippedRowPayload(reason="source tier is unavailable", raw_ref=raw_ref))
        return _envelope(Path(query_label), entries=entries, skipped=skipped, caveats=caveats)

    raw_id = _normalize_raw_ref(raw_ref) if raw_ref is not None else None
    with ExitStack() as stack:
        source_frame = stack.enter_context(
            read_frame(source_db, timeout_class="background-read", tier=ArchiveTier.SOURCE)
        )
        source_conn = source_frame.connection
        source_conn.row_factory = sqlite3.Row
        raw_rows = _select_raw_session_rows(source_conn, raw_id=raw_id, source_path=source_path, limit=limit)
        if not raw_rows:
            skipped.append(
                ImportSkippedRowPayload(
                    reason="no archived raw session matched",
                    source_path=_display_source_path(source_path, redact=redact_paths),
                    raw_ref=raw_ref,
                )
            )
            return _envelope(Path(query_label), entries=entries, skipped=skipped, caveats=caveats)

        index_conn: sqlite3.Connection | None = None
        if index_db.exists():
            index_frame = stack.enter_context(
                read_frame(index_db, timeout_class="background-read", tier=ArchiveTier.INDEX)
            )
            index_conn = index_frame.connection
            index_conn.row_factory = sqlite3.Row
            # Derived projections require identity validation as well as the
            # ordinary tier-version check performed while opening the frame.
            from polylogue.storage.sqlite.connection_profile import assert_tier_schema_supported

            assert_tier_schema_supported(index_conn, index_db, ArchiveTier.INDEX)
        else:
            caveats.append("index tier is unavailable; produced archive row counts are incomplete")
        for row in raw_rows:
            artifact_rows = _select_artifact_rows(source_conn, raw_id=str(row["raw_id"]))
            entry = _archive_entry_from_rows(
                row,
                artifact_rows=artifact_rows,
                source_conn=source_conn,
                index_conn=index_conn,
                redact_paths=redact_paths,
            )
            entries.append(entry)
            skipped.extend(entry.skipped)

    if len(raw_rows) >= limit:
        caveats.append(f"entry limit {limit} reached; remaining archived raw rows omitted")
    return _envelope(Path(query_label), entries=entries, skipped=skipped, caveats=caveats)


def _candidate_paths(path: Path, *, source_name: str) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    yield from _resolve_source_paths(Source(name=source_name, path=path))


def _envelope(
    path: Path,
    *,
    entries: list[ImportExplainEntryPayload],
    skipped: list[ImportSkippedRowPayload],
    caveats: list[str],
) -> ImportExplainPayload:
    produced = ImportProducedRowsPayload(
        sessions=sum(entry.produced.sessions for entry in entries),
        messages=sum(entry.produced.messages for entry in entries),
        blocks=sum(entry.produced.blocks for entry in entries),
        actions=sum(entry.produced.actions for entry in entries),
        raw_records=sum(entry.produced.raw_records for entry in entries),
        session_refs=tuple(ref for entry in entries for ref in entry.produced.session_refs),
    )
    return ImportExplainPayload(
        source_path=str(path),
        entries=tuple(entries),
        produced=produced,
        skipped=tuple(skipped),
        caveats=tuple(caveats),
    )


def _archive_entry_from_rows(
    row: sqlite3.Row,
    *,
    artifact_rows: tuple[sqlite3.Row, ...],
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection | None,
    redact_paths: bool,
) -> ImportExplainEntryPayload:
    raw_id = str(row["raw_id"])
    source_path = _display_source_path(str(row["source_path"]), redact=redact_paths)
    parse_error = _optional_text(row["parse_error"])
    validation_error = _optional_text(row["validation_error"])
    detection_warnings = _loads_warning_tuple(_optional_text(row["detection_warnings_json"]))
    artifact_kind = _archive_artifact_kind(artifact_rows)
    produced = _archive_produced_rows(index_conn, raw_id)
    skipped: list[ImportSkippedRowPayload] = []
    caveats: list[str] = []

    if parse_error is not None:
        skipped.append(
            ImportSkippedRowPayload(
                reason=f"parse error: {parse_error}",
                source_path=source_path,
                raw_ref=f"raw:{raw_id}",
            )
        )
    if validation_error is not None:
        caveats.append(f"validation error: {validation_error}")
    for artifact in artifact_rows:
        decode_error = _optional_text(artifact["decode_error"])
        if decode_error is not None:
            skipped.append(
                ImportSkippedRowPayload(
                    reason=f"decode error: {decode_error}",
                    source_path=source_path,
                    raw_ref=f"raw:{raw_id}",
                )
            )
    if redact_paths and source_path != str(row["source_path"]):
        caveats.append("source path redacted for this surface")
    if artifact_rows:
        artifact_evidence = tuple(
            _evidence(
                f"source.raw_artifacts.{artifact['artifact_id']}",
                matched=bool(artifact["parse_as_session"]),
                reason=str(artifact["classification_reason"]),
            )
            for artifact in artifact_rows
        )
    else:
        artifact_evidence = (_evidence("source.raw_artifacts", matched=False, reason="no artifact row recorded"),)

    # Read the full observation set, not `raw_sessions.capture_mode`: that
    # column caches only the first-known mode, so a byte-identical payload
    # acquired twice by different mechanisms would report one of them as if
    # it were the only fact on record (polylogue-buns AC2 / polylogue-7xg00).
    capture_resolution = read_capture_mode_resolution(source_conn, raw_id)
    if capture_resolution.status == "ambiguous":
        caveats.append(
            "capture mode is ambiguous: "
            + ", ".join(mode.value for mode in capture_resolution.modes)
            + " were all observed for these bytes"
        )

    return ImportExplainEntryPayload(
        raw_ref=f"raw:{raw_id}",
        source_path=source_path,
        artifact_kind=artifact_kind,
        provider_hint=str(row["origin"]),
        detected_origin=str(row["origin"]),
        detected_provider=None,
        capture_mode_status=capture_resolution.status,
        capture_modes=tuple(mode.value for mode in capture_resolution.modes),
        detector="source.raw_sessions",
        detector_evidence=(
            _evidence("source.raw_sessions", matched=True, reason=f"raw_id={raw_id}"),
            *artifact_evidence,
        ),
        parser="archive source/index evidence",
        parser_mode="archived_raw_session",
        schema_resolution=_schema_resolution(row),
        produced=produced,
        skipped=tuple(skipped),
        caveats=tuple(caveats),
        raw_evidence_refs=(f"raw:{raw_id}",)
        + tuple(f"raw-artifact:{artifact['artifact_id']}" for artifact in artifact_rows),
        normalization_warnings=detection_warnings,
    )


def _select_raw_session_rows(
    conn: sqlite3.Connection,
    *,
    raw_id: str | None,
    source_path: str | None,
    limit: int,
) -> tuple[sqlite3.Row, ...]:
    clauses: list[str] = []
    params: list[object] = []
    if raw_id is not None:
        clauses.append("raw_id = ?")
        params.append(raw_id)
    if source_path is not None:
        clauses.append("source_path = ?")
        params.append(source_path)
    where = " AND ".join(clauses) if clauses else "1 = 1"
    return tuple(
        conn.execute(
            f"""
            SELECT
                raw_id, origin, native_id, source_path, source_index, blob_size,
                acquired_at_ms, parsed_at_ms, parse_error, validated_at_ms,
                validation_status, validation_error, detection_warnings_json
            FROM raw_sessions
            WHERE {where}
            ORDER BY source_path, source_index, raw_id
            LIMIT ?
            """,
            (*params, max(0, limit)),
        ).fetchall()
    )


def _select_artifact_rows(conn: sqlite3.Connection, *, raw_id: str) -> tuple[sqlite3.Row, ...]:
    return tuple(
        conn.execute(
            """
            SELECT artifact_id, artifact_kind, classification_reason, parse_as_session,
                   support_status, decode_error
            FROM raw_artifacts
            WHERE raw_id = ?
            ORDER BY source_index, artifact_id
            """,
            (raw_id,),
        ).fetchall()
    )


def _archive_produced_rows(conn: sqlite3.Connection | None, raw_id: str) -> ImportProducedRowsPayload:
    if conn is None:
        return ImportProducedRowsPayload(raw_records=1)
    session_rows = tuple(
        conn.execute(
            """
            SELECT session_id, message_count, tool_use_count
            FROM sessions
            WHERE raw_id = ?
            ORDER BY session_id
            """,
            (raw_id,),
        ).fetchall()
    )
    if not session_rows:
        return ImportProducedRowsPayload(raw_records=1)
    session_ids = tuple(str(row["session_id"]) for row in session_rows)
    placeholders = ",".join("?" for _ in session_ids)
    messages = int(
        conn.execute(f"SELECT COUNT(*) FROM messages WHERE session_id IN ({placeholders})", session_ids).fetchone()[0]
    )
    blocks = int(
        conn.execute(f"SELECT COUNT(*) FROM blocks WHERE session_id IN ({placeholders})", session_ids).fetchone()[0]
    )
    actions = int(
        conn.execute(f"SELECT COUNT(*) FROM actions WHERE session_id IN ({placeholders})", session_ids).fetchone()[0]
    )
    return ImportProducedRowsPayload(
        sessions=len(session_rows),
        messages=messages,
        blocks=blocks,
        actions=actions,
        raw_records=1,
        session_refs=tuple(f"session:{session_id}" for session_id in session_ids),
    )


def _archive_artifact_kind(rows: tuple[sqlite3.Row, ...]) -> str:
    kinds = tuple(dict.fromkeys(str(row["artifact_kind"]) for row in rows if row["artifact_kind"] is not None))
    if not kinds:
        return "raw_session"
    if len(kinds) == 1:
        return kinds[0]
    return "mixed_raw_artifacts"


def _schema_resolution(row: sqlite3.Row) -> str | None:
    status = _optional_text(row["validation_status"])
    if status is not None:
        return status
    if row["validated_at_ms"] is not None:
        return "validated"
    if row["parsed_at_ms"] is not None:
        return "parsed"
    return None


def _normalize_raw_ref(raw_ref: str) -> str:
    return raw_ref.removeprefix("raw:")


def _archive_query_label(*, raw_ref: str | None, source_path: str | None, redact_paths: bool) -> str:
    if raw_ref is not None:
        return f"archive:{raw_ref}"
    return _display_source_path(source_path or "archive:source-path", redact=redact_paths) or "archive:source-path"


def _display_source_path(raw_path: str | None, *, redact: bool) -> str | None:
    if raw_path is None:
        return None
    if not redact:
        return raw_path
    home = os.path.expanduser("~")
    if home and home != "/" and (raw_path == home or raw_path.startswith(home + os.sep)):
        return "~" + raw_path[len(home) :]
    path = Path(raw_path)
    if path.is_absolute():
        return f".../{path.name}" if path.name else "<absolute-path>"
    return raw_path


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _loads_warning_tuple(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return (value,)
    if isinstance(parsed, list):
        return tuple(str(item) for item in parsed)
    return (str(parsed),)


def _explain_file(path: Path, *, provider_hint: Provider) -> ImportExplainEntryPayload:
    # SQLite structural detection takes priority over the raw-path taxonomy
    # classification below: that classification's METADATA_DOCUMENT/
    # parse_as_session=False verdict for these paths exists to protect
    # pre-JSON-decode consumers (e.g. schema sampling) from raw SQLite bytes,
    # not to gate the SQLite-specific parse routes, which have their own
    # structural admission check (looks_like_*_path).
    try:
        evidence = capture_sqlite_read(lambda: inspect_sqlite_source(path))
    except (OSError, ValueError) as exc:
        return _skipped_entry(
            path,
            provider_hint=provider_hint,
            artifact=None,
            reason=f"SQLite inspection failure: {type(exc).__name__}: {exc}",
        )
    if not isinstance(evidence, Measured):
        detail = evidence.detail if isinstance(evidence, Unavailable) else None
        return _skipped_entry(
            path,
            provider_hint=provider_hint,
            artifact=None,
            reason=f"SQLite inspection failure: {detail or 'sqlite_read_failed'}",
        )
    inspection = evidence.value
    if inspection.domain is not None:
        return _explain_sqlite_inspection(path, inspection, provider_hint=provider_hint)

    path_classification = classify_artifact_path(path, provider=provider_hint)
    if path_classification is not None and not path_classification.parse_as_session:
        return _skipped_entry(
            path,
            provider_hint=provider_hint,
            artifact=path_classification,
            reason=path_classification.reason,
            detector_evidence=(_evidence("artifact_taxonomy.path", matched=True, reason=path_classification.reason),),
        )

    if path.suffix.lower() == ".zip":
        return _explain_zip(path, provider_hint=provider_hint, path_classification=path_classification)

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        return _skipped_entry(
            path,
            provider_hint=provider_hint,
            artifact=path_classification,
            reason=f"read failure: {exc}",
        )
    return _explain_bytes(
        raw_bytes,
        stream_name=path.name,
        source_path=str(path),
        provider_hint=provider_hint,
        path_classification=path_classification,
    )


def _explain_sqlite_inspection(
    path: Path, inspection: SQLiteInspection, *, provider_hint: Provider
) -> ImportExplainEntryPayload:
    """Adapt domain evidence proved on one bound connection to the public payload."""
    domain = inspection.domain
    assert domain is not None
    fidelity = inspection.fidelity
    produced = ImportProducedRowsPayload(**inspection.produced)
    if domain == "antigravity_trajectory_db":
        provider = Provider.ANTIGRAVITY
        artifact_kind = "sqlite_trajectory_database"
        signature = "antigravity_trajectory.signature"
        reason = "trajectory_meta and steps tables"
        caveats = [
            "dry-run inspected the SQLite trajectory read-only; import snapshots a consistent logical export before parsing."
        ]
        if not produced.messages:
            caveats.append("trajectory contains no materialized messages; empty evidence remains attributable.")
        if inspection.degraded:
            caveats.append("trajectory contains typed unsupported or degraded steps; coverage is not complete.")
        parser_version = None
    else:
        provider = Provider.HERMES
        assert fidelity is not None
        if domain == "hermes_state_db":
            artifact_kind = "sqlite_state_database"
            signature = "hermes_state_db.signature"
            reason = "required Hermes tables and signature columns"
            version_prefix = "state-db"
        else:
            artifact_kind = "sqlite_verification_evidence_database"
            signature = "hermes_verification_evidence_db.signature"
            reason = "required verification_events/verification_state tables and columns"
            version_prefix = "verification-evidence-db"
        parser_version = None if fidelity.schema_version is None else f"{version_prefix}-v{fidelity.schema_version}"
        caveats = [
            "dry-run inspected the live SQLite database read-only; import snapshots bytes before parsing.",
            *fidelity.caveats,
        ]
    return ImportExplainEntryPayload(
        source_path=str(path),
        artifact_kind=artifact_kind,
        provider_hint=provider_hint.value,
        detected_origin=_origin_value(provider),
        detected_provider=provider.value,
        detector=domain,
        detector_evidence=(_evidence(signature, matched=True, reason=reason),),
        parser=domain,
        parser_version=parser_version,
        parser_mode="logical_export",
        produced=produced,
        caveats=tuple(caveats),
        raw_evidence_refs=(),
        fidelity=None if fidelity is None else _fidelity_payload(fidelity),
    )


def _explain_zip(
    path: Path,
    *,
    provider_hint: Provider,
    path_classification: ArtifactClassification | None,
) -> ImportExplainEntryPayload:
    entries: list[ImportExplainEntryPayload] = []
    skipped: list[ImportSkippedRowPayload] = []
    detector_evidence = [
        _evidence(
            "artifact_taxonomy.path",
            matched=path_classification is not None,
            reason=path_classification.reason if path_classification is not None else None,
        ),
        _evidence("zip.container", matched=True, reason="ZIP container"),
    ]
    container_provider = provider_hint
    try:
        with (
            TemporaryDirectory(prefix="polylogue-zip-explain-") as scratch,
            bind_source_input(path) as captured,
            open_bound_container(
                BlobStore(Path(scratch)),
                captured,
            ) as physical,
            zipfile.ZipFile(physical.stream) as archive,
        ):
            central_directory = archive.infolist()
            entry_ordinals = {id(info): ordinal for ordinal, info in enumerate(central_directory)}
            with zip_member_admission(
                archive, path, central_directory, provider_hint, container_blob_hash=physical.blob_hash
            ) as admission:
                if container_provider is Provider.UNKNOWN and admission.provider_hint is not Provider.UNKNOWN:
                    container_provider = admission.provider_hint
                    detector_evidence.append(
                        _evidence(
                            "zip.member_dominance",
                            matched=True,
                            reason=f"dominant member provider: {container_provider.value}",
                        )
                    )
                validator = ZipEntryValidator(admission.provider_hint, cursor_state=None, zip_path=path)

                for info in validator.filter_entries(central_directory, allowed_path=admission.allowed_path):
                    entry_ordinal = entry_ordinals[id(info)]
                    entry_provider = admission.entry_provider_hint(info, entry_ordinal=entry_ordinal)
                    profile = captured.captured_identity.member_profile_identity(info.filename)
                    profile_identity = None if profile is None else captured_hermes_profile_key(profile[0])
                    zip_coordinate = captured_zip_member_coordinate(
                        captured.captured_identity,
                        entry_name=info.filename,
                        entry_ordinal=entry_ordinal,
                        split_index=0,
                        addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
                        container_blob_hash=physical.blob_hash,
                        decoder_fingerprint=zip_acquisition_fingerprint(container_provider),
                    )
                    path_classification = classify_artifact_path(info.filename, provider=entry_provider)
                    decoded_session_artifact: ArtifactClassification | None = None
                    if path_classification is not None and not path_classification.parse_as_session:
                        try:
                            decoded_session_artifact = zip_entry_session_artifact(
                                archive,
                                info,
                                provider=entry_provider,
                                profile_identity=profile_identity,
                                captured_zip_coordinate=zip_coordinate,
                            )
                        except zipfile.BadZipFile as exc:
                            skipped.append(
                                ImportSkippedRowPayload(
                                    reason=f"zip entry rejected: {exc}",
                                    source_path=f"{path}:{info.filename}",
                                )
                            )
                            continue
                    if (
                        path_classification is not None
                        and not path_classification.parse_as_session
                        and decoded_session_artifact is None
                    ):
                        skipped.append(
                            ImportSkippedRowPayload(
                                reason=path_classification.reason or "not a session artifact",
                                source_path=f"{path}:{info.filename}",
                            )
                        )
                        continue
                    try:
                        entry = _explain_zip_entry(
                            archive,
                            info,
                            source_path=f"{path}:{info.filename}",
                            provider_hint=entry_provider,
                            profile_identity=profile_identity,
                            captured_zip_coordinate=zip_coordinate,
                        )
                    except zipfile.BadZipFile as exc:
                        skipped.append(
                            ImportSkippedRowPayload(
                                reason=f"zip entry rejected: {exc}",
                                source_path=f"{path}:{info.filename}",
                            )
                        )
                        continue
                    entries.append(entry)
                    skipped.extend(entry.skipped)
    except (OSError, zipfile.BadZipFile) as exc:
        return _skipped_entry(
            path,
            provider_hint=provider_hint,
            artifact=path_classification,
            reason=f"zip failure: {exc}",
            detector_evidence=tuple(detector_evidence),
        )

    produced = ImportProducedRowsPayload(
        sessions=sum(entry.produced.sessions for entry in entries),
        messages=sum(entry.produced.messages for entry in entries),
        blocks=sum(entry.produced.blocks for entry in entries),
        actions=sum(entry.produced.actions for entry in entries),
        raw_records=sum(entry.produced.raw_records for entry in entries),
        session_refs=tuple(ref for entry in entries for ref in entry.produced.session_refs),
    )
    return ImportExplainEntryPayload(
        source_path=str(path),
        artifact_kind=path_classification.kind.value if path_classification is not None else "zip",
        provider_hint=provider_hint.value,
        detected_origin=_origin_value(container_provider),
        detected_provider=container_provider.value,
        detector="zip.container",
        detector_evidence=tuple(detector_evidence),
        parser="zip entries",
        produced=produced,
        skipped=tuple(skipped),
        caveats=("ZIP explanation summarizes supported entries; raw bytes are omitted.",),
    )


def _explain_zip_entry(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    source_path: str,
    provider_hint: Provider,
    profile_identity: str | None = None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None,
) -> ImportExplainEntryPayload:
    """Aggregate the existing sealed parser artifact without a session list."""
    import ijson

    from polylogue.sources.detection_projection import DetectorProjection, project_detection_input
    from polylogue.sources.dispatch import detect_provider_from_stream_evidence

    try:
        with open_zip_entry(archive, info) as source:
            detected, evidence = detect_provider_from_stream_evidence(source)
            shape, mode_view = project_detection_input(source, DetectorProjection(fields={"sessions": None}))
    except (ijson.JSONError, json.JSONDecodeError, UnicodeError) as exc:
        # A member that is not decodable JSON is that member's skip, exactly
        # as an undecodable standalone file is; it never aborts the archive.
        return _skipped_entry(
            Path(source_path),
            provider_hint=provider_hint,
            artifact=None,
            reason=f"decode failure: {exc}",
        )
    provider = detected or provider_hint
    refs: list[str] = []
    session_count = messages = blocks = actions = 0
    observer: ParsedSession | None = None
    with prepare_zip_entry(
        archive,
        info,
        provider=provider,
        source_path=source_path,
        profile_identity=profile_identity,
        captured_zip_coordinate=captured_zip_coordinate,
    ) as prepared:
        if prepared.error is not None or prepared.deferred or prepared.blob_hash is None:
            return _skipped_entry(
                Path(source_path),
                provider_hint=provider_hint,
                artifact=None,
                reason=prepared.error or "source preparation deferred",
                detected_provider=provider,
            )
        for session in prepared.iter_sessions():
            session_count += 1
            refs.append(f"session:{session.source_name.value}:{session.provider_session_id}")
            if observer is None and {"hermes:atif-trajectory", "hermes:atof-observer"}.intersection(
                session.ingest_flags
            ):
                observer = session
            for message in session.messages:
                messages += 1
                for block in message.blocks:
                    blocks += 1
                    actions += block.type.value == "tool_use"
        fidelity = None
        if provider is Provider.HERMES:
            fidelity = _fidelity_payload(
                hermes_spans.import_fidelity_declaration(observer)
                if observer is not None
                else hermes_state.json_fallback_fidelity_counts(sessions=session_count, messages=messages)
            )
    parser_mode = (
        "grouped_records"
        if provider in GROUP_PROVIDERS
        else "bundle_record"
        if shape == "sequence"
        else "session_bundle"
        if isinstance(mode_view, dict) and "sessions" in mode_view
        else "single_record"
    )
    return ImportExplainEntryPayload(
        source_path=source_path,
        artifact_kind="session_record_stream" if provider in GROUP_PROVIDERS else "session_document",
        provider_hint=provider_hint.value,
        detected_origin=_origin_value(provider),
        detected_provider=provider.value,
        detector="provider_shape",
        detector_evidence=(_evidence(evidence, matched=provider is not Provider.UNKNOWN, reason=provider.value),),
        parser=provider.value,
        parser_mode=parser_mode,
        produced=ImportProducedRowsPayload(
            sessions=session_count,
            messages=messages,
            blocks=blocks,
            actions=actions,
            raw_records=session_count,
            session_refs=tuple(refs),
        ),
        caveats=(() if session_count else ("parser produced no sessions",))
        + (() if fidelity is None else fidelity.caveats),
        raw_evidence_refs=(),
        fidelity=fidelity,
    )


def _explain_bytes(
    raw_bytes: bytes,
    *,
    stream_name: str,
    source_path: str,
    provider_hint: Provider,
    path_classification: ArtifactClassification | None,
) -> ImportExplainEntryPayload:
    try:
        payload = _load_payload(raw_bytes, stream_name)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return _skipped_entry(
            Path(source_path),
            provider_hint=provider_hint,
            artifact=path_classification,
            reason=f"decode failure: {exc}",
        )

    detected_provider = detect_provider(payload) or provider_hint
    artifact = path_classification or classify_artifact(payload, provider=detected_provider, source_path=source_path)
    detector_evidence = (
        _evidence(
            "provider_shape",
            matched=detected_provider is not Provider.UNKNOWN,
            reason=detected_provider.value
            if detected_provider is not Provider.UNKNOWN
            else "no provider-shaped payload",
        ),
        _evidence("artifact_taxonomy.payload", matched=artifact.parse_as_session, reason=artifact.reason),
    )
    if not artifact.parse_as_session:
        return _skipped_entry(
            Path(source_path),
            provider_hint=provider_hint,
            artifact=artifact,
            reason=artifact.reason,
            detector_evidence=detector_evidence,
            detected_provider=detected_provider,
        )

    try:
        if is_stream_record_provider(source_path, detected_provider):
            stream_payloads = payload if isinstance(payload, list) else [payload]
            sessions = parse_stream_payload(
                detected_provider,
                stream_payloads,
                Path(stream_name).stem,
                source_path=source_path,
            )
        else:
            sessions = parse_payload(
                detected_provider,
                payload,
                Path(stream_name).stem,
                source_path=source_path,
            )
    except Exception as exc:
        return _skipped_entry(
            Path(source_path),
            provider_hint=provider_hint,
            artifact=artifact,
            reason=f"parser failure: {type(exc).__name__}: {exc}",
            detector_evidence=detector_evidence,
            detected_provider=detected_provider,
        )

    fidelity = None
    if detected_provider is Provider.HERMES:
        observer_session = next(
            (
                session
                for session in sessions
                if {"hermes:atif-trajectory", "hermes:atof-observer"}.intersection(session.ingest_flags)
            ),
            None,
        )
        fidelity = _fidelity_payload(
            hermes_spans.import_fidelity_declaration(observer_session)
            if observer_session is not None
            else hermes_state.import_fidelity_declaration(sessions, acquisition_method="json_fallback")
        )
    return ImportExplainEntryPayload(
        source_path=source_path,
        artifact_kind=artifact.kind.value,
        provider_hint=provider_hint.value,
        detected_origin=_origin_value(detected_provider),
        detected_provider=detected_provider.value,
        detector="provider_shape",
        detector_evidence=detector_evidence,
        parser=detected_provider.value,
        parser_mode=_parser_mode(detected_provider, payload),
        produced=_produced_rows(sessions),
        caveats=(
            (() if sessions else ("parser produced no sessions",)) + (() if fidelity is None else fidelity.caveats)
        ),
        raw_evidence_refs=(),
        fidelity=fidelity,
    )


def _fidelity_payload(fidelity: hermes_state.HermesImportFidelity) -> ImportFidelityDeclarationPayload:
    def capability(item: hermes_state.HermesFidelityCapability) -> ImportFidelityCapabilityPayload:
        return ImportFidelityCapabilityPayload(
            status=item.status,
            observed=item.observed,
            expected=item.expected,
            counts=item.counts,
            detail=item.detail,
        )

    return ImportFidelityDeclarationPayload(
        producer=fidelity.producer,
        schema_version=fidelity.schema_version,
        profile_namespace=fidelity.profile_namespace,
        acquisition_method=fidelity.acquisition_method,
        retained_blob_reproducibility=capability(fidelity.retained_blob_reproducibility),
        capabilities={name: capability(item) for name, item in fidelity.capabilities.items()},
        caveats=fidelity.caveats,
    )


def _load_payload(raw_bytes: bytes, stream_name: str) -> JSONValue:
    if is_jsonl_source_path(stream_name):
        return list(_iter_json_stream(BytesIO(raw_bytes), stream_name))
    text = _decode_json_bytes(raw_bytes)
    if text is None:
        raise UnicodeDecodeError("utf-8", raw_bytes, 0, min(len(raw_bytes), 1), "unsupported JSON encoding")
    return cast(JSONValue, json.loads(text))


def _parser_mode(provider: Provider, payload: object) -> str:
    if provider in GROUP_PROVIDERS:
        return "grouped_records"
    if isinstance(payload, list):
        return "bundle_record"
    if isinstance(payload, dict) and "sessions" in payload:
        return "session_bundle"
    return "single_record"


def _produced_rows(sessions: list[ParsedSession]) -> ImportProducedRowsPayload:
    messages = [message for session in sessions for message in session.messages]
    return ImportProducedRowsPayload(
        sessions=len(sessions),
        messages=len(messages),
        blocks=sum(len(message.blocks) for message in messages),
        actions=sum(1 for message in messages for block in message.blocks if block.type.value == "tool_use"),
        raw_records=len(sessions),
        session_refs=tuple(
            f"session:{session.source_name.value}:{session.provider_session_id}" for session in sessions
        ),
    )


def _skipped_entry(
    path: Path,
    *,
    provider_hint: Provider,
    artifact: ArtifactClassification | None,
    reason: str,
    detector_evidence: tuple[ImportDetectorEvidencePayload, ...] = (),
    detected_provider: Provider | None = None,
) -> ImportExplainEntryPayload:
    skipped = ImportSkippedRowPayload(reason=reason, source_path=str(path))
    provider = detected_provider or provider_hint
    return ImportExplainEntryPayload(
        source_path=str(path),
        artifact_kind=artifact.kind.value if artifact is not None else None,
        provider_hint=provider_hint.value,
        detected_origin=_origin_value(provider),
        detected_provider=provider.value,
        detector="artifact_taxonomy" if artifact is not None else "provider_shape",
        detector_evidence=detector_evidence,
        parser=None,
        produced=ImportProducedRowsPayload(),
        skipped=(skipped,),
        caveats=(reason,),
    )


def _origin_value(provider: Provider) -> str:
    return origin_from_provider(provider).value


def _evidence(check: str, *, matched: bool, reason: str | None = None) -> ImportDetectorEvidencePayload:
    return ImportDetectorEvidencePayload(check=check, matched=matched, reason=reason)


__all__ = ["explain_import_archive", "explain_import_path"]
