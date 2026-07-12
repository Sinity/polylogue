"""Import explain payload construction over the existing parser stack."""

from __future__ import annotations

import json
import os
import sqlite3
import zipfile
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import cast

from polylogue.archive.artifact_taxonomy import ArtifactClassification, classify_artifact, classify_artifact_path
from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue
from polylogue.core.sources import origin_from_provider
from polylogue.sources.decoder_zip import (
    MAX_COMPRESSION_RATIO,
    MAX_UNCOMPRESSED_SIZE,
    ZipBombError,
    open_bounded_zip_entry,
)
from polylogue.sources.decoders import _decode_json_bytes, _iter_json_stream
from polylogue.sources.dispatch import GROUP_PROVIDERS, detect_provider, is_jsonl_source_path, parse_payload
from polylogue.sources.parsers import hermes_state
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.source_walk import _resolve_source_paths
from polylogue.surfaces.payloads import (
    ImportDetectorEvidencePayload,
    ImportExplainEntryPayload,
    ImportExplainPayload,
    ImportFidelityCapabilityPayload,
    ImportFidelityDeclarationPayload,
    ImportProducedRowsPayload,
    ImportSkippedRowPayload,
)

_SUPPORTED_ENTRY_SUFFIXES = (".json", ".jsonl", ".jsonl.txt", ".ndjson")


def explain_import_path(
    path: Path,
    *,
    source_name: str = "unknown",
    limit: int = 100,
) -> ImportExplainPayload:
    """Return a bounded import explanation for a file or directory.

    This is intentionally non-mutating: it reads local bytes, runs the same
    detector/parser path used by import, and reports what would be produced
    without staging daemon work or writing raw blobs.
    """

    resolved = path.expanduser().resolve()
    entries: list[ImportExplainEntryPayload] = []
    skipped: list[ImportSkippedRowPayload] = []
    caveats: list[str] = []

    if not resolved.exists():
        skipped.append(ImportSkippedRowPayload(reason="path does not exist", source_path=str(resolved)))
        return _envelope(resolved, entries=entries, skipped=skipped, caveats=caveats)

    for candidate in _candidate_paths(resolved, source_name=source_name):
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
    with _readonly_sqlite(source_db) as source_conn:
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
            index_conn = _readonly_sqlite(index_db)
        else:
            caveats.append("index tier is unavailable; produced archive row counts are incomplete")
        try:
            for row in raw_rows:
                artifact_rows = _select_artifact_rows(source_conn, raw_id=str(row["raw_id"]))
                entry = _archive_entry_from_rows(
                    row,
                    artifact_rows=artifact_rows,
                    index_conn=index_conn,
                    redact_paths=redact_paths,
                )
                entries.append(entry)
                skipped.extend(entry.skipped)
        finally:
            if index_conn is not None:
                index_conn.close()

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

    return ImportExplainEntryPayload(
        raw_ref=f"raw:{raw_id}",
        source_path=source_path,
        artifact_kind=artifact_kind,
        provider_hint=str(row["origin"]),
        detected_origin=str(row["origin"]),
        detected_provider=None,
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


def _readonly_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


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

    if hermes_state.looks_like_state_db_path(path):
        return _explain_hermes_state_db(path, provider_hint=provider_hint)

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


def _explain_hermes_state_db(path: Path, *, provider_hint: Provider) -> ImportExplainEntryPayload:
    """Inspect the real Hermes SQLite parser path without writing a raw blob."""

    try:
        sessions = hermes_state.parse_state_db(path, fallback_id=path.stem, profile_root=path.parent)
    except (OSError, sqlite3.Error, ValueError) as exc:
        return _skipped_entry(
            path,
            provider_hint=provider_hint,
            artifact=None,
            reason=f"Hermes state.db parser failure: {type(exc).__name__}: {exc}",
            detected_provider=Provider.HERMES,
        )
    fidelity = hermes_state.import_fidelity_declaration(sessions, acquisition_method="sqlite_backup")
    return ImportExplainEntryPayload(
        source_path=str(path),
        artifact_kind="sqlite_state_database",
        provider_hint=provider_hint.value,
        detected_origin=_origin_value(Provider.HERMES),
        detected_provider=Provider.HERMES.value,
        detector="hermes_state_db",
        detector_evidence=(
            _evidence("hermes_state_db.signature", matched=True, reason="required Hermes tables and signature columns"),
        ),
        parser="hermes_state_db",
        parser_version=None if fidelity.schema_version is None else f"state-db-v{fidelity.schema_version}",
        parser_mode="sqlite_backup",
        produced=_produced_rows(sessions),
        caveats=(
            "dry-run inspected the live SQLite database read-only; import snapshots bytes before parsing.",
            *fidelity.caveats,
        ),
        raw_evidence_refs=(),
        fidelity=_fidelity_payload(fidelity),
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
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                skip_reason = _zip_entry_skip_reason(info)
                if skip_reason is not None:
                    skipped.append(
                        ImportSkippedRowPayload(
                            reason=skip_reason,
                            source_path=f"{path}:{info.filename}",
                        )
                    )
                    continue
                try:
                    with open_bounded_zip_entry(archive, info.filename) as handle:
                        entry = _explain_bytes(
                            handle.read(MAX_UNCOMPRESSED_SIZE + 1),
                            stream_name=info.filename,
                            source_path=f"{path}:{info.filename}",
                            provider_hint=provider_hint,
                            path_classification=None,
                        )
                except ZipBombError as exc:
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
        detected_origin=_origin_value(provider_hint),
        detected_provider=provider_hint.value,
        detector="zip.container",
        detector_evidence=tuple(detector_evidence),
        parser="zip entries",
        produced=produced,
        skipped=tuple(skipped),
        caveats=("ZIP explanation summarizes supported entries; raw bytes are omitted.",),
    )


def _zip_entry_skip_reason(info: zipfile.ZipInfo) -> str | None:
    if info.is_dir() or not info.filename.lower().endswith(_SUPPORTED_ENTRY_SUFFIXES):
        return "unsupported ZIP entry"
    if info.compress_size > 0 and (info.file_size / info.compress_size) > MAX_COMPRESSION_RATIO:
        return f"zip entry compression ratio {info.file_size / info.compress_size:.1f} exceeds limit"
    if info.file_size > MAX_UNCOMPRESSED_SIZE:
        return f"zip entry file size {info.file_size} exceeds limit"
    return None


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

    fidelity = (
        _fidelity_payload(hermes_state.import_fidelity_declaration(sessions, acquisition_method="json_fallback"))
        if detected_provider is Provider.HERMES
        else None
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
