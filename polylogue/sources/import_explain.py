"""Import explain payload construction over the existing parser stack."""

from __future__ import annotations

import json
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
from polylogue.sources.decoders import _decode_json_bytes, _iter_json_stream
from polylogue.sources.dispatch import GROUP_PROVIDERS, detect_provider, parse_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.source_walk import _resolve_source_paths
from polylogue.surfaces.payloads import (
    ImportDetectorEvidencePayload,
    ImportExplainEntryPayload,
    ImportExplainPayload,
    ImportProducedRowsPayload,
    ImportSkippedRowPayload,
)

_JSONL_SUFFIXES = (".jsonl", ".jsonl.txt", ".ndjson")
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
                if info.is_dir() or not info.filename.lower().endswith(_SUPPORTED_ENTRY_SUFFIXES):
                    skipped.append(
                        ImportSkippedRowPayload(
                            reason="unsupported ZIP entry",
                            source_path=f"{path}:{info.filename}",
                        )
                    )
                    continue
                with archive.open(info) as handle:
                    entries.append(
                        _explain_bytes(
                            handle.read(),
                            stream_name=info.filename,
                            source_path=f"{path}:{info.filename}",
                            provider_hint=provider_hint,
                            path_classification=None,
                        )
                    )
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
        caveats=() if sessions else ("parser produced no sessions",),
        raw_evidence_refs=(),
    )


def _load_payload(raw_bytes: bytes, stream_name: str) -> JSONValue:
    lower = stream_name.lower()
    if lower.endswith(_JSONL_SUFFIXES):
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


__all__ = ["explain_import_path"]
