"""Raw payload decoding and provider inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from polylogue.archive.artifact_taxonomy import (
    ArtifactClassification,
    ArtifactKind,
    classify_artifact,
)
from polylogue.archive.raw_payload.streams import raw_line_stream
from polylogue.core.binary_signatures import detect_binary_signature
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDecodeError, JSONDocument, JSONValue, is_json_value, loads
from polylogue.sources.dispatch import detect_provider

_BINARY_ARTIFACT_MARKER = "unrecognized_binary_artifact"
_UNRECOGNIZED_BINARY_PEEK_BYTES = 32

WireFormat = Literal["json", "jsonl"]
JSONRecord: TypeAlias = JSONDocument


def _decode_provider_utf8(raw: bytes) -> str:
    """Decode provider bytes while preserving UTF-8-encoded surrogate code units.

    Some historical exports contain a lone UTF-16 surrogate encoded directly
    as its three-byte UTF-8 sequence. This is invalid Unicode scalar UTF-8,
    so the active JSON backend correctly rejects it, but Python can preserve the original
    code unit with ``surrogatepass``. Arbitrary malformed byte sequences still
    raise and retain the ordinary malformed-JSONL behavior.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        try:
            return raw.decode("utf-8", errors="surrogatepass")
        except UnicodeDecodeError:
            raise error from None


def _load_json_record(line: str) -> JSONValue:
    try:
        return loads(line)
    except JSONDecodeError:
        # Retry with stdlib json — tolerant of raw control characters
        # (ANSI escape codes in bash output, etc.) that the active JSON backend rejects.
        import json

        return cast("JSONValue", json.loads(line))


def _load_raw_json(raw: bytes | str) -> JSONValue:
    """Parse a JSON document, retaining recoverable provider surrogates."""
    try:
        return loads(raw)
    except (JSONDecodeError, ValueError) as error:
        if not isinstance(raw, bytes):
            raise
        try:
            return _load_json_record(_decode_provider_utf8(raw))
        except (JSONDecodeError, ValueError, UnicodeDecodeError):
            raise error from None


@dataclass(frozen=True)
class RawPayloadEnvelope:
    """Canonical decoded raw payload with inferred runtime semantics."""

    payload: JSONValue
    provider: Provider
    wire_format: WireFormat
    artifact: ArtifactClassification
    malformed_jsonl_lines: int = 0
    malformed_jsonl_detail: str | None = None


def _decode_jsonl_payload(
    raw: Path | bytes | str,
    *,
    jsonl_dict_only: bool = False,
) -> tuple[list[JSONValue], int, str | None]:
    """Decode JSONL incrementally to avoid full-file line splitting.

    When *raw* is a :class:`~pathlib.Path`, lines are streamed directly
    from the file handle — the full file is never loaded into memory.
    """
    lines: list[JSONValue] = []
    malformed_lines = 0
    malformed_detail: str | None = None
    first_line = True
    line_number = 0

    with raw_line_stream(raw) as stream:
        for raw_line in stream:
            line_number += 1
            try:
                line = _decode_provider_utf8(raw_line) if isinstance(raw_line, bytes) else raw_line
            except UnicodeDecodeError as exc:
                malformed_lines += 1
                if malformed_detail is None:
                    malformed_detail = f"line {line_number}: {exc.reason}"
                continue
            if first_line:
                line = line.lstrip("\ufeff")
                first_line = False
            line = line.strip()
            if not line:
                continue
            try:
                parsed = _load_json_record(line)
            except (JSONDecodeError, ValueError) as exc:
                malformed_lines += 1
                if malformed_detail is None:
                    malformed_detail = f"line {line_number}: {exc}"
                continue
            if jsonl_dict_only and not isinstance(parsed, dict):
                continue
            lines.append(parsed)

    if not lines:
        raise ValueError("No valid JSONL records found")
    return lines, malformed_lines, malformed_detail


def _sample_jsonl_payload_with_detail(
    raw: Path | bytes | str,
    *,
    max_samples: int = 64,
    jsonl_dict_only: bool = False,
    scan_full: bool = True,
) -> tuple[list[JSONValue], int, str | None]:
    """Collect a bounded sample of valid JSONL records.

    This is intended for provider/artifact/schema resolution where full-record
    materialization is unnecessary. Set ``scan_full`` when malformed-line
    accounting must reflect the entire source, such as strict validation.
    """
    samples: list[JSONValue] = []
    malformed_lines = 0
    malformed_detail: str | None = None
    valid_records = 0
    first_line = True
    line_number = 0

    with raw_line_stream(raw) as stream:
        for raw_line in stream:
            line_number += 1
            try:
                line = _decode_provider_utf8(raw_line) if isinstance(raw_line, bytes) else raw_line
            except UnicodeDecodeError as exc:
                malformed_lines += 1
                if malformed_detail is None:
                    malformed_detail = f"line {line_number}: {exc.reason}"
                continue
            if first_line:
                line = line.lstrip("\ufeff")
                first_line = False
            line = line.strip()
            if not line:
                continue
            try:
                parsed = _load_json_record(line)
            except (JSONDecodeError, ValueError) as exc:
                malformed_lines += 1
                if malformed_detail is None:
                    malformed_detail = f"line {line_number}: {exc}"
                continue
            if jsonl_dict_only and not isinstance(parsed, dict):
                continue
            valid_records += 1
            if len(samples) < max_samples:
                samples.append(parsed)
            if not scan_full and len(samples) >= max_samples:
                break

    if valid_records == 0:
        raise ValueError("No valid JSONL records found")
    return samples, malformed_lines, malformed_detail


def sample_jsonl_payload(
    raw: Path | bytes | str,
    *,
    max_samples: int = 64,
    jsonl_dict_only: bool = False,
) -> tuple[list[JSONValue], int]:
    samples, malformed_lines, _detail = _sample_jsonl_payload_with_detail(
        raw,
        max_samples=max_samples,
        jsonl_dict_only=jsonl_dict_only,
    )
    return samples, malformed_lines


def _decode_raw_payload(
    raw_content: Path | bytes | str | JSONValue,
    *,
    jsonl_dict_only: bool = False,
    prefer_jsonl: bool = False,
) -> tuple[JSONValue, WireFormat, int, str | None]:
    """Decode JSON payload bytes, with JSONL fallback support.

    When *raw_content* is a :class:`~pathlib.Path`, JSONL files are
    streamed line-by-line from disk (never fully loaded into memory).
    For JSON files the path is read in one shot via the active JSON backend's decode.
    """
    if isinstance(raw_content, Path):
        if prefer_jsonl:
            try:
                payload, malformed_lines, malformed_detail = _decode_jsonl_payload(
                    raw_content,
                    jsonl_dict_only=jsonl_dict_only,
                )
                return payload, "jsonl", malformed_lines, malformed_detail
            except (UnicodeDecodeError, ValueError):
                pass
        raw_bytes = raw_content.read_bytes()
        try:
            return _load_raw_json(raw_bytes), "json", 0, None
        except (JSONDecodeError, ValueError) as exc:
            try:
                payload, malformed_lines, malformed_detail = _decode_jsonl_payload(
                    raw_bytes,
                    jsonl_dict_only=jsonl_dict_only,
                )
            except (UnicodeDecodeError, ValueError):
                raise exc from None
            return payload, "jsonl", malformed_lines, malformed_detail

    if is_json_value(raw_content):
        return raw_content, "json", 0, None

    raw = raw_content if isinstance(raw_content, (bytes, str)) else str(raw_content)
    if prefer_jsonl:
        try:
            payload, malformed_lines, malformed_detail = _decode_jsonl_payload(
                raw,
                jsonl_dict_only=jsonl_dict_only,
            )
            return payload, "jsonl", malformed_lines, malformed_detail
        except (UnicodeDecodeError, ValueError):
            pass
    try:
        return _load_raw_json(raw), "json", 0, None
    except (JSONDecodeError, ValueError) as exc:
        try:
            payload, malformed_lines, malformed_detail = _decode_jsonl_payload(
                raw,
                jsonl_dict_only=jsonl_dict_only,
            )
        except (UnicodeDecodeError, ValueError):
            raise exc from None
        return payload, "jsonl", malformed_lines, malformed_detail


def _infer_payload_provider(
    payload: JSONValue,
    *,
    source_path: str | Path | None,
    fallback_provider: str | Provider,
    payload_provider: str | Provider | None = None,
) -> Provider:
    """Infer canonical provider from payload/path, with fallback."""
    if payload_provider:
        return Provider.from_string(payload_provider)
    fallback_token = Provider.from_string(fallback_provider)
    normalized_path = str(source_path or "").replace("\\", "/").lower()
    if fallback_token is Provider.CLAUDE_CODE and "/subagents/" in normalized_path:
        return fallback_token
    inferred = detect_provider(payload)
    if inferred:
        return inferred
    return fallback_token


def _unrecognized_binary_marker_payload(prefix: bytes, *, source_path: str | Path | None) -> JSONDocument | None:
    """Return a synthetic non-session marker for a recognized-but-unclaimed binary payload.

    polylogue-hbtj2: a raw payload whose leading bytes match a known binary
    format (SQLite being the concrete miscapture this bead fixes -- Hermes/
    Codex state databases swept into ``raw_sessions`` and treated as
    ambiguous session revisions purely because nothing checked the bytes
    before attempting a JSON decode) must be refused as session content
    *before* any JSON/JSONL decode is attempted, not merely tolerated via
    the incidental ``JSONDecodeError`` a binary blob happens to raise. This
    runs after ``_hermes_sqlite_marker_payload`` (so a content-verified,
    dedicated Hermes state.db/verification_evidence.db parse still wins)
    and covers every other case: Codex's own state databases (which
    deliberately never become sessions of their own, see
    ``sources/parsers/codex_state.py``), any other provider's stray SQLite
    file, and the other recognized binary formats in
    ``core.binary_signatures.BINARY_SIGNATURES``.
    """
    signature = detect_binary_signature(prefix)
    if signature is None:
        return None
    return {
        "polylogue_artifact": _BINARY_ARTIFACT_MARKER,
        "binary_format": signature.name,
        "source_path": str(source_path) if source_path is not None else None,
    }


def _binary_artifact_envelope(binary_marker: JSONDocument, *, provider: Provider) -> RawPayloadEnvelope:
    signature_name = binary_marker["binary_format"]
    kind = ArtifactKind.BINARY_DATABASE if signature_name == "sqlite" else ArtifactKind.BINARY_DOCUMENT
    return RawPayloadEnvelope(
        payload=binary_marker,
        provider=provider,
        wire_format="json",
        artifact=ArtifactClassification(
            provider=provider,
            kind=kind,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason=f"unrecognized {signature_name}-shaped binary payload; refused as session content (polylogue-hbtj2)",
        ),
    )


def build_raw_payload_envelope(
    raw_content: Path | bytes | str | JSONValue,
    *,
    source_path: str | Path | None,
    fallback_provider: str | Provider,
    payload_provider: str | Provider | None = None,
    jsonl_dict_only: bool = False,
    sqlite_immutable: bool = False,
) -> RawPayloadEnvelope:
    """Decode raw payload and attach canonical provider/wire-format identity.

    The default preserves live-source marker semantics, where an active SQLite
    database may need its WAL. Callers inspecting retained content-addressed
    blobs must pass ``sqlite_immutable=True`` so SQLite cannot create ``-wal``
    or ``-shm`` namespace entries beside the blob.

    When *raw_content* is a :class:`~pathlib.Path`, JSONL payloads are
    decoded line-by-line from disk before being materialized into a
    Python list. This avoids reading the whole file into one byte string,
    but grouped-provider parses still hold the decoded records in memory.
    """
    provider_for_binary = Provider.from_string(payload_provider or fallback_provider)
    if isinstance(raw_content, Path):
        hermes_marker = _hermes_sqlite_marker_payload(
            raw_content,
            source_path=source_path,
            immutable=sqlite_immutable,
        )
        if hermes_marker is not None:
            provider = Provider.HERMES
            return RawPayloadEnvelope(
                payload=hermes_marker,
                provider=provider,
                wire_format="json",
                artifact=classify_artifact(hermes_marker, provider=provider, source_path=source_path),
            )
        with raw_content.open("rb") as handle:
            binary_prefix = handle.read(_UNRECOGNIZED_BINARY_PEEK_BYTES)
        binary_marker = _unrecognized_binary_marker_payload(binary_prefix, source_path=source_path)
        if binary_marker is not None:
            return _binary_artifact_envelope(binary_marker, provider=provider_for_binary)
    normalized_path = str(source_path or "").lower()
    prefer_jsonl = normalized_path.endswith((".jsonl", ".jsonl.txt", ".ndjson"))
    preferred_provider = payload_provider or fallback_provider
    if not prefer_jsonl:
        runtime_provider = Provider.from_string(preferred_provider)
        prefer_jsonl = runtime_provider in {Provider.CLAUDE_CODE, Provider.CODEX}
    if isinstance(raw_content, bytes):
        binary_marker = _unrecognized_binary_marker_payload(
            raw_content[:_UNRECOGNIZED_BINARY_PEEK_BYTES], source_path=source_path
        )
        if binary_marker is not None:
            return _binary_artifact_envelope(binary_marker, provider=provider_for_binary)
    payload, wire_format, malformed_jsonl_lines, malformed_jsonl_detail = _decode_raw_payload(
        raw_content,
        jsonl_dict_only=jsonl_dict_only,
        prefer_jsonl=prefer_jsonl,
    )
    provider = _infer_payload_provider(
        payload,
        source_path=source_path,
        fallback_provider=fallback_provider,
        payload_provider=payload_provider,
    )
    artifact = classify_artifact(
        payload,
        provider=provider,
        source_path=source_path,
    )
    return RawPayloadEnvelope(
        payload=payload,
        provider=provider,
        wire_format=wire_format,
        artifact=artifact,
        malformed_jsonl_lines=malformed_jsonl_lines,
        malformed_jsonl_detail=malformed_jsonl_detail,
    )


def _hermes_sqlite_marker_payload(
    path: Path,
    *,
    source_path: str | Path | None,
    immutable: bool = False,
) -> JSONDocument | None:
    """Route a raw Hermes SQLite blob (state.db / verification_evidence.db) to its marker payload.

    Binary-capable detection BEFORE any text decode, mirroring the rebuild
    path's provider dispatch (``sources.revision_backfill._parse_one``, which
    probes ``looks_like_sqlite_bytes`` then the same per-artifact
    ``looks_like_*_path`` structural checks used here). ``ingest_record``
    (``pipeline/services/ingest_worker.py``) calls ``build_raw_payload_envelope``
    for every raw, including binary ones, so this is the single place that
    must recognize a SQLite-backed Hermes artifact before the generic
    JSON/JSONL decode path runs and rejects the bytes as invalid UTF-8
    (polylogue-zoc3: ingest and rebuild previously disagreed here for
    ``verification_evidence.db`` raws, which only the rebuild path handled).

    Import lazily: ``dispatch`` imports the Hermes parsers while this module
    imports ``dispatch`` for ordinary JSON provider detection. At runtime the
    module graph is complete, and delegating here keeps raw inspection on the
    same versioned structural contract as actual parsing.
    """
    from polylogue.sources.parsers import hermes_state, hermes_verification

    profile_root = Path(source_path).parent if source_path is not None else None
    if hermes_state.looks_like_state_db_path(path, immutable=immutable):
        return hermes_state.marker_payload(path, profile_root=profile_root, immutable=immutable)
    if hermes_verification.looks_like_verification_evidence_db_path(path, immutable=immutable):
        return hermes_verification.marker_payload(path, profile_root=profile_root, immutable=immutable)
    return None


__all__ = [
    "JSONRecord",
    "JSONValue",
    "RawPayloadEnvelope",
    "WireFormat",
    "build_raw_payload_envelope",
    "sample_jsonl_payload",
]
