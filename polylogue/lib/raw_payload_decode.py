"""Raw payload decoding and provider inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal

import orjson

from polylogue.lib.artifact_taxonomy import ArtifactClassification, classify_artifact
from polylogue.sources.dispatch import detect_provider
from polylogue.types import Provider

WireFormat = Literal["json", "jsonl"]


@dataclass(frozen=True)
class RawPayloadEnvelope:
    """Canonical decoded raw payload with inferred runtime semantics."""

    payload: Any
    provider: Provider
    wire_format: WireFormat
    artifact: ArtifactClassification
    malformed_jsonl_lines: int = 0


def _decode_jsonl_payload(
    raw: Path | bytes | str,
    *,
    jsonl_dict_only: bool = False,
) -> tuple[list[Any], int]:
    """Decode JSONL incrementally to avoid full-file line splitting.

    When *raw* is a :class:`~pathlib.Path`, lines are streamed directly
    from the file handle — the full file is never loaded into memory.
    """
    lines: list[Any] = []
    malformed_lines = 0
    first_line = True

    fh = (
        open(raw, "rb")  # noqa: SIM115 — caller-managed context
        if isinstance(raw, Path)
        else BytesIO(raw)
        if isinstance(raw, bytes)
        else StringIO(raw)
    )

    try:
        for raw_line in fh:
            try:
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            except UnicodeDecodeError:
                malformed_lines += 1
                continue
            if first_line:
                line = line.lstrip("\ufeff")
                first_line = False
            line = line.strip()
            if not line:
                continue
            try:
                parsed = orjson.loads(line)
            except (orjson.JSONDecodeError, ValueError):
                malformed_lines += 1
                continue
            if jsonl_dict_only and not isinstance(parsed, dict):
                continue
            lines.append(parsed)
    finally:
        if isinstance(raw, Path):
            fh.close()

    if not lines:
        raise ValueError("No valid JSONL records found")
    return lines, malformed_lines


def _decode_raw_payload(
    raw_content: Path | bytes | str | Any,
    *,
    jsonl_dict_only: bool = False,
    prefer_jsonl: bool = False,
) -> tuple[Any, WireFormat, int]:
    """Decode JSON payload bytes, with JSONL fallback support.

    When *raw_content* is a :class:`~pathlib.Path`, JSONL files are
    streamed line-by-line from disk (never fully loaded into memory).
    For JSON files the path is read in one shot via ``orjson.loads``.
    """
    if isinstance(raw_content, Path):
        if prefer_jsonl:
            try:
                payload, malformed_lines = _decode_jsonl_payload(
                    raw_content,
                    jsonl_dict_only=jsonl_dict_only,
                )
                return payload, "jsonl", malformed_lines
            except (UnicodeDecodeError, ValueError):
                pass
        raw_bytes = raw_content.read_bytes()
        try:
            return orjson.loads(raw_bytes), "json", 0
        except (orjson.JSONDecodeError, ValueError) as exc:
            try:
                payload, malformed_lines = _decode_jsonl_payload(
                    raw_bytes,
                    jsonl_dict_only=jsonl_dict_only,
                )
            except (UnicodeDecodeError, ValueError):
                raise exc from None
            return payload, "jsonl", malformed_lines

    raw = raw_content if isinstance(raw_content, (bytes, str)) else str(raw_content)
    if prefer_jsonl:
        try:
            payload, malformed_lines = _decode_jsonl_payload(
                raw,
                jsonl_dict_only=jsonl_dict_only,
            )
            return payload, "jsonl", malformed_lines
        except (UnicodeDecodeError, ValueError):
            pass
    try:
        return orjson.loads(raw), "json", 0
    except (orjson.JSONDecodeError, ValueError) as exc:
        try:
            payload, malformed_lines = _decode_jsonl_payload(
                raw,
                jsonl_dict_only=jsonl_dict_only,
            )
        except (UnicodeDecodeError, ValueError):
            raise exc from None
        return payload, "jsonl", malformed_lines


def _infer_payload_provider(
    payload: Any,
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


def build_raw_payload_envelope(
    raw_content: Path | bytes | str | Any,
    *,
    source_path: str | Path | None,
    fallback_provider: str | Provider,
    payload_provider: str | Provider | None = None,
    jsonl_dict_only: bool = False,
) -> RawPayloadEnvelope:
    """Decode raw payload and attach canonical provider/wire-format identity.

    When *raw_content* is a :class:`~pathlib.Path`, JSONL payloads are
    streamed line-by-line from disk — enabling constant-memory parsing
    of multi-GB files.
    """
    normalized_path = str(source_path or "").lower()
    prefer_jsonl = normalized_path.endswith((".jsonl", ".jsonl.txt", ".ndjson"))
    preferred_provider = payload_provider or fallback_provider
    if not prefer_jsonl:
        runtime_provider = Provider.from_string(preferred_provider)
        prefer_jsonl = runtime_provider in {Provider.CLAUDE_CODE, Provider.CODEX}
    payload, wire_format, malformed_jsonl_lines = _decode_raw_payload(
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
    )


__all__ = ["RawPayloadEnvelope", "WireFormat", "build_raw_payload_envelope"]
