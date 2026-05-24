"""Typed storage projection for provider-native events."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from polylogue.core.json import json_document

JSONObject = dict[str, object]


@dataclass(frozen=True, slots=True)
class ProviderEventStorageProjection:
    normalized_kind: str
    payload: JSONObject
    compaction: tuple[str, str | None, int | None, str | None, int, int] | None = None
    turn_context: tuple[str | None, str | None, str | None, str | None, str | None, str | None] | None = None
    tool_call: tuple[str | None, str | None, str | None, int, int, int, int] | None = None
    reasoning: tuple[str | None, str | None, int] | None = None
    ghost_snapshot: tuple[str | None] | None = None


def _string(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _bool_int(value: object) -> int:
    return int(bool(value))


def _payload_body(payload: Mapping[str, object]) -> JSONObject:
    raw = payload.get("raw")
    if isinstance(raw, Mapping):
        return cast(JSONObject, json_document(dict(raw)))
    return cast(JSONObject, json_document(dict(payload)))


def _hash_text(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def _body_chars(value: object) -> tuple[int, int]:
    if isinstance(value, str):
        return len(value), int(bool(value))
    if value is None:
        return 0, 0
    return 0, 1


def project_provider_event_payload(event_type: str, payload_value: object) -> ProviderEventStorageProjection:
    """Project parser payloads into compact typed storage rows.

    The parser payload remains the public reconstructed shape for now, but it is
    no longer stored as a raw JSON catchall in the hot provider_events table.
    Large exact provider bodies are recoverable through raw_id/source provenance.
    """

    payload = json_document(payload_value)
    body = _payload_body(payload)
    normalized = event_type.strip().lower().replace("-", "_") or "provider_native"

    if normalized == "compaction":
        summary = _string(payload.get("summary")) or _string(body.get("message")) or ""
        replacement_history = body.get("replacement_history")
        replacement_count = _int(payload.get("replacement_history_count"))
        if replacement_count is None and isinstance(replacement_history, list):
            replacement_count = len(replacement_history)
        compaction = (
            summary,
            _string(payload.get("trigger")) or _string(body.get("trigger")),
            _int(payload.get("pre_tokens")) or _int(body.get("pre_tokens")) or _int(body.get("preTokens")),
            _string(payload.get("preserved_segment_id")) or _string(body.get("preserved_segment_id")),
            _bool_int(payload.get("is_modern") or body.get("is_modern")),
            replacement_count or 0,
        )
        compact_payload: JSONObject = {
            "summary": compaction[0],
        }
        if compaction[5]:
            compact_payload["replacement_history_count"] = compaction[5]
        if compaction[1] is not None:
            compact_payload["trigger"] = compaction[1]
        if compaction[2] is not None:
            compact_payload["pre_tokens"] = compaction[2]
        if compaction[3] is not None:
            compact_payload["preserved_segment_id"] = compaction[3]
        if compaction[4]:
            compact_payload["is_modern"] = True
        return ProviderEventStorageProjection("compaction", compact_payload, compaction=compaction)

    if normalized == "turn_context":
        turn_context = (
            _string(payload.get("cwd")) or _string(body.get("cwd")),
            _string(payload.get("model")) or _string(body.get("model")),
            _string(payload.get("effort")) or _string(body.get("effort")),
            _string(payload.get("approval_policy")) or _string(body.get("approval_policy")),
            _string(payload.get("sandbox_policy")) or _string(body.get("sandbox_policy")),
            _string(payload.get("summary")) or _string(body.get("summary")),
        )
        compact_payload = {}
        for key, value in zip(
            ("cwd", "model", "effort", "approval_policy", "sandbox_policy", "summary"),
            turn_context,
            strict=True,
        ):
            if value is not None:
                compact_payload[key] = value
        return ProviderEventStorageProjection("turn_context", compact_payload, turn_context=turn_context)

    if normalized in {"function_call", "custom_tool_call", "function_call_output", "custom_tool_call_output"}:
        input_chars, has_input = _body_chars(
            body.get("arguments") if normalized == "function_call" else body.get("input")
        )
        output_chars, has_output = _body_chars(body.get("output"))
        tool_call = (
            _string(payload.get("call_id")) or _string(body.get("call_id")),
            _string(payload.get("name")) or _string(body.get("name")),
            _string(payload.get("status")) or _string(body.get("status")),
            input_chars,
            output_chars,
            has_input,
            has_output,
        )
        compact_payload = {
            "call_id": tool_call[0],
            "name": tool_call[1],
            "status": tool_call[2],
            "input_chars": tool_call[3],
            "output_chars": tool_call[4],
            "has_input_body": bool(tool_call[5]),
            "has_output_body": bool(tool_call[6]),
        }
        return ProviderEventStorageProjection(
            normalized, {k: v for k, v in compact_payload.items() if v is not None}, tool_call=tool_call
        )

    if normalized == "reasoning":
        encrypted = _string(body.get("encrypted_content"))
        summary_value = body.get("summary")
        summary_text: str | None
        if isinstance(summary_value, list):
            summary_text = "\n".join(str(item) for item in summary_value if item)
        else:
            summary_text = _string(summary_value)
        reasoning = (summary_text, _hash_text(encrypted), len(encrypted or ""))
        compact_payload = {}
        if summary_text:
            compact_payload["summary"] = summary_text
        if reasoning[1]:
            compact_payload["encrypted_content_hash"] = reasoning[1]
            compact_payload["encrypted_content_bytes"] = reasoning[2]
        return ProviderEventStorageProjection("reasoning", compact_payload, reasoning=reasoning)

    if normalized == "ghost_snapshot":
        ghost_snapshot = (_string(payload.get("ghost_commit")) or _string(body.get("ghost_commit")),)
        compact_payload = {"ghost_commit": ghost_snapshot[0]} if ghost_snapshot[0] else {}
        return ProviderEventStorageProjection("ghost_snapshot", compact_payload, ghost_snapshot=ghost_snapshot)

    return ProviderEventStorageProjection("provider_native", cast(JSONObject, payload))


__all__ = ["ProviderEventStorageProjection", "project_provider_event_payload"]
