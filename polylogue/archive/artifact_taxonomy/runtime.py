"""Artifact taxonomy classification runtime."""

from __future__ import annotations

from pathlib import Path

from polylogue.archive.artifact_taxonomy.models import ArtifactClassification, ArtifactKind
from polylogue.archive.artifact_taxonomy.support import (
    is_subagent_path,
    looks_like_hook_event,
    looks_like_hook_event_stream,
    looks_like_record_entry,
    looks_like_record_stream,
    looks_like_session_document,
    looks_metadataish_dict,
    looks_metadataish_list,
    normalize_source_path,
    path_only_sidecar_reason,
)
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue, json_document


def classify_artifact_path(
    source_path: str | Path | None,
    *,
    provider: str | Provider,
) -> ArtifactClassification | None:
    """Classify obvious sidecars using only the source path."""
    provider_token = Provider.from_string(provider)
    normalized = normalize_source_path(source_path)
    if not normalized:
        return None

    inner_name = Path(normalized.rsplit(":", 1)[-1]).name.lower()
    if provider_token is Provider.ANTIGRAVITY:
        if inner_name.endswith(".md.metadata.json"):
            return None
        if inner_name.endswith((".pb", ".pbtxt", ".resolved")) or ".resolved." in inner_name:
            return ArtifactClassification(
                provider=provider_token,
                kind=ArtifactKind.METADATA_DOCUMENT,
                parse_as_session=False,
                schema_eligible=False,
                default_priority=0,
                reason="Antigravity opaque or resolved sidecar",
            )
        if inner_name in {
            "browserallowlist.txt",
            "installation_id",
            "knowledge.lock",
            "mcp_config.json",
            "user_settings.pb",
        }:
            return ArtifactClassification(
                provider=provider_token,
                kind=ArtifactKind.METADATA_DOCUMENT,
                parse_as_session=False,
                schema_eligible=False,
                default_priority=0,
                reason="Antigravity configuration sidecar",
            )
    if sidecar_reason := path_only_sidecar_reason(inner_name):
        kind = ArtifactKind.BRIDGE_POINTER if inner_name == "bridge-pointer.json" else ArtifactKind.SESSION_INDEX
        return ArtifactClassification(
            provider=provider_token,
            kind=kind,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason=sidecar_reason,
        )

    if inner_name.startswith("agent-") and inner_name.endswith(".meta.json"):
        return ArtifactClassification(
            provider=provider_token,
            kind=ArtifactKind.AGENT_SIDECAR_META,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="agent sidecar metadata path",
        )

    return None


def classify_artifact(
    payload: JSONValue,
    *,
    provider: str | Provider,
    source_path: str | Path | None = None,
) -> ArtifactClassification:
    """Classify a payload/document into a session or sidecar cohort."""
    provider_token = Provider.from_string(provider)
    explicit = classify_artifact_path(source_path, provider=provider_token)
    if explicit is not None:
        return explicit

    if isinstance(payload, list):
        return _classify_list(payload, provider=provider_token, source_path=source_path)
    if isinstance(payload, dict):
        return _classify_dict(payload, provider=provider_token, source_path=source_path)
    return ArtifactClassification(
        provider=provider_token,
        kind=ArtifactKind.UNKNOWN,
        parse_as_session=False,
        schema_eligible=False,
        default_priority=0,
        reason="non-object payload",
    )


def _classify_list(
    payload: list[JSONValue],
    *,
    provider: Provider,
    source_path: str | Path | None,
) -> ArtifactClassification:
    dict_items = [json_document(item) for item in payload[:32]]
    dict_items = [item for item in dict_items if item]
    if not payload:
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.METADATA_DOCUMENT,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="empty list payload",
        )

    if dict_items and looks_like_hook_event_stream(dict_items):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.HOOK_EVENT,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=100,
            reason="hook event stream",
        )

    if dict_items and looks_like_record_stream(dict_items):
        subagent = is_subagent_path(source_path)
        kind = ArtifactKind.SUBAGENT_SESSION_STREAM if subagent else ArtifactKind.SESSION_RECORD_STREAM
        return ArtifactClassification(
            provider=provider,
            kind=kind,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=90 if subagent else 120,
            reason="record-like JSONL stream",
        )

    if dict_items and any(looks_like_session_document(item) for item in dict_items):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=120,
            reason="bundle of session documents",
        )

    if looks_metadataish_list(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.METADATA_DOCUMENT,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="metadata-oriented list payload",
        )

    return ArtifactClassification(
        provider=provider,
        kind=ArtifactKind.UNKNOWN,
        parse_as_session=False,
        schema_eligible=False,
        default_priority=0,
        reason="unrecognized list payload",
    )


def _classify_dict(
    payload: JSONDocument,
    *,
    provider: Provider,
    source_path: str | Path | None,
) -> ArtifactClassification:
    if provider is Provider.ANTIGRAVITY and _is_antigravity_markdown_export(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=120,
            reason="Antigravity language-server Markdown export",
        )

    if provider is Provider.ANTIGRAVITY and _is_antigravity_brain_metadata(payload, source_path):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=100,
            reason="Antigravity brain artifact metadata with sibling Markdown",
        )

    if looks_like_hook_event(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.HOOK_EVENT,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=100,
            reason="hook event record",
        )

    if looks_like_session_document(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SESSION_DOCUMENT,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=120,
            reason="session-bearing document",
        )

    if is_subagent_path(source_path) and looks_like_record_entry(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SUBAGENT_SESSION_STREAM,
            parse_as_session=True,
            schema_eligible=True,
            default_priority=90,
            reason="subagent record payload",
        )

    if looks_metadataish_dict(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.METADATA_DOCUMENT,
            parse_as_session=False,
            schema_eligible=False,
            default_priority=0,
            reason="metadata-oriented document",
        )

    return ArtifactClassification(
        provider=provider,
        kind=ArtifactKind.UNKNOWN,
        parse_as_session=False,
        schema_eligible=False,
        default_priority=0,
        reason="unrecognized document payload",
    )


def _is_antigravity_brain_metadata(payload: JSONDocument, source_path: str | Path | None) -> bool:
    normalized = normalize_source_path(source_path)
    name = Path(normalized.rsplit(":", 1)[-1]).name.lower() if normalized else ""
    return (
        (not name or name.endswith((".json", ".md.metadata.json")))
        and isinstance(payload.get("artifactType"), str)
        and ("summary" in payload or "updatedAt" in payload)
    )


def _is_antigravity_markdown_export(payload: JSONDocument) -> bool:
    return (
        payload.get("source") == "antigravity_language_server"
        and isinstance(payload.get("cascadeId"), str)
        and isinstance(payload.get("markdown"), str)
    )
