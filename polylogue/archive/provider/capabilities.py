"""Typed provider capability metadata used by verification surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.core.json import JSONDocument, require_json_value
from polylogue.types import Provider


@dataclass(frozen=True, slots=True)
class IdentityMapping:
    """Relationship between provider-native facts and canonical archive fields."""

    native: str
    canonical: str
    note: str

    def to_payload(self) -> JSONDocument:
        return {
            "native": self.native,
            "canonical": self.canonical,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class ProviderCapability:
    """Static capability contract for a provider parser and normalizer."""

    provider: Provider
    parser_identity: str
    parser_source_paths: tuple[str, ...]
    sidecar_spec: str
    tool_use_variant: str
    reasoning_capability: str
    streaming_capability: str
    native_identity_fields: tuple[str, ...]
    canonical_identity_fields: tuple[str, ...]
    identity_mappings: tuple[IdentityMapping, ...]
    timestamp_semantics: str
    coverage_facets: Mapping[str, str]
    partial_coverage: tuple[str, ...]

    @property
    def source_symbol(self) -> str:
        return f"PROVIDER_CAPABILITIES[{self.provider.value}]"

    def to_payload(self) -> JSONDocument:
        return {
            "provider": self.provider.value,
            "parser_identity": self.parser_identity,
            "parser_source_paths": list(self.parser_source_paths),
            "sidecar_spec": self.sidecar_spec,
            "tool_use_variant": self.tool_use_variant,
            "reasoning_capability": self.reasoning_capability,
            "streaming_capability": self.streaming_capability,
            "native_identity_fields": list(self.native_identity_fields),
            "canonical_identity_fields": list(self.canonical_identity_fields),
            "identity_mappings": [mapping.to_payload() for mapping in self.identity_mappings],
            "timestamp_semantics": self.timestamp_semantics,
            "coverage_facets": {
                key: require_json_value(value, context=f"coverage_facets.{key}")
                for key, value in self.coverage_facets.items()
            },
            "partial_coverage": list(self.partial_coverage),
        }


PROVIDER_CAPABILITIES: tuple[ProviderCapability, ...] = (
    ProviderCapability(
        provider=Provider.CODEX,
        parser_identity="Codex JSONL session envelope and response item stream",
        parser_source_paths=(
            "polylogue/sources/parsers/codex.py",
            "polylogue/sources/providers/codex.py",
        ),
        sidecar_spec="none; Codex facts are carried inline in session, response, and context records",
        tool_use_variant="content segments normalized through content_blocks_from_segments()",
        reasoning_capability="partial; reasoning is preserved when exported as content blocks, not inferred",
        streaming_capability="record stream with session metadata, response items, compactions, and context turns",
        native_identity_fields=(
            "session_meta.id",
            "session_meta.created_at",
            "response_item.payload.id",
            "payload.id",
            "git.repository_url",
            "timestamp",
        ),
        canonical_identity_fields=(
            "Conversation.provider_conversation_id",
            "Conversation.created_at",
            "Conversation.updated_at",
            "Message.provider_message_id",
            "Conversation.provider_meta.git.repository_url",
        ),
        identity_mappings=(
            IdentityMapping(
                native="session_meta.id",
                canonical="Conversation.provider_conversation_id",
                note="Session ids remain the native provider conversation id.",
            ),
            IdentityMapping(
                native="payload.id",
                canonical="Message.provider_message_id",
                note="Record item ids remain available as native provider message ids.",
            ),
            IdentityMapping(
                native="git.repository_url",
                canonical="Conversation.provider_meta.git.repository_url",
                note="Repository attribution is retained in provider metadata instead of folded into canonical ids.",
            ),
        ),
        timestamp_semantics="Session and message timestamps are parsed with parse_timestamp() and stored as canonical created/updated timestamps.",
        coverage_facets={
            "reasoning": "partial",
            "streaming": "supported",
            "sidecars": "absent",
            "tool_use": "supported",
            "native_identity": "supported",
            "timestamps": "supported",
        },
        partial_coverage=(
            "reasoning_capability_partial",
            "sidecar_spec_absent",
        ),
    ),
    ProviderCapability(
        provider=Provider.CLAUDE_CODE,
        parser_identity="Claude Code JSONL record stream with typed record models",
        parser_source_paths=(
            "polylogue/sources/parsers/claude_code_parser.py",
            "polylogue/sources/providers/claude_code_record.py",
        ),
        sidecar_spec="isSidechain and agent-* ids are normalized as sidechain/subagent branches",
        tool_use_variant="message.content tool_use/tool_result blocks normalized into canonical content blocks",
        reasoning_capability="supported; thinking blocks are extracted from provider content blocks",
        streaming_capability="record stream grouped by sessionId with parentUuid continuity",
        native_identity_fields=(
            "sessionId",
            "uuid",
            "parentUuid",
            "cwd",
            "message.model",
            "timestamp",
        ),
        canonical_identity_fields=(
            "Conversation.provider_conversation_id",
            "Message.provider_message_id",
            "Message.parent_message_provider_id",
            "Conversation.provider_meta.working_directories",
            "Conversation.provider_meta.models_used",
            "Conversation.created_at",
            "Conversation.updated_at",
        ),
        identity_mappings=(
            IdentityMapping(
                native="sessionId",
                canonical="Conversation.provider_conversation_id",
                note="Session ids remain the native provider conversation id.",
            ),
            IdentityMapping(
                native="uuid",
                canonical="Message.provider_message_id",
                note="Message uuids remain available as native provider message ids.",
            ),
            IdentityMapping(
                native="parentUuid",
                canonical="Message.parent_message_provider_id",
                note="Native parent pointers are preserved for branch and continuity reconstruction.",
            ),
            IdentityMapping(
                native="cwd",
                canonical="Conversation.provider_meta.working_directories",
                note="Working directory facts are retained as provider-native metadata.",
            ),
        ),
        timestamp_semantics="Record timestamps are parsed per message and collapsed into conversation created/updated bounds.",
        coverage_facets={
            "reasoning": "supported",
            "streaming": "supported",
            "sidecars": "partial",
            "tool_use": "supported",
            "native_identity": "supported",
            "timestamps": "supported",
        },
        partial_coverage=("sidecar_artifact_contract_not_modeled",),
    ),
    ProviderCapability(
        provider=Provider.CHATGPT,
        parser_identity="ChatGPT mapping graph export with typed conversation and node models",
        parser_source_paths=(
            "polylogue/sources/parsers/chatgpt.py",
            "polylogue/sources/providers/chatgpt.py",
        ),
        sidecar_spec="none; attachments and tool facts are embedded in mapping-node metadata",
        tool_use_variant="author/recipient/status metadata plus citations and code_execution provider_meta",
        reasoning_capability="supported when exported as thoughts or reasoning_recap content types",
        streaming_capability="snapshot mapping graph, not an append-only stream",
        native_identity_fields=(
            "conversation.id",
            "mapping node id",
            "message.id",
            "parent",
            "children",
            "create_time",
            "update_time",
        ),
        canonical_identity_fields=(
            "Conversation.provider_conversation_id",
            "Message.provider_message_id",
            "Message.parent_message_provider_id",
            "Message.branch_index",
            "Conversation.created_at",
            "Conversation.updated_at",
            "Message.provider_meta.raw",
        ),
        identity_mappings=(
            IdentityMapping(
                native="mapping node id",
                canonical="Message.provider_message_id",
                note="Mapping keys remain the native message identity when message.id is absent.",
            ),
            IdentityMapping(
                native="parent",
                canonical="Message.parent_message_provider_id",
                note="Native mapping graph edges are retained for threaded reconstruction.",
            ),
            IdentityMapping(
                native="message",
                canonical="Message.provider_meta.raw",
                note="Raw message metadata is retained for provider-native facts not projected into canonical fields.",
            ),
        ),
        timestamp_semantics="create_time/update_time floats are normalized into canonical conversation and message timestamps.",
        coverage_facets={
            "reasoning": "supported",
            "streaming": "absent",
            "sidecars": "absent",
            "tool_use": "partial",
            "native_identity": "supported",
            "timestamps": "supported",
        },
        partial_coverage=(
            "streaming_capability_absent",
            "sidecar_spec_absent",
            "tool_use_variant_partial",
        ),
    ),
)


def iter_provider_capabilities() -> tuple[ProviderCapability, ...]:
    """Return the static provider capability contracts."""
    return PROVIDER_CAPABILITIES


__all__ = [
    "IdentityMapping",
    "PROVIDER_CAPABILITIES",
    "ProviderCapability",
    "iter_provider_capabilities",
]
