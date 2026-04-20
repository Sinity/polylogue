"""Schema-observation data models and provider configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from polylogue.lib.json import JSONDocumentList, JSONValue
from polylogue.types import Provider

SchemaSampleGranularity: TypeAlias = Literal["document", "record"]
SchemaClusterPayload: TypeAlias = JSONValue


@dataclass
class ProviderConfig:
    """Configuration for one provider's schema observation behavior."""

    name: Provider
    description: str
    db_provider_name: Provider | None = None
    session_dir: Path | None = None
    max_sessions: int | None = None
    sample_granularity: SchemaSampleGranularity = "document"
    record_type_key: str | None = None
    schema_sample_cap: int | None = None


@dataclass(frozen=True)
class SchemaUnit:
    """Clusterable schema-observation input."""

    cluster_payload: SchemaClusterPayload
    schema_samples: JSONDocumentList
    artifact_kind: str
    conversation_id: str | None = None
    raw_id: str | None = None
    source_path: str | None = None
    bundle_scope: str | None = None
    observed_at: str | None = None
    exact_structure_id: str = ""
    profile_tokens: tuple[str, ...] = ()


PROVIDERS: dict[Provider, ProviderConfig] = {
    Provider.CHATGPT: ProviderConfig(
        name=Provider.CHATGPT,
        description="ChatGPT message format",
        db_provider_name=Provider.CHATGPT,
        sample_granularity="document",
    ),
    Provider.CLAUDE_CODE: ProviderConfig(
        name=Provider.CLAUDE_CODE,
        description="Claude Code message format",
        db_provider_name=Provider.CLAUDE_CODE,
        sample_granularity="record",
        record_type_key="type",
        schema_sample_cap=128,
    ),
    Provider.CLAUDE_AI: ProviderConfig(
        name=Provider.CLAUDE_AI,
        description="Claude AI web message format",
        db_provider_name=Provider.CLAUDE_AI,
        sample_granularity="document",
    ),
    Provider.GEMINI: ProviderConfig(
        name=Provider.GEMINI,
        description="Gemini AI Studio message format",
        db_provider_name=Provider.GEMINI,
        sample_granularity="document",
    ),
    Provider.CODEX: ProviderConfig(
        name=Provider.CODEX,
        description="OpenAI Codex CLI session format",
        db_provider_name=Provider.CODEX,
        session_dir=Path.home() / ".codex/sessions",
        max_sessions=100,
        sample_granularity="record",
        record_type_key="type",
        schema_sample_cap=128,
    ),
}


__all__ = [
    "PROVIDERS",
    "ProviderConfig",
    "SchemaClusterPayload",
    "SchemaSampleGranularity",
    "SchemaUnit",
]
