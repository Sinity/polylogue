"""Schema-observation data models and provider configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocumentList, JSONValue

SchemaSampleGranularity: TypeAlias = Literal["document", "record"]
SchemaClusterPayload: TypeAlias = JSONValue
SCHEMA_SAMPLE_STRING_LIMIT = 1024
ObservationTerminalStatus: TypeAlias = Literal[
    "included",
    "intentionally_excluded",
    "decode_failed",
    "unsupported",
    "quarantined",
]


class ObservationTerminalRecorder(Protocol):
    """Sink for one raw artifact's terminal schema-observation outcome."""

    def __call__(
        self,
        *,
        raw_id: str,
        status: ObservationTerminalStatus,
        artifact_kind: str | None,
        source_path: str | None,
        reason: str | None,
    ) -> None: ...


@dataclass
class ProviderConfig:
    """Configuration for one provider's schema observation behavior."""

    name: Provider
    description: str
    db_source_name: Provider | None = None
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
    session_id: str | None = None
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
        db_source_name=Provider.CHATGPT,
        sample_granularity="document",
    ),
    Provider.CLAUDE_CODE: ProviderConfig(
        name=Provider.CLAUDE_CODE,
        description="Claude Code message format",
        db_source_name=Provider.CLAUDE_CODE,
        sample_granularity="record",
        record_type_key="type",
        schema_sample_cap=128,
    ),
    Provider.CLAUDE_AI: ProviderConfig(
        name=Provider.CLAUDE_AI,
        description="Claude AI web message format",
        db_source_name=Provider.CLAUDE_AI,
        sample_granularity="document",
    ),
    Provider.GEMINI: ProviderConfig(
        name=Provider.GEMINI,
        description="Gemini AI Studio message format",
        db_source_name=Provider.GEMINI,
        sample_granularity="document",
    ),
    Provider.GEMINI_CLI: ProviderConfig(
        name=Provider.GEMINI_CLI,
        description="Gemini CLI local session format",
        db_source_name=Provider.GEMINI_CLI,
        session_dir=Path.home() / ".gemini/tmp",
        sample_granularity="document",
    ),
    Provider.HERMES: ProviderConfig(
        name=Provider.HERMES,
        description="Hermes agent session format",
        db_source_name=Provider.HERMES,
        session_dir=Path.home() / ".hermes/sessions",
        sample_granularity="document",
    ),
    Provider.ANTIGRAVITY: ProviderConfig(
        name=Provider.ANTIGRAVITY,
        description="Antigravity local brain artifact metadata format",
        db_source_name=Provider.ANTIGRAVITY,
        session_dir=Path.home() / ".gemini/antigravity",
        sample_granularity="document",
    ),
    Provider.CODEX: ProviderConfig(
        name=Provider.CODEX,
        description="OpenAI Codex CLI session format",
        db_source_name=Provider.CODEX,
        session_dir=Path.home() / ".codex/sessions",
        max_sessions=100,
        sample_granularity="record",
        record_type_key="type",
        schema_sample_cap=128,
    ),
}


__all__ = [
    "ObservationTerminalRecorder",
    "ObservationTerminalStatus",
    "PROVIDERS",
    "ProviderConfig",
    "SCHEMA_SAMPLE_STRING_LIMIT",
    "SchemaClusterPayload",
    "SchemaSampleGranularity",
    "SchemaUnit",
]
