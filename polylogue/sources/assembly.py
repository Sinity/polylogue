"""Provider assembly layer — generic sidecar discovery and conversation enrichment.

Replaces Claude-specific session index logic with a protocol that any provider
can implement for sidecar discovery and post-parse enrichment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias

from typing_extensions import TypedDict

from polylogue.types import Provider

from .parsers.base import ParsedConversation

if TYPE_CHECKING:
    from .parsers.claude_index import SessionIndexEntry

ClaudeCodeSessionIndex: TypeAlias = dict[str, "SessionIndexEntry"]
CodexThreadNames: TypeAlias = dict[str, str]


class _ClaudeCodeSidecarData(TypedDict, total=False):
    session_index: ClaudeCodeSessionIndex


class _CodexSidecarData(TypedDict, total=False):
    thread_names: CodexThreadNames


class SidecarData(_ClaudeCodeSidecarData, _CodexSidecarData, total=False):
    pass


@dataclass(frozen=True, slots=True)
class TitleResolution:
    """Result of provider-specific title resolution."""

    title: str
    source: str  # e.g. "session-index:summary", "first-user-message", "session-id"


class ProviderAssemblySpec(Protocol):
    """Provider-specific sidecar discovery and conversation enrichment."""

    def discover_sidecars(self, source_paths: list[Path]) -> SidecarData:
        """Discover provider-specific sidecars from source paths.

        Returns opaque provider data keyed by parent directory or similar.
        """
        ...

    def enrich_conversation(
        self,
        conv: ParsedConversation,
        sidecar_data: SidecarData,
    ) -> ParsedConversation:
        """Enrich a parsed conversation using discovered sidecar data."""
        ...


def get_assembly_spec(provider: Provider) -> ProviderAssemblySpec | None:
    """Return the assembly spec for a provider, or None if no enrichment needed."""
    if provider is Provider.CLAUDE_CODE:
        from .assembly_claude_code import ClaudeCodeAssemblySpec

        return ClaudeCodeAssemblySpec()
    if provider is Provider.CODEX:
        from .assembly_codex import CodexAssemblySpec

        return CodexAssemblySpec()
    if provider is Provider.GEMINI:
        from .assembly_gemini import GeminiAssemblySpec

        return GeminiAssemblySpec()
    return None


__all__ = [
    "ClaudeCodeSessionIndex",
    "CodexThreadNames",
    "ProviderAssemblySpec",
    "SidecarData",
    "_CodexSidecarData",
    "_ClaudeCodeSidecarData",
    "TitleResolution",
    "get_assembly_spec",
]
