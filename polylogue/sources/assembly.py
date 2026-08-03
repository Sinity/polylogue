"""Provider assembly layer — generic sidecar discovery and session enrichment.

Replaces Claude-specific session index logic with a protocol that any provider
can implement for sidecar discovery and post-parse enrichment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias

from typing_extensions import TypedDict

from polylogue.core.enums import Provider
from polylogue.storage.blob_store import BlobStore

from .parsers.base import ParsedSession

if TYPE_CHECKING:
    from .parsers.chatgpt_sidecars import ChatGPTAssetIndex
    from .parsers.claude.history import HistoryEntry
    from .parsers.claude.index import SessionIndexEntry

ClaudeCodeSessionIndex: TypeAlias = dict[str, "SessionIndexEntry"]
ClaudeCodeHistoryPasteIndex: TypeAlias = dict[str, list["HistoryEntry"]]
CodexThreadNames: TypeAlias = dict[str, str]
CodexHistoryTitles: TypeAlias = dict[str, str]


class _ClaudeCodeSidecarData(TypedDict, total=False):
    session_index: ClaudeCodeSessionIndex
    history_paste_index: ClaudeCodeHistoryPasteIndex


class _CodexSidecarData(TypedDict, total=False):
    thread_names: CodexThreadNames
    history_titles: CodexHistoryTitles
    state_titles: CodexHistoryTitles


class _ClaudeAISidecarData(TypedDict, total=False):
    # bd polylogue-4zqh3: attachment native id -> (blob_hash_hex, size_bytes)
    # for sole-copy attachment bytes recovered out of band and streamed into
    # the blob store during sidecar discovery. See assembly_claude_ai.py.
    claude_ai_recovered_blobs: dict[str, tuple[str, int]]


class _ChatGPTSidecarData(TypedDict, total=False):
    # bd polylogue-0hwv / polylogue-dt5s: the merged asset-name/sandbox-file
    # resolver built once per source scan from conversation_asset_file_names.json
    # + library_files.json. See sources/assembly_chatgpt.py.
    chatgpt_asset_index: ChatGPTAssetIndex
    # bd polylogue-8ac0: dat asset id -> (blob_hash_hex, size_bytes) for every
    # ``.dat`` member/sibling file whose bytes were streamed into the blob
    # store during sidecar discovery. Attachment resolution joins against this
    # so previously-acquired dat bytes are marked "acquired" without
    # re-hashing (see ``ingest_batch/_core.py``'s ``preacquired_attachment_blobs``).
    chatgpt_dat_blobs: dict[str, tuple[str, int]]


class SidecarData(_ClaudeCodeSidecarData, _CodexSidecarData, _ChatGPTSidecarData, _ClaudeAISidecarData, total=False):
    pass


@dataclass(frozen=True, slots=True)
class TitleResolution:
    """Result of provider-specific title resolution."""

    title: str
    source: str  # e.g. "session-index:summary", "first-user-message", "session-id"


class ProviderAssemblySpec(Protocol):
    """Provider-specific sidecar discovery and session enrichment."""

    def discover_sidecars(
        self,
        source_paths: list[Path],
        *,
        blob_store: BlobStore | None = None,
    ) -> SidecarData:
        """Discover provider-specific sidecars from source paths.

        Returns opaque provider data keyed by parent directory or similar.

        ``blob_store`` is an optional content-addressed blob store/publisher
        a spec may use to stream sidecar-referenced bytes (not just metadata)
        into the archive during discovery -- see ``ChatGPTAssemblySpec``'s
        ``.dat`` asset acquisition (bd polylogue-8ac0). Specs that only need
        sidecar metadata ignore it.
        """
        ...

    def enrich_session(
        self,
        conv: ParsedSession,
        sidecar_data: SidecarData,
    ) -> ParsedSession:
        """Enrich a parsed session using discovered sidecar data."""
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
    if provider is Provider.CHATGPT:
        from .assembly_chatgpt import ChatGPTAssemblySpec

        return ChatGPTAssemblySpec()
    if provider is Provider.CLAUDE_AI:
        from .assembly_claude_ai import ClaudeAIAssemblySpec

        return ClaudeAIAssemblySpec()
    return None


__all__ = [
    "ClaudeCodeHistoryPasteIndex",
    "ClaudeCodeSessionIndex",
    "CodexHistoryTitles",
    "CodexThreadNames",
    "ProviderAssemblySpec",
    "SidecarData",
    "_ChatGPTSidecarData",
    "_ClaudeAISidecarData",
    "_CodexSidecarData",
    "_ClaudeCodeSidecarData",
    "TitleResolution",
    "get_assembly_spec",
]
