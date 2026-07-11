"""Explicit provider/tool registry for semantic transcript cards.

Classification is exact and reviewable. Persisted ``semantic_type`` wins;
otherwise only aliases listed here specialize. Unknown names remain fallback
cards with raw evidence. The registry records only aliases grounded in the
repository's parser fixtures, parser record types, or standing classifier
contract; it does not speculate about a provider's open tool namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polylogue.core.enums import SemanticBlockType
from polylogue.core.json import JSONDocument
from polylogue.rendering.semantic_card_models import SemanticCardKind

MappingEvidenceKind = Literal["fixture_observed", "parser_record_type", "classifier_contract"]
RenderingStatus = Literal["launch", "model_only", "fallback"]


@dataclass(frozen=True, slots=True)
class ToolMapping:
    provider_family: str
    tool_name: str
    semantic_type: str
    card_kind: SemanticCardKind
    rendering_status: RenderingStatus
    evidence_kind: MappingEvidenceKind
    evidence: str

    def to_document(self) -> JSONDocument:
        return {
            "provider_family": self.provider_family,
            "tool_name": self.tool_name,
            "semantic_type": self.semantic_type,
            "card_kind": self.card_kind.value,
            "rendering_status": self.rendering_status,
            "evidence_kind": self.evidence_kind,
            "evidence": self.evidence,
        }


# These are exact aliases found in committed parser fixtures, parser record
# vocabulary, or the existing provider-neutral classifier. Provider tool-name
# spaces remain open: a name absent here is a raw fallback unless upstream has
# already persisted a trusted semantic_type.
_TOOL_MAPPINGS: tuple[ToolMapping, ...] = (
    # Claude Code: observed parser fixtures.
    ToolMapping(
        "claude-code",
        "Bash",
        "shell",
        SemanticCardKind.SHELL,
        "launch",
        "fixture_observed",
        "tests/unit/sources/test_parsers_base.py",
    ),
    ToolMapping(
        "claude-code",
        "Edit",
        "file_edit",
        SemanticCardKind.FILE_EDIT,
        "launch",
        "fixture_observed",
        "tests/unit/sources/test_parsers_base.py",
    ),
    ToolMapping(
        "claude-code",
        "Write",
        "file_write",
        SemanticCardKind.FILE_EDIT,
        "launch",
        "fixture_observed",
        "tests/unit/sources/test_parsers_base.py",
    ),
    ToolMapping(
        "claude-code",
        "NotebookEdit",
        "file_edit",
        SemanticCardKind.FILE_EDIT,
        "launch",
        "fixture_observed",
        "tests/unit/sources/test_parsers_base.py",
    ),
    ToolMapping(
        "claude-code",
        "Task",
        "subagent",
        SemanticCardKind.TASK,
        "model_only",
        "fixture_observed",
        "tests/unit/sources/test_parsers_base.py",
    ),
    ToolMapping(
        "claude-code",
        "Read",
        "file_read",
        SemanticCardKind.FALLBACK,
        "fallback",
        "fixture_observed",
        "tests/unit/sources/test_parsers_base.py",
    ),
    ToolMapping(
        "claude-code",
        "Grep",
        "search",
        SemanticCardKind.FALLBACK,
        "fallback",
        "fixture_observed",
        "tests/unit/sources/test_parsers_base.py",
    ),
    ToolMapping(
        "claude-code",
        "Glob",
        "search",
        SemanticCardKind.FALLBACK,
        "fallback",
        "fixture_observed",
        "tests/unit/sources/test_parsers_base.py",
    ),
    # Existing cross-provider classifier aliases that are meaningful for
    # Claude Code even where the parser fixture corpus has no dedicated row.
    ToolMapping(
        "claude-code",
        "MultiEdit",
        "file_edit",
        SemanticCardKind.FILE_EDIT,
        "launch",
        "classifier_contract",
        "polylogue/archive/viewport/tools.py",
    ),
    ToolMapping(
        "claude-code",
        "WebFetch",
        "web",
        SemanticCardKind.FALLBACK,
        "fallback",
        "classifier_contract",
        "polylogue/archive/viewport/tools.py",
    ),
    ToolMapping(
        "claude-code",
        "WebSearch",
        "web",
        SemanticCardKind.FALLBACK,
        "fallback",
        "classifier_contract",
        "polylogue/archive/viewport/tools.py",
    ),
    ToolMapping(
        "claude-code",
        "AskUserQuestion",
        "agent",
        SemanticCardKind.FALLBACK,
        "fallback",
        "classifier_contract",
        "polylogue/archive/viewport/tools.py",
    ),
    ToolMapping(
        "claude-code",
        "EnterPlanMode",
        "agent",
        SemanticCardKind.FALLBACK,
        "fallback",
        "classifier_contract",
        "polylogue/archive/viewport/tools.py",
    ),
    ToolMapping(
        "claude-code",
        "ExitPlanMode",
        "agent",
        SemanticCardKind.FALLBACK,
        "fallback",
        "classifier_contract",
        "polylogue/archive/viewport/tools.py",
    ),
    # Codex: exact function names observed in fixtures plus explicit envelope
    # record types whose fallback name is defined by the parser.
    ToolMapping(
        "codex",
        "exec_command",
        "shell",
        SemanticCardKind.SHELL,
        "launch",
        "fixture_observed",
        "tests/unit/sources/test_codex_event_stream_contract.py",
    ),
    ToolMapping(
        "codex",
        "shell",
        "shell",
        SemanticCardKind.SHELL,
        "launch",
        "fixture_observed",
        "tests/unit/sources/test_parsers_codex.py",
    ),
    ToolMapping(
        "codex",
        "apply_patch",
        "file_edit",
        SemanticCardKind.FILE_EDIT,
        "launch",
        "fixture_observed",
        "tests/unit/sources/test_parsers_codex.py",
    ),
    ToolMapping(
        "codex",
        "search",
        "search",
        SemanticCardKind.FALLBACK,
        "fallback",
        "fixture_observed",
        "tests/unit/sources/test_parsers_codex.py",
    ),
    ToolMapping(
        "codex",
        "local_shell_call",
        "shell",
        SemanticCardKind.SHELL,
        "launch",
        "parser_record_type",
        "polylogue/sources/parsers/codex.py",
    ),
    ToolMapping(
        "codex",
        "tool_search_call",
        "search",
        SemanticCardKind.FALLBACK,
        "fallback",
        "parser_record_type",
        "polylogue/sources/parsers/codex.py",
    ),
    ToolMapping(
        "codex",
        "web_search_call",
        "web",
        SemanticCardKind.FALLBACK,
        "fallback",
        "parser_record_type",
        "polylogue/sources/parsers/codex.py",
    ),
    ToolMapping(
        "codex",
        "spawn_agent",
        "subagent",
        SemanticCardKind.TASK,
        "model_only",
        "classifier_contract",
        "polylogue/archive/viewport/tools.py",
    ),
    # Gemini CLI and Hermes use open function-name shapes. Only exact names
    # observed in committed fixtures are listed; persisted semantic_type is the
    # safe path for every other provider-private name.
    ToolMapping(
        "gemini-cli",
        "read_file",
        "file_read",
        SemanticCardKind.FALLBACK,
        "fallback",
        "fixture_observed",
        "tests/unit/sources/test_parsers_local_agent.py",
    ),
    ToolMapping(
        "hermes",
        "shell",
        "shell",
        SemanticCardKind.SHELL,
        "launch",
        "fixture_observed",
        "tests/unit/sources/test_parsers_local_agent.py",
    ),
    ToolMapping(
        "hermes",
        "run_shell_command",
        "shell",
        SemanticCardKind.SHELL,
        "launch",
        "fixture_observed",
        "tests/unit/sources/test_parsers_local_agent.py",
    ),
    # ChatGPT recipient names form an open namespace. These are the exact
    # recipient-addressed tools present in parser regressions.
    ToolMapping(
        "chatgpt",
        "canmore.update_textdoc",
        "file_edit",
        SemanticCardKind.FILE_EDIT,
        "launch",
        "fixture_observed",
        "tests/unit/sources/test_parsers_chatgpt.py",
    ),
    ToolMapping(
        "chatgpt",
        "web",
        "web",
        SemanticCardKind.FALLBACK,
        "fallback",
        "fixture_observed",
        "tests/unit/sources/test_parsers_chatgpt.py",
    ),
    ToolMapping(
        "chatgpt",
        "dalle.text2im",
        "other",
        SemanticCardKind.FALLBACK,
        "fallback",
        "fixture_observed",
        "tests/unit/sources/test_parsers_chatgpt.py",
    ),
)

_TOOL_MAPPING_INDEX = {
    (mapping.provider_family.casefold(), mapping.tool_name.casefold()): mapping for mapping in _TOOL_MAPPINGS
}

_SEMANTIC_CARD_KIND: dict[str, SemanticCardKind] = {
    SemanticBlockType.SHELL.value: SemanticCardKind.SHELL,
    SemanticBlockType.GIT.value: SemanticCardKind.SHELL,
    SemanticBlockType.FILE_EDIT.value: SemanticCardKind.FILE_EDIT,
    SemanticBlockType.FILE_WRITE.value: SemanticCardKind.FILE_EDIT,
    SemanticBlockType.SUBAGENT.value: SemanticCardKind.TASK,
}

_PROVIDER_FAMILIES = ("chatgpt", "claude-code", "codex", "gemini-cli", "hermes")


def normalize_provider_family(value: object) -> str:
    """Normalize provider/origin values to registry families."""

    raw = getattr(value, "value", value)
    token = str(raw or "unknown").strip().lower().replace("_", "-")
    aliases = {
        "claude-code-session": "claude-code",
        "codex-session": "codex",
        "gemini-cli-session": "gemini-cli",
        "hermes-session": "hermes",
        "chatgpt-export": "chatgpt",
        "claude-ai-export": "claude-ai",
        "aistudio-drive": "gemini-cli",
    }
    return aliases.get(token, token)


def tool_mapping_rows() -> tuple[ToolMapping, ...]:
    return _TOOL_MAPPINGS


def provider_namespace_documents() -> list[JSONDocument]:
    """Describe the honest behavior outside the finite exact-alias table."""

    counts = dict.fromkeys(_PROVIDER_FAMILIES, 0)
    for row in _TOOL_MAPPINGS:
        counts[row.provider_family] += 1
    return [
        {
            "provider_family": provider,
            "namespace": "open",
            "grounded_exact_aliases": counts[provider],
            "unlisted_behavior": "persisted_semantic_type_then_fallback_raw_evidence",
        }
        for provider in _PROVIDER_FAMILIES
    ]


def card_kind_for_tool(
    *,
    provider_family: str,
    tool_name: str | None,
    semantic_type: str | None,
) -> SemanticCardKind:
    """Resolve a card kind from persisted semantics or exact aliases."""

    if semantic_type:
        persisted = _SEMANTIC_CARD_KIND.get(semantic_type.strip().lower())
        if persisted is not None:
            return persisted
    if tool_name:
        row = _TOOL_MAPPING_INDEX.get((normalize_provider_family(provider_family), tool_name.casefold()))
        if row is not None:
            return row.card_kind
    return SemanticCardKind.FALLBACK


def semantic_type_policy_documents() -> list[JSONDocument]:
    """Map every persisted semantic family to its current honest card behavior."""

    status_by_type: dict[str, RenderingStatus] = {
        SemanticBlockType.FILE_WRITE.value: "launch",
        SemanticBlockType.FILE_EDIT.value: "launch",
        SemanticBlockType.SHELL.value: "launch",
        SemanticBlockType.GIT.value: "launch",
        SemanticBlockType.SUBAGENT.value: "model_only",
    }
    return [
        {
            "semantic_type": semantic_type.value,
            "card_kind": _SEMANTIC_CARD_KIND.get(semantic_type.value, SemanticCardKind.FALLBACK).value,
            "rendering_status": status_by_type.get(semantic_type.value, "fallback"),
            "classification_basis": "persisted_semantic_type",
        }
        for semantic_type in SemanticBlockType
    ]


def tool_mapping_documents() -> list[JSONDocument]:
    return [row.to_document() for row in _TOOL_MAPPINGS]


__all__ = [
    "card_kind_for_tool",
    "MappingEvidenceKind",
    "normalize_provider_family",
    "provider_namespace_documents",
    "RenderingStatus",
    "semantic_type_policy_documents",
    "ToolMapping",
    "tool_mapping_documents",
    "tool_mapping_rows",
]
