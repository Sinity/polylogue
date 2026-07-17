"""Explicit provider/tool registry for semantic transcript rendering.

Persisted semantic types, structural protocol identities, and exact aliases are
separate evidence classes.  Unknown names remain generic fallback cards with
raw evidence; no classification is derived from tool prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polylogue.core.enums import Origin, SemanticBlockType
from polylogue.core.json import JSONDocument
from polylogue.core.tool_identity import parse_mcp_tool_name
from polylogue.rendering.semantic_card_models import SemanticCardKind

MappingEvidenceKind = Literal["fixture_observed", "parser_record_type", "classifier_contract"]
RenderingStatus = Literal["launch", "model_only", "fallback"]
ClassificationBasis = Literal[
    "mcp_tool_identity",
    "persisted_semantic_type",
    "exact_alias",
    "generic_fallback",
]


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


@dataclass(frozen=True, slots=True)
class ToolClassification:
    """One explicit classification decision used by every presentation leaf."""

    card_kind: SemanticCardKind
    semantic_type: str
    basis: ClassificationBasis
    rendering_status: RenderingStatus
    reason: str


@dataclass(frozen=True, slots=True)
class OriginPolicy:
    origin: Origin
    provider_family: str
    namespace: Literal["open", "closed"] = "open"
    unlisted_behavior: str = "structural_mcp_then_persisted_semantic_type_then_fallback_raw_evidence"

    def to_document(self, *, grounded_exact_aliases: int) -> JSONDocument:
        return {
            "origin": self.origin.value,
            "provider_family": self.provider_family,
            "namespace": self.namespace,
            "grounded_exact_aliases": grounded_exact_aliases,
            "unlisted_behavior": self.unlisted_behavior,
        }


# Exact aliases found in committed parser fixtures, parser record vocabulary,
# or the existing provider-neutral classifier.  Provider namespaces stay open.
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
        SemanticCardKind.FILE_READ,
        "launch",
        "fixture_observed",
        "tests/unit/sources/test_parsers_base.py",
    ),
    ToolMapping(
        "claude-code",
        "Grep",
        "search",
        SemanticCardKind.SEARCH,
        "launch",
        "fixture_observed",
        "tests/unit/sources/test_parsers_base.py",
    ),
    ToolMapping(
        "claude-code",
        "Glob",
        "search",
        SemanticCardKind.SEARCH,
        "launch",
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
        SemanticCardKind.WEB,
        "launch",
        "classifier_contract",
        "polylogue/archive/viewport/tools.py",
    ),
    ToolMapping(
        "claude-code",
        "WebSearch",
        "web",
        SemanticCardKind.WEB,
        "launch",
        "classifier_contract",
        "polylogue/archive/viewport/tools.py",
    ),
    ToolMapping(
        "claude-code",
        "AskUserQuestion",
        "agent",
        SemanticCardKind.TASK,
        "model_only",
        "classifier_contract",
        "polylogue/archive/viewport/tools.py",
    ),
    ToolMapping(
        "claude-code",
        "EnterPlanMode",
        "agent",
        SemanticCardKind.TASK,
        "model_only",
        "classifier_contract",
        "polylogue/archive/viewport/tools.py",
    ),
    ToolMapping(
        "claude-code",
        "ExitPlanMode",
        "agent",
        SemanticCardKind.TASK,
        "model_only",
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
        SemanticCardKind.SEARCH,
        "launch",
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
        SemanticCardKind.SEARCH,
        "launch",
        "parser_record_type",
        "polylogue/sources/parsers/codex.py",
    ),
    ToolMapping(
        "codex",
        "web_search_call",
        "web",
        SemanticCardKind.WEB,
        "launch",
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
        SemanticCardKind.FILE_READ,
        "launch",
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
        SemanticCardKind.WEB,
        "launch",
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
    SemanticBlockType.FILE_READ.value: SemanticCardKind.FILE_READ,
    SemanticBlockType.FILE_WRITE.value: SemanticCardKind.FILE_EDIT,
    SemanticBlockType.FILE_EDIT.value: SemanticCardKind.FILE_EDIT,
    SemanticBlockType.SHELL.value: SemanticCardKind.SHELL,
    SemanticBlockType.GIT.value: SemanticCardKind.SHELL,
    SemanticBlockType.SEARCH.value: SemanticCardKind.SEARCH,
    SemanticBlockType.WEB.value: SemanticCardKind.WEB,
    SemanticBlockType.AGENT.value: SemanticCardKind.TASK,
    SemanticBlockType.SUBAGENT.value: SemanticCardKind.TASK,
}

_SEMANTIC_RENDERING_STATUS: dict[str, RenderingStatus] = {
    SemanticBlockType.FILE_READ.value: "launch",
    SemanticBlockType.FILE_WRITE.value: "launch",
    SemanticBlockType.FILE_EDIT.value: "launch",
    SemanticBlockType.SHELL.value: "launch",
    SemanticBlockType.GIT.value: "launch",
    SemanticBlockType.SEARCH.value: "launch",
    SemanticBlockType.WEB.value: "launch",
    SemanticBlockType.AGENT.value: "model_only",
    SemanticBlockType.SUBAGENT.value: "model_only",
}

_SEMANTIC_FALLBACK_REASON: dict[str, str] = {
    SemanticBlockType.OTHER.value: "the persisted semantic family is intentionally generic",
    SemanticBlockType.THINKING.value: "thinking is rendered from typed content blocks, not as a tool card",
}

_ORIGIN_POLICIES: tuple[OriginPolicy, ...] = (
    OriginPolicy(Origin.CLAUDE_CODE_SESSION, "claude-code"),
    OriginPolicy(Origin.CODEX_SESSION, "codex"),
    OriginPolicy(Origin.GEMINI_CLI_SESSION, "gemini-cli"),
    OriginPolicy(Origin.HERMES_SESSION, "hermes"),
    OriginPolicy(Origin.ANTIGRAVITY_SESSION, "antigravity"),
    OriginPolicy(Origin.BEADS_ISSUE, "beads"),
    OriginPolicy(Origin.GROK_EXPORT, "grok"),
    OriginPolicy(Origin.CHATGPT_EXPORT, "chatgpt"),
    OriginPolicy(Origin.CLAUDE_AI_EXPORT, "claude-ai"),
    OriginPolicy(Origin.AISTUDIO_DRIVE, "gemini-cli"),
    OriginPolicy(Origin.UNKNOWN_EXPORT, "unknown"),
)
_ORIGIN_POLICY_INDEX = {policy.origin.value: policy for policy in _ORIGIN_POLICIES}


def normalize_provider_family(value: object) -> str:
    """Normalize an executable origin/provider token to a registry family."""

    raw = getattr(value, "value", value)
    token = str(raw or "unknown").strip().lower().replace("_", "-")
    policy = _ORIGIN_POLICY_INDEX.get(token)
    if policy is not None:
        return policy.provider_family
    aliases = {
        "claude-code": "claude-code",
        "codex": "codex",
        "gemini": "gemini-cli",
        "gemini-cli": "gemini-cli",
        "hermes": "hermes",
        "chatgpt": "chatgpt",
        "claude-ai": "claude-ai",
        "antigravity": "antigravity",
        "beads": "beads",
        "grok": "grok",
        "drive": "gemini-cli",
    }
    return aliases.get(token, token)


def classify_tool(
    *,
    provider_family: str,
    tool_name: str | None,
    semantic_type: str | None,
) -> ToolClassification:
    """Resolve one explicit card policy without prose-derived semantics."""

    mcp_identity = parse_mcp_tool_name(tool_name)
    normalized_semantic = semantic_type.strip().lower() if semantic_type else None
    if mcp_identity is not None:
        return ToolClassification(
            card_kind=SemanticCardKind.MCP,
            semantic_type=normalized_semantic or SemanticBlockType.OTHER.value,
            basis="mcp_tool_identity",
            rendering_status="launch",
            reason=f"tool name structurally identifies MCP server {mcp_identity.server!r}",
        )
    if normalized_semantic:
        kind = _SEMANTIC_CARD_KIND.get(normalized_semantic)
        if kind is not None:
            return ToolClassification(
                card_kind=kind,
                semantic_type=normalized_semantic,
                basis="persisted_semantic_type",
                rendering_status=_SEMANTIC_RENDERING_STATUS[normalized_semantic],
                reason="persisted semantic_type is authoritative",
            )
        return ToolClassification(
            card_kind=SemanticCardKind.FALLBACK,
            semantic_type=normalized_semantic,
            basis="persisted_semantic_type",
            rendering_status="fallback",
            reason=_SEMANTIC_FALLBACK_REASON.get(
                normalized_semantic,
                "the persisted semantic family has no specialized card policy",
            ),
        )
    if tool_name:
        row = _TOOL_MAPPING_INDEX.get((normalize_provider_family(provider_family), tool_name.casefold()))
        if row is not None:
            return ToolClassification(
                card_kind=row.card_kind,
                semantic_type=row.semantic_type,
                basis="exact_alias",
                rendering_status=row.rendering_status,
                reason=f"exact alias is grounded by {row.evidence_kind}: {row.evidence}",
            )
    return ToolClassification(
        card_kind=SemanticCardKind.FALLBACK,
        semantic_type=SemanticBlockType.OTHER.value,
        basis="generic_fallback",
        rendering_status="fallback",
        reason="no persisted semantic_type, MCP identity, or grounded exact alias is present",
    )


def card_kind_for_tool(
    *,
    provider_family: str,
    tool_name: str | None,
    semantic_type: str | None,
) -> SemanticCardKind:
    """Compatibility wrapper returning the shared classification's card kind."""

    return classify_tool(
        provider_family=provider_family,
        tool_name=tool_name,
        semantic_type=semantic_type,
    ).card_kind


def tool_mapping_rows() -> tuple[ToolMapping, ...]:
    return _TOOL_MAPPINGS


def origin_policy_rows() -> tuple[OriginPolicy, ...]:
    return _ORIGIN_POLICIES


def origin_policy_documents() -> list[JSONDocument]:
    counts: dict[str, int] = {}
    for row in _TOOL_MAPPINGS:
        counts[row.provider_family] = counts.get(row.provider_family, 0) + 1
    return [
        policy.to_document(grounded_exact_aliases=counts.get(policy.provider_family, 0)) for policy in _ORIGIN_POLICIES
    ]


def provider_namespace_documents() -> list[JSONDocument]:
    """Backward-compatible name for the exhaustive executable-origin policy."""

    return origin_policy_documents()


def semantic_type_policy_documents() -> list[JSONDocument]:
    """Map every persisted semantic family to an explicit renderer behavior."""

    return [
        {
            "semantic_type": semantic_type.value,
            "card_kind": _SEMANTIC_CARD_KIND.get(semantic_type.value, SemanticCardKind.FALLBACK).value,
            "rendering_status": _SEMANTIC_RENDERING_STATUS.get(semantic_type.value, "fallback"),
            "classification_basis": "persisted_semantic_type",
            "fallback_reason": _SEMANTIC_FALLBACK_REASON.get(semantic_type.value),
        }
        for semantic_type in SemanticBlockType
    ]


def tool_mapping_documents() -> list[JSONDocument]:
    return [row.to_document() for row in _TOOL_MAPPINGS]


__all__ = [
    "card_kind_for_tool",
    "ClassificationBasis",
    "classify_tool",
    "MappingEvidenceKind",
    "normalize_provider_family",
    "OriginPolicy",
    "origin_policy_documents",
    "origin_policy_rows",
    "provider_namespace_documents",
    "RenderingStatus",
    "semantic_type_policy_documents",
    "ToolClassification",
    "ToolMapping",
    "tool_mapping_documents",
    "tool_mapping_rows",
]
