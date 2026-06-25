from __future__ import annotations

from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import (
    BlockType,
    Origin,
    PasteBoundary,
    Provider,
    enum_values,
    nullable_sql_check_in,
    sql_check_in,
    sql_value_list,
)
from polylogue.core.enums import BlockType as TypesBlockType
from polylogue.core.enums import Provider as LegacyProvider


def test_core_enums_preserve_legacy_import_identity() -> None:
    assert LegacyProvider is Provider
    # polylogue.types re-exports the same canonical BlockType object as core.enums.
    assert TypesBlockType is BlockType
    assert Role.normalize("human") is Role.USER
    assert MessageType.normalize("tool-use") is MessageType.TOOL_USE
    assert BranchType.SIDECHAIN.value == "sidechain"


def test_origin_values_match_archive_issue_contract() -> None:
    assert enum_values(Origin) == (
        "claude-code-session",
        "codex-session",
        "gemini-cli-session",
        "hermes-session",
        "antigravity-session",
        "grok-export",
        "chatgpt-export",
        "claude-ai-export",
        "aistudio-drive",
        "unknown-export",
    )


def test_block_type_is_single_canonical_block_vocabulary() -> None:
    # ContentBlockType (parse-side) and BlockType (storage-side) were merged into
    # one canonical enum (#1743); BlockType carries the full stored+parsed block
    # vocabulary, and the blocks.block_type CHECK validates against exactly this set.
    assert enum_values(BlockType) == (
        "text",
        "thinking",
        "reasoning",
        "tool_use",
        "tool_result",
        "image",
        "code",
        "document",
    )


def test_sql_check_helpers_render_stable_sqlite_literals() -> None:
    assert sql_value_list(PasteBoundary) == ("'exact', 'projected', 'whole_message_fallback', 'hash_only'")
    assert sql_check_in("paste_boundary", PasteBoundary) == (
        "paste_boundary IN ('exact', 'projected', 'whole_message_fallback', 'hash_only')"
    )
    assert nullable_sql_check_in("paste_boundary", PasteBoundary) == (
        "(paste_boundary IN ('exact', 'projected', 'whole_message_fallback', 'hash_only') OR paste_boundary IS NULL)"
    )


def test_provider_aliases_still_normalize_during_transition() -> None:
    assert Provider.from_string("openai") is Provider.CHATGPT
    assert Provider.from_string("claude") is Provider.CLAUDE_AI
    assert str(Provider.CODEX) == "codex"
