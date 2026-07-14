from __future__ import annotations

from polylogue.archive.message.roles import ROLE_SQL_VALUES, Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import (
    ROLE_SYNONYMS,
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


def test_role_synonyms_and_sql_values_are_coupled() -> None:
    """Verify ROLE_SYNONYMS and ROLE_SQL_VALUES stay in sync (coupling test).

    ROLE_SYNONYMS in core/enums.py is the single source of truth for role synonyms.
    ROLE_SQL_VALUES in archive/message/roles.py is derived from ROLE_SYNONYMS.
    This test ensures they remain coupled.
    """
    # Every canonical role in ROLE_SYNONYMS has corresponding entry in ROLE_SQL_VALUES
    assert set(ROLE_SYNONYMS.keys()) == set(ROLE_SQL_VALUES.keys())

    # Every synonym in ROLE_SYNONYMS round-trips through Role.normalize
    for canonical_role, synonyms in ROLE_SYNONYMS.items():
        for synonym in synonyms:
            normalized = Role.normalize(synonym)
            assert normalized is canonical_role, (
                f"Synonym {synonym!r} normalized to {normalized}, expected {canonical_role}"
            )

    # ROLE_SQL_VALUES values match ROLE_SYNONYMS (as sorted tuples)
    for canonical_role, synonyms in ROLE_SYNONYMS.items():
        sql_values = ROLE_SQL_VALUES[canonical_role]
        expected_values = tuple(sorted(synonyms))
        assert sql_values == expected_values, (
            f"ROLE_SQL_VALUES[{canonical_role}] = {sql_values}, expected {expected_values}"
        )
