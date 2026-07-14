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


def test_role_synonyms_round_trip_through_normalize() -> None:
    """Coupling test: every synonym in ROLE_SYNONYMS must normalize to the expected canonical role.

    This ensures that ROLE_SYNONYMS (used for SQL filter expansion) and Role.normalize()
    (used for canonicalization) remain synchronized as a single source of truth.
    """
    for role_name, synonyms in ROLE_SYNONYMS.items():
        expected_role = Role(role_name)
        for synonym in synonyms:
            normalized = Role.normalize(synonym)
            assert normalized == expected_role, (
                f"Synonym {synonym!r} should normalize to {expected_role!r}, but got {normalized!r}"
            )


def test_role_sql_values_derived_from_role_synonyms() -> None:
    """Verify that ROLE_SQL_VALUES is correctly derived from ROLE_SYNONYMS.

    ROLE_SQL_VALUES should be generated from ROLE_SYNONYMS and contain all the same mappings.
    """
    for role, sql_values in ROLE_SQL_VALUES.items():
        role_name = role.value
        assert role_name in ROLE_SYNONYMS, f"Role {role_name!r} not found in ROLE_SYNONYMS"
        expected_synonyms = ROLE_SYNONYMS[role_name]
        assert set(sql_values) == expected_synonyms, (
            f"ROLE_SQL_VALUES[{role!r}] should contain {expected_synonyms!r}, but got {set(sql_values)!r}"
        )
