"""Tests for semantic value generation in the synthetic corpus system.

Verifies that SemanticValueGenerator produces role-appropriate values for
message_body, message_role, message_timestamp, and conversation_title
semantic roles, and that _text_for_role handles all known roles correctly.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from polylogue.schemas.synthetic.semantic_values import (
    SemanticValueGenerator,
    _ROLE_TEXTS,
    _text_for_role,
)
from polylogue.schemas.synthetic.showcase import ConversationTheme, _SHOWCASE_THEMES


# ---------------------------------------------------------------------------
# _text_for_role
# ---------------------------------------------------------------------------


class TestTextForRole:
    """Verify _text_for_role returns valid text for all known roles."""

    @pytest.mark.parametrize("role", list(_ROLE_TEXTS.keys()))
    def test_known_roles_return_nonempty_text(self, role: str) -> None:
        rng = random.Random(42)
        text = _text_for_role(rng, role)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_unknown_role_falls_back_to_user_texts(self) -> None:
        rng = random.Random(42)
        text = _text_for_role(rng, "nonexistent_role")
        assert text in _ROLE_TEXTS["user"]

    def test_themed_user_turn_returns_theme_content(self) -> None:
        theme = _SHOWCASE_THEMES[0]
        rng = random.Random(0)
        text = _text_for_role(rng, "user", turn_index=0, theme=theme)
        assert text in theme.user_turns

    def test_themed_assistant_turn_returns_theme_content(self) -> None:
        theme = _SHOWCASE_THEMES[0]
        rng = random.Random(0)
        text = _text_for_role(rng, "assistant", turn_index=1, theme=theme)
        assert text in theme.assistant_turns

    def test_themed_model_role_uses_assistant_turns(self) -> None:
        theme = _SHOWCASE_THEMES[0]
        rng = random.Random(0)
        text = _text_for_role(rng, "model", turn_index=1, theme=theme)
        assert text in theme.assistant_turns

    def test_themed_human_role_uses_user_turns(self) -> None:
        theme = _SHOWCASE_THEMES[0]
        rng = random.Random(0)
        text = _text_for_role(rng, "human", turn_index=0, theme=theme)
        assert text in theme.user_turns

    def test_turn_index_cycles_through_theme_turns(self) -> None:
        """Theme turns should cycle when turn_index exceeds available turns."""
        theme = _SHOWCASE_THEMES[0]
        rng = random.Random(0)
        n_user_turns = len(theme.user_turns)
        # Turn index beyond the number of available turns should wrap
        text = _text_for_role(rng, "user", turn_index=n_user_turns * 2, theme=theme)
        assert text in theme.user_turns


# ---------------------------------------------------------------------------
# SemanticValueGenerator — construction and role cycling
# ---------------------------------------------------------------------------


class TestSemanticValueGeneratorBasics:
    def test_default_role_cycle(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        assert gen.role_cycle == ["user", "assistant"]
        assert gen.current_role == "user"

    def test_custom_role_cycle(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), role_cycle=["human", "model"])
        assert gen.current_role == "human"
        gen.advance_turn()
        assert gen.current_role == "model"
        gen.advance_turn()
        assert gen.current_role == "human"

    def test_turn_index_starts_at_zero(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        assert gen.turn_index == 0

    def test_advance_turn_increments(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        gen.advance_turn()
        assert gen.turn_index == 1
        gen.advance_turn()
        assert gen.turn_index == 2


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — message_role
# ---------------------------------------------------------------------------


class TestSemanticRole:
    def test_generates_role_from_cycle(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = {"x-polylogue-semantic-role": "message_role"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value == "user"

    def test_role_respects_observed_values(self) -> None:
        gen = SemanticValueGenerator(random.Random(42))
        schema = {
            "x-polylogue-semantic-role": "message_role",
            "x-polylogue-values": ["user", "assistant", "system"],
        }
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value in {"user", "assistant", "system"}

    def test_role_uses_cycle_when_cycle_value_in_observed(self) -> None:
        gen = SemanticValueGenerator(
            random.Random(0),
            role_cycle=["human", "assistant"],
        )
        schema = {
            "x-polylogue-semantic-role": "message_role",
            "x-polylogue-values": ["human", "assistant"],
        }
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value == "human"  # current_role is "human"


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — message_body
# ---------------------------------------------------------------------------


class TestSemanticBody:
    def test_generates_nonempty_body_text(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = {"x-polylogue-semantic-role": "message_body"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, str)
        assert len(value) > 0

    def test_body_text_matches_current_role(self) -> None:
        gen = SemanticValueGenerator(random.Random(42))
        schema = {"x-polylogue-semantic-role": "message_body"}

        # Turn 0 = user
        _, user_text = gen.try_generate(schema)
        assert user_text in _ROLE_TEXTS["user"]

        gen.advance_turn()
        # Turn 1 = assistant
        _, assistant_text = gen.try_generate(schema)
        assert assistant_text in _ROLE_TEXTS["assistant"]

    def test_body_with_theme_uses_themed_content(self) -> None:
        theme = _SHOWCASE_THEMES[0]
        gen = SemanticValueGenerator(random.Random(0), theme=theme)
        schema = {"x-polylogue-semantic-role": "message_body"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value in theme.user_turns  # turn 0 = user


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — message_timestamp
# ---------------------------------------------------------------------------


class TestSemanticTimestamp:
    def test_generates_epoch_by_default(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = {"x-polylogue-semantic-role": "message_timestamp"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, float)
        assert value == 1700000000.0  # turn 0

    def test_sequential_timestamps_increase(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = {"x-polylogue-semantic-role": "message_timestamp"}

        _, ts0 = gen.try_generate(schema)
        gen.advance_turn()
        _, ts1 = gen.try_generate(schema)
        gen.advance_turn()
        _, ts2 = gen.try_generate(schema)

        assert ts0 < ts1 < ts2

    def test_iso8601_format(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = {
            "x-polylogue-semantic-role": "message_timestamp",
            "x-polylogue-format": "iso8601",
        }
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, str)
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(value)
        assert parsed.tzinfo is not None

    def test_unix_epoch_str_format(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = {
            "x-polylogue-semantic-role": "message_timestamp",
            "x-polylogue-format": "unix-epoch-str",
        }
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, str)
        assert float(value) == 1700000000.0

    def test_unix_epoch_format(self) -> None:
        gen = SemanticValueGenerator(random.Random(0), base_ts=1700000000.0)
        schema = {
            "x-polylogue-semantic-role": "message_timestamp",
            "x-polylogue-format": "unix-epoch",
        }
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, float)


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — conversation_title
# ---------------------------------------------------------------------------


class TestSemanticTitle:
    def test_generates_title_string(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = {"x-polylogue-semantic-role": "conversation_title"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert isinstance(value, str)
        assert len(value) > 0

    def test_themed_title_uses_theme(self) -> None:
        theme = _SHOWCASE_THEMES[0]
        gen = SemanticValueGenerator(random.Random(0), theme=theme)
        schema = {"x-polylogue-semantic-role": "conversation_title"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value == theme.title

    def test_title_with_observed_values(self) -> None:
        gen = SemanticValueGenerator(random.Random(42))
        schema = {
            "x-polylogue-semantic-role": "conversation_title",
            "x-polylogue-values": ["Alpha", "Beta", "Gamma"],
        }
        handled, value = gen.try_generate(schema)
        assert handled is True
        assert value in {"Alpha", "Beta", "Gamma"}

    def test_title_without_theme_or_values_picks_showcase_theme(self) -> None:
        gen = SemanticValueGenerator(random.Random(42))
        schema = {"x-polylogue-semantic-role": "conversation_title"}
        handled, value = gen.try_generate(schema)
        assert handled is True
        known_titles = {t.title for t in _SHOWCASE_THEMES}
        assert value in known_titles


# ---------------------------------------------------------------------------
# SemanticValueGenerator.try_generate — fallback for unknown/unhandled roles
# ---------------------------------------------------------------------------


class TestSemanticFallback:
    def test_no_semantic_role_returns_not_handled(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = {"type": "string"}
        handled, value = gen.try_generate(schema)
        assert handled is False
        assert value is None

    def test_unknown_semantic_role_returns_not_handled(self) -> None:
        gen = SemanticValueGenerator(random.Random(0))
        schema = {"x-polylogue-semantic-role": "completely_unknown_role"}
        handled, value = gen.try_generate(schema)
        assert handled is False
        assert value is None

    def test_message_container_role_returns_not_handled(self) -> None:
        """message_container is structural and defers to normal generation."""
        gen = SemanticValueGenerator(random.Random(0))
        schema = {"x-polylogue-semantic-role": "message_container"}
        handled, value = gen.try_generate(schema)
        assert handled is False
