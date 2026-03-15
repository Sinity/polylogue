"""Semantic-role-driven value generation for synthetic data.

Uses ``x-polylogue-semantic-role`` annotations on schema fields to generate
contextually appropriate values instead of random noise.  Falls back to
generic text generation when no semantic role is present.

Semantic roles handled:
    - message_role: choose from observed role values
    - message_body: generate plausible content (themed in showcase mode)
    - message_timestamp: sequential timestamps from a base
    - conversation_title: short titles aligned with theme
"""

from __future__ import annotations

import random
from typing import Any

from polylogue.schemas.synthetic.showcase import ConversationTheme, _SHOWCASE_THEMES


# =============================================================================
# Role-based text generation (the original _text_for_role logic)
# =============================================================================

_ROLE_TEXTS: dict[str, list[str]] = {
    "user": [
        "Can you help me debug this issue?",
        "I need to implement a function that processes this data.",
        "What's the best approach for handling errors here?",
        "Could you review this code for potential issues?",
    ],
    "assistant": [
        "I'll analyze the issue. Looking at the code structure...",
        "Here's an implementation:\n\n```python\ndef process(data):\n    return [x for x in data if x]\n```",
        "After reviewing, I found several areas for improvement.",
        "The module structure looks good. A few suggestions:",
    ],
    "system": ["You are a helpful programming assistant."],
    "human": ["Can you explain how this works?", "I'm trying to understand the architecture."],
    "model": ["I'll explain step by step.", "Here's a breakdown of the architecture."],
    "tool": ["Function executed successfully.", "Error: resource not found."],
}


def _text_for_role(
    rng: random.Random,
    role: str,
    *,
    turn_index: int | None = None,
    theme: ConversationTheme | None = None,
) -> str:
    """Generate plausible text content for a given role."""
    if theme is not None and turn_index is not None:
        exchange_idx = max(turn_index, 0) // 2
        if role in {"user", "human"} and theme.user_turns:
            return theme.user_turns[exchange_idx % len(theme.user_turns)]
        if role in {"assistant", "model"} and theme.assistant_turns:
            return theme.assistant_turns[exchange_idx % len(theme.assistant_turns)]
    texts = _ROLE_TEXTS.get(role, _ROLE_TEXTS["user"])
    return rng.choice(texts)


# =============================================================================
# Semantic Value Generator
# =============================================================================


class SemanticValueGenerator:
    """Generate values driven by ``x-polylogue-semantic-role`` annotations.

    This replaces the generic schema-driven string/number generation for
    fields that have been identified as semantically meaningful.  The
    generator is stateful within a single conversation to produce coherent
    turn sequences.
    """

    def __init__(
        self,
        rng: random.Random,
        *,
        theme: ConversationTheme | None = None,
        base_ts: float = 1700000000.0,
        role_cycle: list[str] | None = None,
    ) -> None:
        self.rng = rng
        self.theme = theme
        self.base_ts = base_ts
        self.role_cycle = role_cycle or ["user", "assistant"]
        self._turn_index = 0

    def try_generate(
        self,
        schema: dict[str, Any],
    ) -> tuple[bool, Any]:
        """Attempt semantic generation for a schema node.

        Returns:
            (handled, value) — if handled is False, caller should use
            generic schema-driven generation instead.
        """
        role = schema.get("x-polylogue-semantic-role")
        if role is None:
            return False, None

        match role:
            case "message_role":
                return True, self._generate_role(schema)
            case "message_body":
                return True, self._generate_body(schema)
            case "message_timestamp":
                return True, self._generate_timestamp(schema)
            case "conversation_title":
                return True, self._generate_title(schema)
            case "message_container":
                # Containers are structural — let the normal generator handle them
                return False, None
            case _:
                return False, None

    def advance_turn(self) -> None:
        """Advance the internal turn counter."""
        self._turn_index += 1

    @property
    def current_role(self) -> str:
        """Current role in the alternation cycle."""
        return self.role_cycle[self._turn_index % len(self.role_cycle)]

    @property
    def turn_index(self) -> int:
        return self._turn_index

    def _generate_role(self, schema: dict[str, Any]) -> str:
        """Generate a role value from observed values or cycle."""
        # Prefer observed values from the schema
        if values := schema.get("x-polylogue-values"):
            # Filter to known conversational roles if present
            conversational = [v for v in values if v in {"user", "assistant", "human", "model", "system", "tool"}]
            if conversational:
                return self.current_role if self.current_role in conversational else self.rng.choice(conversational)
            return self.rng.choice(values)
        return self.current_role

    def _generate_body(self, schema: dict[str, Any]) -> str:
        """Generate message body content."""
        return _text_for_role(
            self.rng,
            self.current_role,
            turn_index=self._turn_index,
            theme=self.theme,
        )

    def _generate_timestamp(self, schema: dict[str, Any]) -> Any:
        """Generate a sequential timestamp value.

        Respects the field's format annotation to produce the right
        representation (epoch float, ISO string, etc.).
        """
        from datetime import datetime, timezone

        ts = self.base_ts + self._turn_index * 60  # 1-minute intervals

        fmt = schema.get("x-polylogue-format")
        match fmt:
            case "iso8601":
                return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            case "unix-epoch-str":
                return str(ts)
            case "unix-epoch" | None:
                return ts
            case _:
                return ts

    def _generate_title(self, schema: dict[str, Any]) -> str:
        """Generate a conversation title."""
        if self.theme is not None:
            return self.theme.title

        # Use observed values if available
        if values := schema.get("x-polylogue-values"):
            return self.rng.choice(values)

        # Fallback: pick a theme title
        return self.rng.choice(_SHOWCASE_THEMES).title


__all__ = [
    "SemanticValueGenerator",
    "_ROLE_TEXTS",
    "_text_for_role",
]
