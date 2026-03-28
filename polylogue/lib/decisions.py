"""Decision extraction from conversations.

Identifies explicit decisions where the user confirms an approach
or accepts an assistant's proposal. Heuristic, not ML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.lib.semantic_facts import ConversationSemanticFacts, build_conversation_semantic_facts

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation

_ACCEPTANCE_PATTERNS = (
    "let's go with", "sounds good", "yes, do that", "proceed",
    "that's the approach", "implement that", "go ahead", "approved",
    "let's do it", "ship it", "looks good", "lgtm", "yes please",
    "perfect, do it", "that works", "agreed",
)

_PROPOSAL_PATTERNS = (
    "i recommend", "i suggest", "the approach would be",
    "here's my plan", "i'll ", "my recommendation is",
    "i propose", "the best approach", "we should",
)


@dataclass(frozen=True)
class Decision:
    """An explicit decision point in a conversation."""

    index: int
    summary: str
    confidence: float
    context: str


def extract_decisions(
    conversation: Conversation,
    *,
    facts: ConversationSemanticFacts | None = None,
) -> list[Decision]:
    """Extract decisions from user messages that follow assistant proposals.

    Looks for user acceptance patterns ("let's go with", "proceed", etc.)
    that follow assistant messages containing proposals.
    """
    semantic_facts = facts or build_conversation_semantic_facts(conversation)
    messages = list(semantic_facts.message_facts)
    if len(messages) < 2:
        return []

    decisions: list[Decision] = []
    prev_assistant_text = ""
    prev_had_proposal = False

    for i, msg in enumerate(messages):
        if msg.is_assistant and msg.text:
            text_lower = msg.text.lower()
            prev_had_proposal = any(p in text_lower for p in _PROPOSAL_PATTERNS)
            prev_assistant_text = msg.text[:200]
        elif msg.is_user and msg.text and prev_had_proposal:
            text_lower = msg.text.lower()
            matched_pattern = None
            for pattern in _ACCEPTANCE_PATTERNS:
                if pattern in text_lower:
                    matched_pattern = pattern
                    break
            if matched_pattern:
                # Extract a summary from the user's message
                summary = msg.text[:150].strip()
                decisions.append(Decision(
                    index=i,
                    summary=summary,
                    confidence=0.7,
                    context=prev_assistant_text[:100],
                ))
            prev_had_proposal = False

    return decisions
