"""Heuristic decision extraction from conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.decision_models import Decision
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


def extract_decisions(
    conversation: Conversation,
    *,
    facts: ConversationSemanticFacts | None = None,
) -> list[Decision]:
    semantic_facts = facts or build_conversation_semantic_facts(conversation)
    messages = list(semantic_facts.message_facts)
    if len(messages) < 2:
        return []

    decisions: list[Decision] = []
    prev_assistant_text = ""
    prev_had_proposal = False
    for index, message in enumerate(messages):
        if message.is_assistant and message.text:
            text_lower = message.text.lower()
            prev_had_proposal = any(pattern in text_lower for pattern in _PROPOSAL_PATTERNS)
            prev_assistant_text = message.text[:200]
            continue
        if not (message.is_user and message.text and prev_had_proposal):
            continue
        matched_pattern = next(
            (pattern for pattern in _ACCEPTANCE_PATTERNS if pattern in message.text.lower()),
            None,
        )
        if matched_pattern:
            decisions.append(
                Decision(
                    index=index,
                    summary=message.text[:150].strip(),
                    confidence=0.7,
                    context=prev_assistant_text[:100],
                )
            )
        prev_had_proposal = False

    return decisions


__all__ = ["extract_decisions"]
