"""Semantic action/tool/path slice helpers for grouped query output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.action_events import ActionEvent
    from polylogue.lib.query_spec import ConversationQuerySpec
    from polylogue.lib.semantic_facts import ConversationSemanticFacts


@dataclass(frozen=True, slots=True)
class SemanticStatsSlice:
    path_terms: tuple[str, ...] = ()
    action_terms: tuple[str, ...] = ()
    excluded_action_terms: tuple[str, ...] = ()
    action_text_terms: tuple[str, ...] = ()
    tool_terms: tuple[str, ...] = ()
    excluded_tool_terms: tuple[str, ...] = ()

    @classmethod
    def from_selection(cls, selection: ConversationQuerySpec | None) -> SemanticStatsSlice:
        if selection is None:
            return cls()
        return cls(
            path_terms=selection.path_terms,
            action_terms=selection.action_terms,
            excluded_action_terms=selection.excluded_action_terms,
            action_text_terms=selection.action_text_terms,
            tool_terms=selection.tool_terms,
            excluded_tool_terms=selection.excluded_tool_terms,
        )

    def has_filters(self) -> bool:
        return any(
            (
                self.path_terms,
                self.action_terms,
                self.excluded_action_terms,
                self.action_text_terms,
                self.tool_terms,
                self.excluded_tool_terms,
            )
        )


def normalized_tool_name(action: ActionEvent) -> str:
    return action.normalized_tool_name


def path_matches_slice(action: ActionEvent, path_terms: tuple[str, ...]) -> bool:
    if not path_terms:
        return True
    affected_paths = tuple(path.lower().replace("\\", "/") for path in action.affected_paths)
    if not affected_paths:
        return False
    return any(
        any(term.lower().replace("\\", "/") in path for path in affected_paths)
        for term in path_terms
    )


def action_matches_slice(action: ActionEvent, semantic_slice: SemanticStatsSlice) -> bool:
    if not path_matches_slice(action, semantic_slice.path_terms):
        return False

    if "none" in semantic_slice.action_terms:
        return False
    required_action_terms = {term for term in semantic_slice.action_terms if term != "none"}
    if required_action_terms and action.kind.value not in required_action_terms:
        return False
    blocked_action_terms = {term for term in semantic_slice.excluded_action_terms if term != "none"}
    if action.kind.value in blocked_action_terms:
        return False
    if semantic_slice.action_text_terms:
        search_text = action.search_text.lower()
        if not all(term.lower() in search_text for term in semantic_slice.action_text_terms):
            return False

    tool_name = normalized_tool_name(action)
    if "none" in semantic_slice.tool_terms:
        return False
    required_tool_terms = {term for term in semantic_slice.tool_terms if term != "none"}
    if required_tool_terms and tool_name not in required_tool_terms:
        return False
    blocked_tool_terms = {term for term in semantic_slice.excluded_tool_terms if term != "none"}
    return tool_name not in blocked_tool_terms


def filtered_action_events(
    facts: ConversationSemanticFacts,
    semantic_slice: SemanticStatsSlice,
) -> tuple[ActionEvent, ...]:
    if not semantic_slice.has_filters():
        return facts.action_events
    return tuple(action for action in facts.action_events if action_matches_slice(action, semantic_slice))


__all__ = [
    "SemanticStatsSlice",
    "action_matches_slice",
    "filtered_action_events",
    "normalized_tool_name",
    "path_matches_slice",
]
