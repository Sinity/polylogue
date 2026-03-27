"""Description helpers for typed query specs."""

from __future__ import annotations


def describe_query_spec(spec) -> list[str]:
    parts: list[str] = []
    if spec.query_terms:
        parts.append(f"search: {' '.join(spec.query_terms)}")
    if spec.contains_terms:
        parts.append(f"contains: {', '.join(spec.contains_terms)}")
    if spec.exclude_text_terms:
        parts.append(f"exclude text: {', '.join(spec.exclude_text_terms)}")
    if spec.retrieval_lane != "auto":
        parts.append(f"retrieval: {spec.retrieval_lane}")
    if spec.path_terms:
        parts.append(f"path: {', '.join(spec.path_terms)}")
    if spec.action_terms:
        parts.append(f"action: {', '.join(spec.action_terms)}")
    if spec.excluded_action_terms:
        parts.append(f"exclude action: {', '.join(spec.excluded_action_terms)}")
    if spec.action_sequence:
        parts.append(f"action sequence: {' -> '.join(spec.action_sequence)}")
    if spec.action_text_terms:
        parts.append(f"action text: {', '.join(spec.action_text_terms)}")
    if spec.tool_terms:
        parts.append(f"tool: {', '.join(spec.tool_terms)}")
    if spec.excluded_tool_terms:
        parts.append(f"exclude tool: {', '.join(spec.excluded_tool_terms)}")
    if spec.providers:
        parts.append(f"provider: {', '.join(p.value for p in spec.providers)}")
    if spec.excluded_providers:
        parts.append(f"exclude provider: {', '.join(p.value for p in spec.excluded_providers)}")
    if spec.tags:
        parts.append(f"tag: {', '.join(spec.tags)}")
    if spec.excluded_tags:
        parts.append(f"exclude tag: {', '.join(spec.excluded_tags)}")
    if spec.title:
        parts.append(f"title: {spec.title}")
    if spec.has_types:
        parts.append(f"has: {', '.join(spec.has_types)}")
    if spec.filter_has_tool_use:
        parts.append("has: tool_use (sql)")
    if spec.filter_has_thinking:
        parts.append("has: thinking (sql)")
    if spec.min_messages is not None:
        parts.append(f"min_messages: {spec.min_messages}")
    if spec.max_messages is not None:
        parts.append(f"max_messages: {spec.max_messages}")
    if spec.min_words is not None:
        parts.append(f"min_words: {spec.min_words}")
    if spec.similar_text:
        parts.append(f"similar: {spec.similar_text}")
    if spec.since:
        parts.append(f"since: {spec.since}")
    if spec.until:
        parts.append(f"until: {spec.until}")
    if spec.conversation_id:
        parts.append(f"id: {spec.conversation_id}")
    return parts


def query_spec_has_filters(spec) -> bool:
    return any(
        (
            spec.query_terms,
            spec.contains_terms,
            spec.exclude_text_terms,
            spec.path_terms,
            spec.action_terms,
            spec.excluded_action_terms,
            spec.action_sequence,
            spec.action_text_terms,
            spec.tool_terms,
            spec.excluded_tool_terms,
            spec.providers,
            spec.excluded_providers,
            spec.tags,
            spec.excluded_tags,
            spec.has_types,
            spec.title is not None,
            spec.conversation_id is not None,
            spec.since is not None,
            spec.until is not None,
            spec.latest,
            spec.filter_has_tool_use,
            spec.filter_has_thinking,
            spec.min_messages is not None,
            spec.max_messages is not None,
            spec.min_words is not None,
            spec.similar_text is not None,
        )
    )


__all__ = ["describe_query_spec", "query_spec_has_filters"]
