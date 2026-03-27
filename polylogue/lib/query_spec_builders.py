"""Query-spec construction and compilation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from polylogue.lib.query_plan import ConversationQueryPlan
from polylogue.types import Provider

from .query_spec_normalization import (
    as_tuple,
    normalize_action_sequence,
    normalize_action_terms,
    normalize_tool_terms,
    parse_query_date,
    split_csv,
)


def build_query_spec_from_params(
    spec_cls,
    params: Mapping[str, object],
):
    return spec_cls(
        query_terms=as_tuple(params.get("query")),
        contains_terms=as_tuple(params.get("contains")),
        exclude_text_terms=as_tuple(params.get("exclude_text")),
        retrieval_lane=str(params.get("retrieval_lane") or "auto"),
        path_terms=as_tuple(params.get("path_terms") or params.get("path")),
        action_terms=normalize_action_terms("action", params.get("action")),
        excluded_action_terms=normalize_action_terms("exclude_action", params.get("exclude_action")),
        action_sequence=normalize_action_sequence("action_sequence", params.get("action_sequence")),
        action_text_terms=as_tuple(params.get("action_text")),
        tool_terms=normalize_tool_terms(params.get("tool")),
        excluded_tool_terms=normalize_tool_terms(params.get("exclude_tool")),
        providers=tuple(Provider.from_string(p) for p in split_csv(params.get("provider"))),
        excluded_providers=tuple(Provider.from_string(p) for p in split_csv(params.get("exclude_provider"))),
        tags=split_csv(params.get("tag")),
        excluded_tags=split_csv(params.get("exclude_tag")),
        has_types=as_tuple(params.get("has_type")),
        title=str(params["title"]) if params.get("title") else None,
        conversation_id=str(params["conv_id"]) if params.get("conv_id") else None,
        since=str(params["since"]) if params.get("since") else None,
        until=str(params["until"]) if params.get("until") else None,
        latest=bool(params.get("latest")),
        sort=str(params["sort"]) if params.get("sort") else None,
        reverse=bool(params.get("reverse")),
        limit=int(params["limit"]) if params.get("limit") else None,
        sample=int(params["sample"]) if params.get("sample") else None,
        filter_has_tool_use=bool(params.get("filter_has_tool_use")),
        filter_has_thinking=bool(params.get("filter_has_thinking")),
        min_messages=int(params["min_messages"]) if params.get("min_messages") else None,
        max_messages=int(params["max_messages"]) if params.get("max_messages") else None,
        min_words=int(params["min_words"]) if params.get("min_words") else None,
        similar_text=str(params["similar_text"]) if params.get("similar_text") else None,
    )


def query_spec_to_plan(
    spec,
    *,
    vector_provider=None,
) -> ConversationQueryPlan:
    plan = ConversationQueryPlan(
        query_terms=spec.query_terms,
        contains_terms=spec.contains_terms,
        negative_terms=spec.exclude_text_terms,
        retrieval_lane=spec.retrieval_lane,
        path_terms=spec.path_terms,
        action_terms=spec.action_terms,
        excluded_action_terms=spec.excluded_action_terms,
        action_sequence=spec.action_sequence,
        action_text_terms=spec.action_text_terms,
        tool_terms=spec.tool_terms,
        excluded_tool_terms=spec.excluded_tool_terms,
        providers=spec.providers,
        excluded_providers=spec.excluded_providers,
        tags=spec.tags,
        excluded_tags=spec.excluded_tags,
        has_types=spec.has_types,
        title=spec.title,
        conversation_id=spec.conversation_id,
        since=parse_query_date("since", spec.since),
        until=parse_query_date("until", spec.until),
        sort=spec.sort or "date",
        reverse=spec.reverse,
        limit=spec.limit,
        sample=spec.sample,
        filter_has_tool_use=spec.filter_has_tool_use,
        filter_has_thinking=spec.filter_has_thinking,
        min_messages=spec.min_messages,
        max_messages=spec.max_messages,
        min_words=spec.min_words,
        similar_text=spec.similar_text,
        vector_provider=vector_provider,
    )
    if spec.latest:
        plan = replace(plan, sort="date", limit=1)
    return plan


__all__ = ["build_query_spec_from_params", "query_spec_to_plan"]
