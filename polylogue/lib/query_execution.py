"""Canonical immutable execution plans for conversation queries."""

from __future__ import annotations

import builtins
import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from heapq import heappush, heappushpop
from typing import TYPE_CHECKING, Any, TypeVar

from polylogue.logging import get_logger
from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.storage.search_providers.hybrid import reciprocal_rank_fusion
from polylogue.types import Provider

if TYPE_CHECKING:
    from collections.abc import Callable

    from polylogue.lib.filter_types import SortField
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.protocols import VectorProvider
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)

_T = TypeVar("_T")


def _provider_values(values: tuple[Provider | str, ...]) -> tuple[str, ...]:
    return tuple(str(Provider.from_string(value)) for value in values)


def _conversation_has_branches(conversation: Conversation) -> bool:
    return any(message.branch_index > 0 for message in conversation.messages)


@dataclass(frozen=True)
class ConversationQueryPlan:
    """Canonical immutable execution state for conversation selection."""

    query_terms: tuple[str, ...] = ()
    contains_terms: tuple[str, ...] = ()
    negative_terms: tuple[str, ...] = ()
    retrieval_lane: str = "auto"
    path_terms: tuple[str, ...] = ()
    action_terms: tuple[str, ...] = ()
    excluded_action_terms: tuple[str, ...] = ()
    action_sequence: tuple[str, ...] = ()
    action_text_terms: tuple[str, ...] = ()
    tool_terms: tuple[str, ...] = ()
    excluded_tool_terms: tuple[str, ...] = ()
    providers: tuple[Provider | str, ...] = ()
    excluded_providers: tuple[Provider | str, ...] = ()
    tags: tuple[str, ...] = ()
    excluded_tags: tuple[str, ...] = ()
    has_types: tuple[str, ...] = ()
    title: str | None = None
    conversation_id: str | None = None
    parent_id: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    sort: SortField = "date"
    reverse: bool = False
    limit: int | None = None
    sample: int | None = None
    similar_text: str | None = None
    predicates: tuple[Callable[[Conversation], bool], ...] = ()
    continuation: bool | None = None
    sidechain: bool | None = None
    root: bool | None = None
    has_branches: bool | None = None
    filter_has_tool_use: bool = False
    filter_has_thinking: bool = False
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None
    vector_provider: VectorProvider | None = None

    @property
    def fts_terms(self) -> tuple[str, ...]:
        return self.query_terms + self.contains_terms

    @property
    def sql_pushed(self) -> bool:
        return not self.fts_terms and self.conversation_id is None

    @property
    def record_query(self) -> ConversationRecordQuery:
        provider_values = _provider_values(self.providers)
        provider = provider_values[0] if len(provider_values) == 1 else None
        providers = provider_values if len(provider_values) > 1 else ()
        return ConversationRecordQuery(
            provider=provider,
            providers=providers,
            parent_id=self.parent_id,
            since=self.since.isoformat() if self.since else None,
            until=self.until.isoformat() if self.until else None,
            title_contains=self.title,
            path_terms=self.path_terms,
            action_terms=self.action_terms,
            excluded_action_terms=self.excluded_action_terms,
            tool_terms=self.tool_terms,
            excluded_tool_terms=self.excluded_tool_terms,
            has_tool_use=self.filter_has_tool_use,
            has_thinking=self.filter_has_thinking,
            min_messages=self.min_messages,
            max_messages=self.max_messages,
            min_words=self.min_words,
        )

    def sql_pushdown_params(self) -> dict[str, object]:
        params: dict[str, object] = {}
        provider_values = _provider_values(self.providers)
        if len(provider_values) == 1:
            params["provider"] = provider_values[0]
        elif provider_values:
            params["providers"] = list(provider_values)
        if self.parent_id:
            params["parent_id"] = self.parent_id
        if self.since:
            params["since"] = self.since.isoformat()
        if self.until:
            params["until"] = self.until.isoformat()
        if self.title:
            params["title_contains"] = self.title
        if self.path_terms:
            params["path_terms"] = list(self.path_terms)
        if self.action_terms:
            params["action_terms"] = list(self.action_terms)
        if self.excluded_action_terms:
            params["excluded_action_terms"] = list(self.excluded_action_terms)
        if self.action_sequence:
            params["action_sequence"] = list(self.action_sequence)
        if self.action_text_terms:
            params["action_text_terms"] = list(self.action_text_terms)
        if self.tool_terms:
            params["tool_terms"] = list(self.tool_terms)
        if self.excluded_tool_terms:
            params["excluded_tool_terms"] = list(self.excluded_tool_terms)
        if self.filter_has_tool_use:
            params["has_tool_use"] = True
        if self.filter_has_thinking:
            params["has_thinking"] = True
        if self.min_messages is not None:
            params["min_messages"] = self.min_messages
        if self.max_messages is not None:
            params["max_messages"] = self.max_messages
        if self.min_words is not None:
            params["min_words"] = self.min_words
        return params

    def describe(self) -> list[str]:
        parts: list[str] = []
        if self.fts_terms:
            parts.append(f"contains: {', '.join(self.fts_terms)}")
        if self.negative_terms:
            parts.append(f"exclude text: {', '.join(self.negative_terms)}")
        if self.retrieval_lane != "auto":
            parts.append(f"retrieval: {self.retrieval_lane}")
        if self.path_terms:
            parts.append(f"path: {', '.join(self.path_terms)}")
        if self.action_terms:
            parts.append(f"action: {', '.join(self.action_terms)}")
        if self.excluded_action_terms:
            parts.append(f"exclude action: {', '.join(self.excluded_action_terms)}")
        if self.action_sequence:
            parts.append(f"action sequence: {' -> '.join(self.action_sequence)}")
        if self.action_text_terms:
            parts.append(f"action text: {', '.join(self.action_text_terms)}")
        if self.tool_terms:
            parts.append(f"tool: {', '.join(self.tool_terms)}")
        if self.excluded_tool_terms:
            parts.append(f"exclude tool: {', '.join(self.excluded_tool_terms)}")
        if self.providers:
            parts.append(f"provider: {', '.join(_provider_values(self.providers))}")
        if self.excluded_providers:
            parts.append(f"exclude provider: {', '.join(_provider_values(self.excluded_providers))}")
        if self.tags:
            parts.append(f"tag: {', '.join(self.tags)}")
        if self.excluded_tags:
            parts.append(f"exclude tag: {', '.join(self.excluded_tags)}")
        if self.title:
            parts.append(f"title: {self.title}")
        if self.has_types:
            parts.append(f"has: {', '.join(self.has_types)}")
        if self.filter_has_tool_use:
            parts.append("has_tool_use")
        if self.filter_has_thinking:
            parts.append("has_thinking")
        if self.min_messages is not None:
            parts.append(f"min_messages: {self.min_messages}")
        if self.max_messages is not None:
            parts.append(f"max_messages: {self.max_messages}")
        if self.min_words is not None:
            parts.append(f"min_words: {self.min_words}")
        if self.since:
            parts.append(f"since: {self.since.isoformat()}")
        if self.until:
            parts.append(f"until: {self.until.isoformat()}")
        if self.conversation_id:
            parts.append(f"id: {self.conversation_id}")
        if self.parent_id:
            parts.append(f"parent: {self.parent_id}")
        if self.continuation is True:
            parts.append("continuation")
        if self.continuation is False:
            parts.append("not continuation")
        if self.sidechain is True:
            parts.append("sidechain")
        if self.sidechain is False:
            parts.append("not sidechain")
        if self.root is True:
            parts.append("root")
        if self.root is False:
            parts.append("not root")
        if self.has_branches is True:
            parts.append("has branches")
        if self.has_branches is False:
            parts.append("no branches")
        if self.predicates:
            parts.append(f"custom predicates: {len(self.predicates)}")
        if self.similar_text:
            parts.append(f"similar: {self.similar_text[:30]}")
        return parts

    def has_filters(self) -> bool:
        return any(
            (
                self.fts_terms,
                self.negative_terms,
                self.path_terms,
                self.action_terms,
                self.excluded_action_terms,
                self.action_sequence,
                self.action_text_terms,
                self.tool_terms,
                self.excluded_tool_terms,
                self.providers,
                self.excluded_providers,
                self.tags,
                self.excluded_tags,
                self.has_types,
                self.title is not None,
                self.conversation_id is not None,
                self.parent_id is not None,
                self.since is not None,
                self.until is not None,
                self.similar_text is not None,
                self.continuation is not None,
                self.sidechain is not None,
                self.root is not None,
                self.has_branches is not None,
                self.filter_has_tool_use,
                self.filter_has_thinking,
                self.min_messages is not None,
                self.max_messages is not None,
                self.min_words is not None,
                self.predicates,
            )
        )

    def has_post_filters(self) -> bool:
        return bool(
            self.excluded_providers
            or self.tags
            or self.excluded_tags
            or self.has_types
            or self.predicates
            or self.negative_terms
            or self.path_terms
            or self.action_terms
            or self.excluded_action_terms
            or self.action_sequence
            or self.action_text_terms
            or self.tool_terms
            or self.excluded_tool_terms
            or self.continuation is not None
            or self.sidechain is not None
            or self.root is not None
            or self.has_branches is not None
        )

    def needs_content_loading(self) -> bool:
        if self.fts_terms and self.retrieval_lane in {"actions", "hybrid"}:
            return True
        if self.has_types and any(kind in ("thinking", "tools", "attachments") for kind in self.has_types):
            return True
        if self.negative_terms or self.predicates or self.similar_text:
            return True
        if self.path_terms:
            return True
        if self.action_terms or self.excluded_action_terms:
            return True
        if self.action_sequence:
            return True
        if self.action_text_terms:
            return True
        if self.tool_terms or self.excluded_tool_terms:
            return True
        if self.has_branches is not None:
            return True
        return self.sort in ("messages", "words", "longest", "tokens")

    def can_use_summaries(self) -> bool:
        return not self.needs_content_loading()

    def can_count_in_sql(self) -> bool:
        return not (
            self.fts_terms
            or self.conversation_id
            or self.similar_text
            or self.path_terms
            or self.action_terms
            or self.excluded_action_terms
            or self.action_sequence
            or self.action_text_terms
            or self.tool_terms
            or self.excluded_tool_terms
            or self.predicates
            or self.has_types
            or self.negative_terms
            or self.excluded_providers
            or self.tags
            or self.excluded_tags
            or self.continuation is not None
            or self.sidechain is not None
            or self.root is not None
            or self.has_branches is not None
        )

    def _matches_path_terms(self, conversation: Conversation) -> bool:
        if not self.path_terms:
            return True
        from polylogue.lib.semantic_facts import build_conversation_semantic_facts

        facts = build_conversation_semantic_facts(conversation)
        affected_paths = tuple(path.lower() for action in facts.action_events for path in action.affected_paths)
        if not affected_paths:
            return False
        return all(
            any(term.lower().replace("\\", "/") in path for path in affected_paths)
            for term in self.path_terms
        )

    def _matches_action_terms(self, conversation: Conversation) -> bool:
        if not self.action_terms and not self.excluded_action_terms:
            return True
        from polylogue.lib.semantic_facts import build_conversation_semantic_facts

        facts = build_conversation_semantic_facts(conversation)
        categories = {action.kind.value for action in facts.action_events}
        required_terms = {term for term in self.action_terms if term != "none"}
        if "none" in self.action_terms and categories:
            return False
        if required_terms and not required_terms.issubset(categories):
            return False
        if "none" in self.excluded_action_terms and not categories:
            return False
        return not (
            {term for term in self.excluded_action_terms if term != "none"} & categories
        )

    def _matches_tool_terms(self, conversation: Conversation) -> bool:
        if not self.tool_terms and not self.excluded_tool_terms:
            return True
        from polylogue.lib.semantic_facts import build_conversation_semantic_facts

        facts = build_conversation_semantic_facts(conversation)
        tool_names = {
            (action.tool_name or "unknown").strip().lower()
            for action in facts.action_events
        }
        required_terms = {term for term in self.tool_terms if term != "none"}
        if "none" in self.tool_terms and tool_names:
            return False
        if required_terms and not required_terms.issubset(tool_names):
            return False
        if "none" in self.excluded_tool_terms and not tool_names:
            return False
        return not (
            {term for term in self.excluded_tool_terms if term != "none"} & tool_names
        )

    def _matches_action_sequence(self, conversation: Conversation) -> bool:
        if not self.action_sequence:
            return True
        from polylogue.lib.semantic_facts import build_conversation_semantic_facts

        facts = build_conversation_semantic_facts(conversation)
        if not facts.action_events:
            return False

        index = 0
        target_count = len(self.action_sequence)
        for action in facts.action_events:
            if action.kind.value != self.action_sequence[index]:
                continue
            index += 1
            if index >= target_count:
                return True
        return False

    def _matches_action_text_terms(self, conversation: Conversation) -> bool:
        if not self.action_text_terms:
            return True
        from polylogue.lib.semantic_facts import build_conversation_semantic_facts

        facts = build_conversation_semantic_facts(conversation)
        searchable_events = [action.search_text.lower() for action in facts.action_events if action.search_text]
        if not searchable_events:
            return False
        return all(
            any(term.lower() in event_text for event_text in searchable_events)
            for term in self.action_text_terms
        )

    def effective_fetch_limit(self) -> int | None:
        if self.limit is None:
            return None
        if self.has_post_filters():
            return max(self.limit * 10, 500)
        if self.sample is not None:
            return max(self.sample * 3, 200)
        return max(self.limit * 2, 2)

    def with_limit(self, limit: int | None) -> ConversationQueryPlan:
        return replace(self, limit=limit)

    def _candidate_record_query(self) -> tuple[ConversationRecordQuery, bool]:
        record_query = self.record_query
        if self.path_terms or self.action_terms or self.excluded_action_terms:
            record_query = record_query.without_unstable_semantic_filters()
            return record_query, False
        return record_query, self.sql_pushed

    def fetch_record_query(self) -> ConversationRecordQuery:
        record_query, _ = self._candidate_record_query()
        return record_query.with_limit(self.effective_fetch_limit())

    def _should_batch_post_filter_fetch(self) -> bool:
        return bool(
            self.limit is not None
            and self.limit > 0
            and self.has_post_filters()
            and not self.fts_terms
            and self.conversation_id is None
            and self.sample is None
            and self.sort == "date"
            and not self.reverse
        )

    def _candidate_batch_limit(self) -> int:
        if self.limit is None:
            return 100
        return min(max(self.limit * 2, 100), 200)

    def _search_limit(self) -> int:
        fetch_limit = self.effective_fetch_limit()
        return max(fetch_limit, 100) if fetch_limit is not None else 10000

    async def _fetch_direct_id(
        self,
        repository: ConversationRepository,
        *,
        summaries: bool,
    ) -> builtins.list[Conversation | ConversationSummary]:
        if not self.conversation_id or self.fts_terms:
            return []
        resolved_id = await repository.resolve_id(self.conversation_id)
        if not resolved_id:
            return []
        if summaries:
            item = await repository.get_summary(str(resolved_id))
        else:
            item = await repository.get(str(resolved_id))
        return [item] if item is not None else []

    async def _fetch_search_results(
        self,
        repository: ConversationRepository,
        *,
        summaries: bool,
    ) -> tuple[bool, builtins.list[Conversation | ConversationSummary]]:
        if not self.fts_terms:
            return False, []
        if self.retrieval_lane == "actions":
            if summaries:
                return False, []
            return True, await self._search_action_results(repository, limit=self._search_limit())
        if self.retrieval_lane == "hybrid":
            if summaries:
                return False, []
            return True, await self._search_hybrid_results(repository, limit=self._search_limit())
        query = " ".join(self.fts_terms)
        provider_names = list(_provider_values(self.providers)) or None
        try:
            if summaries:
                results = await repository.search_summaries(query, limit=self._search_limit(), providers=provider_names)
            else:
                results = await repository.search(query, limit=self._search_limit(), providers=provider_names)
            return True, results
        except Exception as exc:
            logger.debug("FTS search failed, falling back to list: %s", exc)
            return False, []

    async def _fetch_candidates(
        self,
        repository: ConversationRepository,
        *,
        summaries: bool,
    ) -> builtins.list[Conversation | ConversationSummary]:
        direct = await self._fetch_direct_id(repository, summaries=summaries)
        if direct:
            return direct

        used_search, search_results = await self._fetch_search_results(repository, summaries=summaries)
        if used_search:
            return search_results

        request = self.fetch_record_query()
        if summaries:
            return await repository.list_summaries_by_query(request)
        return await repository.list_by_query(request)

    def _search_query_text(self) -> str:
        return " ".join(term.strip() for term in self.fts_terms if term.strip()).strip()

    def _search_query_terms(self) -> tuple[str, ...]:
        return tuple(term.lower() for term in self.fts_terms if term.strip())

    def _score_action_search_text(self, search_text: str, *, query_text: str, terms: tuple[str, ...]) -> float:
        haystack = search_text.lower()
        score = 0.0
        if query_text and query_text.lower() in haystack:
            score += 20.0 + len(query_text.split())
        for term in terms:
            if term in haystack:
                score += 4.0
        return score

    def _conversation_action_search_score(self, conversation: Conversation, *, query_text: str, terms: tuple[str, ...]) -> float:
        from polylogue.lib.semantic_facts import build_conversation_semantic_facts

        facts = build_conversation_semantic_facts(conversation)
        matches = [
            self._score_action_search_text(action.search_text, query_text=query_text, terms=terms)
            for action in facts.action_events
            if action.search_text
        ]
        positive = [score for score in matches if score > 0]
        if not positive:
            return 0.0
        return max(positive) + min(len(positive) - 1, 5)

    async def _search_action_results_fallback(
        self,
        repository: ConversationRepository,
        *,
        limit: int,
    ) -> builtins.list[Conversation]:
        request, sql_pushed = self._candidate_record_query()
        batch_limit = self._candidate_batch_limit()
        offset = 0
        query_text = self._search_query_text()
        terms = self._search_query_terms()
        ranked: list[tuple[float, int, Conversation]] = []
        counter = 0

        while True:
            batch = await repository.list_by_query(request.with_limit(batch_limit).with_offset(offset))
            if not batch:
                break
            for conversation in self._apply_common_filters(batch, sql_pushed=sql_pushed):
                score = self._conversation_action_search_score(conversation, query_text=query_text, terms=terms)
                if score <= 0:
                    continue
                entry = (score, counter, conversation)
                counter += 1
                if len(ranked) < limit:
                    heappush(ranked, entry)
                else:
                    heappushpop(ranked, entry)
            if len(batch) < batch_limit:
                break
            offset += batch_limit

        ranked.sort(key=lambda item: (item[0], item[2].updated_at or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        return [conversation for _score, _counter, conversation in ranked]

    async def _search_action_results(
        self,
        repository: ConversationRepository,
        *,
        limit: int,
    ) -> builtins.list[Conversation]:
        query = self._search_query_text()
        provider_names = list(_provider_values(self.providers)) or None
        try:
            return await repository.search_actions(query, limit=limit, providers=provider_names)
        except Exception as exc:
            logger.debug("Persisted action search failed, falling back to hydrated scan: %s", exc)
            return await self._search_action_results_fallback(repository, limit=limit)

    async def _search_hybrid_results(
        self,
        repository: ConversationRepository,
        *,
        limit: int,
    ) -> builtins.list[Conversation]:
        query = self._search_query_text()
        provider_names = list(_provider_values(self.providers)) or None
        text_results = await repository.search(query, limit=limit * 3, providers=provider_names)
        action_results = await self._search_action_results(repository, limit=limit * 3)
        vector_results: builtins.list[Conversation] = []
        if self.vector_provider is not None:
            try:
                vector_results = await repository.search_similar(
                    query,
                    limit=limit * 3,
                    vector_provider=self.vector_provider,
                )
            except Exception as exc:
                logger.debug("Vector contribution to hybrid retrieval unavailable: %s", exc)

        text_ranked = [(str(conversation.id), float(rank)) for rank, conversation in enumerate(text_results, start=1)]
        action_ranked = [(str(conversation.id), float(rank)) for rank, conversation in enumerate(action_results, start=1)]
        vector_ranked = [(str(conversation.id), float(rank)) for rank, conversation in enumerate(vector_results, start=1)]
        fused_ids = [
            conversation_id
            for conversation_id, _score in reciprocal_rank_fusion(text_ranked, action_ranked, vector_ranked)
        ][:limit]

        text_by_id = {str(conversation.id): conversation for conversation in text_results}
        action_by_id = {str(conversation.id): conversation for conversation in action_results}
        vector_by_id = {str(conversation.id): conversation for conversation in vector_results}
        ordered: list[Conversation] = []
        for conversation_id in fused_ids:
            conversation = (
                action_by_id.get(conversation_id)
                or text_by_id.get(conversation_id)
                or vector_by_id.get(conversation_id)
            )
            if conversation is not None:
                ordered.append(conversation)
        return ordered

    async def _fetch_batched_filtered_conversations(
        self,
        repository: ConversationRepository,
    ) -> builtins.list[Conversation]:
        request, sql_pushed = self._candidate_record_query()
        batch_limit = self._candidate_batch_limit()
        offset = 0
        matched: builtins.list[Conversation] = []
        seen_ids: set[str] = set()

        while True:
            batch = await repository.list_by_query(
                request.with_limit(batch_limit).with_offset(offset)
            )
            if not batch:
                break
            filtered_batch = self._apply_full_filters(batch, sql_pushed=sql_pushed)
            for conversation in filtered_batch:
                conversation_id = str(conversation.id)
                if conversation_id in seen_ids:
                    continue
                seen_ids.add(conversation_id)
                matched.append(conversation)
            if self.limit is not None and len(matched) >= self.limit:
                break
            if len(batch) < batch_limit:
                break
            offset += batch_limit

        return matched

    def _apply_common_filters(
        self,
        items: builtins.list[_T],
        *,
        sql_pushed: bool,
    ) -> builtins.list[_T]:
        results = list(items)

        if not sql_pushed:
            provider_set = set(_provider_values(self.providers))
            if provider_set:
                results = [item for item in results if str(item.provider) in provider_set]
            if self.since:
                results = [item for item in results if item.updated_at and item.updated_at >= self.since]
            if self.until:
                results = [item for item in results if item.updated_at and item.updated_at <= self.until]
            if self.title:
                lowered = self.title.lower()
                results = [
                    item
                    for item in results
                    if item.display_title and lowered in item.display_title.lower()
                ]
            if self.parent_id:
                results = [item for item in results if str(item.parent_id or "") == self.parent_id]

        if self.excluded_providers:
            excluded = set(_provider_values(self.excluded_providers))
            results = [item for item in results if str(item.provider) not in excluded]
        if self.tags:
            tag_set = set(self.tags)
            results = [item for item in results if tag_set.intersection(item.tags)]
        if self.excluded_tags:
            excluded_tags = set(self.excluded_tags)
            results = [item for item in results if not excluded_tags.intersection(item.tags)]
        if self.conversation_id:
            results = [item for item in results if str(item.id).startswith(self.conversation_id)]
        if "summary" in self.has_types:
            results = [item for item in results if item.summary]
        if self.continuation is True:
            results = [item for item in results if item.is_continuation]
        if self.continuation is False:
            results = [item for item in results if not item.is_continuation]
        if self.sidechain is True:
            results = [item for item in results if item.is_sidechain]
        if self.sidechain is False:
            results = [item for item in results if not item.is_sidechain]
        if self.root is True:
            results = [item for item in results if item.is_root]
        if self.root is False:
            results = [item for item in results if not item.is_root]

        return results

    def _apply_full_filters(
        self,
        conversations: builtins.list[Conversation],
        *,
        sql_pushed: bool,
    ) -> builtins.list[Conversation]:
        results = self._apply_common_filters(conversations, sql_pushed=sql_pushed)

        if self.has_types:
            for content_type in self.has_types:
                if content_type == "thinking":
                    results = [c for c in results if any(m.is_thinking for m in c.messages)]
                elif content_type == "tools":
                    results = [c for c in results if any(m.is_tool_use for m in c.messages)]
                elif content_type == "attachments":
                    results = [c for c in results if any(m.attachments for m in c.messages)]

        if self.negative_terms:
            negative_terms = [term.lower() for term in self.negative_terms]

            def _has_negative_term(conversation: Conversation) -> bool:
                for message in conversation.messages:
                    if not message.text:
                        continue
                    lowered = message.text.lower()
                    for term in negative_terms:
                        if term in lowered:
                            return True
                return False

            results = [conversation for conversation in results if not _has_negative_term(conversation)]

        if self.has_branches is True:
            results = [conversation for conversation in results if _conversation_has_branches(conversation)]
        if self.has_branches is False:
            results = [conversation for conversation in results if not _conversation_has_branches(conversation)]

        for predicate in self.predicates:
            results = [conversation for conversation in results if predicate(conversation)]

        if self.path_terms:
            results = [conversation for conversation in results if self._matches_path_terms(conversation)]
        if self.action_terms or self.excluded_action_terms:
            results = [conversation for conversation in results if self._matches_action_terms(conversation)]
        if self.action_sequence:
            results = [conversation for conversation in results if self._matches_action_sequence(conversation)]
        if self.action_text_terms:
            results = [conversation for conversation in results if self._matches_action_text_terms(conversation)]
        if self.tool_terms or self.excluded_tool_terms:
            results = [conversation for conversation in results if self._matches_tool_terms(conversation)]

        return results

    def _sort_generic(
        self,
        items: builtins.list[_T],
        key_fn: Callable[[_T], Any],
    ) -> builtins.list[_T]:
        if self.sort == "random":
            shuffled = list(items)
            random.shuffle(shuffled)
            return shuffled
        return sorted(items, key=key_fn, reverse=not self.reverse)

    def _sort_conversations(
        self,
        conversations: builtins.list[Conversation],
    ) -> builtins.list[Conversation]:
        dt_min = datetime.min.replace(tzinfo=timezone.utc)

        def _key(conversation: Conversation) -> Any:
            if self.sort == "date":
                return conversation.updated_at or dt_min
            if self.sort == "messages":
                return len(conversation.messages)
            if self.sort == "words":
                return sum(message.word_count for message in conversation.messages)
            if self.sort == "longest":
                return max((message.word_count for message in conversation.messages), default=0)
            if self.sort == "tokens":
                return sum(len(message.text or "") for message in conversation.messages) // 4
            return conversation.updated_at or dt_min

        return self._sort_generic(conversations, _key)

    def _sort_summaries(
        self,
        summaries: builtins.list[ConversationSummary],
    ) -> builtins.list[ConversationSummary]:
        dt_min = datetime.min.replace(tzinfo=timezone.utc)
        return self._sort_generic(summaries, lambda summary: summary.updated_at or dt_min)

    def _finalize(self, items: builtins.list[_T]) -> builtins.list[_T]:
        results = list(items)
        if self.sample is not None and self.sample < len(results):
            results = random.sample(results, self.sample)
        if self.limit is not None:
            results = results[: self.limit]
        return results

    async def list(self, repository: ConversationRepository) -> builtins.list[Conversation]:
        if self.similar_text:
            candidates = await repository.search_similar(
                self.similar_text,
                limit=self.limit or 10,
                vector_provider=self.vector_provider,
            )
            return self._finalize(self._apply_full_filters(candidates, sql_pushed=False))

        if self._should_batch_post_filter_fetch():
            batched = await self._fetch_batched_filtered_conversations(repository)
            return self._finalize(self._sort_conversations(batched))

        candidates = await self._fetch_candidates(repository, summaries=False)
        filtered = self._apply_full_filters(candidates, sql_pushed=self.sql_pushed)
        return self._finalize(self._sort_conversations(filtered))

    async def list_summaries(
        self,
        repository: ConversationRepository,
    ) -> builtins.list[ConversationSummary]:
        if not self.can_use_summaries():
            msg = (
                "Cannot use list_summaries() with content-dependent filters "
                "(regex, has:thinking, has:tools, etc.). Use list() instead."
            )
            raise ValueError(msg)

        candidates = await self._fetch_candidates(repository, summaries=True)
        filtered = self._apply_common_filters(candidates, sql_pushed=self.sql_pushed)
        return self._finalize(self._sort_summaries(filtered))

    async def first(self, repository: ConversationRepository) -> Conversation | None:
        results = await self.with_limit(1).list(repository)
        return results[0] if results else None

    async def count(self, repository: ConversationRepository) -> int:
        if self.can_count_in_sql():
            return await repository.count_by_query(self.record_query.for_count())

        unbounded = self.with_limit(None)
        if unbounded.can_use_summaries():
            return len(await unbounded.list_summaries(repository))
        return len(await unbounded.list(repository))

    async def delete(self, repository: ConversationRepository) -> int:
        if self.can_use_summaries():
            results: list[Conversation | ConversationSummary] = await self.list_summaries(repository)
        else:
            results = await self.list(repository)

        deleted = 0
        for conversation in results:
            if await repository.delete_conversation(str(conversation.id)):
                deleted += 1
        return deleted


__all__ = ["ConversationQueryPlan", "_conversation_has_branches"]
