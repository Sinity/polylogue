"""Scoped vs global facet computation (#1269 / slice D of #873)."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from polylogue.archive.conversation.models import ConversationSummary
from polylogue.archive.query.facets import (
    FacetBuckets,
    FacetSet,
    compute_facets,
    compute_idf,
)
from polylogue.surfaces.payloads import FacetBucketsPayload, FacetsResponse
from polylogue.types import ConversationId, Provider


def _summary(
    conversation_id: str,
    *,
    provider: Provider,
    message_count: int,
    tags: tuple[str, ...] = (),
) -> ConversationSummary:
    return ConversationSummary(
        id=ConversationId(conversation_id),
        provider=provider,
        title=conversation_id,
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        tags_m2m=tags,
        message_count=message_count,
    )


class TestComputeFacets:
    def test_rolls_provider_and_tag_counts(self) -> None:
        buckets = compute_facets(
            [
                _summary("a", provider=Provider.CHATGPT, message_count=4, tags=("ml",)),
                _summary(
                    "b",
                    provider=Provider.CLAUDE_AI,
                    message_count=2,
                    tags=("ml", "draft"),
                ),
                _summary("c", provider=Provider.CHATGPT, message_count=6, tags=("draft",)),
            ]
        )
        assert buckets.providers == {"chatgpt": 2, "claude-ai": 1}
        assert buckets.tags == {"ml": 2, "draft": 2}
        assert buckets.total_conversations == 3
        assert buckets.total_messages == 12

    def test_empty_input_returns_empty_buckets(self) -> None:
        buckets = compute_facets([])
        assert buckets == FacetBuckets()

    def test_duplicate_tags_on_one_conversation_count_once(self) -> None:
        # Tag list dedupes within a single conversation so a doubly-
        # applied tag does not double-count facet bucket counts.
        buckets = compute_facets(
            [
                _summary(
                    "a",
                    provider=Provider.CHATGPT,
                    message_count=1,
                    tags=("dup", "dup"),
                ),
            ]
        )
        assert buckets.tags == {"dup": 1}


class TestComputeIdf:
    def test_rare_value_has_higher_idf_than_common_value(self) -> None:
        # 10 conversations: "rare" appears in 1, "common" appears in 9
        buckets = FacetBuckets(
            providers={"rare": 1, "common": 9},
            total_conversations=10,
        )
        idf = compute_idf(buckets)
        assert idf["providers"]["rare"] > idf["providers"]["common"]
        assert math.isclose(idf["providers"]["rare"], math.log(10 / 1))
        assert math.isclose(idf["providers"]["common"], math.log(10 / 9))

    def test_empty_universe_returns_empty_map(self) -> None:
        assert compute_idf(FacetBuckets()) == {}

    def test_idf_is_zero_for_value_in_every_conversation(self) -> None:
        buckets = FacetBuckets(
            providers={"x": 5},
            total_conversations=5,
        )
        idf = compute_idf(buckets)
        assert math.isclose(idf["providers"]["x"], 0.0)


class TestFacetsResponseEnvelope:
    def test_carries_scoped_and_global_buckets_with_alias(self) -> None:
        response = FacetsResponse.model_validate(
            {
                "scoped_to_query": True,
                "providers": {"chatgpt": 1},
                "scoped": FacetBucketsPayload(providers={"chatgpt": 1}, total_conversations=1),
                "global": FacetBucketsPayload(
                    providers={"chatgpt": 2, "claude-ai": 1},
                    total_conversations=3,
                ),
                "idf": {"providers": {"chatgpt": math.log(3 / 2)}},
            }
        )
        dumped = response.model_dump(mode="json", by_alias=True)
        assert dumped["scoped_to_query"] is True
        assert dumped["scoped"] == {
            "providers": {"chatgpt": 1},
            "tags": {},
            "total_conversations": 1,
            "total_messages": 0,
        }
        # JSON field uses the alias ``global`` rather than ``global_``.
        assert "global" in dumped
        assert dumped["global"]["providers"] == {"chatgpt": 2, "claude-ai": 1}
        assert dumped["idf"]["providers"]["chatgpt"] == pytest.approx(math.log(3 / 2))

    def test_defaults_are_empty(self) -> None:
        response = FacetsResponse()
        assert response.scoped_to_query is False
        assert response.scoped.total_conversations == 0
        assert response.global_.total_conversations == 0
        assert response.idf == {}


class TestFacetSet:
    def test_holds_both_views(self) -> None:
        fs = FacetSet(
            scoped=FacetBuckets(providers={"chatgpt": 1}, total_conversations=1),
            global_=FacetBuckets(providers={"chatgpt": 2}, total_conversations=2),
            scoped_to_query=True,
        )
        assert fs.scoped.total_conversations == 1
        assert fs.global_.total_conversations == 2
        assert fs.scoped_to_query is True
        assert fs.idf == {}
