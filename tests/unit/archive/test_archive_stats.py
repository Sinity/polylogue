from __future__ import annotations

from polylogue.archive.stats import ArchiveStats


def test_archive_stats_properties_handle_empty_and_partial_embedding_states() -> None:
    empty = ArchiveStats(total_conversations=0, total_messages=0)
    partial = ArchiveStats(
        total_conversations=4,
        total_messages=10,
        providers={"chatgpt": 2, "claude-ai": 2},
        embedded_conversations=2,
        embedded_messages=5,
        pending_embedding_conversations=1,
    )
    stale = ArchiveStats(
        total_conversations=4,
        total_messages=10,
        embedded_conversations=5,
        embedded_messages=5,
        stale_embedding_messages=2,
        messages_missing_embedding_provenance=1,
    )
    fresh = ArchiveStats(
        total_conversations=4,
        total_messages=10,
        embedded_conversations=4,
        embedded_messages=5,
    )

    assert empty.provider_count == 0
    assert empty.avg_messages_per_conversation == 0.0
    assert empty.embedding_coverage == 0.0
    assert empty.embedding_readiness_status == "none"
    assert empty.retrieval_ready is False

    assert partial.provider_count == 2
    assert partial.avg_messages_per_conversation == 2.5
    assert partial.embedding_coverage == 50.0
    assert partial.embedding_readiness_status == "partial"
    assert partial.retrieval_ready is True

    assert stale.embedding_readiness_status == "stale"
    assert stale.retrieval_ready is True
    assert fresh.embedding_readiness_status == "fresh"


def test_archive_stats_to_dict_rounds_and_serializes_derived_fields() -> None:
    stats = ArchiveStats(
        total_conversations=3,
        total_messages=10,
        total_attachments=2,
        providers={"chatgpt": 3},
        embedded_conversations=2,
        embedded_messages=4,
        embedding_oldest_at="2026-04-01T00:00:00Z",
        embedding_newest_at="2026-04-02T00:00:00Z",
        embedding_models={"text-embedding-3-small": 4},
        embedding_dimensions={1536: 4},
        db_size_bytes=2048,
    )

    payload = stats.to_dict()

    assert payload["provider_count"] == 1
    assert payload["embedding_coverage_percent"] == 66.7
    assert payload["embedding_readiness_status"] == "fresh"
    assert payload["retrieval_ready"] is True
    assert payload["avg_messages_per_conversation"] == 3.3
    assert payload["db_size_bytes"] == 2048
