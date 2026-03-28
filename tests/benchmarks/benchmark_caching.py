"""Benchmark search result caching improvements."""

import contextlib
import sqlite3
import tempfile
import time
from pathlib import Path

from polylogue.storage.db import open_connection
from polylogue.storage.index import ensure_index
from polylogue.storage.search import search_messages
from polylogue.storage.search_cache import invalidate_search_cache
from polylogue.storage.store import ConversationRecord, MessageRecord, upsert_conversation, upsert_message


def create_test_database() -> Path:
    """Create a test database with synthetic data for caching benchmarks."""
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = Path(tmpfile.name)

    # Initialize schema
    with open_connection(db_path) as conn:
        pass

    # Create conversations and messages
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    for provider_idx in range(10):
        conv_id = f"conv_{provider_idx}"
        conversation = ConversationRecord(
            conversation_id=conv_id,
            provider_name="chatgpt",
            provider_conversation_id=f"provider_{conv_id}",
            title=f"Test Conversation {provider_idx}",
            created_at=str(1700000000 + provider_idx * 86400),
            updated_at=str(1700000000 + provider_idx * 86400 + 3600),
            provider_meta={},
            content_hash=f"hash_{conv_id}",
        )
        upsert_conversation(conn, conversation)

        for msg_idx in range(100):
            message = MessageRecord(
                message_id=f"{conv_id}_msg_{msg_idx}",
                conversation_id=conv_id,
                provider_message_id=f"provider_{conv_id}_msg_{msg_idx}",
                role="user" if msg_idx % 2 == 0 else "assistant",
                text=f"This is message {msg_idx} with searchable content including keywords like python, database, and optimization.",
                timestamp=str(1700000000 + provider_idx * 86400 + msg_idx * 60),
                provider_meta=None,
                content_hash=f"hash_{conv_id}_msg_{msg_idx}",
            )
            upsert_message(conn, message)

    # Build FTS5 index
    ensure_index(conn)
    conn.execute(
        """
        INSERT INTO messages_fts (message_id, conversation_id, provider_name, content)
        SELECT messages.message_id, messages.conversation_id, conversations.provider_name, messages.text
        FROM messages
        JOIN conversations ON conversations.conversation_id = messages.conversation_id
        WHERE messages.text IS NOT NULL
        """
    )
    conn.commit()
    conn.close()

    return db_path


def benchmark_cache_hit_rate():
    """Benchmark cache effectiveness for repeated queries."""
    db_path = create_test_database()
    archive_root = Path(tempfile.mkdtemp())

    # Clear any existing cache
    invalidate_search_cache()

    results = {}

    # First query (cold cache)
    start = time.perf_counter()
    search_messages("python", archive_root=archive_root, limit=20)
    elapsed_cold = time.perf_counter() - start

    results["first_query_cold_ms"] = elapsed_cold * 1000

    # Same query repeated (hot cache)
    start = time.perf_counter()
    for _ in range(100):
        search_messages("python", archive_root=archive_root, limit=20)
    elapsed_hot = time.perf_counter() - start

    results["repeated_100_hot_ms"] = elapsed_hot * 1000
    results["avg_cached_query_ms"] = (elapsed_hot / 100) * 1000
    results["speedup_factor"] = (elapsed_cold / (elapsed_hot / 100))

    # Different query (different cache key)
    start = time.perf_counter()
    search_messages("database", archive_root=archive_root, limit=20)
    elapsed_different = time.perf_counter() - start

    results["different_query_cold_ms"] = elapsed_different * 1000

    # Cache invalidation test
    invalidate_search_cache()

    start = time.perf_counter()
    search_messages("python", archive_root=archive_root, limit=20)
    elapsed_after_invalidation = time.perf_counter() - start

    results["after_invalidation_ms"] = elapsed_after_invalidation * 1000

    db_path.unlink()

    return results


def benchmark_cache_memory_overhead():
    """Benchmark memory overhead of caching (approximate)."""
    db_path = create_test_database()
    archive_root = Path(tempfile.mkdtemp())

    # Clear cache
    invalidate_search_cache()

    results = {}

    # Run unique queries to fill cache (maxsize=128)
    queries = [f"query_{i}" for i in range(150)]  # More than cache size

    start = time.perf_counter()
    for query in queries:
        try:
            search_messages(query, archive_root=archive_root, limit=20)
        except Exception:
            # Expected for queries that don't match
            pass
    elapsed = time.perf_counter() - start

    results["fill_cache_150_queries_ms"] = elapsed * 1000
    results["avg_per_query_ms"] = (elapsed / len(queries)) * 1000

    # Test LRU eviction (first query should be evicted)
    start = time.perf_counter()
    with contextlib.suppress(Exception):
        search_messages("query_0", archive_root=archive_root, limit=20)
    elapsed = time.perf_counter() - start

    results["lru_evicted_query_ms"] = elapsed * 1000

    db_path.unlink()

    return results


def run_all():
    """Run all caching benchmarks."""
    print("=" * 80)
    print("CACHING BENCHMARKS")
    print("=" * 80)

    print("\n--- Cache Hit Rate ---")
    hit_results = benchmark_cache_hit_rate()
    for name, value in hit_results.items():
        if "speedup" in name:
            print(f"{name}: {value:.2f}x")
        else:
            print(f"{name}: {value:.2f}ms")

    print("\n--- Cache Memory Overhead ---")
    memory_results = benchmark_cache_memory_overhead()
    for name, value in memory_results.items():
        print(f"{name}: {value:.2f}ms")

    return {**hit_results, **memory_results}


if __name__ == "__main__":
    run_all()
