"""Benchmark search operations."""

import sqlite3
import tempfile
import time
from pathlib import Path

from polylogue.storage.db import open_connection
from polylogue.storage.index import ensure_index, update_index_for_conversations
from polylogue.storage.search import escape_fts5_query, search_messages
from polylogue.storage.store import ConversationRecord, MessageRecord, upsert_conversation, upsert_message


def create_test_database(message_count: int = 1000) -> Path:
    """Create a test database with synthetic data.

    Args:
        message_count: Number of messages to create

    Returns:
        Path to the temporary database
    """
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = Path(tmpfile.name)

    # Use open_connection to initialize schema automatically
    with open_connection(db_path) as conn:
        pass  # Schema is initialized automatically

    # Create conversations and messages
    conversations_per_provider = 10
    messages_per_conversation = message_count // conversations_per_provider

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    for provider_idx in range(conversations_per_provider):
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

        for msg_idx in range(messages_per_conversation):
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

    conn.commit()
    conn.close()

    return db_path


def benchmark_fts5_query_escaping():
    """Benchmark FTS5 query escaping."""
    queries = [
        "simple query",
        "query with AND operator",
        "query with special chars: *^(){}[]|&!+-",
        '"quoted query"',
        "query OR another",
        "***",  # Asterisk-only
        "AND",  # Bare operator
    ]

    results = {}

    for query in queries:
        start = time.perf_counter()
        for _ in range(10000):
            escape_fts5_query(query)
        elapsed = time.perf_counter() - start

        # Safe name for dict key
        safe_name = query[:30].replace(" ", "_").replace('"', "")
        results[f"escape_{safe_name}"] = {
            "time_ms": elapsed * 1000,
            "ops_per_sec": 10000 / elapsed,
        }

    return results


def benchmark_fts5_index_build():
    """Benchmark FTS5 index building."""
    results = {}

    for message_count in [100, 1000, 5000]:
        db_path = create_test_database(message_count)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Benchmark index creation
        start = time.perf_counter()
        ensure_index(conn)
        elapsed_create = time.perf_counter() - start

        # Benchmark full index build
        start = time.perf_counter()
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
        elapsed_build = time.perf_counter() - start

        results[f"build_index_{message_count}"] = {
            "create_ms": elapsed_create * 1000,
            "build_ms": elapsed_build * 1000,
            "total_ms": (elapsed_create + elapsed_build) * 1000,
            "messages_per_sec": message_count / (elapsed_create + elapsed_build),
        }

        conn.close()
        db_path.unlink()

    return results


def benchmark_fts5_search():
    """Benchmark FTS5 search queries."""
    db_path = create_test_database(1000)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Build index
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

    # Create archive root for search
    archive_root = Path(tempfile.mkdtemp())

    results = {}

    # Common query
    start = time.perf_counter()
    for _ in range(100):
        search_messages("python", archive_root=archive_root, limit=20)
    elapsed = time.perf_counter() - start
    results["search_common_100"] = {
        "time_ms": elapsed * 1000,
        "ops_per_sec": 100 / elapsed,
    }

    # Multi-word query
    start = time.perf_counter()
    for _ in range(100):
        search_messages("python database", archive_root=archive_root, limit=20)
    elapsed = time.perf_counter() - start
    results["search_multi_word_100"] = {
        "time_ms": elapsed * 1000,
        "ops_per_sec": 100 / elapsed,
    }

    # Rare query (fewer results)
    start = time.perf_counter()
    for _ in range(100):
        search_messages("nonexistent keyword xyz", archive_root=archive_root, limit=20)
    elapsed = time.perf_counter() - start
    results["search_rare_100"] = {
        "time_ms": elapsed * 1000,
        "ops_per_sec": 100 / elapsed,
    }

    db_path.unlink()

    return results


def benchmark_incremental_update():
    """Benchmark incremental index updates."""
    db_path = create_test_database(1000)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Build initial index
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

    results = {}

    # Update single conversation
    start = time.perf_counter()
    update_index_for_conversations(["conv_0"], conn)
    elapsed = time.perf_counter() - start
    results["update_single_conv"] = {
        "time_ms": elapsed * 1000,
    }

    # Update 5 conversations
    start = time.perf_counter()
    update_index_for_conversations([f"conv_{i}" for i in range(5)], conn)
    elapsed = time.perf_counter() - start
    results["update_5_convs"] = {
        "time_ms": elapsed * 1000,
    }

    conn.close()
    db_path.unlink()

    return results


def run_all():
    """Run all search benchmarks."""
    print("=" * 80)
    print("SEARCH BENCHMARKS")
    print("=" * 80)

    print("\n--- FTS5 Query Escaping ---")
    escape_results = benchmark_fts5_query_escaping()
    for name, data in escape_results.items():
        print(f"{name}: {data['time_ms']:.2f}ms ({data['ops_per_sec']:.0f} ops/sec)")

    print("\n--- FTS5 Index Build ---")
    build_results = benchmark_fts5_index_build()
    for name, data in build_results.items():
        print(
            f"{name}: create={data['create_ms']:.2f}ms, build={data['build_ms']:.2f}ms, "
            f"total={data['total_ms']:.2f}ms ({data['messages_per_sec']:.0f} msg/sec)"
        )

    print("\n--- FTS5 Search ---")
    search_results = benchmark_fts5_search()
    for name, data in search_results.items():
        print(f"{name}: {data['time_ms']:.2f}ms ({data['ops_per_sec']:.0f} ops/sec)")

    print("\n--- Incremental Updates ---")
    update_results = benchmark_incremental_update()
    for name, data in update_results.items():
        print(f"{name}: {data['time_ms']:.2f}ms")

    return {**escape_results, **build_results, **search_results, **update_results}


if __name__ == "__main__":
    run_all()
