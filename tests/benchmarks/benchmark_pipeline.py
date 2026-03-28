"""Benchmark pipeline operations."""

import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from polylogue.pipeline.ingest import prepare_ingest
from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage
from polylogue.storage.db import open_connection
from polylogue.storage.repository import ConversationRepository


def create_synthetic_conversation(conv_id: str, message_count: int = 50) -> ParsedConversation:
    """Create a synthetic conversation for benchmarking.

    Args:
        conv_id: Conversation ID
        message_count: Number of messages to include

    Returns:
        ParsedConversation with synthetic data
    """
    messages = []
    for i in range(message_count):
        messages.append(
            ParsedMessage(
                provider_message_id=f"{conv_id}_msg_{i}",
                role="user" if i % 2 == 0 else "assistant",
                text=f"This is message {i} with realistic content that includes code examples, "
                f"explanations, and various formatting. It simulates a typical AI conversation "
                f"with detailed responses and follow-up questions.",
                timestamp=str(1700000000 + i * 60),
                provider_meta=None,
            )
        )

    return ParsedConversation(
        provider_conversation_id=conv_id,
        provider_name="chatgpt",
        title=f"Test Conversation {conv_id}",
        created_at=str(1700000000),
        updated_at=str(1700000000 + message_count * 60),
        messages=messages,
        attachments=[],
    )


def benchmark_prepare_ingest():
    """Benchmark prepare_ingest function."""
    # Create temp database
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = Path(tmpfile.name)

    # Initialize schema
    with open_connection(db_path) as conn:
        pass

    # Create temp archive
    archive_root = Path(tempfile.mkdtemp())

    repository = ConversationRepository()
    conversation = create_synthetic_conversation("test_conv", message_count=50)

    results = {}

    # Single conversation (cold - first insert)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    start = time.perf_counter()
    prepare_ingest(conversation, "test_source", archive_root=archive_root, conn=conn, repository=repository)
    elapsed = time.perf_counter() - start
    conn.close()
    results["prepare_ingest_cold"] = {
        "time_ms": elapsed * 1000,
    }

    # Same conversation (warm - should skip)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    start = time.perf_counter()
    prepare_ingest(conversation, "test_source", archive_root=archive_root, conn=conn, repository=repository)
    elapsed = time.perf_counter() - start
    conn.close()
    results["prepare_ingest_warm_skip"] = {
        "time_ms": elapsed * 1000,
    }

    db_path.unlink()

    return results


def benchmark_parallel_ingestion():
    """Benchmark parallel ingestion with ThreadPoolExecutor."""
    # Create temp database
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = Path(tmpfile.name)

    # Initialize schema
    with open_connection(db_path) as conn:
        pass

    # Create temp archive
    archive_root = Path(tempfile.mkdtemp())

    repository = ConversationRepository()

    # Create multiple conversations
    conversations = [create_synthetic_conversation(f"conv_{i}", message_count=30) for i in range(20)]

    results = {}

    # Sequential processing
    start = time.perf_counter()
    for conv in conversations:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        prepare_ingest(conv, "test_source", archive_root=archive_root, conn=conn, repository=repository)
        conn.close()
    elapsed_sequential = time.perf_counter() - start

    results["sequential_20_convs"] = {
        "time_ms": elapsed_sequential * 1000,
        "convs_per_sec": len(conversations) / elapsed_sequential,
    }

    # Reset database
    db_path.unlink()
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = Path(tmpfile.name)
    with open_connection(db_path) as conn:
        pass

    # Parallel processing (max_workers=4, like production)
    def process_one(conv):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        result = prepare_ingest(conv, "test_source", archive_root=archive_root, conn=conn, repository=repository)
        conn.close()
        return result

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_one, conv) for conv in conversations]
        for fut in futures:
            fut.result()
    elapsed_parallel = time.perf_counter() - start

    results["parallel_20_convs_4workers"] = {
        "time_ms": elapsed_parallel * 1000,
        "convs_per_sec": len(conversations) / elapsed_parallel,
        "speedup": elapsed_sequential / elapsed_parallel,
    }

    db_path.unlink()

    return results


def benchmark_bounded_submission():
    """Benchmark bounded submission pattern."""
    import concurrent.futures

    # Create temp database
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = Path(tmpfile.name)

    with open_connection(db_path):
        pass

    archive_root = Path(tempfile.mkdtemp())
    repository = ConversationRepository()

    conversations = [create_synthetic_conversation(f"conv_{i}", message_count=20) for i in range(50)]

    def process_one(conv):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        result = prepare_ingest(conv, "test_source", archive_root=archive_root, conn=conn, repository=repository)
        conn.close()
        return result

    results = {}

    # Unbounded submission (all at once)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_one, conv): conv for conv in conversations}
        for fut in concurrent.futures.as_completed(futures):
            fut.result()
    elapsed_unbounded = time.perf_counter() - start

    results["unbounded_50_convs"] = {
        "time_ms": elapsed_unbounded * 1000,
    }

    # Reset database
    db_path.unlink()
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = Path(tmpfile.name)
    with open_connection(db_path):
        pass

    # Bounded submission (max 16 in-flight, like production)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        iter(conversations)

        # Initial batch
        for conv in conversations[:16]:
            future = executor.submit(process_one, conv)
            futures[future] = conv

        # Process remaining with bounded submission
        remaining = conversations[16:]
        for conv in remaining:
            # Wait for at least one to complete
            done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                fut.result()
                del futures[fut]

            # Submit new one
            future = executor.submit(process_one, conv)
            futures[future] = conv

        # Drain remaining
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    elapsed_bounded = time.perf_counter() - start

    results["bounded_50_convs_16max"] = {
        "time_ms": elapsed_bounded * 1000,
        "overhead_pct": (elapsed_bounded - elapsed_unbounded) / elapsed_unbounded * 100,
    }

    db_path.unlink()

    return results


def run_all():
    """Run all pipeline benchmarks."""
    print("=" * 80)
    print("PIPELINE BENCHMARKS")
    print("=" * 80)

    print("\n--- Prepare Ingest ---")
    ingest_results = benchmark_prepare_ingest()
    for name, data in ingest_results.items():
        print(f"{name}: {data['time_ms']:.2f}ms")

    print("\n--- Parallel Ingestion ---")
    parallel_results = benchmark_parallel_ingestion()
    for name, data in parallel_results.items():
        if "speedup" in data:
            print(
                f"{name}: {data['time_ms']:.2f}ms ({data['convs_per_sec']:.2f} convs/sec, "
                f"{data['speedup']:.2f}x speedup)"
            )
        else:
            print(f"{name}: {data['time_ms']:.2f}ms ({data['convs_per_sec']:.2f} convs/sec)")

    print("\n--- Bounded Submission ---")
    bounded_results = benchmark_bounded_submission()
    for name, data in bounded_results.items():
        if "overhead_pct" in data:
            print(f"{name}: {data['time_ms']:.2f}ms (overhead: {data['overhead_pct']:.2f}%)")
        else:
            print(f"{name}: {data['time_ms']:.2f}ms")

    return {**ingest_results, **parallel_results, **bounded_results}


if __name__ == "__main__":
    run_all()
