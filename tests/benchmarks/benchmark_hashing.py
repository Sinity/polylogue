"""Benchmark content hashing operations."""

import time
import unicodedata

from polylogue.lib.hashing import hash_text
from polylogue.pipeline.ids import conversation_content_hash
from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage


def benchmark_hash_text():
    """Benchmark hash_text function with various text sizes."""
    results = {}

    # Small text (typical message)
    small_text = "Hello, this is a typical message with some content."
    start = time.perf_counter()
    for _ in range(10000):
        hash_text(small_text)
    elapsed = time.perf_counter() - start
    results["small_text_10k"] = {
        "time_ms": elapsed * 1000,
        "ops_per_sec": 10000 / elapsed,
    }

    # Medium text (long message)
    medium_text = small_text * 100  # ~5KB
    start = time.perf_counter()
    for _ in range(1000):
        hash_text(medium_text)
    elapsed = time.perf_counter() - start
    results["medium_text_1k"] = {
        "time_ms": elapsed * 1000,
        "ops_per_sec": 1000 / elapsed,
    }

    # Large text (conversation)
    large_text = small_text * 1000  # ~50KB
    start = time.perf_counter()
    for _ in range(100):
        hash_text(large_text)
    elapsed = time.perf_counter() - start
    results["large_text_100"] = {
        "time_ms": elapsed * 1000,
        "ops_per_sec": 100 / elapsed,
    }

    # Unicode normalization overhead test
    unicode_text = "café naïve résumé"
    start_normalized = time.perf_counter()
    for _ in range(10000):
        unicodedata.normalize("NFC", unicode_text)
    elapsed_normalized = time.perf_counter() - start_normalized

    start_hash = time.perf_counter()
    for _ in range(10000):
        hash_text(unicode_text)
    elapsed_hash = time.perf_counter() - start_hash

    results["unicode_normalization_overhead_pct"] = (
        (elapsed_hash - elapsed_normalized) / elapsed_normalized * 100
    )

    return results


def benchmark_conversation_hash():
    """Benchmark conversation_content_hash with realistic data."""
    # Create a realistic conversation
    messages = []
    for i in range(50):  # 50 messages
        messages.append(
            ParsedMessage(
                provider_message_id=f"msg_{i}",
                role="user" if i % 2 == 0 else "assistant",
                text=f"This is message {i} with some realistic content that spans multiple words.",
                timestamp=str(1700000000 + i * 3600),
                provider_meta=None,
            )
        )

    conversation = ParsedConversation(
        provider_conversation_id="test_conv",
        provider_name="test",
        title="Test Conversation",
        created_at=str(1700000000),
        updated_at=str(1700000000 + 50 * 3600),
        messages=messages,
        attachments=[],
    )

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        conversation_content_hash(conversation)
    elapsed = time.perf_counter() - start

    return {
        "conversation_hash_100": {
            "time_ms": elapsed * 1000,
            "ops_per_sec": 100 / elapsed,
            "message_count": len(messages),
        }
    }


def run_all():
    """Run all hashing benchmarks."""
    print("=" * 80)
    print("HASHING BENCHMARKS")
    print("=" * 80)

    print("\n--- Text Hashing ---")
    text_results = benchmark_hash_text()
    for name, data in text_results.items():
        if "overhead" in name:
            print(f"{name}: {data:.2f}%")
        else:
            print(f"{name}: {data['time_ms']:.2f}ms ({data['ops_per_sec']:.0f} ops/sec)")

    print("\n--- Conversation Hashing ---")
    conv_results = benchmark_conversation_hash()
    for name, data in conv_results.items():
        print(
            f"{name}: {data['time_ms']:.2f}ms ({data['ops_per_sec']:.0f} ops/sec, "
            f"{data['message_count']} messages)"
        )

    return {**text_results, **conv_results}


if __name__ == "__main__":
    run_all()
