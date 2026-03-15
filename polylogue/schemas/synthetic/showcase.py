"""Showcase conversation themes for human-readable synthetic output.

Each theme provides a coherent narrative arc with user and assistant turns
that produce visually appealing demo/showcase corpora.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationTheme:
    """Narrative theme for visually coherent synthetic conversations."""

    title: str
    instructions: str
    user_turns: tuple[str, ...]
    assistant_turns: tuple[str, ...]


_SHOWCASE_THEMES: tuple[ConversationTheme, ...] = (
    ConversationTheme(
        title="Debugging flaky async pipeline tests",
        instructions="You are a coding assistant focused on debugging async Python services.",
        user_turns=(
            "Our async pipeline test is flaky in CI. Locally it usually passes, but CI fails around 1 in 6 runs.",
            "The failure is around a race between acquisition and parse workers. We see duplicate writes occasionally.",
            "I added timestamps and correlation IDs. The duplicate writes happen when retries overlap with stale tasks.",
            "Would you enforce idempotency at storage or add stricter queue coordination first?",
            "Great. I implemented idempotent upserts and the flake rate dropped to zero in 200 CI runs.",
        ),
        assistant_turns=(
            "Start by capturing a deterministic trace: per-task IDs, queue depth, retry count, and commit boundaries.",
            "That pattern suggests retry re-entry without a completed marker. Guard writes with a stable content key and upsert semantics.",
            "Do both, but prioritize storage idempotency first. Queue coordination reduces load; idempotency guarantees correctness.",
            "Add a regression test that injects delayed worker scheduling and verifies single-write invariants under retries.",
            "Excellent result. Keep the stress test in CI and expose duplicate-write counters in your health metrics.",
        ),
    ),
    ConversationTheme(
        title="Designing schema validation stages",
        instructions="You are a systems assistant specializing in data contracts and pipeline reliability.",
        user_turns=(
            "We split our ingestion into acquire, validate, and parse stages. I want strong guarantees without huge overhead.",
            "Right now we persist raw payloads and parse status, but validation metadata is incomplete.",
            "Should strict validation be default, with an explicit opt-out for advisory mode?",
            "How do we avoid re-validating unchanged payloads every run?",
            "Done. We now persist validation status and only parse rows marked passed or skipped.",
        ),
        assistant_turns=(
            "That stage split is correct. Validation should produce an explicit status artifact consumed by parse.",
            "Persist validated_at, status, error, drift_count, and validation_mode so state survives reruns and compaction.",
            "Yes. Make strict the default and reserve advisory/off for deliberate local workflows or troubleshooting.",
            "Use raw content hash IDs and validate only pending records. Keep a reset path for forced revalidation.",
            "Perfect. Add tests for pending-validation selection and parse gating to protect the contract long-term.",
        ),
    ),
    ConversationTheme(
        title="Improving search index performance",
        instructions="You are a performance-focused assistant for storage and indexing systems.",
        user_turns=(
            "Search index rebuilds are slow after large imports. We need a strategy that scales with conversation growth.",
            "Current profile shows heavy time in repeated per-conversation metadata queries.",
            "I can batch those lookups. Any guidance on cache boundaries?",
            "We also need better observability for stage-level throughput and bottlenecks.",
            "Implemented batch lookup + stage metrics. End-to-end index time improved by 38 percent.",
        ),
        assistant_turns=(
            "First remove N+1 patterns: batch metadata loads and reuse immutable data across worker tasks.",
            "Great target. Build a prepare cache keyed by conversation IDs and warm it per processing batch.",
            "Cache per-run, scoped to candidate IDs only. Avoid global caches that can leak stale rows between runs.",
            "Emit per-stage counters, durations, and throughput. Include queue lag and retry/error distributions.",
            "Strong improvement. Add a benchmark regression gate so future refactors cannot silently erode performance.",
        ),
    ),
)


__all__ = [
    "ConversationTheme",
    "_SHOWCASE_THEMES",
]
