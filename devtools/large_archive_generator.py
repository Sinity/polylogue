"""Generate large synthetic archives for performance validation.

Builds on polylogue's synthetic corpus engine to produce archives
at various scale levels for benchmarking and stress testing.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ScaleLevel(Enum):
    """Scale levels for synthetic archive generation."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    STRETCH = "stretch"


@dataclass(frozen=True)
class ArchiveSpec:
    """Specification for a synthetic archive at a given scale level."""

    level: ScaleLevel
    provider_mix: dict[str, float]
    message_count: int
    conversations: int
    avg_messages_per_conv: int
    content_blocks_ratio: float
    seed: int = 42

    @property
    def messages_per_conversation_range(self) -> range:
        """Derive a message-count range from the average."""
        low = max(2, self.avg_messages_per_conv // 2)
        high = self.avg_messages_per_conv * 2
        return range(low, high + 1)


@dataclass
class ArchiveMetrics:
    """Metrics collected from archive generation."""

    wall_time_s: float = 0.0
    db_size_bytes: int = 0
    message_count: int = 0
    conversation_count: int = 0
    provider_breakdown: dict[str, int] = field(default_factory=dict)


DEFAULT_PROVIDER_MIX: dict[str, float] = {
    "chatgpt": 0.35,
    "claude-ai": 0.30,
    "claude-code": 0.20,
    "gemini": 0.10,
    "codex": 0.05,
}


DEFAULT_SPECS: dict[ScaleLevel, ArchiveSpec] = {
    ScaleLevel.SMALL: ArchiveSpec(
        level=ScaleLevel.SMALL,
        provider_mix=DEFAULT_PROVIDER_MIX,
        message_count=1_000,
        conversations=100,
        avg_messages_per_conv=10,
        content_blocks_ratio=0.3,
    ),
    ScaleLevel.MEDIUM: ArchiveSpec(
        level=ScaleLevel.MEDIUM,
        provider_mix=DEFAULT_PROVIDER_MIX,
        message_count=10_000,
        conversations=500,
        avg_messages_per_conv=20,
        content_blocks_ratio=0.3,
    ),
    ScaleLevel.LARGE: ArchiveSpec(
        level=ScaleLevel.LARGE,
        provider_mix=DEFAULT_PROVIDER_MIX,
        message_count=100_000,
        conversations=2_000,
        avg_messages_per_conv=50,
        content_blocks_ratio=0.3,
    ),
    ScaleLevel.STRETCH: ArchiveSpec(
        level=ScaleLevel.STRETCH,
        provider_mix=DEFAULT_PROVIDER_MIX,
        message_count=1_000_000,
        conversations=10_000,
        avg_messages_per_conv=100,
        content_blocks_ratio=0.3,
    ),
}


def _distribute_conversations(
    total: int,
    provider_mix: dict[str, float],
) -> dict[str, int]:
    """Distribute conversation count across providers by weight.

    Ensures total is preserved by assigning remainders to the largest-share provider.
    """
    allocated: dict[str, int] = {}
    remaining = total
    providers = sorted(provider_mix.items(), key=lambda kv: kv[1], reverse=True)

    for i, (provider, weight) in enumerate(providers):
        if i == len(providers) - 1:
            allocated[provider] = remaining
        else:
            count = max(1, round(total * weight))
            count = min(count, remaining - (len(providers) - i - 1))
            allocated[provider] = count
            remaining -= count

    return allocated


def _available_providers() -> set[str]:
    """Return the set of providers that have both schemas and wire formats."""
    from polylogue.schemas.synthetic import SyntheticCorpus

    return set(SyntheticCorpus.available_providers())


async def generate_archive(spec: ArchiveSpec, output_dir: Path) -> ArchiveMetrics:
    """Generate a synthetic archive at the specified scale level.

    Creates synthetic conversations using polylogue's SyntheticCorpus engine,
    writes them to temporary files, then ingests them through the full pipeline
    into a SQLite database.

    Args:
        spec: Archive specification (scale level, provider mix, counts, etc.)
        output_dir: Directory to write the archive into. Will be created if needed.

    Returns:
        ArchiveMetrics with timing, size, and count data.
    """
    from polylogue.paths import Source
    from polylogue.pipeline.prepare import prepare_records
    from polylogue.schemas.synthetic import SyntheticCorpus
    from polylogue.sources import iter_source_conversations
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.store import RawConversationRecord

    t0 = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "benchmark.db"
    corpus_dir = output_dir / "corpus"
    archive_root = output_dir / "archive"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    # Filter provider mix to only available providers
    available = _available_providers()
    filtered_mix = {p: w for p, w in spec.provider_mix.items() if p in available}
    if not filtered_mix:
        raise ValueError(
            f"No providers from spec are available. "
            f"Requested: {list(spec.provider_mix)}, available: {sorted(available)}"
        )
    # Renormalize weights
    total_weight = sum(filtered_mix.values())
    filtered_mix = {p: w / total_weight for p, w in filtered_mix.items()}

    distribution = _distribute_conversations(spec.conversations, filtered_mix)
    provider_breakdown: dict[str, int] = {}

    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)

    ext_map = {
        "chatgpt": ".json",
        "claude-ai": ".json",
        "gemini": ".json",
        "claude-code": ".jsonl",
        "codex": ".jsonl",
    }

    conversation_count = 0
    message_count = 0

    try:
        async with backend.bulk_connection():
            for provider, conv_count in distribution.items():
                if conv_count <= 0:
                    continue

                corpus = SyntheticCorpus.for_provider(provider)
                ext = ext_map.get(provider, ".json")
                provider_dir = corpus_dir / provider
                provider_dir.mkdir(parents=True, exist_ok=True)

                raw_items = corpus.generate(
                    count=conv_count,
                    messages_per_conversation=spec.messages_per_conversation_range,
                    seed=spec.seed,
                )

                provider_conv_count = 0
                for idx, raw_bytes in enumerate(raw_items):
                    file_path = provider_dir / f"synth-{idx:05d}{ext}"
                    file_path.write_bytes(raw_bytes)

                    # Store raw record
                    raw_id = hashlib.sha256(raw_bytes).hexdigest()
                    raw_record = RawConversationRecord(
                        raw_id=raw_id,
                        provider_name=provider,
                        source_name=provider,
                        source_path=str(file_path),
                        raw_content=raw_bytes,
                        acquired_at=datetime.now(timezone.utc).isoformat(),
                    )
                    await backend.save_raw_conversation(raw_record)

                    # Parse and ingest
                    source = Source(name=provider, path=file_path)
                    for convo in iter_source_conversations(source):
                        await prepare_records(
                            convo,
                            source_name=provider,
                            archive_root=archive_root,
                            backend=backend,
                            repository=repository,
                            raw_id=raw_id,
                        )
                        provider_conv_count += 1
                        message_count += len(convo.messages)

                    # Periodic flush every 100 files
                    if idx > 0 and idx % 100 == 0:
                        await backend.bulk_flush()

                provider_breakdown[provider] = provider_conv_count
                conversation_count += provider_conv_count
    finally:
        await backend.close()

    wall_time = time.monotonic() - t0
    db_size = db_path.stat().st_size if db_path.exists() else 0

    return ArchiveMetrics(
        wall_time_s=round(wall_time, 2),
        db_size_bytes=db_size,
        message_count=message_count,
        conversation_count=conversation_count,
        provider_breakdown=provider_breakdown,
    )


def get_default_spec(level: ScaleLevel) -> ArchiveSpec:
    """Return the default ArchiveSpec for a given scale level."""
    return DEFAULT_SPECS[level]


__all__ = [
    "ArchiveMetrics",
    "ArchiveSpec",
    "DEFAULT_SPECS",
    "ScaleLevel",
    "generate_archive",
    "get_default_spec",
]
