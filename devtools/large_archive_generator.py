"""Generate large synthetic archives for performance validation.

Builds on polylogue's synthetic corpus engine to produce archives
at various scale levels for benchmarking and stress testing.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.scenarios import (
    CorpusScenario,
    CorpusSourceKind,
    CorpusSpec,
    build_corpus_scenarios,
    flatten_corpus_specs,
)
from polylogue.schemas.operator_inference import list_inferred_corpus_scenarios

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

    def corpus_specs(
        self,
        *,
        available_providers: set[str],
        corpus_source: CorpusSourceKind | str = CorpusSourceKind.DEFAULT,
    ) -> tuple[CorpusSpec, ...]:
        """Compile the archive spec into per-provider synthetic corpus specs."""
        return flatten_corpus_specs(
            self.corpus_scenarios(
                available_providers=available_providers,
                corpus_source=corpus_source,
            )
        )

    def corpus_scenarios(
        self,
        *,
        available_providers: set[str],
        corpus_source: CorpusSourceKind | str = CorpusSourceKind.DEFAULT,
    ) -> tuple[CorpusScenario, ...]:
        """Compile the archive spec into named per-provider synthetic corpus scenarios."""
        source_kind = CorpusSourceKind(corpus_source)
        filtered_mix = {
            provider: weight for provider, weight in self.provider_mix.items() if provider in available_providers
        }
        if not filtered_mix:
            raise ValueError(
                f"No providers from spec are available. "
                f"Requested: {list(self.provider_mix)}, available: {sorted(available_providers)}"
            )
        if source_kind is CorpusSourceKind.DEFAULT:
            total_weight = sum(filtered_mix.values())
            normalized_mix = {provider: weight / total_weight for provider, weight in filtered_mix.items()}
            distribution = _distribute_conversations(self.conversations, normalized_mix)
            return build_corpus_scenarios(
                tuple(
                    CorpusSpec.for_provider(
                        provider,
                        count=conversation_count,
                        messages_min=self.messages_per_conversation_range.start,
                        messages_max=self.messages_per_conversation_range.stop - 1,
                        seed=self.seed,
                        origin="generated.large-archive",
                        tags=("synthetic", "benchmark", "scale", self.level.value),
                    )
                    for provider, conversation_count in distribution.items()
                    if conversation_count > 0
                ),
                origin="compiled.large-archive-scenario",
                tags=("synthetic", "benchmark", "scale", self.level.value, "scenario"),
            )

        base_scenarios = tuple(
            scenario for scenario in list_inferred_corpus_scenarios() if scenario.provider in filtered_mix
        )
        if not base_scenarios:
            raise ValueError(f"No inferred corpus scenarios available for providers: {sorted(filtered_mix)}")

        total_weight = sum(filtered_mix.values())
        normalized_mix = {provider: weight / total_weight for provider, weight in filtered_mix.items()}
        distribution = _distribute_conversations(self.conversations, normalized_mix)
        scaled_specs: list[CorpusSpec] = []
        for provider, conversation_count in distribution.items():
            provider_scenarios = tuple(scenario for scenario in base_scenarios if scenario.provider == provider)
            scaled_specs.extend(
                _scale_corpus_specs(
                    tuple(spec for scenario in provider_scenarios for spec in scenario.corpus_specs),
                    total=conversation_count,
                    level=self.level.value,
                    seed=self.seed,
                    messages_min=self.messages_per_conversation_range.start,
                    messages_max=self.messages_per_conversation_range.stop - 1,
                )
            )
        return build_corpus_scenarios(
            tuple(scaled_specs),
            origin="compiled.large-archive-scenario",
            tags=("synthetic", "benchmark", "scale", self.level.value, "scenario", "inferred"),
        )


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


def _scale_integer_weights(total: int, weights: tuple[int, ...]) -> tuple[int, ...]:
    if total <= 0 or not weights:
        return tuple(0 for _ in weights)
    positive_weights = tuple(max(1, weight) for weight in weights)
    total_weight = sum(positive_weights)
    raw = [total * weight / total_weight for weight in positive_weights]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (raw[index] - counts[index], positive_weights[index]),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


def _scale_corpus_specs(
    specs: tuple[CorpusSpec, ...],
    *,
    total: int,
    level: str,
    seed: int,
    messages_min: int,
    messages_max: int,
) -> tuple[CorpusSpec, ...]:
    if total <= 0 or not specs:
        return ()
    counts = _scale_integer_weights(total, tuple(spec.count for spec in specs))
    scaled: list[CorpusSpec] = []
    for spec, count in zip(specs, counts, strict=True):
        if count <= 0:
            continue
        scaled.append(
            replace(
                spec,
                count=count,
                seed=seed,
                messages_min=messages_min,
                messages_max=messages_max,
                origin="generated.large-archive",
                tags=tuple(dict.fromkeys((*spec.tags, "synthetic", "benchmark", "scale", level, "inferred"))),
            )
        )
    return tuple(scaled)


def _available_providers() -> set[str]:
    """Return the set of providers that have both schemas and wire formats."""
    from polylogue.schemas.synthetic import SyntheticCorpus

    return set(SyntheticCorpus.available_providers())


async def generate_archive(
    spec: ArchiveSpec,
    output_dir: Path,
    *,
    corpus_source: CorpusSourceKind | str = CorpusSourceKind.DEFAULT,
) -> ArchiveMetrics:
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
    provider_breakdown: dict[str, int] = {}

    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)

    conversation_count = 0
    message_count = 0

    try:
        async with backend.bulk_connection():
            for corpus_scenario in spec.corpus_scenarios(
                available_providers=_available_providers(),
                corpus_source=corpus_source,
            ):
                provider = corpus_scenario.provider
                provider_conv_count = 0
                for corpus_spec in corpus_scenario.corpus_specs:
                    provider_dir = corpus_dir / provider
                    written = SyntheticCorpus.write_spec_artifacts(
                        corpus_spec,
                        provider_dir,
                        prefix="synth",
                        index_width=5,
                    )

                    for file_path, artifact in zip(written.files, written.batch.artifacts, strict=True):
                        raw_bytes = artifact.raw_bytes
                        # Store raw record
                        raw_id = hashlib.sha256(raw_bytes).hexdigest()
                        raw_record = RawConversationRecord(
                            raw_id=raw_id,
                            provider_name=provider,
                            source_name=provider,
                            source_path=str(file_path),
                            blob_size=len(raw_bytes),
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
                        if provider_conv_count > 0 and provider_conv_count % 100 == 0:
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
