"""Synthetic corpora for raw-authority replay throughput benchmarks (polylogue-amg1).

Builds a population of independent, never-classified raws directly into
source.db -- the exact shape ``census_historical_revision_evidence`` /
``backfill_historical_revision_evidence`` operate on for a CLI
``rebuild-index`` over historical evidence (polylogue-9p8x's original
scenario). Each raw is its own logical source (no cohort/append
relationships), which isolates per-raw census/commit overhead from cohort
classification complexity -- the same isolation polylogue-amg1's own
profiling used to attribute wall time between parse and
``sqlite3.Connection.__exit__`` (commit/fsync).

Two shapes are provided, matching amg1's own recorded measurements:
``SMALL_PAYLOAD_SHAPE`` (200 raws, ~50KB average) and
``LARGE_PAYLOAD_SHAPE`` (80 raws, ~1.7MB average).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

SMALL_PAYLOAD_SHAPE: dict[str, int] = {"raw_count": 200, "avg_payload_bytes": 50_000}
LARGE_PAYLOAD_SHAPE: dict[str, int] = {"raw_count": 80, "avg_payload_bytes": 1_700_000}

# polylogue-odm1: one oversized ("whale") raw alongside a population of
# small independent raws -- the shape of the v42 walk's whale-bearing rebuild
# pages (a multi-hundred-MB Codex rollout sharing a page with many ordinary
# sessions). ``whale_payload_bytes`` is deliberately far smaller than the
# live 442MB receipts: the ratio between spill+reload cost and the
# whale-aware-resident cost is what polylogue-odm1 measures, not the
# absolute byte count, and a multi-hundred-MB fixture would make the test
# suite itself slow. Benchmarks that need the real threshold math to trigger
# on a payload this size shrink ``_ParsedSessionSpill``'s cache-size class
# constants instead of growing the fixture (see
# ``tests/benchmarks/test_whale_census_spill_bench.py``).
WHALE_BEARING_SHAPE: dict[str, int] = {
    "small_raw_count": 40,
    "small_avg_payload_bytes": 20_000,
    "whale_payload_bytes": 40_000_000,
}

# polylogue-nh44: one growing-file cohort shaped after the bead's own live
# evidence (a Codex rollout file re-captured on every scan, each capture a
# byte-superset of the last). ~1MB final size, 50 superseded snapshots plus
# the winner -- the "45GB/46% of a restore is stale snapshots" shape.
REVISION_CHAIN_SHAPE: dict[str, int] = {"superseded_count": 50, "final_payload_bytes": 1_000_000}

# The finished-build comparison uses a synthetic parent whose children each
# retain its complete prefix. The parent deliberately appears last so every
# route has to preserve the same lineage outcome even when input order is
# inconvenient. This is a fixture for controlled route comparisons, not a
# scale claim about an operator corpus.
LARGE_PARENT_SHARED_PREFIX_SHAPE: dict[str, int] = {
    "parent_message_count": 24,
    "child_count": 8,
    "shared_message_bytes": 1_024,
}

FinishedBuildRoute = Literal["retained", "deferred", "shard"]


@dataclass(frozen=True, slots=True)
class FinishedBuildMeasurement:
    """One completed synthetic route measurement.

    The route's phase times are intentionally separate. A caller must not
    treat a partially imported archive as comparable output: canonical
    digests and counts are recorded only after index restoration, derived/FTS
    finalization, and the final checkpoint have all completed.
    """

    route: FinishedBuildRoute
    construction_seconds: float
    import_seconds: float
    index_restoration_seconds: float
    derived_fts_finalization_seconds: float
    checkpoint_seconds: float
    canonical_output_digests: dict[str, str]
    offered_count: int
    ingested_count: int
    refused_count: int
    deferred_count: int
    output_count: int

    def __post_init__(self) -> None:
        if any(
            seconds < 0
            for seconds in (
                self.construction_seconds,
                self.import_seconds,
                self.index_restoration_seconds,
                self.derived_fts_finalization_seconds,
                self.checkpoint_seconds,
            )
        ):
            raise ValueError("finished-build phase times must be non-negative")
        if any(
            count < 0
            for count in (
                self.offered_count,
                self.ingested_count,
                self.refused_count,
                self.deferred_count,
                self.output_count,
            )
        ):
            raise ValueError("finished-build counts must be non-negative")
        if not self.canonical_output_digests:
            raise ValueError("finished-build measurements require canonical output digests")


def _measurement_text_message(index: int, text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=f"measurement-text-{index}",
        role=Role.USER if index % 2 == 0 else Role.ASSISTANT,
        text=text,
        position=index,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def build_large_parent_shared_prefix_sessions(
    *,
    parent_message_count: int = LARGE_PARENT_SHARED_PREFIX_SHAPE["parent_message_count"],
    child_count: int = LARGE_PARENT_SHARED_PREFIX_SHAPE["child_count"],
    shared_message_bytes: int = LARGE_PARENT_SHARED_PREFIX_SHAPE["shared_message_bytes"],
) -> list[ParsedSession]:
    """Build a deterministic parent-plus-children session fixture.

    Every child repeats the parent's full, sizeable message prefix and adds a
    unique tail. Children are returned before their parent to retain the
    historically expensive deferred-tail shape. The fixture is synthetic and
    has no operator content.
    """
    if parent_message_count < 1 or child_count < 1 or shared_message_bytes < 1:
        raise ValueError("large-parent/shared-prefix fixture dimensions must be positive")

    parent_native_id = "measurement-zparent"
    prefix_texts = [
        f"measurement-prefix-{index:04d}-" + ("x" * shared_message_bytes) for index in range(parent_message_count)
    ]

    def session_messages(texts: list[str]) -> list[ParsedMessage]:
        return [_measurement_text_message(index, text) for index, text in enumerate(texts)]

    children = [
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=f"measurement-achild-{index:04d}",
            title=f"measurement child {index}",
            parent_session_provider_id=parent_native_id,
            messages=session_messages([*prefix_texts, f"measurement-child-tail-{index:04d}"]),
        )
        for index in range(child_count)
    ]
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=parent_native_id,
        title="measurement parent",
        messages=session_messages(prefix_texts),
    )
    return [*children, parent]


# Fixed per-record JSON envelope overhead (quotes, keys, braces) that isn't
# part of the padded text field -- subtracted from the target size so the
# generated payload lands close to the requested average.
_ENVELOPE_OVERHEAD_BYTES = 220


def _codex_raw_payload(index: int, *, target_bytes: int) -> bytes:
    """One independent, single-session Codex JSONL raw sized near target_bytes."""
    session_meta = (
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": f"amg1-session-{index:06d}", "timestamp": "2026-06-01T00:00:00Z"},
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    text_len = max(1, target_bytes - len(session_meta) - _ENVELOPE_OVERHEAD_BYTES)
    text = f"amg1-payload-{index:06d}-" + ("x" * text_len)
    response_item = (
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "one",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    return (session_meta + response_item).encode()


def build_independent_raw_corpus(
    archive_root: Path,
    *,
    raw_count: int,
    avg_payload_bytes: int,
    authoritative_source: bool = False,
) -> list[str]:
    """Write ``raw_count`` independent, uncensused, single-session raws.

    Returns the written raw ids in insertion order. The archive root is
    initialized fresh; call once per corpus. ``authoritative_source`` binds
    each raw to its own byte-proven full-revision baseline without deriving
    index rows, for rebuild-route fixtures that must pass source admission.
    """
    initialize_active_archive_root(archive_root)
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for index in range(raw_count):
            payload = _codex_raw_payload(index, target_bytes=avg_payload_bytes)
            native_id = f"amg1-session-{index:06d}"
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=f"synthetic-amg1/session-{index:06d}.jsonl",
                acquired_at_ms=index + 1,
                native_id=native_id if authoritative_source else None,
            )
            if authoritative_source:
                archive.bind_raw_revision(
                    raw_id,
                    RawRevisionEnvelope(
                        logical_source_key=f"codex-session:{native_id}",
                        kind=RawRevisionKind.FULL,
                        source_revision=f"synthetic-amg1:{index:06d}",
                        acquisition_generation=0,
                        baseline_raw_id=raw_id,
                        authority=RawRevisionAuthority.BYTE_PROVEN,
                    ),
                )
            raw_ids.append(raw_id)
    return raw_ids


def build_revision_chain_corpus(
    archive_root: Path,
    *,
    superseded_count: int,
    final_payload_bytes: int,
) -> list[str]:
    """Write one growing-file cohort: ``superseded_count`` older captures plus
    a final winner, all at the same ``source_path`` and each a strict byte
    prefix of the next -- the shape a re-scanned, ever-appended Codex rollout
    file produces on disk (polylogue-nh44). Returns raw ids oldest-first;
    the last id is the winner (the only one that should ever be parsed by an
    optimized census).
    """
    initialize_active_archive_root(archive_root)
    revision_count = superseded_count + 1
    session_meta = (
        json.dumps(
            {"type": "session_meta", "payload": {"id": "nh44-chain-session", "timestamp": "2026-06-01T00:00:00Z"}},
            separators=(",", ":"),
        )
        + "\n"
    )
    per_line_budget = max(1, (final_payload_bytes - len(session_meta)) // revision_count)
    raw_ids: list[str] = []
    payload = session_meta.encode()
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for index in range(revision_count):
            if index > 0:
                text = f"nh44-append-{index:04d}-" + ("x" * per_line_budget)
                response_item = (
                    json.dumps(
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "id": f"msg-{index:04d}",
                                "role": "user" if index % 2 else "assistant",
                                "content": [{"type": "input_text", "text": text}],
                            },
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                payload = payload + response_item.encode()
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path="nh44-chain/session.jsonl",
                acquired_at_ms=index + 1,
            )
            raw_ids.append(raw_id)
    return raw_ids


def build_whale_bearing_corpus(
    archive_root: Path,
    *,
    small_raw_count: int,
    small_avg_payload_bytes: int,
    whale_payload_bytes: int,
) -> tuple[list[str], str]:
    """Write a population of small independent raws plus one oversized
    ("whale") raw, all uncensused single-session Codex records sharing no
    cohort relationship (isolates whale-residency behavior from cohort
    classification, same rationale as ``build_independent_raw_corpus``).

    Returns ``(small_raw_ids, whale_raw_id)`` in insertion order. The archive
    root is initialized fresh; call once per corpus.
    """
    initialize_active_archive_root(archive_root)
    small_raw_ids: list[str] = []
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for index in range(small_raw_count):
            payload = _codex_raw_payload(index, target_bytes=small_avg_payload_bytes)
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=f"synthetic-odm1/session-{index:06d}.jsonl",
                acquired_at_ms=index + 1,
            )
            small_raw_ids.append(raw_id)
        whale_payload = _codex_raw_payload(small_raw_count, target_bytes=whale_payload_bytes)
        whale_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=whale_payload,
            source_path="synthetic-odm1/whale-session.jsonl",
            acquired_at_ms=small_raw_count + 1,
        )
    return small_raw_ids, whale_raw_id


__all__ = [
    "LARGE_PAYLOAD_SHAPE",
    "LARGE_PARENT_SHARED_PREFIX_SHAPE",
    "REVISION_CHAIN_SHAPE",
    "SMALL_PAYLOAD_SHAPE",
    "WHALE_BEARING_SHAPE",
    "FinishedBuildMeasurement",
    "build_independent_raw_corpus",
    "build_large_parent_shared_prefix_sessions",
    "build_revision_chain_corpus",
    "build_whale_bearing_corpus",
]
