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
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

SMALL_PAYLOAD_SHAPE: dict[str, int] = {"raw_count": 200, "avg_payload_bytes": 50_000}
LARGE_PAYLOAD_SHAPE: dict[str, int] = {"raw_count": 80, "avg_payload_bytes": 1_700_000}

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
) -> list[str]:
    """Write ``raw_count`` independent, uncensused, single-session raws.

    Returns the written raw ids in insertion order. The archive root is
    initialized fresh; call once per corpus.
    """
    initialize_active_archive_root(archive_root)
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for index in range(raw_count):
            payload = _codex_raw_payload(index, target_bytes=avg_payload_bytes)
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=f"synthetic-amg1/session-{index:06d}.jsonl",
                acquired_at_ms=index + 1,
            )
            raw_ids.append(raw_id)
    return raw_ids


__all__ = [
    "LARGE_PAYLOAD_SHAPE",
    "SMALL_PAYLOAD_SHAPE",
    "build_independent_raw_corpus",
]
