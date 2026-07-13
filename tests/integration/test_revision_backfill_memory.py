"""Evidence harness for daemon memory retained during raw revision recovery.

Static trace before measurement:

| Candidate | Production site | Trigger | Retention/magnitude hypothesis |
| --- | --- | --- | --- |
| H1 | ``_parse_retained_raw`` | every uncensused raw | parsed Pydantic graph up to one raw payload |
| H2 | ``_ParsedSessionSpill.add`` | every parsed raw | serialized graph reaches SQLite; Python graph must not survive archive-wide |
| H3 | ``session_content_hash`` | accepted replay | full aggregate message text is reprojected and JSON-serialized |
| H4 | ``apply_raw_revision_replay`` | accepted byte chain | aggregate chain plus indexing lives through one replay transaction |

The live py-spy stack identified H2 and then H3/H4.  This test measures all
four boundaries through the real backfill entrypoint; it does not turn an RSS
sample into a brittle budget.  Its deterministic proxies are payload bytes,
parsed graph shape, serialized spill bytes, and text bytes hashed/replayed.
"""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.revision_memory_counter import revision_memory_counter


def _codex_full_revision(session_id: str, *, message_count: int, text_bytes: int) -> bytes:
    records: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-07-13T00:00:00Z"}}
    ]
    for index in range(message_count):
        text = f"{session_id}-{index:04d}-" + ("x" * text_bytes)
        records.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{session_id}-message-{index}",
                    "role": "user" if index % 2 == 0 else "assistant",
                    "timestamp": f"2026-07-13T00:{index % 60:02d}:00Z",
                    "content": [{"type": "input_text" if index % 2 == 0 else "output_text", "text": text}],
                },
            }
        )
    return b"".join(json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n" for record in records)


def test_raw_revision_backfill_measures_parse_spill_hash_and_replay(tmp_path: Path) -> None:
    """A newest full revision yields one replay while both raw graphs pass the spill."""
    initialize_active_archive_root(tmp_path)
    baseline = _codex_full_revision("memory-proof", message_count=12, text_bytes=8_192)
    newest = _codex_full_revision("memory-proof", message_count=24, text_bytes=8_192)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path="memory-proof.jsonl",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                "codex:memory-proof",
                RawRevisionKind.FULL,
                "memory-proof-baseline",
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=newest,
            source_path="memory-proof.jsonl",
            acquired_at_ms=2,
            revision=RawRevisionEnvelope(
                "codex:memory-proof",
                RawRevisionKind.FULL,
                "memory-proof-newest",
                1,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )

    with revision_memory_counter() as counter:
        result = backfill_historical_revision_evidence(tmp_path)

    assert result.scanned == 2
    assert result.replayed_logical_sources == 1
    assert counter.calls_by_site["parse_retained_raw"] == 2
    assert counter.bytes_by_site["parse_retained_raw"] == len(baseline) + len(newest)
    assert counter.calls_by_site["parsed_session_spill.add"] == 2
    assert counter.bytes_by_site["parsed_session_spill.add"] > 0
    assert counter.parsed_sessions == 2
    assert counter.parsed_messages == 36
    assert counter.parsed_text_bytes >= 36 * 8_192
    assert counter.calls_by_site["session_content_hash"] >= 1
    assert counter.bytes_by_site["session_content_hash"] >= 24 * 8_192
    assert counter.calls_by_site["apply_raw_revision_replay"] == 1
    assert counter.bytes_by_site["apply_raw_revision_replay"] >= 24 * 8_192
    phase_names = [phase.name for phase in counter.phases]
    assert phase_names[:5] == [
        "parse_retained_raw",
        "parsed_session_spill.add",
        "parse_retained_raw",
        "parsed_session_spill.add",
        "apply_raw_revision_replay:before",
    ], counter.summary()
    assert phase_names[-1] == "apply_raw_revision_replay:after", counter.summary()
    assert all(
        phase_names.index("apply_raw_revision_replay:before") < index < len(phase_names) - 1
        for index, name in enumerate(phase_names)
        if name == "session_content_hash"
    ), counter.summary()
