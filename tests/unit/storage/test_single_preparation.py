"""A fresh retained raw is prepared and parsed once across all its phases.

A fresh raw needs its parser census, then its byte classification, then its
replay. Each phase used to commit and end its preparation, so the next phase
opened another seal and parsed the same bytes again: three preparations and
two parses for every raw a reset re-ingests. The census and classification now
commit on one tape in place, and the replay continues on the same seal with
the artifact already parsed.

Anti-vacuity: end the preparation at each committed phase (return the census
replacement to ``publish``) and the raw takes two preparations; drop the
carried artifact cache and it is parsed twice.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import polylogue.sources.revision_backfill as revision_backfill
import polylogue.storage.sqlite.reference_seal as reference_seal
from polylogue.core.enums import Provider
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.retained_replay import replay_retained_components


def _rollout(session_id: str) -> bytes:
    records: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-06-01T00:00:00Z"}}
    ]
    for position, text in enumerate(("question", "answer")):
        records.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"m{position}",
                    "role": "user" if position % 2 == 0 else "assistant",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
    return b"".join(json.dumps(record, separators=(",", ":")).encode() + b"\n" for record in records)


def test_fresh_singleton_raw_is_prepared_and_parsed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_rollout("single"),
            source_path="single.jsonl",
            canonical_source_path="single.jsonl",
            acquired_at_ms=1,
            native_id="single",
        )

    seals: list[object] = []
    parses: list[str] = []
    original_seal_init = reference_seal.PreparedIndexMutation.__init__
    original_parse = revision_backfill.prepare_retained_jsonl_artifact

    def counting_seal_init(self: Any, *args: Any, **kwargs: Any) -> None:
        seals.append(self)
        original_seal_init(self, *args, **kwargs)

    def counting_parse(reader: Any, parsed_raw_id: str, *, directory: Path) -> Any:
        parses.append(parsed_raw_id)
        return original_parse(reader, parsed_raw_id, directory=directory)

    monkeypatch.setattr(reference_seal.PreparedIndexMutation, "__init__", counting_seal_init)
    monkeypatch.setattr(revision_backfill, "prepare_retained_jsonl_artifact", counting_parse)

    run = replay_retained_components(tmp_path)

    assert run.replayed_logical_sources == 1
    assert run.components == ((raw_id,),)
    # The census committed in place before the replay, and is still reported.
    assert run.scanned == 1
    assert len(seals) == 1, f"one preparation serves census, classification and replay; opened {len(seals)}"
    assert parses == [raw_id], f"the artifact is parsed once across its phases; parsed {parses}"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT session_id, message_count FROM sessions").fetchall() == [
            ("codex-session:single", 2)
        ]
