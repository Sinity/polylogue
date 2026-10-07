"""Drive document revisions are governed by membership on the retained route.

AI Studio rewrites the whole Drive JSON on every save, so a later capture of a
document is never a byte prefix of an earlier one. Each capture is its own
retained raw; the census parses it into the document's logical cohort
(``aistudio-drive:<id>``), and membership classification decides which capture
holds the content every other capture is contained in. That decision, not
acquisition order or a freshness heuristic, selects the Index head.

Anti-vacuity: choosing the head by acquisition order turns the regression
and arrival-order laws red, since in each the latest capture is not the
containing one.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner

_KEY = "aistudio-drive:neutral-document"
_TURNS = (
    {"id": "t1", "role": "user", "text": "Neutral question"},
    {"id": "t2", "role": "model", "text": "Neutral answer"},
    {"id": "t3", "role": "user", "text": "Neutral follow-up"},
)


def _document(turns: int, *, indent: int | None = None) -> bytes:
    """One Drive document holding its first ``turns`` turns, serialized whole."""
    payload = {"id": "neutral-document", "chunkedPrompt": {"chunks": list(_TURNS[:turns])}}
    return json.dumps(payload, indent=indent).encode()


async def _capture(root: Path, payload: bytes, acquired_at_ms: int) -> str:
    """Acquire one capture of the document's single Drive path and ingest it."""

    def acquire() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.GEMINI,
                payload=payload,
                source_path="Google AI Studio/neutral-document",
                canonical_source_path="Google AI Studio/neutral-document",
                acquired_at_ms=acquired_at_ms,
            )
            archive.commit()
        return raw_id

    raw_id = await run_archive_fixture_write(root, acquire)
    async with prepared_live_convergence_owner(root) as owner:
        (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
    return raw_id


def _decisions(root: Path) -> dict[str, tuple[str, str]]:
    with sqlite3.connect(root / "source.db") as conn:
        return {
            str(raw_id): (str(decision), str(authority))
            for raw_id, decision, authority in conn.execute(
                "SELECT raw_id, decision, revision_authority FROM raw_session_memberships WHERE logical_source_key=?",
                (_KEY,),
            )
        }


def _head(root: Path) -> tuple[str, int]:
    with sqlite3.connect(root / "index.db") as conn:
        rows = conn.execute("SELECT raw_id, message_count FROM sessions WHERE session_id=?", (_KEY,)).fetchall()
    assert len(rows) == 1
    return str(rows[0][0]), int(rows[0][1])


@pytest.mark.asyncio
async def test_a_grown_rewrite_supersedes_its_earlier_capture(tmp_path: Path) -> None:
    earlier = await _capture(tmp_path, _document(2), 1)
    later = await _capture(tmp_path, _document(3), 2)

    assert _decisions(tmp_path) == {
        earlier: ("superseded_prefix", "byte_proven"),
        later: ("applied", "byte_proven"),
    }
    assert _head(tmp_path) == (later, 3)


@pytest.mark.asyncio
async def test_a_reserialized_identical_document_is_one_revision(tmp_path: Path) -> None:
    """A save that only re-serializes the same content is equivalent, not growth."""
    compact = await _capture(tmp_path, _document(2), 1)
    indented = await _capture(tmp_path, _document(2, indent=2), 2)

    decisions = _decisions(tmp_path)
    assert sorted(decision for decision, _authority in decisions.values()) == ["applied", "superseded_equivalent"]
    assert {authority for _decision, authority in decisions.values()} == {"byte_proven"}
    head, messages = _head(tmp_path)
    assert head in {compact, indented} and decisions[head][0] == "applied"
    assert messages == 2


@pytest.mark.asyncio
async def test_a_later_smaller_capture_does_not_replace_the_larger_head(tmp_path: Path) -> None:
    larger = await _capture(tmp_path, _document(3), 1)
    smaller = await _capture(tmp_path, _document(2), 2)

    assert _decisions(tmp_path) == {
        larger: ("applied", "byte_proven"),
        smaller: ("superseded_prefix", "byte_proven"),
    }
    assert _head(tmp_path) == (larger, 3)


@pytest.mark.asyncio
async def test_the_head_is_the_containing_capture_whatever_the_arrival_order(tmp_path: Path) -> None:
    middle = await _capture(tmp_path, _document(2), 1)
    largest = await _capture(tmp_path, _document(3), 2)
    smallest = await _capture(tmp_path, _document(1), 3)

    assert _decisions(tmp_path) == {
        middle: ("superseded_prefix", "byte_proven"),
        largest: ("applied", "byte_proven"),
        smallest: ("superseded_prefix", "byte_proven"),
    }
    assert _head(tmp_path) == (largest, 3)
