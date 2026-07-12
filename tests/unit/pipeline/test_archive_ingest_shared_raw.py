"""Regression coverage for polylogue-sjf6.

A Claude Code resume/fork JSONL file physically carries over ONE boundary
record from its parent session (same content, but tagged with the PARENT's
``sessionId``) before switching to its own session id for the rest of the
file. `_claude_code_grouped_record_specs` (dispatch.py) correctly splits such
a file into two `ParsedSession`s that share the identical captured raw bytes.

Before this fix, `pipeline/services/archive_ingest.py`'s one-shot importer
(`parse_sources_archive` / `write_pair`) wrote a SEPARATE `raw_sessions` row
per split session via `write_raw_and_parsed_result`, whose raw_id is derived
from `deterministic_raw_session_id(..., native_id=session.provider_session_id)`
(see `write_source_raw_session`). Two sessions parsed from the SAME bytes
therefore produced TWO DIFFERENT raw_id rows differentiated only by
native_id -- and the live daemon watcher (`sources/live/batch.py`,
`write_raw_payload`) independently commits a THIRD raw for the same bytes
with native_id always NULL. These extra, duplicate raw rows for
byte-identical content are what the daemon's membership-replay guard
(archive.py:2255, "membership replay cannot retire an unrelated accepted
head", added by rgh2/#2718) later discovers as spurious competing claims on
a `logical_source_key` it already has an accepted head for.

Verified directly against the live production archive (2026-07-12): the
exact production file (a 41 MB, 8984-line JSONL whose first record carries
its logical parent session's id) had TWO raw_sessions rows for the identical
`source_path`, one with native_id set (from a `polylogue import` run) and one
with native_id NULL (from the daemon), joined only by identical blob bytes.

This test proves the fix: `write_pair` now caches raw_ids by
(origin, source_path, source_index, blob_hash) so every session parsed from
the SAME physical raw acquisition is indexed against ONE shared raw_id
(computed WITHOUT native_id, matching the daemon's own raw-identity scheme),
using `write_parsed_for_retained_raw_result` for the second and further
sessions instead of writing a duplicate raw row.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.pipeline.services.archive_ingest import parse_sources_archive


def _write_carryover_chain(root: Path) -> tuple[Path, Path]:
    """Write parent.jsonl (real "parent" session) + child.jsonl (a 1-record
    carryover of parent's tail under `sessionId=parent`, then real "child"
    session content) -- the exact structural shape found in production.
    """
    parent_file = root / "parent.jsonl"
    child_file = root / "child.jsonl"

    def rec(session_id: str, uuid: str, role: str, text: str, parent_uuid: str | None = None) -> str:
        import json as _json

        return _json.dumps(
            {
                "type": role,
                "sessionId": session_id,
                "uuid": uuid,
                "parentUuid": parent_uuid,
                "message": {"role": role, "content": [{"type": "text", "text": text}]},
                "timestamp": "2026-02-13T00:00:00.000Z",
            }
        )

    parent_file.write_text(
        "\n".join(
            [
                rec("parent-session", "p-u1", "user", "p1"),
                rec("parent-session", "p-a1", "assistant", "p2", parent_uuid="p-u1"),
            ]
        )
        + "\n"
    )
    child_file.write_text(
        "\n".join(
            [
                # Boundary carryover: same conversation thread, still tagged
                # with the PARENT's sessionId, referencing parent's last uuid.
                rec(
                    "parent-session",
                    "carryover",
                    "user",
                    "[Request interrupted by user for tool use]",
                    parent_uuid="p-a1",
                ),
                rec("child-session", "c-u1", "user", "c1"),
                rec("child-session", "c-a1", "assistant", "c2", parent_uuid="c-u1"),
            ]
        )
        + "\n"
    )
    return parent_file, child_file


def _raw_rows_for_path(source_db: Path, source_path: str) -> list[tuple[str, str | None]]:
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT raw_id, native_id FROM raw_sessions WHERE source_path = ? ORDER BY raw_id",
            (source_path,),
        ).fetchall()
    finally:
        conn.close()
    return [(str(r[0]), r[1]) for r in rows]


@pytest.mark.asyncio
async def test_grouped_carryover_sessions_share_one_raw_row(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    """Two sessions split from ONE Claude Code file's bytes must NOT produce
    two raw_sessions rows for that file -- the specific bug behind
    polylogue-sjf6's live guard crash."""
    archive_root = workspace_env["archive_root"]
    root = tmp_path / "sessions"
    root.mkdir()
    _parent_file, child_file = _write_carryover_chain(root)
    sources = [Source(name="claude-code", path=child_file)]

    result = await parse_sources_archive(archive_root, sources)
    assert result.parse_failures == 0

    rows = _raw_rows_for_path(archive_root / "source.db", str(child_file))
    # Anti-vacuity: this is the production dependency under test --
    # write_pair's shared-raw cache (pipeline/services/archive_ingest.py).
    # Deleting that cache (reverting to one write_raw_and_parsed_result call
    # per split session, each deriving its own native_id-based raw_id) makes
    # this assertion fail with TWO rows instead of one.
    assert len(rows) == 1, f"expected exactly one raw row for child.jsonl's bytes, got {rows}"

    # Both split sessions (the 1-record carryover under "parent-session" and
    # the real "child-session" content) must still have been indexed.
    conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        native_ids = {
            str(r[0])
            for r in conn.execute(
                "SELECT native_id FROM sessions WHERE native_id IN ('parent-session', 'child-session')"
            )
        }
    finally:
        conn.close()
    assert native_ids == {"parent-session", "child-session"}


@pytest.mark.asyncio
async def test_reingesting_identical_bytes_resolves_to_the_same_raw_id(
    tmp_path: Path, workspace_env: dict[str, Path]
) -> None:
    """AC #1: a deterministic fixture re-acquired twice with byte-identical
    bytes must yield the SAME raw_id (and therefore the same downstream
    logical_source_key) both times -- not a second, native_id-differentiated
    raw row that later collides with the first in membership classification.
    """
    archive_root = workspace_env["archive_root"]
    root = tmp_path / "sessions"
    root.mkdir()
    _parent_file, child_file = _write_carryover_chain(root)
    sources = [Source(name="claude-code", path=child_file)]

    await parse_sources_archive(archive_root, sources)
    first_rows = _raw_rows_for_path(archive_root / "source.db", str(child_file))
    assert len(first_rows) == 1
    first_raw_id = first_rows[0][0]

    # Re-ingest the exact same byte-identical file again (simulating the
    # daemon's later catch-up revisit of unchanged content).
    await parse_sources_archive(archive_root, sources)
    second_rows = _raw_rows_for_path(archive_root / "source.db", str(child_file))

    assert len(second_rows) == 1, f"re-ingest must not create a second raw row, got {second_rows}"
    assert second_rows[0][0] == first_raw_id, "raw_id must be deterministic across re-acquisitions of identical bytes"
