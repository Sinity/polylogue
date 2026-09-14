"""Re-acquisition never strands an attachments row with no reference.

Projection carry-forward exempts every captured attachment belonging to a
surviving message from the stale-attachment sweep, on the assumption that the
restore will put its ``attachment_refs`` row back. The restore is slot-gated,
and the two identities disagree about what a slot is: ``_attachment_position``
derives it from ``provider_attachment_id`` alone, while ``_attachment_id`` also
folds path, name, MIME type and size. A second acquisition that keeps the
provider id but changes the metadata therefore claims the same slot under a
different ``attachment_id``, the old row is never restored, and -- having been
exempted -- it keeps ``ref_count=1`` with no ``attachment_refs`` row at all.

That is not merely untidy. ``blob_liveness`` treats the presence of an
``attachments`` row bearing a blob hash as a live reference, so the superseded
bytes are pinned against collection forever. Repeated provider revisions with a
stable native id accumulate unreachable rows and blobs until the disk fills.

Anti-vacuity: remove the ``refresh_and_sweep_attachment_rows`` call that now
follows ``_restore_captured_projection_rows`` and the second write leaves two
``attachments`` rows, only one of which has a reference.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.live_ingest import write_session_sync


def _session(name: str, size_bytes: int) -> ParsedSession:
    """One message with one attachment whose provider id never changes."""
    return ParsedSession(
        source_name=Provider.UNKNOWN,
        provider_session_id="conv-1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="See attachment",
                timestamp="2024-01-01T00:00:00Z",
            )
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="att-1",
                message_provider_id="m1",
                name=name,
                mime_type="application/pdf",
                size_bytes=size_bytes,
            )
        ],
    )


def test_a_renamed_attachment_does_not_strand_its_predecessor(tmp_path: Path, frozen_clock: object) -> None:
    del frozen_clock
    db = tmp_path / "index.db"
    with open_connection(db):
        pass

    write_session_sync(db, _session("document.pdf", 2048), raw_id="raw-1")
    # A distinct raw_id is what puts the write on the carry-forward path.
    write_session_sync(db, _session("renamed.pdf", 4096), raw_id="raw-2")

    conn = sqlite3.connect(str(db))
    try:
        attachments = conn.execute(
            "SELECT attachment_id, display_name, ref_count FROM attachments ORDER BY attachment_id"
        ).fetchall()
        refs = conn.execute("SELECT attachment_id FROM attachment_refs").fetchall()
        orphans = conn.execute(
            """
            SELECT a.attachment_id, a.display_name, a.ref_count
            FROM attachments AS a
            LEFT JOIN attachment_refs AS r ON r.attachment_id = a.attachment_id
            WHERE r.attachment_id IS NULL
            """
        ).fetchall()
    finally:
        conn.close()

    assert not orphans, f"attachments rows survived with no reference: {orphans}"
    assert len(attachments) == 1, attachments
    assert attachments[0][1] == "renamed.pdf"
    assert attachments[0][2] == 1
    assert [row[0] for row in refs] == [attachments[0][0]]
