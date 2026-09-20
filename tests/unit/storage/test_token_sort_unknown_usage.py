"""``sort=tokens`` must not rank unmeasured usage as a measured zero."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import _summary_order_by
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_rows, write_parsed_session_to_archive


def _session(provider_session_id: str, *, timestamp: str, tokens: int | None) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=provider_session_id,
        created_at=timestamp,
        updated_at=timestamp,
        messages=[
            ParsedMessage(
                provider_message_id=f"{provider_session_id}-m1",
                role=Role.ASSISTANT,
                text="body",
                timestamp=timestamp,
                input_tokens=tokens,
                output_tokens=tokens,
                cache_read_tokens=tokens,
                cache_write_tokens=tokens,
            )
        ],
    )


def test_ascending_token_sort_ranks_measured_zero_ahead_of_unmeasured(tmp_path: Path) -> None:
    """A session whose provider never reported usage is not "the fewest tokens".

    ``unknown`` is written first so the pre-fix ordering (both sessions
    collapsing to a COALESCEd 0 and tie-breaking on ``sort_key_ms``) puts it
    ahead. Anti-vacuity: drop the leading ``NOT EXISTS`` key from
    ``_summary_order_by``'s ``tokens`` branch and the unmeasured session
    returns to the front of the ascending band, making this red.
    """
    path = tmp_path / "index.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        written: dict[str, str] = {}
        for name, timestamp, tokens in (
            ("unknown", "2026-01-01T00:00:00Z", None),
            ("measured-zero", "2026-01-02T00:00:00Z", 0),
            ("measured-ten", "2026-01-03T00:00:00Z", 10),
        ):
            session = _session(name, timestamp=timestamp, tokens=tokens)
            written[name] = write_parsed_session_to_archive(
                conn,
                session,
                content_hash=str(session_content_hash(session)),
                prepared=prepare_session_rows(session),
            )

        order_by = _summary_order_by(sample=False, sort="tokens", reverse=True)
        ascending = [row[0] for row in conn.execute(f"SELECT s.session_id FROM sessions AS s {order_by}")]
        assert ascending == [written["measured-zero"], written["measured-ten"], written["unknown"]]

        descending_order_by = _summary_order_by(sample=False, sort="tokens", reverse=False)
        descending = [row[0] for row in conn.execute(f"SELECT s.session_id FROM sessions AS s {descending_order_by}")]
        assert descending == [written["measured-ten"], written["measured-zero"], written["unknown"]]
    finally:
        conn.close()
