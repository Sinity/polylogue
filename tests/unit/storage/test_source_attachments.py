"""Current-source attachment accounting uses real rows and distinct bytes."""

import hashlib
import sqlite3

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.source_attachments import (
    SourceAttachment,
    record_source_attachments,
    source_attachment_census,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.SOURCE)
    conn.execute(
        "INSERT INTO source_generations VALUES ('g', ?, 'path', 0, NULL, 1)",
        (hashlib.sha256(b"").hexdigest(),),
    )
    return conn


def test_census_keeps_duplicate_references_but_counts_payload_bytes_once() -> None:
    conn = _conn()
    payload = b"payload"
    digest = hashlib.sha256(payload).digest()
    record_source_attachments(
        conn,
        source_generation_id="g",
        observed_at_ms=2,
        attachments=(
            SourceAttachment("a", "aistudio-drive", "drive", 2, "p", payload, digest, 7, "acquired"),
            SourceAttachment("b", "aistudio-drive", "drive", 1, "p", payload, digest, 7, "acquired"),
        ),
    )
    census = source_attachment_census(conn, "g")
    assert census["distinct_payload_bytes"] == 7
    groups = census["groups"]
    assert isinstance(groups, list)
    assert groups[0]["reference_count"] == 3
    assert census["sealable"] is True


def test_acquired_without_true_identity_and_unavailable_without_reason_are_rejected() -> None:
    conn = _conn()
    with pytest.raises(ValueError, match="hash, bytes, and payload identity"):
        record_source_attachments(
            conn,
            source_generation_id="g",
            observed_at_ms=2,
            attachments=(SourceAttachment("a", "o", "c", disposition="acquired"),),
        )
    with pytest.raises(ValueError, match="evidence-backed reason"):
        record_source_attachments(
            conn,
            source_generation_id="g",
            observed_at_ms=2,
            attachments=(SourceAttachment("a", "o", "c", disposition="expired"),),
        )
