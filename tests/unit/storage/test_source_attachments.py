"""Current-source attachment accounting uses real rows and distinct bytes."""

import hashlib
import sqlite3
from typing import Any, cast

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.source_attachments import (
    SourceAttachment,
    SourceAttachmentConflictError,
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
            attachments=(SourceAttachment("a", "aistudio-drive", "c", disposition="acquired"),),
        )
    with pytest.raises(ValueError, match="evidence-backed reason"):
        record_source_attachments(
            conn,
            source_generation_id="g",
            observed_at_ms=2,
            attachments=(SourceAttachment("a", "aistudio-drive", "c", disposition="expired"),),
        )


def test_attachment_writes_are_refused_after_generation_is_sealed() -> None:
    """A sealed census cannot gain a delayed reference.

    Anti-vacuity: remove the generation seal check and this late attachment
    inserts successfully into the supposedly immutable denominator.
    """
    conn = _conn()
    conn.execute("UPDATE source_generations SET sealed_at_ms=3 WHERE source_generation_id='g'")
    conn.commit()

    with pytest.raises(ValueError, match="already sealed"):
        record_source_attachments(
            conn,
            source_generation_id="g",
            observed_at_ms=4,
            attachments=(SourceAttachment("late", "aistudio-drive", "drive", disposition="pending", reason="queued"),),
        )

    assert conn.execute("SELECT COUNT(*) FROM source_attachments").fetchone()[0] == 0


def test_identical_attachment_replay_is_allowed_after_sealing() -> None:
    conn = _conn()
    payload = b"payload"
    attachment = SourceAttachment(
        "a", "aistudio-drive", "drive", 1, "p", payload, hashlib.sha256(payload).digest(), len(payload), "acquired"
    )
    record_source_attachments(conn, source_generation_id="g", observed_at_ms=2, attachments=(attachment,))
    conn.execute("UPDATE source_generations SET sealed_at_ms=3 WHERE source_generation_id='g'")
    conn.commit()

    record_source_attachments(conn, source_generation_id="g", observed_at_ms=9, attachments=(attachment,))

    assert conn.execute("SELECT updated_at_ms FROM source_attachments").fetchone()[0] == 2


def test_attachment_writer_validates_domain_origin_and_storage_disposition_before_batch_writes() -> None:
    conn = _conn()
    with pytest.raises(ValueError, match="origin"):
        record_source_attachments(
            conn,
            source_generation_id="g",
            observed_at_ms=2,
            attachments=(
                SourceAttachment("valid", "aistudio-drive", "drive", disposition="pending", reason="queued"),
                SourceAttachment("invalid", "not-an-origin", "drive", disposition="pending", reason="queued"),
            ),
        )
    assert conn.execute("SELECT COUNT(*) FROM source_attachments").fetchone()[0] == 0

    with pytest.raises(ValueError, match="disposition"):
        record_source_attachments(
            conn,
            source_generation_id="g",
            observed_at_ms=2,
            attachments=(
                SourceAttachment(
                    "invalid-disposition", "aistudio-drive", "drive", disposition=cast(Any, "not-a-state")
                ),
            ),
        )
    assert conn.execute("SELECT COUNT(*) FROM source_attachments").fetchone()[0] == 0

    # source_class deliberately remains an open, nonempty grouping label.
    record_source_attachments(
        conn,
        source_generation_id="g",
        observed_at_ms=2,
        attachments=(
            SourceAttachment("valid", "aistudio-drive", "future-provider-kind", disposition="pending", reason="queued"),
        ),
    )
    assert tuple(conn.execute("SELECT origin, source_class, disposition FROM source_attachments").fetchone()) == (
        "aistudio-drive",
        "future-provider-kind",
        "pending",
    )


def test_sql_bypass_can_store_an_unowned_attachment_origin() -> None:
    """Origin is domain-owned even though source DDL intentionally allows any nonempty token."""
    conn = _conn()
    conn.execute(
        "INSERT INTO source_attachments(source_generation_id, reference_id, origin, source_class, reachability, "
        "reference_count, disposition, reason, observed_at_ms, updated_at_ms) "
        "VALUES ('g', 'direct', 'not-an-origin', 'drive', 'unavailable', 1, 'pending', 'queued', 1, 1)"
    )
    assert conn.execute("SELECT origin FROM source_attachments").fetchone()[0] == "not-an-origin"


def test_identical_replay_is_a_noop() -> None:
    """Idempotent replay of the same immutable input changes nothing (polylogue-8v4rm).

    Anti-vacuity: make ``_apply_replay`` raise unconditionally on a second
    recording and this goes red.
    """
    conn = _conn()
    payload = b"payload"
    digest = hashlib.sha256(payload).digest()
    attachment = SourceAttachment("a", "aistudio-drive", "drive", 1, "p", payload, digest, 7, "acquired")
    record_source_attachments(conn, source_generation_id="g", observed_at_ms=2, attachments=(attachment,))
    record_source_attachments(conn, source_generation_id="g", observed_at_ms=9, attachments=(attachment,))

    rows = conn.execute("SELECT disposition, updated_at_ms FROM source_attachments").fetchall()
    assert [tuple(row) for row in rows] == [("acquired", 2)]


def test_a_conflicting_terminal_fact_is_visible_not_silently_dropped() -> None:
    """A different settled outcome for one reference must not be swallowed.

    ``ON CONFLICT DO NOTHING`` kept whichever arrived first with no signal.
    Anti-vacuity: restore ``ON CONFLICT(...) DO NOTHING`` on the INSERT and
    this goes red -- the second recording becomes a silent no-op.
    """
    conn = _conn()
    payload = b"payload"
    digest = hashlib.sha256(payload).digest()
    record_source_attachments(
        conn,
        source_generation_id="g",
        observed_at_ms=2,
        attachments=(SourceAttachment("a", "aistudio-drive", "drive", 1, "p", payload, digest, 7, "acquired"),),
    )

    with pytest.raises(SourceAttachmentConflictError) as excinfo:
        record_source_attachments(
            conn,
            source_generation_id="g",
            observed_at_ms=3,
            attachments=(
                SourceAttachment("a", "aistudio-drive", "drive", 1, disposition="expired", reason="gone upstream"),
            ),
        )

    assert excinfo.value.reference_id == "a"
    assert conn.execute("SELECT disposition FROM source_attachments").fetchone()[0] == "acquired"


def test_pending_to_terminal_is_allowed_progress() -> None:
    """The one declared transition: a pending reference settles.

    Anti-vacuity: treat every difference as a conflict and this goes red.
    """
    conn = _conn()
    record_source_attachments(
        conn,
        source_generation_id="g",
        observed_at_ms=2,
        attachments=(SourceAttachment("a", "aistudio-drive", "drive", 1, disposition="pending", reason="queued"),),
    )
    payload = b"payload"
    digest = hashlib.sha256(payload).digest()
    record_source_attachments(
        conn,
        source_generation_id="g",
        observed_at_ms=5,
        attachments=(SourceAttachment("a", "aistudio-drive", "drive", 1, "p", payload, digest, 7, "acquired"),),
    )

    row = conn.execute("SELECT disposition, reachability, byte_count, updated_at_ms FROM source_attachments").fetchone()
    assert tuple(row) == ("acquired", "current", 7, 5)


def test_acquired_source_attachment_bytes_have_a_durable_owner() -> None:
    """polylogue-8v4rm AC1: the source tier's own claim keeps the bytes alive.

    Anti-vacuity: drop the ``source_attachments`` entry from ``BLOB_OWNERS``
    and this goes red -- the only liveness surface for those bytes would again
    be the rebuildable index tier.
    """
    from polylogue.storage.blob_liveness import BLOB_OWNERS

    owners = {(owner.tier, owner.table, owner.blob_column) for owner in BLOB_OWNERS}
    assert ("source", "source_attachments", "blob_hash") in owners


@pytest.mark.parametrize("row_factory", [None, sqlite3.Row])
def test_attachment_census_and_generation_seal_use_declared_columns(row_factory: Any) -> None:
    from contextlib import closing

    from polylogue.storage.sqlite.archive_tiers.source_items import seal_source_generation

    with closing(_conn()) as conn:
        conn.row_factory = row_factory
        payload = b"neutral attachment"
        record_source_attachments(
            conn,
            source_generation_id="g",
            observed_at_ms=2,
            attachments=(
                SourceAttachment(
                    "a",
                    "aistudio-drive",
                    "drive",
                    2,
                    "p",
                    payload,
                    hashlib.sha256(payload).digest(),
                    len(payload),
                    "acquired",
                ),
            ),
        )
        census = source_attachment_census(conn, "g")
        assert census["groups"] == [
            {
                "origin": "aistudio-drive",
                "source_class": "drive",
                "reachability": "current",
                "disposition": "acquired",
                "reference_rows": 1,
                "reference_count": 2,
                "distinct_payloads": 1,
                "distinct_blobs": 1,
                "bytes": len(payload),
            }
        ]
        assert census["distinct_payload_bytes"] == len(payload)
        seal_source_generation(conn, source_generation_id="g", sealed_at_ms=3)
        assert tuple(conn.execute("SELECT sealed_at_ms FROM source_generations").fetchone()) == (3,)


@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize("first_kind", ["new", "pending", "same-reference"])
def test_later_attachment_conflict_rolls_back_the_complete_batch(commit: bool, first_kind: str) -> None:
    from contextlib import closing

    with closing(_conn()) as conn:
        settled = SourceAttachment("b", "aistudio-drive", "drive", disposition="expired", reason="original")
        pending = SourceAttachment("a", "aistudio-drive", "drive", disposition="pending", reason="queued")
        initial = (settled, pending) if first_kind == "pending" else (settled,)
        record_source_attachments(conn, source_generation_id="g", observed_at_ms=2, attachments=initial)
        conn.execute("UPDATE source_generations SET created_at_ms=9")
        first = SourceAttachment("a", "aistudio-drive", "drive", disposition="expired", reason="offered")
        second = SourceAttachment("b", "aistudio-drive", "drive", disposition="expired", reason="changed")
        if first_kind == "same-reference":
            second = SourceAttachment("a", "aistudio-drive", "drive", disposition="expired", reason="changed")
        before = [tuple(row) for row in conn.execute("SELECT * FROM source_attachments ORDER BY reference_id")]
        with pytest.raises(SourceAttachmentConflictError):
            record_source_attachments(
                conn, source_generation_id="g", observed_at_ms=3, attachments=(first, second), commit=commit
            )
        assert conn.in_transaction
        conn.commit()
        assert [tuple(row) for row in conn.execute("SELECT * FROM source_attachments ORDER BY reference_id")] == before
        assert tuple(conn.execute("SELECT created_at_ms FROM source_generations").fetchone()) == (9,)


def test_attachment_commit_false_keeps_new_transaction_uncommitted() -> None:
    from contextlib import closing

    with closing(_conn()) as conn:
        conn.commit()
        record_source_attachments(
            conn,
            source_generation_id="g",
            observed_at_ms=2,
            commit=False,
            attachments=(SourceAttachment("a", "aistudio-drive", "drive", disposition="expired", reason="neutral"),),
        )
        assert conn.in_transaction
        conn.rollback()
        assert conn.execute("SELECT COUNT(*) FROM source_attachments").fetchone()[0] == 0
