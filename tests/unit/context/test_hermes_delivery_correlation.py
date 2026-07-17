"""fs1.11: correlate Hermes context_injected events with delivery receipts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.context.compiler import (
    ContextImage,
    ContextSegment,
    ContextSpec,
    context_snapshot_record_from_image,
)
from polylogue.context.hermes_delivery_correlation import correlate_hermes_context_deliveries
from polylogue.core.refs import EvidenceRef
from polylogue.sources.hooks import drain_hook_event_spool, enqueue_hook_event
from polylogue.sources.parsers.hermes_lifecycle import CONTEXT_INJECTED
from polylogue.storage.sqlite.archive_tiers.context_delivery_write import write_context_delivery
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL, USER_SCHEMA_VERSION


def _user_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(USER_DDL)
    conn.execute(f"PRAGMA user_version = {USER_SCHEMA_VERSION}")
    return conn


def _image() -> ContextImage:
    evidence = EvidenceRef(session_id="hermes-session:hermes-conv-1@profile-abc123", message_id="m1")
    segment = ContextSegment(
        segment_id="read-view:hermes-conv-1:messages",
        kind="read_view",
        title="Messages",
        markdown="user: hi\n",
        evidence_refs=(evidence,),
        token_estimate=7,
    )
    return ContextImage(
        spec=ContextSpec(seed_refs=("hermes-session:hermes-conv-1@profile-abc123",), read_views=(), max_tokens=4000),
        segments=(segment,),
        evidence_refs=(evidence,),
        token_estimate=7,
    )


def _source_conn(archive_root: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def test_context_injected_event_correlates_with_its_delivery_receipt(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    spool_root = tmp_path / "hooks"

    user_conn = _user_conn()
    image = _image()
    record = context_snapshot_record_from_image(image, boundary="hermes-turn-start", run_ref=None)
    written = write_context_delivery(
        user_conn,
        image=image,
        record=record,
        recipient_ref="agent:hermes-conv-1",
        delivered_by_ref="user:local",
        delivered_at_ms=1_000,
    )

    enqueue_hook_event(
        event_id="ctx-inject-1",
        provider="hermes",
        event_type=CONTEXT_INJECTED,
        session_id="hermes-conv-1",
        timestamp="2026-07-12T10:00:00Z",
        payload={"snapshot_ref": written.snapshot_ref},
        root=spool_root,
    )
    assert drain_hook_event_spool(archive_root, root=spool_root).acknowledged == 1

    source_conn = _source_conn(archive_root)
    correlations = correlate_hermes_context_deliveries(source_conn, user_conn, hermes_session_native_id="hermes-conv-1")

    assert len(correlations) == 1
    correlation = correlations[0]
    assert correlation.available is True
    assert correlation.snapshot_ref == written.snapshot_ref
    assert correlation.receipt is not None
    assert correlation.receipt.context_image_sha256 == written.context_image_sha256
    assert correlation.rendered_bytes_sha256 == written.context_image_sha256
    assert correlation.token_budget == "4000"
    assert correlation.rendered_token_estimate == "7"
    assert correlation.caveats == ()


def test_context_injected_event_without_matching_receipt_is_explicit_not_silent(tmp_path: Path) -> None:
    """AC: archive outage/missing receipt renders an explicit unavailable state."""

    archive_root = tmp_path / "archive"
    spool_root = tmp_path / "hooks"
    user_conn = _user_conn()

    enqueue_hook_event(
        event_id="ctx-inject-orphan",
        provider="hermes",
        event_type=CONTEXT_INJECTED,
        session_id="hermes-conv-1",
        timestamp="2026-07-12T10:00:00Z",
        payload={"snapshot_ref": "context-snapshot:doesnotexist0000"},
        root=spool_root,
    )
    assert drain_hook_event_spool(archive_root, root=spool_root).acknowledged == 1

    source_conn = _source_conn(archive_root)
    correlations = correlate_hermes_context_deliveries(source_conn, user_conn, hermes_session_native_id="hermes-conv-1")

    assert len(correlations) == 1
    correlation = correlations[0]
    assert correlation.available is False
    assert correlation.receipt is None
    assert correlation.caveats
    assert "no context_deliveries receipt found" in correlation.caveats[0]


def test_context_injected_event_missing_snapshot_ref_is_explicit(tmp_path: Path) -> None:
    """A malformed producer event (no snapshot_ref) is a visible caveat, not a crash."""

    archive_root = tmp_path / "archive"
    spool_root = tmp_path / "hooks"
    user_conn = _user_conn()

    enqueue_hook_event(
        event_id="ctx-inject-malformed",
        provider="hermes",
        event_type=CONTEXT_INJECTED,
        session_id="hermes-conv-1",
        timestamp="2026-07-12T10:00:00Z",
        payload={"turn_id": "turn-1"},  # no snapshot_ref
        root=spool_root,
    )
    assert drain_hook_event_spool(archive_root, root=spool_root).acknowledged == 1

    source_conn = _source_conn(archive_root)
    correlations = correlate_hermes_context_deliveries(source_conn, user_conn, hermes_session_native_id="hermes-conv-1")

    assert len(correlations) == 1
    correlation = correlations[0]
    assert correlation.snapshot_ref is None
    assert correlation.available is False
    assert "carries no snapshot_ref" in correlation.caveats[0]


def test_non_context_injected_hermes_events_are_ignored(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    spool_root = tmp_path / "hooks"
    user_conn = _user_conn()

    enqueue_hook_event(
        event_id="tool-1",
        provider="hermes",
        event_type="tool_start",
        session_id="hermes-conv-1",
        timestamp="2026-07-12T10:00:00Z",
        payload={"tool_call_id": "call-1"},
        root=spool_root,
    )
    assert drain_hook_event_spool(archive_root, root=spool_root).acknowledged == 1

    source_conn = _source_conn(archive_root)
    correlations = correlate_hermes_context_deliveries(source_conn, user_conn, hermes_session_native_id="hermes-conv-1")
    assert correlations == ()
