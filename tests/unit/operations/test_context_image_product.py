"""The context image product reads a supplied archive generation."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.api import Polylogue
from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.operations.context_image_product import context_image_from_pinned_reader
from polylogue.operations.daemon_protocol import daemon_operation_spec, validate_operation_result
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
from tests.infra.storage_records import SessionBuilder


def test_pinned_context_image_preserves_message_and_budget_evidence(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (
        SessionBuilder(root / "index.db", "context-image-pinned")
        .provider("codex")
        .title("Pinned image")
        .add_message("one", role="user", text="Pinned archive evidence")
        .add_message("two", role="assistant", text="A bounded reply")
        .save()
    )
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        session_id = archive.list_summaries(limit=1)[0].session_id
        ops_before = hashlib.sha256((root / "ops.db").read_bytes()).hexdigest()
        image = context_image_from_pinned_reader(
            {"seed_session_id": session_id, "max_sessions": 1, "max_tokens": 100, "include_assertions": False},
            archive=archive,
        )
        missing = context_image_from_pinned_reader(
            {"seed_session_id": "codex-session:missing", "max_sessions": 1, "include_assertions": False},
            archive=archive,
        )
    assert image.spec.seed_refs == (f"session:{session_id}",)
    assert image.projection_spec is not None
    assert image.projection_spec.selection.refs == (f"session:{session_id}",)
    assert any("Pinned archive evidence" in (segment.markdown or "") for segment in image.segments)
    assert image.build_ref
    assert image.ledger
    assert any(gap.reason == "not_found" and gap.view == "messages" for gap in missing.omitted)
    assert hashlib.sha256((root / "ops.db").read_bytes()).hexdigest() == ops_before


def test_explicit_context_image_seeds_obey_max_sessions(tmp_path: Path) -> None:
    """Explicit seeds cannot make the product read past its session budget.

    Anti-vacuity: remove the slice of ``seed_session_ids`` in
    ``context_image_from_pinned_reader`` and the compiled image has two
    explicit sessions despite ``max_sessions=1``.
    """
    root = tmp_path / "archive"
    root.mkdir()
    for name in ("context-budget-a", "context-budget-b"):
        SessionBuilder(root / "index.db", name).provider("codex").add_message(
            "one", role="user", text=f"Evidence from {name}"
        ).save()
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        session_ids = [summary.session_id for summary in archive.list_summaries(limit=2)]
        image = context_image_from_pinned_reader(
            {"seed_session_ids": session_ids, "max_sessions": 1, "include_assertions": False},
            archive=archive,
        )
    assert len(image.spec.seed_refs) == 1


def test_pinned_image_matches_facade_compilation(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (
        SessionBuilder(root / "index.db", "context-image-parity")
        .provider("codex")
        .title("Parity image")
        .add_message("one", role="user", text="Same session evidence")
        .save()
    )
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        session_id = archive.list_summaries(limit=1)[0].session_id
        pinned = context_image_from_pinned_reader(
            {"seed_session_id": session_id, "max_sessions": 1, "include_assertions": False}, archive=archive
        )
        selected = context_image_from_pinned_reader({"max_sessions": 1, "include_assertions": False}, archive=archive)
    facade = Polylogue(archive_root=root, db_path=root / "index.db")
    try:
        direct = run_coroutine_sync(
            facade.context_image_payload(seed_session_id=session_id, max_sessions=1, include_assertions=False)
        )
        direct_selected = run_coroutine_sync(facade.context_image_payload(max_sessions=1, include_assertions=False))
    finally:
        run_coroutine_sync(facade.close())
    assert pinned.spec == direct.spec
    assert pinned.segments == direct.segments
    assert pinned.omitted == direct.omitted
    assert pinned.projection_spec == direct.projection_spec
    assert selected.spec == direct_selected.spec
    assert selected.segments == direct_selected.segments
    assert selected.omitted == direct_selected.omitted


def test_composed_views_compile_on_the_pinned_reader_as_the_facade_does(tmp_path: Path) -> None:
    """The CLI's multi-view read is the declared operation, so it must compile every view.

    Anti-vacuity: restore the pinned source's ``ValueError`` for the temporal or
    chronicle view (or drop ``read_views`` from the payload) and the operation
    either fails or compiles only messages, so the segment kinds differ from
    the facade's ``compile_context`` over the same archive.
    """
    from polylogue.archive.context_models import ContextSpec

    root = tmp_path / "archive"
    root.mkdir()
    (
        SessionBuilder(root / "index.db", "context-image-composed")
        .provider("codex")
        .title("Composed image")
        .add_message("one", role="user", text="Composed archive evidence")
        .add_message("two", role="assistant", text="A composed reply")
        .save()
    )
    views = ["temporal", "chronicle", "messages"]
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        session_id = archive.list_summaries(limit=1)[0].session_id
        result = execute_read_operation(
            "read.context-image",
            {
                "seed_session_ids": [session_id],
                "max_sessions": 1,
                "read_views": views,
                "purpose": "continue",
                "max_messages_per_session": None,
                "max_chars_per_message": None,
                "include_assertions": False,
                "observed_at_ms": 1_700_000_000_000,
            },
            archive=archive,
            serving_identity="test",
        )
    validate_operation_result("read.context-image", result)
    payload = cast(dict[str, Any], result["payload"])
    facade = Polylogue(archive_root=root, db_path=root / "index.db")
    try:
        direct = run_coroutine_sync(
            facade.compile_context(
                ContextSpec(
                    purpose="continue",
                    seed_refs=(f"session:{session_id}",),
                    read_views=tuple(views),
                    include_assertions=False,
                )
            )
        )
    finally:
        run_coroutine_sync(facade.close())
    assert payload["spec"]["read_views"] == views
    assert payload["omitted"] == []
    assert [segment["payload_kind"] for segment in payload["segments"]] == [
        segment.payload_kind for segment in direct.segments
    ]
    assert [segment["markdown"] for segment in payload["segments"]] == [segment.markdown for segment in direct.segments]


def test_context_image_operation_preserves_seed_order_and_projection(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    for number in range(2):
        (
            SessionBuilder(root / "index.db", f"context-image-page-{number}")
            .provider("codex")
            .title(f"Image {number}")
            .add_message("one", role="user", text=f"Evidence {number}")
            .save()
        )
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        refs = [summary.session_id for summary in archive.list_summaries(limit=2)]
        selected = list(reversed(refs))
        result = execute_read_operation(
            "read.context-image",
            {
                "seed_session_ids": selected,
                "max_sessions": 2,
                "max_tokens": 100,
                "include_assertions": False,
                "observed_at_ms": 1_700_000_000_000,
            },
            archive=archive,
            serving_identity="test",
        )
    validate_operation_result("read.context-image", result)
    assert result["view"] == "context-image"
    payload = cast(dict[str, Any], result["payload"])
    assert payload["spec"]["seed_refs"] == [f"session:{session_id}" for session_id in selected]
    assert payload["projection_spec"]["selection"]["refs"] == [f"session:{session_id}" for session_id in selected]
    assert len(payload["segments"]) == 2
    declaration = daemon_operation_spec("read.context-image")
    assert declaration is not None and declaration.fallback.value == "never"


def test_assertion_request_requires_pinned_user_tier(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "context-image-user-tier").provider("codex").save()
    (root / "user.db").unlink()
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        with pytest.raises(ArchiveTierUnavailableError):
            context_image_from_pinned_reader({"max_sessions": 1, "include_assertions": True}, archive=archive)


def test_context_image_assertions_follow_pinned_user_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "context-image-snapshot").provider("codex").save()
    initialize_archive_database(root / "user.db", ArchiveTier.USER)
    with sqlite3.connect(root / "user.db") as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        session_id = archive.list_summaries(limit=1)[0].session_id
        target_ref = f"session:{session_id}"

    def add_claim(assertion_id: str) -> None:
        with sqlite3.connect(root / "user.db") as connection:
            upsert_assertion(
                connection,
                assertion_id=assertion_id,
                target_ref=target_ref,
                kind="decision",
                body_text=f"{assertion_id} evidence",
                author_ref="user:local",
                author_kind="user",
                status="active",
                visibility="private",
                context_policy={"inject": True},
                now_ms=1_700_000_000_000,
            )

    add_claim("visible")
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        archive.begin_read_snapshot()
        assert archive.index_connection is not None
        assert archive.index_connection.execute("SELECT COUNT(*) FROM user_tier.assertions").fetchone()[0] == 1
        add_claim("late")
        image = context_image_from_pinned_reader(
            {"seed_session_id": session_id, "include_assertions": True, "observed_at_ms": 1_700_000_000_001},
            archive=archive,
        )
    assert "assertion:visible" in image.assertion_refs
    assert "assertion:late" not in image.assertion_refs
