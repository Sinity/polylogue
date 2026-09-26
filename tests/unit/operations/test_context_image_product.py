"""The context image product reads a supplied archive generation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from polylogue.api import Polylogue
from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.operations.context_image_product import context_image_from_pinned_reader
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
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
            {"seed_session_id": session_id, "max_sessions": 1, "max_tokens": 100},
            archive=archive,
        )
        missing = context_image_from_pinned_reader(
            {"seed_session_id": "codex-session:missing", "max_sessions": 1},
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
        pinned = context_image_from_pinned_reader({"seed_session_id": session_id, "max_sessions": 1}, archive=archive)
        selected = context_image_from_pinned_reader({"max_sessions": 1}, archive=archive)
    facade = Polylogue(archive_root=root, db_path=root / "index.db")
    try:
        direct = run_coroutine_sync(facade.context_image_payload(seed_session_id=session_id, max_sessions=1))
        direct_selected = run_coroutine_sync(facade.context_image_payload(max_sessions=1))
    finally:
        run_coroutine_sync(facade.close())
    assert pinned.spec == direct.spec
    assert pinned.segments == direct.segments
    assert pinned.omitted == direct.omitted
    assert pinned.projection_spec == direct.projection_spec
    assert selected.spec == direct_selected.spec
    assert selected.segments == direct_selected.segments
    assert selected.omitted == direct_selected.omitted
