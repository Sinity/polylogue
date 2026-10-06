"""The preamble product route consumes a pinned archive and captured inputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from polylogue.operations.context_preamble import execute_context_preamble
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
from polylogue.surfaces.payloads import ContextPreambleProjectState


class _Archive:
    def __init__(self) -> None:
        self.profile_reads = 0

    def list_session_profile_insights(self, **kwargs: object) -> list[object]:
        self.profile_reads += 1
        assert kwargs == {"sort": "last-message", "tier": "merged", "limit": None}
        return []


@pytest.mark.asyncio
async def test_pinned_preamble_uses_captured_git_and_clock_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pinned preamble completes on a thread that is already driving a loop.

    The web reader's admitted read runs this work nested on a compute worker
    whose loop is running, so this test calls it from inside one. Anti-vacuity:
    drive the builder through ``asyncio.run`` again and the call raises
    "asyncio.run() cannot be called from a running event loop".
    """
    import polylogue.context.preamble as preamble

    def refuse_git(_cwd: str | None) -> Any:
        raise AssertionError("product route must use the caller's git observation")

    def refuse_ledger(_api: object, _assembly: object) -> None:
        raise AssertionError("a READ operation must not write the ledger")

    monkeypatch.setattr(preamble, "_git_project_state", refuse_git)
    monkeypatch.setattr(preamble, "_record_preamble_ledger", refuse_ledger)
    archive = _Archive()
    observed_at = datetime(2026, 9, 26, 12, 30, tzinfo=timezone.utc)
    result = execute_context_preamble(
        archive,  # type: ignore[arg-type]
        session_id=None,
        require_session=False,
        cwd="/captured/worktree",
        observed_at=observed_at,
        observed_project_state=(ContextPreambleProjectState(branch="observed", recent_commits=["abc test"]), None),
        source_tool_calls={"context": "polylogue-session-owner"},
    )

    assert result.payload is not None
    assert result.payload.injected_at == "2026-09-26T12:30:00+00:00"
    assert result.payload.project_state is not None
    assert result.payload.project_state.branch == "observed"
    assert result.ledger is not None
    assert result.observed_at_ms == 1_790_425_800_000
    assert archive.profile_reads == 1


def test_pinned_preamble_reads_judged_guidance_from_attached_user_snapshot(tmp_path: Path) -> None:
    import sqlite3

    from polylogue.core.async_bridge import run_coroutine_sync

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    initialize_archive_database(archive_root / "index.db", ArchiveTier.INDEX)
    initialize_archive_database(archive_root / "user.db", ArchiveTier.USER)
    with sqlite3.connect(archive_root / "user.db") as connection:
        upsert_assertion(
            connection,
            assertion_id="guidance-1",
            target_ref="session:codex:seed",
            kind="decision",
            body_text="Keep context as refs.",
            author_ref="user:local",
            author_kind="user",
            status="active",
            visibility="private",
            context_policy={"inject": True},
            now_ms=1_700_000_000_000,
        )
    with ArchiveStore.open_existing(archive_root) as archive:
        archive.begin_read_snapshot()
        result = execute_context_preamble(
            archive,
            session_id="codex:seed",
            require_session=False,
            observed_at=datetime(2026, 9, 26, tzinfo=timezone.utc),
            observed_project_state=(None, None),
        )
        from polylogue.operations.daemon_reads import execute_read_operation

        daemon = execute_read_operation(
            "read.context",
            {
                "session_id": "codex:seed",
                "require_session": False,
                "observed_at": "2026-09-26T00:00:00+00:00",
                "observed_project_state": None,
                "source_tool_calls": {},
            },
            archive=archive,
            serving_identity="test",
        )
    assert result.payload is not None
    assert result.payload.guidance is not None
    assert not isinstance(result.payload.guidance, str)
    assert result.payload.guidance.assertions[0].quoted_evidence is not None
    assert result.payload.guidance.assertions[0].quoted_evidence.text == "Keep context as refs."
    assert daemon["view"] == "context"
    assert isinstance(daemon["payload"], dict)
    assert daemon["payload"]["guidance"] == result.payload.model_dump(mode="json", exclude_none=True)["guidance"]
    assert not (archive_root / "ops.db").exists()

    from polylogue.api import Polylogue

    api = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    try:
        direct = run_coroutine_sync(api.context_preamble_payload("codex:seed", require_session=False))
    finally:
        run_coroutine_sync(api.close())
    assert direct is not None
    assert direct.guidance == result.payload.guidance

    from polylogue.config import Config
    from polylogue.context.scheduler import read_context_ledger
    from polylogue.operations.facade_writers import record_context_ledger_product

    assert result.ledger is not None
    record_context_ledger_product(
        Config(archive_root=archive_root, render_root=tmp_path / "render", sources=[]),
        result.ledger,
        observed_at_ms=result.observed_at_ms,
    )
    with sqlite3.connect(archive_root / "ops.db") as connection:
        rows = read_context_ledger(connection, target_session="codex:seed")
    assert rows
    assert all(row.observed_at_ms == result.observed_at_ms for row in rows)
