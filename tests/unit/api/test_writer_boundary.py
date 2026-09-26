"""Public facade mutations cross the resident daemon's writer boundary."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.api import Polylogue
from polylogue.api.facade_client import FacadeDaemonRequiredError
from polylogue.config import Config
from polylogue.context.preamble import _record_preamble_ledger
from polylogue.context.scheduler import ContextAssembly, schedule_context
from polylogue.core.enums import AssertionKind
from polylogue.core.refs import ExecutionContextRef
from polylogue.daemon.socket_path import daemon_socket_path
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.daemon_operations import running_daemon_operations


def _assembly() -> ContextAssembly:
    return schedule_context(
        (),
        moment="context-preamble",
        target_session=None,
        execution_context=ExecutionContextRef.from_legacy_id("embedded-boundary"),
        token_budget=1,
    )


@pytest.mark.asyncio
async def test_public_facade_refuses_mutation_when_daemon_is_absent(tmp_path: Path) -> None:
    """An initialized archive without a daemon cannot accept a facade write."""
    root = tmp_path / "archive"
    with ArchiveStore(root):
        pass
    archive = Polylogue(archive_root=root)
    try:
        with pytest.raises(FacadeDaemonRequiredError):
            await archive.capture_assertion_candidate(body_text="candidate", kind=AssertionKind.LESSON)
        with sqlite3.connect(root / "user.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0] == 0
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_public_facade_mutations_roundtrip_through_daemon_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real UDS handler owns the commit and returns the typed facade result."""
    root = tmp_path / "archive"
    monkeypatch.setattr("polylogue.daemon.api_auth.resolve_api_auth_token", lambda *_args, **_kwargs: None)
    with running_daemon_operations(root, socket_path=daemon_socket_path(root)):
        archive = Polylogue(archive_root=root)
        try:
            first = await archive.capture_assertion_candidate(
                body_text="synthetic candidate",
                kind=AssertionKind.LESSON,
                author_ref="agent:boundary-test",
                author_kind="agent",
                idempotency_key="stable-key",
            )
            second = await archive.capture_assertion_candidate(
                body_text="synthetic candidate",
                kind=AssertionKind.LESSON,
                author_ref="agent:boundary-test",
                author_kind="agent",
                idempotency_key="stable-key",
            )
            assert second.assertion_id == first.assertion_id
            with sqlite3.connect(root / "user.db") as conn:
                assert (
                    conn.execute(
                        "SELECT COUNT(*) FROM assertions WHERE assertion_id = ?", (first.assertion_id,)
                    ).fetchone()[0]
                    == 1
                )

            from polylogue.context.compiler import ContextImage, ContextSpec

            image = ContextImage(spec=ContextSpec(seed_refs=("session:codex:synthetic",), read_views=()), segments=())
            delivered = await archive.record_context_delivery(
                image=image,
                boundary="explicit-recall",
                recipient_ref="agent:boundary-test",
                delivered_by_ref="user:local",
            )
            repeated = await archive.record_context_delivery(
                image=image,
                boundary="explicit-recall",
                recipient_ref="agent:boundary-test",
                delivered_by_ref="user:local",
            )
            assert delivered.context_image == image
            assert repeated.outcome == "idempotent"
            assert repeated.snapshot_ref == delivered.snapshot_ref
            with sqlite3.connect(root / "user.db") as conn:
                assert conn.execute("SELECT COUNT(*) FROM context_deliveries").fetchone()[0] == 1

            # An unwatched view is stored as an opaque query object. The
            # stricter standing-query operation must not narrow this facade.
            assert await archive.save_view("opaque-view", "Opaque view", '{"custom_filter": true}') is True
            saved_view = await archive.get_view("opaque-view")
            assert saved_view is not None
            assert saved_view["view_id"] == "opaque-view"
        finally:
            await archive.close()


def test_context_preamble_ledger_remains_an_internal_owner(tmp_path: Path) -> None:
    """The internal preamble writer still has its explicit offline owner."""
    root = tmp_path / "archive"
    root.mkdir()
    config = Config(archive_root=root, render_root=tmp_path / "render", sources=[])
    _record_preamble_ledger(SimpleNamespace(config=config), _assembly())
    assert (root / "ops.db").exists()


@pytest.mark.asyncio
async def test_facade_cancelled_after_submission_reports_recoverable_indeterminate_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling the Python await never turns an in-flight write into absence."""
    import asyncio
    import threading

    from polylogue.daemon_client import DaemonClient
    from polylogue.operations.daemon_errors import DaemonMutationIndeterminateError

    root = tmp_path / "archive"
    with ArchiveStore(root):
        pass
    started, release = threading.Event(), threading.Event()

    def pending_operation(
        _client: DaemonClient,
        _operation: str,
        _payload: dict[str, object],
        *,
        archive_root: str,
        request_id: str,
    ) -> None:
        assert archive_root == str(root)
        assert request_id
        started.set()
        release.wait(timeout=5)
        return None

    monkeypatch.setattr(DaemonClient, "operation_to_completion", pending_operation)
    archive = Polylogue(archive_root=root)
    try:
        task = asyncio.create_task(
            archive.capture_assertion_candidate(body_text="candidate", kind=AssertionKind.LESSON)
        )
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(DaemonMutationIndeterminateError) as caught:
            await task
        assert caught.value.request_id
    finally:
        release.set()
        await archive.close()
