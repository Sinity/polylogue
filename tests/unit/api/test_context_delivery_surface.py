"""Surface wiring for the durable context-delivery ledger (polylogue-37t.22).

``polylogue/storage/sqlite/archive_tiers/context_delivery_write.py`` (PR
#2703) persists exact context-delivery receipts, but until this module
nothing outside tests ever called ``write_context_delivery`` -- the read-only
``get_context_delivery`` facade method could resolve a receipt, but no
surface ever recorded one. These tests exercise the write-capable facade
methods (``compile_and_record_context`` / ``record_context_delivery`` /
``list_context_deliveries``) that close that gap, proving compilation,
recording, idempotent replay, drift refusal, and bounded listing all work
end-to-end against a real archive.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast

import pytest

from polylogue import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.context_delivery_write import ArchiveContextDeliveryEnvelope
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.frozen_clock import FrozenClock
from tests.infra.live_ingest import write_index_session


@pytest.fixture
def facade_daemon_writer(monkeypatch: pytest.MonkeyPatch) -> Callable[[Path], AbstractContextManager[object]]:
    """Run context writes through the production daemon operation route."""
    from polylogue.daemon.socket_path import daemon_socket_path
    from tests.infra.daemon_operations import running_daemon_operations

    monkeypatch.setattr("polylogue.daemon.api_auth.resolve_api_auth_token", lambda *_args, **_kwargs: None)

    def start(archive_root: Path) -> AbstractContextManager[object]:
        return running_daemon_operations(archive_root, socket_path=daemon_socket_path(archive_root))

    return start


def _seed_on_writer(archive_root: Path, *, provider_session_id: str, text: str) -> None:
    with ArchiveStore(archive_root) as archive:
        write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=provider_session_id,
                title="Delivery target",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                    )
                ],
            ),
        )


def _seed(archive_root: Path, *, provider_session_id: str, text: str) -> None:
    """Run the synchronous seed off any running event loop."""
    return run_off_event_loop(lambda: _seed_on_writer(archive_root, provider_session_id=provider_session_id, text=text))


async def test_compile_and_record_context_persists_the_exact_compiled_image(
    tmp_path: Path, facade_daemon_writer: Callable[[Path], AbstractContextManager[object]]
) -> None:
    """The delivery boundary records exactly the image compile_context produced."""

    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="delivery-target", text="quoted archival evidence")

    with facade_daemon_writer(archive_root):
        async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
            envelope = await poly.compile_and_record_context(
                recipient_ref="agent:codex-main",
                delivered_by_ref="user:local",
                boundary="explicit-recall",
                query="quoted archival",
                max_sessions=1,
            )

            assert isinstance(envelope, ArchiveContextDeliveryEnvelope)
            assert envelope.outcome == "recorded"
            assert envelope.recipient_ref == "agent:codex-main"
            assert envelope.delivered_by_ref == "user:local"
            assert envelope.boundary == "explicit-recall"
            message_segments = [s for s in envelope.context_image.segments if s.payload_kind == "messages"]
            assert message_segments, "expected a messages segment for the delivered image"
            assert "quoted archival evidence" in (message_segments[0].markdown or "")

            # Fetching the receipt back returns exactly the delivered image.
            fetched = await poly.get_context_delivery(envelope.snapshot_ref, recipient_ref="agent:codex-main")
            assert fetched is not None
            assert fetched.context_image_sha256 == envelope.context_image_sha256

            # A wrong recipient never sees the receipt.
            wrong_recipient = await poly.get_context_delivery(envelope.snapshot_ref, recipient_ref="agent:other")
            assert wrong_recipient is None


async def test_compile_and_record_context_replay_is_idempotent_and_drift_is_rejected(
    tmp_path: Path, facade_daemon_writer: Callable[[Path], AbstractContextManager[object]]
) -> None:
    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="delivery-target", text="quoted archival evidence")

    with facade_daemon_writer(archive_root):
        async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
            first = await poly.compile_and_record_context(
                recipient_ref="agent:codex-main",
                delivered_by_ref="user:local",
                boundary="explicit-recall",
                query="quoted archival",
                max_sessions=1,
            )
            replay = await poly.compile_and_record_context(
                recipient_ref="agent:codex-main",
                delivered_by_ref="user:local",
                boundary="explicit-recall",
                query="quoted archival",
                max_sessions=1,
            )
            assert replay.outcome == "idempotent"
            assert replay.snapshot_ref == first.snapshot_ref
            assert replay.context_image_sha256 == first.context_image_sha256

            listed = await poly.list_context_deliveries(recipient_ref="agent:codex-main")
            assert [item.snapshot_ref for item in listed.items] == [first.snapshot_ref]

            # Same snapshot ref, different recipient: identity drift is rejected.
            from polylogue.operations.daemon_errors import DaemonOperationRejectedError

            with pytest.raises(DaemonOperationRejectedError, match="different delivery identity"):
                await poly.compile_and_record_context(
                    recipient_ref="agent:someone-else",
                    delivered_by_ref="user:local",
                    boundary="explicit-recall",
                    query="quoted archival",
                    max_sessions=1,
                )


async def test_compile_and_record_context_refuses_assertion_read_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A durable assertion read failure cannot produce a clean incomplete receipt.

    Anti-vacuity: changing the adapter to return ``[]`` or letting the raw
    ``sqlite3.Error`` escape makes the typed refusal assertion below fail.
    """

    archive_root = tmp_path / "archive-assertion-read-failure"
    _seed(archive_root, provider_session_id="delivery-target", text="quoted archival evidence")

    with sqlite3.connect(archive_root / "user.db") as conn:
        from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion

        upsert_assertion(
            conn,
            assertion_id="must-include-decision",
            target_ref="session:codex-session:delivery-target",
            kind=AssertionKind.DECISION,
            body_text="This decision must be present in delivered context.",
            author_kind="user",
            status="active",
            visibility="private",
            context_policy={"inject": True},
            now_ms=1_700_000_000_000,
        )

    import polylogue.storage.sqlite.archive_tiers.user_write as user_write

    # The facade reads claims through the controlled archive transaction and
    # imports the reader from its module at call time, so fail it there.
    def fail_assertion_read(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("injected durable user.db read failure")

    monkeypatch.setattr(user_write, "list_assertion_claims", fail_assertion_read)

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        with pytest.raises(ArchiveTierUnavailableError, match="injected durable user.db read failure"):
            await poly.compile_and_record_context(
                recipient_ref="agent:codex-main",
                delivered_by_ref="user:local",
                boundary="explicit-recall",
                query="quoted archival",
                max_sessions=1,
            )

        monkeypatch.undo()
        assert (await poly.list_context_deliveries(recipient_ref="agent:codex-main")).items == ()


async def test_list_context_deliveries_never_includes_full_context_image(
    tmp_path: Path, facade_daemon_writer: Callable[[Path], AbstractContextManager[object]]
) -> None:
    """The bounded list surface is a summary read, not a disclosure surface."""

    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="delivery-target", text="quoted archival evidence")

    with facade_daemon_writer(archive_root):
        async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
            recorded = await poly.compile_and_record_context(
                recipient_ref="agent:codex-main",
                delivered_by_ref="user:local",
                boundary="explicit-recall",
                query="quoted archival",
                max_sessions=1,
            )
            listed = await poly.list_context_deliveries(recipient_ref="agent:codex-main")
            assert len(listed.items) == listed.total == 1
            # Summary metadata identifies the same durable receipt without
            # decoding or disclosing its delivered context image.
            assert not hasattr(listed.items[0], "context_image")
            assert listed.items[0].snapshot_ref == recorded.snapshot_ref
            assert listed.items[0].context_image_sha256 == recorded.context_image_sha256

            unrelated = await poly.list_context_deliveries(recipient_ref="agent:unrelated")
            assert unrelated.items == () and unrelated.total == 0


async def test_record_context_delivery_requires_initialized_user_tier(tmp_path: Path) -> None:
    """A missing user.db fails closed with a typed error, not a silent no-op write."""

    archive_root = tmp_path / "archive-missing-user-tier"
    archive_root.mkdir()
    # Initialize the source and index tiers only -- user.db (the durable
    # receipt ledger) is deliberately never created, mirroring an archive
    # that predates the fs1.11 migration or has not been re-initialized yet.
    with sqlite3.connect(archive_root / "source.db") as source_conn:
        initialize_archive_tier(source_conn, ArchiveTier.SOURCE)
    with sqlite3.connect(archive_root / "index.db") as index_conn:
        initialize_archive_tier(index_conn, ArchiveTier.INDEX)

    from polylogue.archive.context_models import ContextImage
    from polylogue.config import Config
    from polylogue.operations.facade_writers import _archive_record_context_delivery

    with pytest.raises(ValueError, match="context-delivery user tier is not initialized"):
        _archive_record_context_delivery(
            Config(archive_root=archive_root, render_root=archive_root, sources=[]),
            image=cast(ContextImage, object()),  # The guard must run before dereferencing the image.
            boundary="explicit-recall",
            recipient_ref="agent:codex-main",
            delivered_by_ref="user:local",
            run_ref=None,
            inheritance_mode="explicit",
        )


@pytest.mark.frozen_clock_modules("polylogue.api.archive")
async def test_context_scheduler_ledger_has_a_facade_reader(
    tmp_path: Path,
    facade_daemon_writer: Callable[[Path], AbstractContextManager[object]],
    frozen_clock: FrozenClock,
) -> None:
    archive_root = tmp_path / "archive-ledger-reader"
    _seed(archive_root, provider_session_id="ledger-target", text="scheduler evidence")

    with facade_daemon_writer(archive_root):
        async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
            from polylogue.archive.context_models import ContextSpec

            await poly.compile_context(ContextSpec(seed_refs=("session:codex-session:ledger-target",), max_tokens=100))
            records = await poly.list_context_injection_ledger(target_session="codex-session:ledger-target")

    assert records
    assert all(record.observed_at_ms == int(frozen_clock.now().timestamp() * 1000) for record in records)
    assert records[0].row.source == "archive-context"
    assert records[0].row.execution_context_ref.startswith("sha256:")


async def test_context_delivery_decode_failure_refuses_instead_of_absence(
    tmp_path: Path, facade_daemon_writer: Callable[[Path], AbstractContextManager[object]]
) -> None:
    root = tmp_path / "archive"
    _seed(root, provider_session_id="neutral", text="neutral delivery")
    with facade_daemon_writer(root):
        async with Polylogue(archive_root=root, db_path=root / "index.db") as poly:
            receipt = await poly.compile_and_record_context(
                recipient_ref="agent:neutral",
                delivered_by_ref="user:local",
                boundary="explicit-recall",
                query="neutral",
                max_sessions=1,
            )
    with sqlite3.connect(root / "user.db") as conn:
        conn.execute("UPDATE context_deliveries SET metadata_json = '[]'")
    async with Polylogue(archive_root=root, db_path=root / "index.db") as poly:
        with pytest.raises(ArchiveTierUnavailableError):
            await poly.get_context_delivery(receipt.snapshot_ref, recipient_ref="agent:neutral")
        page = await poly.list_context_deliveries(recipient_ref="agent:neutral")
        assert tuple(item.snapshot_ref for item in page.items) == (receipt.snapshot_ref,)
    with sqlite3.connect(root / "user.db") as conn:
        conn.execute("UPDATE context_deliveries SET segment_refs_json = '{}'")
    async with Polylogue(archive_root=root, db_path=root / "index.db") as poly:
        with pytest.raises(ArchiveTierUnavailableError):
            await poly.list_context_deliveries(recipient_ref="agent:neutral")
