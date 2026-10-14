"""Independent metadata windows retain complete directories and durable User tags."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.archive.query.transaction import QueryContinuationStaleError
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop
from tests.infra.live_ingest import write_index_session


def _seed(root: Path) -> str:
    bootstrap_archive_root(root)
    with ArchiveStore(root) as writer:
        session_id = write_index_session(
            writer,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="metadata-windows",
                working_directories=[f"/synthetic/dir-{index}" for index in range(5)],
                messages=[ParsedMessage(provider_message_id="one", role=Role.USER, text="neutral prose")],
            ),
        )
        writer.add_user_tags((session_id,), tuple(f"tag-{index}" for index in range(7)))
    return session_id


@pytest.mark.asyncio
async def test_metadata_progress_is_bounded_and_independent_of_completed_messages(tmp_path: Path) -> None:
    session_id = run_off_event_loop(lambda: _seed(tmp_path))
    facade = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        page = await facade.get_session_page(session_id, limit=2, offset=1)
        assert page is not None and not page.session.messages
        assert len(page.session.working_directories) <= 2
        directories = list(page.session.working_directories)
        tags = list(page.session.tags_m2m)
        progress = page.metadata_page
        assert progress is not None
        assert progress.working_directories_total == 5
        assert progress.user_tags_total == 7
        while progress.continuation is not None:
            page = await facade.get_session_page(
                session_id, limit=2, offset=1, metadata_continuation=progress.continuation
            )
            assert page is not None and not page.session.messages
            assert len(page.session.working_directories) <= 2 and len(page.session.tags_m2m) <= 2
            assert page.metadata_page is not None and page.metadata_page.offset > progress.offset
            directories.extend(page.session.working_directories)
            tags.extend(page.session.tags_m2m)
            progress = page.metadata_page
        assert progress.complete and progress.next_offset is None
        assert directories == [f"/synthetic/dir-{index}" for index in range(5)]
        assert tags == [f"tag-{index}" for index in range(7)]
    finally:
        await facade.close()


@pytest.mark.asyncio
async def test_metadata_continuation_refuses_committed_user_tag_change(tmp_path: Path) -> None:
    session_id = run_off_event_loop(lambda: _seed(tmp_path))
    facade = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        first = await facade.get_session_page(session_id, limit=2)
        assert first is not None and first.metadata_page is not None
        token = first.metadata_page.continuation
        assert token is not None

        def add_tag() -> None:
            with ArchiveStore(tmp_path) as writer:
                writer.add_user_tags((session_id,), ("later",))

        run_off_event_loop(add_tag)
        with pytest.raises(QueryContinuationStaleError):
            await facade.get_session_page(session_id, limit=2, metadata_continuation=token)
    finally:
        await facade.close()


@pytest.mark.uses_real_clock("drives the production resident operation listener")
def test_resident_metadata_drain_has_independent_bounds_and_no_repeated_messages(tmp_path: Path) -> None:
    from tests.infra.daemon_operations import running_daemon_operations

    session_id = ""

    def seed(root: Path) -> None:
        nonlocal session_id
        session_id = _seed(root)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        first = stack.client.operation(
            "session.read",
            {"ref": session_id, "limit": 1, "metadata_limit": 2},
            archive_root=str(stack.archive_root),
        )
        assert first is not None and first["outcome"] == "completed"
        result = first["result"]
        assert len(result["session"]["messages"]) == 1 and result["complete"]
        assert result["metadata_page"]["limit"] == 2
        directories = list(result["session"]["working_directories"])
        tags = list(result["session"]["user_tags"])
        token = result["metadata_page"]["continuation"]
        while token is not None:
            follow = stack.client.operation(
                "session.read",
                {"ref": session_id, "limit": 0, "offset": 1, "metadata_limit": 2, "metadata_continuation": token},
                archive_root=str(stack.archive_root),
            )
            assert follow is not None and follow["outcome"] == "completed"
            page = follow["result"]
            assert page["session"]["messages"] == [] and page["total"] == 1
            assert page["complete"] and page["limit"] == 0 and page["continuation"] is None
            directories.extend(page["session"]["working_directories"])
            tags.extend(page["session"]["user_tags"])
            token = page["metadata_page"]["continuation"]
        assert directories == [f"/synthetic/dir-{index}" for index in range(5)]
        assert tags == [f"tag-{index}" for index in range(7)]


@pytest.mark.asyncio
async def test_http_detail_drains_metadata_without_full_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import Any, cast

    from polylogue.daemon.http import DaemonAPIHandler

    session_id = run_off_event_loop(lambda: _seed(tmp_path))
    facade = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    original = ArchiveStore.read_summary
    projections: list[bool] = []

    def summary(store: ArchiveStore, selected: str, *, include_metadata: bool = True) -> Any:
        projections.append(include_metadata)
        return original(store, selected, include_metadata=include_metadata)

    monkeypatch.setattr(ArchiveStore, "read_summary", summary)
    try:
        payload = await DaemonAPIHandler._do_get_session(cast(Any, None), facade, session_id, limit=1)
        assert isinstance(payload, dict)
        assert len(payload["messages"]) == 1
        assert payload["tags"] == [f"tag-{index}" for index in range(7)]
        assert payload["cwd_display"] == "/synthetic/dir-0"
        assert projections and not any(projections)
    finally:
        await facade.close()


@pytest.mark.parametrize("change", ["widen", "outside", "scope"])
@pytest.mark.asyncio
async def test_metadata_continuation_refuses_invalid_coordinates_or_scope(tmp_path: Path, change: str) -> None:
    from polylogue.archive.query.transaction import QueryContinuation, QueryContinuationInvalidError

    session_id = run_off_event_loop(lambda: _seed(tmp_path))
    facade = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        first = await facade.get_session_page(session_id, limit=2)
        assert first is not None and first.metadata_page is not None
        token = first.metadata_page.continuation
        assert token is not None
        metadata_limit = 3 if change == "widen" else None
        include_user_tags = change != "scope"
        if change == "outside":
            decoded = QueryContinuation.decode(token)
            token = QueryContinuation(decoded.request.next(offset=1000), decoded.result_ref).encode()
        with pytest.raises(QueryContinuationInvalidError):
            await facade.get_session_page(
                session_id,
                limit=1,
                metadata_continuation=token,
                metadata_limit=metadata_limit,
                include_user_tags=include_user_tags,
            )
    finally:
        await facade.close()


@pytest.mark.asyncio
async def test_unrequested_tags_are_not_measured_zero_and_message_narrowing_keeps_metadata_bound(
    tmp_path: Path,
) -> None:
    session_id = run_off_event_loop(lambda: _seed(tmp_path))
    facade = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        first = await facade.get_session_page(session_id, limit=2, include_user_tags=False)
        assert first is not None and first.metadata_page is not None
        assert first.metadata_page.user_tags_total is None and first.session.tags_m2m == ()
        token = first.metadata_page.continuation
        assert token is not None
        follow = await facade.get_session_page(
            session_id, limit=1, metadata_continuation=token, include_user_tags=False
        )
        assert follow is not None and follow.metadata_page is not None
        assert follow.metadata_page.limit == 2 and len(follow.session.working_directories) == 2
        assert follow.metadata_page.user_tags_total is None
    finally:
        await facade.close()


@pytest.mark.asyncio
async def test_http_explicit_user_metadata_does_not_fall_back_to_provider_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace
    from typing import Any, cast

    from polylogue.api.archive import SessionTranscriptPage
    from polylogue.daemon.http import DaemonAPIHandler

    session_id = run_off_event_loop(lambda: _seed(tmp_path))

    def remove_tags() -> None:
        with ArchiveStore(tmp_path) as writer:
            writer.remove_user_tags((session_id,), tuple(f"tag-{index}" for index in range(7)))

    run_off_event_loop(remove_tags)
    facade = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    original = facade.get_session_page

    async def with_provider_header(*args: Any, **kwargs: Any) -> SessionTranscriptPage | None:
        page = await original(*args, **kwargs)
        if page is None:
            return None
        assert page.metadata_page is not None and page.metadata_page.user_tags_total == 0
        return replace(page, session=page.session.model_copy(update={"metadata": {"tags": ["provider-only"]}}))

    monkeypatch.setattr(facade, "get_session_page", with_provider_header)
    try:
        payload = await DaemonAPIHandler._do_get_session(cast(Any, None), facade, session_id, limit=1)
        assert isinstance(payload, dict)
        assert payload["tags"] == []
        assert payload["cwd_display"] == "/synthetic/dir-0"
    finally:
        await facade.close()
