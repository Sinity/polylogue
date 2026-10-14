"""Complete CLI exports consume the resident message and metadata page owners."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from click.testing import CliRunner

from polylogue.cli import archive_query, query_set_read, read_dispatch
from polylogue.cli.operation_kernel import OperationRequest
from polylogue.cli.read_dispatch import ServedBy
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.core.enums import Provider, Role
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.daemon_operations import cli_daemon_archive
from tests.infra.live_ingest import write_index_session


@pytest.mark.uses_real_clock("drives the production resident operation listener")
def test_complete_cli_export_drains_the_actual_independent_metadata_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tags = tuple(f"tag-{index:03d}" for index in range(250))
    session_id = ""

    def seed(root: Path) -> None:
        nonlocal session_id
        with ArchiveStore(root) as writer:
            session_id = write_index_session(
                writer,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="complete-transcript-metadata",
                    working_directories=[f"/synthetic/dir-{index:03d}" for index in range(250)],
                    messages=[ParsedMessage(provider_message_id="one", role=Role.USER, text="neutral prose")],
                ),
            )
            writer.add_user_tags((session_id,), tags)

    calls: list[tuple[dict[str, object], int, int, int]] = []
    root_requests: list[dict[str, object]] = []
    original = read_dispatch.dispatch_read

    def read(
        config: Config,
        request: OperationRequest,
        *,
        daemon_disabled: bool = False,
        selection_epoch: str | None = None,
    ) -> tuple[dict[str, object], ServedBy]:
        result, authority = original(
            config,
            request,
            daemon_disabled=daemon_disabled,
            selection_epoch=selection_epoch,
        )
        assert authority.identity == "daemon"
        if request.operation == "session.read" and request.payload.get("session_projection") == "domain":
            body = result["session"]
            assert isinstance(body, dict)
            messages, directories, user_tags = body["messages"], body["working_directories"], body["tags_m2m"]
            assert isinstance(messages, list) and isinstance(directories, list) and isinstance(user_tags, list)
            calls.append((dict(request.payload), len(messages), len(directories), len(user_tags)))
        elif request.operation == "session.read":
            root_requests.append(dict(request.payload))
            metadata_page = result["metadata_page"]
            assert isinstance(metadata_page, dict) and metadata_page["user_tags_total"] is None
        return result, authority

    monkeypatch.setattr(read_dispatch, "dispatch_read", read)
    monkeypatch.setattr(archive_query, "dispatch_read", read)
    monkeypatch.setattr(read_dispatch, "daemon_route_disabled", lambda **_kwargs: False)
    with cli_daemon_archive(tmp_path / "archive", monkeypatch, seed_archive=seed) as stack:
        config = Config(
            archive_root=stack.archive_root,
            render_root=tmp_path / "render",
            sources=[],
            db_path=stack.archive_root / "index.db",
            api_allow_no_auth=True,
        )
        env = cast(AppEnv, SimpleNamespace(config=config))
        runner = CliRunner()
        with runner.isolation() as (out, _err, _term):
            selection = query_set_read.run_query_set_read(
                env,
                RootModeRequest.from_params({}),
                output_format="json",
                fields=None,
            )
            document = json.loads(out.getvalue())[0]
        assert [row.session_id for row in selection.rows] == [session_id]
        with runner.isolation() as (out, _err, _term), ExitStack() as lifetime:
            root_read = archive_query._read_session_windows(
                config,
                session_id,
                daemon_disabled=False,
                lifetime=lifetime,
            )
            archive_query._emit_session_result(root_read, output_format="json", fields=None)
            root_document = json.loads(out.getvalue())
    assert document["id"] == session_id
    assert document["message_count"] == 1 and len(document["messages"]) == 1
    assert document["messages"][0]["text"] == "neutral prose"
    assert document["tags"] == list(tags)
    assert document["cwd_display"] == "dir-000"
    assert [call[0]["limit"] for call in calls] == [100, 0, 0]
    assert [call[0]["offset"] for call in calls] == [0, 1, 1]
    assert [call[1] for call in calls] == [1, 0, 0]
    assert [call[2] for call in calls] == [100, 100, 50]
    assert [call[3] for call in calls] == [100, 100, 50]
    assert all(call[0]["metadata_limit"] == 100 for call in calls)

    assert len(root_document["messages"]) == 1
    assert "tags" not in root_document and "user_tags" not in root_document
    assert "working_directories" not in root_document
    assert len(root_requests) == 1 and root_requests[0]["include_user_tags"] is False
