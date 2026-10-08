"""Sorted-once export custody on the production daemon read route."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.operations.assertion_export import AssertionExportImages
from polylogue.operations.daemon_reads import DaemonReadDependencies, execute_read_operation
from polylogue.operations.mutation_transaction import AuthorizationMismatchError, MutationPrincipal
from polylogue.operations.operation_context import open_operation_read
from tests.infra.daemon_operations import running_daemon_operations
from tests.infra.storage_records import SessionBuilder


def _seed(root: Path) -> None:
    SessionBuilder(root / "index.db", "neutral-export-image").provider("codex").save()
    with sqlite3.connect(root / "user.db") as user:
        user.executemany(
            "INSERT INTO assertions(assertion_id,target_ref,key,kind,value_json,created_at_ms,updated_at_ms) "
            "VALUES (?,?,?,?,?,?,?)",
            [(f"neutral-{i:04d}", "session:neutral", "neutral", "tag", "{}", 514 - i, 514 - i) for i in range(513)],
        )


def test_export_sorts_once_preserves_date_prefix_and_authenticates_owned_counts(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    _seed(root)
    owner = AssertionExportImages()
    principal = MutationPrincipal("neutral", frozenset(), "cli")
    dependencies = DaemonReadDependencies(assertion_exports=owner, assertion_export_principal=principal)
    sql: list[str] = []
    try:
        with open_operation_read(root) as pinned:
            pinned.archive._conn.set_trace_callback(sql.append)
            first = execute_read_operation(
                "user.assertions.export",
                {"limit": 501, "page_size": 17},
                archive=pinned.archive,
                serving_identity="test",
                dependencies=dependencies,
            )
            ref = first["selection_ref"]
            page = first
            items = page["items"]
            assert isinstance(items, list)
            ids = [row["assertion_id"] for row in items]
            while page["next_offset"] is not None:
                page = execute_read_operation(
                    "user.assertions.export",
                    {"limit": 501, "page_size": 17, "offset": page["next_offset"], "selection_ref": ref},
                    archive=pinned.archive,
                    serving_identity="test",
                    dependencies=dependencies,
                )
                items = page["items"]
                assert isinstance(items, list)
                ids.extend(row["assertion_id"] for row in items)
            assert ids == [f"neutral-{i:04d}" for i in range(512, 11, -1)]
            assert page["total"] == 501
            # Delivery retries can replay the final page until explicit release.
            replay = execute_read_operation(
                "user.assertions.export",
                {"limit": 501, "page_size": 17, "offset": 493, "selection_ref": ref},
                archive=pinned.archive,
                serving_identity="test",
                dependencies=dependencies,
            )
            assert replay["items"] == page["items"]
            with pytest.raises(AuthorizationMismatchError):
                execute_read_operation(
                    "user.assertions.export",
                    {"limit": 501, "selection_ref": ref},
                    archive=pinned.archive,
                    serving_identity="test",
                    dependencies=replace(
                        dependencies, assertion_export_principal=replace(principal, actor_ref="other")
                    ),
                )
        selection_queries = [s for s in sql if "FROM user_tier.assertions" in s and "ORDER BY" in s]
        assert any("PRAGMA temp_store = FILE" in s for s in sql)
        assert len(selection_queries) == 1
        assert "OFFSET" not in selection_queries[0]
        assert not any("COUNT(" in s for s in sql)
        assert owner.release(str(ref), principal)
        assert not owner.release(str(ref), principal)
    finally:
        owner.close()


@pytest.mark.parametrize("change_user", [False, True])
def test_daemon_export_user_change_refuses_but_unrelated_index_ingestion_continues(
    tmp_path: Path, change_user: bool
) -> None:
    root = tmp_path / "archive"
    with running_daemon_operations(root, seed_archive=_seed) as stack:
        first = stack.client.operation("user.assertions.export", {"page_size": 17}, archive_root=str(root))
        assert first is not None
        page = first["result"]
        if change_user:
            with sqlite3.connect(root / "user.db") as user:
                user.execute("DELETE FROM assertions WHERE assertion_id = 'neutral-0000'")
        else:
            SessionBuilder(root / "index.db", "unrelated-arrival").provider("codex").save()
        payload = {"selection_ref": page["selection_ref"], "offset": page["next_offset"], "page_size": 17}
        if change_user:
            refused = stack.client.operation("user.assertions.export", payload, archive_root=str(root))
            assert refused is not None
            assert refused["outcome"] == "rejected"
            assert refused["error"]["code"] == "query_continuation_stale"
        else:
            second = stack.client.operation("user.assertions.export", payload, archive_root=str(root))
            assert second is not None
            assert second["result"]["total"] == 513
            assert second["result"]["snapshot_epoch"] == page["snapshot_epoch"]
        # Release is independent of original User availability, and auth still applies.
        (root / "user.db").rename(root / "unavailable-user.db")
        released = stack.client.operation(
            "user.assertions.export.release", {"selection_ref": page["selection_ref"]}, archive_root=str(root)
        )
        assert released is not None
        assert isinstance(released["result"]["released"], bool)


def test_export_cancellation_retires_partial_image_and_closes_source_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    root = tmp_path / "archive"
    root.mkdir()
    _seed(root)
    original_directory = tempfile.TemporaryDirectory
    scratch: list[Path] = []

    def tracked_directory(*, prefix: str) -> tempfile.TemporaryDirectory[str]:
        directory = original_directory(prefix=prefix, dir=tmp_path)
        scratch.append(Path(directory.name))
        return directory

    monkeypatch.setattr("polylogue.operations.assertion_export.tempfile.TemporaryDirectory", tracked_directory)
    owner = AssertionExportImages()
    principal = MutationPrincipal("neutral", frozenset(), "cli")
    checkpoints = 0

    def cancel_after_progress() -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 23:
            raise InterruptedError("neutral cancellation")

    dependencies = DaemonReadDependencies(
        assertion_exports=owner, assertion_export_principal=principal, raise_if_aborted=cancel_after_progress
    )
    try:
        with open_operation_read(root) as pinned:
            with pytest.raises(InterruptedError):
                execute_read_operation(
                    "user.assertions.export",
                    {},
                    archive=pinned.archive,
                    serving_identity="test",
                    dependencies=dependencies,
                )
            assert pinned.archive._conn.execute("PRAGMA query_only").fetchone()[0] == 1
            assert pinned.archive._conn.execute("SELECT COUNT(*) FROM user_tier.assertions").fetchone()[0] == 513
        assert len(scratch) == 1
        assert not scratch[0].exists()
    finally:
        owner.close()


@pytest.mark.uses_real_clock("starts the real UDS listener and coordinator loop")
def test_daemon_abandoned_equivalent_starts_share_bytes_and_release_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    original_directory = tempfile.TemporaryDirectory
    scratch: list[Path] = []

    def tracked_directory(*args: Any, **kwargs: Any) -> tempfile.TemporaryDirectory[str]:
        directory = original_directory(*args, **kwargs)
        if kwargs.get("prefix") == "polylogue-assertion-export-":
            scratch.append(Path(directory.name))
        return directory

    monkeypatch.setattr("polylogue.operations.assertion_export.tempfile.TemporaryDirectory", tracked_directory)
    root = tmp_path / "archive"
    with running_daemon_operations(root, seed_archive=_seed) as stack:
        # Each completed first-page HTTP exchange disconnects, with no release
        # or continuation. Restoring a fresh image per start makes this red.
        for _ in range(12):
            abandoned = stack.client.operation("user.assertions.export", {"page_size": 1}, archive_root=str(root))
            assert abandoned is not None and abandoned["result"]["total"] == 513

        def start() -> dict[str, object]:
            response = stack.client.operation("user.assertions.export", {"page_size": 1}, archive_root=str(root))
            assert response is not None
            return cast(dict[str, object], response["result"])

        with ThreadPoolExecutor(max_workers=2) as callers:
            left, right = list(callers.map(lambda _: start(), range(2)))
        assert left["selection_ref"] != right["selection_ref"]
        assert len(scratch) == 1 and (scratch[0] / "rows.db").is_file()
        released = stack.client.operation(
            "user.assertions.export.release", {"selection_ref": left["selection_ref"]}, archive_root=str(root)
        )
        assert released is not None and released["result"]["released"] is True
        final_payload = {"selection_ref": right["selection_ref"], "offset": 512, "page_size": 1}
        final = stack.client.operation("user.assertions.export", final_payload, archive_root=str(root))
        replay = stack.client.operation("user.assertions.export", final_payload, archive_root=str(root))
        assert final is not None and replay is not None
        assert final["result"] == replay["result"]
        assert final["result"]["next_offset"] is None
        assert final["result"]["items"][0]["assertion_id"] == "neutral-0000"
    assert not scratch[0].exists()


def test_new_assertion_frame_retires_abandoned_bytes_without_recreating_an_older_pinned_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    from polylogue.archive.query.transaction import QueryContinuationStaleError

    root = tmp_path / "archive"
    root.mkdir()
    _seed(root)
    original_directory = tempfile.TemporaryDirectory
    scratch: list[Path] = []

    def tracked_directory(*, prefix: str) -> tempfile.TemporaryDirectory[str]:
        directory = original_directory(prefix=prefix, dir=tmp_path)
        scratch.append(Path(directory.name))
        return directory

    monkeypatch.setattr("polylogue.operations.assertion_export.tempfile.TemporaryDirectory", tracked_directory)
    owner = AssertionExportImages()
    principal = MutationPrincipal("neutral", frozenset(), "cli")
    dependencies = DaemonReadDependencies(assertion_exports=owner, assertion_export_principal=principal)
    try:
        with open_operation_read(root) as older:
            first = execute_read_operation(
                "user.assertions.export",
                {"page_size": 1},
                archive=older.archive,
                serving_identity="test",
                dependencies=dependencies,
            )
            with sqlite3.connect(root / "user.db") as user:
                user.execute("DELETE FROM assertions WHERE assertion_id = 'neutral-0000'")
            with open_operation_read(root) as newer:
                current = execute_read_operation(
                    "user.assertions.export",
                    {"page_size": 1},
                    archive=newer.archive,
                    serving_identity="test",
                    dependencies=dependencies,
                )
                assert current["total"] == 512
                assert len(scratch) == 2 and not scratch[0].exists() and scratch[1].exists()
                with pytest.raises(QueryContinuationStaleError):
                    execute_read_operation(
                        "user.assertions.export",
                        {"selection_ref": first["selection_ref"], "offset": 1},
                        archive=older.archive,
                        serving_identity="test",
                        dependencies=dependencies,
                    )
                assert scratch[1].exists()
                assert owner.release(str(first["selection_ref"]), principal)
                assert owner.release(str(current["selection_ref"]), principal)
                assert not scratch[1].exists()
    finally:
        owner.close()
