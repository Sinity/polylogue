"""The compact projection is reachable through its declared read operation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.operations.daemon_protocol import daemon_operation_spec, validate_operation_result
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from polylogue.surfaces.compaction import CorpusCompactionPack, compact_sessions
from tests.infra.storage_records import SessionBuilder


def test_read_compact_builds_a_pack_from_the_pinned_archive(tmp_path: Path) -> None:
    """``read.compact`` executes ``compact_sessions`` over archived sessions.

    Anti-vacuity: without the ``read.compact`` branch in
    ``daemon_reads.execute_read_operation`` the call raises "read operation is
    not declared"; without the all-block canonical outcome check in
    ``compaction._successful_tool_result`` the archived successful tool result is kept as an item instead of omitted as ``successful_tool_spam``.
    """

    root = tmp_path / "archive"
    root.mkdir()
    (
        SessionBuilder(root / "index.db", "compact-route")
        .provider("codex")
        .title("Compact route")
        .add_message("ask", role="user", text="Please verify the failing parser", material_origin="human_authored")
        .add_message(
            "ok",
            role="tool",
            text="",
            material_origin="tool_result",
            blocks=[
                {
                    "type": "tool_result",
                    "tool_id": "call-1",
                    "text": "all green",
                    "tool_result_is_error": 0,
                    "tool_result_exit_code": 0,
                }
            ],
        )
        .add_message("reply", role="assistant", text="Fixed the parser", material_origin="assistant_authored")
        .save()
    )
    with open_operation_read(root) as pinned:
        archive = pinned.archive
        session_id = archive.list_summaries(limit=1)[0].session_id
        result = execute_read_operation(
            "read.compact",
            {"session_id": session_id, "params": {}, "projection": {"max_tokens": 4000}},
            archive=archive,
            serving_identity="test",
        )

    validate_operation_result("read.compact", result)
    assert result["view"] == "compact"
    pack = CorpusCompactionPack.model_validate(cast(dict[str, Any], result["payload"]))
    assert pack.projection.max_tokens == 4000
    assert {item.text for item in pack.items} == {"Please verify the failing parser", "Fixed the parser"}
    assert {item.session_id for item in pack.items} == {session_id}
    assert pack.manifest.drop_counts == {"successful_tool_spam": 1}
    declaration = daemon_operation_spec("read.compact")
    assert declaration is not None and declaration.fallback.value == "never"


def test_compact_keeps_a_large_permitted_projection(monkeypatch: Any) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from polylogue.operations import read_view_compact

    value = "λ" * (9 * 1024 * 1024)
    monkeypatch.setattr(read_view_compact, "select_read_view_summaries", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        read_view_compact,
        "compact_sessions",
        lambda *args, **kwargs: SimpleNamespace(model_dump=lambda mode: {"large": value}),
    )
    result = read_view_compact.execute_compact_read({"params": {}}, archive=Mock())
    assert result["payload"] == {"large": value}


def test_pinned_compaction_route_preserves_collapsed_evidence_and_token_accounting(tmp_path: Path) -> None:
    from polylogue.archive.message.token_measure import estimate_tokens

    root = tmp_path / "archive"
    root.mkdir()
    builder = SessionBuilder(root / "index.db", "compact-ladder").provider("codex")
    text = "decision " * 1000
    for index in range(20):
        builder.add_message(f"message-{index}", role="assistant", text=text, material_origin="assistant_authored")
    builder.save()
    with open_operation_read(root) as pinned:
        archive = pinned.archive
        session_id = archive.list_summaries(limit=1)[0].session_id
        result = execute_read_operation(
            "read.compact",
            {"session_id": session_id, "params": {}, "projection": {"max_tokens": 2500}},
            archive=archive,
            serving_identity="test",
        )
    validate_operation_result("read.compact", result)
    pack = CorpusCompactionPack.model_validate(cast(dict[str, Any], result["payload"]))
    assert len(pack.items) == 1
    assert pack.items[0].occurrence_count == 20
    assert len(pack.items[0].refs) == 20
    assert pack.manifest.drop_counts["budget_collapsed"] == 19
    assert pack.manifest.included_tokens_by_session[session_id] + pack.manifest.dropped_tokens_by_session[
        session_id
    ] == (20 * estimate_tokens(text))
    assert pack.outcome.state == "degraded" and pack.token_estimate <= 2500


def _compact_payload(root: Path, session_id: str, *, params: dict[str, object] | None = None) -> CorpusCompactionPack:
    with open_operation_read(root) as pinned:
        result = execute_read_operation(
            "read.compact",
            {"session_id": session_id, "params": params or {}, "projection": {"max_tokens": 4000}},
            archive=pinned.archive,
            serving_identity="test",
        )
    validate_operation_result("read.compact", result)
    return CorpusCompactionPack.model_validate(result["payload"])


@pytest.mark.parametrize("other_outcome", ["unknown", "error", "ok"])
@pytest.mark.parametrize("reverse", [False, True])
def test_compact_retains_a_result_with_any_unproven_or_failed_block(
    tmp_path: Path, other_outcome: str, reverse: bool
) -> None:
    """Suppressing a message from its first successful block loses independent evidence."""
    root = tmp_path / "archive"
    blocks: list[dict[str, object]] = [
        {
            "type": "tool_result",
            "tool_id": "first",
            "text": "green",
            "tool_outcome": "ok",
            "tool_result_is_error": False,
        },
        {"type": "tool_result", "tool_id": "second", "text": "other evidence", "tool_outcome": other_outcome},
    ]
    if other_outcome == "unknown":
        blocks[1]["tool_result_outcome_unknown_reason"] = "not_reported"
    if other_outcome != "unknown":
        blocks[1]["tool_result_is_error"] = other_outcome == "error"
    if reverse:
        blocks.reverse()
    builder = SessionBuilder(root / "index.db", "mixed-outcomes").provider("codex")
    builder.add_message(
        "results", role="tool", text="Independent tool evidence", material_origin="tool_result", blocks=blocks
    )
    builder.save()
    pack = _compact_payload(root, builder.native_session_id())
    expected_text = "other evidence\n\ngreen" if reverse else "green\n\nother evidence"
    assert (expected_text in {item.text for item in pack.items}) == (other_outcome != "ok")
    assert pack.manifest.drop_counts.get("successful_tool_spam", 0) == (1 if other_outcome == "ok" else 0)


def test_compact_child_only_selection_retains_the_parent_branch_point(tmp_path: Path) -> None:
    """A lineage link alone cannot prove that another selected item emitted the prefix."""
    root = tmp_path / "archive"
    parent = SessionBuilder(root / "index.db", "compact-parent").provider("claude-code")
    parent.add_message("shared", role="user", text="Shared authored prefix", material_origin="human_authored").save()
    child = (
        SessionBuilder(root / "index.db", "compact-child")
        .provider("claude-code")
        .parent_session("ext-compact-parent")
        .branch_type("fork")
    )
    child.add_message("shared", role="user", text="Shared authored prefix", material_origin="human_authored")
    child.add_message("tail", role="assistant", text="Child decision", material_origin="assistant_authored").save()
    with open_operation_read(root) as pinned:
        envelope = pinned.archive.read_session_page(child.native_session_id(), limit=100, offset=0)
        assert envelope.lineage_inheritance == "prefix-sharing"
        assert envelope.lineage_branch_point_message_id is not None
    pack = _compact_payload(root, child.native_session_id(), params={"root": False})
    assert {item.text for item in pack.items} == {"Shared authored prefix", "Child decision"}
    assert {item.session_id for item in pack.items} == {child.native_session_id()}
    assert pack.manifest.duplicate_prefix_omissions == 0


@pytest.fixture(scope="module")
def compaction_page_archive(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    root = tmp_path_factory.mktemp("compact-pages")
    builder = SessionBuilder(root / "index.db", "compact-pages").provider("codex")
    for position in range(257):
        builder.add_message(
            str(position), role="user", text=f"Neutral decision {position}", material_origin="human_authored"
        )
    builder.save()
    return root, builder.native_session_id()


@pytest.mark.parametrize("failure", ["none", "late-page", "cancel", "late-frame"])
def test_compact_read_stages_pages_and_closes_before_return_or_failure(
    monkeypatch: pytest.MonkeyPatch, compaction_page_archive: tuple[Path, str], failure: str
) -> None:
    import sqlite3

    from polylogue.archive.message.compaction_store import CompactionStore
    from polylogue.archive.query.execution_control import QueryCancelledError
    from polylogue.operations import read_view_compact
    from polylogue.operations.daemon_reads import DaemonReadDependencies

    root, session_id = compaction_page_archive
    stores: list[CompactionStore] = []
    paths: list[Path] = []
    offsets: list[int] = []
    initialize = CompactionStore.__init__

    def observed(stage: CompactionStore, connection: sqlite3.Connection, checkpoint: Any) -> None:
        initialize(stage, connection, checkpoint)
        stores.append(stage)
        cursor = connection.execute("PRAGMA database_list")
        try:
            paths.append(Path(cursor.fetchone()[2]))
        finally:
            cursor.close()

    monkeypatch.setattr(CompactionStore, "__init__", observed)
    cancelled = False
    compacted = False
    compact = compact_sessions

    def observed_compact(*args: Any, **kwargs: Any) -> CorpusCompactionPack:
        nonlocal compacted
        result = compact(*args, **kwargs)
        compacted = True
        return result

    monkeypatch.setattr(read_view_compact, "compact_sessions", observed_compact)

    def checkpoint() -> None:
        if cancelled:
            raise QueryCancelledError("synthetic compact cancellation")

    with open_operation_read(root) as pinned:
        archive = pinned.archive
        read_page = archive.read_session_page
        check_frame = archive.check_operation_read

        def page(identifier: str, *, offset: int, **kwargs: Any) -> Any:
            nonlocal cancelled
            offsets.append(offset)
            if offset == 256:
                # A full page is on disk before the next page is acquired.
                assert len(stores) == 1
                assert stores[0].scalar("SELECT COUNT(*) FROM compact_messages") == 256
                if failure == "late-page":
                    raise RuntimeError("synthetic late compact page failure")
                if failure == "cancel":
                    cancelled = True
            return read_page(identifier, offset=offset, **kwargs)

        def frame() -> None:
            check_frame()
            if compacted and failure == "late-frame":
                raise RuntimeError("synthetic final compact frame failure")

        monkeypatch.setattr(archive, "read_session_page", page)
        monkeypatch.setattr(archive, "check_operation_read", frame)

        def read() -> dict[str, object]:
            return execute_read_operation(
                "read.compact",
                {"session_id": session_id, "params": {}, "projection": {"max_tokens": 60000}},
                archive=archive,
                serving_identity="test",
                dependencies=DaemonReadDependencies(raise_if_aborted=checkpoint),
            )

        if failure == "none":
            result = read()
            validate_operation_result("read.compact", result)
            pack = CorpusCompactionPack.model_validate(result["payload"])
            assert {item.text for item in pack.items} == {f"Neutral decision {position}" for position in range(257)}
        else:
            with pytest.raises(QueryCancelledError if failure == "cancel" else RuntimeError):
                read()
    assert offsets == [0, 256]
    assert len(stores) == len(paths) == 1
    assert not paths[0].parent.exists()
    with pytest.raises(sqlite3.ProgrammingError):
        stores[0].connection.execute("SELECT 1")
