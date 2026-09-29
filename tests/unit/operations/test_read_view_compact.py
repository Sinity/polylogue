"""The compact projection is reachable through its declared read operation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from polylogue.operations.daemon_protocol import daemon_operation_spec, validate_operation_result
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.compaction import CorpusCompactionPack
from tests.infra.storage_records import SessionBuilder


def test_read_compact_builds_a_pack_from_the_pinned_archive(tmp_path: Path) -> None:
    """``read.compact`` executes ``compact_sessions`` over archived sessions.

    Anti-vacuity: without the ``read.compact`` branch in
    ``daemon_reads.execute_read_operation`` the call raises "read operation is
    not declared"; without the integer ``tool_result_is_error`` handling in
    ``compaction._tool_outcome`` the archived successful tool result (stored as
    ``0``) is kept as an item instead of omitted as ``successful_tool_spam``.
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
    with ArchiveStore.open_existing(root, read_only=True) as archive:
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
