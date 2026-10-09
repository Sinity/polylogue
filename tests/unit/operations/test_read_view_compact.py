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
    from polylogue.surfaces.compaction import estimate_tokens

    root = tmp_path / "archive"
    root.mkdir()
    builder = SessionBuilder(root / "index.db", "compact-ladder").provider("codex")
    text = "decision " * 1000
    for index in range(20):
        builder.add_message(f"message-{index}", role="assistant", text=text, material_origin="assistant_authored")
    builder.save()
    with ArchiveStore.open_existing(root, read_only=True) as archive:
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
