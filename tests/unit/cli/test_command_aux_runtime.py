from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.shell_completion import CompletionItem

from polylogue.cli import shell_completion_values


def _ctx_param() -> tuple[click.Context, click.Parameter]:
    return click.Context(click.Command("polylogue")), click.Option(["--value"])


def test_shell_completion_helpers_cover_csv_prefix_rows_and_trimming(tmp_path: Path) -> None:
    assert shell_completion_values._split_csv_incomplete("alpha,beta") == ("alpha,", "beta")
    assert shell_completion_values._split_csv_incomplete("alpha, beta,g") == ("alpha,beta,", "g")
    assert shell_completion_values._split_csv_incomplete("cla") == ("", "cla")

    item = CompletionItem("tool", type="plain", help="help")
    prefixed = shell_completion_values._with_csv_prefix([item], "alpha,")
    assert [completion.value for completion in prefixed] == ["alpha,tool"]
    assert shell_completion_values._with_csv_prefix([item], "") == [item]

    assert shell_completion_values._trim_help("line one\nline two", limit=9) == "line one…"
    assert shell_completion_values._trim_help("short help") == "short help"

    db = tmp_path / "archive.sqlite"
    with patch("polylogue.cli.shell_completion_values.active_index_db_path", return_value=db):
        assert shell_completion_values._db_exists() is False
        db.write_text("", encoding="utf-8")
        assert shell_completion_values._db_exists() is True

    # Native stats-by completion action: archive.stats_by(group) -> count items.
    action = shell_completion_values._stats_by_items("tool", "", unit="actions")
    mock_archive = MagicMock()
    mock_archive.stats_by.return_value = {"alpha": 7}
    items = action(mock_archive)
    assert [(item.value, item.help) for item in items] == [("alpha", "7 actions")]


def test_completion_functions_cover_origin_session_tag_tool_and_open_targets() -> None:
    ctx, param = _ctx_param()

    origin_items = shell_completion_values.complete_origin_values(ctx, param, "chatgpt,cla")
    assert [item.value for item in origin_items] == ["chatgpt,claude-ai-export", "chatgpt,claude-code-session"]
    action_items = shell_completion_values.complete_action_values(ctx, param, "file")
    action_sequence_items = shell_completion_values.complete_action_sequence_values(ctx, param, "shell,file")
    message_type_items = shell_completion_values.complete_message_type_values(ctx, param, "m")
    retrieval_lane_items = shell_completion_values.complete_retrieval_lane_values(ctx, param, "h")

    # Archive-backed completions now run an ArchiveCompletionAction against an ArchiveStore via
    # ``_run_completion``; drive the action against a mock archive exposing the
    # archive read surface (list_summaries / list_user_tags / stats_by).
    summary = MagicMock()
    summary.session_id = "conv-1"
    summary.title = "Test Conv"
    summary.provider = MagicMock(value="claude-code")

    stats_by_groups = {
        "repo": {"polylogue": 4},
        "cwd": {"/realm/project/polylogue": 2},
        "tool": {"read_file": 7},
    }
    mock_archive = MagicMock()
    mock_archive.list_summaries.return_value = [summary]
    mock_archive.list_user_tags.return_value = {"review": 3}
    mock_archive.stats_by.side_effect = lambda group_by, **_: stats_by_groups[group_by]

    def fake_run_completion(action: object) -> list[object]:
        return list(action(mock_archive))  # type: ignore[operator]

    with (
        patch(
            "polylogue.cli.shell_completion_values._run_completion",
            side_effect=fake_run_completion,
        ),
        patch("polylogue.cli.shell_completion_values._db_exists", return_value=True),
    ):
        session_items = shell_completion_values.complete_session_ids(ctx, param, "conv")
        tag_items = shell_completion_values.complete_tag_values(ctx, param, "alpha,rev")
        repo_items = shell_completion_values.complete_repo_values(ctx, param, "old,poly")
        cwd_items = shell_completion_values.complete_cwd_prefix_values(ctx, param, "/realm")
        tool_items = shell_completion_values.complete_tool_values(ctx, param, "read")

    assert [item.value for item in action_items] == ["file_read", "file_write", "file_edit"]
    assert "shell,file_read" in [item.value for item in action_sequence_items]
    assert [item.value for item in message_type_items] == ["message"]
    assert [item.value for item in retrieval_lane_items] == ["hybrid"]
    assert session_items[0].value == "conv-1"
    assert session_items[0].help is not None and "claude-code" in session_items[0].help
    assert [item.value for item in tag_items] == ["alpha,review"]
    assert tag_items[0].help == "3 sessions"
    assert [item.value for item in repo_items] == ["old,polylogue"]
    assert repo_items[0].help == "4 sessions"
    # Native cwd-prefix completion has no session-cwd aggregate yet, so it
    # returns an empty list by design (well-behaved, no traceback).
    assert cwd_items == []
    assert [item.value for item in tool_items] == ["read_file"]
    assert tool_items[0].help == "7 actions"


def test_completion_source_registry_covers_descriptor_sources() -> None:
    from polylogue.archive.query.fields import query_completion_sources

    assert tuple(shell_completion_values.COMPLETION_SOURCE_HANDLERS) == query_completion_sources()
    assert shell_completion_values.complete_query_source("message_type") is (
        shell_completion_values.complete_message_type_values
    )


def test_query_field_candidates_come_from_expression_registry() -> None:
    from polylogue.archive.query.expression import EXPRESSION_FIELD_REGISTRY

    candidates = shell_completion_values.query_field_candidates("re")
    values = {candidate.value for candidate in candidates}

    assert "repo" in values
    repo_candidate = next(candidate for candidate in candidates if candidate.value == "repo")
    assert repo_candidate.insert == "repo:"
    assert repo_candidate.kind == "query-field"
    assert repo_candidate.source == "EXPRESSION_FIELD_REGISTRY"
    assert EXPRESSION_FIELD_REGISTRY["repo"]["description"] in repo_candidate.description
    payload = repo_candidate.to_payload()
    assert payload["value"] == "repo"
    assert payload["replace_start"] is None
    assert payload["danger"] is False
    assert payload["unsupported_reason"] is None
    click_item = shell_completion_values.query_completion_candidate_to_click_item(repo_candidate)
    assert click_item.value == "repo:"
    assert click_item.type == "plain"


def test_query_field_candidates_include_readable_operator_fields() -> None:
    date_candidates = shell_completion_values.query_field_candidates("d")
    date_candidate = next(candidate for candidate in date_candidates if candidate.value == "date")
    assert date_candidate.insert == "date "
    assert date_candidate.kind == "query-date-field"
    assert date_candidate.source == "DATE_QUERY_FIELD_REGISTRY"
    assert "date between 2026-01-01 and 2026-02-01" in date_candidate.description

    message_candidates = shell_completion_values.query_field_candidates("mess")
    message_candidate = next(candidate for candidate in message_candidates if candidate.value == "messages")
    assert message_candidate.insert == "messages:"
    assert message_candidate.source == "EXPRESSION_FIELD_REGISTRY/COUNT_QUERY_FIELD_REGISTRY"
    assert "messages between 5 and 20" in message_candidate.description

    role_count_candidates = shell_completion_values.query_field_candidates("user_m")
    user_messages_candidate = next(
        candidate for candidate in role_count_candidates if candidate.value == "user_messages"
    )
    assert user_messages_candidate.insert == "user_messages "
    assert user_messages_candidate.source == "EXPRESSION_FIELD_REGISTRY/COUNT_QUERY_FIELD_REGISTRY"
    assert "sessions where user_messages >= 2" in user_messages_candidate.description

    duration_candidates = shell_completion_values.query_field_candidates("dur")
    duration_candidate = next(candidate for candidate in duration_candidates if candidate.value == "duration_ms")
    assert duration_candidate.insert == "duration_ms "
    assert duration_candidate.source == "EXPRESSION_FIELD_REGISTRY/NUMERIC_QUERY_FIELD_REGISTRY"
    assert "duration_ms >= 60000" in duration_candidate.description


def test_query_field_candidates_disappear_with_registry_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.archive.query.expression import EXPRESSION_FIELD_REGISTRY

    monkeypatch.delitem(EXPRESSION_FIELD_REGISTRY, "repo")

    candidates = shell_completion_values.query_field_candidates("re")

    assert "repo" not in {candidate.value for candidate in candidates}


def test_query_structural_unit_candidates_come_from_expression_registry() -> None:
    candidates = shell_completion_values.query_structural_unit_candidates("b")
    assert [candidate.value for candidate in candidates] == ["block"]

    block_candidate = candidates[0]
    assert block_candidate.insert == "block("
    assert block_candidate.kind == "query-structural-unit"
    assert block_candidate.source == "QUERY_UNIT_DESCRIPTORS"
    assert block_candidate.description.startswith("Match sessions with at least one parsed message block")


def test_query_structural_field_candidates_come_from_expression_registry() -> None:
    candidates = shell_completion_values.query_structural_field_candidates("block", "t")
    values = {candidate.value for candidate in candidates}

    assert {"text", "type"}.issubset(values)
    type_candidate = next(candidate for candidate in candidates if candidate.value == "type")
    assert type_candidate.insert == "type:"
    assert type_candidate.kind == "query-structural-field"
    assert type_candidate.source == "QUERY_UNIT_DESCRIPTORS"
    assert "Unit-specific type token" in type_candidate.description
    assert "type:code" in type_candidate.description

    session_candidates = shell_completion_values.query_structural_field_candidates("message", "session.or")
    origin_candidate = next(candidate for candidate in session_candidates if candidate.value == "session.origin")
    assert origin_candidate.insert == "session.origin:"
    assert "Owning session scope" in origin_candidate.description
    assert "session.origin:claude-code-session" in origin_candidate.description


def test_query_structural_session_field_examples_are_executable() -> None:
    from polylogue.archive.query.expression import compile_expression
    from polylogue.archive.query.metadata import structural_query_field_info

    for field in ("session.messages", "session.words", "session.since", "session.origin"):
        info = structural_query_field_info("message", field)
        assert info is not None
        compile_expression(f"exists message({info.example} AND role:assistant)")


def test_query_count_operator_candidates_come_from_expression_registry() -> None:
    candidates = shell_completion_values.query_count_operator_candidates("messages", "b")

    assert [candidate.value for candidate in candidates] == ["between"]
    between_candidate = candidates[0]
    assert between_candidate.insert == "between "
    assert between_candidate.kind == "query-count-operator"
    assert between_candidate.source == "COUNT_QUERY_FIELD_REGISTRY"
    assert "messages between 5 and 20" in between_candidate.description


def test_query_numeric_operator_candidates_come_from_expression_registry() -> None:
    candidates = shell_completion_values.query_numeric_operator_candidates("duration_ms", "b")

    assert [candidate.value for candidate in candidates] == ["between"]
    between_candidate = candidates[0]
    assert between_candidate.insert == "between "
    assert between_candidate.kind == "query-numeric-operator"
    assert between_candidate.source == "NUMERIC_QUERY_FIELD_REGISTRY"
    assert "duration_ms >= 60000" in between_candidate.description


def test_query_pipeline_stage_candidates_come_from_unit_descriptors() -> None:
    candidates = shell_completion_values.query_pipeline_stage_candidates("messages", "g")
    values = [candidate.value for candidate in candidates]

    assert "group by role" in values
    assert "group by session.repo" in values
    role_candidate = next(candidate for candidate in candidates if candidate.value == "group by role")
    assert role_candidate.insert == "group by role"
    assert role_candidate.kind == "query-pipeline-stage"
    assert role_candidate.source == "QUERY_UNIT_DESCRIPTORS"
    assert role_candidate.payload_model == "MessageQueryRowPayload"

    sort_candidates = shell_completion_values.query_pipeline_stage_candidates("messages", "sort")
    sort_values = [candidate.value for candidate in sort_candidates]
    assert "sort by count" in sort_values
    assert "sort by key" in sort_values
    count_sort_candidate = next(candidate for candidate in sort_candidates if candidate.value == "sort by count")
    assert count_sort_candidate.insert == "sort by count desc"

    runtime_candidates = shell_completion_values.query_pipeline_stage_candidates("context-snapshots", "")
    assert [candidate.value for candidate in runtime_candidates] == ["limit", "offset"]


def test_query_expression_value_completion_uses_field_completion_source() -> None:
    ctx, param = _ctx_param()

    origin_items = shell_completion_values.complete_query_expression_fields(ctx, param, "origin:cla")
    assert [item.value for item in origin_items] == ["origin:claude-ai-export", "origin:claude-code-session"]

    with (
        patch("polylogue.cli.shell_completion_values._db_exists", return_value=True),
        patch("polylogue.cli.shell_completion_values._run_completion") as run_completion,
    ):
        mock_archive = MagicMock()
        mock_archive.stats_by.return_value = {"polylogue": 4}
        run_completion.side_effect = lambda action: list(action(mock_archive))
        repo_items = shell_completion_values.complete_query_expression_fields(ctx, param, "repo:poly")

    assert [item.value for item in repo_items] == ["repo:polylogue"]
    assert repo_items[0].help == "4 sessions"


def test_query_expression_completion_handles_structural_contexts() -> None:
    ctx, param = _ctx_param()

    unit_items = shell_completion_values.complete_query_expression_fields(ctx, param, "exists b")
    assert [item.value for item in unit_items] == ["block("]

    field_items = shell_completion_values.complete_query_expression_fields(ctx, param, "exists block(t")
    values = {item.value for item in field_items}
    assert {"text:", "type:"}.issubset(values)


def test_query_expression_completion_handles_terminal_contexts() -> None:
    ctx, param = _ctx_param()

    source_items = shell_completion_values.complete_query_expression_fields(ctx, param, "observed")
    assert "observed-events where " in {item.value for item in source_items}

    event_field_items = shell_completion_values.complete_query_expression_fields(
        ctx,
        param,
        "observed-events where d",
    )
    assert [item.value for item in event_field_items] == ["delivery_state:"]

    snapshot_field_items = shell_completion_values.complete_query_expression_fields(
        ctx,
        param,
        "context-snapshots where bound",
    )
    assert [item.value for item in snapshot_field_items] == ["boundary:"]

    chained_snapshot_items = shell_completion_values.complete_query_expression_fields(
        ctx,
        param,
        "context-snapshots where boundary:session_start AND sess",
    )
    assert "session.repo:" in {item.value for item in chained_snapshot_items}

    pipeline_items = shell_completion_values.complete_query_expression_fields(
        ctx,
        param,
        "messages where role:assistant | g",
    )
    pipeline_values = {item.value for item in pipeline_items}
    assert {"group by role", "group by session.repo"}.issubset(pipeline_values)


def test_query_action_candidates_come_from_action_contracts_with_danger_metadata() -> None:
    candidates = shell_completion_values.query_action_candidates("de")

    delete_candidate = next(candidate for candidate in candidates if candidate.value == "delete")
    assert delete_candidate.source == "ACTION_CONTRACTS"
    assert delete_candidate.kind == "query-action"
    assert delete_candidate.danger is True
    help_text = shell_completion_values.query_completion_candidate_to_click_item(delete_candidate).help
    assert help_text is not None
    assert help_text.startswith("DANGER:")
