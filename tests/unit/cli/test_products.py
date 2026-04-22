"""Tests for the durable archive data products CLI surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner, Result

from polylogue.archive_products import ProviderAnalyticsProduct
from polylogue.cli.click_app import cli
from polylogue.cli.commands.products import _make_callback
from polylogue.products.registry import get_product_type, product_items_payload
from polylogue.storage.action_event_rebuild_runtime import rebuild_action_event_read_model_sync
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.session_product_rebuild import rebuild_session_products_sync
from polylogue.storage.session_product_status import session_product_status_sync
from polylogue.storage.store_constants import SESSION_PRODUCT_MATERIALIZER_VERSION
from tests.infra.storage_records import ConversationBuilder

JsonObject = dict[str, object]
CliWorkspace = dict[str, Path]


def _expect_object(value: object) -> JsonObject:
    assert isinstance(value, dict)
    return value


def _expect_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _expect_object_list(value: object) -> list[JsonObject]:
    values = _expect_list(value)
    return [_expect_object(item) for item in values]


def _expect_int(value: object) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _expect_bool(value: object) -> bool:
    assert isinstance(value, bool)
    return value


def _expect_number(value: object) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def _extract_json(output: str) -> JsonObject:
    data = json.loads(output)
    if isinstance(data, dict) and data.get("status") == "ok":
        return _expect_object(data["result"])
    return _expect_object(data)


def _exception_message(result: Result) -> str:
    return str(result.exception) if result.exception is not None else result.output.strip()


def test_product_items_payload_can_render_cli_and_mcp_keys() -> None:
    product = ProviderAnalyticsProduct(
        provider_name="claude-code",
        conversation_count=1,
        message_count=2,
        user_message_count=1,
        assistant_message_count=1,
        avg_messages_per_conversation=2.0,
        avg_user_words=3.0,
        avg_assistant_words=4.0,
        tool_use_count=1,
        thinking_count=0,
        total_conversations_with_tools=1,
        total_conversations_with_thinking=0,
        tool_use_percentage=100.0,
        thinking_percentage=0.0,
    )
    product_type = get_product_type("provider_analytics")

    cli_payload = product_items_payload([product], product_type)
    mcp_payload = product_items_payload([product], product_type, item_key="items")

    assert cli_payload["count"] == 1
    assert _expect_object_list(cli_payload["provider_analytics"])[0]["product_kind"] == "provider_analytics"
    assert mcp_payload["count"] == 1
    assert _expect_object_list(mcp_payload["items"])[0]["provider_name"] == "claude-code"


def _seed_products(cli_workspace: CliWorkspace) -> None:
    db_path = cli_workspace["db_path"]
    (
        ConversationBuilder(db_path, "conv-root")
        .provider("claude-code")
        .title("Root Thread")
        .created_at("2026-03-01T10:00:00+00:00")
        .updated_at("2026-03-01T10:10:00+00:00")
        .add_message(
            "u1",
            role="user",
            text="Plan the refactor and inspect README",
            timestamp="2026-03-01T10:00:00+00:00",
        )
        .add_message(
            "a1",
            role="assistant",
            text="Inspecting and editing files",
            timestamp="2026-03-01T10:05:00+00:00",
            provider_meta={
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "tool_name": "Read",
                        "semantic_type": "file_read",
                        "input": {"path": "/workspace/polylogue/README.md"},
                    },
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "semantic_type": "file_edit",
                        "input": {"path": "/workspace/polylogue/README.md"},
                    },
                ]
            },
        )
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-child")
        .provider("claude-code")
        .title("Child Thread")
        .parent_conversation("conv-root")
        .branch_type("continuation")
        .created_at("2026-03-01T11:00:00+00:00")
        .updated_at("2026-03-01T11:05:00+00:00")
        .add_message(
            "u2",
            role="user",
            text="Run tests for the refactor",
            timestamp="2026-03-01T11:00:00+00:00",
        )
        .add_message(
            "a2",
            role="assistant",
            text="Running pytest",
            timestamp="2026-03-01T11:04:00+00:00",
            provider_meta={
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "tool_name": "Bash",
                        "semantic_type": "shell",
                        "input": {"command": "pytest -q tests/unit/cli/test_products.py"},
                    }
                ]
            },
        )
        .save()
    )
    with open_connection(db_path) as conn:
        rebuild_session_products_sync(conn)
        rebuild_action_event_read_model_sync(conn)


def test_products_profiles_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["products", "profiles", "--json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    assert _expect_int(payload["count"]) == 2
    first = _expect_object_list(payload["session_profiles"])[0]
    evidence = _expect_object(first["evidence"])
    inference = _expect_object(first["inference"])
    assert _expect_int(first["contract_version"]) == 4
    assert first["product_kind"] == "session_profile"
    assert first["semantic_tier"] == "merged"
    assert evidence["canonical_session_date"] == "2026-03-01"
    assert _expect_number(inference["engaged_duration_ms"]) >= 0
    assert "evidence" in first
    assert "inference" in first
    assert "provenance" in first


def test_products_profiles_format_json_alias(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["products", "profiles", "--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    assert _expect_int(payload["count"]) == 2


def test_products_profiles_inherit_root_format_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--format", "json", "products", "profiles", "--limit", "1"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    assert _expect_int(payload["count"]) == 1


def test_products_enrichments_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--provider",
            "claude-code",
            "products",
            "enrichments",
            "--session-date-since",
            "2026-03-01",
            "--json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    assert _expect_int(payload["count"]) == 2
    first = _expect_object_list(payload["session_enrichments"])[0]
    enrichment_provenance = _expect_object(first["enrichment_provenance"])
    enrichment = _expect_object(first["enrichment"])
    assert _expect_int(first["contract_version"]) == 4
    assert first["product_kind"] == "session_enrichment"
    assert first["semantic_tier"] == "enrichment"
    assert enrichment_provenance["enrichment_family"] == "scored_session_enrichment"
    assert enrichment["support_level"] in {"weak", "moderate", "strong"}
    assert "input_band_summary" in enrichment


def test_products_callback_rejects_unknown_query_fields() -> None:
    callback = _make_callback(get_product_type("session_enrichments"))
    env = SimpleNamespace(operations=MagicMock())
    # Build a minimal Click context to satisfy @click.pass_context
    mock_ctx = click.Context(click.Command("enrichments"))
    mock_ctx.obj = env
    # No parent — root filter inheritance is a no-op

    with pytest.raises(
        SystemExit, match="products enrichments: Unknown query field\\(s\\) for session_enrichments: refined_work_kind"
    ):
        wrapped = getattr(callback, "__wrapped__", None)
        assert callable(wrapped)
        wrapped(mock_ctx, json_mode=False, refined_work_kind="planning")


def test_products_profiles_json_supports_explicit_evidence_and_inference_tiers(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    evidence_result = runner.invoke(
        cli,
        ["products", "profiles", "--tier", "evidence", "--json"],
        catch_exceptions=False,
    )
    inference_result = runner.invoke(
        cli,
        ["products", "profiles", "--tier", "inference", "--json"],
        catch_exceptions=False,
    )

    assert evidence_result.exit_code == 0
    assert inference_result.exit_code == 0

    evidence_payload = _extract_json(evidence_result.output)
    inference_payload = _extract_json(inference_result.output)
    evidence_profile = _expect_object_list(evidence_payload["session_profiles"])[0]
    inference_profile = _expect_object_list(inference_payload["session_profiles"])[0]

    assert evidence_profile["semantic_tier"] == "evidence"
    assert _expect_object(evidence_profile["evidence"])["canonical_session_date"] == "2026-03-01"
    assert evidence_profile["inference"] is None
    assert evidence_profile["inference_provenance"] is None

    assert inference_profile["semantic_tier"] == "inference"
    assert inference_profile["evidence"] is None
    assert _expect_number(_expect_object(inference_profile["inference"])["engaged_duration_ms"]) >= 0
    assert (
        _expect_object(inference_profile["inference_provenance"])["inference_family"] == "heuristic_session_semantics"
    )


def test_products_profiles_json_handles_blank_tier_search_text_from_migrated_rows(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)
    with open_connection(cli_workspace["db_path"]) as conn:
        conn.execute("UPDATE session_profiles SET evidence_search_text = '', inference_search_text = ''")
        conn.commit()

    runner = CliRunner()
    evidence_result = runner.invoke(
        cli,
        ["products", "profiles", "--tier", "evidence", "--json"],
        catch_exceptions=False,
    )
    inference_result = runner.invoke(
        cli,
        ["products", "profiles", "--tier", "inference", "--json"],
        catch_exceptions=False,
    )

    assert evidence_result.exit_code == 0
    assert inference_result.exit_code == 0
    assert _expect_int(_extract_json(evidence_result.output)["count"]) == 2
    assert _expect_int(_extract_json(inference_result.output)["count"]) == 2


def test_products_reconstructs_tiered_payloads_from_blank_migrated_rows(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)
    with open_connection(cli_workspace["db_path"]) as conn:
        conn.execute("UPDATE session_profiles SET evidence_payload_json = '{}', inference_payload_json = '{}'")
        conn.execute("UPDATE session_work_events SET evidence_payload_json = '{}', inference_payload_json = '{}'")
        conn.execute("UPDATE session_phases SET evidence_payload_json = '{}', inference_payload_json = '{}'")
        conn.commit()

    runner = CliRunner()
    evidence_profiles = runner.invoke(
        cli,
        ["products", "profiles", "--tier", "evidence", "--json"],
        catch_exceptions=False,
    )
    inference_profiles = runner.invoke(
        cli,
        ["products", "profiles", "--tier", "inference", "--json"],
        catch_exceptions=False,
    )
    work_events = runner.invoke(cli, ["products", "work-events", "--json"], catch_exceptions=False)
    phases = runner.invoke(cli, ["products", "phases", "--json"], catch_exceptions=False)

    assert evidence_profiles.exit_code == 0
    assert inference_profiles.exit_code == 0
    assert work_events.exit_code == 0
    assert phases.exit_code == 0

    evidence_profiles_payload = _expect_object_list(_extract_json(evidence_profiles.output)["session_profiles"])
    inference_profiles_payload = _expect_object_list(_extract_json(inference_profiles.output)["session_profiles"])
    work_event = _expect_object_list(_extract_json(work_events.output)["session_work_events"])[0]
    phase = _expect_object_list(_extract_json(phases.output)["session_phases"])[0]

    assert all(
        _expect_int(_expect_object(item["evidence"])["message_count"]) >= 1 for item in evidence_profiles_payload
    )
    assert all(
        _expect_object(item["evidence"])["canonical_session_date"] == "2026-03-01" for item in evidence_profiles_payload
    )
    assert all(
        _expect_int(_expect_object(item["inference"])["work_event_count"]) >= 0 for item in inference_profiles_payload
    )
    assert _expect_int(_expect_object(work_event["evidence"])["start_index"]) >= 0
    assert _expect_object(work_event["inference"])["kind"] in {"implementation", "testing", "planning", "debugging"}
    assert _expect_int(_expect_list(_expect_object(phase["evidence"])["message_range"])[0]) >= 0
    assert _expect_number(_expect_object(phase["inference"])["confidence"]) >= 0.0


def test_products_profile_date_filters_and_phases_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    profiles = runner.invoke(
        cli,
        [
            "products",
            "profiles",
            "--session-date-since",
            "2026-03-01",
            "--session-date-until",
            "2026-03-01",
            "--json",
        ],
        catch_exceptions=False,
    )
    phases = runner.invoke(cli, ["products", "phases", "--json"], catch_exceptions=False)

    assert profiles.exit_code == 0
    assert phases.exit_code == 0

    profile_payload = _extract_json(profiles.output)
    phase_payload = _extract_json(phases.output)
    assert _expect_int(profile_payload["count"]) == 2
    assert _expect_int(phase_payload["count"]) >= 1
    assert _expect_object_list(phase_payload["session_phases"])[0]["product_kind"] == "session_phase"


def test_session_product_rebuild_supports_legacy_payload_columns(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        conn.execute("ALTER TABLE session_profiles ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'")
        conn.execute("ALTER TABLE session_work_events ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'")
        conn.execute("ALTER TABLE session_phases ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'")
        conn.commit()
        rebuild_session_products_sync(conn)
        status = session_product_status_sync(conn)

    assert status.profile_row_count == 2
    assert status.profile_rows_ready is True
    assert status.work_event_inference_rows_ready is True
    assert status.phase_inference_rows_ready is True


def test_session_product_rebuild_pages_full_rebuild(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        counts = rebuild_session_products_sync(conn, page_size=1)
        status = session_product_status_sync(conn)

    assert counts.profiles == 2
    assert counts.work_events >= 1
    assert counts.phases >= 1
    assert status.profile_row_count == 2
    assert status.profile_rows_ready is True
    assert status.work_event_inference_rows_ready is True
    assert status.phase_inference_rows_ready is True


def test_session_product_rebuild_sync_reports_progress(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    observed: list[tuple[int, str | None]] = []

    with open_connection(cli_workspace["db_path"]) as conn:
        rebuild_session_products_sync(
            conn,
            page_size=1,
            progress_callback=lambda amount, desc=None: observed.append((_expect_int(amount), desc)),
            progress_total=2,
        )

    assert observed == [
        (1, "Materializing: 1/2"),
        (1, "Materializing: 2/2"),
    ]


def test_session_product_rebuild_preserves_profile_semantics_without_loading_full_provider_meta(
    cli_workspace: CliWorkspace,
) -> None:
    db_path = cli_workspace["db_path"]
    (
        ConversationBuilder(db_path, "conv-heavy")
        .provider("codex")
        .title("Heavy Provider Meta")
        .created_at("2026-03-01T10:00:00+00:00")
        .updated_at("2026-03-01T10:10:00+00:00")
        .add_message(
            "u1",
            role="user",
            text="Continue work on /realm/project/sinex and inspect the branch.",
            timestamp="2026-03-01T10:00:00+00:00",
        )
        .add_message(
            "a1",
            role="assistant",
            text="Inspecting the repository state.",
            timestamp="2026-03-01T10:05:00+00:00",
        )
        .save()
    )
    huge_provider_meta = {
        "git": {
            "branch": "master",
            "repository_url": "git@github.com:Sinity/sinex.git",
        },
        "context_compactions": [{"summary": "Earlier context collapsed."}],
        "raw": {"payload": "x" * 200_000},
    }

    with open_connection(db_path) as conn:
        conn.execute(
            "UPDATE conversations SET provider_meta = ? WHERE conversation_id = ?",
            (json.dumps(huge_provider_meta), "conv-heavy"),
        )
        counts = rebuild_session_products_sync(conn, page_size=1)
        row = conn.execute(
            "SELECT repo_names_json, evidence_payload_json FROM session_profiles WHERE conversation_id = ?",
            ("conv-heavy",),
        ).fetchone()

    assert counts.profiles == 1
    assert row is not None
    repo_names = _expect_list(json.loads(row["repo_names_json"]))
    evidence_payload = _expect_object(json.loads(row["evidence_payload_json"]))
    assert "sinex" in repo_names
    assert _expect_int(evidence_payload["compaction_count"]) == 1


def test_products_threads_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    threads = runner.invoke(cli, ["products", "threads", "--json"], catch_exceptions=False)

    assert threads.exit_code == 0

    threads_payload = _extract_json(threads.output)
    assert _expect_int(threads_payload["count"]) == 1
    assert _expect_object_list(threads_payload["work_threads"])[0]["product_kind"] == "work_thread"


def test_products_tag_and_summary_rollups_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    tags = runner.invoke(cli, ["products", "tags", "--json"], catch_exceptions=False)
    days = runner.invoke(cli, ["products", "day-summaries", "--json"], catch_exceptions=False)
    weeks = runner.invoke(cli, ["products", "week-summaries", "--json"], catch_exceptions=False)

    assert tags.exit_code == 0
    assert days.exit_code == 0
    assert weeks.exit_code == 0

    tag_payload = _extract_json(tags.output)
    day_payload = _extract_json(days.output)
    week_payload = _extract_json(weeks.output)
    assert any(
        item["tag"] == "provider:claude-code" for item in _expect_object_list(tag_payload["session_tag_rollups"])
    )
    assert _expect_int(day_payload["count"]) == 1
    assert _expect_object_list(day_payload["day_session_summaries"])[0]["product_kind"] == "day_session_summary"
    assert _expect_int(week_payload["count"]) == 1
    assert _expect_object_list(week_payload["week_session_summaries"])[0]["product_kind"] == "week_session_summary"


def test_products_analytics_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--provider", "claude-code", "products", "analytics", "--json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    assert _expect_int(payload["count"]) == 1
    item = _expect_object_list(payload["provider_analytics"])[0]
    assert item["product_kind"] == "provider_analytics"
    assert item["provider_name"] == "claude-code"
    assert _expect_int(item["conversation_count"]) == 2
    assert _expect_int(item["tool_use_count"]) == 2


def test_session_product_status_accepts_epoch_backed_conversation_timestamps(cli_workspace: CliWorkspace) -> None:
    db_path = cli_workspace["db_path"]
    (
        ConversationBuilder(db_path, "conv-epoch")
        .provider("claude-code")
        .title("Epoch-backed timestamps")
        .created_at("1740823200.0")
        .updated_at("1740826800.0")
        .add_message("u1", role="user", text="Inspect the archive state")
        .add_message(
            "a1",
            role="assistant",
            text="Inspecting files and planning repairs",
            provider_meta={
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "tool_name": "Read",
                        "semantic_type": "file_read",
                        "input": {"path": "/workspace/polylogue/README.md"},
                    }
                ]
            },
        )
        .save()
    )

    with open_connection(db_path) as conn:
        rebuild_session_products_sync(conn)
        status = session_product_status_sync(conn)

    assert status.profile_row_count == 1
    assert status.stale_profile_row_count == 0
    assert status.stale_work_event_inference_count == 0
    assert status.stale_phase_inference_count == 0
    assert status.profile_rows_ready is True
    assert status.work_event_inference_rows_ready is True
    assert status.phase_inference_rows_ready is True
    assert status.profile_merged_fts_duplicate_count == 0


def test_targeted_session_product_rebuild_does_not_duplicate_profile_fts(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        rebuild_session_products_sync(conn, conversation_ids=["conv-root"])
        status = session_product_status_sync(conn)

    assert status.profile_row_count == 2
    assert status.profile_merged_fts_count == 2
    assert status.profile_merged_fts_duplicate_count == 0
    assert status.profile_merged_fts_ready is True


def test_session_product_status_marks_older_materializer_versions_stale(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        rebuild_session_products_sync(conn)
        conn.execute(
            "UPDATE session_profiles SET materializer_version = ?",
            (SESSION_PRODUCT_MATERIALIZER_VERSION - 1,),
        )
        conn.commit()
        status = session_product_status_sync(conn)

    assert status.profile_row_count == 2
    assert status.stale_profile_row_count == 2
    assert status.profile_rows_ready is False


@pytest.mark.parametrize(
    ("argv", "sql", "expected"),
    [
        (
            ["products", "profiles", "--json"],
            "UPDATE session_profiles SET materializer_version = ?",
            "Session-profile rows are incomplete.",
        ),
        (
            ["products", "work-events", "--json"],
            "UPDATE session_work_events SET materializer_version = ?",
            "Session work-event rows are incomplete.",
        ),
        (
            ["products", "phases", "--json"],
            "UPDATE session_phases SET materializer_version = ?",
            "Session phase rows are incomplete.",
        ),
        (
            ["products", "threads", "--json"],
            "UPDATE work_threads SET materializer_version = ?",
            "Work-thread rows are incomplete.",
        ),
        (
            ["products", "tags", "--json"],
            "UPDATE session_tag_rollups SET materializer_version = ?",
            "Session tag rollups are incomplete.",
        ),
        (
            ["products", "day-summaries", "--json"],
            "UPDATE day_session_summaries SET materializer_version = ?",
            "Day session summaries are incomplete.",
        ),
        (
            ["products", "week-summaries", "--json"],
            "UPDATE day_session_summaries SET materializer_version = ?",
            "Week session summaries are incomplete.",
        ),
    ],
)
def test_products_reject_stale_session_product_surfaces(
    cli_workspace: CliWorkspace, argv: list[str], sql: str, expected: str
) -> None:
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        conn.execute(sql, (SESSION_PRODUCT_MATERIALIZER_VERSION - 1,))
        conn.commit()

    runner = CliRunner()
    result = runner.invoke(cli, argv, catch_exceptions=False)

    assert result.exit_code == 1
    message = _exception_message(result)
    assert expected in message
    assert "polylogue doctor --repair --target session_products" in message


def test_products_profiles_reject_incomplete_profile_search_index(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        conn.execute("DELETE FROM session_profiles_fts")
        conn.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["products", "profiles", "--query", "refactor", "--json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    message = _exception_message(result)
    assert "Session-profile merged search index is incomplete." in message
    assert "polylogue doctor --repair --target session_products" in message
