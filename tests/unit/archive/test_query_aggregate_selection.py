from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.archive.query.expression import parse_unit_source_expression
from polylogue.archive.query.predicate import QueryExistsPredicate
from polylogue.archive.query.unit_results import query_unit_rows
from polylogue.operations.daemon_reads import _aggregate_payload
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope
from tests.infra.storage_records import SessionBuilder


def test_date_groups_preserve_pre_epoch_fractional_seconds(workspace_env: dict[str, Path]) -> None:
    index = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index, "pre-epoch")
        .provider("codex")
        .created_at("1969-12-31T23:59:59.999000+00:00")
        .updated_at("1969-12-31T23:59:59.999000+00:00")
        .add_message("message", role="user", text="neutral request")
        .save()
    )
    with ArchiveStore.open_existing(index.parent) as archive:
        assert archive.stats_by("day") == {"1969-12-31": 1}
        assert archive.stats_by("month") == {"1969-12": 1}
        assert archive.stats_by("year") == {"1969": 1}


@pytest.mark.parametrize("mode", ["count", "stats", "stats_by"])
def test_latest_aggregate_reduces_the_canonical_one_session_window(workspace_env: dict[str, Path], mode: str) -> None:
    index = workspace_env["archive_root"] / "index.db"
    for name, provider, day in (("old", "claude-code", "01"), ("new", "codex", "02")):
        (
            SessionBuilder(index, name)
            .provider(provider)
            .created_at(f"2026-01-{day}T00:00:00+00:00")
            .updated_at(f"2026-01-{day}T00:00:00+00:00")
            .add_message(f"{name}-message", role="user", text="neutral request")
            .save()
        )
    with ArchiveStore.open_existing(index.parent) as archive:
        result = _aggregate_payload({"mode": mode, "group_by": "origin", "params": {"latest": True}}, archive=archive)
    assert result["outcome"]["state"] == "ok"
    if mode == "count":
        assert result["count"] == 1
    elif mode == "stats_by":
        assert result["groups"] == {"codex-session": 1}
    else:
        assert result["stats"]["total_sessions"] == 1
        assert result["stats"]["origins"] == {"codex-session": 1}


@pytest.mark.parametrize("group", ["role", "role, session.origin"])
def test_named_aggregate_sorts_keys_before_paging(workspace_env: dict[str, Path], group: str) -> None:
    index = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index, "ordered")
        .provider("codex")
        .add_message("a", role="assistant", text="one two")
        .add_message("u", role="user", text="one two three")
        .save()
    )
    query = f"messages where role:(user|assistant) | group by {group} | agg count, sum:word_count | sort by key desc | limit 1"
    source = parse_unit_source_expression(query)
    assert source is not None
    with ArchiveStore.open_existing(index.parent) as archive:
        first = query_unit_rows(archive, source, query=query, limit=1)
        second = query_unit_rows(archive, source, query=query, limit=1, offset=1)
    assert isinstance(first, QueryUnitAggregateEnvelope)
    assert isinstance(second, QueryUnitAggregateEnvelope)
    keys = [page.items[0].group_key for page in (first, second)]
    if "," in group:
        assert [json.loads(key)["role"] for key in keys] == ["user", "assistant"]
    else:
        assert keys == ["user", "assistant"]
    assert first.items[0].metrics == {"count": 1, "sum_word_count": 3.0}
    assert first.next_offset == 1
    assert second.next_offset is None


@pytest.mark.parametrize("terminal", ["count", "group by role, session.origin | count", "agg count, sum:word_count"])
def test_or_aggregate_predicate_intersects_the_external_scope(workspace_env: dict[str, Path], terminal: str) -> None:
    index = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index, "excluded")
        .provider("claude-code")
        .add_message("excluded-user", role="user", text="one two three")
        .save()
    )
    (
        SessionBuilder(index, "included")
        .provider("codex")
        .add_message("included-assistant", role="assistant", text="one two")
        .save()
    )
    query = f"messages where role:user OR role:assistant | {terminal}"
    source = parse_unit_source_expression(query)
    assert source is not None
    with ArchiveStore.open_existing(index.parent) as archive:
        page = query_unit_rows(archive, source, query=query, limit=20, session_filters={"origin": "codex-session"})
    assert isinstance(page, QueryUnitAggregateEnvelope)
    assert sum(row.count for row in page.items) == 1
    if terminal.startswith("agg"):
        assert page.items[0].metrics == {"count": 1, "sum_word_count": 2.0}


def test_or_predicate_keeps_row_scope_and_exists_correlation(workspace_env: dict[str, Path]) -> None:
    index = workspace_env["archive_root"] / "index.db"
    for name, provider, role in (
        ("excluded", "claude-code", "user"),
        ("included", "codex", "assistant"),
        ("unrelated", "codex", "system"),
    ):
        SessionBuilder(index, name).provider(provider).add_message(name, role=role, text="neutral").save()
    source = parse_unit_source_expression("messages where role:user OR role:assistant")
    assert source is not None
    with ArchiveStore.open_existing(index.parent) as archive:
        rows = archive.query_message_projection(
            source.predicate, fields=("role",), session_filters={"origin": "codex-session"}
        )
        count = archive.count_sessions(boolean_predicate=QueryExistsPredicate(unit="message", child=source.predicate))
    assert [row["role"] for row in rows] == ["assistant"]
    assert count == 2


@pytest.mark.parametrize("terminal", ["count", "agg count, p50:word_count"])
@pytest.mark.parametrize("group", ["session.repo", "role, session.repo"])
def test_group_null_is_distinct_from_a_literal_missing_label(
    workspace_env: dict[str, Path], terminal: str, group: str
) -> None:
    index = workspace_env["archive_root"] / "index.db"
    for name, repo, text in (("absent", None, "one two"), ("literal", "[missing]", "one two three")):
        (
            SessionBuilder(index, name)
            .provider("codex")
            .git_repository_url(repo)
            .add_message(f"{name}-message", role="assistant", text=text)
            .save()
        )
    query = f"messages where role:assistant | group by {group} | {terminal} | sort by key asc"
    source = parse_unit_source_expression(query)
    assert source is not None
    with ArchiveStore.open_existing(index.parent) as archive:
        page = query_unit_rows(archive, source, query=query, limit=20)
    assert isinstance(page, QueryUnitAggregateEnvelope)
    assert len(page.items) == 2
    if "," in group:
        keys = [json.loads(row.group_key)["session.repo"] for row in page.items]
    else:
        keys = [row.group_key for row in page.items]
    assert keys == [None, "[missing]"]
    assert [row.count for row in page.items] == [1, 1]
    if terminal.startswith("agg"):
        assert [row.metrics["p50_word_count"] for row in page.items] == [2.0, 3.0]
