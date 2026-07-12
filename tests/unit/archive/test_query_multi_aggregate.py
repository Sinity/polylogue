from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.query.expression import ExpressionCompileError, parse_unit_source_expression
from polylogue.archive.query.unit_results import query_unit_rows
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope
from tests.infra.storage_records import SessionBuilder


def test_multi_field_group_count_reports_proportions_and_denominator(workspace_env: dict[str, Path]) -> None:
    index_db = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index_db, "claude")
        .provider("claude-code")
        .add_message("user", role="user", text="aggregate")
        .add_message("assistant", role="assistant", text="aggregate")
        .save()
    )
    (
        SessionBuilder(index_db, "codex")
        .provider("codex")
        .add_message("assistant", role="assistant", text="aggregate")
        .save()
    )
    source = parse_unit_source_expression("messages where text:aggregate | group by role, session.origin | count")
    assert source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="multi-aggregate", limit=20)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    assert envelope.pipeline_stages == (
        {"kind": "group", "fields": ["role", "session.origin"]},
        {"kind": "count", "metric": "count"},
        {"kind": "terminal", "action": "count"},
    )
    assert envelope.pipeline is not None
    result = cast(dict[str, object], envelope.pipeline["result"])
    assert {key: result[key] for key in result if key != "groups"} == {
        "group_by": ["role", "session.origin"],
        "aggregate": ["count", "proportion"],
        "denominator": {"kind": "all_matching_rows", "n": 3},
        "n": 3,
        "missing_counts": {"role": 0, "session.origin": 0},
        "unknown_counts": {"role": 0, "session.origin": 0},
    }
    groups = cast(list[dict[str, object]], result["groups"])
    assert {
        (
            cast(dict[str, str], item["group"])["role"],
            cast(dict[str, str], item["group"])["session.origin"],
            item["count"],
            item["proportion"],
        )
        for item in groups
    } == {
        ("assistant", "claude-code-session", 1, 1 / 3),
        ("assistant", "codex-session", 1, 1 / 3),
        ("user", "claude-code-session", 1, 1 / 3),
    }
    assert {
        (json.loads(row.group_key)["role"], json.loads(row.group_key)["session.origin"], row.count)
        for row in envelope.items
        if row.group_key is not None
    } == {
        ("assistant", "claude-code-session", 1),
        ("assistant", "codex-session", 1),
        ("user", "claude-code-session", 1),
    }


def test_multi_field_group_rejects_each_unsupported_field() -> None:
    with pytest.raises(ExpressionCompileError, match=r"group by nope.*action rows"):
        parse_unit_source_expression("actions where tool:bash | group by tool, nope | count")
