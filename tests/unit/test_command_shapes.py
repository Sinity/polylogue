from polylogue.insights.command_shapes import (
    CommandShapeUsageQuery,
    build_command_shape_usage,
    normalize_command_shapes,
)


def test_normalization_collapses_arguments_wrappers_and_expands_pipelines() -> None:
    assert normalize_command_shapes("env FOO=bar bash -lc 'foo bar --flag=/tmp/a | foo bar --flag=/tmp/b'") == (
        "foo bar",
        "foo bar",
    )


def test_normalization_drops_path_arguments_without_dropping_subcommands() -> None:
    assert normalize_command_shapes("foo bar /tmp/one --flag=/tmp/two") == ("foo bar",)


def test_builder_aggregates_shape_counts_and_last_use() -> None:
    rows = [
        {
            "origin": "codex",
            "session_id": "s1",
            "repository": "polylogue",
            "tool_command": "foo bar",
            "occurred_at_ms": 1000,
        },
        {
            "origin": "codex",
            "session_id": "s2",
            "repository": "polylogue",
            "tool_command": "foo bar --x /tmp/a",
            "occurred_at_ms": 3000,
        },
        {
            "origin": "codex",
            "session_id": "s2",
            "repository": "polylogue",
            "tool_command": "other status",
            "occurred_at_ms": 2000,
        },
    ]
    result = build_command_shape_usage(rows, CommandShapeUsageQuery(), materialized_at="now")
    assert [(item.command_shape, item.execution_count, item.session_count) for item in result] == [
        ("foo bar", 2, 2),
        ("other status", 1, 1),
    ]
    assert result[0].last_used_at == "1970-01-01T00:00:03+00:00"
