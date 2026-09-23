"""Semantic witness for the command-shape family decision."""

from polylogue.analysis.command_shapes import (
    CommandShapeUsageQuery,
    build_command_shape_usage,
    normalize_command_shapes,
)


def test_command_shape_library_preserves_shell_semantics_before_aggregation() -> None:
    """A SQL-style grouping would lose pipeline and wrapper boundaries."""

    command = "env CI=1 bash -lc 'pytest tests/unit --maxfail=1 | rg failed'"
    assert normalize_command_shapes(command) == ("pytest", "rg failed")

    rows = [
        {
            "origin": "codex",
            "repository": "polylogue",
            "session_id": "s1",
            "tool_command": command,
            "occurred_at_ms": 1_000,
        },
        {
            "origin": "codex",
            "repository": "polylogue",
            "session_id": "s2",
            "tool_command": "pytest tests/unit --maxfail=1",
            "occurred_at_ms": 2_000,
        },
    ]
    result = build_command_shape_usage(rows, CommandShapeUsageQuery(), materialized_at="now")

    assert [(item.command_shape, item.execution_count, item.session_count) for item in result] == [
        ("pytest", 2, 2),
        ("rg failed", 1, 1),
    ]
    assert all(item.provenance.materializer_version == 1 for item in result)
