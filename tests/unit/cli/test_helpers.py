"""Focused CLI helper contracts for summary rendering."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from polylogue.cli.helpers import print_summary
from polylogue.cli.types import AppEnv
from polylogue.config import Config
from polylogue.services import build_runtime_services


@pytest.fixture
def config() -> Config:
    return Config(
        archive_root=Path("/data/archive"),
        render_root=Path("/data/archive/rendered"),
        sources=[],
    )


def _make_env(config: Config, *, plain: bool) -> tuple[AppEnv, StringIO]:
    buffer = StringIO()
    ui = MagicMock()
    ui.plain = plain
    ui.console = Console(file=buffer, width=120, force_terminal=False, color_system=None)
    env = AppEnv(ui=ui, services=build_runtime_services(config=config, backend=MagicMock()))
    return env, buffer


def _health_report(*, cached=None, age_seconds=None, checks=None):
    return SimpleNamespace(cached=cached, age_seconds=age_seconds, checks=checks or [])


def _check(name: str, status: str, detail: str):
    return SimpleNamespace(name=name, status=status, detail=detail)


def _metric(
    provider_name: str,
    conversation_count: int,
    message_count: int = 0,
    avg_messages_per_conversation: float = 0.0,
    avg_user_words: float = 0.0,
    avg_assistant_words: float = 0.0,
    tool_use_count: int = 0,
    tool_use_percentage: float = 0.0,
    thinking_count: int = 0,
    thinking_percentage: float = 0.0,
):
    return SimpleNamespace(
        provider_name=provider_name,
        conversation_count=conversation_count,
        message_count=message_count,
        avg_messages_per_conversation=avg_messages_per_conversation,
        avg_user_words=avg_user_words,
        avg_assistant_words=avg_assistant_words,
        tool_use_count=tool_use_count,
        tool_use_percentage=tool_use_percentage,
        thinking_count=thinking_count,
        thinking_percentage=thinking_percentage,
    )


def _run_summary(
    config: Config,
    *,
    verbose: bool,
    plain: bool,
    last_run=None,
    cached_health: str = "OK",
    health=None,
    counts=None,
    metrics=None,
    analytics_error: Exception | None = None,
):
    env, buffer = _make_env(config, plain=plain)
    analytics_patches = {
        "polylogue.cli.helpers.compute_provider_comparison": AsyncMock(return_value=metrics),
        "polylogue.cli.helpers.get_provider_counts": AsyncMock(return_value=counts),
    }
    if analytics_error is not None:
        analytics_patches["polylogue.cli.helpers.compute_provider_comparison"] = AsyncMock(side_effect=analytics_error)
        analytics_patches["polylogue.cli.helpers.get_provider_counts"] = AsyncMock(side_effect=analytics_error)

    with (
        patch("polylogue.cli.helpers.latest_run", new_callable=AsyncMock, return_value=last_run),
        patch("polylogue.cli.helpers.cached_health_summary", return_value=cached_health) as mock_cached,
        patch("polylogue.cli.helpers.get_health", return_value=health) as mock_get_health,
        patch("polylogue.cli.helpers.format_sources_summary", return_value="inbox"),
        patch("polylogue.cli.helpers.compute_provider_comparison", analytics_patches["polylogue.cli.helpers.compute_provider_comparison"]),
        patch("polylogue.cli.helpers.get_provider_counts", analytics_patches["polylogue.cli.helpers.get_provider_counts"]),
    ):
        print_summary(env, verbose=verbose)

    title, lines = env.ui.summary.call_args.args
    return {
        "title": title,
        "lines": lines,
        "console": buffer.getvalue(),
        "mock_cached": mock_cached,
        "mock_get_health": mock_get_health,
    }


def test_print_summary_basic_contract(config: Config) -> None:
    last_run = SimpleNamespace(run_id="run-123", timestamp="2025-01-15T12:30:45Z")
    result = _run_summary(
        config,
        verbose=False,
        plain=False,
        last_run=last_run,
        counts=[("claude", 7), ("chatgpt", 3)],
    )

    assert result["title"] == "Polylogue"
    assert result["lines"] == [
        "Archive: /data/archive",
        "Render: /data/archive/rendered",
        "Sources: inbox",
        "Last run: run-123 (2025-01-15T12:30:45Z)",
        "Health: OK",
    ]
    assert "Archive:" in result["console"]
    assert "10 conversations" in result["console"]
    assert "claude:" in result["console"]
    result["mock_cached"].assert_called_once()
    result["mock_get_health"].assert_not_called()


@pytest.mark.parametrize(
    ("plain", "status", "expected_indicator"),
    [
        (False, "ok", "[green]✓[/green]"),
        (False, "warning", "[yellow]![/yellow]"),
        (False, "error", "[red]✗[/red]"),
        (True, "ok", "OK"),
        (True, "warning", "WARN"),
        (True, "error", "ERR"),
    ],
    ids=["rich-ok", "rich-warning", "rich-error", "plain-ok", "plain-warning", "plain-error"],
)
def test_print_summary_verbose_health_matrix(config: Config, plain: bool, status: str, expected_indicator: str) -> None:
    report = _health_report(cached=True, age_seconds=30, checks=[_check("database", status, "detail")])
    result = _run_summary(config, verbose=True, plain=plain, health=report, counts=[])

    assert result["lines"][:4] == [
        "Archive: /data/archive",
        "Render: /data/archive/rendered",
        "Sources: inbox",
        "Last run: none",
    ]
    assert result["lines"][4] == "Health (cached=True, age=30s)"
    assert result["lines"][5] == f"  {expected_indicator} database: detail"
    result["mock_get_health"].assert_called_once()
    result["mock_cached"].assert_not_called()


def test_print_summary_verbose_analytics_deep_dive_contract(config: Config) -> None:
    metrics = [
        _metric(
            "claude",
            7,
            message_count=70,
            avg_messages_per_conversation=10.0,
            avg_user_words=15,
            avg_assistant_words=20,
            tool_use_count=5,
            tool_use_percentage=71.4,
            thinking_count=4,
            thinking_percentage=57.1,
        ),
        _metric(
            "chatgpt",
            3,
            message_count=21,
            avg_messages_per_conversation=7.0,
            avg_user_words=9,
            avg_assistant_words=13,
        ),
    ]
    result = _run_summary(
        config,
        verbose=True,
        plain=False,
        health=_health_report(cached=None, age_seconds=None, checks=[]),
        counts=[("claude", 7), ("chatgpt", 3)],
        metrics=metrics,
    )

    assert result["lines"][4] == "Health"
    assert "Archive:" in result["console"]
    assert "10 conversations" in result["console"]
    assert "Deep Dive:" in result["console"]
    assert "claude" in result["console"]
    assert "Messages: 70" in result["console"]
    assert "Tool Use: 5" in result["console"]
    assert "Thinking: 4" in result["console"]
    assert "chatgpt" in result["console"]


def test_print_summary_analytics_errors_are_silent(config: Config) -> None:
    result = _run_summary(
        config,
        verbose=False,
        plain=True,
        analytics_error=RuntimeError("boom"),
    )

    assert result["title"] == "Polylogue"
    assert result["lines"][-1] == "Health: OK"
    assert "Archive:" not in result["console"]


def test_print_summary_omits_deep_dive_when_no_verbose_metrics(config: Config) -> None:
    result = _run_summary(
        config,
        verbose=False,
        plain=True,
        counts=[("claude", 1)],
        metrics=[_metric("claude", 1, message_count=2, avg_messages_per_conversation=2.0)],
    )

    assert "Archive:" in result["console"]
    assert "Deep Dive:" not in result["console"]
