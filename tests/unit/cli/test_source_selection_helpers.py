"""Focused tests for CLI source-selection and path helpers."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.cli import helpers
from polylogue.cli.types import AppEnv
from polylogue.config import Config, Source
from polylogue.services import build_runtime_services


@pytest.fixture
def helpers_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    config_dir = tmp_path / "config"
    for directory in (data_dir, state_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_dir))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    return {
        "data_dir": data_dir,
        "state_dir": state_dir,
        "config_dir": config_dir,
    }


@pytest.mark.parametrize(
    ("command", "message", "expected"),
    [("test_cmd", "something broke", "test_cmd: something broke"), ("test_cmd", "", "test_cmd:")],
)
def test_fail_raises_system_exit(command: str, message: str, expected: str) -> None:
    with pytest.raises(SystemExit, match=expected):
        helpers.fail(command, message)


def test_source_state_path_defaults_without_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    result = helpers.source_state_path()
    assert result == tmp_path / ".local" / "state" / "polylogue" / "last-source.json"


def test_source_state_path_uses_xdg_state_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", "/custom/state")
    result = helpers.source_state_path()
    assert result == Path("/custom/state/polylogue/last-source.json")


def test_load_last_source_returns_none_for_missing(helpers_workspace: dict[str, Path]) -> None:
    assert helpers.load_last_source() is None


def test_load_last_source_returns_none_for_invalid_json(helpers_workspace: dict[str, Path]) -> None:
    path = helpers.source_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")
    assert helpers.load_last_source() is None


def test_save_and_load_last_source_roundtrip(helpers_workspace: dict[str, Path]) -> None:
    helpers.save_last_source("chatgpt")
    path = helpers.source_state_path()
    assert json.loads(path.read_text(encoding="utf-8")) == {"source": "chatgpt"}
    assert helpers.load_last_source() == "chatgpt"


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        (("chatgpt",), ["chatgpt"]),
        (("chatgpt", "claude-ai"), {"chatgpt", "claude-ai"}),
        (("chatgpt", "chatgpt"), ["chatgpt"]),
        ((), None),
    ],
)
def test_resolve_sources_valid_cases(sources: tuple[str, ...], expected: list[str] | set[str] | None) -> None:
    config = MagicMock()
    config.sources = [
        Source(name="chatgpt", path=Path("/data")),
        Source(name="claude-ai", path=Path("/data2")),
    ]
    result = helpers.resolve_sources(config, sources, "test_cmd")
    if isinstance(expected, set):
        assert set(result or []) == expected
    else:
        assert result == expected


@pytest.mark.parametrize("sources", [("unknown",), ("chatgpt", "unknown")])
def test_resolve_sources_rejects_unknown_sources(sources: tuple[str, ...]) -> None:
    config = MagicMock()
    config.sources = [Source(name="chatgpt", path=Path("/data"))]
    with pytest.raises(SystemExit):
        helpers.resolve_sources(config, sources, "test_cmd")


def test_resolve_sources_expands_last(helpers_workspace: dict[str, Path]) -> None:
    helpers.save_last_source("chatgpt")
    config = MagicMock()
    config.sources = [Source(name="chatgpt", path=Path("/data"))]
    assert helpers.resolve_sources(config, ("last",), "test_cmd") == ["chatgpt"]


def test_resolve_sources_rejects_last_without_saved_source(helpers_workspace: dict[str, Path]) -> None:
    config = MagicMock()
    config.sources = []
    with pytest.raises(SystemExit):
        helpers.resolve_sources(config, ("last",), "test_cmd")


def test_resolve_sources_rejects_last_combined_with_other_sources(helpers_workspace: dict[str, Path]) -> None:
    config = MagicMock()
    config.sources = [Source(name="chatgpt", path=Path("/data"))]
    with pytest.raises(SystemExit):
        helpers.resolve_sources(config, ("last", "chatgpt"), "test_cmd")


def _config_with_sources(tmp_path: Path, names: list[str]) -> Config:
    return Config(
        archive_root=tmp_path / "archive",
        render_root=tmp_path / "render",
        sources=[Source(name=name, path=tmp_path / name) for name in names],
    )


def test_maybe_prompt_sources_returns_selected_sources_if_provided(
    helpers_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    env = MagicMock()
    env.ui.plain = True
    result = helpers.maybe_prompt_sources(env, _config_with_sources(tmp_path, ["one", "two"]), ["one"], "sync")
    assert result == ["one"]


def test_maybe_prompt_sources_returns_none_in_plain_mode(
    helpers_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    env = MagicMock()
    env.ui.plain = True
    result = helpers.maybe_prompt_sources(env, _config_with_sources(tmp_path, ["one", "two"]), None, "sync")
    assert result is None


def test_maybe_prompt_sources_skips_prompt_for_single_source(
    helpers_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    env = MagicMock()
    env.ui.plain = False
    env.ui.choose = MagicMock()
    result = helpers.maybe_prompt_sources(env, _config_with_sources(tmp_path, ["one"]), None, "sync")
    assert result is None
    env.ui.choose.assert_not_called()


def test_maybe_prompt_sources_prompts_and_persists_selection(
    helpers_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    env = MagicMock()
    env.ui.plain = False
    env.ui.choose.return_value = "two"

    result = helpers.maybe_prompt_sources(env, _config_with_sources(tmp_path, ["one", "two"]), None, "sync")

    assert result == ["two"]
    assert helpers.load_last_source() == "two"
    prompt, options = env.ui.choose.call_args.args
    assert prompt == "Select source for sync"
    assert options == ["all", "one", "two"]


def test_maybe_prompt_sources_prioritizes_saved_selection(
    helpers_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    helpers.save_last_source("two")
    env = MagicMock()
    env.ui.plain = False
    env.ui.choose.return_value = "two"

    helpers.maybe_prompt_sources(env, _config_with_sources(tmp_path, ["one", "two"]), None, "sync")

    _, options = env.ui.choose.call_args.args
    assert options[0] == "two"
    assert options[1:] == ["all", "one"]


def test_maybe_prompt_sources_rejects_empty_choice(
    helpers_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    env = MagicMock()
    env.ui.plain = False
    env.ui.choose.return_value = None
    with pytest.raises(SystemExit):
        helpers.maybe_prompt_sources(env, _config_with_sources(tmp_path, ["one", "two"]), None, "sync")


@pytest.mark.parametrize(
    "case_id",
    [
        "nonexistent-root",
        "empty-root",
        "single-markdown",
        "single-html",
        "latest-mtime-wins",
        "missing-candidate-is-skipped",
    ],
)
def test_latest_render_path_contract(tmp_path: Path, case_id: str) -> None:
    render_root = tmp_path / "render"
    expected: Path | None = None

    if case_id == "nonexistent-root":
        render_root = tmp_path / "missing"
    elif case_id == "empty-root":
        render_root.mkdir()
    elif case_id == "single-markdown":
        conv_dir = render_root / "conv1"
        conv_dir.mkdir(parents=True)
        expected = conv_dir / "conversation.md"
        expected.write_text("# Test", encoding="utf-8")
    elif case_id == "single-html":
        conv_dir = render_root / "conv1"
        conv_dir.mkdir(parents=True)
        expected = conv_dir / "conversation.html"
        expected.write_text("<html>test</html>", encoding="utf-8")
    elif case_id == "latest-mtime-wins":
        import os

        conv1 = render_root / "conv1"
        conv2 = render_root / "conv2"
        conv1.mkdir(parents=True)
        conv2.mkdir(parents=True)
        older = conv1 / "conversation.md"
        expected = conv2 / "conversation.html"
        older.write_text("old", encoding="utf-8")
        expected.write_text("new", encoding="utf-8")
        os.utime(older, (100, 100))
        os.utime(expected, (200, 200))
    elif case_id == "missing-candidate-is-skipped":
        conv_dir = render_root / "conv1"
        conv_dir.mkdir(parents=True)
        expected = conv_dir / "conversation.md"
        expected.write_text("# Existing", encoding="utf-8")
        original_rglob = Path.rglob

        def fake_rglob(self: Path, pattern: str):
            if self == render_root and pattern in {"conversation.md", "conversation.html"}:
                missing = render_root / "deleted" / pattern
                return list(original_rglob(self, pattern)) + [missing]
            return original_rglob(self, pattern)

        with patch.object(Path, "rglob", fake_rglob):
            assert helpers.latest_render_path(render_root) == expected
        return

    assert helpers.latest_render_path(render_root) == expected


# ---------------------------------------------------------------------------
# Merged from test_helpers.py (2026-03-15)
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> Config:
    return Config(
        archive_root=Path("/data/archive"),
        render_root=Path("/data/archive/rendered"),
        sources=[],
    )


def _make_env(config: Config, *, plain: bool) -> tuple[AppEnv, StringIO]:
    from rich.console import Console
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
    archive_stats=None,
    analytics_error: Exception | None = None,
):
    from polylogue.cli.helpers import print_summary
    env, buffer = _make_env(config, plain=plain)
    total_conversations = sum(count for _, count in counts or [])
    if archive_stats is None:
        archive_stats = SimpleNamespace(
            total_conversations=total_conversations,
            embedded_conversations=0,
            embedded_messages=0,
            pending_embedding_conversations=0,
            embedding_coverage=0.0,
        )
    env.repository.get_archive_stats = AsyncMock(return_value=archive_stats)
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
        counts=[("claude-ai", 7), ("chatgpt", 3)],
    )

    assert result["title"] == "Polylogue"
    assert result["lines"] == [
        "Archive: /data/archive",
        "Render: /data/archive/rendered",
        "Sources: inbox",
        "Last run: run-123 (2025-01-15T12:30:45Z)",
        "Embeddings: 0/10 convs, 0 msgs (0.0%)",
        "Health: OK",
    ]
    assert "Archive:" in result["console"]
    assert "10 conversations" in result["console"]
    assert "claude-ai:" in result["console"]
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
    assert result["lines"][4] == "Embeddings: 0/0 convs, 0 msgs (0.0%)"
    assert result["lines"][5] == "Health (cached=True, age=30s)"
    assert result["lines"][6] == f"  {expected_indicator} database: detail"
    result["mock_get_health"].assert_called_once()
    result["mock_cached"].assert_not_called()


def test_print_summary_verbose_analytics_deep_dive_contract(config: Config) -> None:
    metrics = [
        _metric(
            "claude-ai",
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
        counts=[("claude-ai", 7), ("chatgpt", 3)],
        metrics=metrics,
    )

    assert result["lines"][4] == "Embeddings: 0/10 convs, 0 msgs (0.0%)"
    assert result["lines"][5] == "Health"
    assert "Archive:" in result["console"]
    assert "10 conversations" in result["console"]
    assert "Deep Dive:" in result["console"]
    assert "claude-ai" in result["console"]
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
    assert result["lines"][-2] == "Embeddings: 0/0 convs, 0 msgs (0.0%)"
    assert "Archive:" not in result["console"]


def test_print_summary_omits_deep_dive_when_no_verbose_metrics(config: Config) -> None:
    result = _run_summary(
        config,
        verbose=False,
        plain=True,
        counts=[("claude-ai", 1)],
        metrics=[_metric("claude-ai", 1, message_count=2, avg_messages_per_conversation=2.0)],
    )

    assert "Archive:" in result["console"]
    assert "Deep Dive:" not in result["console"]


def test_print_summary_shows_pending_embeddings(config: Config) -> None:
    result = _run_summary(
        config,
        verbose=False,
        plain=True,
        counts=[("claude-ai", 7), ("chatgpt", 3)],
        archive_stats=SimpleNamespace(
            total_conversations=10,
            embedded_conversations=4,
            embedded_messages=120,
            pending_embedding_conversations=6,
            embedding_coverage=40.0,
        ),
    )

    assert "Embeddings: 4/10 convs, 120 msgs (40.0%), pending 6" in result["lines"]
