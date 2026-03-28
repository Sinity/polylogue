"""Pinned static-site regressions and exact-output smokes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from polylogue.cli.commands.site import site_command
from polylogue.cli.types import AppEnv
from polylogue.lib.models import ConversationSummary
from polylogue.services import build_runtime_services
from polylogue.site.builder import SiteBuilder, SiteConfig


async def _empty_messages(*args, **kwargs):
    if False:  # pragma: no cover
        yield None


async def _iter_messages(payloads):
    for index, (role, text) in enumerate(payloads, start=1):
        yield MagicMock(
            id=f"msg-{index}",
            message_id=f"msg-{index}",
            role=role,
            text=text,
            sort_key=f"2024-01-15T10:{index:02d}:00+00:00",
        )


async def _summary_pages(*pages):
    for page in pages:
        if page:
            yield page


def _configure_summary_pages(repo: AsyncMock, *pages) -> None:
    repo.iter_summary_pages = MagicMock(
        side_effect=lambda *args, **kwargs: _summary_pages(*pages)
    )
    repo.iter_messages = _empty_messages


async def _collect_indexes(builder: SiteBuilder):
    return [conversation async for conversation in builder._iter_conversation_indexes()]


def _site_env(*, backend: AsyncMock, repository: AsyncMock) -> AppEnv:
    ui = MagicMock()
    ui.plain = True
    ui.console = MagicMock()
    ui.summary = MagicMock()
    repository.backend = backend
    return AppEnv(ui=ui, services=build_runtime_services(backend=backend, repository=repository))


def _summary(
    conversation_id: str,
    *,
    provider: str = "claude",
    title: str | None = "Test Conversation",
    summary: str | None = None,
) -> ConversationSummary:
    return ConversationSummary(
        id=conversation_id,
        provider=provider,
        title=title,
        created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
        metadata={"summary": summary} if summary is not None else {},
    )


def test_build_index_requests_all_summaries_without_silent_cap(tmp_path):
    """Regression: root archive scan must use paged summary iteration without a silent cap."""
    backend = AsyncMock()
    backend.get_message_counts_batch.return_value = {"test-conv-001": 3}
    repository = AsyncMock()
    _configure_summary_pages(repository, [_summary("test-conv-001")])
    builder = SiteBuilder(
        output_dir=tmp_path / "site",
        config=SiteConfig(title="Test Archive"),
        backend=backend,
        repository=repository,
    )

    asyncio.run(_collect_indexes(builder))

    repository.iter_summary_pages.assert_called_once_with(page_size=500, provider=None)


def test_conversation_index_no_source_attribute_reference(tmp_path):
    """Regression: ConversationIndex must not touch the removed conv.source attribute."""
    backend = AsyncMock()
    backend.get_message_counts_batch.return_value = {"test-conv-001": 3}
    repository = AsyncMock()
    _configure_summary_pages(repository, [_summary("test-conv-001")])
    builder = SiteBuilder(
        output_dir=tmp_path / "site",
        config=SiteConfig(title="Test Archive"),
        backend=backend,
        repository=repository,
    )

    indexes = asyncio.run(_collect_indexes(builder))

    assert len(indexes) == 1
    index = indexes[0]
    assert index.id == "test-conv-001"
    assert index.title == "Test Conversation"
    assert index.provider == "claude"
    assert not hasattr(index, "source")
    assert index.message_count == 3


def test_site_command_no_rich_markup_in_output(workspace_env):
    """Regression: plain CLI output must not leak Rich markup tags."""
    runner = CliRunner()
    output_dir = workspace_env["archive_root"] / "site"
    backend = AsyncMock()
    backend.get_message_counts_batch.return_value = {}
    repository = AsyncMock()
    _configure_summary_pages(repository, [])
    env = _site_env(backend=backend, repository=repository)

    result = runner.invoke(site_command, ["--output", str(output_dir)], obj=env)

    assert result.exit_code == 0
    for token in ("[bold]", "[/bold]", "[green]", "[/green]", "[red]", "[/red]"):
        assert token not in result.output


def test_site_command_error_handling(workspace_env):
    """Pinned error path: site command should abort cleanly when the builder fails."""
    runner = CliRunner()
    output_dir = workspace_env["archive_root"] / "site"
    env = AppEnv(ui=MagicMock())

    with patch("polylogue.site.builder.SiteBuilder.build", side_effect=RuntimeError("Database error")):
        result = runner.invoke(site_command, ["--output", str(output_dir)], obj=env)

    assert result.exit_code != 0
    assert "Error building site: Database error" in result.output


def test_build_generates_valid_html(tmp_path):
    """Pinned smoke: generated HTML keeps the basic template skeleton intact."""
    backend = AsyncMock()
    backend.get_message_counts_batch.return_value = {"conv-123": 1}
    repository = AsyncMock()
    _configure_summary_pages(repository, [_summary("conv-123", provider="test", title="Test")])
    builder = SiteBuilder(
        output_dir=tmp_path / "site",
        config=SiteConfig(title="Test Archive", include_dashboard=True),
        backend=backend,
        repository=repository,
    )

    builder.build()

    index_html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    dashboard_html = (tmp_path / "site" / "dashboard.html").read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in index_html
    assert "Test Archive" in index_html
    assert "Test" in index_html
    assert "<!DOCTYPE html>" in dashboard_html
    assert "Dashboard" in dashboard_html or "Archive" in dashboard_html


def test_build_conversation_page_keeps_tail_messages(tmp_path):
    """Regression: conversation pages must include messages beyond the old 500-message cap."""
    summary = _summary("conv-123", title="Long Conversation")
    payloads = [("user" if index % 2 == 0 else "assistant", f"message {index}") for index in range(501)]
    backend = AsyncMock()
    backend.get_message_counts_batch.return_value = {"conv-123": 501}
    repository = AsyncMock()
    repository.iter_messages = lambda *args, **kwargs: _iter_messages(payloads)
    _configure_summary_pages(repository, [summary])
    repository.iter_messages = lambda *args, **kwargs: _iter_messages(payloads)

    builder = SiteBuilder(
        output_dir=tmp_path / "site",
        config=SiteConfig(title="Test Archive"),
        backend=backend,
        repository=repository,
    )
    builder.build()

    conversation_html = next((tmp_path / "site").rglob("conversation.html")).read_text(encoding="utf-8")
    assert "message 0" in conversation_html
    assert "message 500" in conversation_html


def test_build_conversation_page_preserves_long_message_bodies(tmp_path):
    """Regression: conversation pages must not silently truncate long message bodies."""
    summary = _summary("conv-456", title="Long Message Conversation")
    long_text = ("abcdef " * 900) + "tail-marker"
    backend = AsyncMock()
    backend.get_message_counts_batch.return_value = {"conv-456": 1}
    repository = AsyncMock()
    _configure_summary_pages(repository, [summary])
    repository.iter_messages = lambda *args, **kwargs: _iter_messages([("assistant", long_text)])

    builder = SiteBuilder(
        output_dir=tmp_path / "site",
        config=SiteConfig(title="Test Archive"),
        backend=backend,
        repository=repository,
    )
    builder.build()

    conversation_html = next((tmp_path / "site").rglob("conversation.html")).read_text(encoding="utf-8")
    assert "tail-marker" in conversation_html
    assert "[... truncated ...]" not in conversation_html
