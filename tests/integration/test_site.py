"""Pinned static-site regressions and exact-output smokes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from polylogue.archive.models import ConversationSummary
from polylogue.cli.click_app import cli
from polylogue.site.builder import SiteBuilder, SiteConfig
from polylogue.site.models import ConversationIndex
from polylogue.types import ConversationId, Provider
from tests.infra.storage_records import ConversationBuilder

WorkspaceEnv = dict[str, Path]
MessagePayload = tuple[str, str]


async def _empty_messages(*args: object, **kwargs: object) -> AsyncIterator[None]:
    if False:  # pragma: no cover
        yield None


async def _iter_messages(payloads: Sequence[MessagePayload]) -> AsyncIterator[MagicMock]:
    for index, (role, text) in enumerate(payloads, start=1):
        yield MagicMock(
            id=f"msg-{index}",
            message_id=f"msg-{index}",
            role=role,
            text=text,
            sort_key=f"2024-01-15T10:{index:02d}:00+00:00",
        )


async def _summary_pages(*pages: Sequence[ConversationSummary]) -> AsyncIterator[Sequence[ConversationSummary]]:
    for page in pages:
        if page:
            yield page


def _configure_summary_pages(repo: AsyncMock, *pages: Sequence[ConversationSummary]) -> None:
    repo.iter_summary_pages = MagicMock(side_effect=lambda *args, **kwargs: _summary_pages(*pages))
    repo.iter_messages = _empty_messages


def _make_backend() -> AsyncMock:
    backend = AsyncMock()
    backend.queries = MagicMock()
    backend.queries.get_message_counts_batch = AsyncMock(return_value={})
    return backend


async def _collect_indexes(builder: SiteBuilder) -> list[ConversationIndex]:
    return [conversation async for conversation in builder._iter_conversation_indexes()]


def _summary(
    conversation_id: str,
    *,
    provider: str = "claude-ai",
    title: str | None = "Test Conversation",
    summary: str | None = None,
) -> ConversationSummary:
    return ConversationSummary(
        id=ConversationId(conversation_id),
        provider=Provider.from_string(provider),
        title=title,
        created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
        metadata={"summary": summary} if summary is not None else {},
    )


def test_build_index_requests_all_summaries_without_silent_cap(tmp_path: Path) -> None:
    """Regression: root archive scan must use paged summary iteration without a silent cap."""
    backend = _make_backend()
    backend.queries.get_message_counts_batch.return_value = {"test-conv-001": 3}
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


def test_conversation_index_uses_summary_provider_metadata(tmp_path: Path) -> None:
    """Conversation index construction uses summary provider metadata."""
    backend = _make_backend()
    backend.queries.get_message_counts_batch.return_value = {"test-conv-001": 3}
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
    assert index.provider == "claude-ai"
    assert index.message_count == 3


def test_run_site_no_rich_markup_in_output(workspace_env: WorkspaceEnv) -> None:
    """Regression: plain CLI output must not leak Rich markup tags."""
    runner = CliRunner()
    output_dir = workspace_env["archive_root"] / "site"

    result = runner.invoke(
        cli,
        ["--plain", "run", "site", "--output", str(output_dir)],
    )

    assert result.exit_code == 0
    for token in ("[bold]", "[/bold]", "[green]", "[/green]", "[red]", "[/red]"):
        assert token not in result.output


def test_run_site_builds_browsable_static_site(workspace_env: WorkspaceEnv) -> None:
    """Smoke: `polylogue run site` writes linked HTML pages from a seeded archive."""
    db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
    (
        ConversationBuilder(db_path, "site-smoke-1")
        .provider("chatgpt")
        .title("Smoke Conversation")
        .updated_at("2026-03-22T12:00:00+00:00")
        .add_message("m1", role="user", text="site smoke body")
        .save()
    )

    runner = CliRunner()
    output_dir = workspace_env["archive_root"] / "site-smoke"
    result = runner.invoke(
        cli,
        [
            "--plain",
            "run",
            "site",
            "--output",
            str(output_dir),
            "--title",
            "Smoke Archive",
            "--search-provider",
            "lunr",
        ],
    )

    assert result.exit_code == 0, result.output
    index_html = (output_dir / "index.html").read_text(encoding="utf-8")
    dashboard_html = (output_dir / "dashboard.html").read_text(encoding="utf-8")
    conversation_path = output_dir / "chatgpt" / "site-smoke-1" / "conversation.html"
    conversation_html = conversation_path.read_text(encoding="utf-8")
    search_index = json.loads((output_dir / "search-index.json").read_text(encoding="utf-8"))

    assert "<!DOCTYPE html>" in index_html
    assert "Smoke Archive" in index_html
    assert "Smoke Conversation" in index_html
    assert "dashboard.html" in index_html
    assert "<!DOCTYPE html>" in dashboard_html
    assert "Smoke Archive" in dashboard_html
    assert "<!DOCTYPE html>" in conversation_html
    assert "site smoke body" in conversation_html
    assert search_index == [
        {
            "id": "site-smoke-1",
            "title": "Smoke Conversation",
            "provider": "chatgpt",
            "preview": "",
            "path": "chatgpt/site-smoke-1/conversation.html",
        }
    ]


def test_run_site_error_handling(workspace_env: WorkspaceEnv) -> None:
    """Pinned error path: run site stage should record error when the builder fails."""
    runner = CliRunner()
    output_dir = workspace_env["archive_root"] / "site"

    with patch("polylogue.site.builder.SiteBuilder.build", side_effect=RuntimeError("Database error")):
        result = runner.invoke(
            cli,
            ["--plain", "run", "site", "--output", str(output_dir)],
        )

    # The run pipeline catches site errors via SiteStageOutcome.error
    # and still completes (exit_code 0) — the error is logged, not raised.
    assert result.exit_code == 0


def test_build_generates_valid_html(tmp_path: Path) -> None:
    """Pinned smoke: generated HTML keeps the basic template skeleton intact."""
    backend = _make_backend()
    backend.queries.get_message_counts_batch.return_value = {"conv-123": 1}
    repository = AsyncMock()
    _configure_summary_pages(repository, [_summary("conv-123", provider="chatgpt", title="Test")])
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


def test_build_conversation_page_keeps_tail_messages(tmp_path: Path) -> None:
    """Regression: conversation pages must include messages beyond the old 500-message cap."""
    summary = _summary("conv-123", title="Long Conversation")
    payloads: list[MessagePayload] = [
        ("user" if index % 2 == 0 else "assistant", f"message {index}") for index in range(501)
    ]
    backend = _make_backend()
    backend.queries.get_message_counts_batch.return_value = {"conv-123": 501}
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


def test_build_conversation_page_preserves_long_message_bodies(tmp_path: Path) -> None:
    """Regression: conversation pages must not silently truncate long message bodies."""
    summary = _summary("conv-456", title="Long Message Conversation")
    long_text = ("abcdef " * 900) + "tail-marker"
    backend = _make_backend()
    backend.queries.get_message_counts_batch.return_value = {"conv-456": 1}
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
