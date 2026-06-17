"""Comprehensive end-to-end workflow tests.

CRITICAL GAP: Only 1 e2e test existed before this file.

Tests cover complete workflows:
- Import → Store → Query → Render → API for each provider
- Incremental sync (add/update/delete)
- Multi-source concurrent sync
- Error recovery and partial success
- Search accuracy validation
- Format conversion workflows
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import pytest

from polylogue.api import Polylogue
from polylogue.config import Config, Source
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

pytestmark = pytest.mark.slow

WorkflowRepos: TypeAlias = tuple[Config, SessionRepository, SessionRepository, Path, Path]
ProviderName: TypeAlias = Literal["chatgpt", "claude-ai", "claude-code", "codex", "gemini"]
RenderFormat: TypeAlias = Literal["markdown", "html"]


class SyntheticSourceFactory(Protocol):
    def __call__(
        self,
        provider: str,
        count: int = 1,
        messages_per_session: range = range(4, 12),
        seed: int = 42,
    ) -> Source: ...


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
async def temp_config_and_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[WorkflowRepos, None]:
    """Create temporary config and storage repositories for testing."""
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root = tmp_path / "render"
    render_root.mkdir(parents=True, exist_ok=True)

    # The split-file backend, the config archive root, and the db_path used by
    # update_index/search must all point at the same archive. db_path is the
    # backend's index tier under archive_root (not a standalone test.db).
    db_path = archive_root / "index.db"

    # Create minimal config
    config = Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=[],
    )

    # Create backend and repositories. SQLiteBackend bootstraps the full split
    # archive (source/index/embeddings/user/ops) under archive_root.
    backend = SQLiteBackend(db_path=db_path)
    storage_repo = SessionRepository(backend=backend)
    conv_repo = SessionRepository(backend=backend)

    yield config, storage_repo, conv_repo, archive_root, db_path


@pytest.fixture
def chatgpt_sample_source(synthetic_source: SyntheticSourceFactory) -> Source:
    """Source with synthetic ChatGPT data."""
    return synthetic_source("chatgpt")


@pytest.fixture
def gemini_sample_source(synthetic_source: SyntheticSourceFactory) -> Source:
    """Source with synthetic Gemini data."""
    return synthetic_source("gemini")


# =============================================================================
# PER-PROVIDER WORKFLOW TESTS (4 tests parametrized)
# =============================================================================


@pytest.mark.parametrize(
    "provider",
    [
        "chatgpt",
        "claude-ai",
        "claude-code",
        "codex",
        "gemini",
    ],
)
async def test_full_workflow_per_provider(
    provider: ProviderName, synthetic_source: SyntheticSourceFactory, temp_config_and_repo: WorkflowRepos
) -> None:
    """Import → Store → Query → Render for each provider."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    source = synthetic_source(provider, count=1, messages_per_session=range(4, 12))

    # 1. IMPORT: Run ingestion
    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )
    parse_result = await service.parse_sources([source])

    # Build FTS index for search tests (INDEX stage of pipeline)
    from polylogue.storage.index import update_index_for_sessions
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as conn:
        update_index_for_sessions(list(parse_result.processed_ids), conn)

    # Verify import
    assert parse_result.counts["sessions"] > 0, f"No sessions imported from {provider}"
    assert parse_result.counts["messages"] > 0, f"No messages imported from {provider}"

    # 2. STORE: Query back from database
    from polylogue.storage.search import search_messages

    convs = await conv_repo.list()
    expected_origin = origin_from_provider(Provider.from_string(provider)).value
    convs = [c for c in convs if str(c.origin) == expected_origin]
    assert len(convs) > 0, f"No sessions found for {provider}"

    # 3. QUERY: Verify session structure
    conv = convs[0]
    assert conv.id is not None
    assert len(conv.messages) > 0
    assert any(m.text for m in conv.messages), "All messages are empty"

    # 4. RENDER: Generate markdown output
    from polylogue.rendering.formatting import format_session

    markdown = format_session(conv, "markdown", None)
    assert len(markdown) > 0
    # The archive renderer renders each content block separately, so a
    # multi-block message's flattened ``text`` (parts joined with blank lines)
    # is not a verbatim substring. Assert on the first line of a message, which
    # is the first block's leading text and renders verbatim.
    first_lines = [m.text.strip().splitlines()[0] for m in conv.messages if m.text and m.text.strip()]
    assert any(line and line in markdown for line in first_lines), "No message content found in rendered markdown"

    # 5. SEARCH: Full-text search works
    first_text = next((message.text for message in conv.messages if message.text), None)
    if first_text:
        first_words = first_text.split()[:3]
        search_term = " ".join(first_words)
        search_result = search_messages(
            search_term,
            archive_root=archive_root,
            db_path=db_path,
        )
        found_ids = {r.session_id for r in search_result.hits}
        assert conv.id in found_ids, f"Search failed for '{search_term}'"


# =============================================================================
# RENDER FORMAT TESTS (2 tests parametrized)
# =============================================================================


@pytest.mark.parametrize("format", ["markdown", "html"])
async def test_render_formats(
    format: RenderFormat, temp_config_and_repo: WorkflowRepos, chatgpt_sample_source: Source
) -> None:
    """Verify each output format works end-to-end."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Import data
    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )
    await service.parse_sources([chatgpt_sample_source])

    # Get session
    convs = await conv_repo.list()
    convs = [c for c in convs if c.origin == "chatgpt-export"]
    assert len(convs) > 0

    output = ""
    if format == "markdown":
        from polylogue.rendering.formatting import format_session

        output = format_session(convs[0], "markdown", None)
        assert "#" in output or "**" in output or "*" in output
    elif format == "html":
        from polylogue.rendering.renderers.html import render_session_html

        output = render_session_html(convs[0])
        assert "<html" in output.lower() or "<!doctype html" in output.lower()

    # Verify content present (at least one message should be there)
    assert any(m.text in output for m in convs[0].messages if m.text), "No message text found in output"


# =============================================================================
# INCREMENTAL SYNC TESTS (3 tests)
# =============================================================================


async def test_incremental_sync_no_duplicates(
    temp_config_and_repo: WorkflowRepos, chatgpt_sample_source: Source
) -> None:
    """Syncing same source twice doesn't create duplicates."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )

    # First sync
    result1 = await service.parse_sources([chatgpt_sample_source])
    count1 = result1.counts["sessions"]

    # Second sync (should deduplicate)
    result2 = await service.parse_sources([chatgpt_sample_source])
    count2 = result2.counts["sessions"]

    # Second sync should add 0 sessions
    assert count2 == 0, f"Expected 0 new convs, got {count2}"

    # Total count should match first sync
    all_convs = await conv_repo.list()
    assert len(all_convs) == count1


async def test_incremental_sync_with_updates(temp_config_and_repo: WorkflowRepos) -> None:
    """Modified sessions are updated, not duplicated."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Create initial source with 1 session
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        conv_before_update: JSONDocument = {
            "id": "test-conv-1",
            "title": "Version 1",
            "mapping": {
                "node1": {
                    "id": "node1",
                    "message": {
                        "id": "msg1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Original question"]},
                    },
                    "children": ["node2"],
                },
                "node2": {
                    "id": "node2",
                    "parent": "node1",
                    "message": {
                        "id": "msg2",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Original answer"]},
                    },
                },
            },
        }
        json.dump(conv_before_update, f)
        source_path_before_update = Path(f.name)

    try:
        # First sync
        source_before_update = Source(name="test", path=source_path_before_update)
        service = ParsingService(
            repository=storage_repo,
            archive_root=archive_root,
            config=config,
        )
        await service.parse_sources([source_before_update])

        # Modify session
        with open(source_path_before_update, "w") as f:
            conv_after_update: JSONDocument = {
                "id": "test-conv-1",
                "title": "Updated version",
                "mapping": {
                    "node1": {
                        "id": "node1",
                        "message": {
                            "id": "msg1",
                            "author": {"role": "user"},
                            "content": {"parts": ["Original question"]},
                        },
                        "children": ["node2"],
                    },
                    "node2": {
                        "id": "node2",
                        "parent": "node1",
                        "message": {
                            "id": "msg2",
                            "author": {"role": "assistant"},
                            "content": {"parts": ["Updated answer"]},
                        },
                    },
                },
            }
            json.dump(conv_after_update, f)

        # Second sync
        await service.parse_sources([source_before_update])

        # Should still have 1 session
        all_convs = await conv_repo.list()
        assert len(all_convs) == 1

        # Should have updated content
        conv = all_convs[0]
        assert conv.title == "Updated version"
        assert "Updated answer" in [m.text for m in conv.messages]

    finally:
        source_path_before_update.unlink()


async def test_sync_handles_deleted_sessions(temp_config_and_repo: WorkflowRepos) -> None:
    """Sessions removed from source are NOT deleted from archive.

    This is expected behavior - archive is append-only.
    """
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Create source with 2 sessions
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        conv1 = {
            "id": "conv-1",
            "title": "First",
            "mapping": {
                "n1": {
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Q1"]},
                    }
                }
            },
        }
        json.dump(conv1, f)
        path1 = Path(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        conv2 = {
            "id": "conv-2",
            "title": "Second",
            "mapping": {
                "n1": {
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Q2"]},
                    }
                }
            },
        }
        json.dump(conv2, f)
        path2 = Path(f.name)

    try:
        # Sync both
        service = ParsingService(
            repository=storage_repo,
            archive_root=archive_root,
            config=config,
        )
        await service.parse_sources([Source(name="s1", path=path1), Source(name="s2", path=path2)])

        # Should have 2 sessions
        assert len(await conv_repo.list()) == 2

        # Remove second source, sync again
        await service.parse_sources([Source(name="s1", path=path1)])

        # Should STILL have 2 sessions (archive is append-only)
        assert len(await conv_repo.list()) == 2

    finally:
        path1.unlink()
        path2.unlink()


# =============================================================================
# MULTI-SOURCE TESTS (2 tests)
# =============================================================================


async def test_multi_source_concurrent_sync(
    temp_config_and_repo: WorkflowRepos, chatgpt_sample_source: Source, gemini_sample_source: Source
) -> None:
    """Multiple sources can sync concurrently."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    sources = [chatgpt_sample_source, gemini_sample_source]

    # Sync all sources
    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )
    await service.parse_sources(sources)

    # Should have sessions from both
    all_convs = await conv_repo.list()
    assert len(all_convs) >= 2

    # Each provider should be present
    providers = {c.origin for c in all_convs}
    assert len(providers) >= 2


async def test_multi_source_isolated_namespaces(temp_config_and_repo: WorkflowRepos) -> None:
    """Each source can have different sessions."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Create 2 sources with different sessions
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        conv = {
            "id": "conv-1",
            "title": "Source 1 Conv",
            "mapping": {
                "n1": {
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["From source 1"]},
                    }
                }
            },
        }
        json.dump(conv, f)
        path1 = Path(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        conv = {
            "id": "conv-2",
            "title": "Source 2 Conv",
            "mapping": {
                "n1": {
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["From source 2"]},
                    }
                }
            },
        }
        json.dump(conv, f)
        path2 = Path(f.name)

    try:
        service = ParsingService(
            repository=storage_repo,
            archive_root=archive_root,
            config=config,
        )
        await service.parse_sources(
            [
                Source(name="source1", path=path1),
                Source(name="source2", path=path2),
            ]
        )

        # Should have 2 sessions from different sources
        all_convs = await conv_repo.list()
        assert len(all_convs) == 2

        titles = {c.title for c in all_convs}
        assert "Source 1 Conv" in titles
        assert "Source 2 Conv" in titles

    finally:
        path1.unlink()
        path2.unlink()


# =============================================================================
# ERROR RECOVERY TESTS (3 tests)
# =============================================================================


async def test_sync_with_malformed_file_skips_gracefully(temp_config_and_repo: WorkflowRepos) -> None:
    """Malformed files are skipped, don't crash entire sync."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Create invalid JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{invalid json")
        bad_path = Path(f.name)

    try:
        source = Source(name="bad", path=bad_path)
        service = ParsingService(
            repository=storage_repo,
            archive_root=archive_root,
            config=config,
        )

        # Should not crash
        result = await service.parse_sources([source])

        # Should report 0 sessions
        assert result.counts["sessions"] == 0

    finally:
        bad_path.unlink()


async def test_sync_with_missing_file_reports_error(temp_config_and_repo: WorkflowRepos) -> None:
    """Missing source files are reported, don't crash."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    source = Source(name="missing", path=Path("/nonexistent/file.json"))
    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )

    # Should not crash
    result = await service.parse_sources([source])

    # Should report 0 sessions
    assert result.counts["sessions"] == 0


async def test_sync_partial_success_with_mixed_sources(
    temp_config_and_repo: WorkflowRepos, chatgpt_sample_source: Source
) -> None:
    """If some sources fail, successful ones still sync."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Mix good and bad sources
    sources = [
        chatgpt_sample_source,
        Source(name="bad", path=Path("/nonexistent.json")),
    ]

    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )
    result = await service.parse_sources(sources)

    # Good source should succeed
    assert result.counts["sessions"] > 0

    # Should have sessions from good source
    convs = await conv_repo.list()
    convs = [c for c in convs if c.origin == "chatgpt-export"]
    assert len(convs) > 0


# =============================================================================
# SEARCH ACCURACY TESTS (2 tests)
# =============================================================================


async def test_search_accuracy_basic_terms(temp_config_and_repo: WorkflowRepos, chatgpt_sample_source: Source) -> None:
    """Search returns correct sessions for basic queries."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )
    await service.parse_sources([chatgpt_sample_source])

    # Build search index
    from polylogue.storage.index import rebuild_index
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as conn:
        rebuild_index(conn)

    # Get all sessions
    all_convs = await conv_repo.list()
    assert len(all_convs) > 0

    # Find a message with enough words for a meaningful search query
    target_conv = all_convs[0]
    words = []
    for msg in target_conv.messages:
        if msg.text:
            words = msg.text.split()[:5]
            if len(words) >= 3:
                break

    if len(words) < 3:
        pytest.skip("No message with 3+ words for meaningful search test")

    # Search for these words
    search_term = " ".join(words[:3])
    from polylogue.storage.search import search_messages

    result = search_messages(search_term, archive_root=archive_root, db_path=db_path)
    result_ids = {r.session_id for r in result.hits}
    assert target_conv.id in result_ids, f"Failed to find session with '{search_term}'"


async def test_search_with_special_characters(temp_config_and_repo: WorkflowRepos) -> None:
    """Search handles special FTS5 characters correctly."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Create session with special characters
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        conv = {
            "id": "special-chars",
            "title": "Special Test",
            "mapping": {
                "n1": {
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Question with * and ? and ()"]},
                    }
                },
                "n2": {
                    "parent": "n1",
                    "message": {
                        "id": "m2",
                        "author": {"role": "assistant"},
                        "content": {"parts": ['Answer with "quotes" and {braces}']},
                    },
                },
            },
        }
        json.dump(conv, f)
        path = Path(f.name)

    try:
        service = ParsingService(
            repository=storage_repo,
            archive_root=archive_root,
            config=config,
        )
        await service.parse_sources([Source(name="test", path=path)])

        # Build search index
        from polylogue.storage.index import rebuild_index
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(db_path) as conn:
            rebuild_index(conn)

        from polylogue.storage.search import search_messages

        # Search with special chars (should be escaped)
        result1 = search_messages("quotes", archive_root=archive_root, db_path=db_path)
        assert len(result1.hits) > 0

        result2 = search_messages("braces", archive_root=archive_root, db_path=db_path)
        assert len(result2.hits) > 0

        # FTS5 operators should be escaped and not cause errors
        # (*, ?, OR, AND, NOT, etc.)
        search_messages("*", archive_root=archive_root, db_path=db_path)
        # Should not crash

    finally:
        path.unlink()


# =============================================================================
# API INGEST TESTS
# =============================================================================


async def test_daemon_owned_api_ingest_lands_source_session(
    workspace_env: dict[str, Path], chatgpt_sample_source: Source
) -> None:
    """The daemon-owned API ingest path lands source sessions in the archive."""
    from polylogue.config import get_config

    config = get_config()
    config.sources = [chatgpt_sample_source]

    async with Polylogue(archive_root=config.archive_root, db_path=config.db_path) as polylogue:
        result = await polylogue.parse_sources([chatgpt_sample_source])
        stored = await polylogue.list_sessions(limit=5)

    assert result is not None
    assert result.counts["sessions"] > 0
    assert len(stored) == 1
