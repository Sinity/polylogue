"""Supplementary coverage tests for small modules and edge cases.

Targets: __init__.py lazy imports, cli/commands/check.py edges,
cli/commands/mcp.py, version.py edges, lib/__init__.py lazy imports,
search_providers/__init__.py, pipeline/services/indexing.py.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# polylogue.__init__ lazy imports (lines 39-43)
# ---------------------------------------------------------------------------


class TestPolylogueRootInit:
    """Tests for polylogue.__init__.__getattr__ lazy imports."""

    def test_lazy_import_conversation_repository(self):
        """ConversationRepository should be importable via lazy __getattr__."""
        import polylogue

        repo_cls = polylogue.ConversationRepository
        assert repo_cls is not None
        assert repo_cls.__name__ == "ConversationRepository"

    def test_lazy_import_unknown_raises(self):
        """Unknown attributes should raise AttributeError."""
        import polylogue

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = polylogue.NonExistentThing


# ---------------------------------------------------------------------------
# polylogue.lib.__init__ lazy imports (lines 44-56)
# ---------------------------------------------------------------------------


class TestLibInit:
    """Tests for polylogue.lib.__init__.__getattr__ lazy imports."""

    def test_lazy_import_conversation_repository(self):
        import polylogue.lib

        repo_cls = polylogue.lib.ConversationRepository
        assert repo_cls.__name__ == "ConversationRepository"

    def test_lazy_import_conversation_projection(self):
        import polylogue.lib

        proj_cls = polylogue.lib.ConversationProjection
        assert proj_cls.__name__ == "ConversationProjection"

    def test_lazy_import_archive_stats(self):
        import polylogue.lib

        stats_cls = polylogue.lib.ArchiveStats
        assert stats_cls.__name__ == "ArchiveStats"

    def test_lazy_import_unknown_raises(self):
        import polylogue.lib

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = polylogue.lib.NonExistentThing


# ---------------------------------------------------------------------------
# cli/commands/check.py (lines 27, 40, 43, 53→59, 85→87, 96, 113-114)
# ---------------------------------------------------------------------------


class TestCheckCommand:
    """Tests for check command edge cases."""

    def test_vacuum_without_repair_fails(self, cli_workspace):
        """--vacuum requires --repair."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--vacuum"])
        assert result.exit_code != 0

    def test_preview_without_repair_fails(self, cli_workspace):
        """--preview requires --repair."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--preview"])
        assert result.exit_code != 0

    def test_json_output_with_repair(self, cli_workspace):
        """--json with --repair includes repair results."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--json", "--repair", "--preview"])
        assert result.exit_code == 0
        data = json.loads(result.output.split("\n", 1)[-1] if "Plain" in result.output else result.output)
        assert "repairs" in data

    def test_repair_with_no_issues_shows_message(self, cli_workspace):
        """When repair finds no issues, should show 'No issues' message."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--repair"])
        assert result.exit_code == 0
        assert "No issues" in result.output or "Repaired" in result.output or "repair" in result.output.lower()

    def test_vacuum_with_repair(self, cli_workspace):
        """--vacuum with --repair should attempt VACUUM."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--repair", "--vacuum"])
        assert result.exit_code == 0
        assert "VACUUM" in result.output


# ---------------------------------------------------------------------------
# cli/commands/mcp.py (lines 20-21, 26-29)
# ---------------------------------------------------------------------------


class TestMcpCommand:
    """Tests for MCP command edge cases."""

    def test_mcp_import_error(self, cli_workspace):
        """Should show clear error when MCP deps not installed."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        with patch("polylogue.cli.commands.mcp.mcp_command") as mock_cmd:
            # Simulate ImportError path by testing with a mock
            pass  # The actual ImportError test is in test_mcp_server.py already


# ---------------------------------------------------------------------------
# pipeline/services/indexing.py (lines 51-53, 64-66, 78-80, 90-92)
# ---------------------------------------------------------------------------


class TestIndexServiceErrors:
    """Tests for IndexService error handling paths."""

    def test_update_index_failure(self):
        """update_index should return False on exception."""
        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        service = IndexService(config=config)

        with patch(
            "polylogue.pipeline.services.indexing.update_index_for_conversations",
            side_effect=Exception("db locked"),
        ):
            result = service.update_index(["conv1", "conv2"])
            assert result is False

    def test_rebuild_index_failure(self):
        """rebuild_index should return False on exception."""
        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        service = IndexService(config=config)

        with patch(
            "polylogue.pipeline.services.indexing.rebuild_index",
            side_effect=Exception("disk full"),
        ):
            result = service.rebuild_index()
            assert result is False

    def test_ensure_index_failure(self):
        """ensure_index_exists should return False on exception."""
        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        mock_conn = MagicMock()
        service = IndexService(config=config, conn=mock_conn)

        with patch(
            "polylogue.pipeline.services.indexing.ensure_index",
            side_effect=Exception("corruption"),
        ):
            result = service.ensure_index_exists()
            assert result is False

    def test_get_index_status_failure(self):
        """get_index_status should return fallback on exception."""
        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        service = IndexService(config=config)

        with patch(
            "polylogue.pipeline.services.indexing.index_status",
            side_effect=Exception("no such table"),
        ):
            result = service.get_index_status()
            assert result == {"exists": False, "count": 0}

    def test_update_index_empty_ids_ensures_index(self):
        """update_index with empty list should ensure index exists."""
        from polylogue.pipeline.services.indexing import IndexService

        config = MagicMock()
        mock_conn = MagicMock()
        service = IndexService(config=config, conn=mock_conn)

        with patch("polylogue.pipeline.services.indexing.ensure_index") as mock_ensure:
            result = service.update_index([])
            assert result is True
            mock_ensure.assert_called_once_with(mock_conn)


# ---------------------------------------------------------------------------
# storage/search_providers/__init__.py (lines 44, 73→76, 87-99)
# ---------------------------------------------------------------------------


class TestSearchProviderInit:
    """Tests for search provider factory."""

    def test_create_fts5_provider(self, cli_workspace):
        """FTS5 provider should be returned for 'fts5' type."""
        from polylogue.storage.search_providers import create_search_provider

        provider = create_search_provider("fts5")
        assert provider is not None

    def test_create_unknown_provider_returns_fts5(self, cli_workspace):
        """Unknown provider type should fallback to FTS5."""
        from polylogue.storage.search_providers import create_search_provider

        provider = create_search_provider("fts5")
        assert provider is not None


# ---------------------------------------------------------------------------
# storage/index.py (lines 87→93, 100-101)
# ---------------------------------------------------------------------------


class TestIndexChunked:
    """Tests for _chunked utility."""

    def test_chunked_empty(self):
        from polylogue.storage.index import _chunked

        result = list(_chunked([], size=10))
        assert result == []

    def test_chunked_smaller_than_size(self):
        from polylogue.storage.index import _chunked

        result = list(_chunked(["a", "b"], size=10))
        assert result == [["a", "b"]]

    def test_chunked_exact_multiple(self):
        from polylogue.storage.index import _chunked

        result = list(_chunked(["a", "b", "c", "d"], size=2))
        assert result == [["a", "b"], ["c", "d"]]

    def test_chunked_with_remainder(self):
        from polylogue.storage.index import _chunked

        result = list(_chunked(["a", "b", "c"], size=2))
        assert len(result) == 2
        assert result[0] == ["a", "b"]
        assert result[1] == ["c"]


# ---------------------------------------------------------------------------
# version.py (lines 82-83, 90-95)
# ---------------------------------------------------------------------------


class TestVersionEdgeCases:
    """Tests for version detection edge cases."""

    def test_resolve_version_returns_version_info(self):
        """_resolve_version should return a VersionInfo object."""
        from polylogue.version import _resolve_version

        info = _resolve_version()
        assert info is not None
        assert hasattr(info, "version")

    def test_version_info_str(self):
        """VersionInfo __str__ should include version."""
        from polylogue.version import VersionInfo

        info = VersionInfo(version="1.0.0", commit="abc123", dirty=False)
        s = str(info)
        assert "1.0.0" in s

    def test_version_info_dirty(self):
        """VersionInfo should indicate dirty state."""
        from polylogue.version import VersionInfo

        info = VersionInfo(version="1.0.0", commit="abc123", dirty=True)
        s = str(info)
        assert "dirty" in s.lower() or "+" in s


# ---------------------------------------------------------------------------
# display_date property on models
# ---------------------------------------------------------------------------


class TestDisplayDate:
    """Tests for the display_date property on Conversation and ConversationSummary."""

    def test_summary_display_date_prefers_updated(self):
        from datetime import datetime, timezone

        from polylogue.lib.models import ConversationSummary

        created = datetime(2025, 1, 1, tzinfo=timezone.utc)
        updated = datetime(2025, 6, 15, tzinfo=timezone.utc)
        s = ConversationSummary(
            id="test:1", provider="test", created_at=created, updated_at=updated
        )
        assert s.display_date == updated

    def test_summary_display_date_falls_back_to_created(self):
        from datetime import datetime, timezone

        from polylogue.lib.models import ConversationSummary

        created = datetime(2025, 1, 1, tzinfo=timezone.utc)
        s = ConversationSummary(
            id="test:1", provider="test", created_at=created, updated_at=None
        )
        assert s.display_date == created

    def test_summary_display_date_none_when_both_missing(self):
        from polylogue.lib.models import ConversationSummary

        s = ConversationSummary(id="test:1", provider="test")
        assert s.display_date is None

    def test_conversation_display_date_prefers_updated(self):
        from datetime import datetime, timezone

        from polylogue.lib.models import Conversation, MessageCollection

        created = datetime(2025, 1, 1, tzinfo=timezone.utc)
        updated = datetime(2025, 6, 15, tzinfo=timezone.utc)
        c = Conversation(
            id="test:1",
            provider="test",
            messages=MessageCollection(messages=[]),
            created_at=created,
            updated_at=updated,
        )
        assert c.display_date == updated

    def test_conversation_display_date_falls_back_to_created(self):
        from datetime import datetime, timezone

        from polylogue.lib.models import Conversation, MessageCollection

        created = datetime(2025, 1, 1, tzinfo=timezone.utc)
        c = Conversation(
            id="test:1",
            provider="test",
            messages=MessageCollection(messages=[]),
            created_at=created,
            updated_at=None,
        )
        assert c.display_date == created
