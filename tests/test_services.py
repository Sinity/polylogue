"""Tests for the services module (replaced DI container)."""

from __future__ import annotations

from polylogue.services import get_backend, get_repository, get_service_config, reset


class TestServices:
    def test_get_backend_returns_sqlite(self, workspace_env):
        from polylogue.storage.backends.sqlite import SQLiteBackend

        backend = get_backend()
        assert isinstance(backend, SQLiteBackend)

    def test_get_repository_returns_repo(self, workspace_env):
        from polylogue.storage.repository import ConversationRepository

        repo = get_repository()
        assert isinstance(repo, ConversationRepository)

    def test_get_config_returns_config(self):
        from polylogue.config import Config

        config = get_service_config()
        assert isinstance(config, Config)
        assert config.archive_root is not None

    def test_reset_clears_singletons(self, workspace_env):
        repo1 = get_repository()
        reset()
        repo2 = get_repository()
        assert repo1 is not repo2

    def test_singleton_returns_same_instance(self, workspace_env):
        repo1 = get_repository()
        repo2 = get_repository()
        assert repo1 is repo2

    def test_backend_singleton_returns_same_instance(self, workspace_env):
        backend1 = get_backend()
        backend2 = get_backend()
        assert backend1 is backend2

    def test_repository_uses_same_backend(self, workspace_env):
        repo1 = get_repository()
        repo2 = get_repository()
        assert repo1.backend is repo2.backend

    def test_reset_affects_backend_singleton(self, workspace_env):
        backend1 = get_backend()
        reset()
        backend2 = get_backend()
        assert backend1 is not backend2
