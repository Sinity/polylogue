from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from polylogue.version import VersionInfo, _get_git_info, _resolve_version

# =============================================================================
# Merged from test_lazy_exports.py (2024-03-15)
# =============================================================================


@pytest.mark.parametrize(
    "name",
    [
        "ArchiveStats",
        "Conversation",
        "Message",
        "Polylogue",
        "PolylogueError",
        "SearchResult",
        "SyncPolylogue",
    ],
)
def test_lazy_import_archive_core_exports_root(name: str) -> None:
    import polylogue

    assert getattr(polylogue, name).__name__ == name


def test_lazy_import_unknown_raises_root() -> None:
    import polylogue

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = polylogue.NonExistentThing


@pytest.mark.parametrize(
    "name",
    [
        "ArchiveCoverage",
        "ConversationAttribution",
        "ConversationRepository",
        "DaySessionSummary",
        "Decision",
        "ModelPricing",
        "SessionProfile",
        "WeekSessionSummary",
        "WorkEvent",
        "WorkEventKind",
        "WorkThread",
        "build_session_profile",
        "build_session_threads",
        "estimate_cost",
        "harmonize_session_cost",
        "infer_auto_tags",
    ],
)
def test_root_does_not_export_semantic_or_storage_helpers(name: str) -> None:
    import polylogue

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = getattr(polylogue, name)


@pytest.mark.parametrize(
    "name",
    [
        "Attachment",
        "BranchType",
        "Conversation",
        "ConversationProjection",
        "DialoguePair",
        "Message",
        "MessageCollection",
        "Role",
    ],
)
def test_lazy_import_domain_exports_lib(name: str) -> None:
    import polylogue.lib

    assert getattr(polylogue.lib, name).__name__ == name


@pytest.mark.parametrize("name", ["ArchiveStats", "ConversationRepository"])
def test_lib_root_does_not_export_archive_runtime_surfaces(name: str) -> None:
    import polylogue.lib

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = getattr(polylogue.lib, name)


def test_lazy_import_unknown_raises_lib() -> None:
    import polylogue.lib

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = polylogue.lib.NonExistentThing


@pytest.mark.parametrize(
    "name",
    ["HarmonizedMessage", "SchemaValidator", "ValidationResult", "validate_provider_export"],
)
def test_runtime_schema_exports_are_narrow(name: str) -> None:
    import polylogue.schemas

    assert getattr(polylogue.schemas, name).__name__ == name


@pytest.mark.parametrize("name", ["SchemaDiff", "SchemaRegistry"])
def test_schema_root_does_not_export_tooling_registry_surfaces(name: str) -> None:
    import polylogue.schemas

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = getattr(polylogue.schemas, name)


class TestVersionInfo:
    def test_version_only_and_repr(self) -> None:
        info = VersionInfo(version="1.0.0")
        assert str(info) == "1.0.0"
        assert info.full == "1.0.0"
        assert info.short == "1.0.0"
        assert "1.0.0" in repr(info)

    @pytest.mark.parametrize("commit,dirty,has_dirty_suffix", [("abc123def456", False, False), ("abc123def456", True, True)])
    def test_version_with_commit(self, commit: str, dirty: bool, has_dirty_suffix: bool) -> None:
        info = VersionInfo(version="1.0.0", commit=commit, dirty=dirty)
        rendered = str(info)
        assert "1.0.0+" in rendered
        assert info.short == "1.0.0"
        assert ("-dirty" in rendered) is has_dirty_suffix

    @pytest.mark.parametrize("version,commit", [("2.5.3", "fedcba9876543210"), ("3.2.1", "deadbeef12345678")])
    def test_version_properties(self, version: str, commit: str) -> None:
        info = VersionInfo(version=version, commit=commit)
        assert str(info) == f"{version}+{commit[:8]}"
        assert info.full == f"{version}+{commit[:8]}"
        assert info.short == version

    def test_version_dirty_without_commit_and_equality(self) -> None:
        assert str(VersionInfo(version="1.0.0", dirty=True)) == "1.0.0"
        assert VersionInfo(version="1.0.0") == VersionInfo(version="1.0.0")
        assert {field.name for field in fields(VersionInfo)} == {"version", "commit", "dirty"}


class TestGetGitInfo:
    def test_valid_git_repo(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        if (repo_root / ".git").exists():
            commit, dirty = _get_git_info(repo_root)
            assert commit is not None
            assert len(commit) == 40
            assert isinstance(dirty, bool)

    @pytest.mark.parametrize("path_factory", [lambda tmp_path: tmp_path / "nonexistent", lambda tmp_path: tmp_path])
    def test_returns_none_for_missing_or_non_git_paths(self, tmp_path, path_factory) -> None:
        commit, dirty = _get_git_info(path_factory(tmp_path))
        assert commit is None
        assert dirty is False

    def test_timeout_returns_none(self, tmp_path, monkeypatch) -> None:
        import subprocess

        def mock_run(*args, **kwargs):
            raise subprocess.TimeoutExpired("git", 2)

        monkeypatch.setattr(subprocess, "run", mock_run)
        commit, dirty = _get_git_info(tmp_path)
        assert commit is None
        assert dirty is False

    def test_returns_tuple_and_dirty_is_bool(self, tmp_path) -> None:
        result = _get_git_info(tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[1], bool)


class TestResolveVersion:
    def test_returns_version_info_and_attributes(self) -> None:
        result = _resolve_version()
        assert isinstance(result, VersionInfo)
        assert result.version not in {"", "unknown"}
        assert result.commit is None or isinstance(result.commit, str)
        assert isinstance(result.dirty, bool)

    def test_resolve_version_consistency_and_fallback(self, monkeypatch) -> None:
        result1 = _resolve_version()
        result2 = _resolve_version()
        assert isinstance(result1, VersionInfo)
        assert isinstance(result2, VersionInfo)
        assert result1.version == result2.version

        from importlib.metadata import PackageNotFoundError

        import polylogue.version as version_module

        def mock_metadata_version(name):
            raise PackageNotFoundError(name)

        original = version_module.metadata_version
        monkeypatch.setattr(version_module, "metadata_version", mock_metadata_version)
        try:
            result = _resolve_version()
        finally:
            monkeypatch.setattr(version_module, "metadata_version", original)

        assert result.version is not None
        assert len(result.version) > 0


class TestVersionConstants:
    def test_all_version_exports(self) -> None:
        from polylogue import version
        from polylogue.version import POLYLOGUE_VERSION, VERSION_INFO

        assert isinstance(POLYLOGUE_VERSION, str)
        assert len(POLYLOGUE_VERSION) > 0
        assert isinstance(VERSION_INFO, VersionInfo)
        assert VERSION_INFO.version in POLYLOGUE_VERSION
        assert hasattr(version, "POLYLOGUE_VERSION")
        assert hasattr(version, "VERSION_INFO")
        assert hasattr(version, "VersionInfo")
        if hasattr(version, "__all__"):
            assert {"POLYLOGUE_VERSION", "VERSION_INFO", "VersionInfo"}.issubset(set(version.__all__))
