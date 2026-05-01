from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as importlib_metadata_version
from pathlib import Path

import pytest

import polylogue.version as version_module
from polylogue.version import VersionInfo, _get_embedded_build_info, _get_git_info, _resolve_version

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

    @pytest.mark.parametrize(
        "commit,dirty,has_dirty_suffix", [("abc123def456", False, False), ("abc123def456", True, True)]
    )
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
    def test_returns_none_for_missing_or_non_git_paths(
        self,
        tmp_path: Path,
        path_factory: Callable[[Path], Path],
    ) -> None:
        commit, dirty = _get_git_info(path_factory(tmp_path))
        assert commit is None
        assert dirty is False

    def test_timeout_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        def mock_run(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired("git", 2)

        monkeypatch.setattr(subprocess, "run", mock_run)
        commit, dirty = _get_git_info(tmp_path)
        assert commit is None
        assert dirty is False

    def test_returns_tuple_and_dirty_is_bool(self, tmp_path: Path) -> None:
        result = _get_git_info(tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[1], bool)

    def test_reads_head_commit_without_git_executable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        git_dir = tmp_path / ".git"
        refs_dir = git_dir / "refs" / "heads"
        refs_dir.mkdir(parents=True)
        commit = "a" * 40
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (refs_dir / "main").write_text(f"{commit}\n", encoding="utf-8")

        def missing_git(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("git")

        monkeypatch.setattr("polylogue.version.subprocess.run", missing_git)
        resolved_commit, dirty = _get_git_info(tmp_path)
        assert resolved_commit == commit
        assert dirty is False


class TestResolveVersion:
    def test_returns_version_info_and_attributes(self) -> None:
        result = _resolve_version()
        assert isinstance(result, VersionInfo)
        assert result.version not in {"", "unknown"}
        assert isinstance(result.commit, str)
        assert isinstance(result.dirty, bool)

    def test_resolve_version_consistency_and_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result1 = _resolve_version()
        result2 = _resolve_version()
        assert isinstance(result1, VersionInfo)
        assert isinstance(result2, VersionInfo)
        assert result1.version == result2.version

        def mock_metadata_version(name: str) -> None:
            raise PackageNotFoundError(name)

        original = importlib_metadata_version
        monkeypatch.setattr("polylogue.version.metadata_version", mock_metadata_version)
        try:
            result = _resolve_version()
        finally:
            monkeypatch.setattr("polylogue.version.metadata_version", original)

        assert result.version is not None
        assert len(result.version) > 0

    def test_source_checkout_requires_git_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
        monkeypatch.setattr(version_module, "_get_git_info", lambda _: (None, False))
        with pytest.raises(RuntimeError, match="source checkout is missing git commit metadata"):
            _resolve_version(tmp_path)

    def test_built_artifact_uses_embedded_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
        monkeypatch.setattr(version_module, "_get_embedded_build_info", lambda: ("deadbeefdeadbeef", True))
        result = _resolve_version(tmp_path)
        assert result.version == "1.2.3"
        assert result.commit == "deadbeefdeadbeef"
        assert result.dirty is True

    def test_built_artifact_requires_embedded_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")

        def raise_missing_metadata() -> tuple[str, bool]:
            raise RuntimeError("built package is missing embedded git metadata")

        monkeypatch.setattr(version_module, "_get_embedded_build_info", raise_missing_metadata)
        with pytest.raises(RuntimeError, match="built package is missing embedded git metadata"):
            _resolve_version(tmp_path)


class TestEmbeddedBuildInfo:
    def test_unknown_build_commit_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(__import__("sys").modules, "polylogue._build_info", raising=False)

        class BuildInfo:
            BUILD_COMMIT = "unknown"
            BUILD_DIRTY = False

        monkeypatch.setitem(__import__("sys").modules, "polylogue._build_info", BuildInfo())
        with pytest.raises(RuntimeError, match="embedded build metadata is incomplete"):
            _get_embedded_build_info()


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
