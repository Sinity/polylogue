# mypy: disable-error-code="arg-type,attr-defined"

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import polylogue.version as version_module
from polylogue.version import (
    _detect_git_dirty,
    _get_embedded_build_info,
    _read_head_commit,
    _read_packed_ref,
    _resolve_base_version,
    _resolve_common_git_dir,
    _resolve_git_dir,
    _resolve_version,
)


def test_resolve_git_dir_supports_directory_relative_file_and_invalid_file(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo-dir"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    assert _resolve_git_dir(repo_dir) == repo_dir / ".git"

    repo_file = tmp_path / "repo-file"
    repo_file.mkdir()
    worktrees = tmp_path / "worktrees" / "repo"
    worktrees.mkdir(parents=True)
    (repo_file / ".git").write_text(f"gitdir: {Path('../worktrees/repo')}\n", encoding="utf-8")
    assert _resolve_git_dir(repo_file) == worktrees.resolve()

    repo_invalid = tmp_path / "repo-invalid"
    repo_invalid.mkdir()
    (repo_invalid / ".git").write_text("not-a-gitdir\n", encoding="utf-8")
    assert _resolve_git_dir(repo_invalid) is None

    repo_missing = tmp_path / "repo-missing"
    repo_missing.mkdir()
    assert _resolve_git_dir(repo_missing) is None


def test_read_packed_ref_ignores_comments_and_malformed_lines(tmp_path: Path) -> None:
    git_dir = tmp_path / "git"
    git_dir.mkdir()
    (git_dir / "packed-refs").write_text(
        f"# pack-refs\n^ignore-me\nnot a valid line\n{'a' * 40} refs/heads/main\n",
        encoding="utf-8",
    )

    assert _read_packed_ref(git_dir, "refs/heads/main") == "a" * 40
    assert _read_packed_ref(git_dir, "refs/heads/missing") is None


def test_read_packed_ref_handles_missing_file_and_read_errors(tmp_path: Path) -> None:
    git_dir = tmp_path / "git"
    git_dir.mkdir()
    assert _read_packed_ref(git_dir, "refs/heads/main") is None

    packed_refs = git_dir / "packed-refs"
    packed_refs.write_text("placeholder", encoding="utf-8")
    real_read_text = Path.read_text

    def _read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == packed_refs:
            raise OSError("broken")
        return real_read_text(self, *args, **kwargs)

    with patch("pathlib.Path.read_text", new=_read_text):
        assert _read_packed_ref(git_dir, "refs/heads/main") is None


def test_resolve_common_git_dir_supports_relative_and_empty_commondir(tmp_path: Path) -> None:
    git_dir = tmp_path / "git"
    git_dir.mkdir()
    assert _resolve_common_git_dir(git_dir) == git_dir

    (git_dir / "commondir").write_text("../common\n", encoding="utf-8")
    assert _resolve_common_git_dir(git_dir) == (tmp_path / "common").resolve()

    (git_dir / "commondir").write_text("\n", encoding="utf-8")
    assert _resolve_common_git_dir(git_dir) == git_dir


def test_resolve_git_dir_and_common_dir_handle_read_errors(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git_file = repo / ".git"
    git_file.write_text("gitdir: worktree\n", encoding="utf-8")

    common_dir = tmp_path / "git"
    common_dir.mkdir()
    commondir = common_dir / "commondir"
    commondir.write_text("../common\n", encoding="utf-8")
    real_read_text = Path.read_text

    def _read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self in {git_file, commondir}:
            raise OSError("broken")
        return real_read_text(self, *args, **kwargs)

    with patch("pathlib.Path.read_text", new=_read_text):
        assert _resolve_git_dir(repo) is None
        assert _resolve_common_git_dir(common_dir) == common_dir


def test_read_head_commit_supports_direct_ref_common_dir_and_packed_refs(tmp_path: Path) -> None:
    direct_repo = tmp_path / "direct"
    direct_repo.mkdir()
    git_dir = direct_repo / ".git"
    refs_dir = git_dir / "refs" / "heads"
    refs_dir.mkdir(parents=True)
    direct_commit = "a" * 40
    (git_dir / "HEAD").write_text(f"{direct_commit}\n", encoding="utf-8")
    assert _read_head_commit(direct_repo) == direct_commit

    ref_repo = tmp_path / "ref"
    ref_repo.mkdir()
    git_dir = ref_repo / ".git"
    refs_dir = git_dir / "refs" / "heads"
    refs_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (refs_dir / "main").write_text(f"{'b' * 40}\n", encoding="utf-8")
    assert _read_head_commit(ref_repo) == "b" * 40

    common_repo = tmp_path / "common-repo"
    common_repo.mkdir()
    git_dir = common_repo / ".git"
    git_dir.mkdir()
    common_dir = tmp_path / "common-dir"
    (common_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "commondir").write_text("../../common-dir\n", encoding="utf-8")
    (common_dir / "refs" / "heads" / "main").write_text(f"{'c' * 40}\n", encoding="utf-8")
    assert _read_head_commit(common_repo) == "c" * 40

    packed_repo = tmp_path / "packed"
    packed_repo.mkdir()
    git_dir = packed_repo / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "packed-refs").write_text(f"{'d' * 40} refs/heads/main\n", encoding="utf-8")
    assert _read_head_commit(packed_repo) == "d" * 40


def test_read_head_commit_returns_none_for_invalid_or_missing_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git_dir = repo / ".git"
    git_dir.mkdir()
    assert _read_head_commit(repo) is None

    (git_dir / "HEAD").write_text("not-a-ref\n", encoding="utf-8")
    assert _read_head_commit(repo) is None

    (git_dir / "HEAD").write_text("ref:\n", encoding="utf-8")
    assert _read_head_commit(repo) is None


def test_read_head_commit_handles_head_and_ref_read_errors(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git_dir = repo / ".git"
    refs_dir = git_dir / "refs" / "heads"
    refs_dir.mkdir(parents=True)
    head_path = git_dir / "HEAD"
    ref_path = refs_dir / "main"
    packed_refs = git_dir / "packed-refs"

    head_path.write_text("ref: refs/heads/main\n", encoding="utf-8")
    ref_path.write_text("invalid\n", encoding="utf-8")
    packed_refs.write_text(f"{'e' * 40} refs/heads/main\n", encoding="utf-8")

    real_read_text = Path.read_text

    def _read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == head_path:
            raise OSError("broken-head")
        return real_read_text(self, *args, **kwargs)

    with patch("pathlib.Path.read_text", new=_read_text):
        assert _read_head_commit(repo) is None

    def _read_ref_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == ref_path:
            raise OSError("broken-ref")
        return real_read_text(self, *args, **kwargs)

    with patch("pathlib.Path.read_text", new=_read_ref_text):
        assert _read_head_commit(repo) == "e" * 40


def test_detect_git_dirty_handles_clean_dirty_and_failures(tmp_path: Path) -> None:
    with patch("polylogue.version.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=" M file\n")):
        assert _detect_git_dirty(tmp_path) is True

    with patch("polylogue.version.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="")):
        assert _detect_git_dirty(tmp_path) is False

    with patch("polylogue.version.subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="")):
        assert _detect_git_dirty(tmp_path) is False

    with patch("polylogue.version.subprocess.run", side_effect=FileNotFoundError("git")):
        assert _detect_git_dirty(tmp_path) is False


def test_resolve_base_version_prefers_pyproject_then_metadata(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
    assert _resolve_base_version(tmp_path) == "1.2.3"

    (tmp_path / "pyproject.toml").unlink()
    with patch("polylogue.version.metadata_version", return_value="9.9.9"):
        assert _resolve_base_version(tmp_path) == "9.9.9"


def test_resolve_base_version_raises_when_metadata_is_unavailable(tmp_path: Path) -> None:
    with patch("polylogue.version.metadata_version", side_effect=version_module.PackageNotFoundError("polylogue")):
        with pytest.raises(RuntimeError, match="unable to resolve package version"):
            _resolve_base_version(tmp_path)


def test_get_embedded_build_info_supports_success_and_incomplete_metadata() -> None:
    fake_module = SimpleNamespace(BUILD_COMMIT="deadbeef", BUILD_DIRTY=True)
    with patch.dict(sys.modules, {"polylogue._build_info": fake_module}):
        assert _get_embedded_build_info() == ("deadbeef", True)

    incomplete_module = SimpleNamespace(BUILD_COMMIT="unknown", BUILD_DIRTY=False)
    with patch.dict(sys.modules, {"polylogue._build_info": incomplete_module}):
        with pytest.raises(RuntimeError, match="embedded build metadata is incomplete"):
            _get_embedded_build_info()


def test_get_embedded_build_info_raises_when_module_is_missing() -> None:
    real_import = __import__

    def _import(
        name: str, globals: object = None, locals: object = None, fromlist: tuple[str, ...] = (), level: int = 0
    ) -> object:
        if name == "polylogue._build_info":
            raise ImportError("missing-build-info")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=_import):
        with pytest.raises(RuntimeError, match="built package is missing embedded git metadata"):
            _get_embedded_build_info()


def test_resolve_version_uses_embedded_metadata_for_non_git_paths(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")

    with patch("polylogue.version._get_embedded_build_info", return_value=("deadbeef", False)):
        info = _resolve_version(tmp_path)

    assert info.version == "1.2.3"
    assert info.commit == "deadbeef"


def test_lazy_package_exports_cover_pipeline_services_storage_and_mcp() -> None:
    import polylogue.mcp
    import polylogue.pipeline
    import polylogue.pipeline.services
    import polylogue.storage

    assert polylogue.mcp.server.__name__.endswith("server")
    assert polylogue.pipeline.run_sources.__name__ == "run_sources"
    assert polylogue.pipeline.services.ValidationService.__name__ == "ValidationService"
    assert polylogue.storage.ConversationRepository.__name__ == "ConversationRepository"

    with pytest.raises(AttributeError):
        _ = polylogue.mcp.missing
    with pytest.raises(AttributeError):
        _ = polylogue.pipeline.missing
    with pytest.raises(AttributeError):
        _ = polylogue.pipeline.services.missing
    with pytest.raises(AttributeError):
        _ = polylogue.storage.missing
