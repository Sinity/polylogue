from __future__ import annotations

import contextlib
import re
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from pathlib import Path


@dataclass
class VersionInfo:
    version: str
    commit: str | None = None
    dirty: bool = False

    def __str__(self) -> str:
        if not self.commit:
            return self.version
        short = self.commit[:8]
        suffix = "-dirty" if self.dirty else ""
        return f"{self.version}+{short}{suffix}"

    @property
    def full(self) -> str:
        return str(self)

    @property
    def short(self) -> str:
        return self.version


def _get_git_info(repo_root: Path) -> tuple[str | None, bool]:
    """Get git commit hash and dirty state from repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            return None, False
        commit = result.stdout.strip()

        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        dirty = bool(dirty_result.stdout.strip())
        return commit, dirty
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None, False


def _resolve_base_version(repo_root: Path) -> str:
    version = "unknown"

    pyproject_path = repo_root / "pyproject.toml"
    if pyproject_path.exists():
        try:
            text = pyproject_path.read_text(encoding="utf-8")
            match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
            if match:
                version = match.group(1)
        except (OSError, UnicodeDecodeError):
            pass

    if version == "unknown":
        with contextlib.suppress(PackageNotFoundError):
            version = metadata_version("polylogue")

    if version == "unknown":
        raise RuntimeError("unable to resolve package version from metadata or pyproject.toml")

    return version


def _get_embedded_build_info() -> tuple[str, bool]:
    try:
        from polylogue._build_info import BUILD_COMMIT, BUILD_DIRTY
    except ImportError as exc:
        raise RuntimeError("built package is missing embedded git metadata") from exc

    if not BUILD_COMMIT or BUILD_COMMIT == "unknown":
        raise RuntimeError("embedded build metadata is incomplete")

    return BUILD_COMMIT, BUILD_DIRTY


def _resolve_version(repo_root: Path | None = None) -> VersionInfo:
    """Resolve the Polylogue version and exact build identity."""
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    version = _resolve_base_version(repo_root)

    if (repo_root / ".git").exists():
        commit, dirty = _get_git_info(repo_root)
        if commit is None:
            raise RuntimeError("source checkout is missing git commit metadata")
    else:
        commit, dirty = _get_embedded_build_info()

    return VersionInfo(version=version, commit=commit, dirty=dirty)


VERSION_INFO = _resolve_version()
POLYLOGUE_VERSION = VERSION_INFO.full

__all__ = ["POLYLOGUE_VERSION", "VERSION_INFO", "VersionInfo"]
