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


def _resolve_version() -> VersionInfo:
    """Resolve the Polylogue version from package metadata or pyproject.toml.

    Includes git commit info when running from a source checkout.
    """
    version = "unknown"
    commit: str | None = None
    dirty = False

    with contextlib.suppress(PackageNotFoundError):
        version = metadata_version("polylogue")

    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"

    if version == "unknown" and pyproject_path.exists():
        try:
            text = pyproject_path.read_text(encoding="utf-8")
            match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
            if match:
                version = match.group(1)
        except (OSError, UnicodeDecodeError):
            pass

    # Check for git info in source checkout
    if (repo_root / ".git").exists():
        commit, dirty = _get_git_info(repo_root)
    else:
        # Nix build or installed package — try build-time info
        try:
            from polylogue._build_info import BUILD_COMMIT, BUILD_DIRTY
            commit = BUILD_COMMIT if BUILD_COMMIT != "unknown" else None
            dirty = BUILD_DIRTY
        except ImportError:
            pass

    return VersionInfo(version=version, commit=commit, dirty=dirty)


VERSION_INFO = _resolve_version()
POLYLOGUE_VERSION = VERSION_INFO.full

__all__ = ["POLYLOGUE_VERSION", "VERSION_INFO", "VersionInfo"]
