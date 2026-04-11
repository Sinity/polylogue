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


def _resolve_git_dir(repo_root: Path) -> Path | None:
    git_entry = repo_root / ".git"
    if git_entry.is_dir():
        return git_entry
    if not git_entry.is_file():
        return None
    try:
        text = git_entry.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    prefix = "gitdir:"
    if not text.startswith(prefix):
        return None
    target = text[len(prefix) :].strip()
    git_dir = Path(target)
    if not git_dir.is_absolute():
        git_dir = (repo_root / git_dir).resolve()
    return git_dir


def _read_packed_ref(git_dir: Path, ref_name: str) -> str | None:
    packed_refs = git_dir / "packed-refs"
    if not packed_refs.exists():
        return None
    try:
        lines = packed_refs.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line or line.startswith(("#", "^")):
            continue
        try:
            commit, name = line.split(" ", 1)
        except ValueError:
            continue
        if name == ref_name:
            return commit
    return None


def _read_head_commit(repo_root: Path) -> str | None:
    git_dir = _resolve_git_dir(repo_root)
    if git_dir is None:
        return None
    head_path = git_dir / "HEAD"
    try:
        head_text = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head_text:
        return None
    if re.fullmatch(r"[0-9a-f]{40}", head_text):
        return head_text
    prefix = "ref:"
    if not head_text.startswith(prefix):
        return None
    ref_name = head_text[len(prefix) :].strip()
    if not ref_name:
        return None
    ref_path = git_dir / ref_name
    if ref_path.exists():
        try:
            commit = ref_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return commit if re.fullmatch(r"[0-9a-f]{40}", commit) else None
    return _read_packed_ref(git_dir, ref_name)


def _detect_git_dirty(repo_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            return False
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _get_git_info(repo_root: Path) -> tuple[str | None, bool]:
    """Get git commit hash and dirty state from repository."""
    commit = _read_head_commit(repo_root)
    if commit is None:
        return None, False
    return commit, _detect_git_dirty(repo_root)


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
