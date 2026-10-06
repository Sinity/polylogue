from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

sys.dont_write_bytecode = True

_BUILD_INFO_RELATIVE = "polylogue/_build_info.py"


def _git_metadata(repo_root: Path) -> tuple[str, bool]:
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    dirty = bool(_run_git(repo_root, "status", "--porcelain"))
    return commit, dirty


# Build metadata is mandatory, so this timeout decides whether a workspace
# provision succeeds at all. Concurrent provisioning puts many git processes on
# one repository at once, and `status --porcelain` there is far from instant.
GIT_TIMEOUT_SECONDS = 60


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        joined = " ".join(args)
        raise RuntimeError(f"unable to resolve git metadata via `git {joined}`") from exc
    return result.stdout.strip()


def _render_build_info(commit: str, dirty: bool) -> str:
    return f'from __future__ import annotations\n\nBUILD_COMMIT = "{commit}"\nBUILD_DIRTY = {dirty}\n'


class CustomBuildHook(BuildHookInterface):
    """Embed ``polylogue/_build_info.py`` without writing into the source tree.

    A checkout renders the module into a private temporary directory and
    force-includes it at its package path, so a read-only checkout (a sandboxed
    test runner, a mounted source) builds the same artifact. An unpacked sdist
    or a Nix source tree already carries the module and registers it as is.
    """

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        del version
        self._generated_dir: tempfile.TemporaryDirectory[str] | None = None
        repo_root = Path(self.root)
        build_info_path = repo_root / _BUILD_INFO_RELATIVE

        if (repo_root / ".git").exists():
            try:
                commit, dirty = _git_metadata(repo_root)
            except RuntimeError:
                if build_info_path.exists():
                    self._register_build_info_artifact(build_data)
                    return
                raise
            self._generated_dir = tempfile.TemporaryDirectory(prefix="polylogue-build-info-")
            generated = Path(self._generated_dir.name) / "_build_info.py"
            generated.write_text(_render_build_info(commit, dirty), encoding="utf-8")
            force_include = build_data.setdefault("force_include", {})
            if not isinstance(force_include, dict):
                raise RuntimeError("unexpected hatch build-data shape for force_include")
            force_include[str(generated)] = _BUILD_INFO_RELATIVE
            return

        if build_info_path.exists():
            self._register_build_info_artifact(build_data)
            return

        raise RuntimeError(
            "build metadata is mandatory: expected a git checkout or an embedded polylogue/_build_info.py"
        )

    def finalize(self, version: str, build_data: dict[str, object], artifact_path: str) -> None:
        del version, build_data, artifact_path
        if self._generated_dir is not None:
            self._generated_dir.cleanup()
            self._generated_dir = None

    def _register_build_info_artifact(self, build_data: dict[str, object]) -> None:
        artifacts = build_data.setdefault("artifacts", [])
        if not isinstance(artifacts, list):
            raise RuntimeError("unexpected hatch build-data shape for artifacts")
        if _BUILD_INFO_RELATIVE not in artifacts:
            artifacts.append(_BUILD_INFO_RELATIVE)
