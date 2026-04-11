from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

sys.dont_write_bytecode = True


def _git_metadata(repo_root: Path) -> tuple[str, bool]:
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    dirty = bool(_run_git(repo_root, "status", "--porcelain"))
    return commit, dirty


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        joined = " ".join(args)
        raise RuntimeError(f"unable to resolve git metadata via `git {joined}`") from exc
    return result.stdout.strip()


def _render_build_info(commit: str, dirty: bool) -> str:
    return f'from __future__ import annotations\n\nBUILD_COMMIT = "{commit}"\nBUILD_DIRTY = {dirty}\n'


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        del version
        self._generated_build_info = False
        self._build_info_path = Path(self.root) / "polylogue" / "_build_info.py"

        if (Path(self.root) / ".git").exists():
            commit, dirty = _git_metadata(Path(self.root))
            self._build_info_path.write_text(_render_build_info(commit, dirty), encoding="utf-8")
            self._generated_build_info = True
            self._register_build_info_artifact(build_data)
            return

        if self._build_info_path.exists():
            self._register_build_info_artifact(build_data)
            return

        raise RuntimeError(
            "build metadata is mandatory: expected a git checkout or an embedded polylogue/_build_info.py"
        )

    def finalize(self, version: str, build_data: dict[str, object], artifact_path: str) -> None:
        del version, build_data, artifact_path
        if self._generated_build_info:
            self._build_info_path.unlink(missing_ok=True)

    def _register_build_info_artifact(self, build_data: dict[str, object]) -> None:
        relative_path = str(self._build_info_path.relative_to(self.root))
        artifacts = build_data.setdefault("artifacts", [])
        if not isinstance(artifacts, list):
            raise RuntimeError("unexpected hatch build-data shape for artifacts")
        if relative_path not in artifacts:
            artifacts.append(relative_path)
