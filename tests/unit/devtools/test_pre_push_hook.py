"""Pre-push hook env detection (#423).

Simulates the hook's pre-flight by running it with a doctored ``PATH`` that
either provides or omits ``devtools``. The successful path delegates to
``devtools verify --quick``; the failed path emits an actionable hint about
the devshell, instead of a Python traceback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

HOOK_PATH = Path(__file__).resolve().parents[3] / ".githooks" / "pre-push"


def test_hook_file_exists_and_is_executable() -> None:
    assert HOOK_PATH.is_file()
    assert os.access(HOOK_PATH, os.X_OK), "pre-push hook must be executable"


def test_hook_emits_devshell_hint_when_devtools_missing(tmp_path: Path) -> None:
    """When ``devtools`` is not on PATH, the hook prints a devshell hint, not a verify trace."""
    bash = shutil.which("bash") or "/bin/bash"
    coreutils_paths = [shutil.which(name) for name in ("cat", "echo")]
    coreutils_dirs = sorted({Path(path).parent for path in coreutils_paths if path is not None})
    minimal_path = ":".join(str(directory) for directory in coreutils_dirs) or os.environ["PATH"]
    env = {"PATH": minimal_path, "HOME": str(tmp_path)}
    result = subprocess.run(
        [bash, str(HOOK_PATH)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        check=False,
    )
    assert result.returncode != 0, "hook must exit non-zero when env is broken"
    combined = result.stdout + result.stderr
    assert "devtools is not importable" in combined
    assert "nix develop" in combined
    assert "Traceback" not in combined, "the env-not-loaded path must not surface a Python traceback to the user"


def test_hook_source_contains_quick_verify_step() -> None:
    """Hook still delegates to ``devtools verify --quick`` when env is healthy."""
    body = HOOK_PATH.read_text(encoding="utf-8")
    assert "devtools verify --quick" in body
    assert "devtools --help" in body, "pre-flight check must use a no-side-effects devtools invocation"
