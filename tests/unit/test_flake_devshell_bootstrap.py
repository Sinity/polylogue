from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAP_START = "          # The venv must track the devShell interpreter."
_BOOTSTRAP_END = "          # Activate venv"


def _bootstrap_block() -> str:
    flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
    start = flake.index(_BOOTSTRAP_START)
    end = flake.index(_BOOTSTRAP_END, start)
    return textwrap.dedent(flake[start:end])


def _write_identity_python(path: Path, identity: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' {identity!r}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("existing_identity", "expected_notice"),
    [
        pytest.param(None, "creating virtual environment", id="initial-creation"),
        pytest.param("3.14.5 True", "interpreter changed", id="identity-repair"),
    ],
)
def test_devshell_venv_bootstrap_binds_uv_to_active_python(
    tmp_path: Path,
    existing_identity: str | None,
    expected_notice: str,
) -> None:
    """Execute the literal flake hook for creation and interpreter repair.

    The fake uv intentionally creates no environment unless it receives the
    active devshell executable via ``--python``. This fails against a bare
    ``uv venv`` invocation, so it protects the real route where uv otherwise
    selects its managed GIL interpreter instead of the Nix free-threaded one.
    """
    bin_dir = tmp_path / "bin"
    active_python = bin_dir / "python3"
    _write_identity_python(active_python, "3.14.4 False")
    if existing_identity is not None:
        _write_identity_python(tmp_path / ".venv" / "bin" / "python", existing_identity)

    uv_args = tmp_path / "uv-args"
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$@" > "$UV_ARGS"\n'
        'test "${1:-}" = venv\n'
        'test "${2:-}" = --python\n'
        'test "${3:-}" = "$EXPECTED_PYTHON"\n'
        "mkdir -p .venv/bin\n"
        "printf '%s\\n' '#!/usr/bin/env bash' \"printf '%s\\\\n' '3.14.4 False'\" > .venv/bin/python\n"
        "chmod +x .venv/bin/python\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    result = subprocess.run(
        ["bash", "-c", _bootstrap_block()],
        cwd=tmp_path,
        env=os.environ
        | {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "UV_ARGS": str(uv_args),
            "EXPECTED_PYTHON": str(active_python),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert expected_notice in result.stderr
    assert uv_args.read_text(encoding="utf-8").splitlines() == [
        "venv",
        "--python",
        str(active_python),
    ]
