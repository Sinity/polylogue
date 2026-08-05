from __future__ import annotations

import subprocess
from pathlib import Path


def test_webui_generate_check_script_resolves_from_ci_working_directory() -> None:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["npm", "run", "generate:check"],
        cwd=root / "webui",
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "render webui-design-system: sync OK" in result.stdout
