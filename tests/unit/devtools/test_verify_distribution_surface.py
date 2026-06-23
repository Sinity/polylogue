from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path

import pytest

from devtools import verify_distribution_surface as surface


def test_verify_wheel_surface_accepts_runtime_scripts(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path, entry_points=_runtime_entry_points())

    surface._verify_wheel_surface(wheel)


def test_verify_distribution_surface_builds_sdist_wheel_and_smokes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(cmd: tuple[str, ...], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        del cwd, env
        calls.append(cmd)
        if cmd[:2] == ("uv", "build") and "--sdist" in cmd:
            dist_dir = Path(cmd[cmd.index("--out-dir") + 1])
            dist_dir.mkdir(parents=True, exist_ok=True)
            _write_wheel(dist_dir, entry_points=_runtime_entry_points())
            _write_sdist(dist_dir / "polylogue-0.1.0.tar.gz")
        elif cmd[:2] == ("uv", "build") and "--wheel" in cmd:
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_wheel(out_dir, entry_points=_runtime_entry_points())
        elif cmd[:2] == ("uv", "venv"):
            venv = Path(cmd[2])
            bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
            bin_dir.mkdir(parents=True, exist_ok=True)
            for script in (*surface.RUNTIME_SCRIPTS, "python"):
                (bin_dir / script).write_text("", encoding="utf-8")

    monkeypatch.setattr(surface, "_run", fake_run)

    surface.verify_distribution_surface(tmp_path)

    rendered = [" ".join(call[:2]) for call in calls]
    assert rendered.count("uv build") == 2
    assert rendered.count("uv venv") == 2
    import_probes = [call for call in calls if len(call) >= 4 and call[1:3] == ("-I", "-c")]
    assert len(import_probes) == 2
    assert all("polylogue.archive.query.expression" in call[3] for call in import_probes)
    smoke_commands = [" ".join(call) for call in calls]
    assert sum("polylogue --plain ops diagnostics workload --json" in call for call in smoke_commands) == 2
    assert sum("polylogue --plain ops diagnostics space --json" in call for call in smoke_commands) == 2


def test_smoke_env_removes_source_pythonpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/repo/source")

    env = surface._smoke_env(tmp_path / "archive")

    assert "PYTHONPATH" not in env
    assert env["POLYLOGUE_FORCE_PLAIN"] == "1"


def _runtime_entry_points() -> str:
    return "\n".join(
        [
            "[console_scripts]",
            "polylogue = polylogue.cli:main",
            "polylogued = polylogue.daemon.cli:main",
            "polylogue-mcp = polylogue.mcp.cli:main",
            "",
        ]
    )


def _write_wheel(
    directory: Path,
    *,
    entry_points: str,
    extra_files: dict[str, str] | None = None,
) -> Path:
    wheel = directory / "polylogue-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("polylogue/__init__.py", "")
        archive.writestr("polylogue/_build_info.py", 'BUILD_COMMIT = "deadbeef"\nBUILD_DIRTY = False\n')
        archive.writestr("polylogue-0.1.0.dist-info/entry_points.txt", entry_points)
        for name, content in (extra_files or {}).items():
            archive.writestr(name, content)
    return wheel


def _write_sdist(path: Path) -> None:
    root = path.parent / "polylogue-0.1.0"
    (root / "polylogue").mkdir(parents=True)
    (root / "polylogue" / "_build_info.py").write_text(
        'BUILD_COMMIT = "deadbeef"\nBUILD_DIRTY = False\n',
        encoding="utf-8",
    )
    with tarfile.open(path, "w:gz") as archive:
        archive.add(root, arcname=root.name)
