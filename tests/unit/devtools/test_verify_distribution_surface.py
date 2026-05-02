from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path

import pytest
import tomllib

from devtools import verify_distribution_surface as surface


def test_verify_wheel_surface_accepts_runtime_scripts_without_devtools(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path, entry_points=_runtime_entry_points())

    surface._verify_wheel_surface(wheel)


def test_verify_wheel_surface_rejects_devtools_script(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path, entry_points=f"{_runtime_entry_points()}devtools = devtools.__main__:main\n")

    with pytest.raises(surface.DistributionVerificationError, match="devtools"):
        surface._verify_wheel_surface(wheel)


def test_verify_wheel_surface_rejects_devtools_package(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path, entry_points=_runtime_entry_points(), extra_files={"devtools/__main__.py": ""})

    with pytest.raises(surface.DistributionVerificationError, match="devtools package"):
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
    assert not any("devtools" in call for call in calls)


def test_pyproject_keeps_devtools_source_only() -> None:
    data = tomllib.loads((surface.ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "devtools" not in data["project"]["scripts"]
    assert data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["polylogue"]


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
