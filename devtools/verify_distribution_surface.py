"""Verify installed distribution artifacts expose the intended runtime surface."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SCRIPTS = ("polylogue", "polylogued", "polylogue-mcp")


class DistributionVerificationError(RuntimeError):
    """Raised when a distribution artifact violates the supported surface."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify wheel/sdist installed distribution behavior.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for temporary build/install artifacts. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the generated work directory for inspection.",
    )
    args = parser.parse_args(argv)

    if args.work_dir is not None:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="polylogue-distribution-"))
        cleanup = not args.keep_work_dir

    try:
        verify_distribution_surface(work_dir)
    except DistributionVerificationError as exc:
        print(f"verify-distribution-surface: FAILED: {exc}", file=sys.stderr)
        if not cleanup:
            print(f"verify-distribution-surface: work dir: {work_dir}", file=sys.stderr)
        return 1
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)

    if args.keep_work_dir or args.work_dir is not None:
        print(f"verify-distribution-surface: work dir: {work_dir}", file=sys.stderr)
    print("verify-distribution-surface: ok", file=sys.stderr)
    return 0


def verify_distribution_surface(work_dir: Path) -> None:
    """Build and smoke installed wheel artifacts from checkout and unpacked sdist."""
    work_dir = work_dir.resolve()
    dist_dir = work_dir / "dist"
    _run(("uv", "build", "--out-dir", str(dist_dir), "--sdist", "--wheel", str(ROOT)), cwd=ROOT)

    wheel = _single_artifact(dist_dir, "*.whl")
    sdist = _single_artifact(dist_dir, "*.tar.gz")
    _verify_wheel_surface(wheel)
    _smoke_installed_wheel(wheel, work_dir / "wheel-install")

    unpacked = _unpack_sdist(sdist, work_dir / "unpacked-sdist")
    if (unpacked / ".git").exists():
        raise DistributionVerificationError("unpacked sdist unexpectedly contains .git")
    if not (unpacked / "polylogue" / "_build_info.py").exists():
        raise DistributionVerificationError("sdist is missing embedded polylogue/_build_info.py")

    sdist_wheel_dir = work_dir / "sdist-wheel"
    _run(("uv", "build", "--out-dir", str(sdist_wheel_dir), "--wheel", str(unpacked)), cwd=work_dir)
    sdist_wheel = _single_artifact(sdist_wheel_dir, "*.whl")
    _verify_wheel_surface(sdist_wheel)
    _smoke_installed_wheel(sdist_wheel, work_dir / "sdist-wheel-install")


def _verify_wheel_surface(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        if "polylogue/_build_info.py" not in names:
            raise DistributionVerificationError(f"{wheel.name} is missing polylogue/_build_info.py")
        entry_points = _read_entry_points(archive)
    for script in RUNTIME_SCRIPTS:
        if f"{script} =" not in entry_points:
            raise DistributionVerificationError(f"{wheel.name} is missing console script {script}")


def _read_entry_points(archive: zipfile.ZipFile) -> str:
    matches = [name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")]
    if not matches:
        raise DistributionVerificationError("wheel is missing entry_points.txt")
    return archive.read(matches[0]).decode()


def _smoke_installed_wheel(wheel: Path, install_dir: Path) -> None:
    wheel = wheel.resolve()
    install_dir = install_dir.resolve()
    install_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = install_dir / "venv"
    _run(("uv", "venv", str(venv_dir)), cwd=install_dir)
    python = _venv_python(venv_dir)
    _run(("uv", "pip", "install", "--python", str(python), str(wheel)), cwd=install_dir)
    env = _smoke_env(install_dir / "archive")
    bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    _run((str(bin_dir / "polylogue"), "--version"), cwd=install_dir, env=env)
    _run((str(bin_dir / "polylogue"), "--help"), cwd=install_dir, env=env)
    _run((str(bin_dir / "polylogue"), "--plain", "count"), cwd=install_dir, env=env)
    _run((str(python), "-m", "polylogue", "--version"), cwd=install_dir, env=env)
    _run((str(bin_dir / "polylogued"), "--help"), cwd=install_dir, env=env)
    _run((str(bin_dir / "polylogue-mcp"), "--help"), cwd=install_dir, env=env)


def _smoke_env(archive_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root.resolve())
    env["POLYLOGUE_FORCE_PLAIN"] = "1"
    return env


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _single_artifact(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise DistributionVerificationError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def _unpack_sdist(sdist: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(sdist) as archive:
        _safe_extract(archive, output_dir)
    roots = [path for path in output_dir.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise DistributionVerificationError(f"expected one unpacked sdist root, found {len(roots)}")
    return roots[0]


def _safe_extract(archive: tarfile.TarFile, output_dir: Path) -> None:
    destination = output_dir.resolve()
    for member in archive.getmembers():
        target = (output_dir / member.name).resolve()
        if destination not in (target, *target.parents):
            raise DistributionVerificationError(f"unsafe sdist member path: {member.name}")
    try:
        archive.extractall(output_dir, filter="data")
    except TypeError:
        archive.extractall(output_dir)


def _run(
    cmd: Iterable[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    rendered = " ".join(cmd)
    print(f"verify-distribution-surface: {rendered}", file=sys.stderr)
    result = subprocess.run(tuple(cmd), cwd=cwd, env=env, text=True, capture_output=True, check=False)
    if result.returncode == 0:
        return
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    raise DistributionVerificationError(f"command failed ({result.returncode}): {rendered}\n{output}")


if __name__ == "__main__":
    raise SystemExit(main())
