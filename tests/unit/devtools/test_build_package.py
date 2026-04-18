from __future__ import annotations

from pathlib import Path
from typing import Any

from devtools import build_package


def test_build_package_uses_repo_local_out_link(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    class Result:
        returncode = 0

    def fake_run(cmd: list[str], check: bool) -> Result:
        del check
        calls.append(cmd)
        return Result()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("devtools.build_package.subprocess.run", fake_run)

    assert build_package.main([]) == 0
    assert calls == [["nix", "build", ".#polylogue", "--out-link", ".local/result"]]


def test_build_package_accepts_custom_package_and_out_link(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    class Result:
        returncode = 0

    def fake_run(cmd: list[str], check: bool) -> Result:
        del check
        calls.append(cmd)
        return Result()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("devtools.build_package.subprocess.run", fake_run)

    assert build_package.main(["--package", ".#api-python", "--out-link", ".local/api-result"]) == 0
    assert calls == [["nix", "build", ".#api-python", "--out-link", ".local/api-result"]]
