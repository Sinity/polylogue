"""Contracts for low-write archive template clones."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.infra.archive_templates import clone_archive_template


def test_clone_requests_reflink_before_copy_fallback(monkeypatch, tmp_path: Path) -> None:
    template = tmp_path / "template"
    destination = tmp_path / "clone"
    template.mkdir()
    (template / "index.db").write_bytes(b"snapshot")
    calls: list[list[str]] = []

    def no_reflink(*args: object, **kwargs: object) -> None:
        argv = list(args[0])
        calls.append(argv)
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", no_reflink)
    clone_archive_template(template, destination)

    assert calls == [["cp", "-a", "--reflink=always", f"{template}/.", str(destination)]]
    assert (destination / "index.db").read_bytes() == b"snapshot"
    assert (destination / "index.db").stat().st_mode & 0o200


def test_clone_keeps_template_unchanged_after_private_write(tmp_path: Path) -> None:
    template = tmp_path / "template"
    destination = tmp_path / "clone"
    template.mkdir()
    (template / "index.db").write_bytes(b"snapshot")

    clone_archive_template(template, destination)
    (destination / "index.db").write_bytes(b"private")

    assert (template / "index.db").read_bytes() == b"snapshot"
    assert (destination / "index.db").read_bytes() == b"private"
