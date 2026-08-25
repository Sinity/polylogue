from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devtools import render_cli_reference, render_devtools_reference, render_docs_surface, render_pages


@pytest.fixture
def configured_pages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config = tmp_path / "pages.toml"
    config.write_text("site_name = 'test'\n", encoding="utf-8")
    monkeypatch.setattr(render_pages, "_default_config_path", lambda: config)
    return config


def _patch_feeders(monkeypatch: pytest.MonkeyPatch, result: object = 0) -> None:
    for module in (render_cli_reference, render_devtools_reference, render_docs_surface):
        monkeypatch.setattr(module, "main", lambda _argv, result=result: result)


def test_render_pages_stops_before_assembly_on_nonzero_feeder(
    monkeypatch: pytest.MonkeyPatch, configured_pages: Path, tmp_path: Path
) -> None:
    del configured_pages
    calls: list[str] = []
    monkeypatch.setattr(render_cli_reference, "main", lambda _argv: 7)

    def render_devtools(_argv: list[str] | None) -> int:
        calls.append("devtools")
        return 0

    def render_docs(_argv: list[str] | None) -> int:
        calls.append("docs")
        return 0

    monkeypatch.setattr(render_devtools_reference, "main", render_devtools)
    monkeypatch.setattr(render_docs_surface, "main", render_docs)
    monkeypatch.setattr(render_pages, "ROOT", tmp_path)

    def should_not_build(**_kwargs: object) -> Path:
        raise AssertionError("a failed feeder must not assemble the site")

    from devtools import pages_builder

    monkeypatch.setattr(pages_builder, "build_site_with_pagefind", should_not_build)
    assert render_pages.main(["--output", str(tmp_path / "site")]) == 7


def test_render_pages_feeder_exception_is_nonzero_and_actionable(
    monkeypatch: pytest.MonkeyPatch, configured_pages: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del configured_pages
    monkeypatch.setattr(render_cli_reference, "main", lambda _argv: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(render_devtools_reference, "main", lambda _argv: 0)
    monkeypatch.setattr(render_docs_surface, "main", lambda _argv: 0)

    assert render_pages.main(["--skip-pagefind"]) == 1
    captured = capsys.readouterr()
    assert "diagnosis: render_feeder_exception" in captured.err
    assert "boom" in captured.err
    assert "Warning:" not in captured.err


def test_render_pages_pagefind_failure_is_nonzero_without_site_success(
    monkeypatch: pytest.MonkeyPatch,
    configured_pages: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del configured_pages
    _patch_feeders(monkeypatch)
    from devtools import pages_builder

    output = tmp_path / "site"
    monkeypatch.setattr(
        pages_builder,
        "build_site",
        lambda config_path=None, output_dir=None: output,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("pagefind")),
    )

    assert render_pages.main(["--output", str(output)]) == 1
    captured = capsys.readouterr()
    assert "diagnosis: render_pagefind_failed" in captured.err
    assert "Site built:" not in captured.err
    assert "Warning:" not in captured.err
