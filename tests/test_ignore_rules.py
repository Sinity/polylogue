from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.importers.base import ImportResult
from polylogue.local_sync import sync_codex_sessions


def _fake_import(session_id: str, *, base_dir, output_dir, collapse_threshold, collapse_thresholds=None, html=False, html_theme="light", force=False, allow_dirty=False, registrar=None, attachment_ocr=False):
    slug = Path(session_id).stem
    md_path = output_dir / slug / "conversation.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("stub", encoding="utf-8")
    return ImportResult(markdown_path=md_path, html_path=None, attachments_dir=None, document=None, slug=slug)


def test_polylogueignore_skips_sessions(monkeypatch, tmp_path):
    base_dir = tmp_path / "sessions"
    base_dir.mkdir(parents=True)
    keep = base_dir / "keep.jsonl"
    skip = base_dir / "skipme.jsonl"
    keep.write_text("{}", encoding="utf-8")
    skip.write_text("{}", encoding="utf-8")
    (base_dir / ".polylogueignore").write_text("*skip*.jsonl\n", encoding="utf-8")

    monkeypatch.setattr("polylogue.importers.codex.import_codex_session", _fake_import)

    result = sync_codex_sessions(
        base_dir=base_dir,
        output_dir=tmp_path / "out",
        collapse_threshold=10,
        collapse_thresholds=None,
        html=False,
        html_theme="light",
        force=False,
        prune=False,
        diff=False,
        sessions=None,
        registrar=None,
        ui=None,
        attachment_ocr=False,
    )

    slugs = {item.slug for item in result.written}
    assert len(slugs) == 1
    assert next(iter(slugs)).startswith("keep")
    assert result.skipped == 0
