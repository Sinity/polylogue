"""Antigravity source-level acquisition contracts.

The periodic daemon reconciler covers a working language-server export. These
tests exercise the shared source iterator when that supported export surface
is unavailable, which is where brain metadata used to bypass artifact
classification and become synthetic conversation sessions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.sources.parsers.antigravity import AntigravityBinaryUnavailableError
from polylogue.sources.source_parsing import iter_antigravity_language_server_sessions


def _write_brain_sidecar(root: Path) -> Path:
    metadata_path = root / "brain" / "work-session" / "plan.md.metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.with_name("plan.md").write_text("# Plan\n\nInspect the archive.\n", encoding="utf-8")
    metadata_path.write_text(
        '{"artifactType":"ARTIFACT_TYPE_OTHER","summary":"Plan","updatedAt":"2026-08-04T08:00:00Z"}',
        encoding="utf-8",
    )
    return metadata_path


def test_unavailable_language_server_never_promotes_brain_sidecars_to_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing exporter leaves the conversation coverage gap visible.

    This enters the real Antigravity source iterator that batch import uses.
    Metadata is an artifact, so it cannot substitute for a conversation
    trajectory when the language-server export is unavailable.
    """
    root = tmp_path / "antigravity"
    (root / "conversations").mkdir(parents=True)
    _write_brain_sidecar(root)

    def unavailable_export(*_args: object, **_kwargs: object) -> object:
        raise AntigravityBinaryUnavailableError("test language server unavailable")

    monkeypatch.setattr(
        "polylogue.sources.source_parsing.antigravity.iter_language_server_exports",
        unavailable_export,
    )
    caplog.set_level(logging.WARNING, logger="polylogue.sources.source_parsing")

    sessions = list(iter_antigravity_language_server_sessions(Source(name="antigravity", path=root)))

    assert sessions == []
    assert "no session coverage" in caplog.messages[-1]
