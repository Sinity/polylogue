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
from polylogue.core.enums import Provider
from polylogue.sources.parsers import antigravity
from polylogue.sources.parsers.antigravity import AntigravityBinaryUnavailableError
from polylogue.sources.source_parsing import iter_antigravity_language_server_sessions
from polylogue.sources.source_walk import census_source_root


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
    Before the fix, its fallback parsed this sidecar into a degraded session,
    bypassing the same taxonomy rule that excludes the file from live ingest
    and schema inference.
    """
    root = tmp_path / "antigravity"
    (root / "conversations").mkdir(parents=True)
    _write_brain_sidecar(root)

    def unavailable_export(*_args: object, **_kwargs: object) -> object:
        raise AntigravityBinaryUnavailableError("test language server unavailable")

    monkeypatch.setattr(
        "polylogue.sources.source_parsing.antigravity.iter_language_server_export_results",
        unavailable_export,
    )
    caplog.set_level(logging.WARNING, logger="polylogue.sources.source_parsing")

    sessions = list(iter_antigravity_language_server_sessions(Source(name="antigravity", path=root)))

    assert sessions == []
    assert "antigravity_coverage_gap" in caplog.messages[-1]


def test_source_census_accounts_for_all_roles_and_unknown_items(tmp_path: Path) -> None:
    root = tmp_path / "antigravity"
    (root / "conversations").mkdir(parents=True)
    (root / "conversations" / "cascade.pb").write_bytes(b"opaque")
    (root / "brain" / "work").mkdir(parents=True)
    (root / "brain" / "work" / "plan.md").write_text("# plan", encoding="utf-8")
    (root / "brain" / "work" / "plan.md.metadata.json").write_text("{}", encoding="utf-8")
    (root / "settings" / "opaque.bin").parent.mkdir()
    (root / "settings" / "opaque.bin").write_bytes(b"unknown")

    source_census = antigravity.census_source(root)
    assert source_census.counts == {
        antigravity.AntigravitySourceRole.CONVERSATION_PROTOBUF: 1,
        antigravity.AntigravitySourceRole.BRAIN_DOCUMENT: 1,
        antigravity.AntigravitySourceRole.METADATA_SIDECAR: 1,
        antigravity.AntigravitySourceRole.UNKNOWN: 1,
    }
    assert source_census.unknown_count == 1
    assert source_census.unexplained_items == ()
    source_census.assert_conserved()

    root_census = census_source_root(root, provider=Provider.ANTIGRAVITY)
    assert root_census.candidate_count == 4
    assert root_census.disposition_counts == {"session": 1, "non_session": 2, "unsupported": 1}
    assert root_census.is_complete


def test_source_census_rejects_mutation_during_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "antigravity"
    root.mkdir()
    source = root / "settings.bin"
    source.write_bytes(b"before")
    original_digest = antigravity._file_digest

    def digest_then_mutate(path: Path) -> str:
        digest = original_digest(path)
        path.write_bytes(b"after")
        return digest

    monkeypatch.setattr(antigravity, "_file_digest", digest_then_mutate)
    with pytest.raises(antigravity.AntigravitySourceMutationError):
        antigravity.census_source(root)
