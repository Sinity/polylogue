"""Observe actual closed-file Raw parse bodies without changing their authority."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from polylogue import logging as plog
from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.core import work_observation
from polylogue.core.enums import Provider, ValidationMode
from polylogue.storage.derived import raw


def test_neutral_parse_body_refs_are_distinct_but_validation_mode_does_not_change_parse_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(work_observation, "ENABLED", True)
    payload = (
        b'{"type":"user","message":{"role":"user","content":"synthetic"},"uuid":"m0",'
        b'"sessionId":"observed","timestamp":"2026-06-02T00:00:00Z","isSidechain":false}\n'
    )
    descriptor = (
        Provider.CLAUDE_CODE,
        sha256(payload).hexdigest(),
        "neutral.jsonl",
        RawRevisionKind.FULL,
        len(payload),
    )
    operand = raw._NeutralParserOperand(descriptor, None, None, None, None, None)
    previous = plog.set_level("info")
    try:
        with plog.capture() as records:
            for ordinal, mode in enumerate((ValidationMode.OFF, ValidationMode.ADVISORY)):
                capture_root = tmp_path / f"capture-{ordinal}"
                capture_root.mkdir()
                source = capture_root / "neutral.jsonl"
                source.write_bytes(payload)
                captured = raw._CapturedNeutralRaw(descriptor, None, None, None, None, None, source)
                key = raw._neutral_artifact_key("raw-one", operand, mode)
                artifact = raw._parse_captured_neutral("raw-one", captured, key, {}, tmp_path)
                try:
                    assert artifact.error is None
                    assert artifact.blob_hash == descriptor[1]
                    assert artifact.observed_parse_key is not None
                    assert artifact.observed_parse_artifact_ref is not None
                    artifact.verify_files(full=True)
                finally:
                    artifact.discard()
        bodies = [r for r in records if r["event"] == "work.body.ok"]
        entries = [r for r in records if r["event"] == "work.body.started"]
        assert len(entries) == len(bodies) == 2
        assert all(r["component"] == "raw" and r["operation"] == "parse" for r in bodies)
        assert bodies[0]["work_key"] == bodies[1]["work_key"]
        assert bodies[0]["artifact_ref"] != bodies[1]["artifact_ref"]
    finally:
        plog.set_level(previous)


def test_neutral_parse_off_does_not_observe_recipe_or_artifact_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(work_observation, "ENABLED", False)

    def forbidden() -> str:
        raise AssertionError("disabled diagnostics observed a parse recipe")

    monkeypatch.setattr(raw, "raw_authority_parser_fingerprint", forbidden)
    payload = b'{"type":"user","message":{"role":"user","content":"synthetic"},"uuid":"m0"}\n'
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    source = capture_root / "neutral.jsonl"
    source.write_bytes(payload)
    descriptor = (
        Provider.CLAUDE_CODE,
        sha256(payload).hexdigest(),
        "neutral.jsonl",
        RawRevisionKind.FULL,
        len(payload),
    )
    captured = raw._CapturedNeutralRaw(descriptor, None, None, None, None, None, source)
    operand = raw._NeutralParserOperand(descriptor, None, None, None, None, None)
    with plog.capture() as records:
        artifact = raw._parse_captured_neutral(
            "raw-one", captured, raw._neutral_artifact_key("raw-one", operand, ValidationMode.OFF), {}, tmp_path
        )
    try:
        assert artifact.error is None
        assert artifact.observed_parse_key is artifact.observed_parse_artifact_ref is None
        assert not any(str(r["event"]).startswith("work.") for r in records)
    finally:
        artifact.discard()
