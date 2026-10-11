"""Observe actual closed-file Raw parse bodies without changing their authority."""

from __future__ import annotations

import tempfile
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from polylogue import logging as plog
from polylogue.archive.revision_authority import RawRevisionKind, raw_authority_parser_fingerprint
from polylogue.core import work_observation
from polylogue.core.enums import Provider, ValidationMode
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.sources.sidecar_evidence import CapturedSidecarResolver, RetainedSidecarScope
from polylogue.storage.derived import raw
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead


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
                artifact = raw._parse_captured_neutral(
                    "raw-one",
                    captured,
                    key,
                    raw.NeutralPreparationContext(
                        CapturedSidecarResolver({}), raw_authority_parser_fingerprint(), SchemaRegistry(), 0
                    ),
                    tmp_path,
                )
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
            "raw-one",
            captured,
            raw._neutral_artifact_key("raw-one", operand, ValidationMode.OFF),
            raw.NeutralPreparationContext(CapturedSidecarResolver({}), None, SchemaRegistry(), 0),
            tmp_path,
        )
    try:
        assert artifact.error is None
        assert artifact.observed_parse_key is artifact.observed_parse_artifact_ref is None
        assert not any(str(r["event"]).startswith("work.") for r in records)
    finally:
        artifact.discard()


@pytest.mark.parametrize("changed", ["absence", "witness"])
def test_exact_sidecar_context_amortizes_only_within_owned_read(changed: str) -> None:
    """Same-path Raw revisions share construction; a fresh witness never does."""
    descriptor = (Provider.CLAUDE_CODE, "a" * 64, "session.jsonl", RawRevisionKind.FULL, 30)

    class Resolver:
        def __init__(self, scope: RetainedSidecarScope) -> None:
            self.scope = scope
            self.calls = 0

        def claude_code_scope(self, _path: str) -> RetainedSidecarScope:
            self.calls += 1
            return self.scope

    class Read:
        def __init__(self, scope: RetainedSidecarScope) -> None:
            self.resolver = Resolver(scope)
            self.resolver_calls = 0

        def retained_sidecar_resolver(self) -> Resolver:
            self.resolver_calls += 1
            return self.resolver

        def raw_profile_identity(self, _raw_id: str) -> None:
            return None

        def raw_revision_file_mtime(self, _raw_id: str) -> None:
            return None

        def raw_captured_zip_coordinate(self, _raw_id: str) -> None:
            return None

    initial_scope = RetainedSidecarScope(scope_key="owned", available=True, witness=(("file", "sidecar", "a"),))
    read = Read(initial_scope)
    owner = raw._NeutralOperandRead(cast(PreparedSessionSourceRead, read))
    first = owner.operand("one", descriptor=descriptor)
    second = owner.operand("two", descriptor=descriptor)
    assert read.resolver_calls == read.resolver.calls == 1
    assert first.sidecar_signature is second.sidecar_signature
    assert first.sidecar_signature_digest is second.sidecar_signature_digest
    assert first.sidecar_signature_bytes > 0
    key = raw._neutral_artifact_key("one", first, ValidationMode.ADVISORY)
    assert isinstance(key[0], tuple)
    assert key[0][-1] == ("sidecar-signature-digest", first.sidecar_signature_digest)
    changed_scope = (
        RetainedSidecarScope(scope_key="owned")
        if changed == "absence"
        else RetainedSidecarScope(scope_key="owned", available=True, witness=(("file", "sidecar", "b"),))
    )
    fresh_read = Read(changed_scope)
    fresh = raw._NeutralOperandRead(cast(PreparedSessionSourceRead, fresh_read)).operand("one", descriptor=descriptor)
    assert fresh_read.resolver.calls == 1
    assert fresh.sidecar_signature != first.sidecar_signature
    assert raw._neutral_parser_cache_identity("one", fresh) != raw._neutral_parser_cache_identity("one", first)
    assert raw._neutral_artifact_key("one", fresh, ValidationMode.ADVISORY) != key


def test_neutral_page_observes_distinct_backing_custody_until_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(work_observation, "ENABLED", True)
    monkeypatch.setattr(raw, "raw_authority_parser_fingerprint", lambda: "synthetic-context")
    owner = tempfile.TemporaryDirectory(dir=tmp_path)
    staged = Path(owner.name) / "input.jsonl"
    staged.write_bytes(b"neutral")
    descriptor = (Provider.CODEX, "a" * 64, "session.jsonl", RawRevisionKind.FULL, 7)
    capture = raw._CapturedNeutralRaw(descriptor, None, None, None, None, None, staged)
    operand = raw._NeutralParserOperand(descriptor, None, None, None, None, None)
    with plog.capture() as records:
        page = raw.NeutralRawPreparation(
            owner,
            {"one": capture, "two": capture},
            {"one": operand, "two": operand},
            {},
            {},
            {},
            sidecar_backings={staged: 7},
        )
        assert page.unique_backing_bytes == 7  # Repeated offered coordinates do not create another file.
        assert page.unique_context_count == 1
        assert staged.is_file()
        assert not [event for event in records if event["event"] == "storage.raw_observation.neutral_context"]
        page.observe_custody("acquire")
        page.close()
    events = [event for event in records if event["event"] == "storage.raw_observation.neutral_context"]
    assert [event["operation"] for event in events] == ["acquire", "retire"]
    assert all(event["outcome"] == "ok" for event in events)
    assert all(event["unique_backing_bytes"] == 7 for event in events)
    assert all(event["unique_context_count"] == 1 for event in events)
    assert not staged.exists()
    assert not page.contexts
    assert page.unique_backing_bytes == page.shared_context_bytes == 0
