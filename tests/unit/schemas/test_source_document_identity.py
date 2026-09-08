"""Document revision identity preserves chats, profiles, and artifact families."""

from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument
from polylogue.schemas.source_document_identity import native_document_identity


def test_gemini_chat_revisions_share_identity_without_collapsing_siblings(tmp_path: Path) -> None:
    """Path identity double-counts copies; sessionId alone loses sibling chats."""
    original: JSONDocument = {"sessionId": "process", "kind": "main", "startTime": "2026-01-01T00:00:00Z"}
    updated: JSONDocument = {**original, "lastUpdated": "2026-01-02T00:00:00Z"}
    identity = native_document_identity(Provider.GEMINI_CLI, original, tmp_path / "first.json")
    assert identity is not None
    assert native_document_identity(Provider.GEMINI_CLI, updated, tmp_path / "copy.json") == identity
    assert (
        native_document_identity(Provider.GEMINI_CLI, {**original, "kind": "subagent"}, tmp_path / "child.json")
        != identity
    )
    assert (
        native_document_identity(
            Provider.GEMINI_CLI, {**original, "startTime": "2026-01-01T01:00:00Z"}, tmp_path / "reset.json"
        )
        != identity
    )


def test_hermes_snapshot_copies_share_identity_within_their_profile(tmp_path: Path) -> None:
    """Raw session IDs alone collapse separate profiles; paths split saved copies."""
    payload: JSONDocument = {"session_id": "session", "messages": []}
    identity = native_document_identity(Provider.HERMES, payload, tmp_path / "profile-a/sessions/session.json")
    assert identity is not None
    assert (
        native_document_identity(Provider.HERMES, payload, tmp_path / "profile-a/sessions/saved/copy.json") == identity
    )
    assert native_document_identity(Provider.HERMES, payload, tmp_path / "profile-b/sessions/session.json") != identity
    trajectory: JSONDocument = {
        "session_id": "session",
        "schema_version": "ATIF-v1",
        "agent": {},
        "steps": [],
    }
    assert native_document_identity(Provider.HERMES, trajectory, tmp_path / "profile-a/trajectory.json") != identity


def test_declared_document_ids_are_path_independent(tmp_path: Path) -> None:
    """Fallback paths must not split copies when the provider supplies an ID."""
    cases: tuple[tuple[Provider, JSONDocument], ...] = (
        (Provider.GEMINI, {"id": "prompt"}),
        (Provider.DRIVE, {"id": "prompt"}),
        (Provider.ANTIGRAVITY, {"cascadeId": "cascade"}),
    )
    for provider, payload in cases:
        assert native_document_identity(provider, payload, tmp_path / "a.json") == native_document_identity(
            provider, payload, tmp_path / "b.json"
        )
        assert native_document_identity(provider, {}, tmp_path / "a.json") is None


def test_source_inference_replaces_copied_chat_revision_and_keeps_subagent(tmp_path: Path) -> None:
    """An unwired native identity helper counts all three captures as current."""
    import json

    from polylogue.schemas.generation.evidence import SchemaEvidence
    from polylogue.schemas.source_inference import SchemaSourceInput, infer_sources

    source_root = tmp_path / "sources"
    source_root.mkdir()
    common: JSONDocument = {
        "sessionId": "process",
        "kind": "main",
        "startTime": "2026-01-01T00:00:00Z",
        "messages": [{"id": "message", "type": "user", "content": "neutral"}],
    }
    for name, payload in (
        ("original", {**common, "lastUpdated": "2026-01-01T00:01:00Z"}),
        ("copy", {**common, "lastUpdated": "2026-01-01T00:01:00.100Z", "latest_copy": True}),
        ("child", {**common, "kind": "subagent", "lastUpdated": "2026-01-01T00:03:00Z"}),
    ):
        (source_root / f"{name}.json").write_text(json.dumps(payload))
    result = infer_sources(
        (SchemaSourceInput("gemini-cli", source_root),), cache_path=tmp_path / "evidence.sqlite", max_workers=1
    )
    assert result.record_count == 2
    evidence = SchemaEvidence.from_json(result.evidence_by_element["session_document"][0])
    assert evidence.current_source_count == 2
    assert evidence.historical_source_count == 1
    assert "$.latest_copy" in evidence.fields
