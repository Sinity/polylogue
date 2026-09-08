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
