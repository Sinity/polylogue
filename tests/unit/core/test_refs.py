from __future__ import annotations

import pytest

from polylogue.core.refs import (
    EvidenceRef,
    ObjectRef,
    normalize_object_ref_text,
    normalize_public_ref_text,
    parse_public_ref,
)


@pytest.mark.parametrize(
    ("raw", "kind", "object_id", "qualifiers"),
    [
        ("session:session-1", "session", "session-1", ()),
        ("session:codex-session:demo", "session", "codex-session:demo", ()),
        ("message:m-7", "message", "m-7", ()),
        ("message:codex-session:demo:message-1", "message", "codex-session:demo:message-1", ()),
        ("block:m1:2", "block", "m1", ("2",)),
        ("block:codex-session:demo:message-1:2", "block", "codex-session:demo:message-1", ("2",)),
        ("file:polylogue/insights/transforms.py", "file", "polylogue/insights/transforms.py", ()),
        ("branch:feature/demo", "branch", "feature/demo", ()),
        ("commit:abc1234", "commit", "abc1234", ()),
        ("commit:a8cd1c1516b29068ec9ce1493f262d663407ffa5", "commit", "a8cd1c1516b29068ec9ce1493f262d663407ffa5", ()),
        ("check-run:ruff check", "check-run", "ruff check", ()),
        ("github-issue:Sinity/polylogue#1883", "github-issue", "Sinity/polylogue#1883", ()),
        ("github-review:1911", "github-review", "1911", ()),
        ("user:sinity", "user", "sinity", ()),
        ("repo:polylogue", "repo", "polylogue", ()),
        ("insight:session-1", "insight", "session-1", ()),
        ("transform:recovery_digest_v0@v1", "transform", "recovery_digest_v0@v1", ()),
        ("assertion:note-1", "assertion", "note-1", ()),
        ("saved_view:view-1", "saved_view", "view-1", ()),
        ("recall_pack:pack-1", "recall_pack", "pack-1", ()),
        ("agent:codex/main", "agent", "codex/main", ()),
        ("tool-call:codex-session:demo:tool-1", "tool-call", "codex-session:demo:tool-1", ()),
        ("subagent-report:codex-session:demo:tool-2", "subagent-report", "codex-session:demo:tool-2", ()),
    ],
)
def test_object_ref_parses_and_formats_existing_assertion_ref_shapes(
    raw: str,
    kind: str,
    object_id: str,
    qualifiers: tuple[str, ...],
) -> None:
    ref = ObjectRef.parse(raw)

    assert ref.kind == kind
    assert ref.object_id == object_id
    assert ref.qualifiers == qualifiers
    assert ref.format() == raw


@pytest.mark.parametrize("raw", ["session", "session:", "message:m1:", "block:m1:", "unknown:x"])
def test_object_ref_rejects_unsupported_or_lossy_shapes(raw: str) -> None:
    with pytest.raises(ValueError):
        ObjectRef.parse(raw)


@pytest.mark.parametrize(
    ("raw", "session_id", "message_id", "block_index", "kind", "object_ref"),
    [
        ("codex-session:demo", "codex-session:demo", None, None, "session", "session:codex-session:demo"),
        ("codex-session:demo::m2", "codex-session:demo", "m2", None, "message", "message:m2"),
        ("codex-session:demo::m2::0", "codex-session:demo", "m2", 0, "block", "block:m2:0"),
    ],
)
def test_evidence_ref_parses_formats_and_projects_to_object_ref(
    raw: str,
    session_id: str,
    message_id: str | None,
    block_index: int | None,
    kind: str,
    object_ref: str,
) -> None:
    ref = EvidenceRef.parse(raw)

    assert ref.session_id == session_id
    assert ref.message_id == message_id
    assert ref.block_index == block_index
    assert ref.ref_kind == kind
    assert ref.format() == raw
    assert ref.to_object_ref().format() == object_ref


@pytest.mark.parametrize("raw", ["", "session::", "session::m::-1", "session::m::not-int", "a::b::c::d"])
def test_evidence_ref_rejects_unsupported_or_lossy_shapes(raw: str) -> None:
    with pytest.raises(ValueError):
        EvidenceRef.parse(raw)


def test_public_ref_parser_prefers_object_refs_and_accepts_recovery_evidence_refs() -> None:
    object_ref = parse_public_ref("session:codex-session:demo")
    evidence_ref = parse_public_ref("codex-session:demo::m2::0")

    assert isinstance(object_ref, ObjectRef)
    assert object_ref.format() == "session:codex-session:demo"
    assert isinstance(evidence_ref, EvidenceRef)
    assert evidence_ref.to_object_ref().format() == "block:m2:0"
    assert normalize_object_ref_text("repo:polylogue") == "repo:polylogue"
    assert normalize_public_ref_text("codex-session:demo::m2") == "codex-session:demo::m2"


def test_object_ref_normalizer_rejects_unscoped_raw_strings() -> None:
    with pytest.raises(ValueError):
        normalize_object_ref_text("demo-fixture-world")
