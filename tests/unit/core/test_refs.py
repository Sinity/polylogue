from __future__ import annotations

import pytest

from polylogue.core.refs import EvidenceRef, ObjectRef


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
        ("check-run:ruff check", "check-run", "ruff check", ()),
        ("github-issue:Sinity/polylogue#1883", "github-issue", "Sinity/polylogue#1883", ()),
        ("github-review:1911", "github-review", "1911", ()),
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
