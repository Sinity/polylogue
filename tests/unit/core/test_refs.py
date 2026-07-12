from __future__ import annotations

import pytest

from polylogue.core.refs import (
    EvidenceRef,
    ObjectRef,
    delegation_edge_object_id,
    normalize_object_ref_text,
    normalize_public_ref_text,
    parse_delegation_edge_object_id,
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
        ("attachment:sha256:abc", "attachment", "sha256:abc", ()),
        ("paste_span:codex-session:demo:m1:0:4", "paste_span", "codex-session:demo:m1:0:4", ()),
        ("work_event:codex-session:demo:work_event:1", "work_event", "codex-session:demo:work_event:1", ()),
        ("phase:codex-session:demo:phase:1", "phase", "codex-session:demo:phase:1", ()),
        ("thread:codex-session:demo", "thread", "codex-session:demo", ()),
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
        ("transform:session_digest_v0@v1", "transform", "session_digest_v0@v1", ()),
        ("assertion:note-1", "assertion", "note-1", ()),
        ("saved_view:view-1", "saved_view", "view-1", ()),
        ("recall_pack:pack-1", "recall_pack", "pack-1", ()),
        ("agent:codex/main", "agent", "codex/main", ()),
        ("tool-call:codex-session:demo:tool-1", "tool-call", "codex-session:demo:tool-1", ()),
        ("subagent-report:codex-session:demo:tool-2", "subagent-report", "codex-session:demo:tool-2", ()),
        # Analysis-provenance object kinds (polylogue-rxdo.1). Resolvers are
        # stubbed pending the storage tiers; the ref shape round-trips today.
        ("query:sha256:deadbeef", "query", "sha256:deadbeef", ()),
        ("query-run:sha256:deadbeef:run-1", "query-run", "sha256:deadbeef:run-1", ()),
        ("result-set:sha256:deadbeef:run-1:result-1", "result-set", "sha256:deadbeef:run-1:result-1", ()),
        ("finding:finding-hash-1", "finding", "finding-hash-1", ()),
        ("cohort:cohort-1", "cohort", "cohort-1", ()),
        ("analysis:analysis-1", "analysis", "analysis-1", ()),
        ("annotation-batch:batch-1", "annotation-batch", "batch-1", ()),
        # polylogue-lph4: delegation attempt identity. Action-observed rows
        # key off the instruction tool-use block id verbatim (already
        # colon-bearing, opaque); edge-only rows use the deterministic
        # edge:<parent>::<child> relation identity (polylogue-y964 states
        # with no parent-side dispatch action to key off).
        (
            "delegation:claude-code-session:parent:dispatch:0",
            "delegation",
            "claude-code-session:parent:dispatch:0",
            (),
        ),
        (
            "delegation:edge:codex-session:parent::codex-session:child",
            "delegation",
            "edge:codex-session:parent::codex-session:child",
            (),
        ),
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


@pytest.mark.parametrize(
    "raw",
    [
        "query:sha256:deadbeef",
        "query-run:sha256:deadbeef:run-1",
        "result-set:sha256:deadbeef:run-1:result-1",
        "finding:finding-hash-1",
        "cohort:cohort-1",
        "analysis:analysis-1",
        "annotation-batch:batch-1",
    ],
)
def test_normalize_object_ref_text_accepts_analysis_provenance_kinds(raw: str) -> None:
    """polylogue-rxdo.1: the analysis-provenance kinds normalize/round-trip today.

    Resolution is stubbed pending storage (see resolve_ref tests); the ref
    grammar itself must accept these kinds without any special-casing.
    """
    assert normalize_object_ref_text(raw) == raw
    assert normalize_public_ref_text(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [
        "query:",
        "query-run:",
        "result-set:",
        "finding:",
        "cohort:",
        "analysis:",
        "annotation-batch:",
    ],
)
def test_object_ref_rejects_empty_ids_for_analysis_provenance_kinds(raw: str) -> None:
    """New kinds reuse the shared empty-id guard; they don't weaken validation."""
    with pytest.raises(ValueError):
        normalize_object_ref_text(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "delegation:claude-code-session:parent:dispatch:0",
        "delegation:edge:codex-session:parent::codex-session:child",
    ],
)
def test_normalize_object_ref_text_accepts_delegation_kind(raw: str) -> None:
    assert normalize_object_ref_text(raw) == raw
    assert normalize_public_ref_text(raw) == raw


def test_object_ref_rejects_empty_id_for_delegation_kind() -> None:
    with pytest.raises(ValueError):
        normalize_object_ref_text("delegation:")


def test_delegation_edge_object_id_round_trips() -> None:
    object_id = delegation_edge_object_id("claude-code-session:parent", "claude-code-session:child")

    assert object_id == "edge:claude-code-session:parent::claude-code-session:child"
    assert parse_delegation_edge_object_id(object_id) == ("claude-code-session:parent", "claude-code-session:child")

    ref = ObjectRef(kind="delegation", object_id=object_id)
    assert ref.format() == f"delegation:{object_id}"
    assert ObjectRef.parse(ref.format()) == ref


@pytest.mark.parametrize(
    "object_id",
    [
        "claude-code-session:parent:dispatch:0",  # action-observed shape, no edge: prefix
        "edge:only-one-segment",
        "edge:",
    ],
)
def test_parse_delegation_edge_object_id_returns_none_for_non_edge_shapes(object_id: str) -> None:
    assert parse_delegation_edge_object_id(object_id) is None


@pytest.mark.parametrize(
    ("parent_session_id", "child_session_id"),
    [("", "child"), ("parent", "")],
)
def test_delegation_edge_object_id_rejects_empty_segments(parent_session_id: str, child_session_id: str) -> None:
    with pytest.raises(ValueError):
        delegation_edge_object_id(parent_session_id, child_session_id)
