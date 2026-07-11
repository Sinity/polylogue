from __future__ import annotations

from types import SimpleNamespace

import pytest

from polylogue.context.compiler import (
    ContextImage,
    ContextSegment,
    ContextSpec,
    compile_query_unit_context_segment,
    context_snapshot_record_from_image,
)
from polylogue.core.refs import EvidenceRef, ObjectRef


def test_context_spec_allows_unit_query_only_recipes() -> None:
    spec = ContextSpec(unit_queries=("runs where session.id:codex-session:compiler",), read_views=())

    assert spec.seed_refs == ()
    assert spec.read_views == ()
    assert spec.unit_queries == ("runs where session.id:codex-session:compiler",)


def test_query_unit_context_segment_projects_refs() -> None:
    envelope = SimpleNamespace(
        model_dump=lambda **_: {
            "unit": "run",
            "query": "runs where session.id:codex-session:compiler",
            "total": 1,
            "items": [
                {
                    "unit": "run",
                    "run_ref": "run:codex-session:compiler",
                    "session_id": "codex-session:compiler",
                    "context_snapshot_ref": "context-snapshot:codex-session:compiler:session_start",
                    "lineage_refs": ["run:codex-session:compiler"],
                    "transcript_ref": "codex-session:compiler",
                    "evidence_refs": ["codex-session:compiler::m1"],
                    "summary": "main run",
                }
            ],
        }
    )

    segment = compile_query_unit_context_segment(envelope)

    assert segment.kind == "query_unit"
    assert segment.object_refs == (
        ObjectRef(kind="run", object_id="codex-session:compiler"),
        ObjectRef(kind="context-snapshot", object_id="codex-session:compiler:session_start"),
        ObjectRef(kind="session", object_id="codex-session:compiler"),
    )
    assert segment.evidence_refs == (
        EvidenceRef(session_id="codex-session:compiler", message_id="m1"),
        EvidenceRef(session_id="codex-session:compiler"),
    )


def _context_image() -> ContextImage:
    segment = ContextSegment(
        segment_id="query-unit:run:abc123",
        kind="query_unit",
        title="runs where session.id:codex-session:compiler",
        markdown="- run:codex-session:compiler: main run",
        payload_kind="run",
        object_refs=(ObjectRef(kind="run", object_id="codex-session:compiler"),),
        evidence_refs=(EvidenceRef(session_id="codex-session:compiler"),),
        token_estimate=4,
    )
    return ContextImage(
        spec=ContextSpec(
            purpose="handoff",
            unit_queries=("runs where session.id:codex-session:compiler",),
            read_views=(),
        ),
        segments=(segment,),
        object_refs=segment.object_refs,
        evidence_refs=segment.evidence_refs,
        token_estimate=segment.token_estimate,
    )


def test_context_snapshot_record_is_explicit_delivery_boundary() -> None:
    image = _context_image()

    record = context_snapshot_record_from_image(
        image,
        boundary="handoff",
        run_ref="run:local-review",
    )
    record_again = context_snapshot_record_from_image(
        image,
        boundary="handoff",
        run_ref="run:local-review",
    )
    different_boundary = context_snapshot_record_from_image(
        image,
        boundary="review",
        run_ref="run:local-review",
    )

    assert record.snapshot_ref.startswith("context-snapshot:")
    assert record.snapshot_ref == record_again.snapshot_ref
    assert record.snapshot_ref != different_boundary.snapshot_ref
    assert record.run_ref == "run:local-review"
    assert record.boundary == "handoff"
    assert record.inheritance_mode == "explicit"
    assert record.segment_refs == ("query-unit:run:abc123",)
    assert record.evidence_refs == image.evidence_refs
    assert record.metadata["purpose"] == "handoff"
    assert record.metadata["read_views"] == "[]"
    assert record.metadata["unit_queries"] == '["runs where session.id:codex-session:compiler"]'
    assert record.metadata["token_estimate"] == str(image.token_estimate)
    assert record.metadata["include_candidates"] == "false"
    assert len(record.metadata["context_image_sha256"]) == 64

    changed_image = image.model_copy(
        update={"segments": (image.segments[0].model_copy(update={"markdown": "changed exact text"}),)}
    )
    changed_record = context_snapshot_record_from_image(
        changed_image,
        boundary="handoff",
        run_ref="run:local-review",
    )
    assert changed_record.snapshot_ref != record.snapshot_ref


def test_context_snapshot_record_requires_delivery_boundary() -> None:
    image = _context_image()

    with pytest.raises(ValueError, match="delivery boundary"):
        context_snapshot_record_from_image(image, boundary="")
    with pytest.raises(ValueError, match="delivery boundary"):
        context_snapshot_record_from_image(image, boundary="   ")


def test_context_spec_requires_an_explicit_seed() -> None:
    with pytest.raises(ValueError, match="requires seed_query, seed_refs, or unit_queries"):
        ContextSpec()

    spec = ContextSpec(seed_query="sessions where repo:polylogue", max_tokens=1200)
    assert spec.seed_query == "sessions where repo:polylogue"
    assert spec.include_candidates is False
