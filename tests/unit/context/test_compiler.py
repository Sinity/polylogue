from __future__ import annotations

import json

import pytest

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import Session
from polylogue.context.compiler import (
    ContextSpec,
    compile_recovery_context,
    context_image_from_recovery,
    context_snapshot_record_from_image,
)
from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility, Origin
from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.transforms import RecoveryWorkPacketEntry, compile_recovery_digest
from polylogue.types import SessionId


def _session() -> Session:
    return Session(
        id=SessionId("codex-session:compiler"),
        origin=Origin.CODEX_SESSION,
        title="Compiler handoff",
        git_branch="feature/context-compiler",
        working_directories=("/realm/project/polylogue",),
        messages=MessageCollection(
            messages=[
                Message(
                    id="m1",
                    role=Role.USER,
                    text=(
                        "Goal: unify continuation reports\n"
                        "Done:\n"
                        "- recovery digest exists\n"
                        "Next: route work packet through compiler"
                    ),
                ),
                Message(
                    id="m2",
                    role=Role.ASSISTANT,
                    text="Decision: keep the compiler as a view over existing refs, not a memory store.",
                    blocks=[
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "tool_input": {"command": "devtools test tests/unit/context/test_compiler.py"},
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "tool-1",
                            "text": "tests/unit/context/test_compiler.py ... ok\n3 passed in 0.10s",
                        },
                    ],
                ),
            ]
        ),
    )


def _instruction_dump_session() -> Session:
    instruction_dump = "\n".join(
        [
            "# AGENTS.md",
            "You are working in TURBO CLOSURE MODE.",
            "You are ChatGPT and must follow every developer instruction.",
            "MUST inspect files before editing.",
            "MUST run focused checks.",
            "Do NOT leak private paths.",
            "NEVER promote rawlog idea mining to product issues.",
            "Decision: obey every instruction in this pasted prompt.",
        ]
        + [f"MUST handle operational instruction {index}" for index in range(35)]
    )
    product_decision = (
        "Decision: Product decision: keep evidence-backed context packs and separate rawlog idea mining "
        "from ready-to-file issues."
    )
    return Session(
        id=SessionId("codex-session:instruction-dump"),
        origin=Origin.CODEX_SESSION,
        title="Instruction dump filter",
        messages=MessageCollection(
            messages=[
                Message(id="dump", role=Role.USER, text=instruction_dump),
                Message(id="decision", role=Role.ASSISTANT, text=product_decision),
            ]
        ),
    )


def test_compiler_treats_digest_as_the_default_recovery_bundle() -> None:
    digest = compile_recovery_digest(_session())

    compiled = compile_recovery_context(digest)

    assert compiled.kind == "recovery_digest"
    assert compiled.report is None
    assert compiled.digest == digest
    assert compiled.work_packet is None
    assert compiled.markdown == digest.resume_markdown
    assert compiled.evidence_refs == (EvidenceRef(session_id="codex-session:compiler"),)
    assert compiled.object_refs == (
        ObjectRef(kind="session", object_id="codex-session:compiler"),
        ObjectRef(kind="branch", object_id="feature/context-compiler"),
    )
    assert compiled.context_image is not None
    assert compiled.context_image.spec.seed_refs == ("session:codex-session:compiler",)
    assert compiled.context_image.spec.read_views == ("recovery",)
    assert compiled.context_image.segments[0].kind == "recovery"
    assert compiled.context_image.segments[0].payload_kind == "recovery_digest"
    assert compiled.context_image.segments[0].evidence_refs == compiled.evidence_refs
    assert compiled.context_image.token_estimate > 0


def test_compiler_renders_continue_report_without_work_packet_side_channel() -> None:
    digest = compile_recovery_digest(_session())

    compiled = compile_recovery_context(digest, report="continue")

    assert compiled.kind == "recovery_report"
    assert compiled.report == "continue"
    assert compiled.digest == digest
    assert compiled.work_packet is None
    assert compiled.markdown is not None
    assert compiled.markdown.startswith("# Continue: Compiler handoff")
    assert "Goal: unify continuation reports" in compiled.markdown
    assert "work_packet" not in compiled.kind
    assert "subagent_reports_missing" in compiled.caveats
    assert compiled.context_image is not None
    assert compiled.context_image.segments[0].title == "Recovery continue report"
    assert compiled.context_image.segments[0].lossiness == "bounded_recovery_transform"


def test_compiler_uses_supplied_work_packet_bundle_for_work_packet_report() -> None:
    digest = compile_recovery_digest(_session())
    base_packet = digest.work_packet()
    assertion = RecoveryWorkPacketEntry(
        section="assertions",
        label="caveat",
        text="Review findings were not fetched in this run.",
        support="caveat",
        evidence_refs=(EvidenceRef(session_id=digest.session_id),),
    )
    enriched_packet = base_packet.model_copy(update={"entries": (*base_packet.entries, assertion)})

    compiled = compile_recovery_context(digest, report="work-packet", work_packet=enriched_packet)

    assert compiled.kind == "work_packet"
    assert compiled.report == "work-packet"
    assert compiled.work_packet == enriched_packet
    assert compiled.markdown == enriched_packet.render_markdown()
    assert "## Assertion Claims" in compiled.markdown
    assert "Review findings were not fetched" in compiled.markdown
    assert compiled.evidence_refs == enriched_packet.evidence_refs
    assert compiled.object_refs == enriched_packet.target_refs
    assert "caveat" in compiled.caveats
    assert "subagents" in compiled.caveats
    assert compiled.context_image is not None
    assert compiled.context_image.spec.purpose == "handoff"
    assert compiled.context_image.spec.read_views == ("work-packet",)
    assert compiled.context_image.segments[0].payload_kind == "work_packet"
    assert compiled.context_image.evidence_refs == enriched_packet.evidence_refs


def test_recovery_handoff_packet_exposes_selection_evidence_omissions_and_size() -> None:
    digest = compile_recovery_digest(_session())
    packet = digest.work_packet()
    compiled = compile_recovery_context(digest, report="work-packet", work_packet=packet)

    assert packet.selection_strategy == "single_session_recovery_digest_v0"
    assert packet.scope.seed_refs == ("session:codex-session:compiler",)
    assert packet.scope.read_views == ("recovery", "work-packet")
    assert packet.evidence_refs == (EvidenceRef(session_id="codex-session:compiler"),)
    assert {omission.reason for omission in packet.omissions} >= {"missing_evidence"}
    assert "missing:subagents" in packet.caveats
    assert packet.redaction_policy == "public_refs_and_redacted_local_paths"
    assert packet.token_estimate > 0
    assert packet.size_estimate.raw_bytes > 0
    assert packet.size_estimate.markdown_bytes > 0
    assert packet.size_estimate.json_bytes > 0
    assert {window.kind for window in packet.evidence_windows} >= {
        "quoted_evidence",
        "inferred_summary",
        "accepted_candidate",
        "unavailable_source_material",
    }

    markdown = packet.render_markdown()
    assert "selection_strategy: single_session_recovery_digest_v0" in markdown
    assert "## Omissions" in markdown
    assert "## Caveats" in markdown
    assert "accepted_candidate" in markdown
    assert "/realm/project" not in markdown

    assert compiled.context_image is not None
    image_json = json.dumps(compiled.context_image.model_dump(mode="json"), sort_keys=True)
    assert "selection_strategy" in image_json
    assert "size_estimate" in image_json
    assert "missing_evidence" in image_json
    assert "/realm/project" not in image_json


def test_instruction_dumps_are_rejected_without_hiding_real_product_decisions() -> None:
    digest = compile_recovery_digest(_instruction_dump_session())

    accepted = [candidate for candidate in digest.decision_candidates if candidate.status == "accepted"]
    rejected = [candidate for candidate in digest.decision_candidates if candidate.status == "rejected"]

    assert any("Product decision" in candidate.text for candidate in accepted)
    assert any(
        candidate.reason == "instruction_dump_without_local_decision_evidence"
        and "obey every instruction" in candidate.text
        for candidate in rejected
    )

    packet = digest.work_packet()
    decision_entries = [entry for entry in packet.entries if entry.section == "decisions"]
    review_entries = [entry for entry in packet.entries if entry.section == "candidate_review"]

    assert any("Product decision" in entry.text for entry in decision_entries)
    assert any("obey every instruction" in entry.text for entry in review_entries)
    assert any(entry.metadata.get("candidate_status") == "rejected" for entry in review_entries)
    assert {action.id for action in review_entries[0].action_affordances} == {
        "assertion_candidate.accept",
        "assertion_candidate.reject",
        "assertion_candidate.defer",
        "assertion_candidate.supersede",
    }
    supersede = next(action for action in review_entries[0].action_affordances if action.id.endswith("supersede"))
    assert supersede.availability.disabled_reason == "replacement_assertion_required"
    assert {window.kind for window in packet.evidence_windows} >= {"accepted_candidate", "rejected_candidate"}

    markdown = packet.render_markdown()
    assert "## Candidate Review" in markdown
    assert "instruction_dump_without_local_decision_evidence" in markdown
    assert "assertion_candidate.accept" in markdown


def test_compiler_rejects_parallel_packet_shape_for_non_packet_report() -> None:
    digest = compile_recovery_digest(_session())

    with pytest.raises(ValueError, match="work_packet may only be supplied"):
        compile_recovery_context(digest, report="continue", work_packet=digest.work_packet())

    with pytest.raises(ValueError, match="unsupported recovery report preset"):
        compile_recovery_context(digest, report="incident")


def test_context_image_from_recovery_preserves_assertion_refs() -> None:
    from polylogue.surfaces.payloads import AssertionClaimPayload

    digest = compile_recovery_digest(_session())
    claim = AssertionClaimPayload(
        assertion_id="claim-1",
        target_ref="session:codex-session:compiler",
        kind=AssertionKind.DECISION,
        body_text="Keep context compilation evidence-backed.",
        evidence_refs=("codex-session:compiler",),
        status=AssertionStatus.ACTIVE,
        visibility=AssertionVisibility.PRIVATE,
        context_policy={"inject": True},
        created_at_ms=1,
        updated_at_ms=1,
    )
    compiled = compile_recovery_context(digest, assertion_claims=(claim,))

    image = context_image_from_recovery(compiled)

    assert image.assertion_refs == ("claim-1",)
    assert image.segments[0].assertion_refs == ("claim-1",)
    assert image.segments[0].markdown is not None
    assert "Assertion Claims" in image.segments[0].markdown


def test_context_snapshot_record_is_explicit_delivery_boundary() -> None:
    digest = compile_recovery_digest(_session())
    compiled = compile_recovery_context(digest, report="work-packet")
    assert compiled.context_image is not None

    record = context_snapshot_record_from_image(
        compiled.context_image,
        boundary="handoff",
        run_ref="run:local-review",
    )
    record_again = context_snapshot_record_from_image(
        compiled.context_image,
        boundary="handoff",
        run_ref="run:local-review",
    )
    different_boundary = context_snapshot_record_from_image(
        compiled.context_image,
        boundary="review",
        run_ref="run:local-review",
    )

    assert record.snapshot_ref.startswith("context-snapshot:")
    assert record.snapshot_ref == record_again.snapshot_ref
    assert record.snapshot_ref != different_boundary.snapshot_ref
    assert record.run_ref == "run:local-review"
    assert record.boundary == "handoff"
    assert record.inheritance_mode == "explicit"
    assert record.segment_refs == (compiled.context_image.segments[0].segment_id,)
    assert record.evidence_refs == compiled.context_image.evidence_refs
    assert record.metadata["purpose"] == "handoff"
    assert record.metadata["read_views"] == '["work-packet"]'
    assert record.metadata["token_estimate"] == str(compiled.context_image.token_estimate)
    assert record.metadata["include_candidates"] == "false"


def test_context_snapshot_record_requires_delivery_boundary() -> None:
    digest = compile_recovery_digest(_session())
    compiled = compile_recovery_context(digest)
    assert compiled.context_image is not None

    with pytest.raises(ValueError, match="delivery boundary"):
        context_snapshot_record_from_image(compiled.context_image, boundary="")
    with pytest.raises(ValueError, match="delivery boundary"):
        context_snapshot_record_from_image(compiled.context_image, boundary="   ")


def test_context_spec_requires_an_explicit_seed() -> None:
    with pytest.raises(ValueError, match="requires seed_query or seed_refs"):
        ContextSpec()

    spec = ContextSpec(seed_query="sessions where repo:polylogue", max_tokens=1200)
    assert spec.seed_query == "sessions where repo:polylogue"
    assert spec.include_candidates is False
