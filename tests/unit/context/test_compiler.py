from __future__ import annotations

import pytest

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import Session
from polylogue.context.compiler import compile_recovery_context
from polylogue.core.enums import Origin
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


def test_compiler_rejects_parallel_packet_shape_for_non_packet_report() -> None:
    digest = compile_recovery_digest(_session())

    with pytest.raises(ValueError, match="work_packet may only be supplied"):
        compile_recovery_context(digest, report="continue", work_packet=digest.work_packet())

    with pytest.raises(ValueError, match="unsupported recovery report preset"):
        compile_recovery_context(digest, report="incident")
