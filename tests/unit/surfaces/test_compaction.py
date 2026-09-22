from __future__ import annotations

from polylogue.context.compiler import ContextImage
from polylogue.surfaces.compaction import CompactProjectionSpec, compact_sessions


def _sessions() -> list[dict[str, object]]:
    return [
        {
            "id": "codex:a",
            "messages": [
                {"id": "m1", "text": '{"protocol": true}', "material_origin": "runtime_protocol"},
                {
                    "id": "m2",
                    "text": "run tests failed",
                    "material_origin": "tool_result",
                    "blocks": [{"type": "tool_result", "tool_result_is_error": True, "tool_result_exit_code": 2}],
                },
                {"id": "m3", "text": "fixed the test and verify it", "material_origin": "assistant_authored"},
                {
                    "id": "m4",
                    "text": "success output",
                    "material_origin": "tool_result",
                    "blocks": [{"type": "tool_result", "tool_result_is_error": False, "tool_result_exit_code": 0}],
                },
            ],
        },
        {
            "id": "codex:b",
            "messages": [
                {"id": "m1", "text": "shared prefix", "material_origin": "human_authored"},
                {"id": "m5", "text": "child decision", "material_origin": "assistant_authored"},
            ],
        },
    ]


def test_compaction_manifest_filters_spam_and_keeps_structured_failure_fix() -> None:
    """A failing tool result and the message that fixes it survive; the rest do not.

    Anti-vacuity: delete the ``successful_tool_spam`` branch in
    ``compact_sessions`` and ``drop_counts["successful_tool_spam"]`` raises
    ``KeyError`` because the successful ``m4`` is kept instead. Deleting the
    ``duplicate_lineage_prefix`` branch makes ``duplicate_prefix_omissions``
    read 0. The fixture reaches the production structural-outcome reader
    through the same ``tool_result_is_error``/``tool_result_exit_code`` fields
    ``_tool_outcome`` consults, and the parent/child link aliases are the ones
    ``compact_sessions`` accepts at line 188.
    """
    pack = compact_sessions(
        _sessions(),
        session_links=[
            {"parent_session_id": "codex:a", "child_session_id": "codex:b", "branch_point_message_id": "m1"}
        ],
    )

    refs = {item.anchor.ref.format() for item in pack.items}
    assert "codex:a::m2::1" in refs
    assert "codex:a::m3::2" in refs
    assert "codex:a::m1::0" not in refs
    assert pack.manifest.drop_counts_by_material_origin["runtime_protocol"] == 1
    assert pack.manifest.drop_counts["successful_tool_spam"] == 1
    assert pack.manifest.duplicate_prefix_omissions == 1
    assert all(item.anchor.ref in item.refs for item in pack.items)


def test_compaction_budget_is_deterministic_and_clips_before_dropping() -> None:
    """The budget degrades by clipping first, and the same input compacts alike.

    Anti-vacuity: reordering ``CompactProjectionSpec.degradation_order`` so
    ``skeleton_only`` precedes ``clip`` makes the third assertion red, and
    removing the ``budget_clip`` drop accounting makes the last one red. The
    determinism assertion is red for any per-call identifier folded into the
    pack, which ``pack_ref`` deliberately derives from the kept anchors.
    """
    sessions = [
        {
            "id": "s",
            "messages": [
                {"id": str(i), "text": "decision " * 20, "material_origin": "assistant_authored"} for i in range(20)
            ],
        }
    ]
    spec = CompactProjectionSpec(max_tokens=60)
    first = compact_sessions(sessions, spec=spec)
    second = compact_sessions(sessions, spec=spec)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.token_estimate <= 60
    assert first.manifest.degradation_order[:3] == ("clip", "collapse_runs_to_counts", "skeleton_only")
    assert first.manifest.drop_counts["budget_clip"] >= 1


def test_compaction_pack_is_not_context_image() -> None:
    """A compaction pack is its own type with its own ref namespace.

    Anti-vacuity: returning a ``ContextImage`` from ``compact_sessions``, or
    changing ``pack_ref`` to any prefix other than ``compact:``, makes this
    red. The two surfaces are separately owned and a caller that receives one
    where it expects the other reads a different projection contract.
    """
    pack = compact_sessions(
        [{"id": "s", "messages": [{"id": "m", "text": "evidence", "material_origin": "human_authored"}]}]
    )
    assert not isinstance(pack, ContextImage)
    assert pack.pack_ref.startswith("compact:")
