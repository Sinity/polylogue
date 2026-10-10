from __future__ import annotations

import json
from typing import cast

import pytest

from polylogue.archive.context_models import ContextImage
from polylogue.surfaces.compaction import (
    CompactionBudgetTooSmallError,
    CompactProjectionSpec,
    CorpusCompactionPack,
    compact_sessions,
    estimate_serialized_tokens,
)


def _wire_tokens(pack: CorpusCompactionPack) -> int:
    return estimate_serialized_tokens(
        json.dumps(pack.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


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
    # Anchors name messages, never their enumeration position.
    assert "codex:a::m2" in refs
    assert "codex:a::m3" in refs
    assert "codex:a::m1" not in refs
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
                {"id": str(i), "text": "decision " * 100, "material_origin": "assistant_authored"} for i in range(20)
            ],
        }
    ]
    spec = CompactProjectionSpec(max_tokens=800)
    first = compact_sessions(sessions, spec=spec)
    second = compact_sessions(sessions, spec=spec)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.token_estimate <= spec.max_tokens
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


def test_compaction_uses_canonical_lineage_and_message_refs() -> None:
    """Archive links deduplicate inherited prefixes and anchors name messages.

    Anti-vacuity: swapping src/resolved-dst back to parent/child direction
    retains the child's inherited prefix; passing the enumeration position to
    EvidenceRef makes the first ref contain a bogus block suffix.
    """
    sessions = [
        {"id": "parent", "messages": [{"id": "m0", "text": "shared", "material_origin": "human_authored"}]},
        {
            "id": "child",
            "messages": [
                {"id": "m0", "text": "shared", "material_origin": "human_authored"},
                {"id": "m1", "text": "new", "material_origin": "human_authored"},
            ],
        },
    ]
    pack = compact_sessions(
        sessions,
        session_links=[
            {"src_session_id": "child", "resolved_dst_session_id": "parent", "branch_point_message_id": "m0"}
        ],
    )
    child_refs = {item.anchor.ref.format() for item in pack.items if item.session_id == "child"}
    assert child_refs == {"child::m1"}
    assert all("::0" not in ref and "::1" not in ref for ref in child_refs)


def test_compaction_budget_counts_serialized_omissions() -> None:
    """The advertised estimate is the estimate of the whole emitted pack.

    Anti-vacuity: measuring compact JSON by whitespace-split words sees the
    100 omission objects as about one word, so all of them stay in the pack
    and its real wire estimate exceeds the budget; measuring a partial probe
    instead of the emitted pack breaks the equality.
    """
    sessions = [
        {
            "id": "s",
            "messages": [
                {"id": str(i), "text": "private protocol details " * 10, "material_origin": "runtime_protocol"}
                for i in range(100)
            ],
        }
    ]
    pack = compact_sessions(sessions, spec=CompactProjectionSpec(max_tokens=800))
    assert pack.token_estimate == _wire_tokens(pack)
    assert pack.token_estimate <= 800
    assert 0 < len(pack.omissions) < 100
    assert "omission_rows_truncated" in pack.manifest.unknown
    assert pack.manifest.drop_counts["filtered_material_origin"] == 100


def test_serialized_estimate_charges_unspaced_and_multilingual_text() -> None:
    """Text without whitespace still costs in proportion to its bytes.

    Anti-vacuity: a word-count-only estimator reports 1 for both inputs.
    """
    assert estimate_serialized_tokens(json.dumps([{"a": 1}] * 200, separators=(",", ":"))) >= 400
    assert estimate_serialized_tokens("\u8a18\u9332" * 200) >= 300


def test_compaction_refuses_a_budget_below_its_envelope() -> None:
    """A budget smaller than the empty typed pack is refused, not mislabelled.

    Anti-vacuity: breaking out of the budget loop instead of raising returns a
    pack whose estimate exceeds ``max_tokens``.
    """
    with pytest.raises(CompactionBudgetTooSmallError) as refusal:
        compact_sessions([], spec=CompactProjectionSpec(max_tokens=1))
    assert refusal.value.envelope_tokens > 1


def test_compaction_does_not_invent_canonical_content_hashes() -> None:
    """An input without an archive content hash keeps no anchor hash.

    Anti-vacuity: restore the text-only sha256 fallback and the ``missing``
    anchor carries a hash the archive never computed.
    """
    pack = compact_sessions(
        [
            {
                "id": "s",
                "messages": [
                    {"id": "missing", "text": "evidence", "material_origin": "human_authored"},
                    {"id": "known", "text": "evidence", "material_origin": "human_authored", "content_hash": "a" * 64},
                ],
            }
        ]
    )
    hashes = {item.anchor.ref.message_id: item.anchor.content_hash for item in pack.items}
    assert hashes == {"missing": None, "known": "a" * 64}


def test_compaction_identity_commits_to_content_projection_and_provenance() -> None:
    """Packs that differ in text, projection or provenance get different refs.

    Anti-vacuity: hash only the kept anchor refs and all four packs, which
    retain the same single anchor, share one ``pack_ref``.
    """

    def pack(text: str = "evidence", budget: int = 60_000, run: str | None = None) -> CorpusCompactionPack:
        return compact_sessions(
            [{"id": "s", "messages": [{"id": "m", "text": text, "material_origin": "human_authored"}]}],
            spec=CompactProjectionSpec(max_tokens=budget),
            query_run_ref=run,
        )

    first = pack()
    assert first.pack_ref == pack().pack_ref
    assert first.token_estimate == _wire_tokens(first)
    refs = {first.pack_ref, pack("changed evidence").pack_ref, pack(budget=59_000).pack_ref}
    refs.add(pack(run="query-run:other").pack_ref)
    assert len(refs) == 4


def test_compaction_markdown_preserves_fidelity_manifest_and_omission_anchors() -> None:
    """The Markdown rendering carries the whole manifest and every omission anchor.

    Anti-vacuity: render only the aggregate drop counts and the per-origin and
    per-session maps, ``unknown`` and the omitted anchors are missing.
    """
    pack = compact_sessions(
        [
            {
                "id": "child",
                "parent_id": "missing-parent",
                "messages": [
                    {"id": "omitted", "text": "protocol", "material_origin": "runtime_protocol"},
                    {"id": "kept", "text": "authored evidence", "material_origin": "human_authored"},
                ],
            }
        ]
    )
    markdown = pack.render_markdown()
    assert "lineage_unresolved" in markdown
    assert "drop_counts_by_material_origin" in markdown
    assert "included_tokens_by_session" in markdown
    assert "dropped_tokens_by_session" in markdown
    assert pack.omissions
    for omission in pack.omissions:
        assert omission.anchor.ref.format() in markdown


def test_long_unbroken_runs_are_weighted_by_their_size() -> None:
    """Anti-vacuity: counting a run as one word estimates ``"!" * 100000`` at
    one token, so a tiny budget would accept it.
    """
    from polylogue.surfaces.compaction import estimate_serialized_tokens, estimate_tokens

    run = "!" * 100_000
    assert estimate_tokens(run) >= 10_000
    assert estimate_serialized_tokens(f'{{"text":"{run}"}}') >= 10_000
    assert estimate_tokens("one two three") == 3


def _repeated_messages(count: int, text: str) -> list[dict[str, object]]:
    return [
        {
            "id": "s",
            "messages": [
                {
                    "id": str(index),
                    "text": text,
                    "material_origin": "assistant_authored",
                    "content_hash": f"{index:064x}",
                }
                for index in range(count)
            ],
        }
    ]


def test_clipped_source_tokens_partition_the_original_prose() -> None:
    from polylogue.surfaces.compaction import estimate_tokens

    text = "evidence " * 1000
    pack = compact_sessions(_repeated_messages(1, text), spec=CompactProjectionSpec(max_tokens=600))
    assert len(pack.items) == 1 and pack.items[0].degradation == "clip"
    included = pack.manifest.included_tokens_by_session["s"]
    dropped = pack.manifest.dropped_tokens_by_session["s"]
    assert included > 0 and dropped > 0
    assert included + dropped == estimate_tokens(text)
    assert included == estimate_tokens(pack.items[0].text.removesuffix(" …"))
    assert pack.token_estimate == _wire_tokens(pack) <= 600


@pytest.mark.parametrize("budget,stage", [(800, "collapse_runs_to_counts"), (700, "skeleton_only")])
def test_budget_ladder_retains_run_counts_and_source_references(budget: int, stage: str) -> None:
    from polylogue.surfaces.compaction import estimate_tokens

    text = "decision " * 1000
    # The current typed skeleton and all 20 source refs cost 692 wire tokens;
    # 700 holds them while requiring prose removal. 620 exercises index-only.
    pack = compact_sessions(_repeated_messages(20, text), spec=CompactProjectionSpec(max_tokens=budget))
    assert len(pack.items) == 1
    item = pack.items[0]
    assert item.degradation == stage
    assert item.occurrence_count == 20
    assert {ref.message_id for ref in item.refs} == {str(index) for index in range(20)}
    assert item.anchor.ref in item.refs and item.anchor.content_hash == "0" * 64
    assert pack.manifest.drop_counts["budget_collapsed"] == 19
    assert pack.manifest.included_tokens_by_session["s"] + pack.manifest.dropped_tokens_by_session["s"] == (
        20 * estimate_tokens(text)
    )
    assert pack.outcome.state == "degraded"
    assert pack.token_estimate == _wire_tokens(pack) <= budget
    markdown = pack.render_markdown()
    assert f"Occurrences: 20; degradation: {stage}" in markdown
    assert all(ref.format() in markdown for ref in item.refs)
    if stage == "skeleton_only":
        assert item.text == "" and pack.manifest.included_tokens_by_session["s"] == 0


@pytest.mark.parametrize("budget", [400, 620])
def test_index_only_budget_pack_reports_source_loss_instead_of_empty_selection(budget: int) -> None:
    from polylogue.surfaces.compaction import estimate_tokens

    text = "decision " * 1000
    pack = compact_sessions(_repeated_messages(20, text), spec=CompactProjectionSpec(max_tokens=budget))
    assert pack.items == ()
    assert pack.manifest.dropped_tokens_by_session["s"] == 20 * estimate_tokens(text)
    assert pack.manifest.drop_counts["budget_skeleton"] == 1
    assert pack.manifest.drop_counts["budget_drop"] == 1
    assert "index_only_pack_failure" in pack.manifest.unknown
    assert pack.outcome.state == "degraded"
    gaps = pack.outcome.detail["gaps"]
    assert isinstance(gaps, list)
    assert "index_only_pack_failure" in gaps
    assert pack.token_estimate == _wire_tokens(pack) <= budget
    with pytest.raises(CompactionBudgetTooSmallError):
        compact_sessions(_repeated_messages(20, text), spec=CompactProjectionSpec(max_tokens=60))


def test_collapse_uses_original_adjacent_text_not_equal_clipped_prefixes() -> None:
    sessions = _repeated_messages(2, "same words " * 1000)
    messages = cast(list[dict[str, str]], sessions[0]["messages"])
    messages[1]["text"] += "different ending"
    pack = compact_sessions(sessions, spec=CompactProjectionSpec(max_tokens=800))
    assert "budget_collapsed" not in pack.manifest.drop_counts
    assert all(item.occurrence_count == 1 for item in pack.items)
