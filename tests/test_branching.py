from __future__ import annotations

from polylogue.branching import MessageRecord, build_branch_plan


def _make_record(message_id: str, parent_id: str | None) -> MessageRecord:
    return MessageRecord(
        message_id=message_id,
        parent_id=parent_id,
        role="user",
        text=f"Message {message_id}",
        token_count=1,
        word_count=1,
        timestamp=None,
        attachments=0,
        chunk={"id": message_id},
        links=[],
        metadata={},
    )


def test_build_branch_plan_handles_deep_chains_without_recursion_error() -> None:
    depth = 1500
    records = []
    parent: str | None = None
    for index in range(depth):
        message_id = f"m{index}"
        records.append(_make_record(message_id, parent))
        parent = message_id

    plan = build_branch_plan(records)
    canonical_messages = plan.messages_for_branch(plan.canonical_branch_id)
    assert len(canonical_messages) == depth
    assert canonical_messages[0] == "m0"
    assert canonical_messages[-1] == f"m{depth - 1}"
    assert len(plan.branches) == 1


def test_build_branch_plan_handles_cycles_gracefully() -> None:
    records = [
        _make_record("a", "c"),  # Cycle a -> b -> c -> a
        _make_record("b", "a"),
        _make_record("c", "b"),
    ]

    plan = build_branch_plan(records)
    canonical_messages = plan.messages_for_branch(plan.canonical_branch_id)
    assert canonical_messages == ["a", "b", "c"]
    assert len(plan.branches) == 1
