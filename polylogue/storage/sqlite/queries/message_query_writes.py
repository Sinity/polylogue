"""Write queries for messages."""

from __future__ import annotations

from polylogue.storage.runtime import MessageRecord


def topo_sort_messages(records: list[MessageRecord]) -> list[MessageRecord]:
    ids_in_batch = {r.message_id for r in records}
    no_parent: list[MessageRecord] = []
    has_parent: list[MessageRecord] = []
    for r in records:
        if r.parent_message_id and r.parent_message_id in ids_in_batch:
            has_parent.append(r)
        else:
            no_parent.append(r)
    if not has_parent:
        return records
    ordered: list[MessageRecord] = list(no_parent)
    inserted_ids = {r.message_id for r in ordered}
    remaining = list(has_parent)
    max_passes = len(remaining) + 1
    for _ in range(max_passes):
        if not remaining:
            break
        next_remaining: list[MessageRecord] = []
        for r in remaining:
            if r.parent_message_id in inserted_ids:
                ordered.append(r)
                inserted_ids.add(r.message_id)
            else:
                next_remaining.append(r)
        remaining = next_remaining
    ordered.extend(remaining)
    return ordered


__all__ = ["topo_sort_messages"]
