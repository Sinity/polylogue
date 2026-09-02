from __future__ import annotations

from polylogue.storage.sqlite.archive_tiers.durable_references import durable_reference_relations


def test_result_set_members_use_successor_manifests() -> None:
    relations = durable_reference_relations("user")
    result_members = next(relation for relation in relations if relation.table == "result_set_members")

    assert result_members.transition == "successor"
    assert result_members.identity_columns == ("result_set_id", "rank")
    assert tuple(field.column for field in result_members.fields) == ("member_ref",)
