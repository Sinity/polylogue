from __future__ import annotations

import pytest

from polylogue.core.query_identity import (
    JsonValue,
    canonical_query_plan,
    query_hash_for_plan,
    query_ref,
    query_run_ref,
    require_supported_definition_protocol_version,
    result_set_ref,
)


def test_query_hash_normalizes_field_aliases_and_actual_commutative_kind() -> None:
    left: dict[str, JsonValue] = {
        "kind": "AND",
        "children": [
            {"kind": "field", "field": "provider", "values": ["codex-session"]},
            {"kind": "field", "field": "title", "values": ["alpha"]},
        ],
    }
    right: dict[str, JsonValue] = {
        "kind": "and",
        "children": [
            {"kind": "field", "field": "title", "values": ["alpha"]},
            {"kind": "field", "field": "origin", "values": ["codex-session"]},
        ],
    }

    assert query_hash_for_plan(
        left, grain="session", lane="dialogue", rank_policy="mixed", field_aliases={"provider": "origin"}
    ) == query_hash_for_plan(right, grain="session", lane="dialogue", rank_policy="mixed")


def test_query_hash_preserves_non_commutative_pipeline_order() -> None:
    first: dict[str, JsonValue] = {"pipeline": [{"limit": 10}, {"sort": "date"}]}
    second: dict[str, JsonValue] = {"pipeline": [{"sort": "date"}, {"limit": 10}]}

    assert query_hash_for_plan(first, grain="session", lane="dialogue", rank_policy="mixed") != query_hash_for_plan(
        second, grain="session", lane="dialogue", rank_policy="mixed"
    )


def test_query_identity_includes_grain_lane_and_rank_policy() -> None:
    ast: dict[str, JsonValue] = {"field": "origin", "value": "codex-session"}
    digest = query_hash_for_plan(ast, grain="session", lane="dialogue", rank_policy="mixed")

    assert digest != query_hash_for_plan(ast, grain="message", lane="dialogue", rank_policy="mixed")
    assert digest != query_hash_for_plan(ast, grain="session", lane="hybrid", rank_policy="mixed")
    assert digest != query_hash_for_plan(ast, grain="session", lane="dialogue", rank_policy="bm25")


def test_public_query_object_refs_use_the_shared_registered_kinds() -> None:
    digest = "a" * 64

    assert query_ref(digest).format() == f"query:{digest}"
    assert query_run_ref("qr_01JTEST").format() == "query-run:qr_01JTEST"
    assert result_set_ref("rs_01JTEST").format() == "result-set:rs_01JTEST"


@pytest.mark.parametrize("run_id", ["", "run_01JTEST", "qr_"])
def test_query_run_ref_rejects_noncontract_ids(run_id: str) -> None:
    with pytest.raises(ValueError):
        query_run_ref(run_id)


def test_canonical_query_plan_retains_relative_time_as_dynamic_ast() -> None:
    dynamic: dict[str, JsonValue] = {"field": "since", "relative": "7d"}

    assert canonical_query_plan(dynamic, grain="session", lane="dialogue", rank_policy="mixed")["ast"] == dynamic


def test_definition_protocol_version_is_bound_into_query_identity() -> None:
    plan: dict[str, JsonValue] = {"field": "origin", "value": "codex-session"}

    assert query_hash_for_plan(
        plan,
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        definition_protocol_version="polylogue.query-definition.v1",
    ) != query_hash_for_plan(
        plan,
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        definition_protocol_version="polylogue.query-definition.v2",
    )


def test_unsupported_definition_protocol_version_fails_closed_at_execution_boundary() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        require_supported_definition_protocol_version("polylogue.query-definition.v999")


def test_query_canonicalization_preserves_opaque_operands_and_mapping_keys() -> None:
    opaque: dict[str, JsonValue] = {
        "é": "composed",
        "e\u0301": "decomposed",
        "kind": "AND",
        "field": "provider",
        "op": "OR",
        "children": ["second", "first"],
    }
    ast: dict[str, JsonValue] = {"kind": "field", "field": "id", "values": ["cafe\u0301"], "literal": opaque}
    canonical = canonical_query_plan(
        ast, grain="session", lane="dialogue", rank_policy="mixed", field_aliases={"provider": "origin"}
    )
    assert canonical["ast"] == ast
    changed: dict[str, JsonValue] = {**ast, "values": ["café"]}
    assert query_hash_for_plan(ast, grain="session", lane="dialogue", rank_policy="mixed") != query_hash_for_plan(
        changed, grain="session", lane="dialogue", rank_policy="mixed"
    )


def test_typed_directional_sequence_order_remains_semantic() -> None:
    from typing import cast

    from polylogue.archive.query.predicate import QueryFieldPredicate, QuerySequencePredicate

    steps = (QueryFieldPredicate("action", ("read",)), QueryFieldPredicate("action", ("write",)))
    forward = cast(dict[str, JsonValue], QuerySequencePredicate(steps=steps).to_payload())
    reverse = cast(dict[str, JsonValue], QuerySequencePredicate(steps=tuple(reversed(steps))).to_payload())
    assert query_hash_for_plan(forward, grain="session", lane="dialogue", rank_policy="mixed") != query_hash_for_plan(
        reverse, grain="session", lane="dialogue", rank_policy="mixed"
    )


def test_query_canonicalization_preserves_bound_field_coordinates_and_unknown_nodes() -> None:
    bound: dict[str, JsonValue] = {
        "kind": "field",
        "field": "provider",
        "values": ["codex-session"],
        "field_ref": {"scope": "session", "name": "origin", "source_name": "session.provider", "unit": "message"},
    }
    expected: dict[str, JsonValue] = {**bound, "field": "origin"}
    assert (
        canonical_query_plan(
            bound, grain="session", lane="dialogue", rank_policy="mixed", field_aliases={"provider": "origin"}
        )["ast"]
        == expected
    )
    unknown: dict[str, JsonValue] = {"kind": "OpaqueKind", "field": "provider", "children": [bound]}
    assert (
        canonical_query_plan(
            unknown, grain="session", lane="dialogue", rank_policy="mixed", field_aliases={"provider": "origin"}
        )["ast"]
        == unknown
    )
