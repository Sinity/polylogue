from __future__ import annotations

import json
import subprocess
import sys
from typing import cast, get_args, get_type_hints


def test_query_completions_do_not_import_expression_parser() -> None:
    script = """
import json
import sys
from polylogue.archive.query.completions import query_field_candidates

candidates = query_field_candidates("re")
print(json.dumps({
    "values": [candidate.value for candidate in candidates],
    "expression_loaded": "polylogue.archive.query.expression" in sys.modules,
}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert "repo" in payload["values"]
    assert payload["expression_loaded"] is False


def test_terminal_query_completion_payloads_are_lightweight() -> None:
    from polylogue.archive.query.completions import query_completion_payload

    source_payload = query_completion_payload("terminal-source", incomplete="observed")
    source_candidates = [
        cast(dict[str, object], candidate) for candidate in cast(list[object], source_payload["candidates"])
    ]
    source_values = [candidate["value"] for candidate in source_candidates]
    assert "observed-events" in source_values
    source_candidate = next(candidate for candidate in source_candidates if candidate["value"] == "observed-events")
    assert source_candidate["insert"] == "observed-events where "
    assert source_candidate["kind"] == "query-terminal-source"
    assert source_candidate["source"] == "QUERY_UNIT_DESCRIPTORS"

    field_payload = query_completion_payload("terminal-field", unit="context-snapshots", incomplete="bound")
    field_candidates = [
        cast(dict[str, object], candidate) for candidate in cast(list[object], field_payload["candidates"])
    ]
    assert [candidate["value"] for candidate in field_candidates] == ["boundary"]
    field_candidate = field_candidates[0]
    assert field_candidate["insert"] == "boundary:"
    assert field_candidate["kind"] == "query-terminal-field"
    assert field_candidate["source"] == "QUERY_UNIT_DESCRIPTORS"


def test_query_unit_descriptors_own_terminal_aliases() -> None:
    from polylogue.archive.query.metadata import (
        query_unit_descriptor,
        query_unit_descriptors,
        structural_query_fields,
        structural_query_units,
        terminal_query_cli_surfaces,
        terminal_query_examples,
        terminal_query_fields,
        terminal_query_source_list,
        terminal_query_source_pairs,
        terminal_query_unit,
    )

    observed_descriptor = query_unit_descriptor("observed-events")
    context_descriptor = query_unit_descriptor("context-snapshot")
    message_descriptor = query_unit_descriptor("messages")
    assert observed_descriptor is not None
    assert context_descriptor is not None
    assert message_descriptor is not None
    assert terminal_query_unit("messages") == "message"
    assert terminal_query_unit("context-snapshots") == "context-snapshot"
    assert observed_descriptor.plural_source == "observed-events"
    assert observed_descriptor.exists_supported is False
    assert observed_descriptor.lowerer_kind == "runtime_transform"
    assert observed_descriptor.payload_model == "ObservedEventQueryRowPayload"
    assert message_descriptor.exists_supported is True
    assert message_descriptor.lowerer_kind == "sql"
    assert message_descriptor.payload_model == "MessageQueryRowPayload"
    assert context_descriptor.source_aliases == ("context-snapshot", "context-snapshots")
    assert terminal_query_source_list() == "messages/actions/blocks/assertions/runs/observed-events/context-snapshots"
    assert ("assertions", "assertion") in terminal_query_source_pairs()
    assert structural_query_units() == ("action", "assertion", "block", "message")
    assert tuple(descriptor.unit for descriptor in query_unit_descriptors(exists_supported=True)) == (
        "message",
        "action",
        "block",
        "assertion",
    )
    assert tuple(descriptor.unit for descriptor in query_unit_descriptors(lowerer_kind="runtime_transform")) == (
        "run",
        "observed-event",
        "context-snapshot",
    )
    assert structural_query_fields("context-snapshot") == ()
    assert "boundary" in terminal_query_fields("context-snapshots")
    assert "polylogue --format json runs where ..." in terminal_query_cli_surfaces()
    assert "runs where session.repo:polylogue AND role:main" in terminal_query_examples()


def test_query_unit_schema_surfaces_are_descriptor_derived() -> None:
    from devtools.render_cli_output_schemas import SCHEMAS
    from polylogue.archive.query.metadata import terminal_query_cli_surfaces

    query_unit_schema = next(schema for schema in SCHEMAS if schema.name == "query-unit-envelope")
    for surface in terminal_query_cli_surfaces():
        assert surface in query_unit_schema.surfaces


def test_query_unit_payload_and_structural_types_track_descriptors() -> None:
    from polylogue.archive.query.metadata import query_unit_descriptors, structural_query_units
    from polylogue.archive.query.predicate import QueryExistsPredicate
    from polylogue.surfaces.payloads import QueryUnitKind

    assert tuple(get_args(QueryUnitKind)) == tuple(
        descriptor.unit for descriptor in query_unit_descriptors(terminal_supported=True)
    )
    assert frozenset(get_args(get_type_hints(QueryExistsPredicate)["unit"])) == frozenset(structural_query_units())


def test_terminal_query_completion_payloads_expose_payload_model() -> None:
    from polylogue.archive.query.completions import query_completion_payload

    source_payload = query_completion_payload("terminal-source", incomplete="runs")
    source_candidates = [
        cast(dict[str, object], candidate) for candidate in cast(list[object], source_payload["candidates"])
    ]
    run_candidate = next(candidate for candidate in source_candidates if candidate["value"] == "runs")
    assert run_candidate["payload_model"] == "RunQueryRowPayload"

    field_payload = query_completion_payload("terminal-field", unit="assertions", incomplete="kind")
    field_candidates = [
        cast(dict[str, object], candidate) for candidate in cast(list[object], field_payload["candidates"])
    ]
    field_candidate = next(candidate for candidate in field_candidates if candidate["value"] == "kind")
    assert field_candidate["payload_model"] == "AssertionQueryRowPayload"
