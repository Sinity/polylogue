"""JSON Schema contract tests for stable CLI output surfaces (#1272).

These tests assert that the published schemas under
``docs/schemas/cli-output/`` actually match what the live CLI emits.

Approach:
1. Load each published schema as JSON.
2. Build the corresponding Pydantic payload from a representative
   instance (synthetic ``Session``/``SessionSummary``).
3. Validate the model's ``model_dump(mode='json')`` against the schema
   using ``jsonschema``.

If a payload model gains/loses a field, the schema must be regenerated
(``devtools render-cli-output-schemas``), and these tests will start
failing until that happens — closing the drift loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.render_cli_output_schemas import SCHEMAS, _build_schema

SCHEMAS_DIR = Path("docs/schemas/cli-output")


pytest.importorskip("jsonschema", reason="jsonschema not installed")


def _load_published_schema(name: str) -> dict[str, object]:
    target = SCHEMAS_DIR / f"{name}.schema.json"
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.parametrize("entry", SCHEMAS, ids=lambda e: e.name)
def test_published_schema_matches_current_model(entry: object) -> None:
    """Published JSON Schema files must be in sync with the Pydantic models.

    Equivalent to running ``devtools render-cli-output-schemas --check``,
    but expressed per-schema so failure messages name the drifting surface.
    """
    from devtools.render_cli_output_schemas import CliOutputSchema

    cli_entry: CliOutputSchema = entry  # type: ignore[assignment]
    target = SCHEMAS_DIR / f"{cli_entry.name}.schema.json"
    expected = _build_schema(cli_entry)
    actual = target.read_text(encoding="utf-8")
    assert actual == expected, (
        f"Published schema `{target}` is out of sync with "
        f"`{cli_entry.model.__name__}`. Run "
        "`devtools render-cli-output-schemas` to refresh."
    )


def test_machine_error_payload_validates_against_schema() -> None:
    """A real MachineErrorPayload must validate against the published schema."""
    import jsonschema

    from polylogue.surfaces.payloads import MachineErrorPayload

    schema = _load_published_schema("machine-error")
    payload = MachineErrorPayload(
        code="invalid_arguments",
        message="missing --provider",
        command=["polylogue", "list"],
        details={"hint": "supply at least one filter"},
    )
    jsonschema.validate(instance=payload.to_dict(), schema=schema)


def test_machine_success_payload_validates_against_schema() -> None:
    """A real MachineSuccessPayload must validate against the published schema."""
    import jsonschema

    from polylogue.surfaces.payloads import MachineSuccessPayload

    schema = _load_published_schema("machine-success")
    payload = MachineSuccessPayload(result={"count": 7, "items": ["a", "b"]})
    instance = payload.model_dump(mode="json")
    jsonschema.validate(instance=instance, schema=schema)


def test_mutation_result_payload_validates_against_schema() -> None:
    """A real MutationResultPayload must validate against the published schema."""
    import jsonschema

    from polylogue.surfaces.payloads import MutationResultPayload

    schema = _load_published_schema("mutation-result")
    payload = MutationResultPayload(
        status="preview",
        operation="delete",
        affected_count=0,
        session_count=2,
        session_ids=("session-a", "session-b"),
    )
    instance = payload.model_dump(mode="json", exclude_none=True)
    jsonschema.validate(instance=instance, schema=schema)


def test_query_error_payload_validates_against_schema() -> None:
    """A real QueryErrorPayload must validate against the published schema."""
    import jsonschema

    from polylogue.surfaces.payloads import QueryErrorPayload

    schema = _load_published_schema("query-error")
    payload = QueryErrorPayload(error="invalid_cursor", detail="cursor expired", field="cursor")
    instance = payload.model_dump(mode="json", exclude_none=True)
    jsonschema.validate(instance=instance, schema=schema)


def test_session_list_row_payload_validates_against_schema() -> None:
    """SessionListRowPayload must validate against the published schema."""
    import jsonschema

    from polylogue.surfaces.payloads import (
        SessionFlagsPayload,
        SessionListRowPayload,
        TargetRefPayload,
    )

    schema = _load_published_schema("session-list-row")
    payload = SessionListRowPayload(
        id="claude-ai:abc123",
        origin="claude-ai-export",
        title="Example session",
        target_ref=TargetRefPayload.session("claude-ai:abc123"),
        anchor="session-claude-ai-abc123",
        date="2026-05-18T12:00:00+00:00",
        messages=42,
        tags=("review", "lab"),
        summary="Discussion of frobulators.",
        words=1234,
        repo="polylogue",
        cwd_display="/realm/project/polylogue",
        flags=SessionFlagsPayload(has_thinking=True, has_tool_use=True, has_paste=False),
    )
    instance = payload.model_dump(mode="json", exclude_none=True)
    jsonschema.validate(instance=instance, schema=schema)


def test_ndjson_list_output_validates_against_schema() -> None:
    """Each line of `--format ndjson` list output must validate against the schema."""
    import jsonschema

    from polylogue.cli.query_output_contracts import StructuredRowsDocument
    from polylogue.surfaces.payloads import (
        SessionListRowPayload,
        TargetRefPayload,
        model_json_document,
    )

    schema = _load_published_schema("session-list-row")
    rows = tuple(
        model_json_document(
            SessionListRowPayload(
                id=f"claude-ai:row-{i}",
                origin="claude-ai-export",
                title=f"Row {i}",
                target_ref=TargetRefPayload.session(f"claude-ai:row-{i}"),
                anchor=f"session-claude-ai-row-{i}",
                date="2026-05-18T12:00:00+00:00",
                messages=i + 1,
                words=10 * (i + 1),
            ),
            exclude_none=True,
        )
        for i in range(3)
    )

    document = StructuredRowsDocument(
        rows=rows,
        csv_headers=(),
        csv_rows=(),
        text_lines=(),
    )
    ndjson_body = document.render("ndjson")

    lines = ndjson_body.split("\n")
    assert len(lines) == 3
    for line in lines:
        instance = json.loads(line)
        jsonschema.validate(instance=instance, schema=schema)


def test_ndjson_render_is_one_doc_per_line() -> None:
    """`ndjson` output is contract-shaped: one JSON document per line, no array."""
    from polylogue.cli.query_output_contracts import StructuredRowsDocument

    document = StructuredRowsDocument(
        rows=({"id": "1", "title": "one"}, {"id": "2", "title": "two"}),
        csv_headers=(),
        csv_rows=(),
        text_lines=(),
    )

    body = document.render("ndjson")
    lines = body.split("\n")
    assert len(lines) == 2
    # Each line individually parses as JSON; the whole body is not a JSON array.
    for line in lines:
        parsed = json.loads(line)
        assert "id" in parsed and "title" in parsed
    assert not body.lstrip().startswith("[")


def test_ndjson_empty_rows_render_empty_string() -> None:
    """Empty result set in ndjson is an empty string, not `[]`."""
    from polylogue.cli.query_output_contracts import StructuredRowsDocument

    document = StructuredRowsDocument(rows=(), csv_headers=(), csv_rows=(), text_lines=())
    assert document.render("ndjson") == ""
