from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from devtools.render_webui_client import ContractGenerationError, generate, render

FIXTURE_SCHEMA = Path("tests/fixtures/openapi/webui-client.yaml")
FIXTURE_GOLDEN = Path("tests/fixtures/openapi/webui-client.generated.ts")
COMMITTED_SCHEMA = Path("docs/openapi/search.yaml")
COMMITTED_CLIENT = Path("webui/src/api/generated.ts")


def test_webui_client_generator_matches_golden_fixture() -> None:
    rendered = generate(FIXTURE_SCHEMA)

    assert rendered == FIXTURE_GOLDEN.read_text(encoding="utf-8")
    assert "export type QueryPage = Page<Item, ItemEnvelope>;" in rendered
    assert "cursor === null ? parameters : { bucket: parameters.bucket, continuation: cursor }" in rendered
    assert 'qualification: "page" as const' in rendered


def test_webui_client_check_mode_detects_drift(tmp_path: Path) -> None:
    output = tmp_path / "generated.ts"
    output.write_text(generate(FIXTURE_SCHEMA), encoding="utf-8")

    assert render(FIXTURE_SCHEMA, output, check=True) == 0

    output.write_text(output.read_text(encoding="utf-8") + "// drift\n", encoding="utf-8")
    assert render(FIXTURE_SCHEMA, output, check=True) == 1


def test_committed_webui_client_matches_generated_openapi() -> None:
    assert render(COMMITTED_SCHEMA, COMMITTED_CLIENT, check=True) == 0


def test_webui_client_generator_fails_closed_for_unknown_page_schema(tmp_path: Path) -> None:
    document = yaml.safe_load(FIXTURE_SCHEMA.read_text(encoding="utf-8"))
    document["paths"]["/api/items/{bucket}"]["get"]["x-polylogue-page"]["response_schemas"] = ["MissingEnvelope"]
    schema = tmp_path / "invalid.yaml"
    schema.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    assert render(schema, tmp_path / "generated.ts", check=False) == 2


def test_webui_client_generator_is_deterministic() -> None:
    assert generate(FIXTURE_SCHEMA) == generate(FIXTURE_SCHEMA)


def test_webui_client_generator_fails_closed_for_request_bodies(tmp_path: Path) -> None:
    document = yaml.safe_load(FIXTURE_SCHEMA.read_text(encoding="utf-8"))
    document["paths"]["/api/items/{bucket}"]["get"]["requestBody"] = {
        "content": {"application/json": {"schema": {"type": "object"}}}
    }
    schema = tmp_path / "request-body.yaml"
    schema.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ContractGenerationError, match="request bodies are not supported"):
        generate(schema)


def test_webui_client_generator_fails_closed_for_unknown_coverage_qualification(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load(FIXTURE_SCHEMA.read_text(encoding="utf-8"))
    page = document["paths"]["/api/items/{bucket}"]["get"]["x-polylogue-page"]
    page["coverage"] = {
        "total_property": "total",
        "exactness_property": "exactness",
        "exact_values": ["exact"],
        "qualified_values": ["approximately-right"],
    }
    schema = tmp_path / "coverage.yaml"
    schema.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ContractGenerationError, match="unsupported qualifications"):
        generate(schema)
