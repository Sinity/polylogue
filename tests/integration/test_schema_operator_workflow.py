"""Integration test for the schema operator workflow.

End-to-end flow: infer -> list -> compare -> promote -> explain
using Click's CliRunner for product-facing inspection and devtools
entrypoints for maintenance actions against a real SchemaRegistry backed by a
temp directory.

Since schema infer requires DB access with real provider data,
we test it via a pre-populated registry and exercise the CLI
commands that read/write from it.
"""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from devtools import schema_audit, schema_promote
from polylogue.cli import cli
from polylogue.schemas.registry import SchemaRegistry

pytestmark = pytest.mark.integration

JsonObject = dict[str, object]
JsonObjectList = list[JsonObject]


def _expect_object(value: object, label: str = "value") -> JsonObject:
    if isinstance(value, dict):
        return value
    raise AssertionError(f"Expected {label} to be a JSON object, got {type(value).__name__}")


def _expect_object_list(value: object, label: str = "value") -> JsonObjectList:
    if not isinstance(value, list):
        raise AssertionError(f"Expected {label} to be a JSON array, got {type(value).__name__}")
    objects: JsonObjectList = []
    for item in value:
        objects.append(_expect_object(item, label=f"{label} item"))
    return objects


def _expect_string_list(value: object, label: str = "value") -> list[str]:
    if not isinstance(value, list):
        raise AssertionError(f"Expected {label} to be a JSON array, got {type(value).__name__}")
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise AssertionError(f"Expected {label} item to be a string, got {type(item).__name__}")
        strings.append(item)
    return strings


def _extract_json(output: str) -> JsonObject:
    """Extract JSON from CLI output, stripping any non-JSON trailer (e.g. plain-mode banner)."""
    # Find the start of JSON (first { or [)
    for i, ch in enumerate(output):
        if ch in "{[":
            # Try to parse from this position
            # Use raw_decode to find where the JSON ends
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(output, i)
            if isinstance(obj, dict):
                return obj
            raise ValueError(f"Expected JSON object in output, got {type(obj).__name__}: {output!r}")
    raise ValueError(f"No JSON found in output: {output!r}")


def _extract_result_json(output: str) -> JsonObject:
    """Extract JSON and unwrap the standard success envelope when present."""
    data = _extract_json(output)
    if isinstance(data, dict) and data.get("status") == "ok" and "result" in data:
        result = data["result"]
        if isinstance(result, dict):
            return result
        raise ValueError(f"Expected result JSON object, got {type(result).__name__}: {output!r}")
    return data


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def schema_storage(tmp_path: Path) -> Path:
    """Isolated schema storage root."""
    root = tmp_path / "schemas"
    root.mkdir()
    return root


@pytest.fixture
def seeded_registry(schema_storage: Path) -> SchemaRegistry:
    """Registry pre-seeded with two versions of a test provider schema."""
    registry = SchemaRegistry(storage_root=schema_storage)

    schema_v1 = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "title": "test-provider export format",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "create_time": {"type": "number"},
            "mapping": {"type": "object"},
        },
        "required": ["id", "title"],
    }

    schema_v2 = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "title": "test-provider export format v2",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "create_time": {"type": "number"},
            "mapping": {"type": "object"},
            "status": {"type": "string"},  # new field
            "tags": {"type": "array"},  # new field
        },
        "required": ["id"],  # title no longer required
    }

    registry.register_schema("test-provider", schema_v1)
    registry.register_schema("test-provider", schema_v2)

    return registry


def _patch_registry(registry: SchemaRegistry) -> AbstractContextManager[object]:
    """Context manager to patch SchemaRegistry() calls to return our seeded instance."""
    return patch(
        "polylogue.schemas.registry.SchemaRegistry",
        return_value=registry,
    )


# =============================================================================
# schema list
# =============================================================================


class TestSchemaListCommand:
    def test_list_all_providers_json(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(cli, ["schema", "list", "--json"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        outer = _extract_json(result.output)
        assert outer["status"] == "ok"
        data = _extract_result_json(result.output)
        providers = [entry["provider"] for entry in _expect_object_list(data["providers"], "providers")]
        assert "test-provider" in providers
        snapshot = next(
            entry
            for entry in _expect_object_list(data["providers"], "providers")
            if entry["provider"] == "test-provider"
        )
        assert snapshot["corpus_spec_count"] == 1
        assert snapshot["corpus_scenario_count"] == 1

    def test_list_specific_provider_json(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(cli, ["schema", "list", "--provider", "test-provider", "--json"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        outer = _extract_json(result.output)
        assert outer["status"] == "ok"
        data = _extract_result_json(result.output)
        assert data["provider"] == "test-provider"
        assert "v1" in _expect_string_list(data["versions"], "versions")
        assert "v2" in _expect_string_list(data["versions"], "versions")
        corpus_specs = _expect_object_list(data["corpus_specs"], "corpus_specs")
        corpus_scenarios = _expect_object_list(data["corpus_scenarios"], "corpus_scenarios")
        assert len(corpus_specs) == 1
        assert len(corpus_scenarios) == 1
        assert corpus_specs[0]["provider"] == "test-provider"
        assert corpus_scenarios[0]["provider"] == "test-provider"

    def test_list_specific_provider_text(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(cli, ["schema", "list", "--provider", "test-provider"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "test-provider" in result.output
        assert "v1" in result.output
        assert "v2" in result.output
        assert "Suggested synthetic scenarios:" in result.output
        assert "Suggested synthetic corpus specs:" in result.output

    def test_list_unknown_provider(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(cli, ["schema", "list", "--provider", "nonexistent"])

        assert result.exit_code == 0  # not an error, just "no schemas found"
        assert "No schemas found" in result.output or "nonexistent" in result.output


class TestSchemaAuditCommand:
    def test_schema_audit_json_scopes_checks_by_provider(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = schema_audit.main(["--json"])

        captured = capsys.readouterr()
        assert exit_code == 0, f"devtools failed: {captured.out}{captured.err}"
        outer = _extract_json(captured.out)
        assert outer["status"] == "ok"
        data = _extract_result_json(captured.out)
        checks = _expect_object_list(data["checks"], "checks")
        assert checks
        assert any(check.get("provider") for check in checks[:-1])


# =============================================================================
# schema compare
# =============================================================================


class TestSchemaCompareCommand:
    def test_compare_json_output(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(
                cli,
                ["schema", "compare", "--provider", "test-provider", "--from", "v1", "--to", "v2", "--json"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        outer = _extract_json(result.output)
        assert outer["status"] == "ok"
        data = _extract_result_json(result.output)
        assert data["provider"] == "test-provider"
        assert data["version_a"] == "v1"
        assert data["version_b"] == "v2"
        assert data["has_changes"] is True
        assert "status" in _expect_string_list(data["added_properties"], "added_properties")
        assert "tags" in _expect_string_list(data["added_properties"], "added_properties")

    def test_compare_classifies_additive_changes(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(
                cli,
                ["schema", "compare", "--provider", "test-provider", "--from", "v1", "--to", "v2", "--json"],
            )

        data = _extract_result_json(result.output)
        classified = _expect_object_list(data["classified_changes"], "classified_changes")
        added = [c for c in classified if c["kind"] == "added"]
        assert len(added) == 2
        added_paths = {c["path"] for c in added}
        assert "status" in added_paths
        assert "tags" in added_paths

    def test_compare_classifies_requiredness_changes(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(
                cli,
                ["schema", "compare", "--provider", "test-provider", "--from", "v1", "--to", "v2", "--json"],
            )

        data = _extract_result_json(result.output)
        classified = _expect_object_list(data["classified_changes"], "classified_changes")
        req = [c for c in classified if c["kind"] == "requiredness"]
        assert len(req) >= 1
        assert any(c["path"] == "title" for c in req)

    def test_compare_text_output(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(
                cli,
                ["schema", "compare", "--provider", "test-provider", "--from", "v1", "--to", "v2"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "v1 -> v2" in result.output
        assert "status" in result.output

    def test_compare_markdown_output(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(
                cli,
                [
                    "schema",
                    "compare",
                    "--provider",
                    "test-provider",
                    "--from",
                    "v1",
                    "--to",
                    "v2",
                    "--markdown",
                ],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "# Schema Diff" in result.output
        assert "| Path | Detail |" in result.output

    def test_compare_type_mutation(self, runner: CliRunner, schema_storage: Path) -> None:
        """Register schemas where a field changes type, verify classification."""
        registry = SchemaRegistry(storage_root=schema_storage)
        registry.register_schema(
            "mut-prov",
            {"type": "object", "properties": {"count": {"type": "integer"}}},
        )
        registry.register_schema(
            "mut-prov",
            {"type": "object", "properties": {"count": {"type": "string"}}},
        )

        with _patch_registry(registry):
            result = runner.invoke(
                cli,
                ["schema", "compare", "--provider", "mut-prov", "--from", "v1", "--to", "v2", "--json"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        outer = _extract_json(result.output)
        assert outer["status"] == "ok"
        data = _extract_result_json(result.output)
        type_mutations = [
            c
            for c in _expect_object_list(data["classified_changes"], "classified_changes")
            if c["kind"] == "type_mutation"
        ]
        assert len(type_mutations) == 1
        assert type_mutations[0]["path"] == "count"
        detail = str(type_mutations[0]["detail"])
        assert "integer" in detail
        assert "string" in detail

    def test_compare_subtractive(self, runner: CliRunner, schema_storage: Path) -> None:
        """Register schemas where fields are removed, verify classification."""
        registry = SchemaRegistry(storage_root=schema_storage)
        registry.register_schema(
            "sub-prov",
            {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "integer"},
                    "c": {"type": "boolean"},
                },
            },
        )
        registry.register_schema(
            "sub-prov",
            {"type": "object", "properties": {"a": {"type": "string"}}},
        )

        with _patch_registry(registry):
            result = runner.invoke(
                cli,
                ["schema", "compare", "--provider", "sub-prov", "--from", "v1", "--to", "v2", "--json"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        outer = _extract_json(result.output)
        assert outer["status"] == "ok"
        data = _extract_result_json(result.output)
        assert sorted(_expect_string_list(data["removed_properties"], "removed_properties")) == ["b", "c"]
        removed = [
            c for c in _expect_object_list(data["classified_changes"], "classified_changes") if c["kind"] == "removed"
        ]
        assert len(removed) == 2

    def test_compare_nonexistent_version_fails(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(
                cli,
                ["schema", "compare", "--provider", "test-provider", "--from", "v1", "--to", "v99"],
            )

        assert result.exit_code != 0


# =============================================================================
# schema explain
# =============================================================================


class TestSchemaExplainCommand:
    def test_explain_json_output(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(cli, ["schema", "explain", "--provider", "test-provider", "--json"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        outer = _extract_json(result.output)
        assert outer["status"] == "ok"
        data = _extract_result_json(result.output)
        assert "schema" in data
        assert "package" in data
        schema = _expect_object(data["schema"], "schema")
        properties = _expect_object(schema["properties"], "schema.properties")
        assert "id" in properties

    def test_explain_latest_version(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(cli, ["schema", "explain", "--provider", "test-provider"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # Latest is v2, which has "status" and "tags"
        assert "status" in result.output
        assert "tags" in result.output

    def test_explain_specific_version(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(
                cli,
                ["schema", "explain", "--provider", "test-provider", "--version", "v1"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # v1 has properties but NOT "status"/"tags"
        assert "id" in result.output
        assert "title" in result.output

    def test_explain_nonexistent_fails(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(
                cli,
                ["schema", "explain", "--provider", "nonexistent"],
            )

        assert result.exit_code != 0

    def test_explain_shows_metadata(self, runner: CliRunner, seeded_registry: SchemaRegistry) -> None:
        with _patch_registry(seeded_registry):
            result = runner.invoke(cli, ["schema", "explain", "--provider", "test-provider"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Metadata" in result.output
        assert "Version" in result.output

    def test_explain_json_includes_annotations(self, runner: CliRunner, schema_storage: Path) -> None:
        """Schemas with x-polylogue-* annotations surface in explain --json."""
        registry = SchemaRegistry(storage_root=schema_storage)
        registry.register_schema(
            "ann-prov",
            {
                "type": "object",
                "properties": {
                    "msg": {
                        "type": "string",
                        "x-polylogue-semantic-role": "message_body",
                        "x-polylogue-format": "markdown",
                    },
                },
                "x-polylogue-foreign-keys": [{"source": "$.msg_id", "target": "$.id"}],
            },
        )

        with _patch_registry(registry):
            result = runner.invoke(cli, ["schema", "explain", "--provider", "ann-prov", "--json"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        outer = _extract_json(result.output)
        assert outer["status"] == "ok"
        data = _extract_result_json(result.output)
        schema = _expect_object(data["schema"], "schema")
        properties = _expect_object(schema["properties"], "schema.properties")
        msg = _expect_object(properties["msg"], "schema.properties.msg")
        assert msg["x-polylogue-semantic-role"] == "message_body"
        assert "x-polylogue-foreign-keys" in schema


# =============================================================================
# devtools schema-promote (end-to-end with clustering)
# =============================================================================


class TestSchemaPromoteCommand:
    def test_promote_creates_new_version(
        self,
        capsys: pytest.CaptureFixture[str],
        schema_storage: Path,
    ) -> None:
        """Full promote workflow: cluster -> save manifest -> promote via devtools."""
        registry = SchemaRegistry(storage_root=schema_storage)
        samples = [
            {"id": "1", "title": "a", "count": 1},
            {"id": "2", "title": "b", "count": 2},
        ]
        manifest = registry.cluster_samples("promo-prov", samples)
        registry.save_cluster_manifest(manifest)
        cluster_id = manifest.clusters[0].cluster_id

        with _patch_registry(registry):
            exit_code = schema_promote.main(["--provider", "promo-prov", "--cluster", cluster_id])

        output = capsys.readouterr().out
        assert exit_code == 0, f"devtools failed: {output}"
        assert "v1" in output
        assert "Promoted" in output or "promoted" in output

    def test_promote_json_output(self, capsys: pytest.CaptureFixture[str], schema_storage: Path) -> None:
        registry = SchemaRegistry(storage_root=schema_storage)
        samples = [{"key": "val"}]
        manifest = registry.cluster_samples("promo-json", samples)
        registry.save_cluster_manifest(manifest)
        cluster_id = manifest.clusters[0].cluster_id

        with _patch_registry(registry):
            exit_code = schema_promote.main(["--provider", "promo-json", "--cluster", cluster_id, "--json"])

        output = capsys.readouterr().out
        assert exit_code == 0, f"devtools failed: {output}"
        outer = _extract_json(output)
        assert outer["status"] == "ok"
        data = _extract_result_json(output)
        assert data["provider"] == "promo-json"
        assert data["cluster_id"] == cluster_id
        assert data["package_version"] == "v1"
        package = _expect_object(data["package"], "package")
        assert package["version"] == "v1"
        assert data["schema"] is not None

    def test_promote_already_promoted_fails(
        self,
        capsys: pytest.CaptureFixture[str],
        schema_storage: Path,
    ) -> None:
        registry = SchemaRegistry(storage_root=schema_storage)
        samples = [{"x": 1}]
        manifest = registry.cluster_samples("dup-promo", samples)
        registry.save_cluster_manifest(manifest)
        cluster_id = manifest.clusters[0].cluster_id

        # First promotion succeeds
        registry.promote_cluster("dup-promo", cluster_id)

        with _patch_registry(registry):
            exit_code = schema_promote.main(["--provider", "dup-promo", "--cluster", cluster_id])

        assert exit_code != 0
        assert "schema-promote:" in capsys.readouterr().err

    def test_promote_nonexistent_cluster_fails(
        self,
        capsys: pytest.CaptureFixture[str],
        schema_storage: Path,
    ) -> None:
        registry = SchemaRegistry(storage_root=schema_storage)
        samples = [{"x": 1}]
        manifest = registry.cluster_samples("ne-promo", samples)
        registry.save_cluster_manifest(manifest)

        with _patch_registry(registry):
            exit_code = schema_promote.main(["--provider", "ne-promo", "--cluster", "nonexistent-id"])

        assert exit_code != 0
        assert "schema-promote:" in capsys.readouterr().err

    def test_promote_no_manifest_fails(self, capsys: pytest.CaptureFixture[str], schema_storage: Path) -> None:
        registry = SchemaRegistry(storage_root=schema_storage)

        with _patch_registry(registry):
            exit_code = schema_promote.main(["--provider", "no-manifest", "--cluster", "any-id"])

        assert exit_code != 0
        assert "schema-promote:" in capsys.readouterr().err


# =============================================================================
# Full operator workflow: list -> compare -> promote -> explain
# =============================================================================


class TestFullOperatorWorkflow:
    def test_end_to_end_workflow(
        self,
        runner: CliRunner,
        capsys: pytest.CaptureFixture[str],
        schema_storage: Path,
    ) -> None:
        """Simulate an operator's complete schema management workflow."""
        registry = SchemaRegistry(storage_root=schema_storage)

        # Step 1: Register initial schema (simulating infer output)
        schema_v1 = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "message": {"type": "string"},
                "timestamp": {"type": "number"},
            },
            "required": ["id"],
        }
        registry.register_schema("workflow-prov", schema_v1)

        # Step 2: List — verify v1 is visible
        with _patch_registry(registry):
            result = runner.invoke(
                cli,
                ["schema", "list", "--provider", "workflow-prov", "--json"],
            )
        assert result.exit_code == 0
        outer = _extract_json(result.output)
        assert outer["status"] == "ok"
        data = _extract_result_json(result.output)
        assert "v1" in _expect_string_list(data["versions"], "versions")

        # Step 3: Register v2 with structural changes
        schema_v2 = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "message": {"type": "string"},
                "timestamp": {"type": "number"},
                "model": {"type": "string"},  # new
                "tokens": {"type": "integer"},  # new
            },
            "required": ["id", "model"],
        }
        registry.register_schema("workflow-prov", schema_v2)

        # Step 4: Compare v1 vs v2 — verify additive + requiredness changes
        with _patch_registry(registry):
            result = runner.invoke(
                cli,
                [
                    "schema",
                    "compare",
                    "--provider",
                    "workflow-prov",
                    "--from",
                    "v1",
                    "--to",
                    "v2",
                    "--json",
                ],
            )
        assert result.exit_code == 0
        outer = _extract_json(result.output)
        assert outer["status"] == "ok"
        diff = _extract_result_json(result.output)
        assert diff["has_changes"] is True
        assert "model" in _expect_string_list(diff["added_properties"], "added_properties")
        assert "tokens" in _expect_string_list(diff["added_properties"], "added_properties")
        # "id" was required in both, but "model" is new (only in v2) so shows as additive,
        # not as a requiredness change. Check that "id" remained consistent (no req change).
        added_kinds = [
            c for c in _expect_object_list(diff["classified_changes"], "classified_changes") if c["kind"] == "added"
        ]
        assert any(c["path"] == "model" for c in added_kinds)

        # Step 5: Cluster some samples and promote
        samples = [
            {"id": "1", "message": "hi", "timestamp": 1.0, "model": "gpt-4", "tokens": 100},
            {"id": "2", "message": "bye", "timestamp": 2.0, "model": "gpt-4", "tokens": 200},
        ]
        manifest = registry.cluster_samples("workflow-prov", samples)
        registry.save_cluster_manifest(manifest)
        cluster_id = manifest.clusters[0].cluster_id

        with _patch_registry(registry):
            exit_code = schema_promote.main(["--provider", "workflow-prov", "--cluster", cluster_id, "--json"])
        output = capsys.readouterr().out
        assert exit_code == 0
        outer = _extract_json(output)
        assert outer["status"] == "ok"
        promo = _extract_result_json(output)
        assert promo["package_version"] == "v3"

        # Step 6: Explain the promoted version
        with _patch_registry(registry):
            result = runner.invoke(
                cli,
                ["schema", "explain", "--provider", "workflow-prov", "--version", "v3", "--json"],
            )
        assert result.exit_code == 0
        outer = _extract_json(result.output)
        assert outer["status"] == "ok"
        schema_v3 = _extract_result_json(result.output)
        schema = _expect_object(schema_v3["schema"], "schema")
        assert schema["x-polylogue-anchor-profile-family-id"] == cluster_id
        assert schema["x-polylogue-observed-artifact-count"] == 2
        assert "x-polylogue-promoted-at" in schema
        assert schema["x-polylogue-version"] == 3

        # Step 7: Final list shows all 3 versions
        with _patch_registry(registry):
            result = runner.invoke(
                cli,
                ["schema", "list", "--provider", "workflow-prov", "--json"],
            )
        assert result.exit_code == 0
        outer = _extract_json(result.output)
        assert outer["status"] == "ok"
        final = _extract_result_json(result.output)
        assert _expect_string_list(final["versions"], "versions") == ["v1", "v2", "v3"]
