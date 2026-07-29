"""Tests for the schema-vs-parser field coverage join (polylogue-2qx.3).

Exercises the static join the drift sentinel's fourth classification
(``KNOWN_FIELD_UNREAD``) depends on: which committed-schema field names does
no parser module for a provider reference as a string constant.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from polylogue.schemas.schema_parser_coverage import (
    parser_referenced_field_names,
    payload_unread_field_names,
    schema_known_field_names,
    unread_field_names,
)


def _write_schema(root: Path, provider: str, element_kind: str, schema: dict[str, object]) -> None:
    element_dir = root / provider / "versions" / "v1" / "elements"
    element_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(schema).encode("utf-8")
    (element_dir / f"{element_kind}.schema.json.gz").write_bytes(gzip.compress(payload))


def _write_parser(root: Path, name: str, source: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


class TestSchemaKnownFieldNames:
    def test_collects_nested_and_dynamic_property_names(self, tmp_path: Path) -> None:
        schema_root = tmp_path / "providers"
        _write_schema(
            schema_root,
            "example",
            "session_document",
            {
                "type": "object",
                "properties": {
                    "sessionId": {"type": "string"},
                    "toolUseResult": {
                        "type": "object",
                        "properties": {"oldTodos": {"type": "array"}},
                    },
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"stop_reason": {"type": "string"}},
                        },
                    },
                    "extra": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {"agentId": {"type": "string"}},
                        },
                    },
                },
            },
        )
        names = schema_known_field_names("example", schema_root=schema_root)
        assert {"sessionId", "toolUseResult", "oldTodos", "messages", "stop_reason", "agentId"} <= names

    def test_missing_provider_directory_returns_empty(self, tmp_path: Path) -> None:
        assert schema_known_field_names("nonexistent-provider", schema_root=tmp_path / "providers") == frozenset()


class TestParserReferencedFieldNames:
    def test_collects_string_constants_from_configured_modules(self, tmp_path: Path) -> None:
        parser_root = tmp_path / "parsers"
        _write_parser(
            parser_root,
            "example_parser.py",
            'def parse(record):\n    return record.get("sessionId"), record["messages"]\n',
        )
        names = parser_referenced_field_names(
            "example",
            parser_root=parser_root,
        )
        # "example" has no PROVIDER_PARSERS module mapping, so this must be
        # empty even though the file exists -- the mapping, not directory
        # scanning, decides which modules own a provider's wire format.
        assert names == frozenset()


class TestUnreadFieldNames:
    def test_field_read_by_parser_is_excluded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import polylogue.schemas.schema_parser_coverage as coverage

        monkeypatch.setitem(coverage.PROVIDER_PARSERS, "example", ("example_parser.py",))
        schema_root = tmp_path / "providers"
        parser_root = tmp_path / "parsers"
        _write_schema(
            schema_root,
            "example",
            "session_document",
            {
                "type": "object",
                "properties": {
                    "sessionId": {"type": "string"},
                    "structuredPatch": {"type": "array"},
                },
            },
        )
        _write_parser(
            parser_root,
            "example_parser.py",
            'def parse(record):\n    return record.get("sessionId")\n',
        )
        coverage.clear_cache()
        try:
            unread = unread_field_names("example", schema_root=schema_root, parser_root=parser_root)
            assert "structuredPatch" in unread
            assert "sessionId" not in unread
        finally:
            coverage.clear_cache()

    def test_no_schema_known_fields_yields_no_unread_fields(self, tmp_path: Path) -> None:
        assert unread_field_names("nonexistent-provider", schema_root=tmp_path / "providers") == frozenset()


class TestPayloadUnreadFieldNames:
    def test_intersects_payload_keys_with_provider_unread_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import polylogue.schemas.schema_parser_coverage as coverage

        monkeypatch.setitem(coverage.PROVIDER_PARSERS, "example", ("example_parser.py",))
        schema_root = tmp_path / "providers"
        parser_root = tmp_path / "parsers"
        _write_schema(
            schema_root,
            "example",
            "session_document",
            {
                "type": "object",
                "properties": {
                    "sessionId": {"type": "string"},
                    "structuredPatch": {"type": "array"},
                    "agentId": {"type": "string"},
                },
            },
        )
        _write_parser(
            parser_root,
            "example_parser.py",
            'def parse(record):\n    return record.get("sessionId")\n',
        )
        coverage.clear_cache()
        try:
            result = payload_unread_field_names(
                "example",
                ["sessionId", "structuredPatch", "somethingNotInSchema"],
                schema_root=schema_root,
                parser_root=parser_root,
            )
            # sessionId is read by the parser, somethingNotInSchema isn't a
            # schema-known field at all (that's NEW_FIELD's job, not this
            # join's) -- only structuredPatch is schema-known AND unread.
            assert result == ["structuredPatch"]
        finally:
            coverage.clear_cache()

    def test_empty_payload_field_names_short_circuits(self, tmp_path: Path) -> None:
        assert payload_unread_field_names("example", [], schema_root=tmp_path / "providers") == []
