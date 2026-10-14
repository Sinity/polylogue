"""The original generated owner retains schema support once, without per-row copies."""

from __future__ import annotations

from contextlib import closing

import pytest

from polylogue.schemas.synthetic.models import SchemaRecord
from polylogue.schemas.synthetic.relations import ForeignKeyGraph


def test_original_schema_capture_preserves_exact_branch_overlay_and_lifetime() -> None:
    owner = ForeignKeyGraph()
    schema: SchemaRecord = {"type": "object", "properties": {"e\u0301": {"type": "string", "enum": ["e\u0301"]}}}
    captured = owner.capture_schema(schema)
    original = owner.new_record(captured, path="$.original")
    overlay: SchemaRecord = {**captured, "x-polylogue-frequency": 0.25}
    selected = owner.new_record(overlay, path="$.branch")
    assert original.selected_schema == schema
    assert selected.selected_schema == overlay
    assert selected.original_path == "$.branch"
    # A changed original input gets a distinct captured snapshot; previously
    # generated rows keep their exact support, not the caller's later mutation.
    properties = schema["properties"]
    assert isinstance(properties, dict)
    properties["added"] = {"type": "integer"}
    changed = owner.new_record(owner.capture_schema(schema))
    assert changed.selected_schema == schema
    old_properties = original.selected_schema["properties"]
    overlay_properties = selected.selected_schema["properties"]
    assert isinstance(old_properties, dict) and isinstance(overlay_properties, dict)
    assert "added" not in old_properties
    assert "added" not in overlay_properties
    assert "e\u0301" in old_properties
    owner.close()
    with pytest.raises(RuntimeError):
        _ = selected.selected_schema
    assert not owner._captured_schema_nodes


def test_generated_schema_storage_scales_with_support_not_record_count() -> None:
    owner = ForeignKeyGraph()
    schema: SchemaRecord = {
        "type": "object",
        "properties": {
            f"neutral_{index}": {"type": "string", "enum": [f"neutral_value_{index}"]} for index in range(1000)
        },
    }
    try:
        captured = owner.capture_schema(schema)
        connection = owner._owner()
        with closing(connection.execute("SELECT count(*) FROM generated_schema_nodes")) as cursor:
            original_nodes = cursor.fetchone()[0]
        for _ in range(10_003):
            owner.new_record(captured)
        with closing(connection.execute("SELECT count(*) FROM generated_schema_nodes")) as cursor:
            assert cursor.fetchone()[0] == original_nodes
        with closing(
            connection.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        ) as cursor:
            assert cursor.fetchone()[0] < 2 * 1024 * 1024
    finally:
        owner.close()


def test_uncaptured_schema_changes_are_not_hidden_by_native_interning() -> None:
    owner = ForeignKeyGraph()
    schema: SchemaRecord = {"type": "object", "properties": {"original": {"type": "string"}}}
    try:
        first = owner.new_record(schema)
        properties = schema["properties"]
        assert isinstance(properties, dict)
        properties["original"] = {"type": "number"}
        second = owner.new_record(schema)
        assert first.selected_schema != second.selected_schema
        assert first.selected_schema["properties"] == {"original": {"type": "string"}}
        assert second.selected_schema == schema
        cyclic: SchemaRecord = {}
        cyclic["cycle"] = cyclic
        with pytest.raises(ValueError):
            owner.new_record(cyclic)
    finally:
        owner.close()


def test_repeated_wire_reads_share_captured_support_and_only_restore_overlay_parent() -> None:
    import tracemalloc

    from polylogue.schemas.synthetic import SyntheticCorpus
    from polylogue.schemas.synthetic.build_wire_formats import _wire_schema
    from polylogue.schemas.synthetic.models import GeneratedRecord
    from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS

    schema: SchemaRecord = {
        "type": "object",
        "properties": {
            f"neutral_{index}": {"type": "string", "enum": [f"neutral_value_{index}"]} for index in range(1000)
        },
    }
    corpus = SyntheticCorpus(schema, PROVIDER_WIRE_FORMATS["codex"], "codex")
    owner = corpus._relation_solver.fk_graph
    try:
        captured = owner.capture_schema(schema)
        overlay: SchemaRecord = {**captured, "x-polylogue-frequency": 0.25}
        rows = owner.new_array()
        for index in range(1003):
            owner.append_array_item(rows, owner.new_record(captured if index % 2 == 0 else overlay))
        properties = captured["properties"]
        assert isinstance(properties, dict)
        tracemalloc.start()
        try:
            for index, row in enumerate(rows):
                assert isinstance(row, GeneratedRecord)
                if index % 2 == 0:
                    assert row.selected_schema is captured
                else:
                    assert row.selected_schema["properties"] is properties
                    assert row.selected_schema["x-polylogue-frequency"] == 0.25
                # This is the actual producer-side wire lookup, including its
                # repeated selected_schema accesses and original row currency.
                assert _wire_schema(corpus, row, ("neutral_0",)) is properties["neutral_0"]
            _, peak = tracemalloc.get_traced_memory()
            assert peak < 2 * 1024 * 1024
        finally:
            tracemalloc.stop()
    finally:
        corpus._relation_solver.close()
