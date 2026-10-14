"""Bookkeeping facts must survive genuine inference and anonymous generation."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import cache
from typing import Any

import pytest

from polylogue.core.enums import BlockType
from polylogue.schemas.audit.workflow import audit_schema_bundle_privacy
from polylogue.schemas.generation.workflow import build_provider_bundle_from_sources, persist_generated_provider_bundle
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.source_inference import SchemaSourceInput
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.sources.dispatch import parse_payload
from tests.infra.bookkeeping_sources import bookkeeping_records

CLAUSES = (
    "turn_carriers",
    "completion_types",
    "message_roles",
    "instructions",
    "reasoning",
    "relocation",
    "attribution",
)


def _check(clause: str, records: list[dict[str, Any]], provider: str, *, authored_instructions: bool = False) -> None:
    sessions = parse_payload(provider, records, "neutral-bookkeeping")
    assert len(sessions) == 1
    parsed = sessions[0]
    if provider == "claude-code":
        users = [record for record in records if record.get("type") == "user"]
        assert users and all(record.get("origin") == {"kind": "human"} for record in users)
    payloads = [record.get("payload", {}) for record in records]
    if clause == "turn_carriers":
        headers = [p for r, p in zip(records, payloads, strict=True) if r["type"] == "session_meta"]
        assert len(headers) == 1
        active = None
        turns = []
        event_count = 0
        for record, payload in zip(records, payloads, strict=True):
            if record["type"] == "turn_context":
                active = payload.get("turn_id")
                assert isinstance(active, str) and active
                turns.append(active)
            if record["type"] == "event_msg" and payload.get("type") == "agent_message":
                assert active is not None
                assert payload["thread_id"] == headers[0]["id"]
                assert (
                    payload["turn_id"]
                    == payload["metadata"]["turn_id"]
                    == payload["internal_chat_message_metadata_passthrough"]["turn_id"]
                    == active
                )
                event_count += 1
        assert len(set(turns)) >= 2 and event_count >= 2
    elif clause == "completion_types":
        items = [p.get("item") for p in payloads if p.get("type") == "item_completed"]
        assert items and all(isinstance(item, dict) for item in items), items
        assert {item["type"] for item in items} <= {"CommandExecution", "FileChange"}
    elif clause == "message_roles":
        events = [
            p
            for r, p in zip(records, payloads, strict=True)
            if r["type"] == "event_msg" and p.get("type") == "agent_message"
        ]
        users = [
            p
            for r, p in zip(records, payloads, strict=True)
            if r["type"] == "response_item" and p.get("type") == "message"
        ]
        assert events and users
        assert all("role" not in p for p in events)
        assert all(p["role"] == "user" for p in users)
        expected = [("assistant", p["message"]) for p in events]
        expected += [("user", part["text"]) for p in users for part in p["content"]]
        actual = [
            (str(message.role), block.text)
            for message in parsed.messages
            for block in message.blocks
            if block.type is BlockType.TEXT
        ]
        for pair in expected:
            assert pair in actual
    elif clause == "instructions":
        contexts = [p for r, p in zip(records, payloads, strict=True) if r["type"] == "turn_context"]
        assert len(contexts) >= 2
        values = [p.get("user_instructions") for p in contexts]
        assert all(isinstance(value, str) and value for value in values)
        if authored_instructions:
            assert len(values) == 3 and values[0] == values[1] and values[2] != values[0]
        assert all("developer_instructions" not in p for p in contexts)
        assert parsed.instructions_text == values[0]
        changes = [
            e.payload["instructions"] for e in parsed.session_events if e.event_type == "codex_instructions_changed"
        ]
        assert changes == list(dict.fromkeys(value for value in values[1:] if value != values[0]))
    elif clause == "reasoning":
        summaries = [p.get("summary") for p in payloads if p.get("type") == "reasoning"]
        assert summaries
        expected_summaries: list[str] = []
        for summary in summaries:
            assert summary and all(isinstance(part.get("text"), str) and part["text"] for part in summary)
            expected_summaries.append("\n\n".join(part["text"] for part in summary))
        actual_thinking = [
            block.text for message in parsed.messages for block in message.blocks if block.type is BlockType.THINKING
        ]
        assert actual_thinking == expected_summaries
    elif clause == "relocation":
        paths = [r["relocatedCwd"] for r in records if r["type"] == "relocated"]
        assert paths and all(isinstance(path, str) and path for path in paths)
        assert all(path in parsed.working_directories for path in paths)
    elif clause == "attribution":
        keys = {
            "attributionAgent": "agent",
            "attributionSkill": "skill",
            "attributionMcpServer": "mcp_server",
            "attributionMcpTool": "mcp_tool",
            "attributionPlugin": "plugin",
        }
        assistants = [r for r in records if r["type"] == "assistant"]
        assert assistants
        expected_attribution = []
        for record in assistants:
            value = {target: record[source] for source, target in keys.items()}
            assert all(isinstance(item, str) and item for item in value.values())
            value["summary"] = f"{value['mcp_server']}:{value['mcp_tool']}"
            expected_attribution.append(value)
        actual_attribution = [
            event.payload for event in parsed.session_events if event.event_type == "claude_capability_attribution"
        ]
        assert actual_attribution == expected_attribution


@pytest.mark.parametrize("clause", CLAUSES)
def test_native_bookkeeping_oracle(clause: str) -> None:
    provider = "claude-code" if clause in {"relocation", "attribution"} else "codex"
    _check(clause, bookkeeping_records(provider, 1), provider, authored_instructions=True)


@pytest.mark.parametrize("clause", CLAUSES)
def test_native_oracle_rejects_lost_bookkeeping(clause: str) -> None:
    provider = "claude-code" if clause in {"relocation", "attribution"} else "codex"
    records: list[dict[str, Any]] = bookkeeping_records(provider, 1)
    if clause == "turn_carriers":
        event = next(r["payload"] for r in records if r.get("payload", {}).get("type") == "agent_message")
        event["metadata"]["turn_id"] = "stale-turn"
    elif clause == "completion_types":
        event = next(r["payload"] for r in records if r.get("payload", {}).get("type") == "item_completed")
        event["item"]["type"] = "anonymous-item"
    elif clause == "message_roles":
        event = next(r["payload"] for r in records if r.get("payload", {}).get("type") == "agent_message")
        event["role"] = "record:event_msg:agent"
    elif clause == "instructions":
        contexts = [r["payload"] for r in records if r["type"] == "turn_context"]
        contexts[1]["user_instructions"] = contexts[2]["user_instructions"]
    elif clause == "reasoning":
        event = next(r["payload"] for r in records if r.get("payload", {}).get("type") == "reasoning")
        event["summary"] = []
    elif clause == "relocation":
        next(r for r in records if r["type"] == "relocated")["relocatedCwd"] = ""
    elif clause == "attribution":
        next(r for r in records if r["type"] == "assistant")["attributionAgent"] = ""
    with pytest.raises(AssertionError):
        _check(clause, records, provider, authored_instructions=True)


@pytest.fixture(scope="module")
def generated_bookkeeping(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], list[list[dict[str, Any]]]]:
    root = tmp_path_factory.mktemp("bookkeeping")
    packages = root / "packages"
    for provider in ("claude-code", "codex"):
        sources = root / provider
        sources.mkdir()
        for number in range(1, 4):
            source_records: list[dict[str, Any]] = bookkeeping_records(provider, number)
            if provider == "codex":
                # Original repeated-instruction law: the measured scope is constant.
                for record in source_records:
                    if record["type"] == "turn_context":
                        record["payload"]["user_instructions"] = "Neutral first instruction."
            (sources / f"session-{number}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in source_records))
        bundle = build_provider_bundle_from_sources(
            provider,
            source_inputs=(SchemaSourceInput(provider, sources),),
            cache_path=root / f"{provider}.sqlite3",
            max_workers=1,
            privacy_config=None,
            prior_catalog=None,
        )
        assert bundle.result.success, (provider, bundle.result.error, bundle.result.phase_receipt)
        receipt = bundle.result.phase_receipt["source"]
        assert isinstance(receipt, dict)
        assert receipt["source_terminal_outcomes"] == {"included": 3}
        persist_generated_provider_bundle(packages, provider, bundle)
    registry = SchemaRegistry(storage_root=packages)
    audit = audit_schema_bundle_privacy(registry=registry)
    assert audit.failed == 0, audit.format_text()

    @cache
    def generate(provider: str) -> list[list[dict[str, Any]]]:
        selection = select_synthetic_schema(provider, registry_factory=lambda: registry)
        generated = SyntheticCorpus.from_selection(selection).iter_sessions(count=4, seed=31)
        try:
            result = [[json.loads(line) for line in b"".join(session.chunks).splitlines()] for session in generated]
        finally:
            generated.close()
        assert len(result) == 4
        return result

    return generate


@pytest.mark.parametrize("clause", CLAUSES)
def test_inferred_bookkeeping_matches_generated_native_fields(
    clause: str, generated_bookkeeping: Callable[[str], list[list[dict[str, Any]]]]
) -> None:
    provider = "claude-code" if clause in {"relocation", "attribution"} else "codex"
    generated = generated_bookkeeping(provider)
    if clause == "completion_types":
        assert {
            record["payload"]["item"]["type"]
            for records in generated
            for record in records
            if record.get("payload", {}).get("type") == "item_completed"
        } == {"CommandExecution", "FileChange"}
    for records in generated:
        _check(clause, records, provider)
        if clause == "instructions":
            values = [record["payload"]["user_instructions"] for record in records if record["type"] == "turn_context"]
            assert len(set(values)) == 1


@pytest.mark.parametrize("extra", ("progress", "relocated", "neutral private discriminator"))
def test_partial_safe_values_are_not_an_exhaustive_field_enum(extra: str) -> None:
    from polylogue.core.json import JSONDocument
    from polylogue.schemas.field_stats.collection import _collect_field_stats
    from polylogue.schemas.field_stats.evidence import (
        deserialize_field_stats,
        finalize_field_stats,
        serialize_field_stats,
    )
    from polylogue.schemas.generation.field_annotations import annotate_schema

    rows: list[JSONDocument] = [{"type": "user"}, {"type": "assistant"}, {"type": extra}]
    observed = _collect_field_stats(rows)
    retained = {path: deserialize_field_stats(serialize_field_stats(stats)) for path, stats in observed.items()}
    finalize_field_stats(retained)
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "x-polylogue-semantic-role": "message_role",
                "x-polylogue-values": ["user", "assistant"],
            }
        },
    }
    annotate_schema(schema, retained)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    leaf = properties["type"]
    assert isinstance(leaf, dict)
    assert "x-polylogue-values" not in leaf
    distribution = leaf["x-polylogue-observed-distribution"]
    assert isinstance(distribution, dict)
    assert distribution["type_counts"] == {"string": 3}
    assert extra not in json.dumps(schema)


@pytest.mark.parametrize("nullable_numeric", (False, True))
def test_complete_safe_string_enum_keeps_rare_members_and_other_types(nullable_numeric: bool) -> None:
    from polylogue.core.json import JSONDocument
    from polylogue.schemas.field_stats.collection import _collect_field_stats
    from polylogue.schemas.generation.field_annotations import annotate_schema
    from polylogue.schemas.synthetic.observed_values import existing_value_is_supported

    rows: list[JSONDocument] = [{"type": kind} for kind in ["user"] * 50 + ["assistant"] * 50 + ["tool"]]
    if nullable_numeric:
        rows.extend([{"type": None}, {"type": 7}])
    schema: JSONDocument = {
        "type": "object",
        "properties": {
            "type": {
                "type": ["string", "null", "integer"] if nullable_numeric else "string",
                "x-polylogue-semantic-role": "message_role",
            }
        },
    }
    annotate_schema(schema, _collect_field_stats(rows))
    properties = schema["properties"]
    assert isinstance(properties, dict)
    leaf = properties["type"]
    assert isinstance(leaf, dict)
    values = leaf["x-polylogue-values"]
    assert isinstance(values, list)
    assert set(values) == {"user", "assistant", "tool"}
    if nullable_numeric:
        assert existing_value_is_supported(leaf, None)
        assert existing_value_is_supported(leaf, 7)
