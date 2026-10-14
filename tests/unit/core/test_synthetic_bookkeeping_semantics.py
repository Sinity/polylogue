"""Bookkeeping facts must survive genuine inference and anonymous generation."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import cache

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


def _check(clause: str, records: list[dict], provider: str) -> None:
    sessions = parse_payload(provider, records, "neutral-bookkeeping")
    assert len(sessions) == 1
    parsed = sessions[0]
    payloads = [record.get("payload", {}) for record in records]
    if clause == "turn_carriers":
        headers = [p for r, p in zip(records, payloads, strict=True) if r["type"] == "session_meta"]
        assert len(headers) == 1
        active = None
        turns = []
        events = 0
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
                events += 1
        assert len(set(turns)) >= 2 and events >= 2
    elif clause == "completion_types":
        items = [p.get("item") for p in payloads if p.get("type") == "item_completed"]
        assert items and all(isinstance(item, dict) for item in items), items
        assert {item["type"] for item in items} == {"CommandExecution", "FileChange"}
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
        assert len(contexts) == 3
        values = [p.get("user_instructions") for p in contexts]
        assert all(isinstance(value, str) and value for value in values)
        assert values[0] == values[1] and values[2] != values[0]
        assert all("developer_instructions" not in p for p in contexts)
        assert parsed.instructions_text == values[0]
        changes = [
            e.payload["instructions"] for e in parsed.session_events if e.event_type == "codex_instructions_changed"
        ]
        assert changes == [values[2]]
    elif clause == "reasoning":
        summaries = [p.get("summary") for p in payloads if p.get("type") == "reasoning"]
        assert summaries
        expected = []
        for summary in summaries:
            assert summary and all(isinstance(part.get("text"), str) and part["text"] for part in summary)
            expected.append("\n\n".join(part["text"] for part in summary))
        actual = [
            block.text for message in parsed.messages for block in message.blocks if block.type is BlockType.THINKING
        ]
        assert actual == expected
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
        expected = []
        for record in assistants:
            value = {target: record[source] for source, target in keys.items()}
            assert all(isinstance(item, str) and item for item in value.values())
            value["summary"] = f"{value['mcp_server']}:{value['mcp_tool']}"
            expected.append(value)
        actual = [
            event.payload for event in parsed.session_events if event.event_type == "claude_capability_attribution"
        ]
        assert actual == expected


@pytest.mark.parametrize("clause", CLAUSES)
def test_native_bookkeeping_oracle(clause: str) -> None:
    provider = "claude-code" if clause in {"relocation", "attribution"} else "codex"
    _check(clause, bookkeeping_records(provider, 1), provider)


@pytest.mark.parametrize("clause", CLAUSES)
def test_native_oracle_rejects_lost_bookkeeping(clause: str) -> None:
    provider = "claude-code" if clause in {"relocation", "attribution"} else "codex"
    records = bookkeeping_records(provider, 1)
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
        _check(clause, records, provider)


@pytest.fixture(scope="module")
def generated_bookkeeping(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], list[list[dict]]]:
    root = tmp_path_factory.mktemp("bookkeeping")
    packages = root / "packages"
    for provider in ("claude-code", "codex"):
        sources = root / provider
        sources.mkdir()
        for number in range(1, 4):
            (sources / f"session-{number}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in bookkeeping_records(provider, number))
            )
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
        assert receipt["source_terminal_outcomes"] == {"included": 3}
        persist_generated_provider_bundle(packages, provider, bundle)
    registry = SchemaRegistry(storage_root=packages)
    audit = audit_schema_bundle_privacy(registry=registry)
    assert audit.failed == 0, audit.format_text()

    @cache
    def generate(provider: str) -> list[list[dict]]:
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
    clause: str, generated_bookkeeping: Callable[[str], list[list[dict]]]
) -> None:
    provider = "claude-code" if clause in {"relocation", "attribution"} else "codex"
    for records in generated_bookkeeping(provider):
        _check(clause, records, provider)
