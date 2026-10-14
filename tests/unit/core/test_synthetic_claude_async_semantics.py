"""Generated Claude async evidence keeps its original task, call and child joins."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree

from polylogue.core.enums import BlockType
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.audit.workflow import audit_schema_bundle_privacy
from polylogue.schemas.generation.workflow import build_provider_bundle_from_sources, persist_generated_provider_bundle
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.source_inference import SchemaSourceInput
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.sources.dispatch import parse_payload
from tests.infra.claude_async_sources import write_claude_async_sources

_REQUIRED = {"foreground", "failed_completion", "unknown_ack", "ambiguous_completion", "open_agent", "no_child_claim"}


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _child_stem(identity: str) -> str:
    return identity if identity.startswith("agent-") else "agent-" + identity


def _notification(record: dict[str, object]) -> dict[str, str] | None:
    """Read only the neutral cohort's trusted native XML carrier, independently."""
    origin = record.get("origin")
    attachment = record.get("attachment")
    trusted = (
        isinstance(origin, dict)
        and origin.get("kind") == "task-notification"
        or record.get("type") == "queue-operation"
        or isinstance(attachment, dict)
        and attachment.get("type") == "queued_command"
    )
    if not trusted:
        return None
    message = record.get("message")
    content = message.get("content") if isinstance(message, dict) else record.get("content")
    if not isinstance(content, str):
        return None
    try:
        envelope = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return None
    if envelope.tag != "task-notification":
        return None
    fields: dict[str, str] = {}
    for field in envelope:
        assert field.tag not in fields and not list(field)
        fields[field.tag] = (field.text or "").strip()
    assert {"task-id", "output-file", "status", "summary"} <= fields.keys()
    assert all(fields.values())
    return fields


def _native_async_oracle(main: Path) -> set[str]:
    """Use native IDs and physical files as expectations, never parser join helpers."""
    records = _records(main)
    calls: dict[str, dict[str, object]] = {}
    open_calls: set[str] = set()
    starts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    start_pairs: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    expected: dict[tuple[str, str], tuple[bool | None, int | None, str | None]] = {}
    progress: dict[str, int] = defaultdict(int)
    results: dict[str, int] = defaultdict(int)
    children: dict[str, set[str]] = defaultdict(set)
    notifications: list[dict[str, str]] = []
    witnesses: set[str] = set()
    physical_children = main.parent / main.stem / "subagents"
    for record in records:
        if record.get("type") == "progress":
            data = record.get("data")
            if isinstance(data, dict) and data.get("type") == "agent_progress":
                parent = record.get("parentToolUseID")
                child = data.get("agentId")
                if isinstance(parent, str) and parent:
                    progress[parent] += 1
                if not any(calls[identity]["name"] == "Agent" for identity in open_calls):
                    assert "agentId" not in data, "unbound progress invented a child claim"
                    witnesses.add("no_child_claim")
                if isinstance(child, str) and child:
                    assert isinstance(parent, str) and parent in open_calls, "progress lost its open call"
                    assert calls[parent]["name"] == "Agent", "progress borrowed a non-Agent call"
                    children[parent].add(_child_stem(child))
                    witnesses.add("open_agent")
        notification = _notification(record)
        if notification is not None:
            notifications.append(notification)
        message = record.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                identity = block.get("id")
                assert isinstance(identity, str) and identity
                calls[identity] = block
                open_calls.add(identity)
            elif block.get("type") == "tool_result":
                identity = block.get("tool_use_id")
                assert isinstance(identity, str) and identity in calls
                key = str(record["uuid"]), identity
                error = block.get("is_error")
                assert isinstance(error, bool)
                expected[key] = error, None, None
                evidence = record.get("toolUseResult")
                name = calls[identity]["name"]
                if name == "Bash":
                    operand = calls[identity].get("input")
                    assert isinstance(operand, dict) and isinstance(evidence, dict)
                    background = operand.get("run_in_background") is True
                    assert background == ("backgroundTaskId" in evidence), "Bash call/result subtype was lost"
                    if background:
                        task = evidence["backgroundTaskId"]
                        assert isinstance(task, str) and task
                        starts[task].append(key)
                        start_pairs[task, identity].append(key)
                        expected[key] = None, None, "distrusted"
                    else:
                        assert "stdout" in evidence
                        witnesses.add("foreground")
                if name == "Agent":
                    assert isinstance(evidence, dict) and isinstance(evidence.get("agentId"), str)
                    agent = evidence["agentId"]
                    assert isinstance(agent, str)
                    stem = _child_stem(agent)
                    children[identity].add(stem)
                    assert len(children[identity]) == 1, "progress and result name different children"
                    child = physical_children / (stem + ".jsonl")
                    metadata = physical_children / (stem + ".meta.json")
                    assert child.is_file() and metadata.is_file(), "dispatch lost its actual child or metadata"
                    meta = json.loads(metadata.read_bytes())
                    assert meta["toolUseId"] == identity and meta["agentId"] == agent
                    assert parse_payload("claude-code", _records(child), child.stem, source_path=str(child))
                    results[identity] += 1
                open_calls.discard(identity)
    # A progress claim must name a physical child even if this sampled
    # transcript ends before its result. No result timing is invented here.
    for parent in progress:
        for stem in children[parent]:
            assert (physical_children / (stem + ".jsonl")).is_file(), "progress names no physical child"
            meta = json.loads((physical_children / (stem + ".meta.json")).read_bytes())
            assert meta["toolUseId"] == parent, "progress borrowed another dispatch's child"
    notification_exits: dict[tuple[str, str | None], int] = {}
    for notification in notifications:
        task, tool = notification["task-id"], notification.get("tool-use-id")
        summary = notification["summary"]
        # This source cohort contains the fixed public failed-command protocol.
        # Its numeric operand comes from generated bytes, never from the parser.
        assert notification["status"] == "failed" and summary.startswith('Background command "')
        command, separator, code = summary.rpartition('" failed with exit code ')
        assert command and separator and code.isdecimal(), "known completion grammar was lost"
        exit_code = int(code)
        assert exit_code != 0
        notification_exits[task, tool] = exit_code
        candidates = start_pairs[task, tool] if tool is not None else starts[task]
        if len(candidates) != 1:
            if tool is None and len(candidates) > 1:
                witnesses.add("ambiguous_completion")
            continue
        expected[candidates[0]] = True, exit_code, None
        witnesses.add("failed_completion")
    notified_tasks = {row["task-id"] for row in notifications}
    if any(
        task not in notified_tasks and expected[key] == (None, None, "distrusted")
        for task, keys in starts.items()
        for key in keys
    ):
        witnesses.add("unknown_ack")
    parsed = parse_payload("claude-code", records, main.stem, source_path=str(main))
    actual = {
        (message.provider_message_id, block.tool_id): block
        for session in parsed
        for message in session.messages
        for block in message.blocks
        if block.type is BlockType.TOOL_RESULT
    }
    assert expected.keys() <= actual.keys()
    for key, outcome in expected.items():
        block = actual[key]
        assert (block.is_error, block.exit_code, block.outcome_unknown_reason) == outcome
    events = {
        event.payload["parent_tool_use_id"]: event.payload
        for session in parsed
        for event in session.session_events
        if event.event_type == "claude_delegation_progress"
    }
    for parent in progress.keys() | results.keys():
        event = events[parent]
        assert event["progress_tick_count"] == progress[parent]
        assert event["dispatch_result_count"] == results[parent]
        expected_children = children[parent]
        assert event["child_provider_id"] == (next(iter(expected_children)) if len(expected_children) == 1 else None)
    completions = [
        event
        for session in parsed
        for event in session.session_events
        if event.event_type == "background_task_completion"
    ]
    actual_notifications = {
        (event.payload["task_id"], event.payload["tool_use_id"]): event.payload for event in completions
    }
    assert actual_notifications.keys() == notification_exits.keys()
    for row in notifications:
        notification_key = row["task-id"], row.get("tool-use-id")
        payload = actual_notifications[notification_key]
        assert payload["status"] == row["status"]
        assert payload["output_file"] == row["output-file"]
        assert payload["summary"] == row["summary"]
        assert payload["exit_code"] == notification_exits[notification_key]
    return witnesses


def test_neutral_async_sources_have_real_native_task_and_child_joins(tmp_path: Path) -> None:
    mains = write_claude_async_sources(tmp_path / "sources")
    for main in mains:
        assert _native_async_oracle(main) == _REQUIRED


def test_inferred_claude_async_records_keep_task_and_open_child_joins(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    write_claude_async_sources(sources)
    bundle = build_provider_bundle_from_sources(
        "claude-code",
        source_inputs=(SchemaSourceInput("claude-code", sources),),
        cache_path=tmp_path / "evidence.sqlite3",
        max_workers=1,
        privacy_config=None,
        prior_catalog=None,
    )
    assert bundle.result.success, (bundle.result.error, bundle.result.phase_receipt)
    published = json.dumps(bundle.package_workload_profiles, default=dict)
    assert "neutral-child-" not in published and "task-failure" not in published
    packages = tmp_path / "packages"
    persist_generated_provider_bundle(packages, "claude-code", bundle)
    registry = SchemaRegistry(storage_root=packages)
    audit = audit_schema_bundle_privacy(registry=registry)
    assert audit.failed == 0, audit.format_text()
    selection = select_synthetic_schema("claude-code", registry_factory=lambda: registry)
    written = SyntheticCorpus.write_selection_artifacts(
        selection, CorpusSpec(provider="claude-code", count=8, seed=31), tmp_path / "generated", prefix="parent"
    )
    observed: set[str] = set()
    for main in written.session_files:
        observed.update(_native_async_oracle(main))
    assert observed == _REQUIRED
