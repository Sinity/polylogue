"""Generated Claude async evidence keeps its original task, call and child joins."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections import defaultdict
from html import unescape
from pathlib import Path

import pytest

from polylogue.core.enums import BlockType
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.audit.workflow import audit_schema_bundle_privacy
from polylogue.schemas.generation.workflow import build_provider_bundle_from_sources, persist_generated_provider_bundle
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.source_inference import SchemaSourceInput
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.models import SyntheticSchemaSelection
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.sources.dispatch import parse_payload
from tests.infra.claude_async_sources import claude_async_records, write_claude_async_sources

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
    # The native carrier uses HTML text semantics, including raw ampersands.
    # Read its fixed direct fields independently of production notification helpers.
    envelope = re.fullmatch(r"\s*<task-notification>\s*(.*?)\s*</task-notification>\s*", content, re.DOTALL)
    if envelope is None:
        return None
    body = envelope.group(1)
    fields: dict[str, str] = {}
    offset = 0
    for field in re.finditer(r"<([a-z-]+)>([^<]*)</\1>\s*", body):
        assert not body[offset : field.start()].strip()
        assert field.group(1) not in fields
        fields[field.group(1)] = unescape(field.group(2)).strip()
        offset = field.end()
    assert not body[offset:].strip()
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
            if not candidates:
                witnesses.add("unmatched_pair" if tool is not None else "unmatched_task")
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
        assert (block.is_error, block.exit_code, block.outcome_unknown_reason) == outcome, (
            key,
            outcome,
            (block.is_error, block.exit_code, block.outcome_unknown_reason),
        )
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
    mains = write_claude_async_sources(tmp_path / "sources", unmatched=True)
    for main in mains:
        assert _native_async_oracle(main) == _REQUIRED | {"unmatched_pair", "unmatched_task"}


def _selection_from_sources(
    tmp_path: Path, *, unmatched: bool = False, repeated_ack: bool = False, ambiguous_only: bool = False
) -> SyntheticSchemaSelection:
    sources = tmp_path / "sources"
    if ambiguous_only:
        sources.mkdir()
        isolated: list[Path] = []
        for number in range(1, 4):
            rows = claude_async_records(number)
            selected = [rows[0]]
            for row in rows[1:]:
                message = row.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if (
                    isinstance(content, list)
                    and any(
                        isinstance(block, dict)
                        and str(block.get("id", block.get("tool_use_id", ""))).startswith("call-ambiguous-")
                        for block in content
                    )
                    or isinstance(content, str)
                    and "<task-id>task-ambiguous</task-id>" in content
                ):
                    selected.append(row)
            main = sources / f"neutral-parent-{number}.jsonl"
            main.write_text("".join(json.dumps(row) + "\n" for row in selected))
            assert "ambiguous_completion" in _native_async_oracle(main)
            isolated.append(main)
        mains = tuple(isolated)
    else:
        mains = write_claude_async_sources(sources, unmatched=unmatched)
    if repeated_ack:
        for index, main in enumerate(mains):
            records = _records(main)
            position = next(
                i for i, row in enumerate(records) if row.get("toolUseResult") == {"backgroundTaskId": "task-failure"}
            )
            duplicate = dict(records[position])
            duplicate["uuid"] = str(uuid.UUID(int=9000 + index, version=4))
            records.insert(position + 1, duplicate)
            main.write_text("".join(json.dumps(row) + "\n" for row in records))
            assert "failed_completion" not in _native_async_oracle(main)
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
    return selection


@pytest.fixture(scope="module")
def generated_async(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("inferred-async")
    selection = _selection_from_sources(root)
    written = SyntheticCorpus.write_selection_artifacts(
        selection, CorpusSpec(provider="claude-code", count=8, seed=31), root / "generated", prefix="parent"
    )
    assert len(written.session_files) == 8
    return root / "generated"


def _check_generated_cohort(root: Path) -> None:
    observed: set[str] = set()
    mains = sorted(root.glob("*/parent-*.jsonl"))
    assert len(mains) == 8
    for main in mains:
        observed.update(_native_async_oracle(main))
    # Mixed finite samples need not visit every support class. The isolated
    # ambiguous source below proves that class without a visitation guarantee.
    assert _REQUIRED - {"ambiguous_completion"} <= observed <= _REQUIRED, observed


def test_inferred_claude_async_records_keep_task_and_open_child_joins(generated_async: Path) -> None:
    _check_generated_cohort(generated_async)


@pytest.mark.parametrize("loss", ("open_parent", "metadata_dispatch", "notification_provenance"))
def test_generated_native_oracle_rejects_lost_async_relations(generated_async: Path, tmp_path: Path, loss: str) -> None:
    root = tmp_path / "mutated"
    shutil.copytree(generated_async, root)
    changed = False
    for main in sorted(root.glob("*/parent-*.jsonl")):
        records = _records(main)
        calls = {
            block["id"]: block
            for record in records
            for message in [record.get("message")]
            if isinstance(message, dict)
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_use"
        }
        other_call = next((identity for identity, block in calls.items() if block["name"] != "Agent"), None)
        for record in records:
            data = record.get("data")
            if loss == "open_parent" and not changed and isinstance(data, dict) and data.get("agentId") and other_call:
                record["parentToolUseID"] = other_call
                changed = True
            if loss == "notification_provenance" and _notification(record) is not None:
                record["origin"] = {"kind": "human"}
                changed = True
        if loss == "metadata_dispatch" and not changed and other_call:
            files = list((main.parent / main.stem / "subagents").glob("*.meta.json"))
            if files:
                metadata = json.loads(files[0].read_text())
                metadata["toolUseId"] = other_call
                files[0].write_text(json.dumps(metadata))
                changed = True
        main.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert changed
    with pytest.raises(AssertionError):
        _check_generated_cohort(root)


def test_unmatched_notification_evidence_is_not_promoted_to_an_executable_pair(tmp_path: Path) -> None:
    selection = _selection_from_sources(tmp_path, unmatched=True)
    with pytest.raises(ValueError, match="unmatched notification relation"):
        SyntheticCorpus.write_selection_artifacts(
            selection, CorpusSpec(provider="claude-code", count=8, seed=31), tmp_path / "generated", prefix="parent"
        )


def test_repeated_background_acknowledgements_are_not_one_completion(tmp_path: Path) -> None:
    selection = _selection_from_sources(tmp_path, repeated_ack=True)
    with pytest.raises(ValueError, match="repeated(_ack notification relation| background acknowledgement)"):
        SyntheticCorpus.write_selection_artifacts(
            selection, CorpusSpec(provider="claude-code", count=8, seed=31), tmp_path / "generated", prefix="parent"
        )


@pytest.mark.parametrize("escaped", (False, True))
def test_native_notification_text_keeps_raw_and_escaped_ampersand_identity(tmp_path: Path, escaped: bool) -> None:
    mains = write_claude_async_sources(tmp_path / "sources")
    main = mains[0]
    text = main.read_text().replace("call-failure----", "&!!!!!!!!!!!!!!!")
    if escaped:
        text = text.replace(
            "<tool-use-id>&!!!!!!!!!!!!!!!</tool-use-id>", "<tool-use-id>&amp;!!!!!!!!!!!!!!!</tool-use-id>"
        )
    main.write_text(text)
    assert "failed_completion" in _native_async_oracle(main)


def test_inferred_ambiguous_task_only_source_keeps_acknowledgements_unknown(tmp_path: Path) -> None:
    selection = _selection_from_sources(tmp_path, ambiguous_only=True)
    written = SyntheticCorpus.write_selection_artifacts(
        selection, CorpusSpec(provider="claude-code", count=8, seed=31), tmp_path / "generated", prefix="parent"
    )
    observed: set[str] = set()
    for main in written.session_files:
        observed.update(_native_async_oracle(main))
    assert "ambiguous_completion" in observed
    assert "failed_completion" not in observed
