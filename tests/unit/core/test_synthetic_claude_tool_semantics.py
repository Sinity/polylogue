"""Measured Claude tools preserve their named input and result classes."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from polylogue.core.enums import BlockType
from polylogue.core.json import JSONDocument
from polylogue.schemas.audit.workflow import audit_schema_bundle_privacy
from polylogue.schemas.generation.workflow import build_provider_bundle_from_sources, persist_generated_provider_bundle
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.source_inference import SchemaSourceInput
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic import runtime as synthetic_runtime
from polylogue.schemas.synthetic.models import SchemaRecord
from polylogue.schemas.synthetic.record_profiles import PublicRecordPlan
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.sources.dispatch import parse_payload
from tests.infra.claude_tool_sources import write_mixed_claude_tool_sources


def _tool_calls(records: list[JSONDocument]) -> list[JSONDocument]:
    calls: list[JSONDocument] = []
    for record in records:
        message = record.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            calls.extend(block for block in content if isinstance(block, dict) and block.get("type") == "tool_use")
    return calls


def _assert_available_calls(records: list[JSONDocument]) -> int:
    unused: set[str] = set()
    peak = 0
    for record in records:
        message = record.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                identity = str(block["id"])
                assert identity not in unused
                unused.add(identity)
                peak = max(peak, len(unused))
            elif block.get("type") == "tool_result":
                identity = str(block["tool_use_id"])
                assert identity in unused
                unused.remove(identity)
    return peak


def _assert_named_inputs(records: list[JSONDocument]) -> set[str]:
    names: set[str] = set()
    for call in _tool_calls(records):
        name = call["name"]
        assert isinstance(name, str)
        names.add(name)
        operand = call["input"]
        assert isinstance(operand, dict)
        if name == "Read":
            assert set(operand) == {"file_path"}
        elif name == "Edit":
            assert set(operand) == {"file_path", "old_string", "new_string"}
        elif name == "Write":
            assert set(operand) == {"file_path", "content"}
        elif name == "TodoWrite":
            assert isinstance(operand["todos"], list) and operand["todos"]
        elif name == "Bash":
            assert "command" in operand and "file_path" not in operand
        elif name == "Agent":
            assert set(operand) == {"prompt", "subagent_type"}
    return names


def _assert_result_classes(records: list[JSONDocument], *, result_arrays: bool = False) -> set[str]:
    calls = {str(call["id"]): call for call in _tool_calls(records)}
    expected_edits: dict[str, JSONDocument] = {}
    expected_todos: list[JSONDocument] = []
    names: set[str] = set()
    for record in records:
        message = record.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for result in content:
            if not isinstance(result, dict) or result.get("type") != "tool_result":
                continue
            identity = str(result["tool_use_id"])
            call = calls[identity]
            name = call["name"]
            assert isinstance(name, str)
            names.add(name)
            if result_arrays:
                assert isinstance(result["content"], list) if name == "Edit" else isinstance(result["content"], str)
            evidence = record.get("toolUseResult")
            if name == "Read":
                assert evidence is None
            elif name in {"Edit", "Write"}:
                assert isinstance(evidence, dict) and "filePath" in evidence and "newString" in evidence
                assert "oldTodos" not in evidence and "agentId" not in evidence
                if name == "Edit":
                    assert "oldString" in evidence and "structuredPatch" in evidence
                expected_edits[identity] = evidence
            elif name == "TodoWrite":
                assert isinstance(evidence, dict) and set(evidence) == {"oldTodos", "newTodos"}
                expected_todos.append({"old_todos": evidence["oldTodos"], "new_todos": evidence["newTodos"]})
            elif name == "Bash":
                assert isinstance(evidence, dict) and "filePath" not in evidence and "agentId" not in evidence
                assert "stdout" in evidence or "backgroundTaskId" in evidence
            elif name == "Agent":
                assert isinstance(evidence, dict) and set(evidence) == {"agentId"}
    parsed = parse_payload("claude-code", records, "generated-mixed-tool-results")
    actual_edits = {
        block.tool_id: block.file_edit
        for session in parsed
        for message in session.messages
        for block in message.blocks
        if block.file_edit is not None
    }
    assert actual_edits.keys() == expected_edits.keys()
    for identity, evidence in expected_edits.items():
        edit = actual_edits[identity]
        assert edit is not None
        assert edit.file_path == evidence["filePath"]
        assert edit.new_string == evidence["newString"]
        if "oldString" in evidence:
            assert edit.old_string == evidence["oldString"]
        if "structuredPatch" in evidence:
            assert edit.structured_patch == evidence["structuredPatch"]
    assert [
        event.payload
        for session in parsed
        for event in session.session_events
        if event.event_type == "claude_todo_state"
    ] == expected_todos
    return names


def _qualify_mixed_tools(tmp_path: Path, *, multiple_calls: bool = False, result_arrays: bool = False) -> None:
    sources = tmp_path / "sources"
    originals = write_mixed_claude_tool_sources(sources, multiple_calls=multiple_calls, result_arrays=result_arrays)
    for records in originals:
        assert _assert_named_inputs(records) == {
            "Read",
            "Edit",
            "Write",
            "TodoWrite",
            "Bash",
            "Agent",
            "private_tool_neutral",
        }
        parsed = parse_payload("claude-code", records, "neutral-mixed-tools")
        blocks = [block for session in parsed for message in session.messages for block in message.blocks]
        assert {block.tool_name for block in blocks if block.type is BlockType.TOOL_USE} >= {"Read", "Edit"}
        assert sum(block.file_edit is not None for block in blocks) == 2
        _assert_result_classes(records, result_arrays=result_arrays)
    bundle = build_provider_bundle_from_sources(
        "claude-code",
        source_inputs=(SchemaSourceInput("claude-code", sources),),
        cache_path=tmp_path / "evidence.sqlite3",
        max_workers=1,
        privacy_config=None,
        prior_catalog=None,
    )
    assert bundle.result.success, (bundle.result.error, bundle.result.phase_receipt)
    assert "neutral.txt" not in json.dumps(bundle.package_workload_profiles, default=dict)
    assert "private_tool_neutral" not in json.dumps(bundle.package_workload_profiles, default=dict)
    packages = tmp_path / "packages"
    persist_generated_provider_bundle(packages, "claude-code", bundle)
    registry = SchemaRegistry(storage_root=packages)
    audit = audit_schema_bundle_privacy(registry=registry)
    assert audit.failed == 0, audit.format_text()
    corpus = SyntheticCorpus.from_selection(select_synthetic_schema("claude-code", registry_factory=lambda: registry))
    generated = corpus.iter_sessions(count=8, seed=31)
    names: set[str] = set()
    result_names: set[str] = set()
    mutation_checked = False
    multicall_seen = False
    try:
        for session in generated:
            records = [json.loads(line) for line in b"".join(session.chunks).splitlines()]
            multicall_seen |= any(len(_tool_calls([record])) > 1 for record in records)
            available = _assert_available_calls(records)
            if multiple_calls and len(_tool_calls(records)) > 1:
                assert available > 0
            names.update(_assert_named_inputs(records))
            result_names.update(_assert_result_classes(records, result_arrays=result_arrays))
            if not mutation_checked and any(call["name"] == "Edit" for call in _tool_calls(records)):
                damaged = json.loads(json.dumps(records))
                for call in _tool_calls(damaged):
                    if call["name"] == "Edit":
                        call["name"] = "Read"
                with pytest.raises(AssertionError):
                    _assert_named_inputs(damaged)
                with pytest.raises(AssertionError):
                    _assert_result_classes(damaged, result_arrays=result_arrays)
                mutation_checked = True
    finally:
        generated.close()
    assert {"Read", "Edit", "Write", "TodoWrite", "Bash", "Agent"} <= names & result_names
    assert names - {"Read", "Edit", "Write", "TodoWrite", "Bash", "Agent"}
    assert "private_tool_neutral" not in names
    assert mutation_checked
    if multiple_calls:
        assert multicall_seen


def test_inferred_mixed_claude_tools_preserve_named_semantics(tmp_path: Path) -> None:
    _qualify_mixed_tools(tmp_path)


def test_inferred_mixed_claude_result_modes_follow_call_classes(tmp_path: Path) -> None:
    _qualify_mixed_tools(tmp_path, result_arrays=True)


def test_inferred_multicall_claude_tools_preserve_named_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = PublicRecordPlan._bind_dispatch_fields

    def bind_fields(plan: PublicRecordPlan, rng: random.Random) -> None:
        try:
            original(plan, rng)
        except ValueError as error:
            cursor = plan.owner._live_tool_owner().execute(
                "SELECT calls,results,tool_class,count(*) FROM generated_tool_identities GROUP BY calls,results,tool_class ORDER BY calls,results,tool_class"
            )
            try:
                inventory = cursor.fetchall()
            finally:
                cursor.close()
            raise AssertionError(
                f"No eligible dispatcher for {plan.kind}; actual call inventory {inventory!r}"
            ) from error

    monkeypatch.setattr(PublicRecordPlan, "_bind_dispatch_fields", bind_fields)
    _qualify_mixed_tools(tmp_path, multiple_calls=True)


def test_multicall_qualification_detects_unconditioned_count_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = synthetic_runtime._sampled_length

    def without_floor(schema: SchemaRecord, name: str, rng: random.Random, *, minimum: int = 0) -> int | None:
        return original(schema, name, rng)

    monkeypatch.setattr(synthetic_runtime, "_sampled_length", without_floor)
    with pytest.raises(ValueError, match="eligible|matching|complete path"):
        _qualify_mixed_tools(tmp_path, multiple_calls=True)


def test_result_mode_qualification_detects_independent_class_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = PublicRecordPlan._claude_result_forms

    def without_mode(
        plan: PublicRecordPlan,
        mode: str,
        relation: str,
        prior_mode: str,
        prior_count: int,
        widths: tuple[int, ...] | None,
        tool_class: str | None,
    ) -> list[SchemaRecord]:
        native = plan.content["native_tool_field_forms"]
        assert isinstance(native, dict)
        forms = native["tool_result"]
        assert isinstance(forms, list)
        modes = {form["result_mode"] for form in forms if isinstance(form, dict)}
        return [
            form
            for candidate in sorted(str(value) for value in modes)
            for form in original(plan, candidate, relation, prior_mode, prior_count, widths, tool_class)
        ]

    monkeypatch.setattr(PublicRecordPlan, "_claude_result_forms", without_mode)
    with pytest.raises(AssertionError):
        _qualify_mixed_tools(tmp_path, result_arrays=True)
