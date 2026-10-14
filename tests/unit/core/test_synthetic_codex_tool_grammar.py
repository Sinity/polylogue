"""Generated tool programs must reach the production semantic parser."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from polylogue.core.enums import BlockType
from polylogue.core.json import JSONDocument
from polylogue.schemas.audit.workflow import audit_schema_bundle_privacy
from polylogue.schemas.generation.workflow import build_provider_bundle_from_sources, persist_generated_provider_bundle
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.source_inference import SchemaSourceInput
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.schemas.tool_input_syntax import measured_tool_syntax, validated_syntax_parts
from polylogue.sources.dispatch import parse_payload

_PATCH = "*** Begin Patch\n*** Update File: neutral.txt\n@@\n-old\n+new\n*** End Patch"
_PROGRAM = f'await tools.exec_command({{"cmd":"printf neutral"}});\nawait tools.apply_patch({json.dumps(_PATCH)});'
_PROGRAM += '\n// tools.apply_patch("*** Update File: commented_private.txt");\n/* tools.exec_command({"cmd":"commented_private_command"}); */'


def _records(case: str, number: int, body_size: int = 0) -> list[JSONDocument]:
    patch = _PATCH if not body_size else _PATCH.replace("+new", "+" + "q" * body_size)
    return [
        {
            "type": "session_meta",
            "payload": {"id": str(uuid.UUID(int=number, version=4)), "timestamp": "2025-01-01T00:00:00Z"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Neutral edit."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "call-neutral",
                "name": "apply_patch" if case == "patch" else "exec",
                "arguments": patch if case == "patch" else _PROGRAM,
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-neutral",
                "output": {"output": "neutral done", "exit_code": 0, "is_error": False},
            },
        },
    ]


def _patch_paths(records: list[JSONDocument]) -> list[str]:
    sessions = parse_payload("codex", records, "neutral-tool-program")
    paths: list[str] = []
    for session in sessions:
        for message in session.messages:
            for block in message.blocks:
                if block.tool_name != "apply_patch" or block.tool_input is None:
                    continue
                path = block.tool_input.get("path")
                if isinstance(path, str):
                    paths.append(path)
    return paths


@pytest.mark.parametrize(("case", "body_size"), (("patch", 0), ("program", 0), ("patch", 2 << 20)))
def test_measured_tool_program_reaches_actual_patch_path(tmp_path: Path, case: str, body_size: int) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    for number in range(1, 4):
        records = _records(case, number, body_size)
        assert _patch_paths(records) == ["neutral.txt"]
        original_results = [
            block
            for parsed in parse_payload("codex", records, "original-tool")
            for message in parsed.messages
            for block in message.blocks
            if block.type is BlockType.TOOL_RESULT
        ]
        assert len(original_results) == 1
        assert original_results[0].exit_code == 0 and original_results[0].is_error is False
        (sources / f"session-{number}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
    bundle = build_provider_bundle_from_sources(
        "codex",
        source_inputs=(SchemaSourceInput("codex", sources),),
        cache_path=tmp_path / "evidence.sqlite3",
        max_workers=1,
        privacy_config=None,
        prior_catalog=None,
    )
    assert bundle.result.success, (bundle.result.error, bundle.result.phase_receipt)
    measured_profile = json.dumps(bundle.package_workload_profiles, default=dict)
    assert "neutral.txt" not in measured_profile
    assert "printf neutral" not in measured_profile
    assert "commented_private" not in measured_profile
    root = tmp_path / "packages"
    persist_generated_provider_bundle(root, "codex", bundle)
    registry = SchemaRegistry(storage_root=root)
    audit = audit_schema_bundle_privacy(registry=registry)
    assert audit.failed == 0, audit.format_text()
    selection = select_synthetic_schema("codex", registry_factory=lambda: registry)
    corpus = SyntheticCorpus.from_selection(selection)
    generated = corpus.iter_sessions(count=4, seed=31)
    paths: list[str] = []
    try:
        for session in generated:
            records = [json.loads(line) for line in b"".join(session.chunks).splitlines()]
            observed = _patch_paths(records)
            assert observed == ["x" * len("neutral.txt")], records
            blocks = [
                block
                for parsed in parse_payload("codex", records, "generated-tool")
                for message in parsed.messages
                for block in message.blocks
            ]
            if case == "program":
                assert {"exec_command", "apply_patch"} <= {block.tool_name for block in blocks}
                assert sum(block.tool_name == "exec_command" for block in blocks) == 1
            outcomes = [block for block in blocks if block.type is BlockType.TOOL_RESULT]
            native_outputs: list[JSONDocument] = []
            for record in records:
                payload = record.get("payload")
                if isinstance(payload, dict) and payload.get("type") == "function_call_output":
                    native_outputs.append(payload)
            assert len(native_outputs) == len(outcomes) == 1
            output = native_outputs[0]["output"]
            assert isinstance(output, dict)
            assert type(output["exit_code"]) is int and type(output["is_error"]) is bool
            # Numeric privacy does not publish an observed zero magnitude.
            # This checks conservation of the generated native verdict, not
            # a claim that the original all-success joint has been recovered.
            assert outcomes[0].exit_code == output["exit_code"]
            assert outcomes[0].is_error == output["is_error"]
            assert outcomes[0].tool_id == native_outputs[0]["call_id"]
            # The production parser must lose the path when the measured
            # header is removed, even though byte length and newlines remain.
            damaged = json.loads(json.dumps(records).replace("*** Update File: ", "x" * len("*** Update File: ")))
            assert _patch_paths(damaged) == []
            if case == "program":
                damaged = json.loads(json.dumps(records).replace("tools.apply_patch", "x" * len("tools.apply_patch")))
                assert _patch_paths(damaged) == []
            paths.extend(observed)
    finally:
        generated.close()
    assert len(paths) == 4


@pytest.mark.parametrize(
    "suffix",
    ('; "private_unterminated_Ω', '; "private_escaped_end_\\"', "; /* private_comment_Ω", "; // private_comment_Ω"),
)
def test_malformed_suffix_cannot_publish_a_private_literal(suffix: str) -> None:
    evidence = measured_tool_syntax({"name": "exec", "arguments": _PROGRAM + suffix})
    assert "private" not in evidence and "Ω" not in evidence
    syntax = json.loads(evidence)
    for _ in validated_syntax_parts(syntax["arguments"]):
        pass


def test_public_words_in_command_values_are_anonymous_content() -> None:
    source = 'tools.exec_command({"cmd":"command"});\n' + _PROGRAM
    syntax = json.loads(measured_tool_syntax({"name": "exec", "arguments": source}))
    assert ["literal", "cmd"] in syntax["arguments"]
    assert ["literal", "command"] not in syntax["arguments"]
    formatted = json.loads(measured_tool_syntax({"name": "exec", "arguments": "00000000-0000-4000-8000-000000000001"}))
    assert "arguments" not in formatted


@pytest.mark.parametrize("body", ("q" * (2 << 20), "qΩ" * (1 << 20)))
def test_unknown_names_remain_anonymous_and_large_patch_bodies_are_runs(body: str) -> None:
    assert measured_tool_syntax({"name": "private_tool", "arguments": _PATCH}) == ""
    assert measured_tool_syntax({"name": {"private": "value"}, "arguments": _PATCH}) == ""
    patch = "*** Begin Patch\n*** Update File: private_path.txt\n+" + body + "\n*** End Patch"
    evidence = measured_tool_syntax({"name": "apply_patch", "arguments": patch})
    assert len(evidence) < 1024
    assert "private_path" not in evidence
    parts = json.loads(evidence)["arguments"]
    width = 0
    for kind, value in validated_syntax_parts(parts):
        if kind == "literal":
            assert isinstance(value, str)
            width += len(value)
        else:
            assert isinstance(value, int)
            width += value
    assert width == len(patch)
