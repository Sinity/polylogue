"""Measured native call classes and verdicts survive the current driver."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from polylogue.core.enums import BlockType
from polylogue.schemas.audit.workflow import audit_schema_bundle_privacy
from polylogue.schemas.generation.workflow import build_provider_bundle_from_sources, persist_generated_provider_bundle
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.source_inference import SchemaSourceInput
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.sources.dispatch import parse_payload
from tests.infra.codex_tool_sources import PRIVATE_TOOL, mixed_codex_tool_records, write_mixed_codex_tool_sources

_FREEFORM_HEADER = re.compile(
    r"\AExit code: (-?\d+)\nWall time: [0-9.]+ seconds\n(?:Output:\n|Total output lines: \d+\n)"
)
_EXEC_HEADER = re.compile(r"\AWall time: [0-9.]+ seconds\nProcess exited with code (-?\d+)(?=\n|$)")


def _native_outcomes(records: list[dict[str, object]]) -> Counter[tuple[bool, bool]]:
    """Read emitted native fields directly, without the parser's oracle."""
    calls: dict[str, str] = {}
    outcomes: Counter[tuple[bool, bool]] = Counter()
    expected: dict[str, tuple[bool | None, int | None]] = {}
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        kind = payload.get("type")
        identity = payload.get("call_id")
        if kind in {"function_call", "custom_tool_call"}:
            assert isinstance(identity, str) and isinstance(payload.get("name"), str)
            calls[identity] = payload["name"]
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            assert isinstance(identity, str) and identity in calls
            output = payload.get("output")
            name = calls[identity]
            if name not in {"exec_command", "apply_patch"}:
                assert name not in {"exec", "shell_command"}
                assert not isinstance(output, str) or "Process exited" not in output
                assert not isinstance(output, dict) or "exit_code" not in output
                expected[identity] = None, None
                outcomes[False, False] += 1
            elif isinstance(output, dict):
                code, error = output.get("exit_code"), output.get("is_error")
                assert type(code) is int and type(error) is bool
                assert error == (code != 0), (code, error)
                expected[identity] = error, code
                outcomes[True, error] += 1
            else:
                assert isinstance(output, str)
                match = (_EXEC_HEADER if name == "exec_command" else _FREEFORM_HEADER).match(output)
                assert match is not None, output
                code = int(match[1])
                expected[identity] = code != 0, code
                outcomes[True, code != 0] += 1
    actual = {
        block.tool_id: (block.is_error, block.exit_code)
        for session in parse_payload("codex", records, "neutral-native-results")
        for message in session.messages
        for block in message.blocks
        if block.type is BlockType.TOOL_RESULT
    }
    assert actual == expected
    return outcomes


def _qualify_results(tmp_path: Path, *, custom: bool = False) -> None:
    original = mixed_codex_tool_records(1, custom=custom)
    assert _native_outcomes(original) == {(True, False): 2, (True, True): 2, (False, False): 2}
    sources = write_mixed_codex_tool_sources(tmp_path / "sources", custom=custom)
    bundle = build_provider_bundle_from_sources(
        "codex",
        source_inputs=(SchemaSourceInput("codex", sources),),
        cache_path=tmp_path / "evidence.sqlite3",
        max_workers=1,
        privacy_config=None,
        prior_catalog=None,
    )
    assert bundle.result.success, (bundle.result.error, bundle.result.phase_receipt)
    published = json.dumps(bundle.package_workload_profiles, default=dict)
    assert PRIVATE_TOOL not in published and "printf neutral" not in published and "neutral plain" not in published
    root = tmp_path / "packages"
    persist_generated_provider_bundle(root, "codex", bundle)
    registry = SchemaRegistry(storage_root=root)
    audit = audit_schema_bundle_privacy(registry=registry)
    assert audit.failed == 0, audit.format_text()
    corpus = SyntheticCorpus.from_selection(select_synthetic_schema("codex", registry_factory=lambda: registry))
    sessions = corpus.iter_sessions(count=8, seed=41)
    observed: Counter[tuple[bool, bool]] = Counter()
    try:
        for session in sessions:
            records = [json.loads(line) for line in b"".join(session.chunks).splitlines()]
            assert PRIVATE_TOOL not in json.dumps(records)
            observed.update(_native_outcomes(records))
    finally:
        sessions.close()
    assert all(observed[key] for key in ((True, False), (True, True), (False, False))), observed


@pytest.mark.parametrize("custom", (False, True))
def test_inferred_codex_results_follow_call_class_and_measured_verdict(tmp_path: Path, custom: bool) -> None:
    _qualify_results(tmp_path, custom=custom)


def test_outcome_binding_loss_is_detected_by_native_oracle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.schemas.synthetic.record_profiles import PublicRecordPlan

    monkeypatch.setattr(PublicRecordPlan, "_bind_codex_result", lambda *_args: None)
    with pytest.raises(AssertionError):
        _qualify_results(tmp_path)


def test_call_class_binding_loss_is_detected_by_native_oracle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.schemas.synthetic.relations import ForeignKeyGraph

    original = ForeignKeyGraph.select_tool_identity
    lost_class = 0

    def without_class(
        self, relation, rng, prior_result_mode=None, prior_count_bucket=None, widths=None, tool_class=None
    ):
        nonlocal lost_class
        if tool_class is not None:
            other = "function_call:" if tool_class != "function_call:" else "function_call:exec_command"
            if self.tool_identity_count(relation, prior_result_mode, prior_count_bucket, widths, other):
                lost_class += 1
                return original(self, relation, rng, prior_result_mode, prior_count_bucket, widths, other)
        return original(self, relation, rng, prior_result_mode, prior_count_bucket, widths, tool_class)

    monkeypatch.setattr(ForeignKeyGraph, "select_tool_identity", without_class)
    with pytest.raises(AssertionError):
        _qualify_results(tmp_path)
    assert lost_class > 0


@pytest.mark.parametrize(
    ("source", "error"),
    (
        (
            "Command: /private/neutral-secret\nChunk ID: a9abcdef\nWall time: 0.25 seconds\nProcess exited with code 0\nprivate body Ω",
            False,
        ),
        ("Wall time: 0.25 seconds\nProcess completed with exit code 17\nprivate body Ω", True),
        ("Exit code: -17\nWall time: 0.25 seconds\nTotal output lines: 2\nprivate body Ω", True),
    ),
)
def test_result_header_grammar_is_public_and_keeps_only_verdict_class(source: str, error: bool) -> None:
    from polylogue.schemas.tool_input_syntax import measured_result_syntax, validated_syntax_parts

    descriptor = measured_result_syntax(source)
    encoded = json.dumps(descriptor)
    assert "private" not in encoded and "a9abcdef" not in encoded and "Ω" not in encoded
    rendered = "".join(
        value if kind == "literal" else {"ascii": "x", "non_ascii": "ñ", "digits": "1", "zeros": "0"}[kind] * value
        for kind, value in validated_syntax_parts(descriptor["output_syntax"])
    )
    assert len(rendered) == len(source) and rendered.count("\n") == source.count("\n")
    records = mixed_codex_tool_records(1)[:2] + [
        {
            "type": "response_item",
            "payload": {"type": "function_call", "call_id": "header-1", "name": "exec_command", "arguments": "{}"},
        },
        {
            "type": "response_item",
            "payload": {"type": "function_call_output", "call_id": "header-1", "output": rendered},
        },
    ]
    results = [
        block
        for session in parse_payload("codex", records, "neutral-header")
        for message in session.messages
        for block in message.blocks
        if block.type is BlockType.TOOL_RESULT
    ]
    assert len(results) == 1 and results[0].is_error is error
    assert (results[0].exit_code != 0) is error
    assert measured_result_syntax("private prefix\n" + source) == {}


@pytest.mark.parametrize(
    ("limits", "verdict", "expected"),
    (
        ([0, 0], "zero", [(0, 0, 1)]),
        ([0, 0], "nonzero", []),
        ([-4, -2], "nonzero", [(-4, -2, 3)]),
        ([-2, 2], "nonzero", [(-2, -1, 2), (1, 2, 2)]),
    ),
)
def test_numeric_verdict_conditions_the_existing_support(
    limits: list[int], verdict: str, expected: list[tuple[int, int, int]]
) -> None:
    from polylogue.schemas.synthetic.observed_values import numeric_verdict_support

    assert numeric_verdict_support({"type": "integer", "x-polylogue-range": limits}, verdict) == expected


def test_result_tokens_cannot_turn_argument_numbers_into_public_literals() -> None:
    from polylogue.schemas.tool_input_syntax import measured_tool_syntax

    descriptor = json.loads(
        measured_tool_syntax({"name": "exec", "arguments": 'await tools.exec_command({"private":0});'})
    )
    assert ["literal", "0"] not in descriptor["arguments"]
    assert "private" not in json.dumps(descriptor)


def test_result_form_eligibility_intersects_call_class_and_identity_width() -> None:
    from polylogue.schemas.synthetic.record_profiles import PublicRecordPlan
    from polylogue.schemas.synthetic.relations import ForeignKeyGraph

    owner = ForeignKeyGraph()
    owner._owner()
    try:
        owner.observe_tool_identity(
            "exec-wide", "call", "new", {"type": "string"}, tool_class="function_call:exec_command"
        )
        owner.observe_tool_identity("anon-6", "call", "new", {"type": "string"}, tool_class="function_call:")
        plan = object.__new__(PublicRecordPlan)
        plan.owner = owner
        plan.content = {
            "native_tool_field_forms": {
                "function_call_output": [
                    {"result_mode": "native-output:string", "syntax": {"call_class": name}}
                    for name in ("function_call:exec_command", "function_call:")
                ]
            }
        }
        forms = plan._codex_native_forms("function_call_output", "native-output:string", "call", (6,))
        assert [form["syntax"]["call_class"] for form in forms] == ["function_call:"]
        # Removing the width intersection admits the unavailable exec+6 pair.
        unconditioned = plan._codex_native_forms("function_call_output", "native-output:string", "call", None)
        assert len(unconditioned) == 2
        with pytest.raises(ValueError):
            owner.select_tool_identity("call", random.Random(31), widths=(6,), tool_class="function_call:exec_command")
    finally:
        owner.close()
