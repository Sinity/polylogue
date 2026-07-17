from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import jsonschema
import pytest

from devtools.render_semantic_card_fixtures import (
    _load_case,
    build_case_transcript,
    render_case,
    render_case_entries,
)
from devtools.render_semantic_card_fixtures import main as fixture_main
from polylogue.core.enums import Origin
from polylogue.core.json import JSONDocument, require_json_document
from polylogue.rendering.block_models import coerce_renderable_block
from polylogue.rendering.semantic_card_models import (
    SemanticCard,
    SemanticCardField,
    SemanticCardKind,
    SemanticCardSource,
)
from polylogue.rendering.semantic_card_placement import semantic_card_placement_for_messages
from polylogue.rendering.semantic_card_registry import (
    card_kind_for_tool,
    provider_namespace_documents,
    semantic_type_policy_documents,
    tool_mapping_rows,
)
from polylogue.rendering.semantic_cards import (
    build_semantic_transcript,
    lineage_descriptor_from_archive_envelope,
)
from polylogue.rendering.semantic_markdown import render_semantic_card_markdown

CASES_ROOT = Path("tests/data/semantic_cards/cases")
SCHEMA_PATH = Path("docs/schemas/semantic-card-v1.schema.json")
TRANSCRIPT_SCHEMA_PATH = Path("docs/schemas/semantic-transcript-v1.schema.json")
CASE_PATHS = sorted(CASES_ROOT.glob("*.json"))


def _case(path: Path) -> dict[str, object]:
    return _load_case(path)


def _expected(case: dict[str, object]) -> list[JSONDocument]:
    value = case.get("expected_cards")
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return [require_json_document(item, context="expected semantic card") for item in value]


def test_attribute_block_metadata_retains_persisted_display_fields() -> None:
    block = SimpleNamespace(
        type=SimpleNamespace(value="image"),
        text=None,
        metadata=json.dumps(
            {
                "language": "python",
                "url": "https://example.test/plot.png",
                "name": "plot",
                "media_type": "image/png",
            }
        ),
    )

    rendered = coerce_renderable_block(block)

    assert rendered is not None
    assert rendered.language == "python"
    assert rendered.url == "https://example.test/plot.png"
    assert rendered.name == "plot"
    assert rendered.mime_type == "image/png"


@pytest.mark.parametrize("case_path", CASE_PATHS, ids=lambda path: path.stem)
def test_semantic_card_golden_cases(case_path: Path) -> None:
    case = _case(case_path)
    actual = render_case(case, case_path=case_path)
    assert actual == _expected(case)


@pytest.mark.parametrize("case_path", CASE_PATHS, ids=lambda path: path.stem)
def test_semantic_card_golden_cases_satisfy_public_schema(case_path: Path) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for card in _expected(_case(case_path)):
        jsonschema.validate(instance=card, schema=schema)


def test_golden_corpus_covers_every_card_kind_and_provider_family() -> None:
    providers: set[object] = set()
    kinds: set[object] = set()
    for path in CASE_PATHS:
        case = _case(path)
        providers.add(case.get("provider_family"))
        kinds.update(card.get("kind") for card in _expected(case))

    assert providers == {
        "antigravity",
        "chatgpt",
        "claude-ai",
        "claude-code",
        "codex",
        "gemini-cli",
        "grok",
        "hermes",
    }
    assert kinds == {
        "attachment",
        "fallback",
        "file_edit",
        "file_read",
        "lineage",
        "mcp",
        "search",
        "shell",
        "task",
        "web",
    }


def test_registry_is_explicit_unique_and_evidence_backed() -> None:
    rows = tool_mapping_rows()
    identities = [(row.provider_family.casefold(), row.tool_name.casefold()) for row in rows]
    assert len(identities) == len(set(identities))
    assert {row.provider_family for row in rows} == {
        "claude-code",
        "codex",
        "gemini-cli",
        "chatgpt",
        "hermes",
    }
    assert {row.rendering_status for row in rows} == {"launch", "model_only", "fallback"}
    assert {row.evidence_kind for row in rows} == {
        "fixture_observed",
        "parser_record_type",
        "classifier_contract",
    }
    for row in rows:
        evidence_path = Path(row.evidence)
        assert evidence_path.exists(), row
        assert row.tool_name.casefold() in evidence_path.read_text(encoding="utf-8").casefold(), row


def test_semantic_type_policy_is_exhaustive_over_persisted_vocabulary() -> None:
    rows = semantic_type_policy_documents()
    assert {row["semantic_type"] for row in rows} == {
        "other",
        "file_read",
        "file_write",
        "file_edit",
        "shell",
        "git",
        "search",
        "web",
        "agent",
        "subagent",
        "thinking",
    }
    by_type = {str(row["semantic_type"]): row for row in rows}
    assert by_type["shell"]["card_kind"] == "shell"
    assert by_type["file_edit"]["card_kind"] == "file_edit"
    assert by_type["subagent"]["rendering_status"] == "model_only"
    assert by_type["search"]["card_kind"] == "search"
    assert by_type["web"]["card_kind"] == "web"
    assert by_type["file_read"]["card_kind"] == "file_read"


def test_provider_namespace_policy_discloses_open_world_fallback() -> None:
    policies = provider_namespace_documents()
    assert {item["origin"] for item in policies} == {origin.value for origin in Origin}
    assert {item["provider_family"] for item in policies} == {
        "antigravity",
        "beads",
        "chatgpt",
        "claude-ai",
        "claude-code",
        "codex",
        "gemini-cli",
        "grok",
        "hermes",
        "unknown",
    }
    assert all(item["namespace"] == "open" for item in policies)
    assert all(
        item["unlisted_behavior"] == "structural_mcp_then_persisted_semantic_type_then_fallback_raw_evidence"
        for item in policies
    )


def test_unknown_tool_remains_fallback_even_when_input_looks_like_shell() -> None:
    assert (
        card_kind_for_tool(
            provider_family="codex",
            tool_name="unregistered_magic",
            semantic_type=None,
        ).value
        == "fallback"
    )


def test_persisted_semantic_type_is_provider_neutral() -> None:
    assert (
        card_kind_for_tool(
            provider_family="unknown",
            tool_name="provider_private_name",
            semantic_type="shell",
        ).value
        == "shell"
    )


def test_null_outcome_does_not_infer_success_from_prose() -> None:
    messages: list[dict[str, object]] = [
        {
            "id": "use",
            "role": "assistant",
            "message_type": "tool_use",
            "blocks": [
                {
                    "id": "use-block",
                    "type": "tool_use",
                    "tool_name": "exec_command",
                    "tool_id": "t",
                    "tool_input": {"command": "verify"},
                }
            ],
        },
        {
            "id": "result",
            "role": "tool",
            "message_type": "tool_result",
            "blocks": [
                {
                    "id": "result-block",
                    "type": "tool_result",
                    "tool_id": "t",
                    "text": "SUCCESS everything passed",
                }
            ],
        },
    ]
    card = build_semantic_transcript(messages, session_id="s", provider_family="codex").cards[0]
    assert card.outcome is not None
    assert card.outcome.state.value == "unknown"
    assert any("no structural" in caveat for caveat in card.caveats)


def test_explicit_success_does_not_infer_failure_from_error_word() -> None:
    case_path = CASES_ROOT / "claude-bash-explicit-success.json"
    card = render_case(_case(case_path), case_path=case_path)[0]
    assert cast(dict[str, object], card["outcome"])["state"] == "succeeded"


def test_large_result_preview_discloses_exact_omission() -> None:
    case_path = CASES_ROOT / "codex-exec-failure-ten-thousand-lines.json"
    card = render_case(_case(case_path), case_path=case_path)[0]
    preview = cast(list[dict[str, object]], card["previews"])[0]
    assert preview["line_count"] == 10_000
    assert preview["omitted_lines"] == 9_936
    assert preview["strategy"] == "head_tail"
    assert "line 00001" in cast(str, preview["text"])
    assert "line 10000" in cast(str, preview["text"])


def test_non_utf8_preview_is_marked_not_silently_dropped() -> None:
    case_path = CASES_ROOT / "hermes-shell-non-utf8.json"
    card = render_case(_case(case_path), case_path=case_path)[0]
    preview = cast(list[dict[str, object]], card["previews"])[0]
    assert preview["encoding_replacements"] == 1
    assert "\ufffd" in cast(str, preview["text"])
    assert any("invalid UTF-8" in caveat for caveat in cast(list[str], card["caveats"]))


def test_interleaved_subagent_results_pair_by_tool_id() -> None:
    case_path = CASES_ROOT / "claude-interleaved-subagents.json"
    cards = render_case(_case(case_path), case_path=case_path)
    sources = [cast(dict[str, object], card["source"]) for card in cards]
    assert [source["tool_id"] for source in sources] == ["t1", "t2"]
    assert [source["result_message_id"] for source in sources] == ["m-result-1", "m-result-2"]


def test_wrong_frozen_expectation_fails_the_comparison() -> None:
    """Anti-vacuity: a deliberately wrong expected card cannot pass."""

    case_path = CASES_ROOT / "claude-bash-explicit-success.json"
    case = _case(case_path)
    actual = render_case(case, case_path=case_path)
    wrong = copy.deepcopy(_expected(case))
    wrong[0]["kind"] = "attachment"
    with pytest.raises(AssertionError):
        assert actual == wrong


def test_fixture_checker_rejects_corrupted_frozen_card(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Executable anti-vacuity control for the golden-fixture command."""

    source = CASES_ROOT / "claude-bash-explicit-success.json"
    corrupted = _case(source)
    expected = corrupted.get("expected_cards")
    assert isinstance(expected, list)
    first = expected[0]
    assert isinstance(first, dict)
    first["kind"] = "attachment"
    path = tmp_path / "corrupted.json"
    path.write_text(json.dumps(corrupted), encoding="utf-8")

    assert fixture_main(["--check", "--fixture", str(path)]) == 1
    assert "full expected_cards contract differs" in capsys.readouterr().err


def test_pure_renderer_modules_do_not_import_storage_api_or_daemon() -> None:
    modules = (
        Path("polylogue/core/tool_identity.py"),
        Path("polylogue/rendering/block_models.py"),
        Path("polylogue/rendering/semantic_card_models.py"),
        Path("polylogue/rendering/semantic_card_registry.py"),
        Path("polylogue/rendering/semantic_cards.py"),
        Path("polylogue/rendering/semantic_markdown.py"),
    )
    forbidden = ("polylogue.storage", "polylogue.api", "polylogue.daemon", "polylogue.insights")
    violations: list[str] = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names.append(node.module)
            for name in names:
                if name.startswith(forbidden):
                    violations.append(f"{path}:{getattr(node, 'lineno', '?')}: {name}")
    assert violations == []


def test_card_construction_is_deterministic_and_does_not_mutate_input() -> None:
    case_path = CASES_ROOT / "claude-edit-diff.json"
    case = _case(case_path)
    before = copy.deepcopy(case)

    first = render_case(case, case_path=case_path)
    second = render_case(case, case_path=case_path)

    assert first == second
    assert case == before


def test_conflicting_error_flag_and_zero_exit_code_is_failed_with_caveat() -> None:
    messages: list[dict[str, object]] = [
        {
            "id": "use",
            "role": "assistant",
            "message_type": "tool_use",
            "blocks": [
                {
                    "id": "use-block",
                    "type": "tool_use",
                    "tool_name": "exec_command",
                    "tool_id": "t",
                    "tool_input": {"command": "verify"},
                }
            ],
        },
        {
            "id": "result",
            "role": "tool",
            "message_type": "tool_result",
            "blocks": [
                {
                    "id": "result-block",
                    "type": "tool_result",
                    "tool_id": "t",
                    "text": "provider disagrees",
                    "tool_result_is_error": True,
                    "tool_result_exit_code": 0,
                }
            ],
        },
    ]
    card = build_semantic_transcript(messages, session_id="s", provider_family="codex").cards[0]
    assert card.outcome is not None
    assert card.outcome.state.value == "failed"
    assert any("explicit error flag" in caveat for caveat in card.caveats)


def test_markdown_inline_code_preserves_backticks() -> None:
    case_path = CASES_ROOT / "codex-apply-patch-missing-result.json"
    card_document = render_case(_case(case_path), case_path=case_path)[0]
    # Build a small direct card through the public model to exercise a path
    # value that itself contains Markdown delimiters.

    card = SemanticCard(
        kind=SemanticCardKind.FILE_EDIT,
        title="File edit",
        source=SemanticCardSource(session_id="s"),
        fields=(SemanticCardField("path", "src/`odd`.py"),),
    )
    rendered = render_semantic_card_markdown(card)
    assert "`` src/`odd`.py ``" in rendered
    assert card_document["kind"] == "file_edit"


@pytest.mark.parametrize("case_path", CASE_PATHS, ids=lambda path: path.stem)
def test_semantic_transcript_golden_cases_satisfy_public_schema(case_path: Path) -> None:
    """Production dependency: SemanticTranscript.to_document and transcript schema.

    Mutation control: removing an entry discriminator or source coordinate makes
    jsonschema validation fail instead of allowing a card-only compatibility
    projection to stand in for the full transcript.
    """

    schema = json.loads(TRANSCRIPT_SCHEMA_PATH.read_text(encoding="utf-8"))
    document = build_case_transcript(_case(case_path), case_path=case_path).to_document()
    jsonschema.validate(instance=document, schema=schema)


def test_result_serialized_before_use_is_paired_once() -> None:
    """Production dependency: semantic_cards._pair_tool_results whole-document FIFO.

    Mutation control: restoring one-pass pairing emits an orphan fallback before
    the shell card and this exact one-entry assertion fails.
    """

    case_path = CASES_ROOT / "result-before-use.json"
    entries = render_case_entries(_case(case_path), case_path=case_path)
    assert [entry["entry_type"] for entry in entries] == ["card"]
    card = cast(dict[str, object], entries[0]["card"])
    assert card["kind"] == "shell"
    assert cast(dict[str, object], card["source"])["result_message_id"] == "m-result"


def test_empty_typed_thinking_compacts_by_contiguous_typed_run() -> None:
    """Production dependency: typed THINKING/REASONING run compaction.

    Mutation control: treating whitespace as prose, grouping across the text
    separator, or dropping empty blocks changes entry types/counts/source ids.
    """

    case_path = CASES_ROOT / "empty-thinking-run.json"
    entries = render_case_entries(_case(case_path), case_path=case_path)
    notices = [cast(dict[str, object], entry["notice"]) for entry in entries if entry["entry_type"] == "notice"]
    assert [notice["count"] for notice in notices] == [2, 1]
    first_sources = cast(list[dict[str, object]], notices[0]["sources"])
    assert [(source["block_type"], source["block_id"]) for source in first_sources] == [
        ("thinking", "think-a"),
        ("reasoning", "reason-a"),
    ]
    assert first_sources[0]["inherited_prefix"] is True
    assert first_sources[0]["variant_index"] == 1


def test_mixed_result_context_prevents_protocol_row_suppression() -> None:
    """Production dependency: shared placement over all semantic entry kinds.

    Mutation control: suppressing every paired result message loses both context
    prose entries and makes ``is_suppressed('m-result')`` true.
    """

    case_path = CASES_ROOT / "mixed-tool-result-context.json"
    case = _case(case_path)
    placement = semantic_card_placement_for_messages(
        cast(list[dict[str, object]], case["messages"]),
        session_id=str(case["session_id"]),
        provider_family=str(case["provider_family"]),
    )
    result_entries = placement.entries_for("m-result")
    assert placement.is_suppressed("m-result") is False
    assert [entry["entry_type"] for entry in result_entries] == ["prose", "prose"]
    assert [cast(dict[str, object], entry["prose"])["text"] for entry in result_entries] == [
        "Operator context beside the protocol row.",
        "Independent envelope context: cached branch state retained.",
    ]


def test_pure_paired_result_row_is_suppressed() -> None:
    """Production dependency: placement suppression is conditional on no independent entry."""

    case_path = CASES_ROOT / "result-before-use.json"
    case = _case(case_path)
    placement = semantic_card_placement_for_messages(
        cast(list[dict[str, object]], case["messages"]),
        session_id=str(case["session_id"]),
        provider_family=str(case["provider_family"]),
    )
    assert placement.is_suppressed("m-result") is True
    assert placement.entries_for("m-result") == []


def test_chatgpt_web_recipient_extracts_structured_query_without_raw_json_title() -> None:
    """Production dependency: explicit ChatGPT web input structure and specialized web builder."""

    case_path = CASES_ROOT / "chatgpt-web.json"
    card = render_case(_case(case_path), case_path=case_path)[0]
    assert card["kind"] == "web"
    assert card["summary"] == "semantic transcript renderer"
    assert any(
        field["label"] == "query" and field["value"] == "semantic transcript renderer"
        for field in cast(list[dict[str, object]], card["fields"])
    )


def test_mcp_identity_precedes_generic_persisted_family() -> None:
    """Production dependency: structural MCP identity has highest registry precedence."""

    kind = card_kind_for_tool(
        provider_family="unknown",
        tool_name="mcp__github__get_issue",
        semantic_type="other",
    )
    assert kind is SemanticCardKind.MCP


def test_message_attachment_envelope_retains_attachment_identity() -> None:
    case_path = CASES_ROOT / "message-attachment-envelope.json"
    entries = render_case_entries(_case(case_path), case_path=case_path)
    card = cast(dict[str, object], entries[1]["card"])
    source = cast(dict[str, object], card["source"])
    assert source["attachment_id"] == "att-7"
    assert source["origin"] == "claude-ai-export"
    assert _field_value(card, "size") == "2048 bytes"


def _field_value(card: dict[str, object], label: str) -> str | None:
    for field in cast(list[dict[str, object]], card["fields"]):
        if field["label"] == label:
            return cast(str, field["value"])
    return None


def test_archive_lineage_descriptor_preserves_exact_composition_authority() -> None:
    """Production dependency: archive envelope facts, not inferred ancestry.

    Mutation control: forcing resolved/root/completeness certainty or dropping
    the branch point changes this exact bounded descriptor.
    """

    descriptor = lineage_descriptor_from_archive_envelope(
        SimpleNamespace(
            session_id="child",
            origin="codex-session",
            parent_session_id="parent",
            root_session_id="root",
            branch_type="fork",
            lineage_complete=True,
            lineage_truncation_reason=None,
            lineage_inheritance="prefix-sharing",
            lineage_branch_point_message_id="parent:p1",
            active_leaf_message_id="child:c2",
        )
    )
    assert descriptor.root_session_id == "root"
    assert descriptor.parent_session_id == "parent"
    assert descriptor.resolved is True
    assert descriptor.lineage_complete is True
    assert descriptor.inherited_prefix is True
    assert descriptor.branch_point_message_id == "parent:p1"
    assert descriptor.availability.value == "complete"


def test_archive_lineage_descriptor_does_not_invent_resolution_when_truncated() -> None:
    descriptor = lineage_descriptor_from_archive_envelope(
        SimpleNamespace(
            session_id="child",
            origin="codex-session",
            parent_session_id="parent",
            root_session_id=None,
            branch_type="fork",
            lineage_complete=False,
            lineage_truncation_reason="depth_limit",
            lineage_inheritance="prefix-sharing",
            lineage_branch_point_message_id="parent:p1",
            active_leaf_message_id=None,
        )
    )
    assert descriptor.root_session_id is None
    assert descriptor.resolved is None
    assert descriptor.availability.value == "partial"
    assert descriptor.lineage_truncation_reason == "depth_limit"
