"""Original endpoint replies use the ordinary Grok semantic owner."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.hydration import archive_envelope_to_session
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, MaterialOrigin
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.parsers import grok
from polylogue.sources.parsers.base import AdmissionUnit
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


@pytest.fixture
def bundle() -> dict[str, Any]:
    return cast(
        dict[str, Any], json.loads((Path(__file__).parents[3] / "fixtures/grok/native-bundle.json").read_text())
    )


def test_native_structured_only_turns_and_identity_survive_ordinary_parser(bundle: dict[str, Any]) -> None:
    # Restoring the export text guard or synthetic IDs removes these rows.
    session = grok.parse_conversation(bundle, "filename")
    assert session.provider_session_id == "native-conversation"
    assert [message.provider_message_id for message in session.messages] == [
        "human",
        "reasoning",
        "search",
        "attachment",
    ]
    assert [message.parent_message_provider_id for message in session.messages] == [
        None,
        "human",
        "reasoning",
        "search",
    ]
    assert session.messages[1].text is None
    assert session.messages[2].text == ""
    assert session.messages[1].blocks[0].type is BlockType.THINKING
    assert session.messages[1].blocks[0].text == "Check the source\nCompare results"
    results = [block for message in session.messages for block in message.blocks if block.type is BlockType.TOOL_RESULT]
    assert [block.is_error for block in results] == [False, None, None, True]
    assert results[1].outcome_unknown_reason == "not_reported"
    assert cast(dict[str, Any], results[2].metadata)["results"][0]["url"] == "https://example.org/document"
    assert cast(dict[str, Any], results[3].metadata)["raw"]["output"] == {"body": "complete"}
    assert session.active_leaf_message_provider_id == "attachment"
    assert all(message.is_active_path for message in session.messages)
    assert session.updated_at == "2026-01-01T00:00:04+00:00"
    assert session.attachments[0].provider_attachment_id == "file-1"
    assert session.attachments[0].message_provider_id == "attachment"
    assert session.attachments[0].source_url == "users/synthetic/file-1/content"
    assert session.attachments[0].size_bytes == 12
    assert session.attachments[1].source_url == "https://example.org/image"
    assert session.session_events[-1].payload["reply"] == bundle["response_nodes"]
    assert session.session_events[0].payload["partial"] is True


def test_native_direct_nested_replies_and_replay_have_same_semantics(bundle: dict[str, Any]) -> None:
    nested = deepcopy(bundle)
    nested["conversation"] = {"conversation": nested["conversation"]}
    nested["responses"] = nested["responses"]["responses"]
    assert grok.looks_like_native_bundle(nested)
    ordinary = grok.parse_conversation(bundle, "first")
    replay = grok.parse_native_bundle(nested, "second")[0]
    assert session_revision_projection(ordinary) == session_revision_projection(replay)


def test_native_fork_keeps_parent_edges_without_guessing_selected_leaf(bundle: dict[str, Any]) -> None:
    bundle["responses"]["responses"].append(
        {
            "responseId": "alternate",
            "createTime": "2026-01-01T00:00:05Z",
            "parentResponseId": "human",
            "sender": "assistant",
            "steps": [{"text": ["Alternative"]}],
        }
    )
    session = grok.parse_native_bundle(bundle, "filename")[0]
    assert session.messages[-1].parent_message_provider_id == "human"
    assert session.active_leaf_message_provider_id is None
    assert all(message.is_active_path is None for message in session.messages)


def test_native_absent_and_repeated_ids_keep_private_asset_owner_coordinates(bundle: dict[str, Any]) -> None:
    response = bundle["responses"]["responses"][-1]
    bundle["responses"]["responses"] = [deepcopy(response), deepcopy(response)]
    session = grok.parse_native_bundle(bundle, "filename")[0]
    assert [message.variant_index for message in session.messages] == [0, 1]
    assert [
        asset.owner_coordinate.physical_key
        for asset in session.attachments
        if asset.provider_attachment_id == "file-1" and asset.owner_coordinate is not None
    ] == [(0, 0), (1, 1)]
    del bundle["responses"]["responses"][0]["responseId"]
    session = grok.parse_native_bundle(bundle, "filename")[0]
    assert session.messages[0].provider_message_id == ""
    assert session.attachments[0].message_provider_id is None
    assert session.attachments[0].owner_coordinate is not None
    assert session.attachments[0].owner_coordinate.physical_key == (0, 0)


def test_native_unknown_structures_retain_raw_evidence(bundle: dict[str, Any]) -> None:
    bundle["responses"]["responses"] = [
        {
            "responseId": "future",
            "sender": "assistant",
            "steps": [{"future": "step"}],
            "toolResponses": [{"future": "tool"}],
            "imageAttachments": [{"future": "image"}],
        }
    ]
    session = grok.parse_native_bundle(bundle, "filename")[0]
    assert len(session.messages) == 1
    assert [cast(dict[str, Any], block.metadata)["raw"] for block in session.messages[0].blocks] == [
        {"future": "step"},
        {"future": "tool"},
        [{"future": "image"}],
    ]


def test_native_predicate_does_not_claim_idless_account_export() -> None:
    assert not grok.looks_like_native_bundle({"conversation": {"title": "Export"}, "responses": []})
    assert not grok.looks_like_native_bundle({"conversation": {"conversationId": "native"}, "responses": {}})


def test_native_asset_metadata_only_and_malformed_records_are_visible(bundle: dict[str, Any]) -> None:
    bundle["responses"]["responses"] = [
        17,
        {
            "responseId": "asset-only",
            "sender": "assistant",
            "fileAttachmentAssetMetadata": [
                {"assetId": "asset", "name": "asset.txt", "key": "users/synthetic/asset/content"}
            ],
        },
    ]
    session = grok.parse_native_bundle(bundle, "filename")[0]
    assert [message.provider_message_id for message in session.messages] == ["asset-only"]
    assert session.attachments[0].provider_attachment_id == "asset"
    assert session.attachments[0].source_url == "users/synthetic/asset/content"
    assert session.unit_accounting is not None
    assert session.unit_accounting.expected[AdmissionUnit.MESSAGE] == 2
    assert session.session_events[0].event_type == "grok_response_refusal"
    session.unit_accounting.assert_conserved()


def test_native_result_unsupported_outcome_is_not_reported_as_absence(
    bundle: dict[str, Any], workspace_env: dict[str, Path]
) -> None:
    bundle["responses"]["responses"] = [
        {"responseId": "tool", "sender": "assistant", "toolResponses": [{"isError": "future-status"}]}
    ]
    session = grok.parse_native_bundle(bundle, "filename")[0]
    assert session.messages[0].blocks[0].is_error is None
    assert session.messages[0].blocks[0].outcome_unknown_reason == "unsupported_construct"
    with ArchiveStore(workspace_env["archive_root"]) as archive:
        _, stored_id = archive.write_raw_and_parsed(
            session,
            payload=json.dumps(bundle).encode(),
            source_path="/example/grok-native.json",
            acquired_at_ms=1735689600000,
        )
        hydrated = archive.read_session(stored_id)
        assert hydrated.messages[0].blocks[0].tool_outcome == "unknown"
        assert hydrated.messages[0].blocks[0].tool_result_outcome_unknown_reason == "unsupported_construct"


def test_native_missing_message_id_does_not_pair_tools_by_ordinal(bundle: dict[str, Any]) -> None:
    bundle["responses"]["responses"] = [
        {"sender": "assistant", "toolResponses": [{"toolName": "reader", "text": "Output"}]}
    ]
    session = grok.parse_native_bundle(bundle, "filename")[0]
    assert session.messages[0].provider_message_id == ""
    assert all(block.tool_id is None for block in session.messages[0].blocks)


@pytest.mark.parametrize("sender", ["human", "user"])
def test_native_human_sender_authorship_survives_persisted_hydration(
    bundle: dict[str, Any], workspace_env: dict[str, Path], sender: str
) -> None:
    archive_root = workspace_env["archive_root"]
    response = bundle["responses"]["responses"][0]
    response["sender"] = sender
    response["message"] = "# AGENTS.md instructions for a sample project\nPlease explain this document."
    session = grok.parse_native_bundle(bundle, "human-context")[0]
    assert session.messages[0].material_origin is MaterialOrigin.HUMAN_AUTHORED
    assert session.messages[0].message_type is MessageType.CONTEXT
    with ArchiveStore(archive_root) as archive:
        _, stored_id = archive.write_raw_and_parsed(
            session,
            payload=json.dumps(bundle).encode(),
            source_path="/example/grok-native.json",
            acquired_at_ms=1735689600000,
        )
        hydrated = archive_envelope_to_session(archive.read_session(stored_id))
        message = next(iter(hydrated.messages))
        assert message.material_origin is MaterialOrigin.HUMAN_AUTHORED
        assert message.message_type is MessageType.CONTEXT
        assert message.is_human_authored
        summary = archive.read_summary(stored_id)
        assert summary.authored_user_message_count == 1
        assert summary.authored_user_word_count > 0


def test_native_human_sender_does_not_reclassify_structured_tool_output(bundle: dict[str, Any]) -> None:
    bundle["responses"]["responses"] = [
        {
            "responseId": "tool-output",
            "sender": "human",
            "toolResponses": [{"text": "# AGENTS.md instructions for a sample project", "is_error": False}],
        }
    ]
    message = grok.parse_native_bundle(bundle, "structured")[0].messages[0]
    assert message.message_type is MessageType.TOOL_RESULT
    assert message.material_origin is MaterialOrigin.TOOL_RESULT
    assert message.blocks[0].is_error is False
