"""Tests for ChatGPTAssemblySpec — bd polylogue-0hwv / polylogue-dt5s.

Exercises the real ``sources/assembly.py`` protocol wiring end to end: on-disk
sidecar discovery (both extracted-directory and ZIP-bundle shapes) and
attachment enrichment for both ``.dat`` id resolution and sandbox-file tiered
resolution.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.sources.assembly_chatgpt import ChatGPTAssemblySpec
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.sources.parsers.chatgpt_sidecars import ChatGPTAssetIndex


def _session(
    attachments: list[ParsedAttachment],
    *,
    provider_session_id: str = "conv-1",
    source_name: Provider = Provider.CHATGPT,
) -> ParsedSession:
    return ParsedSession(
        source_name=source_name,
        provider_session_id=provider_session_id,
        title="t",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.ASSISTANT, text="hi")],
        attachments=attachments,
    )


class TestDiscoverSidecarsFromDirectory:
    def test_reads_both_sidecar_files(self, tmp_path: Path) -> None:
        (tmp_path / "library_files.json").write_text(
            json.dumps([{"file_id": "file_abc", "file_name": "notes.md"}]), encoding="utf-8"
        )
        (tmp_path / "conversation_asset_file_names.json").write_text(
            json.dumps({"file-xyz.dat": "image.png"}), encoding="utf-8"
        )
        (tmp_path / "conversations-000.json").write_text("[]", encoding="utf-8")

        sidecar_data = ChatGPTAssemblySpec().discover_sidecars(
            [
                tmp_path / "conversations-000.json",
                tmp_path / "library_files.json",
                tmp_path / "conversation_asset_file_names.json",
            ]
        )
        index = sidecar_data["chatgpt_asset_index"]
        assert index.resolve_dat("file_abc") is not None
        assert index.resolve_dat("file-xyz") is not None

    def test_climbs_to_parent_dir_when_only_one_shard_path_given(self, tmp_path: Path) -> None:
        """Single-file daemon catch-up re-parse passes only one shard path."""
        (tmp_path / "library_files.json").write_text(
            json.dumps([{"file_id": "file_abc", "file_name": "notes.md"}]), encoding="utf-8"
        )
        (tmp_path / "conversations-014.json").write_text("[]", encoding="utf-8")

        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([tmp_path / "conversations-014.json"])
        index = sidecar_data["chatgpt_asset_index"]
        assert index.resolve_dat("file_abc") is not None

    def test_no_sidecars_present_returns_empty_index(self, tmp_path: Path) -> None:
        (tmp_path / "conversations-000.json").write_text("[]", encoding="utf-8")
        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([tmp_path / "conversations-000.json"])
        assert sidecar_data["chatgpt_asset_index"].is_empty is True


class TestDiscoverSidecarsFromZip:
    def test_reads_sidecar_members_from_zip(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("library_files.json", json.dumps([{"file_id": "file_abc", "file_name": "notes.md"}]))
            zf.writestr("conversation_asset_file_names.json", json.dumps({"file-xyz.dat": "image.png"}))
            zf.writestr("conversations-000.json", "[]")

        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path])
        index = sidecar_data["chatgpt_asset_index"]
        assert index.resolve_dat("file_abc") is not None
        assert index.resolve_dat("file-xyz") is not None

    def test_zip_missing_sidecars_returns_empty_index(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations-000.json", "[]")

        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path])
        assert sidecar_data["chatgpt_asset_index"].is_empty is True


class TestEnrichSession:
    def _index(self) -> ChatGPTAssetIndex:
        return ChatGPTAssetIndex.build(
            library_files_payload=[
                {
                    "file_id": "file-abc",
                    "file_name": "library-name.png",
                    "mime_type": "image/png",
                    "file_size_bytes": 999,
                    "sha256_digest": "deadbeef",
                },
                {
                    "file_id": "file_lib1",
                    "file_name": "output.csv",
                    "origination_message_id": "m1",
                    "origination_thread_id": "conv-1",
                    "mime_type": "text/csv",
                },
            ],
            asset_file_names_payload={},
        )

    def test_no_op_when_no_attachments(self) -> None:
        conv = _session([])
        spec = ChatGPTAssemblySpec()
        result = spec.enrich_session(conv, {"chatgpt_asset_index": self._index()})
        assert result is conv

    def test_no_op_for_non_chatgpt_provider(self) -> None:
        attachment = ParsedAttachment(provider_attachment_id="file-abc", message_provider_id="m1")
        conv = _session([attachment], source_name=Provider.CLAUDE_AI)
        spec = ChatGPTAssemblySpec()
        result = spec.enrich_session(conv, {"chatgpt_asset_index": self._index()})
        assert result is conv

    def test_no_op_when_index_empty(self) -> None:
        attachment = ParsedAttachment(provider_attachment_id="file-abc", message_provider_id="m1")
        conv = _session([attachment])
        spec = ChatGPTAssemblySpec()
        result = spec.enrich_session(conv, {"chatgpt_asset_index": ChatGPTAssetIndex.empty()})
        assert result is conv

    def test_dat_attachment_resolved_and_event_recorded(self) -> None:
        attachment = ParsedAttachment(provider_attachment_id="file-abc", message_provider_id="m1")
        conv = _session([attachment])
        spec = ChatGPTAssemblySpec()

        result = spec.enrich_session(conv, {"chatgpt_asset_index": self._index()})

        resolved = result.attachments[0]
        assert resolved.name == "library-name.png"
        assert resolved.mime_type == "image/png"
        assert resolved.size_bytes == 999
        assert resolved.provider_file_id == "file-abc"
        events = [e for e in result.session_events if e.event_type == "chatgpt_asset_resolution"]
        assert len(events) == 1
        assert events[0].payload["provider_sha256"] == "deadbeef"
        assert events[0].payload["resolution_source"] == "library_files"

    def test_does_not_overwrite_existing_attachment_fields(self) -> None:
        attachment = ParsedAttachment(
            provider_attachment_id="file-abc",
            message_provider_id="m1",
            name="original-name.png",
            mime_type="image/original",
            size_bytes=1,
        )
        conv = _session([attachment])
        spec = ChatGPTAssemblySpec()

        result = spec.enrich_session(conv, {"chatgpt_asset_index": self._index()})

        # The attachment keeps its own provider-reported fields; only the
        # sidecar-derived event and the still-unset provider_file_id change.
        resolved = result.attachments[0]
        assert resolved.name == "original-name.png"
        assert resolved.mime_type == "image/original"
        assert resolved.size_bytes == 1
        assert resolved.provider_file_id == "file-abc"

    def test_unresolvable_dat_attachment_is_unchanged(self) -> None:
        attachment = ParsedAttachment(provider_attachment_id="file-totally-unknown", message_provider_id="m1")
        conv = _session([attachment])
        spec = ChatGPTAssemblySpec()

        result = spec.enrich_session(conv, {"chatgpt_asset_index": self._index()})

        assert result is conv

    def test_sandbox_attachment_resolves_via_message_id_tier1(self) -> None:
        attachment = ParsedAttachment(
            provider_attachment_id="sandbox:m1:/mnt/data/output.csv",
            message_provider_id="m1",
            name="output.csv",
            attachment_kind="sandbox_file",
            source_url="sandbox:/mnt/data/output.csv",
        )
        conv = _session([attachment], provider_session_id="conv-1")
        spec = ChatGPTAssemblySpec()

        result = spec.enrich_session(conv, {"chatgpt_asset_index": self._index()})

        resolved = result.attachments[0]
        assert resolved.provider_file_id == "file_lib1"
        # attachment_kind stays sandbox_file -- there is still nothing
        # fetchable over HTTP for it, resolution only strengthens identity.
        assert resolved.attachment_kind == "sandbox_file"
        events = [e for e in result.session_events if e.event_type == "chatgpt_sandbox_file_resolution"]
        assert len(events) == 1
        assert events[0].payload["resolution_tier"] == 1

    def test_unresolved_sandbox_attachment_still_records_tier6_event(self) -> None:
        attachment = ParsedAttachment(
            provider_attachment_id="sandbox:m-unrelated:/mnt/data/mystery.bin",
            message_provider_id="m-unrelated",
            name="mystery.bin",
            attachment_kind="sandbox_file",
            source_url="sandbox:/mnt/data/mystery.bin",
        )
        conv = _session([attachment], provider_session_id="conv-unrelated")
        spec = ChatGPTAssemblySpec()

        result = spec.enrich_session(conv, {"chatgpt_asset_index": self._index()})

        resolved = result.attachments[0]
        assert resolved.provider_file_id is None
        assert resolved.attachment_kind == "sandbox_file"
        events = [e for e in result.session_events if e.event_type == "chatgpt_sandbox_file_resolution"]
        assert len(events) == 1
        assert events[0].payload["resolution_tier"] == 6
        assert "resolved_file_id" not in events[0].payload
