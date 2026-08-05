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

import pytest

from polylogue.archive import zip_admission as zip_admission_module
from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.sources.assembly_chatgpt import ChatGPTAssemblySpec
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.sources.parsers.chatgpt_sidecars import ChatGPTAssetIndex
from polylogue.storage.blob_store import BlobStore
from tests.infra.source_builders import ChatGPTExportBuilder


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

    def test_duplicate_sidecar_name_does_not_replace_first_admitted_member(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "duplicate-sidecar.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("library_files.json", json.dumps([{"file_id": "file-first", "file_name": "first.md"}]))
            zf.writestr("library_files.json", json.dumps([{"file_id": "file-second", "file_name": "second.md"}]))

        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path])
        index = sidecar_data["chatgpt_asset_index"]

        assert index.resolve_dat("file-first") is not None
        assert index.resolve_dat("file-second") is None

    @pytest.mark.parametrize("limit_name", ["MAX_UNCOMPRESSED_SIZE", "MAX_COMPRESSION_RATIO"])
    def test_rejects_json_sidecar_before_open_for_size_and_ratio_limits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        limit_name: str,
    ) -> None:
        zip_path = tmp_path / f"rejected-{limit_name}.zip"
        sidecar_bytes = b'{"file_id":"file-abc","file_name":"notes.md"}' + (b" " * 2048)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("library_files.json", sidecar_bytes)

        monkeypatch.setattr(zip_admission_module, limit_name, 1)
        opened: list[object] = []

        def fail_if_open(_archive: zipfile.ZipFile, member: object, *args: object, **kwargs: object) -> object:
            opened.append(member)
            raise AssertionError("rejected JSON sidecar must not be opened")

        monkeypatch.setattr(zipfile.ZipFile, "open", fail_if_open)

        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path])

        assert opened == []
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


class TestAcquireDatBlobsFromZip:
    """bd polylogue-8ac0 — streaming ``.dat`` asset bytes into the blob store."""

    def test_dat_member_streamed_into_blob_store(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "export.zip"
        dat_bytes = b"the real attachment bytes"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversation_asset_file_names.json", json.dumps({"file-xyz.dat": "photo.png"}))
            zf.writestr("file-xyz.dat", dat_bytes)

        store = BlobStore(tmp_path / "blobs")
        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path], blob_store=store)

        dat_blobs = sidecar_data.get("chatgpt_dat_blobs")
        assert dat_blobs is not None
        assert "file-xyz" in dat_blobs
        blob_hash, size = dat_blobs["file-xyz"]
        assert size == len(dat_bytes)
        assert store.exists(blob_hash)
        assert store.read_all(blob_hash) == dat_bytes

    def test_no_blob_store_leaves_dat_blobs_absent(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file-xyz.dat", b"bytes")

        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path])
        assert "chatgpt_dat_blobs" not in sidecar_data

    def test_non_dat_members_are_not_streamed(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", "[]")

        store = BlobStore(tmp_path / "blobs")
        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path], blob_store=store)
        assert "chatgpt_dat_blobs" not in sidecar_data

    def test_many_dat_members_obey_aggregate_limit_before_second_read(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        zip_path = tmp_path / "aggregate-dat.zip"
        first_bytes = b"first attachment"
        second_bytes = b"second attachment"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file-first.dat", first_bytes)
            zf.writestr("file-second.dat", second_bytes)

        monkeypatch.setattr(zip_admission_module, "MAX_AGGREGATE_UNCOMPRESSED_SIZE", len(first_bytes))
        original_open = zipfile.ZipFile.open
        opened: list[str] = []

        def track_open(
            archive: zipfile.ZipFile,
            member: str | zipfile.ZipInfo,
        ) -> object:
            info = member if isinstance(member, zipfile.ZipInfo) else archive.getinfo(member)
            opened.append(info.filename)
            return original_open(archive, member)

        monkeypatch.setattr(zipfile.ZipFile, "open", track_open)
        store = BlobStore(tmp_path / "blobs")

        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path], blob_store=store)

        assert opened == ["file-first.dat"]
        dat_blobs = sidecar_data.get("chatgpt_dat_blobs")
        assert dat_blobs is not None
        assert set(dat_blobs) == {"file-first"}

    def test_json_and_dat_members_share_aggregate_limit_before_dat_read(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        zip_path = tmp_path / "aggregate-cross-type.zip"
        json_bytes = b"[]"
        dat_bytes = b"attachment"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("library_files.json", json_bytes)
            zf.writestr("file-xyz.dat", dat_bytes)

        monkeypatch.setattr(zip_admission_module, "MAX_AGGREGATE_UNCOMPRESSED_SIZE", len(json_bytes))
        original_open = zipfile.ZipFile.open
        opened: list[str] = []

        def track_open(
            archive: zipfile.ZipFile,
            member: str | zipfile.ZipInfo,
        ) -> object:
            info = member if isinstance(member, zipfile.ZipInfo) else archive.getinfo(member)
            opened.append(info.filename)
            return original_open(archive, member)

        monkeypatch.setattr(zipfile.ZipFile, "open", track_open)
        store = BlobStore(tmp_path / "blobs")

        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path], blob_store=store)

        assert opened == ["library_files.json"]
        assert "chatgpt_dat_blobs" not in sidecar_data


class TestAcquireDatBlobsFromDirectory:
    def test_dat_sibling_streamed_into_blob_store(self, tmp_path: Path) -> None:
        (tmp_path / "conversations-000.json").write_text("[]", encoding="utf-8")
        dat_bytes = b"library file bytes"
        (tmp_path / "file_abc.dat").write_bytes(dat_bytes)

        store = BlobStore(tmp_path / "blobs")
        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([tmp_path / "conversations-000.json"], blob_store=store)

        dat_blobs = sidecar_data.get("chatgpt_dat_blobs")
        assert dat_blobs is not None
        blob_hash, size = dat_blobs["file_abc"]
        assert size == len(dat_bytes)
        assert store.read_all(blob_hash) == dat_bytes


class TestEnrichSessionAcquiresBlobs:
    def test_dat_attachment_gets_precomputed_blob(self) -> None:
        attachment = ParsedAttachment(provider_attachment_id="file-xyz", message_provider_id="m1")
        conv = _session([attachment])
        spec = ChatGPTAssemblySpec()

        result = spec.enrich_session(
            conv,
            {
                "chatgpt_asset_index": ChatGPTAssetIndex.empty(),
                "chatgpt_dat_blobs": {"file-xyz": ("ab" * 32, 5)},
            },
        )

        resolved = result.attachments[0]
        assert resolved.precomputed_blob == ("ab" * 32, 5)
        assert resolved.size_bytes == 5
        events = [e for e in result.session_events if e.event_type == "chatgpt_asset_resolution"]
        assert events == []  # no library_files/asset_names hit, only the blob join

    def test_does_not_overwrite_inline_bytes(self) -> None:
        attachment = ParsedAttachment(
            provider_attachment_id="file-xyz",
            message_provider_id="m1",
            inline_bytes=b"already carrying real bytes",
        )
        conv = _session([attachment])
        spec = ChatGPTAssemblySpec()

        result = spec.enrich_session(
            conv,
            {
                "chatgpt_asset_index": ChatGPTAssetIndex.empty(),
                "chatgpt_dat_blobs": {"file-xyz": ("cd" * 32, 5)},
            },
        )

        resolved = result.attachments[0]
        assert resolved.precomputed_blob is None
        assert resolved.inline_bytes == b"already carrying real bytes"

    def test_sandbox_attachment_never_joins_dat_blobs(self) -> None:
        attachment = ParsedAttachment(
            provider_attachment_id="sandbox:m-unrelated:/mnt/data/mystery.bin",
            message_provider_id="m-unrelated",
            name="mystery.bin",
            attachment_kind="sandbox_file",
            source_url="sandbox:/mnt/data/mystery.bin",
        )
        conv = _session([attachment], provider_session_id="conv-unrelated")
        spec = ChatGPTAssemblySpec()

        result = spec.enrich_session(
            conv,
            {
                "chatgpt_asset_index": ChatGPTAssetIndex.empty(),
                "chatgpt_dat_blobs": {"mystery.bin": ("ef" * 32, 5)},
            },
        )

        assert result.attachments[0].precomputed_blob is None


class TestZipBundleEndToEndBlobAcquisition:
    """Regression coverage for the ZIP-bundle sidecar-enrichment wiring gap.

    ``process_zip`` used to hardcode ``sidecar_data={}`` for every entry's
    ``_ParseContext``, so ``enrich_session`` never actually ran for
    ZIP-shaped ChatGPT sources (the common shape for a GDPR/Takeout export) --
    only unit tests exercising ``discover_sidecars``/``enrich_session`` in
    isolation existed, none of them through the real ``parse_one_source_path``
    entry point. This drives a synthetic ZIP export all the way through.
    """

    def test_dat_bytes_acquired_through_real_zip_ingest_path(self, tmp_path: Path) -> None:
        from polylogue.sources.source_parsing import parse_one_source_path

        conversation = (
            ChatGPTExportBuilder("conv-1")
            .title("t")
            .add_node(
                "user",
                "hello",
                metadata={
                    "attachments": [
                        {"id": "file-xyz", "name": "photo.png", "mime_type": "image/png", "size": 5},
                    ]
                },
            )
            .build()
        )

        zip_path = tmp_path / "export.zip"
        dat_bytes = b"the real bytes"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", json.dumps([conversation]))
            zf.writestr("conversation_asset_file_names.json", json.dumps({"file-xyz.dat": "photo.png"}))
            zf.writestr("file-xyz.dat", dat_bytes)

        blob_root = tmp_path / "blobs"
        store = BlobStore(blob_root)
        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path], blob_store=store)
        assert "file-xyz" in sidecar_data.get("chatgpt_dat_blobs", {})

        results = list(
            parse_one_source_path(
                str(zip_path),
                file_mtime=None,
                source_name="chatgpt",
                sidecar_data=sidecar_data,
                capture_raw=False,
                blob_root=blob_root,
                blob_store=store,
            )
        )
        assert len(results) == 1
        _, session = results[0]
        assert len(session.attachments) == 1
        resolved = session.attachments[0]
        blob_hash, size = sidecar_data["chatgpt_dat_blobs"]["file-xyz"]
        assert resolved.precomputed_blob == (blob_hash, size)
        assert resolved.name == "photo.png"
