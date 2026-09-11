"""Tests for ChatGPTAssemblySpec — bd polylogue-0hwv / polylogue-dt5s.

Exercises the real ``sources/assembly.py`` protocol wiring end to end: on-disk
sidecar discovery (both extracted-directory and ZIP-bundle shapes) and
attachment enrichment for both asset-member id resolution and sandbox-file
tiered resolution.
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


class TestAcquireAssetBlobsFromZip:
    """bd polylogue-8ac0 / polylogue-1nd1s — streaming asset bytes into the blob store."""

    def test_dat_member_streamed_into_blob_store(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "export.zip"
        dat_bytes = b"the real attachment bytes"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversation_asset_file_names.json", json.dumps({"file-xyz.dat": "photo.png"}))
            zf.writestr("file-xyz.dat", dat_bytes)

        store = BlobStore(tmp_path / "blobs")
        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path], blob_store=store)

        asset_blobs = sidecar_data.get("chatgpt_asset_blobs")
        assert asset_blobs is not None
        assert "file-xyz" in asset_blobs
        blob_hash, size = asset_blobs["file-xyz"]
        assert size == len(dat_bytes)
        assert store.exists(blob_hash)
        assert store.read_all(blob_hash) == dat_bytes

    def test_no_blob_store_leaves_asset_blobs_absent(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file-xyz.dat", b"bytes")

        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path])
        assert "chatgpt_asset_blobs" not in sidecar_data

    def test_non_dat_members_are_not_streamed(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", "[]")

        store = BlobStore(tmp_path / "blobs")
        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path], blob_store=store)
        assert "chatgpt_asset_blobs" not in sidecar_data

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
        asset_blobs = sidecar_data.get("chatgpt_asset_blobs")
        assert asset_blobs is not None
        assert set(asset_blobs) == {"file-first"}

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
        assert "chatgpt_asset_blobs" not in sidecar_data

    def test_extension_carrying_members_are_acquired(self, tmp_path: Path) -> None:
        """bd polylogue-1nd1s: an export that names assets by real extension.

        The 2026-04-23 export ships its assets as ``.png``/``.webp``/``.jpg``
        and extension-less members under per-conversation subdirectories
        instead of as flat ``file-<id>.dat``. Admitting only JSON and ``.dat``
        filtered every one of them out before any reader saw it, leaving the
        conversations' asset pointers with no bytes to bind to. Red if
        admission goes back to selecting members by suffix.
        """
        zip_path = tmp_path / "export.zip"
        # member name -> (bytes, the asset id the name carries)
        members = {
            "conv-1/image/file_0000000000ac6243a75c01ca3ff57b84-c5e08f86.png": (
                b"screenshot bytes",
                "file_0000000000ac6243a75c01ca3ff57b84",
            ),
            "dalle-generations/file-0nRBfsDLFpRdQVKa9WffrQCE-3f165a39.webp": (
                b"generated image",
                "file-0nRBfsDLFpRdQVKa9WffrQCE",
            ),
            "file-05HzDzfH11W85mpcVKafDihK-1f3bd27e-ce02": (
                b"no extension at all",
                "file-05HzDzfH11W85mpcVKafDihK",
            ),
            "03adfe6a4b1e5a0#file_00000000cc4c7243aa6bdd0537ca804e#p_0.jpg-p_0.jpg": (
                b"page render",
                "file_00000000cc4c7243aa6bdd0537ca804e",
            ),
            "conv-1/audio/file_66f6b1408b20203c-e7b106a6.wav": (
                b"voice note",
                "file_66f6b1408b20203c",
            ),
            "file_000000007e9471f48ae216a0f081438b-issues-open.json": (
                b'{"real": "asset"}',
                "file_000000007e9471f48ae216a0f081438b",
            ),
        }
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations-000.json", "[]")
            zf.writestr("chat.html", b"<html>the whole transcript render</html>")
            for name, (payload, _asset_id) in members.items():
                zf.writestr(name, payload)

        store = BlobStore(tmp_path / "blobs")
        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path], blob_store=store)

        asset_blobs = sidecar_data.get("chatgpt_asset_blobs")
        assert asset_blobs is not None
        assert set(asset_blobs) == {asset_id for _payload, asset_id in members.values()}
        for payload, asset_id in members.values():
            blob_hash, size = asset_blobs[asset_id]
            assert size == len(payload)
            assert store.read_all(blob_hash) == payload

    def test_non_asset_members_are_never_acquired(self, tmp_path: Path) -> None:
        """The transcript render and the sidecars are not assets.

        ``conversation_asset_file_names.json`` contains the substring
        ``file_names``; a substring match would stream the lookup table into
        the blob store as if it were an asset. Red if the id match stops being
        anchored.
        """
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations-000.json", "[]")
            zf.writestr("chat.html", b"<html>render</html>")
            zf.writestr("conversation_asset_file_names.json", json.dumps({"file-xyz.dat": "photo.png"}))
            zf.writestr("library_files.json", json.dumps([]))
            zf.writestr("user.json", "{}")

        store = BlobStore(tmp_path / "blobs")
        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path], blob_store=store)

        assert "chatgpt_asset_blobs" not in sidecar_data


class TestAcquireAssetBlobsFromDirectory:
    def test_dat_sibling_streamed_into_blob_store(self, tmp_path: Path) -> None:
        (tmp_path / "conversations-000.json").write_text("[]", encoding="utf-8")
        dat_bytes = b"library file bytes"
        (tmp_path / "file_abc.dat").write_bytes(dat_bytes)

        store = BlobStore(tmp_path / "blobs")
        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([tmp_path / "conversations-000.json"], blob_store=store)

        asset_blobs = sidecar_data.get("chatgpt_asset_blobs")
        assert asset_blobs is not None
        blob_hash, size = asset_blobs["file_abc"]
        assert size == len(dat_bytes)
        assert store.read_all(blob_hash) == dat_bytes

    def test_extension_carrying_assets_in_subdirectories_are_acquired(self, tmp_path: Path) -> None:
        """bd polylogue-1nd1s: an extracted export nests assets one level down.

        The extension-carrying export shape puts a conversation's assets under
        ``<conversation>/image/`` and ``<conversation>/audio/``, so a
        non-recursive sibling glob reaches none of them. Red if the directory
        walk stops descending.
        """
        (tmp_path / "conversations-000.json").write_text("[]", encoding="utf-8")
        image_dir = tmp_path / "conv-1" / "image"
        image_dir.mkdir(parents=True)
        image_bytes = b"nested screenshot bytes"
        (image_dir / "file_0000000000ac6243a75c01ca3ff57b84-c5e08f86.png").write_bytes(image_bytes)

        store = BlobStore(tmp_path / "blobs")
        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([tmp_path / "conversations-000.json"], blob_store=store)

        asset_blobs = sidecar_data.get("chatgpt_asset_blobs")
        assert asset_blobs is not None
        blob_hash, size = asset_blobs["file_0000000000ac6243a75c01ca3ff57b84"]
        assert size == len(image_bytes)
        assert store.read_all(blob_hash) == image_bytes


class TestEnrichSessionAcquiresBlobs:
    def test_dat_attachment_gets_precomputed_blob(self) -> None:
        attachment = ParsedAttachment(provider_attachment_id="file-xyz", message_provider_id="m1")
        conv = _session([attachment])
        spec = ChatGPTAssemblySpec()

        result = spec.enrich_session(
            conv,
            {
                "chatgpt_asset_index": ChatGPTAssetIndex.empty(),
                "chatgpt_asset_blobs": {"file-xyz": ("ab" * 32, 5)},
            },
        )

        resolved = result.attachments[0]
        assert resolved.precomputed_blob == ("ab" * 32, 5)
        assert resolved.size_bytes == 5
        events = [e for e in result.session_events if e.event_type == "chatgpt_asset_resolution"]
        assert events == []  # no library_files/asset_names hit, only the blob join

    def test_screenshot_pointer_attachment_joins_its_blob(self) -> None:
        """A ``sediment://`` screenshot row binds the blob named by its bare id.

        This is the join the computer-use screenshots exist for: acquired
        asset bytes are keyed by the bare file id, the attachment carries the
        full pointer. Red if the pointer's scheme is not stripped on the way
        into the blob lookup.
        """
        attachment = ParsedAttachment(
            provider_attachment_id="sediment://file_shot1",
            message_provider_id="m1",
            attachment_kind="computer_screenshot",
        )
        conv = _session([attachment])
        spec = ChatGPTAssemblySpec()

        result = spec.enrich_session(
            conv,
            {
                "chatgpt_asset_index": ChatGPTAssetIndex.empty(),
                "chatgpt_asset_blobs": {"file_shot1": ("ab" * 32, 27100)},
            },
        )

        assert result.attachments[0].precomputed_blob == ("ab" * 32, 27100)

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
                "chatgpt_asset_blobs": {"file-xyz": ("cd" * 32, 5)},
            },
        )

        resolved = result.attachments[0]
        assert resolved.precomputed_blob is None
        assert resolved.inline_bytes == b"already carrying real bytes"

    def test_sandbox_attachment_never_joins_asset_blobs(self) -> None:
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
                "chatgpt_asset_blobs": {"mystery.bin": ("ef" * 32, 5)},
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
        assert "file-xyz" in sidecar_data.get("chatgpt_asset_blobs", {})

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
        blob_hash, size = sidecar_data["chatgpt_asset_blobs"]["file-xyz"]
        assert resolved.precomputed_blob == (blob_hash, size)
        assert resolved.name == "photo.png"

    def test_extension_carrying_asset_binds_to_its_pointer_attachment(self, tmp_path: Path) -> None:
        """bd polylogue-1nd1s + polylogue-91kys, end to end through real ingest.

        An asset named by its real extension is acquired, the content part
        that points at it produces the attachment row, and the two join on the
        bare file id. Red if either half regresses: an unadmitted member
        leaves ``precomputed_blob`` unset, and a pointer that reaches storage
        as block metadata alone leaves no attachment for the bytes to bind to.
        """
        from polylogue.sources.source_parsing import parse_one_source_path

        pointer = "sediment://file_0000000000ac6243a75c01ca3ff57b84"
        conversation = {
            "id": "conv-1",
            "conversation_id": "conv-1",
            "title": "generated image",
            "create_time": 1704067200.0,
            "current_node": "a1",
            "mapping": {
                "u1": {
                    "id": "u1",
                    "message": {
                        "id": "u1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["draw something"]},
                        "create_time": 1704067200.0,
                    },
                    "parent": None,
                    "children": ["a1"],
                },
                "a1": {
                    "id": "a1",
                    "message": {
                        "id": "a1",
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "multimodal_text",
                            "parts": [
                                {
                                    "content_type": "image_asset_pointer",
                                    "asset_pointer": pointer,
                                    "size_bytes": 16,
                                    "width": 1024,
                                    "height": 768,
                                }
                            ],
                        },
                        "create_time": 1704067201.0,
                    },
                    "parent": "u1",
                    "children": [],
                },
            },
        }

        zip_path = tmp_path / "export.zip"
        image_bytes = b"the real image!!"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations-000.json", json.dumps([conversation]))
            zf.writestr(
                "conv-1/image/file_0000000000ac6243a75c01ca3ff57b84-c5e08f86.png",
                image_bytes,
            )

        blob_root = tmp_path / "blobs"
        store = BlobStore(blob_root)
        sidecar_data = ChatGPTAssemblySpec().discover_sidecars([zip_path], blob_store=store)

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
        attachment = next(a for a in session.attachments if a.provider_attachment_id == pointer)
        assert attachment.provider_file_id == "file_0000000000ac6243a75c01ca3ff57b84"
        blob_hash, size = sidecar_data["chatgpt_asset_blobs"]["file_0000000000ac6243a75c01ca3ff57b84"]
        assert attachment.precomputed_blob == (blob_hash, size)
        assert store.read_all(blob_hash) == image_bytes


# ---------------------------------------------------------------------------
# polylogue-ximhz: the asset maps must be rebuildable from retained bytes,
# scoped to the export they were acquired from.
#
# Anti-vacuity: make ``chatgpt_export_scope`` return a constant and
# ``test_two_exports_are_two_scopes`` goes red; the full production-route
# proofs (attachment name, payload and no cross-binding with every original
# file deleted) are in
# ``tests/unit/pipeline/test_ingest_worker_assembly.py``.
# ---------------------------------------------------------------------------


def test_two_exports_are_two_scopes() -> None:
    """One export's retained maps never address another export's members."""
    from polylogue.sources.retained_assembly import chatgpt_export_scope

    zip_scope = chatgpt_export_scope("/archive/exports/first.zip:conversations.json")
    other_zip_scope = chatgpt_export_scope("/archive/exports/second.zip:conversations.json")
    dir_scope = chatgpt_export_scope("/archive/exports/extracted/conversations.json")

    assert zip_scope == "/archive/exports/first.zip:"
    assert other_zip_scope == "/archive/exports/second.zip:"
    assert dir_scope == "/archive/exports/extracted/"
    assert len({zip_scope, other_zip_scope, dir_scope}) == 3


def test_retained_asset_member_names_still_carry_their_provider_id() -> None:
    """The retained member coordinate is what the attachment join resolves."""
    from polylogue.sources.assembly_chatgpt import _member_asset_id

    assert _member_asset_id("file-ABCdef123.dat") == "file-ABCdef123"
    assert _member_asset_id("file-ABCdef123.png") == "file-ABCdef123"
    assert _member_asset_id("conversations.json") is None


def test_zip_export_retains_asset_members_and_maps_byte_exact(tmp_path: Path) -> None:
    """A bundled export's asset bytes are acquired, not decoded as a payload.

    A ChatGPT export normally arrives as a ZIP. Its asset members are
    arbitrary binary, so the payload-splitting route's UTF-8 decode fails the
    whole archive read rather than the one member -- the raw-only declaration
    has to divert them to byte-preserving acquisition
    (``source_acquisition_components.iter_zip_entry_raw_data``).

    Anti-vacuity: remove the ``path_declaration_refuses_session`` branch there
    and this test fails with the asset member missing and the archive skipped.
    """
    import zipfile

    from polylogue.config import Source
    from polylogue.sources.source_acquisition import iter_source_raw_data
    from polylogue.storage.blob_store import BlobStore

    root = tmp_path / "inbox"
    root.mkdir()
    asset_bytes = b"\x89PNG\r\n\x1a\nsynthetic"
    archive = root / "chatgpt-export.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("conversations.json", json.dumps([]))
        handle.writestr("library_files.json", json.dumps([{"id": "file-ABC", "name": "d.png"}]))
        handle.writestr("conversation_asset_file_names.json", json.dumps({"file-ABC": "d.png"}))
        handle.writestr("dalle-generations/file-ABC.webp", asset_bytes)

    store = BlobStore(tmp_path / "blobs")
    acquired = {
        raw.source_path.rsplit(":", 1)[-1]: raw
        for raw in iter_source_raw_data(Source(name="chatgpt", path=root), blob_store=store)
    }

    assert "dalle-generations/file-ABC.webp" in acquired
    assert "library_files.json" in acquired
    assert "conversation_asset_file_names.json" in acquired
    asset = acquired["dalle-generations/file-ABC.webp"]
    assert store.read_all(asset.blob_hash or "") == asset_bytes
