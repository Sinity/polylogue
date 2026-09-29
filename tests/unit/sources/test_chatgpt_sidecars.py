"""Tests for ChatGPT export sidecar resolvers (bd polylogue-0hwv / polylogue-dt5s).

Fixture shapes and tier counts mirror the measured spec recorded on the two
beads (2026-07-31, against the real 2026-07-29 export): two id namespaces
(``file-<b64ish>`` conversation assets, ``file_<32hex>`` library files), a
six-tier sandbox-file resolver where tier 5 (ambiguous) is possible but tier
1-4 dominate, and library_files preferred over conversation_asset_file_names
whenever both name the same id.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.sources.assembly import SidecarData
from polylogue.sources.assembly_chatgpt import ChatGPTAssemblySpec
from polylogue.sources.parsers.base import ParsedAttachment, ParsedSession, ParsedSessionEvent
from polylogue.sources.parsers.chatgpt_sidecars import (
    ChatGPTAssetIndex,
    parse_asset_file_names,
    parse_library_files,
)
from polylogue.sources.prepared_message_sink import SqliteMessageStore


def test_sidecar_enrichment_updates_prepared_rows_without_collecting(tmp_path: Path) -> None:
    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        attachments = store.new_attachment_sink()
        events = store.new_event_sink()
        attachments.extend(
            ParsedAttachment(provider_attachment_id=f"file-{index}", message_provider_id=f"message-{index}")
            for index in range(3)
        )
        session = ParsedSession(source_name=Provider.CHATGPT, provider_session_id="conversation", messages=[])
        session = session.model_copy(update={"attachments": attachments, "session_events": events})
        index = ChatGPTAssetIndex.build(
            library_files_payload=[],
            asset_file_names_payload={f"file-{index}.dat": f"asset-{index}.png" for index in range(3)},
        )
        returned = ChatGPTAssemblySpec().enrich_session(session, {"chatgpt_asset_index": index})
        assert returned is session
        assert [attachment.name for attachment in attachments] == [f"asset-{index}.png" for index in range(3)]
        assert [event.event_type for event in events] == ["chatgpt_asset_resolution"] * 3
    finally:
        store.close()


_RENDITION_ID = "file_00000000cc4c7243aa6bdd0537ca804e"
_RENDITION_BLOBS = {
    f"{_RENDITION_ID}#03adfe6a4b1e5a0#{_RENDITION_ID}#p_0.jpg-p_0.jpg": ("a0" * 32, 11),
    f"{_RENDITION_ID}#03adfe6a4b1e5a0#{_RENDITION_ID}#p_1.jpg-p_1.jpg": ("a1" * 32, 18),
}


def _assert_every_rendition_is_archived(attachments: list[ParsedAttachment]) -> None:
    blobs = [attachment.precomputed_blob for attachment in attachments]
    assert all(blob is not None for blob in blobs)
    assert sorted(blob for blob in blobs if blob is not None) == sorted(_RENDITION_BLOBS.values())
    assert len({attachment.provider_attachment_id for attachment in attachments}) == 2
    assert len({attachment.name for attachment in attachments}) == 2
    assert {attachment.mime_type for attachment in attachments} == {"image/jpeg"}
    assert {attachment.provider_file_id for attachment in attachments} == {_RENDITION_ID}


def test_every_duplicate_asset_rendition_becomes_its_own_attachment() -> None:
    """Two members normalizing to one asset id both receive an attachment and blob.

    Anti-vacuity: binding only the lexically first member to the pointer
    leaves ``p_1``'s blob with no attachment reference.
    """
    pointer = ParsedAttachment(provider_attachment_id=f"sediment://{_RENDITION_ID}", message_provider_id="m1")
    session = ParsedSession(
        source_name=Provider.CHATGPT, provider_session_id="conversation", messages=[], attachments=[pointer]
    )

    sidecars: SidecarData = {"chatgpt_asset_index": ChatGPTAssetIndex.empty(), "chatgpt_asset_blobs": _RENDITION_BLOBS}
    returned = ChatGPTAssemblySpec().enrich_session(session, sidecars)

    _assert_every_rendition_is_archived(list(returned.attachments))
    members = sorted(str(event.payload["member_name"]) for event in returned.session_events)
    assert members == sorted(key.split("#", 1)[1] for key in _RENDITION_BLOBS)


def test_renditions_keep_a_provider_file_id_the_pointer_already_carried() -> None:
    """A provider-native file id on the pointer survives rendition expansion.

    Anti-vacuity: overwrite ``provider_file_id`` with the member's normalized
    asset id and every rendition reports ``file_000...cc4c...`` instead.
    """
    pointer = ParsedAttachment(
        provider_attachment_id=f"sediment://{_RENDITION_ID}",
        provider_file_id="file-provider-native",
        message_provider_id="m1",
    )
    session = ParsedSession(
        source_name=Provider.CHATGPT, provider_session_id="conversation", messages=[], attachments=[pointer]
    )
    sidecars: SidecarData = {"chatgpt_asset_index": ChatGPTAssetIndex.empty(), "chatgpt_asset_blobs": _RENDITION_BLOBS}

    returned = ChatGPTAssemblySpec().enrich_session(session, sidecars)

    assert len(returned.attachments) == 2
    assert {attachment.provider_file_id for attachment in returned.attachments} == {"file-provider-native"}


def test_prepared_carrier_appends_renditions_and_rolls_them_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prepared-carrier route archives every rendition, and a failure undoes the appends.

    Anti-vacuity: resolving only the first member leaves one attachment row;
    not restoring the attachment count after rollback leaves ``len`` at 3
    while the carrier holds 2 rows.
    """
    from polylogue.sources import assembly_chatgpt

    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        attachments = store.new_attachment_sink()
        events = store.new_event_sink()
        attachments.append(ParsedAttachment(provider_attachment_id=_RENDITION_ID, message_provider_id="m1"))
        session = ParsedSession(source_name=Provider.CHATGPT, provider_session_id="conversation", messages=[])
        session = session.model_copy(update={"attachments": attachments, "session_events": events})
        sidecars: SidecarData = {
            "chatgpt_asset_index": ChatGPTAssetIndex.empty(),
            "chatgpt_asset_blobs": _RENDITION_BLOBS,
        }

        assert ChatGPTAssemblySpec().enrich_session(session, sidecars) is session
        _assert_every_rendition_is_archived(list(attachments))
        assert len(events) == 2

        attachments = store.new_attachment_sink()
        events = store.new_event_sink()
        attachments.extend(
            [
                ParsedAttachment(provider_attachment_id=_RENDITION_ID, message_provider_id="m1"),
                ParsedAttachment(provider_attachment_id="file-other", message_provider_id="m2"),
            ]
        )
        session = session.model_copy(update={"attachments": attachments, "session_events": events})
        original = assembly_chatgpt._resolve_attachment_renditions
        calls = 0

        def fail_second(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("injected rendition failure")
            return original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(assembly_chatgpt, "_resolve_attachment_renditions", fail_second)
        with pytest.raises(ValueError, match="injected rendition failure"):
            ChatGPTAssemblySpec().enrich_session(session, sidecars)
        assert [attachment.provider_attachment_id for attachment in attachments] == [_RENDITION_ID, "file-other"]
        assert len(events) == 0
    finally:
        store.close()


def test_sidecar_enrichment_rolls_back_prepared_rows_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources import assembly_chatgpt

    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        attachments = store.new_attachment_sink()
        events = store.new_event_sink()
        attachments.extend(
            ParsedAttachment(provider_attachment_id=f"file-{index}", message_provider_id=f"message-{index}")
            for index in range(2)
        )
        session = ParsedSession(source_name=Provider.CHATGPT, provider_session_id="conversation", messages=[])
        session = session.model_copy(update={"attachments": attachments, "session_events": events})
        index = ChatGPTAssetIndex.build(
            library_files_payload=[],
            asset_file_names_payload={"file-0.dat": "asset.png"},
        )
        original = assembly_chatgpt._resolve_attachment
        seen = 0

        def failing_resolver(
            attachment: ParsedAttachment,
            asset_index: ChatGPTAssetIndex,
            *,
            thread_id: str,
            asset_blobs: Mapping[str, tuple[str, int]],
        ) -> tuple[ParsedAttachment, ParsedSessionEvent | None]:
            nonlocal seen
            seen += 1
            if seen == 2:
                raise ValueError("injected sidecar failure")
            return original(attachment, asset_index, thread_id=thread_id, asset_blobs=asset_blobs)

        monkeypatch.setattr(assembly_chatgpt, "_resolve_attachment", failing_resolver)
        with pytest.raises(ValueError, match="injected sidecar failure"):
            ChatGPTAssemblySpec().enrich_session(session, {"chatgpt_asset_index": index})
        assert [attachment.name for attachment in attachments] == [None, None]
        assert len(events) == 0
    finally:
        store.close()


def _library_entry(
    file_id: str,
    *,
    file_name: str | None = "report.csv",
    origination_message_id: str | None = None,
    origination_thread_id: str | None = None,
    sha256_digest: str | None = None,
    mime_type: str | None = "text/csv",
    file_size_bytes: int | None = 128,
) -> dict[str, object]:
    return {
        "file_id": file_id,
        "file_name": file_name,
        "file_extension": "csv",
        "mime_type": mime_type,
        "file_size_bytes": file_size_bytes,
        "sha256_digest": sha256_digest,
        "origination_message_id": origination_message_id,
        "origination_thread_id": origination_thread_id,
        "library_artifact_type": "other",
        "directory_id": "libdir_1",
        "created_at": "2026-07-28T12:00:00Z",
        "file_upload_time": "2026-07-28T12:00:00Z",
        "file_processed_time": "2026-07-28T12:00:05Z",
    }


class TestParseLibraryFiles:
    def test_parses_known_fields(self) -> None:
        records = parse_library_files([_library_entry("file_abc", file_name="notes.md", sha256_digest="deadbeef")])
        assert set(records) == {"file_abc"}
        record = records["file_abc"]
        assert record.file_name == "notes.md"
        assert record.sha256_digest == "deadbeef"
        assert record.mime_type == "text/csv"

    def test_skips_entries_without_file_id(self) -> None:
        assert parse_library_files([{"file_name": "x.txt"}]) == {}

    def test_non_list_payload_returns_empty(self) -> None:
        assert parse_library_files({"not": "a list"}) == {}
        assert parse_library_files(None) == {}

    def test_skips_non_dict_entries(self) -> None:
        assert parse_library_files(["not-a-dict", 42]) == {}


class TestParseAssetFileNames:
    def test_strips_dat_suffix(self) -> None:
        names = parse_asset_file_names({"file-abc123.dat": "image.png"})
        assert names == {"file-abc123": "image.png"}

    def test_keeps_keys_without_dat_suffix(self) -> None:
        names = parse_asset_file_names({"file-abc123": "image.png"})
        assert names == {"file-abc123": "image.png"}

    def test_non_dict_payload_returns_empty(self) -> None:
        assert parse_asset_file_names([1, 2, 3]) == {}
        assert parse_asset_file_names(None) == {}

    def test_skips_non_string_values(self) -> None:
        assert parse_asset_file_names({"file-x.dat": 123}) == {}


class TestChatGPTAssetIndexResolveDat:
    def test_prefers_library_files_over_asset_names(self) -> None:
        index = ChatGPTAssetIndex.build(
            library_files_payload=[_library_entry("file-abc", file_name="library-name.png", mime_type="image/png")],
            asset_file_names_payload={"file-abc.dat": "asset-name.png"},
        )
        resolved = index.resolve_dat("file-abc")
        assert resolved is not None
        assert resolved.name == "library-name.png"
        assert resolved.mime_type == "image/png"
        assert resolved.source == "library_files"

    def test_falls_back_to_asset_names(self) -> None:
        index = ChatGPTAssetIndex.build(
            library_files_payload=[],
            asset_file_names_payload={"file-only-named.dat": "image.png"},
        )
        resolved = index.resolve_dat("file-only-named")
        assert resolved is not None
        assert resolved.name == "image.png"
        assert resolved.mime_type is None
        assert resolved.source == "conversation_asset_file_names"

    def test_unknown_id_returns_none(self) -> None:
        index = ChatGPTAssetIndex.build(library_files_payload=[], asset_file_names_payload={})
        assert index.resolve_dat("file-unknown") is None

    def test_strips_file_service_prefix(self) -> None:
        index = ChatGPTAssetIndex.build(
            library_files_payload=[],
            asset_file_names_payload={"file-abc.dat": "image.png"},
        )
        resolved = index.resolve_dat("file-service://file-abc")
        assert resolved is not None
        assert resolved.name == "image.png"

    def test_strips_sediment_prefix(self) -> None:
        """``sediment://`` names the same id space ``file-service://`` does.

        Red if only the ``file-service://`` scheme is stripped: the
        computer-use screenshots (every one a ``sediment://`` pointer) miss
        the sidecar and the acquired blob keyed by their bare id.
        """
        index = ChatGPTAssetIndex.build(
            library_files_payload=[_library_entry("file_shot1", file_name="screenshot.png")],
            asset_file_names_payload={},
        )
        resolved = index.resolve_dat("sediment://file_shot1")
        assert resolved is not None
        assert resolved.name == "screenshot.png"

    def test_empty_index_reports_is_empty(self) -> None:
        assert ChatGPTAssetIndex.empty().is_empty is True
        assert ChatGPTAssetIndex.build(library_files_payload=[], asset_file_names_payload={}).is_empty is True
        assert ChatGPTAssetIndex.build(library_files_payload=[_library_entry("file_x")]).is_empty is False


class TestChatGPTAssetIndexResolveSandbox:
    def test_tier1_exact_message_id_and_name(self) -> None:
        index = ChatGPTAssetIndex.build(
            library_files_payload=[
                _library_entry(
                    "file_a", file_name="out.csv", origination_message_id="msg-1", origination_thread_id="th-1"
                )
            ]
        )
        resolution = index.resolve_sandbox(message_id="msg-1", thread_id="th-1", file_name="out.csv")
        assert resolution.tier == 1
        assert resolution.method == "message_id+name"
        assert resolution.file is not None
        assert resolution.file.file_id == "file_a"
        assert resolution.matched_name == "out.csv"

    def test_tier2_message_id_matches_but_name_differs(self) -> None:
        index = ChatGPTAssetIndex.build(
            library_files_payload=[
                _library_entry(
                    "file_a", file_name="renamed.csv", origination_message_id="msg-1", origination_thread_id="th-1"
                )
            ]
        )
        resolution = index.resolve_sandbox(message_id="msg-1", thread_id="th-1", file_name="different.csv")
        assert resolution.tier == 2
        assert resolution.method == "message_id"
        assert resolution.file is not None
        assert resolution.file.file_id == "file_a"
        # The library name is the label, not the identity key -- id alone
        # decided this join (bd polylogue-dt5s tier-2 spec).
        assert resolution.matched_name == "renamed.csv"

    def test_tier3_thread_id_and_name_when_message_id_absent(self) -> None:
        index = ChatGPTAssetIndex.build(
            library_files_payload=[_library_entry("file_a", file_name="out.csv", origination_thread_id="th-1")]
        )
        resolution = index.resolve_sandbox(message_id=None, thread_id="th-1", file_name="out.csv")
        assert resolution.tier == 3
        assert resolution.file is not None
        assert resolution.file.file_id == "file_a"

    def test_tier4_globally_unique_name_with_no_id_evidence(self) -> None:
        index = ChatGPTAssetIndex.build(library_files_payload=[_library_entry("file_a", file_name="unique.csv")])
        resolution = index.resolve_sandbox(
            message_id="unrelated-msg", thread_id="unrelated-thread", file_name="unique.csv"
        )
        assert resolution.tier == 4
        assert resolution.method == "global_name_unique"
        assert resolution.file is not None
        assert resolution.file.file_id == "file_a"

    def test_tier5_ambiguous_name_records_no_file(self) -> None:
        index = ChatGPTAssetIndex.build(
            library_files_payload=[
                _library_entry("file_a", file_name="shared.csv"),
                _library_entry("file_b", file_name="shared.csv"),
            ]
        )
        resolution = index.resolve_sandbox(message_id="unrelated", thread_id="unrelated", file_name="shared.csv")
        assert resolution.tier == 5
        assert resolution.method == "global_name_ambiguous"
        assert resolution.file is None
        assert resolution.matched_name == "shared.csv"

    def test_tier6_unresolved_when_nothing_matches(self) -> None:
        index = ChatGPTAssetIndex.build(library_files_payload=[_library_entry("file_a", file_name="other.csv")])
        resolution = index.resolve_sandbox(message_id="unrelated", thread_id="unrelated", file_name="missing.csv")
        assert resolution.tier == 6
        assert resolution.method == "unresolved"
        assert resolution.file is None
        assert resolution.matched_name is None

    def test_message_id_join_takes_priority_over_thread_and_global(self) -> None:
        # Two library files share a name globally; only one is tied to the
        # requesting message id. The id join must win over any name-only tier.
        index = ChatGPTAssetIndex.build(
            library_files_payload=[
                _library_entry("file_a", file_name="dup.csv", origination_message_id="msg-1"),
                _library_entry("file_b", file_name="dup.csv", origination_message_id="msg-2"),
            ]
        )
        resolution = index.resolve_sandbox(message_id="msg-1", thread_id=None, file_name="dup.csv")
        assert resolution.tier == 1
        assert resolution.file is not None
        assert resolution.file.file_id == "file_a"
