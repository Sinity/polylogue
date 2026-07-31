"""Tests for ChatGPT export sidecar resolvers (bd polylogue-0hwv / polylogue-dt5s).

Fixture shapes and tier counts mirror the measured spec recorded on the two
beads (2026-07-31, against the real 2026-07-29 export): two id namespaces
(``file-<b64ish>`` conversation assets, ``file_<32hex>`` library files), a
six-tier sandbox-file resolver where tier 5 (ambiguous) is possible but tier
1-4 dominate, and library_files preferred over conversation_asset_file_names
whenever both name the same id.
"""

from __future__ import annotations

from polylogue.sources.parsers.chatgpt_sidecars import (
    ChatGPTAssetIndex,
    parse_asset_file_names,
    parse_library_files,
)


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
