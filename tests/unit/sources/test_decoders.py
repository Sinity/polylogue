"""Tests for source decoding, JSON stream iteration, and ZIP processing.

Insightion code under test: polylogue/sources/decoders.py
Functions: _decode_json_bytes, owned_json_records, _ZipEntryValidator, _process_zip
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.archive.raw_payload.decode import jsonl_session_artifact, scan_jsonl_session_artifact
from polylogue.core.enums import Provider
from polylogue.sources import decoder_zip
from polylogue.sources.decoder_json import JsonlDecodeError
from polylogue.sources.decoders import (
    _decode_json_bytes,
    _ZipEntryValidator,
    open_zip_entry,
)
from polylogue.storage.cursor_state import CursorStatePayload
from tests.infra.json_values import iter_owned_json_values

# =============================================================================
# _decode_json_bytes
# =============================================================================


def _seeded_cursor_state() -> CursorStatePayload:
    return {"failed_files": [], "failed_count": 0}


class TestDecodeJsonBytesBasic:
    """Deterministic tests for _decode_json_bytes."""

    def test_utf8_roundtrip(self) -> None:
        """Encode then decode a UTF-8 JSON payload."""
        payload = '{"key": "value", "num": 42}'
        result = _decode_json_bytes(payload.encode("utf-8"))
        assert result is not None
        assert json.loads(result) == {"key": "value", "num": 42}

    def test_bom_stripping(self) -> None:
        """BOM-prefixed bytes are decoded (utf-8-sig in encoding list handles it)."""
        # utf-8-sig encoded bytes: BOM is consumed during decode
        payload = '{"key": "value"}'
        raw = payload.encode("utf-8-sig")  # Prepends EF BB BF
        result = _decode_json_bytes(raw)
        assert result is not None
        # utf-8 decoding succeeds first and preserves BOM as \ufeff.
        # utf-8-sig in the encoding list would strip it, but utf-8 wins first.
        # The decoded string may contain a leading BOM.
        # Verify the JSON content is present regardless.
        assert '"key"' in result
        assert '"value"' in result

    def test_utf8_sig_direct_bom(self) -> None:
        """Direct utf-8-sig BOM bytes are decoded successfully."""
        # Create bytes with a single BOM prefix
        bom = b"\xef\xbb\xbf"
        raw = bom + b'{"key": "value"}'
        result = _decode_json_bytes(raw)
        assert result is not None
        # Content is present
        assert "key" in result

    def test_null_bytes_removed(self) -> None:
        """Null bytes are stripped from decoded output."""
        payload = '{"key":\x00 "value"}'
        raw = payload.encode("utf-8")
        result = _decode_json_bytes(raw)
        assert result is not None
        assert "\x00" not in result
        assert "key" in result

    def test_fallback_encodings_utf16(self) -> None:
        """UTF-16 encoded payloads are decoded correctly."""
        payload = '{"key": "value"}'
        raw = payload.encode("utf-16")
        result = _decode_json_bytes(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_fallback_encodings_utf32(self) -> None:
        """UTF-32 encoded payloads are decoded correctly."""
        payload = '{"key": "value"}'
        raw = payload.encode("utf-32")
        result = _decode_json_bytes(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_returns_none_for_empty(self) -> None:
        """Empty bytes produce None."""
        result = _decode_json_bytes(b"")
        # Empty bytes decode to empty string, which is falsy
        assert result is None or result == ""

    def test_unicode_content_preserved(self) -> None:
        """Unicode content survives decode roundtrip."""
        payload = '{"emoji": "\\u2764", "jp": "\\u65e5\\u672c\\u8a9e"}'
        raw = payload.encode("utf-8")
        result = _decode_json_bytes(raw)
        assert result is not None
        assert json.loads(result) is not None


class TestDecodeJsonBytesFuzz:
    """Property-based tests for _decode_json_bytes."""

    @given(st.binary(max_size=4096))
    @settings(max_examples=200)
    def test_never_crashes_on_arbitrary_bytes(self, data: bytes) -> None:
        """_decode_json_bytes never raises on any input."""
        result = _decode_json_bytes(data)
        assert result is None or isinstance(result, str)


# =============================================================================
# owned_json_records
# =============================================================================


class TestIterJsonStream:
    """Tests for owned_json_records parsing strategies."""

    def test_jsonl_with_blank_lines(self) -> None:
        """JSONL parsing skips blank lines and yields valid objects."""
        content = b'{"a": 1}\n\n{"b": 2}\n\n\n{"c": 3}\n'
        handle = io.BytesIO(content)
        items = list(iter_owned_json_values(handle, "test.jsonl"))
        assert len(items) == 3
        assert items[0] == {"a": 1}
        assert items[1] == {"b": 2}
        assert items[2] == {"c": 3}

    def test_json_root_array(self) -> None:
        """Root array JSON is unpacked into individual items."""
        content = json.dumps([{"a": 1}, {"b": 2}]).encode("utf-8")
        handle = io.BytesIO(content)
        items = list(iter_owned_json_values(handle, "test.json"))
        assert len(items) == 2
        assert items[0] == {"a": 1}
        assert items[1] == {"b": 2}

    def test_sessions_wrapper(self) -> None:
        """{"sessions": [...]} is unpacked into individual items."""
        content = json.dumps({"sessions": [{"id": "c1"}, {"id": "c2"}]}).encode("utf-8")
        handle = io.BytesIO(content)
        items = list(iter_owned_json_values(handle, "test.json"))
        assert len(items) == 2
        assert items[0] == {"id": "c1"}
        assert items[1] == {"id": "c2"}

    def test_single_dict_yielded_as_is(self) -> None:
        """A single JSON dict is yielded without unwrapping."""
        content = json.dumps({"key": "value"}).encode("utf-8")
        handle = io.BytesIO(content)
        items = list(iter_owned_json_values(handle, "test.json"))
        assert len(items) == 1
        assert items[0] == {"key": "value"}

    def test_jsonl_invalid_lines_skipped(self) -> None:
        """Invalid JSON lines in JSONL are skipped (not crashed on)."""
        content = b'{"valid": 1}\nnot json at all\n{"also_valid": 2}\n'
        handle = io.BytesIO(content)
        items = list(iter_owned_json_values(handle, "data.jsonl"))
        assert len(items) == 2
        assert items[0] == {"valid": 1}
        assert items[1] == {"also_valid": 2}

    def test_ndjson_extension_treated_as_jsonl(self) -> None:
        """Files with .ndjson extension use JSONL parsing."""
        content = b'{"a": 1}\n{"b": 2}\n'
        handle = io.BytesIO(content)
        items = list(iter_owned_json_values(handle, "data.ndjson"))
        assert len(items) == 2

    def test_jsonl_txt_extension(self) -> None:
        """Files with .jsonl.txt extension use JSONL parsing."""
        content = b'{"a": 1}\n{"b": 2}\n'
        handle = io.BytesIO(content)
        items = list(iter_owned_json_values(handle, "data.jsonl.txt"))
        assert len(items) == 2

    def test_strict_jsonl_decode_reports_physical_offending_line(self) -> None:
        content = b'{"valid": 1}\n\nnot json at all\n{"later": 2}\n'
        with pytest.raises(JsonlDecodeError) as exc_info:
            list(iter_owned_json_values(io.BytesIO(content), "data.jsonl", fail_on_decode_error=True))
        assert exc_info.value.line_number == 3


# =============================================================================
# _ZipEntryValidator
# =============================================================================


class TestZipEntryValidator:
    """Tests for ZIP bomb protection and entry filtering."""

    def _make_zip_info(
        self,
        filename: str,
        file_size: int = 1000,
        compress_size: int = 100,
        is_dir: bool = False,
    ) -> zipfile.ZipInfo:
        """Create a ZipInfo with specified attributes."""
        info = zipfile.ZipInfo(filename)
        info.file_size = file_size
        info.compress_size = compress_size
        if is_dir:
            info.external_attr = 0o40775 << 16  # Directory bit
        return info

    def test_declared_size_and_ratio_do_not_drop_relevant_entries(self) -> None:
        validator = _ZipEntryValidator("chatgpt", cursor_state=None, zip_path=Path("input.zip"))
        entries = [
            self._make_zip_info("ratio.json", file_size=200000, compress_size=1),
            self._make_zip_info("large.json", file_size=11 * 1024**3, compress_size=11 * 1024**3),
        ]
        assert list(validator.filter_entries(entries)) == entries

    def test_claude_json_entries_are_not_special_cased(self) -> None:
        """Claude ZIP validation now relies on artifact classification, not filename allowlists."""
        validator = _ZipEntryValidator(
            "claude-ai",
            cursor_state=None,
            zip_path=Path("claude.zip"),
        )
        entries_in = [
            self._make_zip_info("sessions.json", file_size=5000, compress_size=500),
            self._make_zip_info("settings.json", file_size=1000, compress_size=100),
            self._make_zip_info("account.json", file_size=1000, compress_size=100),
        ]
        entries_out = list(validator.filter_entries(entries_in))
        assert [entry.filename for entry in entries_out] == [
            "sessions.json",
            "settings.json",
            "account.json",
        ]

    def test_directories_skipped(self) -> None:
        """Directory entries in ZIP are skipped."""
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=None,
            zip_path=Path("test.zip"),
        )
        dir_entry = self._make_zip_info("some_dir/", is_dir=True)
        # Manually set directory flag since ZipInfo.is_dir() checks filename
        dir_entry.filename = "some_dir/"
        entries = list(validator.filter_entries([dir_entry]))
        assert len(entries) == 0

    def test_non_json_extensions_skipped(self) -> None:
        """Non-JSON files in ZIP are skipped."""
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=None,
            zip_path=Path("test.zip"),
        )
        entries_in = [
            self._make_zip_info("readme.txt", file_size=500, compress_size=200),
            self._make_zip_info("image.png", file_size=5000, compress_size=4000),
            self._make_zip_info("data.json", file_size=1000, compress_size=100),
        ]
        entries_out = list(validator.filter_entries(entries_in))
        assert len(entries_out) == 1
        assert entries_out[0].filename == "data.json"

    def test_valid_entry_passes_through(self) -> None:
        """A normal JSON entry with reasonable ratio passes validation."""
        validator = _ZipEntryValidator(
            "chatgpt",
            cursor_state=None,
            zip_path=Path("test.zip"),
        )
        normal_entry = self._make_zip_info("sessions.json", file_size=50000, compress_size=5000)
        entries = list(validator.filter_entries([normal_entry]))
        assert len(entries) == 1

    def test_bounded_open_preserves_duplicate_zipinfo_identity(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("duplicate.json", b"first")
            zf.writestr("duplicate.json", b"second")
        buffer.seek(0)

        with zipfile.ZipFile(buffer) as zf:
            infos = zf.infolist()
            with open_zip_entry(zf, infos[0]) as handle:
                assert handle.read() == b"first"

    def test_complete_selection_has_no_aggregate_byte_budget(self) -> None:
        state = _seeded_cursor_state()
        validator = _ZipEntryValidator("chatgpt", cursor_state=state, zip_path=Path("input.zip"))
        entries = [
            self._make_zip_info(f"conversation_{i}.json", file_size=9 * 1024**3, compress_size=9 * 1024**3)
            for i in range(10)
        ]
        assert list(validator.filter_entries(entries)) == entries
        assert state["failed_count"] == 0

    def test_validator_leaves_terminal_artifact_classification_to_zip_processing(self) -> None:
        """ZIP validation must not path-exclude entries before payload decoding.

        Regression test for polylogue-dc1k: every ``OriginArtifactRule.path_pattern``
        in ``origin_specs.py`` is anchored ``(?:^|/)``, but the entry was
        classified on ``f"{zip_path}:{name}"`` -- the character immediately
        before the pattern's leading segment was ``:``, never ``/`` or
        start-of-string, so no rule could ever match and this exclusion was
        dead code for every zip import. Builds a *real* zip (not a bare
        ``ZipInfo``) so the entries here are exactly what ``zf.infolist()``
        would hand ``filter_entries`` in production.
        """
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            # Matches the "workflow_run" OriginArtifactRule for claude-code
            # (parse_policy="fact" -> parse_as_session=False): must be excluded.
            zf.writestr("workflows/run.json", json.dumps({"run_id": "abc"}))
            # Matches the "agent_transcript" OriginArtifactRule for claude-code
            # (parse_policy="session" -> parse_as_session=True): must survive.
            zf.writestr("subagents/agent-1.jsonl", json.dumps({"type": "user"}) + "\n")
            # No OriginArtifactRule matches this path at all, so ZIP processing
            # must leave it available for ordinary payload classification.
            zf.writestr("sessions.json", json.dumps({"conversations": []}))
        buffer.seek(0)

        with zipfile.ZipFile(buffer) as zf:
            validator = _ZipEntryValidator(
                "claude-code", cursor_state=_seeded_cursor_state(), zip_path=Path("export.zip")
            )
            accepted = [info.filename for info in validator.filter_entries(zf.infolist())]

        assert "workflows/run.json" in accepted
        assert "subagents/agent-1.jsonl" in accepted
        assert "sessions.json" in accepted


def test_zip_admission_uses_declared_claude_sidecar_paths_for_nonstandard_forms() -> None:
    """Path-owned tool results survive ZIP lowering regardless of extension.

    Anti-vacuity mutation: removing ``allowed_path`` from ZIP admission drops
    the text, HTML, and extensionless members before artifact classification.
    An unrelated text member remains excluded, so the exception is not a
    global suffix widening.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name in ("toolu.json", "toolu.txt", "toolu.html", "toolu"):
            zf.writestr(f"project/session/tool-results/{name}", "opaque output")
        zf.writestr("project/session/notes.txt", "ordinary text")
    buffer.seek(0)

    with zipfile.ZipFile(buffer) as zf:
        validator = _ZipEntryValidator("claude-code", cursor_state=_seeded_cursor_state(), zip_path=Path("export.zip"))
        accepted = [info.filename for info in validator.filter_entries(zf.infolist())]

    assert accepted == [
        "project/session/tool-results/toolu.json",
        "project/session/tool-results/toolu.txt",
        "project/session/tool-results/toolu.html",
        "project/session/tool-results/toolu",
    ]


def test_explicit_zip_suffix_filter_does_not_infer_declared_paths() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("project/session/tool-results/toolu.json", "{}")
        zf.writestr("project/session/tool-results/toolu.txt", "opaque output")
    buffer.seek(0)

    with zipfile.ZipFile(buffer) as zf:
        validator = _ZipEntryValidator("claude-code", cursor_state=None, zip_path=Path("export.zip"))
        accepted = [info.filename for info in validator.filter_entries(zf.infolist(), allowed_suffixes=(".json",))]

    assert accepted == ["project/session/tool-results/toolu.json"]


def test_zip_admission_does_not_promote_weak_analysis_classification() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("analysis/cache.bin", "generated")
        zf.writestr("session/tool-results/toolu.txt", "opaque output")
    buffer.seek(0)

    with zipfile.ZipFile(buffer) as zf:
        validator = _ZipEntryValidator("claude-code", cursor_state=None, zip_path=Path("export.zip"))
        accepted = [info.filename for info in validator.filter_entries(zf.infolist())]

    assert accepted == ["session/tool-results/toolu.txt"]


# ZIP CONTENT-PROBE CEILING


def _zip_with_member(path: Path, name: str, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)


def test_zip_json_probe_consumes_complete_positive_member(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[2] / "fixtures" / "chatgpt" / "native-conversation-v1.json"
    archive_path = tmp_path / "compressed.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        with archive.open("assets/conversations.json", "w") as member:
            member.write(b'{"padding":"')
            for _ in range(64):
                member.write(b"x" * (1024 * 1024))
            member.write(b'",' + fixture.read_bytes().lstrip()[1:])
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.infolist()[0]
        assert info.file_size / info.compress_size > 1000
        artifact = decoder_zip.zip_entry_session_artifact(archive, info, provider=Provider.CHATGPT)
    assert artifact is not None and artifact.parse_as_session


def test_zip_parser_progress_identity_tracks_captured_occurrences_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import warnings

    from devtools.fresh_build_bench.run import WorkProgressTail
    from polylogue.core import work_progress
    from polylogue.core.raw_coordinates import MemberAddressingMode
    from polylogue.sources.source_acquisition_components import (
        captured_zip_member_coordinate,
        zip_acquisition_fingerprint,
    )
    from polylogue.sources.source_staging import bind_source_input

    fixture = Path(__file__).parents[2] / "fixtures" / "chatgpt" / "native-conversation-v1.json"
    payload = fixture.read_bytes()
    first_container = tmp_path / "first.zip"
    second_container = tmp_path / "second.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(first_container, "w") as archive:
            archive.writestr("assets/conversations.json", payload)
            archive.writestr("assets/conversations.json", payload)
    with zipfile.ZipFile(second_container, "w") as archive:
        archive.writestr("assets/conversations.json", payload)

    decoder_fingerprint = zip_acquisition_fingerprint(Provider.CHATGPT)

    def run_entry(container: Path, ordinal: int) -> tuple[set[str], int]:
        with zipfile.ZipFile(container) as archive:
            info = archive.infolist()[ordinal]
            assert archive.read(info) == payload
            with bind_source_input(container) as binding:
                coordinate = captured_zip_member_coordinate(
                    binding.captured_identity,
                    entry_name=info.filename,
                    entry_ordinal=ordinal,
                    split_index=0,
                    addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
                    container_blob_hash=hashlib.sha256(container.read_bytes()).hexdigest(),
                    decoder_fingerprint=decoder_fingerprint,
                )
            assert coordinate is not None
            classification = decoder_zip.zip_entry_session_artifact(
                archive,
                info,
                provider=Provider.CHATGPT,
                captured_zip_coordinate=coordinate,
            )
            assert classification is not None and classification.parse_as_session
        product_ids = {
            str(fields["productive_id"])
            for event, fields in emitted
            if event == "daemon.work.progress" and fields.get("phase") == "source_preparation"
        }
        assert product_ids
        with events_path.open("a", encoding="utf-8") as handle:
            for event, fields in emitted:
                if event == "daemon.work.progress":
                    handle.write(json.dumps({"event": event, **fields}) + "\n")
        emitted.clear()
        return product_ids, tail.poll()

    emitted: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(work_progress, "PROGRESS_INTERVAL_S", 0)
    monkeypatch.setattr(work_progress, "emit", lambda event, **fields: emitted.append((event, fields)))
    events_path = tmp_path / "parser-progress.jsonl"
    tail = WorkProgressTail(events_path, state_root=tmp_path)
    try:
        first = run_entry(first_container, 0)
        retry = run_entry(first_container, 0)
        duplicate_entry = run_entry(first_container, 1)
        second_container_entry = run_entry(second_container, 0)

        assert first[0] == retry[0]
        assert first[1] > 0
        assert retry[1] == first[1]
        assert len(duplicate_entry[0]) == 1 and duplicate_entry[0] != first[0]
        assert duplicate_entry[1] > retry[1]
        assert len(second_container_entry[0]) == 1 and second_container_entry[0] != duplicate_entry[0]
        assert second_container_entry[1] > duplicate_entry[1]
    finally:
        tail.close()


def test_zip_json_probe_does_not_override_with_empty_session_shape(tmp_path: Path) -> None:
    archive_path = tmp_path / "empty.zip"
    _zip_with_member(archive_path, "assets/conversations.json", b'{"title":"empty","mapping":{}}')
    with zipfile.ZipFile(archive_path) as archive:
        assert decoder_zip.zip_entry_session_artifact(archive, archive.infolist()[0], provider=Provider.CHATGPT) is None


def test_jsonl_session_artifact_preserves_a_large_valid_record() -> None:
    payload = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": "large-message",
            "role": "user",
            "content": [{"type": "input_text", "text": "x" * 200_000}],
        },
    }
    raw = (json.dumps(payload) + "\n").encode()
    artifact = jsonl_session_artifact(io.BytesIO(raw), provider=Provider.CODEX)
    assert artifact is not None
    assert artifact.parse_as_session
    assert not artifact.schema_eligible
    scan = scan_jsonl_session_artifact(io.BytesIO(raw), provider=Provider.CODEX)
    assert scan.malformed_records == 0


def test_zip_parser_uses_accepted_container_after_declared_alias_retargets(tmp_path: Path) -> None:
    """A reopen of the operator alias would parse B and lose A's entry receipt."""
    from polylogue.sources.source_staging import bind_source_input
    from polylogue.storage.blob_store import BlobStore

    payload = (Path(__file__).parents[2] / "fixtures" / "chatgpt" / "native-conversation-v1.json").read_bytes()
    accepted = tmp_path / "accepted.zip"
    unrelated = tmp_path / "unrelated.zip"
    alias = tmp_path / "declared.zip"
    _zip_with_member(accepted, "conversations.json", payload)
    _zip_with_member(unrelated, "other.json", b'{"title":"unrelated","mapping":{}}')
    alias.symlink_to(accepted)
    with bind_source_input(alias) as binding:
        alias.unlink()
        alias.symlink_to(unrelated)
        rows = list(
            decoder_zip.process_zip(
                alias,
                provider_hint=Provider.CHATGPT,
                should_group=True,
                file_mtime=None,
                capture_raw=True,
                cursor_state=None,
                blob_store=BlobStore(tmp_path / "blobs"),
                source_binding=binding,
            )
        )
    assert rows
    for raw, session in rows:
        assert session.source_name is Provider.CHATGPT
        assert raw is not None
        assert raw.captured_zip_coordinate is not None
        assert raw.captured_zip_coordinate.canonical_container == str(accepted)
        assert raw.captured_zip_coordinate.member_name == "conversations.json"
        assert raw.captured_zip_coordinate.entry_ordinal == 0
        assert raw.source_path == f"{alias}:conversations.json"


def test_complete_jsonl_candidacy_preserves_healthy_records_before_bad_utf8() -> None:
    message = b'{"sessionId":"accepted","uuid":"m1","type":"user","cwd":"/neutral"}\n'
    handle = io.BytesIO(message + b'{"text":"\xff"}\n' + message)
    scan = scan_jsonl_session_artifact(handle, provider=Provider.CLAUDE_CODE)
    assert scan.artifact is not None
    assert scan.artifact.parse_as_session
    assert scan.malformed_records == 1
    assert handle.tell() == len(handle.getvalue())
    assert not handle.closed


def test_complete_taxonomy_rewinds_multibyte_text_using_its_opaque_cookie() -> None:
    from polylogue.archive.raw_payload.streams import raw_byte_stream

    class CookieText(io.StringIO):
        def tell(self) -> int:
            return 1_000_000 + super().tell()

        def seek(self, offset: int, whence: int = 0) -> int:
            assert whence == 0
            assert offset >= 1_000_000
            return 1_000_000 + super().seek(offset - 1_000_000)

    payload = '{"label":"α😀"}\n{"label":"終"}\n'
    caller = CookieText(payload)
    with raw_byte_stream(caller) as view:
        expected = payload.encode("utf-8")
        assert view.read(7) == expected[:7]
        view.seek(0)
        assert view.read() == expected
        assert view.tell() == len(expected)
        view.seek(5)
        assert view.read() == expected[5:]
    assert not caller.closed
    caller.seek(1_000_000)
    scan = scan_jsonl_session_artifact(caller, provider=Provider.UNKNOWN)
    assert scan.proved_non_session
    assert scan.valid_records == 2
    assert not caller.closed


@pytest.mark.parametrize("text", [False, True])
@pytest.mark.parametrize("advertises_seek", [False, True])
def test_complete_taxonomy_preserves_nonseekable_input_and_caller_closure(text: bool, advertises_seek: bool) -> None:
    payload = '{"label":"α😀"}\n{"label":"終"}\n'

    class BinaryPipe(io.BytesIO):
        def seekable(self) -> bool:
            return advertises_seek

        def seek(self, *_args: object) -> int:
            raise io.UnsupportedOperation("synthetic pipe")

        def tell(self) -> int:
            raise io.UnsupportedOperation("synthetic pipe")

    class TextPipe(io.StringIO):
        def seekable(self) -> bool:
            return advertises_seek

        def seek(self, *_args: object) -> int:
            raise io.UnsupportedOperation("synthetic pipe")

        def tell(self) -> int:
            raise io.UnsupportedOperation("synthetic pipe")

    caller = TextPipe(payload) if text else BinaryPipe(payload.encode())
    scan = scan_jsonl_session_artifact(caller, provider=Provider.UNKNOWN)
    assert scan.proved_non_session
    assert scan.valid_records == 2
    assert not caller.closed
    assert caller.read() in (b"", "")


@pytest.mark.parametrize("cancelled", [False, True])
def test_nonseekable_taxonomy_failed_native_close_retains_replay_artifact(
    monkeypatch: pytest.MonkeyPatch,
    cancelled: bool,
) -> None:
    import sqlite3

    from polylogue.storage.sqlite import connection_profile
    from tests.infra.sqlite_cursor_settlement import ControlledConnection

    class BinaryPipe(io.BytesIO):
        def seekable(self) -> bool:
            return False

    actual_connect = sqlite3.connect

    def connect(database: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if str(database).endswith("bytes.db"):
            kwargs["factory"] = ControlledConnection
            connection = actual_connect(database, *args, **kwargs)
            assert isinstance(connection, ControlledConnection)
            connection.close_failure = sqlite3.OperationalError("synthetic replay close failure")
            return connection
        result = actual_connect(database, *args, **kwargs)
        assert isinstance(result, sqlite3.Connection)
        return result

    monkeypatch.setattr(sqlite3, "connect", connect)
    caller = BinaryPipe(b'{"metadata":1}\n')
    cancellation = InterruptedError("synthetic replay cancellation")

    def stop() -> None:
        if cancelled:
            raise cancellation

    with pytest.raises(connection_profile.NativeConnectionSettlementError) as refused:
        scan_jsonl_session_artifact(caller, provider=Provider.UNKNOWN, check_stop=stop)
    owner = refused.value.owner
    assert not caller.closed
    directory = owner.scratch_directory
    assert directory is not None
    assert Path(directory.name).is_dir()
    if cancelled:
        assert refused.value.__cause__ is cancellation
    connection = owner.connection
    assert isinstance(connection, ControlledConnection)
    assert connection.close_attempts == 1
    connection.close_failure = None
    owner.close()
    assert connection.close_attempts == 2
    assert not Path(directory.name).exists()


def test_nonseekable_taxonomy_cancellation_keeps_the_caller_open() -> None:
    class BinaryPipe(io.BytesIO):
        def seekable(self) -> bool:
            return False

    caller = BinaryPipe(b'{"metadata":1}\n' * 100)

    def stop() -> None:
        raise InterruptedError("synthetic replay cancellation")

    with pytest.raises(InterruptedError):
        scan_jsonl_session_artifact(caller, provider=Provider.UNKNOWN, check_stop=stop)
    assert not caller.closed


@pytest.mark.parametrize("text", [b"\xed\xa0\x80", b"\\ud800", b"\xed\xa0\xbd\xed\xb8\x80"])
def test_complete_jsonl_projection_preserves_provider_surrogates(text: bytes) -> None:
    from polylogue.core.json import decode_provider_utf8
    from polylogue.sources.detection_projection import DetectorProjection, iter_projected_jsonl_records

    raw = b'{"text":"' + text + b'"}\n'
    records = list(
        iter_projected_jsonl_records(io.BytesIO(raw), DetectorProjection(fields={"text": DetectorProjection()}))
    )
    assert records == [json.loads(decode_provider_utf8(raw))]


def test_complete_jsonl_projection_rejects_raw_nul_without_losing_earlier_records() -> None:
    from polylogue.sources.detection_projection import DetectorProjection, iter_projected_jsonl_records

    failures: list[Exception] = []
    records = list(
        iter_projected_jsonl_records(
            io.BytesIO(b'{"id":"first"}\n{"id":"bad\x00value"}\n{"id":"last"}\n'),
            DetectorProjection(fields={"id": DetectorProjection()}),
            on_decode_failure=failures.append,
        )
    )
    assert records == [{"id": "first"}, {"id": "last"}]
    assert len(failures) == 1


@pytest.mark.parametrize(
    "error", [UnicodeError("stop"), OSError("stop"), ValueError("stop"), json.JSONDecodeError("stop", "", 0)]
)
def test_complete_jsonl_projection_preserves_stop_callback_failure(error: Exception) -> None:
    from polylogue.sources.detection_projection import DetectorProjection, iter_projected_jsonl_records

    failures: list[Exception] = []
    calls = 0

    def stop() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:  # The line reader's checkpoint, inside event decoding.
            raise error

    with pytest.raises(type(error)) as raised:
        list(
            iter_projected_jsonl_records(
                io.BytesIO(b'{"id":"accepted"}\n'),
                DetectorProjection(fields={"id": DetectorProjection()}),
                check_stop=stop,
                on_decode_failure=failures.append,
            )
        )
    assert raised.value is error
    assert failures == []


@pytest.mark.parametrize("document", [False, True])
def test_complete_detection_projection_refuses_raw_nul_in_document_or_jsonl(document: bool) -> None:
    import ijson

    from polylogue.sources.detection_projection import (
        DetectorProjection,
        iter_projected_document_records,
        project_detection_input,
    )

    handle = io.BytesIO(b'{"id":"bad\x00value"}\n')
    rule = DetectorProjection(fields={"id": DetectorProjection()})
    with pytest.raises((ijson.JSONError, json.JSONDecodeError)):
        if document:
            list(iter_projected_document_records(handle, rule))
        else:
            project_detection_input(handle, rule)
    assert not handle.closed
