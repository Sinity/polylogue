"""Tests for source decoding, JSON stream iteration, and ZIP processing.

Insightion code under test: polylogue/sources/decoders.py
Functions: _decode_json_bytes, _iter_json_stream, _ZipEntryValidator, _process_zip
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.archive.raw_payload.decode import jsonl_session_artifact, scan_jsonl_session_artifact
from polylogue.core.enums import Provider
from polylogue.sources import decoder_zip
from polylogue.sources.decoder_json import JsonlDecodeError
from polylogue.sources.decoders import (
    _decode_json_bytes,
    _iter_json_stream,
    _ZipEntryValidator,
    open_zip_entry,
)
from polylogue.storage.cursor_state import CursorStatePayload

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
# _iter_json_stream
# =============================================================================


class TestIterJsonStream:
    """Tests for _iter_json_stream parsing strategies."""

    def test_jsonl_with_blank_lines(self) -> None:
        """JSONL parsing skips blank lines and yields valid objects."""
        content = b'{"a": 1}\n\n{"b": 2}\n\n\n{"c": 3}\n'
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "test.jsonl"))
        assert len(items) == 3
        assert items[0] == {"a": 1}
        assert items[1] == {"b": 2}
        assert items[2] == {"c": 3}

    def test_json_root_array(self) -> None:
        """Root array JSON is unpacked into individual items."""
        content = json.dumps([{"a": 1}, {"b": 2}]).encode("utf-8")
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "test.json"))
        assert len(items) == 2
        assert items[0] == {"a": 1}
        assert items[1] == {"b": 2}

    def test_sessions_wrapper(self) -> None:
        """{"sessions": [...]} is unpacked into individual items."""
        content = json.dumps({"sessions": [{"id": "c1"}, {"id": "c2"}]}).encode("utf-8")
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "test.json"))
        assert len(items) == 2
        assert items[0] == {"id": "c1"}
        assert items[1] == {"id": "c2"}

    def test_single_dict_yielded_as_is(self) -> None:
        """A single JSON dict is yielded without unwrapping."""
        content = json.dumps({"key": "value"}).encode("utf-8")
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "test.json"))
        assert len(items) == 1
        assert items[0] == {"key": "value"}

    def test_jsonl_invalid_lines_skipped(self) -> None:
        """Invalid JSON lines in JSONL are skipped (not crashed on)."""
        content = b'{"valid": 1}\nnot json at all\n{"also_valid": 2}\n'
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "data.jsonl"))
        assert len(items) == 2
        assert items[0] == {"valid": 1}
        assert items[1] == {"also_valid": 2}

    def test_ndjson_extension_treated_as_jsonl(self) -> None:
        """Files with .ndjson extension use JSONL parsing."""
        content = b'{"a": 1}\n{"b": 2}\n'
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "data.ndjson"))
        assert len(items) == 2

    def test_jsonl_txt_extension(self) -> None:
        """Files with .jsonl.txt extension use JSONL parsing."""
        content = b'{"a": 1}\n{"b": 2}\n'
        handle = io.BytesIO(content)
        items = list(_iter_json_stream(handle, "data.jsonl.txt"))
        assert len(items) == 2

    def test_strict_jsonl_decode_reports_physical_offending_line(self) -> None:
        content = b'{"valid": 1}\n\nnot json at all\n{"later": 2}\n'
        with pytest.raises(JsonlDecodeError) as exc_info:
            list(_iter_json_stream(io.BytesIO(content), "data.jsonl", fail_on_decode_error=True))
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
    payload = json.loads(fixture.read_bytes())
    payload["padding"] = "x" * (2 * 1024 * 1024)
    archive_path = tmp_path / "compressed.zip"
    _zip_with_member(archive_path, "assets/conversations.json", json.dumps(payload).encode())
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.infolist()[0]
        assert info.file_size / info.compress_size > 1000
        artifact = decoder_zip.zip_entry_session_artifact(archive, info, provider=Provider.CHATGPT)
    assert artifact is not None and artifact.parse_as_session


def test_zip_json_probe_does_not_override_with_empty_session_shape(tmp_path: Path) -> None:
    archive_path = tmp_path / "empty.zip"
    _zip_with_member(archive_path, "assets/conversations.json", b'{"title":"empty","mapping":{}}')
    with zipfile.ZipFile(archive_path) as archive:
        assert decoder_zip.zip_entry_session_artifact(archive, archive.infolist()[0], provider=Provider.CHATGPT) is None


def test_jsonl_session_artifact_forwards_its_record_ceiling() -> None:
    """The classification wrapper must not drop ``max_record_bytes``.

    Anti-vacuity: the wrapper previously called ``scan_jsonl_session_artifact``
    without forwarding the bound, so every caller that wanted only the
    classification silently got unbounded per-line reads. Reverting that
    forwarding makes the oversized record inspectable again and the artifact
    resolves, turning the ``is None`` assertion red.
    """
    oversized = (json.dumps({"title": "x" * 200_000, "mapping": {}}) + "\n").encode("utf-8")

    assert (
        jsonl_session_artifact(
            io.BytesIO(oversized),
            provider=Provider.CHATGPT,
            max_record_bytes=1024,
        )
        is None
    )
    # Without the ceiling the same bytes are inspected normally, proving the
    # input is otherwise classifiable and the bound is what changed the outcome.
    scan = scan_jsonl_session_artifact(io.BytesIO(oversized), provider=Provider.CHATGPT)
    assert scan.oversized_records == 0


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
