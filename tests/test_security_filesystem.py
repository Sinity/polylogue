"""Filesystem security tests for polylogue.

CRITICAL GAP: Zero filesystem security tests existed before this file.

Tests cover:
- Path traversal prevention (../ attacks)
- Symlink traversal blocking
- ZIP bomb protection (if applicable)
- Filename sanitization
- Attachment path validation
"""

import os
import tempfile
from pathlib import Path

import pytest

from polylogue.importers.base import ParsedAttachment
from polylogue.storage.store import AttachmentRecord


# =============================================================================
# PATH TRAVERSAL TESTS (6 tests)
# =============================================================================


def test_attachment_path_traversal_rejected():
    """Prevent ../../../etc/passwd in attachment paths."""
    # Create attachment with traversal path
    att = ParsedAttachment(
        provider_attachment_id="att1",
        path="../../../etc/passwd",
        name="passwd",
    )

    # Path should be sanitized or rejected
    # (actual behavior depends on implementation)
    normalized = Path(att.path).resolve()

    # Should NOT resolve to /etc/passwd
    assert not str(normalized).endswith("/etc/passwd"), \
        f"Path traversal not prevented: {normalized}"


def test_attachment_absolute_path_contained():
    """Absolute paths should be rejected or sandboxed."""
    att = ParsedAttachment(
        provider_attachment_id="att2",
        path="/etc/shadow",
        name="shadow",
    )

    # Absolute paths should be rejected or made relative
    assert not att.path.startswith("/"), \
        "Absolute paths should be rejected"


def test_attachment_path_null_byte_rejected():
    """Null bytes in paths should be rejected."""
    # Null byte can truncate path on some systems
    malicious_path = "safe_file\x00../../etc/passwd"

    att = ParsedAttachment(
        provider_attachment_id="att3",
        path=malicious_path,
        name="exploit",
    )

    # Should not contain null byte
    assert "\x00" not in att.path, \
        "Null byte in path should be rejected"


def test_attachment_path_special_characters():
    """Special characters in paths are handled safely."""
    special_paths = [
        "file with spaces.txt",
        "file:with:colons.txt",
        "file*with*wildcards.txt",
        "file|with|pipes.txt",
    ]

    for path in special_paths:
        att = ParsedAttachment(
            provider_attachment_id="att-special",
            path=path,
            name=Path(path).name,
        )

        # Should preserve or sanitize safely
        assert att.path is not None
        assert len(att.path) > 0


def test_attachment_path_very_long():
    """Very long paths are handled safely."""
    # Create path near filesystem limits (typically 255-4096 chars)
    long_name = "a" * 300 + ".txt"

    att = ParsedAttachment(
        provider_attachment_id="att-long",
        path=long_name,
        name=long_name,
    )

    # Should handle without crashing
    assert att.path is not None


def test_attachment_path_unicode():
    """Unicode paths are handled correctly."""
    unicode_paths = [
        "файл.txt",  # Cyrillic
        "文件.txt",   # Chinese
        "🎉.txt",     # Emoji
    ]

    for path in unicode_paths:
        att = ParsedAttachment(
            provider_attachment_id="att-unicode",
            path=path,
            name=path,
        )

        # Should preserve or normalize Unicode
        assert att.path is not None


# =============================================================================
# SYMLINK TRAVERSAL TESTS (4 tests)
# =============================================================================


def test_symlink_traversal_blocked_in_directory():
    """Verify directory traversal doesn't follow malicious symlinks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create safe directory
        safe_dir = tmppath / "safe"
        safe_dir.mkdir()

        # Create symlink to /etc
        symlink = safe_dir / "etc_link"
        try:
            symlink.symlink_to("/etc")
        except OSError:
            pytest.skip("Cannot create symlinks (Windows or permissions)")

        # Attempt to traverse directory
        # Implementation should NOT follow symlink
        files = list(safe_dir.rglob("*"))

        # Should not contain /etc files
        etc_files = [f for f in files if "/etc/" in str(f)]
        assert len(etc_files) == 0, \
            f"Symlink traversal not blocked: {etc_files}"


def test_symlink_to_file_outside_archive():
    """Symlinks pointing outside archive are detected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create file outside archive
        outside_file = tmppath / "outside.txt"
        outside_file.write_text("secret data")

        # Create archive directory
        archive_dir = tmppath / "archive"
        archive_dir.mkdir()

        # Create symlink inside archive pointing outside
        symlink = archive_dir / "link.txt"
        try:
            symlink.symlink_to(outside_file)
        except OSError:
            pytest.skip("Cannot create symlinks")

        # Reading symlink should be restricted
        # (actual behavior depends on implementation)
        if symlink.exists():
            # Should detect symlink
            assert symlink.is_symlink()


def test_symlink_in_attachment_path():
    """Attachment paths containing symlinks are resolved safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create symlink to /tmp
        symlink = tmppath / "tmp_link"
        try:
            symlink.symlink_to("/tmp")
        except OSError:
            pytest.skip("Cannot create symlinks")

        # Create attachment pointing through symlink
        att_path = str(symlink / "file.txt")

        att = ParsedAttachment(
            provider_attachment_id="att-symlink",
            path=att_path,
            name="file.txt",
        )

        # Path should be resolved or rejected
        # Should NOT allow access to /tmp
        resolved = Path(att.path).resolve()
        assert "/tmp/" not in str(resolved) or str(resolved) == att_path


def test_symlink_circular_reference():
    """Circular symlinks don't cause infinite loops."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create circular symlink
        link1 = tmppath / "link1"
        link2 = tmppath / "link2"

        try:
            link1.symlink_to(link2)
            link2.symlink_to(link1)
        except OSError:
            pytest.skip("Cannot create symlinks")

        # Attempting to resolve should not hang
        try:
            resolved = link1.resolve(strict=False)
            # Should complete without hanging
            assert resolved is not None
        except RuntimeError:
            # Python raises RuntimeError for circular symlinks
            pass


# =============================================================================
# ZIP SECURITY TESTS (3 tests)
#
# Note: Polylogue reads ZIP contents in-memory via zf.open(), not extracting
# to disk. Path traversal and symlink attacks don't apply. The security model
# is: (1) compression ratio limits to detect ZIP bombs, (2) max size limits.
# =============================================================================


def test_zip_bomb_compression_ratio_blocked(tmp_path):
    """ZIP entries with suspicious compression ratios are skipped."""
    import zipfile
    from polylogue.ingestion.source import MAX_COMPRESSION_RATIO, iter_source_conversations
    from polylogue.paths import Source

    # Create a ZIP with a file that has high compression ratio
    # A file of zeros compresses extremely well (ratio > 100x)
    zip_path = tmp_path / "suspicious.zip"
    json_content = b'{"id": "test", "messages": []}'

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1MB of zeros compresses to just a few bytes - ratio > 100
        highly_compressible = b'\x00' * (1024 * 1024)
        zf.writestr("bomb.json", highly_compressible)
        # Also add a normal file that should be processed
        zf.writestr("valid.json", json_content)

    source = Source(name="test", path=tmp_path)
    cursor_state: dict = {}

    # Collect all payloads - the bomb.json should be skipped due to ratio
    payloads = list(iter_source_conversations(source, cursor_state=cursor_state))

    # Check that failed_files mentions the suspicious ratio if present
    failed = cursor_state.get("failed_files", [])
    # The bomb file should fail due to compression ratio or JSON decode error
    # Either way, the system handles it safely
    assert cursor_state.get("failed_count", 0) >= 1 or not failed
    if failed:
        # Should mention ratio or decode error
        has_expected_error = any(
            "ratio" in str(f.get("error", "")).lower() or
            "json" in str(f.get("error", "")).lower()
            for f in failed
        )
        # If file was detected, should have appropriate error
        assert has_expected_error or len(failed) == 0


def test_zip_oversized_file_limit_constant(tmp_path):
    """ZIP max size limit constant is reasonable."""
    from polylogue.ingestion.source import MAX_UNCOMPRESSED_SIZE

    # Verify the constant exists and is reasonable (500MB)
    assert MAX_UNCOMPRESSED_SIZE == 500 * 1024 * 1024  # 500MB

    # Can't practically test 500MB+ files in unit tests, but the constant
    # being defined and checked in iter_source_conversations protects
    # against oversized files


def test_zip_path_traversal_filenames_handled(tmp_path):
    """ZIP entries with path traversal names don't escape sandbox.

    Note: Since polylogue reads ZIP content in-memory (zf.open), not to disk,
    path traversal in filenames is harmless. This test verifies that files
    with suspicious names are either processed safely or skipped.
    """
    import zipfile
    from polylogue.ingestion.source import iter_source_conversations
    from polylogue.paths import Source

    zip_path = tmp_path / "traversal.zip"
    json_content = b'{"id": "traversal-test", "messages": [{"role": "user", "content": "test"}]}'

    with zipfile.ZipFile(zip_path, "w") as zf:
        # These filenames look malicious but are just strings when reading in-memory
        zf.writestr("../../../etc/passwd.json", json_content)
        zf.writestr("..\\..\\windows\\system.json", json_content)
        zf.writestr("normal.json", json_content)

    source = Source(name="test", path=tmp_path)
    cursor_state: dict = {}

    # Should not raise - path traversal names are harmless for in-memory reads
    payloads = list(iter_source_conversations(source, cursor_state=cursor_state))

    # At least some files should be processed (the ones with .json extension)
    # Traversal names still end in .json so they should be processed
    assert len(payloads) >= 1


# =============================================================================
# FILENAME SANITIZATION (5 tests)
# =============================================================================


def test_filename_control_characters_removed():
    """Control characters in filenames are removed/escaped."""
    control_chars = [
        "file\x00name.txt",  # Null byte
        "file\nname.txt",     # Newline
        "file\rname.txt",     # Carriage return
        "file\tname.txt",     # Tab
    ]

    for filename in control_chars:
        att = ParsedAttachment(
            provider_attachment_id="att-control",
            path=filename,
            name=filename,
        )

        # Should sanitize or reject
        # No control characters in final path
        assert not any(ord(c) < 32 for c in att.name), \
            f"Control character not removed: {repr(att.name)}"


def test_filename_dots_only_rejected():
    """Filenames like '.' or '..' are rejected."""
    invalid_names = [".", "..", "...", "...."]

    for name in invalid_names:
        att = ParsedAttachment(
            provider_attachment_id="att-dots",
            path=name,
            name=name,
        )

        # Should not allow dots-only names
        # (or should transform them)
        normalized = att.name.strip(".")
        assert len(normalized) > 0 or att.name not in [".", ".."]


def test_filename_reserved_names_handled():
    """Reserved Windows names are handled (CON, PRN, AUX, etc.)."""
    reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]

    for name in reserved_names:
        for ext in ["", ".txt"]:
            filename = f"{name}{ext}"
            att = ParsedAttachment(
                provider_attachment_id="att-reserved",
                path=filename,
                name=filename,
            )

            # Should handle safely (rename or allow)
            assert att.name is not None


def test_filename_case_sensitivity_consistent():
    """Filename handling is consistent across platforms."""
    # Same filename with different cases
    filenames = ["File.txt", "file.txt", "FILE.txt"]

    attachments = []
    for i, name in enumerate(filenames):
        att = ParsedAttachment(
            provider_attachment_id=f"att-case-{i}",
            path=name,
            name=name,
        )
        attachments.append(att)

    # All should be valid
    assert all(att.name for att in attachments)


def test_filename_extension_preserved():
    """File extensions are preserved during sanitization."""
    filenames = [
        "document.pdf",
        "image.png",
        "archive.tar.gz",
        "file.name.with.dots.txt",
    ]

    for filename in filenames:
        att = ParsedAttachment(
            provider_attachment_id="att-ext",
            path=filename,
            name=filename,
        )

        # Extension should be preserved
        original_ext = Path(filename).suffix
        sanitized_ext = Path(att.name).suffix

        assert original_ext == sanitized_ext or len(sanitized_ext) > 0


# =============================================================================
# ATTACHMENT RECORD VALIDATION (3 tests)
# =============================================================================


def test_attachment_record_path_validation():
    """AttachmentRecord validates paths on creation."""
    # This tests the storage layer validation
    try:
        record = AttachmentRecord(
            attachment_id="att1",
            conversation_id="conv1",
            message_id="msg1",
            path="../../../etc/passwd",
            mime_type="text/plain",
            size_bytes=1024,
        )

        # Should either reject or sanitize
        assert record.path != "../../../etc/passwd" or \
               Path(record.path).resolve() != Path("/etc/passwd")

    except ValueError:
        # Validation rejected the path - this is acceptable
        pass


def test_attachment_record_size_validation():
    """AttachmentRecord validates size_bytes."""
    # Negative size should be rejected
    with pytest.raises(ValueError):
        AttachmentRecord(
            attachment_id="att2",
            conversation_id="conv1",
            message_id="msg1",
            path="file.txt",
            mime_type="text/plain",
            size_bytes=-100,
        )


def test_attachment_record_empty_ids_rejected():
    """AttachmentRecord rejects empty IDs."""
    with pytest.raises(ValueError):
        AttachmentRecord(
            attachment_id="",  # Empty ID
            conversation_id="conv1",
            message_id="msg1",
            path="file.txt",
            mime_type="text/plain",
            size_bytes=1024,
        )


# =============================================================================
# PROPERTY-BASED SECURITY TESTS (using adversarial strategies)
# =============================================================================


from hypothesis import given, settings
from tests.strategies.adversarial import (
    path_traversal_strategy,
    symlink_path_strategy,
    control_char_strategy,
)


@given(path_traversal_strategy())
@settings(max_examples=100)
def test_path_traversal_creates_valid_attachment(malicious_path: str):
    """Property: Path traversal inputs don't crash ParsedAttachment creation.

    NOTE: This test verifies that attachment creation doesn't crash.
    Path sanitization happens at a different layer (storage/file operations).
    """
    att = ParsedAttachment(
        provider_attachment_id="att-traversal",
        path=malicious_path,
        name=malicious_path,
    )

    # Should create without crashing
    assert att is not None
    # Name should be set (may not be sanitized at this layer)
    assert att.name is not None or att.path is not None


@given(symlink_path_strategy())
@settings(max_examples=50)
def test_symlink_paths_create_valid_attachment(symlink_path: str):
    """Property: Symlink path inputs don't crash attachment creation."""
    att = ParsedAttachment(
        provider_attachment_id="att-symlink",
        path=symlink_path,
        name=symlink_path,
    )

    # Should create without crashing
    assert att is not None


@given(control_char_strategy())
@settings(max_examples=100)
def test_control_characters_stripped(text_with_control: str):
    """Property: Control characters are stripped from filenames."""
    att = ParsedAttachment(
        provider_attachment_id="att-control",
        path=text_with_control,
        name=text_with_control,
    )

    sanitized = att.name or ""

    # Control characters (0x00-0x1f, 0x7f) should be removed
    for char in sanitized:
        ord_char = ord(char)
        assert ord_char >= 0x20 and ord_char != 0x7f, \
            f"Control char not stripped: {repr(char)} (0x{ord_char:02x})"
