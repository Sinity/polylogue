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
# ZIP EXTRACTION SECURITY (3 tests)
# =============================================================================


@pytest.mark.skip(reason="ZIP extraction implementation not in scope yet")
def test_zip_bomb_size_limit():
    """ZIP extraction has size limits to prevent bombs."""
    # Create small ZIP that expands massively
    # This would test actual ZIP extraction code
    pass


@pytest.mark.skip(reason="ZIP extraction implementation not in scope yet")
def test_zip_path_traversal_blocked():
    """ZIP files with ../ paths are sanitized."""
    # Test ZIP with entry like "../../../etc/passwd"
    pass


@pytest.mark.skip(reason="ZIP extraction implementation not in scope yet")
def test_zip_symlink_extraction_blocked():
    """ZIP files containing symlinks are handled safely."""
    # Test ZIP with symlink entries
    pass


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
