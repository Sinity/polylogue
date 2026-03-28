"""Attachment and filename security contracts at the parser boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings

from polylogue.sources.parsers.base import ParsedAttachment
from tests.infra.adversarial_cases import (
    CONTROL_CHAR_FILENAMES,
    DOTS_ONLY_FILENAMES,
    PATH_TRAVERSAL_STRINGS,
    RESERVED_FILENAMES,
)
from tests.infra.strategies.adversarial import (
    control_char_strategy,
    path_traversal_strategy,
    symlink_path_strategy,
)


def test_attachment_path_traversal_rejected() -> None:
    att = ParsedAttachment(
        provider_attachment_id="att1",
        path="../../../etc/passwd",
        name="passwd",
    )
    normalized = Path(att.path).resolve()
    assert not str(normalized).endswith("/etc/passwd")


def test_attachment_absolute_path_preserved() -> None:
    att = ParsedAttachment(
        provider_attachment_id="att2",
        path="/etc/shadow",
        name="shadow",
    )
    assert att.path == "/etc/shadow"


def test_attachment_path_null_byte_rejected() -> None:
    att = ParsedAttachment(
        provider_attachment_id="att3",
        path="safe_file\x00../../etc/passwd",
        name="exploit",
    )
    assert "\x00" not in att.path


@pytest.mark.parametrize("path", ["file with spaces.txt", "file:with:colons.txt", "file*with*wildcards.txt", "file|with|pipes.txt"])
def test_attachment_path_special_characters(path: str) -> None:
    att = ParsedAttachment(
        provider_attachment_id="att-special",
        path=path,
        name=Path(path).name,
    )
    assert att.path is not None
    assert len(att.path) > 0


def test_attachment_path_very_long() -> None:
    long_name = "a" * 300 + ".txt"
    att = ParsedAttachment(
        provider_attachment_id="att-long",
        path=long_name,
        name=long_name,
    )
    assert att.path is not None


@pytest.mark.parametrize("path", ["файл.txt", "文件.txt", "🎉.txt"])
def test_attachment_path_unicode(path: str) -> None:
    att = ParsedAttachment(
        provider_attachment_id="att-unicode",
        path=path,
        name=path,
    )
    assert att.path is not None


@pytest.mark.parametrize("filename", CONTROL_CHAR_FILENAMES)
def test_filename_control_characters_removed(filename: str) -> None:
    att = ParsedAttachment(
        provider_attachment_id="att-control",
        path=filename,
        name=filename,
    )
    assert not any(ord(c) < 32 for c in att.name)


@pytest.mark.parametrize("name", DOTS_ONLY_FILENAMES)
def test_filename_dots_only_rejected(name: str) -> None:
    att = ParsedAttachment(
        provider_attachment_id="att-dots",
        path=name,
        name=name,
    )
    assert att.name == "file"


@pytest.mark.parametrize("name", RESERVED_FILENAMES)
def test_filename_reserved_names_handled(name: str) -> None:
    for ext in ["", ".txt"]:
        filename = f"{name}{ext}"
        att = ParsedAttachment(
            provider_attachment_id="att-reserved",
            path=filename,
            name=filename,
        )
        assert att.name is not None


def test_filename_case_sensitivity_consistent() -> None:
    attachments = [
        ParsedAttachment(provider_attachment_id=f"att-case-{i}", path=name, name=name)
        for i, name in enumerate(["File.txt", "file.txt", "FILE.txt"])
    ]
    assert all(att.name for att in attachments)


@pytest.mark.parametrize(
    "filename",
    ["document.pdf", "image.png", "archive.tar.gz", "file.name.with.dots.txt"],
)
def test_filename_extension_preserved(filename: str) -> None:
    att = ParsedAttachment(
        provider_attachment_id="att-ext",
        path=filename,
        name=filename,
    )
    assert Path(att.name).suffix or Path(filename).suffix == Path(att.name).suffix


@pytest.mark.parametrize("path", PATH_TRAVERSAL_STRINGS)
def test_known_path_traversal_inputs_do_not_crash(path: str) -> None:
    att = ParsedAttachment(
        provider_attachment_id="att-known-traversal",
        path=path,
        name=path,
    )
    assert att is not None
    assert att.name is not None or att.path is not None


@given(path_traversal_strategy())
@settings(max_examples=100)
def test_path_traversal_creates_valid_attachment(malicious_path: str) -> None:
    att = ParsedAttachment(
        provider_attachment_id="att-traversal",
        path=malicious_path,
        name=malicious_path,
    )
    assert att is not None
    assert att.name is not None or att.path is not None


@given(symlink_path_strategy())
@settings(max_examples=50)
def test_symlink_paths_create_valid_attachment(symlink_path: str) -> None:
    att = ParsedAttachment(
        provider_attachment_id="att-symlink",
        path=symlink_path,
        name=symlink_path,
    )
    assert att is not None


@given(control_char_strategy())
@settings(max_examples=100)
def test_control_characters_stripped(text_with_control: str) -> None:
    att = ParsedAttachment(
        provider_attachment_id="att-control-prop",
        path=text_with_control,
        name=text_with_control,
    )
    sanitized = att.name or ""
    for char in sanitized:
        ord_char = ord(char)
        assert ord_char >= 0x20 and ord_char != 0x7F
