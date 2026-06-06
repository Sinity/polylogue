"""Regression coverage for path-extraction heuristics in ToolCall (#1622).

The naive shape filter accepted any token with a slash as a path,
which surfaced sed expressions, Python attribute access, email
addresses in angle brackets, and version numbers in
``ToolCall.affected_paths``. Tighten the filter so non-path tokens
are rejected up front.
"""

from __future__ import annotations

import pytest

from polylogue.archive.viewport.models import ToolCall
from polylogue.archive.viewport.tools import (
    clean_metadata_path_candidate,
    clean_path_candidate,
    looks_like_path_candidate,
)


@pytest.mark.parametrize(
    "value",
    [
        "/dst_provider_name/!s/provider_name/source_name/g",  # sed expression
        "s/old/new/g",  # bare sed
        "!s/foo/bar/",  # negated sed
    ],
)
def test_clean_path_candidate_rejects_sed_substitution(value: str) -> None:
    assert clean_path_candidate(value) is None


@pytest.mark.parametrize(
    "value",
    [
        "<noreply@anthropic.com>",
        "<placeholder>",
        "{template}",
    ],
)
def test_clean_path_candidate_rejects_angle_brace_wrappers(value: str) -> None:
    assert clean_path_candidate(value) is None


@pytest.mark.parametrize(
    "value",
    [
        "4.7",  # version number — `.7` is not an extension shape
        "<noreply@anthropic.com>",  # rejected upstream by clean_path_candidate
        "insight.provider_name",  # Python attribute access — long suffix
        "kwargs/fields",  # slash but no extension
        "dataclass/function",  # same
        "/dst_provider_name/!s/provider_name/source_name/g",  # sed
    ],
)
def test_looks_like_path_candidate_rejects_known_false_positives(value: str) -> None:
    assert not looks_like_path_candidate(value)


@pytest.mark.parametrize(
    "value",
    [
        "/realm/project/polylogue/file.py",
        "./file.py",
        "../file.txt",
        "README.md",
        "Makefile",
        "polylogue/archive.py",
        "data/raw.json",
        "/etc/hosts",  # absolute, no extension — still a path
        "~/.local/share/polylogue/index.db",
    ],
)
def test_looks_like_path_candidate_accepts_real_paths(value: str) -> None:
    assert looks_like_path_candidate(value)


def test_tool_call_affected_paths_filters_input_field_junk() -> None:
    """ToolCall.affected_paths applies the shape filter to input fields too."""
    call = ToolCall(
        name="custom-tool",
        input={
            "file_path": "4.7",  # model version — not a path
            "path": "kwargs/fields",  # Python attr — not a path
            "file": "/realm/project/polylogue/file.py",  # real path
        },
    )
    assert call.affected_paths == ["/realm/project/polylogue/file.py"]


def test_clean_metadata_path_candidate_consistency() -> None:
    """The metadata-path helper composes the cleaner and shape filter."""
    assert clean_metadata_path_candidate("README.md") == "README.md"
    assert clean_metadata_path_candidate("kwargs/fields") is None
    assert clean_metadata_path_candidate("4.7") is None
