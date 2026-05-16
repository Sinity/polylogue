"""Path-traversal boundary properties for `polylogue.core.security.sanitize_path`.

`sanitize_path` is the single chokepoint used by ingestion record validators
(`sources/parsers/base_models.py`, `storage/runtime/archive/records.py`) to
neutralize attacker-controlled path strings before they reach the
filesystem. The function's contract:

* never raise on arbitrary text
* never return an absolute path
* never return a path containing `..`
* never return a path containing a NUL byte or ASCII control byte
* return ``_blocked_<sha256-prefix>`` (a content-addressed scrub token) when
  the input contains traversal segments, symlinks in any parent component,
  or starts with `/`
* keep already-safe inputs as plain relative POSIX-style paths

These property tests hammer the function with adversarial strings
(traversal, absolute, unicode normalization, NUL, control bytes, symlink
parents) and assert the contract above. A regression in the sanitizer or a
new bypass technique should make at least one property fail.
"""

from __future__ import annotations

import os
from pathlib import Path
from string import printable

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from polylogue.core.security import sanitize_path

_TRAVERSAL_SEGMENTS = (
    "..",
    "../",
    "..\\",
    "../../",
    "../../../etc/passwd",
    "safe/../../etc/passwd",
    "%2e%2e/%2e%2e/etc/passwd",  # URL-encoded; sanitizer should NOT decode
    "..%2fetc%2fpasswd",
    "....//",
    "..;/",
)

_ABSOLUTE_PREFIXES = ("/", "//", "/etc/passwd", "/var/log/auth.log", "//server/share")

_CONTROL_AND_NULL = (
    "foo\x00bar",
    "\x00",
    "foo\x01bar",
    "foo\x7fbar",
    "foo\nbar",
    "foo\rbar",
    "\x00../etc/passwd",
)

_UNICODE_TRICKS = (
    "fo\u202eo/bar",
    "fo\u2066o/bar",
    "fo\u00a0o/bar",  # NBSP — should not crash, may or may not survive trim
    "ＡＢＣ/file",  # full-width characters
    "fo\u0301o",  # combining acute accent (NFD)
)


_ALL_HOSTILE = _TRAVERSAL_SEGMENTS + _ABSOLUTE_PREFIXES + _CONTROL_AND_NULL + _UNICODE_TRICKS


hostile_path_strategy: st.SearchStrategy[str] = st.one_of(
    st.sampled_from(_ALL_HOSTILE),
    st.text(
        alphabet=st.characters(blacklist_categories=["Cs"]),
        min_size=0,
        max_size=200,
    ),
    st.lists(st.sampled_from(_ALL_HOSTILE), min_size=1, max_size=4).map("/".join),
)


@settings(max_examples=120, deadline=None)
@given(raw=hostile_path_strategy)
def test_sanitize_path_never_raises(raw: str) -> None:
    """`sanitize_path` is total: never raises for any string."""
    sanitize_path(raw)


@settings(max_examples=120, deadline=None)
@given(raw=hostile_path_strategy)
def test_sanitize_path_returns_safe_shape(raw: str) -> None:
    """Output is None, the input unchanged-but-safe, or a `_blocked_` token."""
    result = sanitize_path(raw)
    if result is None:
        return
    assert isinstance(result, str)
    assert "\x00" not in result, f"NUL byte leaked through for {raw!r}: {result!r}"
    for ch in result:
        assert ord(ch) >= 32 and ord(ch) != 127, f"Control byte 0x{ord(ch):02x} leaked through for {raw!r}: {result!r}"
    if result.startswith("_blocked_"):
        # _blocked_<12 hex chars>
        suffix = result[len("_blocked_") :]
        assert len(suffix) == 12 and all(c in "0123456789abcdef" for c in suffix), (
            f"Malformed _blocked_ token for {raw!r}: {result!r}"
        )
        return
    assert not result.startswith("/"), f"Absolute path leaked through for {raw!r}: {result!r}"
    assert ".." not in result.split("/"), f"Traversal segment in result {result!r} for {raw!r}"


@pytest.mark.parametrize("raw", _TRAVERSAL_SEGMENTS)
def test_known_traversal_payloads_are_blocked(raw: str) -> None:
    result = sanitize_path(raw)
    assert result is not None
    assert result.startswith("_blocked_") or ".." not in result.split("/"), (
        f"{raw!r} produced {result!r} — traversal segment survived"
    )


@pytest.mark.parametrize("raw", _ABSOLUTE_PREFIXES)
def test_absolute_paths_are_blocked(raw: str) -> None:
    result = sanitize_path(raw)
    assert result is not None
    assert result.startswith("_blocked_"), f"Absolute path {raw!r} produced {result!r} (expected _blocked_ token)"


@pytest.mark.parametrize("raw", _CONTROL_AND_NULL)
def test_null_and_control_bytes_stripped_or_blocked(raw: str) -> None:
    result = sanitize_path(raw)
    if result is None:
        return
    assert "\x00" not in result
    for ch in result:
        assert ord(ch) >= 32 and ord(ch) != 127


def test_symlink_parent_is_blocked(tmp_path: Path) -> None:
    """If any parent component of the input resolves to a symlink, block it.

    This mirrors the sanitizer's symlink guard: even a non-traversal path
    that happens to live below a symlinked directory must not be trusted.
    """
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    sym_link = tmp_path / "link"
    try:
        os.symlink(real_dir, sym_link)
    except OSError:
        pytest.skip("symlink creation not permitted in this environment")

    candidate = str(sym_link / "child" / "file.txt")
    result = sanitize_path(candidate)
    assert result is not None
    # The sanitizer also blocks absolute paths, which covers this case.
    assert result.startswith("_blocked_"), f"Path under symlinked parent should be _blocked_, got {result!r}"


@given(
    raw=st.text(alphabet=printable, min_size=1, max_size=40).filter(
        lambda s: ".." not in s and "\x00" not in s and not s.startswith("/")
    )
)
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
def test_clean_relative_paths_pass_through(raw: str) -> None:
    """A clean relative path with no traversal/symlink must not be blocked."""
    assume(not any(ord(c) < 32 or ord(c) == 127 for c in raw))
    result = sanitize_path(raw)
    if result is None:
        return
    assert not result.startswith("_blocked_"), f"Clean input {raw!r} got scrubbed to {result!r}"
