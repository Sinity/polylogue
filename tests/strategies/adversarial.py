"""Adversarial hypothesis strategies for security testing.

These strategies generate malformed, malicious, or edge-case inputs
for testing security properties and robustness.
"""

from __future__ import annotations

from typing import Any

from hypothesis import strategies as st


# =============================================================================
# Path Traversal Strategies
# =============================================================================


@st.composite
def path_traversal_strategy(draw: st.DrawFn) -> str:
    """Generate path traversal attack strings.

    Tests the path sanitizer in ParsedAttachment.
    """
    traversal_patterns = [
        # Basic traversal
        "../",
        "..\\",
        "../../../",
        "..\\..\\..\\",
        # URL encoded
        "%2e%2e/",
        "%2e%2e%2f",
        "..%2f",
        "%2e%2e\\",
        # Double URL encoded
        "%252e%252e/",
        # Unicode normalization attacks
        "..%c0%af",
        "..%c1%9c",
        # Null byte injection
        "../\x00",
        "file.txt\x00.jpg",
        # Mixed separators
        "..\\../",
        "..\\/",
        # Long paths
        "../" * 20,
        # Absolute paths
        "/etc/passwd",
        "C:\\Windows\\System32",
        # Special directories
        "~/.ssh/id_rsa",
        "$HOME/.bashrc",
        # Hidden files
        ".hidden/../../../etc/passwd",
    ]

    pattern = draw(st.sampled_from(traversal_patterns))

    # Optionally combine with random prefix/suffix
    if draw(st.booleans()):
        prefix = draw(st.text(min_size=0, max_size=10, alphabet="abcdefghijklmnop/\\"))
        pattern = prefix + pattern

    if draw(st.booleans()):
        suffix = draw(st.text(min_size=0, max_size=10, alphabet="abcdefghijklmnop.txt"))
        pattern = pattern + suffix

    return pattern


@st.composite
def symlink_path_strategy(draw: st.DrawFn) -> str:
    """Generate paths that might involve symlinks."""
    return draw(st.sampled_from([
        "/tmp/link/../../../etc/passwd",
        "symlink/../secret",
        "./link/./link/../target",
    ]))


# =============================================================================
# SQL/FTS Injection Strategies
# =============================================================================


@st.composite
def sql_injection_strategy(draw: st.DrawFn) -> str:
    """Generate SQL injection attack strings.

    Tests the FTS5 query escaper and any raw SQL paths.
    """
    injection_patterns = [
        # Basic SQL injection
        "'; DROP TABLE messages; --",
        "' OR '1'='1",
        "' OR 1=1 --",
        "'; DELETE FROM conversations; --",
        "' UNION SELECT * FROM sqlite_master --",
        # FTS5 specific
        "test OR 1=1",
        'test" OR "1"="1',
        "test AND 1=1",
        "* OR MATCH",
        "NEAR(test, 5)",
        # Boolean operators
        "NOT test",
        "test OR test",
        "test AND NOT test",
        # Wildcards
        "*",
        "te*st",
        "test*",
        # Column prefixes (FTS5)
        "text:test",
        "role:user",
        # Special characters
        '"test"',
        "'test'",
        "test; SELECT",
        "test/* comment */",
        # Unicode tricks
        "test\u0000",
        "test\u001f",
    ]

    return draw(st.sampled_from(injection_patterns))


@st.composite
def fts5_operator_strategy(draw: st.DrawFn) -> str:
    """Generate FTS5 operator strings that need escaping."""
    operators = ["AND", "OR", "NOT", "NEAR", "MATCH"]
    return draw(st.sampled_from(operators))


# =============================================================================
# Malformed JSON Strategies
# =============================================================================


@st.composite
def malformed_json_strategy(draw: st.DrawFn) -> str:
    """Generate malformed JSON strings for parser fuzzing."""
    malformed_patterns = [
        # Truncated
        '{"key": "val',
        '{"key": [1, 2, ',
        '{"key": {"nested":',
        # Invalid syntax
        "{'single': 'quotes'}",
        '{key: "no quotes on key"}',
        '{"trailing": "comma",}',
        '{"duplicate": 1, "duplicate": 2}',
        # Type confusion
        '{"number": NaN}',
        '{"number": Infinity}',
        '{"number": -Infinity}',
        # Encoding issues
        '{"unicode": "\\uDEAD"}',  # Invalid surrogate
        '{"unicode": "\\x00"}',  # Invalid escape
        # Deep nesting
        '{"a":' * 100 + '"deep"' + '}' * 100,
        # Large strings
        '{"key": "' + 'x' * 10000 + '"}',
        # Empty/whitespace
        "",
        "   ",
        "\n\n\n",
        # Not JSON
        "null",
        "true",
        "42",
        '"just a string"',
        "undefined",
        "None",
    ]

    return draw(st.sampled_from(malformed_patterns))


@st.composite
def edge_case_json_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate edge-case but valid JSON structures."""
    return draw(st.one_of(
        # Empty containers
        st.just({}),
        st.just([]),
        st.just({"messages": []}),
        # Null values
        st.just({"key": None}),
        st.just({"messages": [None]}),
        # Very long strings
        st.builds(lambda n: {"text": "x" * n}, st.integers(min_value=10000, max_value=100000)),
        # Deep nesting
        st.builds(
            lambda depth: _nested_dict(depth),
            st.integers(min_value=50, max_value=100),
        ),
        # Many keys
        st.builds(
            lambda n: {f"key_{i}": i for i in range(n)},
            st.integers(min_value=100, max_value=1000),
        ),
        # Unicode edge cases
        st.just({"text": "\u0000\u001f\u007f"}),
        st.just({"text": "\U0001F600"}),  # Emoji
        st.just({"text": "\u202e"}),  # RTL override
    ))


def _nested_dict(depth: int) -> dict[str, Any]:
    """Create a deeply nested dict."""
    if depth <= 0:
        return {"leaf": "value"}
    return {"nested": _nested_dict(depth - 1)}


# =============================================================================
# Control Character Strategies
# =============================================================================


@st.composite
def control_char_strategy(draw: st.DrawFn) -> str:
    """Generate strings with control characters."""
    control_chars = [
        "\x00",  # Null
        "\x01",  # SOH
        "\x07",  # Bell
        "\x08",  # Backspace
        "\x0b",  # Vertical tab
        "\x0c",  # Form feed
        "\x1b",  # Escape
        "\x7f",  # DEL
    ]

    base_text = draw(st.text(min_size=1, max_size=20))
    inject_char = draw(st.sampled_from(control_chars))
    position = draw(st.integers(min_value=0, max_value=len(base_text)))

    return base_text[:position] + inject_char + base_text[position:]


# =============================================================================
# DoS / Resource Exhaustion Strategies
# =============================================================================


@st.composite
def regex_dos_strategy(draw: st.DrawFn) -> str:
    """Generate strings that might cause regex DoS (ReDoS).

    Tests timestamp parsing and other regex-based operations.
    """
    # Patterns that might cause backtracking explosion
    return draw(st.sampled_from([
        "a" * 30 + "!",  # Classic ReDoS for (a+)+
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
        "0" * 50 + "x",  # For numeric parsing
        "1" * 20 + "." + "1" * 20 + "." + "1" * 20,  # Version-like
        "2024-01-01" + "-01" * 50,  # Date-like
    ]))


@st.composite
def large_input_strategy(draw: st.DrawFn, max_size: int = 1_000_000) -> str:
    """Generate large inputs for memory/performance testing."""
    size = draw(st.integers(min_value=100_000, max_value=max_size))
    char = draw(st.sampled_from(["a", "0", " ", "\n"]))
    return char * size
