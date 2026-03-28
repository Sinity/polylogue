"""Shared assertion helpers for rendered and textual test output."""


def assert_messages_ordered(markdown_text: str, *expected_order: str) -> None:
    """Assert messages appear in the given order in rendered output."""
    indices = []
    for text in expected_order:
        try:
            idx = markdown_text.index(text)
            indices.append((idx, text))
        except ValueError as exc:
            raise AssertionError(f"Expected text '{text}' not found in markdown") from exc

    for i in range(len(indices) - 1):
        if indices[i][0] >= indices[i + 1][0]:
            raise AssertionError(
                f"Order violation: '{indices[i][1]}' (index {indices[i][0]}) "
                f"should come before '{indices[i + 1][1]}' (index {indices[i + 1][0]})"
            )


def assert_contains_all(text: str, *expected: str) -> None:
    """Assert text contains all expected substrings."""
    for expected_text in expected:
        assert expected_text in text, f"Expected '{expected_text}' not found in text"


def assert_not_contains_any(text: str, *unexpected: str) -> None:
    """Assert text does not contain any of the given substrings."""
    for unexpected_text in unexpected:
        assert unexpected_text not in text, f"Unexpected '{unexpected_text}' found in text"
