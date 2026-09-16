"""polylogue-lb15e: the agent-facing surface must declare what it clipped.

A bounded projection that hands back a bare ``text[:n]`` is not merely lossy --
it is indistinguishable from a complete short value, so a model reading it has
no way to know it is reasoning about a fragment. Every clip on this surface
carries the shared marker plus the untruncated length and an explicit flag.
"""

from __future__ import annotations

from polylogue.mcp.archive_support import TRUNCATION_MARKER, clip_with_marker


def test_clip_with_marker_declares_the_clip_and_respects_the_budget() -> None:
    """Anti-vacuity: returning ``text[:max_chars], False`` fails the flag and
    marker assertions; dropping the budget arithmetic fails the length bound.
    """
    clipped, truncated = clip_with_marker("a" * 500, 300)
    assert truncated is True
    assert clipped.endswith(TRUNCATION_MARKER)
    assert len(clipped) == 300, "the marker must fit inside the budget, not extend it"


def test_clip_with_marker_leaves_a_complete_value_untouched() -> None:
    """A value inside the budget must not gain a marker or a truthy flag."""
    assert clip_with_marker("short", 300) == ("short", False)
    assert clip_with_marker("exact", 5) == ("exact", False)


def test_fenced_code_blocks_report_their_real_length_and_truncation() -> None:
    """An agent may reason about or run this code, so a fragment must say so.

    Anti-vacuity: restoring ``{"language": ..., "code": code[:300]}`` drops both
    new keys and makes this red.
    """
    from polylogue.mcp.server_support import _extract_fenced_code

    long_code = "x" * 500
    blocks = _extract_fenced_code(f"intro\n```py\n{long_code}\n```\ntail")
    assert len(blocks) == 1
    block = blocks[0]
    assert block["truncated"] is True
    assert block["code_length"] == len(long_code) + 1  # the fence's trailing newline
    assert block["code"].endswith(TRUNCATION_MARKER)


def test_short_fenced_code_block_is_not_marked_truncated() -> None:
    """A complete snippet must be reported as complete, or the flag is noise."""
    from polylogue.mcp.server_support import _extract_fenced_code

    blocks = _extract_fenced_code("```py\nprint(1)\n```")
    assert len(blocks) == 1
    assert blocks[0]["truncated"] is False
    assert TRUNCATION_MARKER not in blocks[0]["code"]
