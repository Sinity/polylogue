"""Unit tests for the reader compare diff logic (#1124).

Exercises :mod:`polylogue.daemon.compare` directly with synthetic reader
payload dicts so the diff behaviour is pinned independently of the daemon
HTTP path. The HTTP smoke is covered separately in ``test_web_reader``.
"""

from __future__ import annotations

from typing import Any

import pytest

from polylogue.daemon import compare


def _make_msg(
    msg_id: str,
    role: str,
    text: str,
    *,
    anchor: str | None = None,
) -> dict[str, Any]:
    return {
        "id": msg_id,
        "role": role,
        "text": text,
        "anchor": anchor if anchor is not None else f"message-{msg_id}",
    }


def _make_payload(
    conv_id: str,
    *,
    origin: str = "claude-code-session",
    model: str | None = "claude-opus",
    title: str = "Session",
    repo: str | None = None,
    tags: list[str] | None = None,
    messages: list[dict[str, Any]] | None = None,
    word_count: int = 0,
) -> dict[str, Any]:
    return {
        "id": conv_id,
        "title": title,
        "origin": origin,
        "model": model,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-02T00:00:00",
        "message_count": len(messages or []),
        "word_count": word_count,
        "repo": repo,
        "branch_type": "main",
        "session_id": None,
        "tags": tags or [],
        "messages": messages or [],
    }


class TestMetadataDiff:
    def test_equal_fields_marked_equal(self) -> None:
        left = _make_payload("c1")
        right = _make_payload("c2")
        diff = compare.build_metadata_diff(left, right)
        assert diff["origin"]["status"] == "equal"
        assert diff["model"]["status"] == "equal"

    def test_changed_field_records_both_sides(self) -> None:
        left = _make_payload("c1", model="claude-opus")
        right = _make_payload("c2", model="claude-sonnet")
        diff = compare.build_metadata_diff(left, right)
        assert diff["model"] == {
            "left": "claude-opus",
            "right": "claude-sonnet",
            "status": "changed",
        }

    def test_missing_one_side_marked_missing(self) -> None:
        left = _make_payload("c1", repo="acme/widget")
        right = _make_payload("c2", repo=None)
        diff = compare.build_metadata_diff(left, right)
        assert diff["repo"]["status"] == "missing"
        assert diff["repo"]["left"] == "acme/widget"
        assert diff["repo"]["right"] is None

    def test_both_sides_absent_field_omitted(self) -> None:
        left = _make_payload("c1", repo=None)
        right = _make_payload("c2", repo=None)
        diff = compare.build_metadata_diff(left, right)
        assert "repo" not in diff

    def test_tag_order_does_not_matter_for_status(self) -> None:
        left = _make_payload("c1", tags=["b", "a"])
        right = _make_payload("c2", tags=["a", "b"])
        diff = compare.build_metadata_diff(left, right)
        assert diff["tags"]["status"] == "equal"

    def test_tag_membership_differs_changed(self) -> None:
        left = _make_payload("c1", tags=["a", "b"])
        right = _make_payload("c2", tags=["a", "c"])
        diff = compare.build_metadata_diff(left, right)
        assert diff["tags"]["status"] == "changed"


class TestAlignment:
    def test_sequential_when_no_shared_anchors(self) -> None:
        left = _make_payload(
            "c1",
            messages=[_make_msg("m1", "user", "hi"), _make_msg("m2", "assistant", "hello")],
        )
        right = _make_payload(
            "c2",
            messages=[
                _make_msg("n1", "user", "different prompt"),
                _make_msg("n2", "assistant", "different reply"),
                _make_msg("n3", "user", "follow-up"),
            ],
        )
        alignment, pairs = compare.align_messages(left, right)
        assert alignment == "sequential"
        assert len(pairs) == 3
        # Third pair only has a right side (added) since left ran out.
        assert pairs[2]["diff_status"] == "added"
        assert pairs[2]["left"] is None
        assert pairs[2]["right"]["id"] == "n3"

    def test_anchor_aligned_when_shared_messages_present(self) -> None:
        shared = _make_msg("shared", "user", "common prompt", anchor="anchor-shared")
        left = _make_payload(
            "c1",
            messages=[
                shared,
                _make_msg("l1", "assistant", "left response", anchor="anchor-l1"),
            ],
        )
        right = _make_payload(
            "c2",
            messages=[
                shared,
                _make_msg("r1", "assistant", "right response", anchor="anchor-r1"),
                _make_msg("r2", "user", "extra", anchor="anchor-r2"),
            ],
        )
        alignment, pairs = compare.align_messages(left, right)
        assert alignment == "anchor"
        # First pair is the shared anchor → both sides present, equal text.
        assert pairs[0]["diff_status"] == "equal"
        assert pairs[0]["left"]["id"] == "shared"
        assert pairs[0]["right"]["id"] == "shared"

    def test_changed_status_when_text_differs(self) -> None:
        left = _make_payload(
            "c1",
            messages=[_make_msg("m1", "user", "version A")],
        )
        right = _make_payload(
            "c2",
            messages=[_make_msg("m1", "user", "version B")],
        )
        alignment, pairs = compare.align_messages(left, right)
        # Same anchor "message-m1" → anchor alignment, content differs.
        assert alignment == "anchor"
        assert pairs[0]["diff_status"] == "changed"
        assert pairs[0]["role_match"] is True

    def test_added_and_removed_status(self) -> None:
        left = _make_payload(
            "c1",
            messages=[_make_msg("m1", "user", "hi")],
        )
        right = _make_payload(
            "c2",
            messages=[_make_msg("m1", "user", "hi"), _make_msg("m2", "assistant", "added")],
        )
        alignment, pairs = compare.align_messages(left, right)
        assert pairs[0]["diff_status"] == "equal"
        assert pairs[1]["diff_status"] == "added"

    def test_handles_missing_side(self) -> None:
        right = _make_payload("c2", messages=[_make_msg("m1", "user", "hi")])
        alignment, pairs = compare.align_messages(None, right)
        # No shared anchors possible — sequential fallback with left=None pairs.
        assert alignment == "sequential"
        assert pairs[0]["left"] is None
        assert pairs[0]["right"]["id"] == "m1"
        assert pairs[0]["diff_status"] == "added"


class TestEnvelope:
    def test_envelope_carries_alignment_and_metadata_diff(self) -> None:
        left = _make_payload(
            "c1",
            model="claude-opus",
            messages=[_make_msg("m1", "user", "hello")],
        )
        right = _make_payload(
            "c2",
            model="claude-sonnet",
            messages=[_make_msg("m1", "user", "hello")],
        )
        env = compare.build_compare_envelope(left, right, "c1", "c2", "prompt")
        assert env["mode"] == "compare"
        assert env["align"] == "prompt"
        assert env["alignment"] == "anchor"
        assert env["degraded_count"] == 0
        assert env["degraded_sides"] == []
        assert env["metadata_diff"]["model"]["status"] == "changed"
        assert env["pairs"][0]["diff_status"] == "equal"

    def test_envelope_marks_degraded_sides(self) -> None:
        right = _make_payload("c2", messages=[_make_msg("m1", "user", "hi")])
        env = compare.build_compare_envelope(None, right, "missing", "c2", "prompt")
        assert env["degraded_count"] == 1
        assert env["degraded_sides"] == ["left"]
        assert env["metadata_diff"] == {}
        # Missing-side placeholder uses the missing_session_target shape.
        assert env["left"]["status"] == "missing"
        assert env["left"]["session_id"] == "missing"

    def test_envelope_both_sides_missing(self) -> None:
        env = compare.build_compare_envelope(None, None, "x", "y", "prompt")
        assert env["degraded_count"] == 2
        assert env["degraded_sides"] == ["left", "right"]
        assert env["pairs"] == []
        assert env["total"] == 0

    @pytest.mark.parametrize("align", sorted(compare.COMPARE_ALIGN_MODES))
    def test_envelope_records_align_value(self, align: str) -> None:
        env = compare.build_compare_envelope(_make_payload("c1"), _make_payload("c2"), "c1", "c2", align)
        assert env["align"] == align
