"""Tests for resolve_paste_boundary_state and the paste boundary evidence model.

Covers the decision matrix in ``polylogue/archive/message/paste_detection.py``
that determines whether paste-span boundaries are exact (content-level
recoverable), projected (marker-inferred), hash-only (pastedContents recorded
but content unrecoverable), whole_message_fallback (heuristic), or absent.
"""

from __future__ import annotations

from polylogue.archive.message.paste_detection import (
    _PASTE_BOUNDARY_STATES,
    detect_paste,
    resolve_paste_boundary_state,
)


class TestPasteBoundaryStateVocabulary:
    """The closed vocabulary of boundary states is well-defined."""

    def test_vocabulary_is_closed(self) -> None:
        assert frozenset({"exact", "projected", "whole_message_fallback", "hash_only"}) == _PASTE_BOUNDARY_STATES


class TestResolvePasteBoundaryState:
    """Unit tests for resolve_paste_boundary_state decision matrix."""

    # ------------------------------------------------------------------
    # Exact — history with content
    # ------------------------------------------------------------------

    def test_all_three_sources_agree_exact(self) -> None:
        """When history has paste + content, exact always wins regardless
        of hook and text heuristics."""
        state = resolve_paste_boundary_state(
            message_text="[Pasted text #1] long content",
            history_has_paste=True,
            history_has_content=True,
            hook_has_paste=True,
        )
        assert state == "exact"

    def test_history_only_exact(self) -> None:
        """History with paste + content alone (no hook, no heuristic text)
        still returns exact."""
        state = resolve_paste_boundary_state(
            message_text="normal typed message",
            history_has_paste=True,
            history_has_content=True,
            hook_has_paste=False,
        )
        assert state == "exact"

    def test_history_exact_trumps_hook(self) -> None:
        """Even when hook_has_paste is also True, history exact wins."""
        state = resolve_paste_boundary_state(
            message_text=None,
            history_has_paste=True,
            history_has_content=True,
            hook_has_paste=True,
        )
        assert state == "exact"

    # ------------------------------------------------------------------
    # Projected — hook markers only (no history content)
    # ------------------------------------------------------------------

    def test_hook_only_projected(self) -> None:
        """Hook paste marker with no history content → projected."""
        state = resolve_paste_boundary_state(
            message_text=None,
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=True,
        )
        assert state == "projected"

    def test_hook_projected_even_with_normal_text(self) -> None:
        """Hook paste still returns projected when message text is ordinary
        (does not itself match heuristics)."""
        state = resolve_paste_boundary_state(
            message_text="short typed message",
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=True,
        )
        assert state == "projected"

    def test_hook_trumps_heuristic_only(self) -> None:
        """Hook paste marker takes priority over heuristic text patterns."""
        state = resolve_paste_boundary_state(
            message_text="x" * 5000,  # would be whole_message_fallback by itself
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=True,
        )
        assert state == "projected"

    # ------------------------------------------------------------------
    # Hash-only — history asserts paste but content is unrecoverable
    # ------------------------------------------------------------------

    def test_hash_only(self) -> None:
        """History records paste but no content → hash_only."""
        state = resolve_paste_boundary_state(
            message_text="normal message",
            history_has_paste=True,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state == "hash_only"

    def test_hash_only_with_normal_text(self) -> None:
        """Hash-only even when message text is not pasted (the authoritative
        history source says paste existed, even if we can't see it)."""
        state = resolve_paste_boundary_state(
            message_text=None,
            history_has_paste=True,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state == "hash_only"

    # ------------------------------------------------------------------
    # Whole-message fallback — heuristic text patterns only
    # ------------------------------------------------------------------

    def test_heuristic_only_long_text(self) -> None:
        """Text-only paste detection without hook or history → whole_message_fallback."""
        state = resolve_paste_boundary_state(
            message_text="x" * 5000,
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state == "whole_message_fallback"

    def test_in_text_paste_marker_is_projected(self) -> None:
        """Text containing a ground-truth [Pasted text #N] marker without
        hook/history evidence resolves to projected (known boundary), not the
        weaker whole_message_fallback proxy state."""
        state = resolve_paste_boundary_state(
            message_text="Look at [Pasted text #1] for details",
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state == "projected"

    def test_heuristic_only_forwarding_pattern(self) -> None:
        """Text matching chatlog forwarding without hook/history → whole_message_fallback."""
        state = resolve_paste_boundary_state(
            message_text="previous chatlog attached below\n\n...",
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state == "whole_message_fallback"

    def test_heuristic_only_code_fence_dominant(self) -> None:
        """Code-fence-dominant text without hook/history → whole_message_fallback."""
        code = "Here is the code:\n```\n" + ("x" * 800) + "\n```\n"
        state = resolve_paste_boundary_state(
            message_text=code,
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state == "whole_message_fallback"

    # ------------------------------------------------------------------
    # No evidence
    # ------------------------------------------------------------------

    def test_no_evidence_any_source(self) -> None:
        """When all sources are clean, returns None."""
        state = resolve_paste_boundary_state(
            message_text="Just a normal question about the codebase.",
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state is None

    def test_no_evidence_none_text(self) -> None:
        """None text with no other evidence → None."""
        state = resolve_paste_boundary_state(
            message_text=None,
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state is None

    def test_no_evidence_empty_text(self) -> None:
        """Empty string text with no other evidence → None."""
        state = resolve_paste_boundary_state(
            message_text="",
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state is None

    # ------------------------------------------------------------------
    # Priority chain verified
    # ------------------------------------------------------------------

    def test_priority_exact_over_hash_only(self) -> None:
        """History exact always beats history hash-only (content availability
        is the tie-breaker)."""
        state = resolve_paste_boundary_state(
            message_text=None,
            history_has_paste=True,
            history_has_content=True,
            hook_has_paste=False,
        )
        assert state == "exact"

    def test_priority_hook_over_hash_only(self) -> None:
        """Hook projected beats hash-only (marker location is finer-grained
        than a content-absent history assertion)."""
        state = resolve_paste_boundary_state(
            message_text=None,
            history_has_paste=True,
            history_has_content=False,
            hook_has_paste=True,
        )
        assert state == "projected"

    def test_priority_hash_over_heuristic(self) -> None:
        """Hash-only beats whole_message_fallback (history assertion is more
        authoritative than text heuristics)."""
        state = resolve_paste_boundary_state(
            message_text="x" * 5000,
            history_has_paste=True,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state == "hash_only"


class TestDetectPasteIntegration:
    """Sanity checks that detect_paste integrates with the boundary resolver
    as expected (caller-side contract)."""

    def test_detect_paste_returns_zero_for_normal_text(self) -> None:
        assert detect_paste("How do I fix this test?") == 0

    def test_detect_paste_returns_one_for_long_text(self) -> None:
        assert detect_paste("x" * 5000) == 1

    def test_resolver_uses_heuristic_for_whole_message_fallback(self) -> None:
        """Resolver returns whole_message_fallback for a heuristic-only proxy."""
        state = resolve_paste_boundary_state(
            message_text="x" * 5000,
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state == "whole_message_fallback"

    def test_resolver_uses_marker_for_projected(self) -> None:
        """A ground-truth paste marker in text resolves to projected, not the
        weaker whole_message_fallback proxy state."""
        state = resolve_paste_boundary_state(
            message_text="see [Pasted text #1] above",
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state == "projected"

    def test_resolver_returns_none_when_detect_paste_returns_zero(self) -> None:
        """Resolver returns None when detect_paste returns 0 and no other evidence."""
        state = resolve_paste_boundary_state(
            message_text="short message",
            history_has_paste=False,
            history_has_content=False,
            hook_has_paste=False,
        )
        assert state is None
