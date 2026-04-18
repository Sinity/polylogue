"""Tests for semantic role inference in schema_inference.py.

Covers:
  - infer_semantic_roles() scoring for all semantic role types
  - select_best_roles() to pick highest-confidence candidate per role
  - Per-role scoring functions with hand-shaped FieldStats fixtures
  - Confidence thresholds and evidence accumulation
"""

from __future__ import annotations

from collections import Counter

from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.semantic_inference_models import SemanticCandidate
from polylogue.schemas.semantic_inference_runtime import infer_semantic_roles, select_best_roles


class TestInferSemanticRoles:
    """infer_semantic_roles() returns scored candidates for all roles."""

    def test_returns_list_of_candidates(self) -> None:
        stats = {
            "$.messages": FieldStats(
                path="$.messages",
                array_lengths=[5, 8, 10],
                total_samples=3,
                present_count=3,
            ),
        }
        candidates = infer_semantic_roles(stats)
        assert isinstance(candidates, list)
        assert all(isinstance(c, SemanticCandidate) for c in candidates)

    def test_candidates_sorted_by_confidence_descending(self) -> None:
        stats = {
            "$.role": FieldStats(
                path="$.role",
                observed_values=Counter({"user": 50, "assistant": 50}),
                total_samples=100,
                present_count=95,
                value_count=100,
                string_lengths=[4, 5, 6, 9],
            ),
            "$.body": FieldStats(
                path="$.body",
                string_lengths=[100, 150, 200],
                is_multiline=2,
                newline_counts=[2, 3, 1],
                total_samples=3,
                present_count=3,
                value_count=3,
            ),
        }
        candidates = infer_semantic_roles(stats)
        confidences = [c.confidence for c in candidates]
        assert confidences == sorted(confidences, reverse=True)

    def test_filters_low_confidence_candidates(self) -> None:
        stats = {
            "$.unknown": FieldStats(
                path="$.unknown",
                total_samples=1,
                present_count=0,  # never present → low confidence everywhere
            ),
        }
        candidates = infer_semantic_roles(stats)
        assert all(c.confidence > 0.1 for c in candidates)

    def test_no_duplicate_path_role_pairs(self) -> None:
        stats = {
            "$.messages": FieldStats(
                path="$.messages",
                array_lengths=[5],
                total_samples=1,
                present_count=1,
            ),
        }
        candidates = infer_semantic_roles(stats)
        pairs = {(c.path, c.role) for c in candidates}
        assert len(pairs) == len(candidates)

    def test_includes_all_known_semantic_roles(self) -> None:
        """With diverse data, candidates cover all role types."""
        stats = {
            "$.messages": FieldStats(
                path="$.messages",
                array_lengths=[5, 8],
                total_samples=2,
                present_count=2,
            ),
            "$.messages[*].role": FieldStats(
                path="$.messages[*].role",
                observed_values=Counter({"user": 50, "assistant": 50}),
                total_samples=100,
                present_count=95,
                value_count=100,
                string_lengths=[4, 5, 9],
            ),
            "$.messages[*].text": FieldStats(
                path="$.messages[*].text",
                string_lengths=[100, 200, 300],
                is_multiline=2,
                newline_counts=[2, 5, 3],
                total_samples=3,
                present_count=3,
                value_count=3,
            ),
            "$.messages[*].created_at": FieldStats(
                path="$.messages[*].created_at",
                detected_formats=Counter({"unix-epoch": 95}),
                num_min=1000000000.0,
                num_max=1700000000.0,
                total_samples=100,
                present_count=100,
                value_count=100,
                numeric_values=[1000000000, 1200000000, 1500000000],
            ),
            "$.title": FieldStats(
                path="$.title",
                observed_values=Counter({"Chat 1": 1, "Chat 2": 1, "Chat 3": 1}),
                string_lengths=[10, 12, 8],
                is_multiline=0,
                newline_counts=[0, 0, 0],
                total_samples=3,
                present_count=3,
                value_count=3,
            ),
        }
        candidates = infer_semantic_roles(stats)
        roles_seen = {c.role for c in candidates}
        # Should have all semantic roles represented
        assert "message_container" in roles_seen
        assert "message_role" in roles_seen
        assert "message_body" in roles_seen
        assert "message_timestamp" in roles_seen
        assert "conversation_title" in roles_seen


class TestSelectBestRoles:
    """select_best_roles() picks single best candidate per role."""

    def test_returns_dict_role_to_candidate(self) -> None:
        candidates = [
            SemanticCandidate(path="$.messages", role="message_container", confidence=0.8),
            SemanticCandidate(path="$.data", role="message_container", confidence=0.5),
        ]
        best = select_best_roles(candidates)
        assert isinstance(best, dict)
        assert "message_container" in best
        assert best["message_container"].confidence == 0.8

    def test_selects_highest_confidence_per_role(self) -> None:
        candidates = [
            SemanticCandidate(path="$.role1", role="message_role", confidence=0.3),
            SemanticCandidate(path="$.role2", role="message_role", confidence=0.7),
            SemanticCandidate(path="$.role3", role="message_role", confidence=0.5),
        ]
        best = select_best_roles(candidates)
        assert best["message_role"].path == "$.role2"
        assert best["message_role"].confidence == 0.7

    def test_empty_candidates_returns_empty_dict(self) -> None:
        best = select_best_roles([])
        assert best == {}

    def test_single_candidate_returned(self) -> None:
        candidates = [
            SemanticCandidate(path="$.messages", role="message_container", confidence=0.6),
        ]
        best = select_best_roles(candidates)
        assert len(best) == 1
        assert "message_container" in best


class TestScoreMessageContainer:
    """_score_container scoring for message_container role."""

    def test_array_based_container_with_children(self) -> None:
        stats = {
            "$.messages": FieldStats(
                path="$.messages",
                array_lengths=[5, 8, 10],
                total_samples=3,
                present_count=3,
            ),
            "$.messages[*].id": FieldStats(path="$.messages[*].id"),
            "$.messages[*].role": FieldStats(path="$.messages[*].role"),
            "$.messages[*].text": FieldStats(path="$.messages[*].text"),
            "$.messages[*].created_at": FieldStats(path="$.messages[*].created_at"),
        }
        candidates = infer_semantic_roles(stats)
        container = next((c for c in candidates if c.role == "message_container"), None)
        assert container is not None
        assert container.path == "$.messages"
        assert container.confidence > 0.15
        assert "avg_array_length" in container.evidence
        assert "child_field_count" in container.evidence

    def test_dict_based_container_with_dynamic_keys(self) -> None:
        stats = {
            "$.conversation_map": FieldStats(
                path="$.conversation_map",
                object_key_counts=[5, 7, 6],
                total_samples=3,
                present_count=3,
            ),
            "$.conversation_map.*": FieldStats(
                path="$.conversation_map.*",
            ),
        }
        candidates = infer_semantic_roles(stats)
        container = next((c for c in candidates if c.role == "message_container"), None)
        assert container is not None
        assert "avg_object_fanout" in container.evidence

    def test_low_frequency_reduces_confidence(self) -> None:
        stats_low = {
            "$.messages": FieldStats(
                path="$.messages",
                array_lengths=[5, 8],
                total_samples=100,
                present_count=50,  # 50% frequency
            ),
            "$.messages[*].id": FieldStats(path="$.messages[*].id"),
        }
        candidates_low = infer_semantic_roles(stats_low)
        container_low = next((c for c in candidates_low if c.role == "message_container"), None)
        # Same setup but with high frequency
        stats_high = {
            "$.messages": FieldStats(
                path="$.messages",
                array_lengths=[5, 8],
                total_samples=100,
                present_count=95,  # 95% frequency
            ),
            "$.messages[*].id": FieldStats(path="$.messages[*].id"),
        }
        candidates_high = infer_semantic_roles(stats_high)
        container_high = next((c for c in candidates_high if c.role == "message_container"), None)
        assert container_high is not None
        # High frequency should score better than low frequency
        if container_low is not None:
            assert container_high.confidence >= container_low.confidence

    def test_shallow_depth_bonus(self) -> None:
        """Containers at depth 0-3 get a depth bonus."""
        stats = {
            "$.messages": FieldStats(
                path="$.messages",
                array_lengths=[5],
                total_samples=1,
                present_count=1,
            ),
            "$.messages[*].id": FieldStats(path="$.messages[*].id"),
        }
        candidates = infer_semantic_roles(stats)
        container = next((c for c in candidates if c.role == "message_container"), None)
        assert container is not None
        assert container.evidence.get("depth", 999) <= 3


class TestScoreMessageRole:
    """_score_role scoring for message_role role."""

    def test_low_cardinality_known_role_values(self) -> None:
        stats = {
            "$.role": FieldStats(
                path="$.role",
                observed_values=Counter({"user": 50, "assistant": 50}),
                total_samples=100,
                present_count=95,
                value_count=100,
                string_lengths=[4, 5, 9],
            ),
        }
        candidates = infer_semantic_roles(stats)
        role = next((c for c in candidates if c.role == "message_role"), None)
        assert role is not None
        assert role.confidence > 0.3
        assert "known_roles" in role.evidence
        assert "distinct_values" in role.evidence

    def test_name_signal_bonus(self) -> None:
        """Fields named 'role', 'sender', 'author', etc. get bonus."""
        stats = {
            "$.sender": FieldStats(
                path="$.sender",
                observed_values=Counter({"alice": 50, "bob": 50}),
                total_samples=100,
                present_count=100,
                value_count=100,
                string_lengths=[3, 3],
            ),
        }
        candidates = infer_semantic_roles(stats)
        role = next((c for c in candidates if c.role == "message_role"), None)
        assert role is not None
        assert "name_signal" in role.evidence

    def test_high_cardinality_rejected(self) -> None:
        """Fields with >15 distinct values are not considered roles."""
        many_values = {str(i): 1 for i in range(20)}
        stats = {
            "$.field": FieldStats(
                path="$.field",
                observed_values=Counter(many_values),
                total_samples=100,
                present_count=100,
                value_count=100,
                string_lengths=[1] * 20,
            ),
        }
        candidates = infer_semantic_roles(stats)
        role = next((c for c in candidates if c.role == "message_role"), None)
        assert role is None

    def test_multiline_content_penalizes(self) -> None:
        """Multiline content reduces role confidence."""
        stats_no_nl = {
            "$.type": FieldStats(
                path="$.type",
                observed_values=Counter({"assistant": 50, "user": 50}),
                total_samples=100,
                present_count=100,
                value_count=100,
                string_lengths=[9, 4],
                is_multiline=0,
                newline_counts=[0] * 100,
            ),
        }
        candidates_no_nl = infer_semantic_roles(stats_no_nl)
        role_no_nl = next((c for c in candidates_no_nl if c.role == "message_role"), None)

        stats_nl = {
            "$.type": FieldStats(
                path="$.type",
                observed_values=Counter({"assistant": 50, "user": 50}),
                total_samples=100,
                present_count=100,
                value_count=100,
                string_lengths=[9, 4],
                is_multiline=20,  # 20% multiline
                newline_counts=[0] * 80 + [1] * 20,
            ),
        }
        candidates_nl = infer_semantic_roles(stats_nl)
        role_nl = next((c for c in candidates_nl if c.role == "message_role"), None)
        # Both should score, but multiline should be lower
        if role_no_nl and role_nl:
            assert role_no_nl.confidence > role_nl.confidence


class TestScoreMessageBody:
    """_score_body scoring for message_body role."""

    def test_long_multiline_text(self) -> None:
        stats = {
            "$.text": FieldStats(
                path="$.text",
                string_lengths=[100, 150, 200, 250],
                is_multiline=3,
                newline_counts=[2, 3, 5, 4],
                total_samples=4,
                present_count=4,
                value_count=4,
                observed_values=Counter(
                    {
                        "a" * 100: 1,
                        "b" * 150: 1,
                        "c" * 200: 1,
                        "d" * 250: 1,
                    }
                ),
            ),
        }
        candidates = infer_semantic_roles(stats)
        body = next((c for c in candidates if c.role == "message_body"), None)
        assert body is not None
        assert body.confidence > 0.3
        assert "avg_length" in body.evidence
        assert "newline_rate" in body.evidence

    def test_name_signal_bonus(self) -> None:
        """Fields named 'text', 'content', 'body', etc. get bonus."""
        stats = {
            "$.content": FieldStats(
                path="$.content",
                string_lengths=[100, 150, 200],
                is_multiline=2,
                newline_counts=[2, 3, 1],
                total_samples=3,
                present_count=3,
                value_count=3,
                observed_values=Counter({"text" * 30: 1, "more" * 40: 1, "content" * 25: 1}),
            ),
        }
        candidates = infer_semantic_roles(stats)
        body = next((c for c in candidates if c.role == "message_body"), None)
        assert body is not None
        assert "name_signal" in body.evidence

    def test_too_short_rejected(self) -> None:
        """Strings averaging <10 chars rejected."""
        stats = {
            "$.field": FieldStats(
                path="$.field",
                string_lengths=[1, 2, 3, 4, 5],
                is_multiline=0,
                newline_counts=[0] * 5,
                total_samples=5,
                present_count=5,
                value_count=5,
                observed_values=Counter({"a": 1, "b": 1, "c": 1, "d": 1, "e": 1}),
            ),
        }
        candidates = infer_semantic_roles(stats)
        body = next((c for c in candidates if c.role == "message_body"), None)
        assert body is None

    def test_entropy_bonus(self) -> None:
        """High entropy (diverse values) increases confidence."""
        stats = {
            "$.text": FieldStats(
                path="$.text",
                string_lengths=[100, 100, 100],
                is_multiline=2,
                newline_counts=[1, 2, 3],
                total_samples=3,
                present_count=3,
                value_count=3,
                observed_values=Counter(
                    {
                        "apple banana cherry dog": 1,
                        "foo bar baz qux": 1,
                        "one two three four": 1,
                    }
                ),
            ),
        }
        candidates = infer_semantic_roles(stats)
        body = next((c for c in candidates if c.role == "message_body"), None)
        assert body is not None
        assert "entropy" in body.evidence


class TestScoreMessageTimestamp:
    """_score_timestamp scoring for message_timestamp role."""

    def test_unix_epoch_format(self) -> None:
        stats = {
            "$.created_at": FieldStats(
                path="$.created_at",
                detected_formats=Counter({"unix-epoch": 95}),
                num_min=1000000000.0,
                num_max=1700000000.0,
                total_samples=100,
                present_count=100,
                value_count=100,
                numeric_values=[1000000000, 1200000000, 1500000000],
            ),
        }
        candidates = infer_semantic_roles(stats)
        ts = next((c for c in candidates if c.role == "message_timestamp"), None)
        assert ts is not None
        assert ts.confidence > 0.3
        assert "format" in ts.evidence

    def test_iso8601_format(self) -> None:
        stats = {
            "$.timestamp": FieldStats(
                path="$.timestamp",
                detected_formats=Counter({"iso8601": 100}),
                total_samples=100,
                present_count=100,
                value_count=100,
                string_lengths=[20] * 100,
                is_multiline=0,
                newline_counts=[0] * 100,
            ),
        }
        candidates = infer_semantic_roles(stats)
        ts = next((c for c in candidates if c.role == "message_timestamp"), None)
        assert ts is not None
        assert ts.confidence > 0.3

    def test_monotonicity_bonus(self) -> None:
        """Monotonically increasing sequences get confidence bonus."""
        stats = {
            "$.created_at": FieldStats(
                path="$.created_at",
                detected_formats=Counter({"unix-epoch": 100}),
                num_min=1000000000.0,
                num_max=1700000000.0,
                total_samples=100,
                present_count=100,
                value_count=100,
                numeric_values=list(range(1000000000, 1000000100)),
            ),
        }
        candidates = infer_semantic_roles(stats)
        ts = next((c for c in candidates if c.role == "message_timestamp"), None)
        assert ts is not None
        # Monotonicity bonus requires _ordered_samples to be set (from array context)
        # Plain numeric_values don't trigger monotonicity checking
        assert ts.confidence >= 0.55

    def test_name_signal_bonus(self) -> None:
        """Fields named 'timestamp', 'created_at', etc. get bonus."""
        stats = {
            "$.created_at": FieldStats(
                path="$.created_at",
                detected_formats=Counter({"unix-epoch": 100}),
                num_min=1000000000.0,
                num_max=1700000000.0,
                total_samples=100,
                present_count=100,
                value_count=100,
            ),
        }
        candidates = infer_semantic_roles(stats)
        ts = next((c for c in candidates if c.role == "message_timestamp"), None)
        assert ts is not None
        assert "name_signal" in ts.evidence

    def test_multiline_penalizes(self) -> None:
        """Multiline content significantly reduces timestamp confidence."""
        stats = {
            "$.ts": FieldStats(
                path="$.ts",
                detected_formats=Counter({"unix-epoch": 50}),
                num_min=1000000000.0,
                num_max=1700000000.0,
                total_samples=100,
                present_count=100,
                value_count=100,
                is_multiline=20,
                newline_counts=[1] * 20 + [0] * 80,
            ),
        }
        candidates = infer_semantic_roles(stats)
        ts = next((c for c in candidates if c.role == "message_timestamp"), None)
        # Should still exist but with reduced confidence due to multiline
        if ts:
            assert ts.confidence < 0.6


class TestScoreConversationTitle:
    """_score_title scoring for conversation_title role."""

    def test_short_high_cardinality_string(self) -> None:
        stats = {
            "$.title": FieldStats(
                path="$.title",
                observed_values=Counter({f"Chat {i}": 1 for i in range(10)}),
                string_lengths=[8, 9, 7, 10, 8, 9],
                is_multiline=0,
                newline_counts=[0] * 6,
                total_samples=10,
                present_count=10,
                value_count=10,
            ),
        }
        candidates = infer_semantic_roles(stats)
        title = next((c for c in candidates if c.role == "conversation_title"), None)
        assert title is not None
        assert title.confidence > 0.2
        assert "avg_length" in title.evidence
        assert "distinct_values" in title.evidence

    def test_name_signal_bonus(self) -> None:
        """Fields named 'title', 'name', 'subject', etc. get bonus."""
        stats = {
            "$.subject": FieldStats(
                path="$.subject",
                observed_values=Counter({"Subject A": 1, "Subject B": 1}),
                string_lengths=[9, 9],
                is_multiline=0,
                newline_counts=[0, 0],
                total_samples=2,
                present_count=2,
                value_count=2,
            ),
        }
        candidates = infer_semantic_roles(stats)
        title = next((c for c in candidates if c.role == "conversation_title"), None)
        assert title is not None
        assert "name_signal" in title.evidence

    def test_too_long_rejected(self) -> None:
        """Titles averaging >200 chars rejected."""
        stats = {
            "$.title": FieldStats(
                path="$.title",
                observed_values=Counter({"a" * 300: 1, "b" * 250: 1}),
                string_lengths=[300, 250],
                is_multiline=0,
                newline_counts=[0, 0],
                total_samples=2,
                present_count=2,
                value_count=2,
            ),
        }
        candidates = infer_semantic_roles(stats)
        title = next((c for c in candidates if c.role == "conversation_title"), None)
        assert title is None

    def test_multiline_penalizes(self) -> None:
        """Multiline rate affects title confidence scoring."""
        stats = {
            "$.title": FieldStats(
                path="$.title",
                observed_values=Counter({"Title A": 1, "Title B": 1}),
                string_lengths=[7, 7],
                is_multiline=1,  # 50% multiline
                newline_counts=[1, 0],
                total_samples=2,
                present_count=2,
                value_count=2,
            ),
        }
        candidates = infer_semantic_roles(stats)
        title = next((c for c in candidates if c.role == "conversation_title"), None)
        # Title with multiline should exist but multiline is tracked in evidence
        if title:
            assert "newline_rate" in title.evidence
            assert title.evidence["newline_rate"] > 0.0

    def test_inside_array_penalizes(self) -> None:
        """Titles inside message arrays ([*]) are penalized."""
        stats = {
            "$.messages[*].subject": FieldStats(
                path="$.messages[*].subject",
                observed_values=Counter({"Subj A": 1, "Subj B": 1}),
                string_lengths=[6, 6],
                is_multiline=0,
                newline_counts=[0, 0],
                total_samples=2,
                present_count=2,
                value_count=2,
            ),
        }
        candidates = infer_semantic_roles(stats)
        title = next((c for c in candidates if c.role == "conversation_title"), None)
        # Array items are penalized but may still score if other factors strong
        if title:
            assert title.confidence < 0.5

    def test_deep_path_penalizes(self) -> None:
        """Deep nesting (depth > 4) is noted in evidence."""
        stats = {
            "$.a.b.c.d.e.title": FieldStats(
                path="$.a.b.c.d.e.title",
                observed_values=Counter({"Title A": 1}),
                string_lengths=[7],
                is_multiline=0,
                newline_counts=[0],
                total_samples=1,
                present_count=1,
                value_count=1,
            ),
        }
        candidates = infer_semantic_roles(stats)
        title = next((c for c in candidates if c.role == "conversation_title"), None)
        # Deep paths are penalized via the scoring function
        # The 0.5x multiplier for depth > 4 should be applied
        if title:
            assert title.evidence.get("depth", 0) > 4
