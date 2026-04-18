"""Direct tests for polylogue.schemas.field_stats module.

Covers is_dynamic_key(), _detect_string_format(), FieldStats computed
properties, and _collect_field_stats() sample collection.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.schemas.field_stats import (
    FieldStats,
    _collect_field_stats,
    _detect_string_format,
    is_dynamic_key,
)

# =============================================================================
# is_dynamic_key
# =============================================================================


class TestIsDynamicKey:
    @pytest.mark.parametrize(
        "key",
        [
            "550e8400-e29b-41d4-a716-446655440000",
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        ],
    )
    def test_uuid_detected(self, key: str) -> None:
        assert is_dynamic_key(key)

    @pytest.mark.parametrize(
        "key",
        [
            "a" * 24,
            "abcdef1234567890abcdef12",
            "0" * 32,
        ],
    )
    def test_hex_id_detected(self, key: str) -> None:
        assert is_dynamic_key(key)

    @pytest.mark.parametrize(
        "key",
        [
            "msg-abc123",
            "node-550e8400",
            "conv-def456",
            "item-789abc",
            "att-abc-def",
        ],
    )
    def test_prefixed_id_detected(self, key: str) -> None:
        assert is_dynamic_key(key)

    @pytest.mark.parametrize(
        "key",
        [
            "mapping",
            "title",
            "content",
            "type",
            "role",
            "abc",
            "my-field",
            "status_code",
        ],
    )
    def test_static_keys_not_dynamic(self, key: str) -> None:
        assert not is_dynamic_key(key)


# =============================================================================
# _detect_string_format
# =============================================================================


class TestDetectStringFormat:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("550e8400-e29b-41d4-a716-446655440000", "uuid4"),
            ("550e8400-e29b-0000-0000-446655440000", "uuid"),
            ("abcdef1234567890abcdef12", "hex-id"),
            ("2024-01-15T10:30:00Z", "iso8601"),
            ("2024-01-15 10:30:00", "iso8601"),
            ("1705312200", "unix-epoch-str"),
            ("1705312200.123", "unix-epoch-str"),
            ("https://example.com/path", "url"),
            ("http://api.test.io", "url"),
            ("application/json", "mime-type"),
            ("text/plain", "mime-type"),
            ("image/png", "mime-type"),
            ("user@example.com", "email"),
        ],
    )
    def test_format_detection(self, value: str, expected: str) -> None:
        assert _detect_string_format(value) == expected

    def test_base64_detection(self) -> None:
        b64_value = "A" * 44 + "=="  # 46 chars, valid base64 pattern
        assert _detect_string_format(b64_value) == "base64"

    def test_empty_string_returns_none(self) -> None:
        assert _detect_string_format("") is None

    def test_long_string_returns_none(self) -> None:
        assert _detect_string_format("a" * 501) is None

    def test_plain_text_returns_none(self) -> None:
        assert _detect_string_format("hello world") is None

    @given(st.text(max_size=500))
    @settings(max_examples=100)
    def test_never_raises(self, value: str) -> None:
        result = _detect_string_format(value)
        assert result is None or isinstance(result, str)


# =============================================================================
# FieldStats computed properties
# =============================================================================


class TestFieldStatsProperties:
    def test_frequency_with_data(self) -> None:
        stats = FieldStats(path="$.x", total_samples=10, present_count=7)
        assert stats.frequency == pytest.approx(0.7)

    def test_frequency_no_samples(self) -> None:
        stats = FieldStats(path="$.x", total_samples=0, present_count=0)
        assert stats.frequency == 0.0

    def test_dominant_format_clear_winner(self) -> None:
        stats = FieldStats(
            path="$.x",
            detected_formats=Counter({"uuid": 9, "hex-id": 1}),
            value_count=10,
        )
        assert stats.dominant_format == "uuid"

    def test_dominant_format_no_winner(self) -> None:
        stats = FieldStats(
            path="$.x",
            detected_formats=Counter({"uuid": 4, "hex-id": 6}),
            value_count=10,
        )
        # uuid at 40% doesn't pass 80% threshold; hex-id at 60% doesn't either
        assert stats.dominant_format is None

    def test_dominant_format_empty(self) -> None:
        stats = FieldStats(path="$.x", value_count=0)
        assert stats.dominant_format is None

    def test_is_enum_like_low_cardinality(self) -> None:
        stats = FieldStats(path="$.x", observed_values=Counter({"a": 5, "b": 3}))
        assert stats.is_enum_like

    def test_is_enum_like_high_cardinality(self) -> None:
        stats = FieldStats(
            path="$.x",
            observed_values=Counter({f"val_{i}": 1 for i in range(51)}),
        )
        assert not stats.is_enum_like

    def test_is_enum_like_empty(self) -> None:
        stats = FieldStats(path="$.x")
        assert not stats.is_enum_like

    def test_string_length_stats(self) -> None:
        stats = FieldStats(path="$.x", string_lengths=[3, 5, 7, 9])
        result = stats.string_length_stats
        assert result is not None
        assert result["min"] == 3
        assert result["max"] == 9
        assert result["avg"] == 6.0
        assert result["stddev"] > 0

    def test_string_length_stats_single(self) -> None:
        stats = FieldStats(path="$.x", string_lengths=[10])
        result = stats.string_length_stats
        assert result is not None
        assert result["stddev"] == 0.0

    def test_string_length_stats_empty(self) -> None:
        stats = FieldStats(path="$.x")
        assert stats.string_length_stats is None

    def test_newline_rate(self) -> None:
        stats = FieldStats(path="$.x", is_multiline=3, value_count=10)
        assert stats.newline_rate == pytest.approx(0.3)

    def test_newline_rate_zero_values(self) -> None:
        stats = FieldStats(path="$.x", is_multiline=0, value_count=0)
        assert stats.newline_rate == 0.0

    def test_monotonicity_score_increasing(self) -> None:
        stats = FieldStats(path="$.x", _ordered_samples=[[1, 2, 3, 4, 5]])
        assert stats.monotonicity_score == pytest.approx(1.0)

    def test_monotonicity_score_decreasing(self) -> None:
        stats = FieldStats(path="$.x", _ordered_samples=[[5, 4, 3, 2, 1]])
        assert stats.monotonicity_score == pytest.approx(0.0)

    def test_monotonicity_score_none(self) -> None:
        stats = FieldStats(path="$.x")
        assert stats.monotonicity_score is None

    def test_avg_array_length(self) -> None:
        stats = FieldStats(path="$.x", array_lengths=[2, 4, 6])
        assert stats.avg_array_length == pytest.approx(4.0)

    def test_avg_array_length_none(self) -> None:
        stats = FieldStats(path="$.x")
        assert stats.avg_array_length is None

    def test_approximate_entropy_uniform(self) -> None:
        stats = FieldStats(
            path="$.x",
            observed_values=Counter({"a": 10, "b": 10}),
            value_count=20,
        )
        # Uniform 2-symbol distribution → entropy = 1.0
        assert stats.approximate_entropy == pytest.approx(1.0)

    def test_approximate_entropy_single_value(self) -> None:
        stats = FieldStats(
            path="$.x",
            observed_values=Counter({"a": 10}),
            value_count=10,
        )
        assert stats.approximate_entropy == pytest.approx(0.0)

    def test_approximate_entropy_none(self) -> None:
        stats = FieldStats(path="$.x", value_count=0)
        assert stats.approximate_entropy is None


# =============================================================================
# _collect_field_stats
# =============================================================================


class TestCollectFieldStats:
    def test_discovers_all_top_level_keys(self) -> None:
        samples: list[dict[str, Any]] = [
            {"a": 1, "b": "hello"},
            {"b": "world", "c": True},
            {"a": 2, "c": False, "d": None},
        ]
        stats = _collect_field_stats(samples)
        # Root + all top-level keys should be present
        discovered_keys = {s.path.split(".")[-1] for s in stats.values() if "." in s.path}
        assert {"a", "b", "c", "d"} <= discovered_keys

    def test_total_samples_consistent(self) -> None:
        samples = [{"x": i} for i in range(5)]
        stats = _collect_field_stats(samples)
        for s in stats.values():
            assert s.total_samples == 5

    def test_present_count_tracks_non_null(self) -> None:
        samples: list[dict[str, Any]] = [{"x": 1}, {"x": None}, {"x": 3}]
        stats = _collect_field_stats(samples)
        assert stats["$.x"].present_count == 2

    def test_nested_fields_discovered(self) -> None:
        samples: list[dict[str, Any]] = [{"outer": {"inner": "value"}}]
        stats = _collect_field_stats(samples)
        assert "$.outer.inner" in stats

    def test_array_items_use_wildcard_path(self) -> None:
        samples: list[dict[str, Any]] = [{"items": [1, 2, 3]}]
        stats = _collect_field_stats(samples)
        assert "$.items[*]" in stats

    def test_dynamic_keys_use_wildcard(self) -> None:
        samples: list[dict[str, Any]] = [{"mapping": {"550e8400-e29b-41d4-a716-446655440000": {"v": 1}}}]
        stats = _collect_field_stats(samples)
        assert "$.mapping.*" in stats

    def test_conversation_ids_tracked(self) -> None:
        samples: list[dict[str, Any]] = [{"status": "active"}, {"status": "active"}, {"status": "pending"}]
        conv_ids: list[str | None] = ["conv1", "conv1", "conv2"]
        stats = _collect_field_stats(samples, conversation_ids=conv_ids)
        status_stats = stats["$.status"]
        assert "active" in status_stats.value_conversation_ids
        assert status_stats.value_conversation_ids["active"] == {"conv1"}
        assert status_stats.value_conversation_ids["pending"] == {"conv2"}

    @given(
        st.lists(
            st.fixed_dictionaries({"key": st.text(min_size=1, max_size=20)}),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_collect_discovers_all_keys_property(self, samples: list[dict[str, str]]) -> None:
        stats = _collect_field_stats(samples)
        # All provided keys should be discovered
        for sample in samples:
            for key in sample:
                assert f"$.{key}" in stats
