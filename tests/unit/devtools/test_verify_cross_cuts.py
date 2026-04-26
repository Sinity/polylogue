"""Tests for ``devtools verify-cross-cuts``."""

from __future__ import annotations

import pytest

from devtools import verify_cross_cuts


def test_expected_for_runtime_suffix() -> None:
    assert verify_cross_cuts.expected_for("query_runtime.py")["lifecycle"] == "runtime"
    assert verify_cross_cuts.expected_for("session_profile_runtime.py")["lifecycle"] == "runtime"


def test_expected_for_models_suffix() -> None:
    assert verify_cross_cuts.expected_for("query_models.py")["lifecycle"] == "model"


def test_expected_for_sync_prefix() -> None:
    assert verify_cross_cuts.expected_for("sync_product_queries.py")["api"] == "sync"
    assert verify_cross_cuts.expected_for("sync.py")["api"] == "sync"


def test_expected_for_facade_prefix() -> None:
    assert verify_cross_cuts.expected_for("facade.py")["api"] == "async"
    assert verify_cross_cuts.expected_for("facade_archive.py")["api"] == "async"


def test_expected_for_layer_writes() -> None:
    assert verify_cross_cuts.expected_for("repository_writes.py")["layer"] == "write"
    assert verify_cross_cuts.expected_for("repository_write_conversations.py")["layer"] == "write"


def test_expected_for_layer_reads() -> None:
    assert verify_cross_cuts.expected_for("repository_archive_reads.py")["layer"] == "read"


def test_expected_for_unsuffixed_returns_empty() -> None:
    assert verify_cross_cuts.expected_for("dates.py") == {}


def test_committed_yaml_is_clean(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed projection's cross-cut tags should be internally consistent."""
    rc = verify_cross_cuts.main([])
    captured = capsys.readouterr()
    assert rc == 0, captured.out
    assert "blocking=False" in captured.out


def test_parse_cross_cut_inline_mapping() -> None:
    parsed = verify_cross_cuts.parse_cross_cut("{ layer: read, lifecycle: runtime }")
    assert parsed == {"layer": "read", "lifecycle": "runtime"}


def test_parse_cross_cut_empty() -> None:
    assert verify_cross_cuts.parse_cross_cut("") == {}
    assert verify_cross_cuts.parse_cross_cut("{}") == {}
