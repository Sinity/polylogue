"""Embedding cache currency follows the purchased tier's admitted topology."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from polylogue.operations.embedding_readiness import EmbeddingReadinessUnavailableError, embedding_readiness_currency


def test_split_purchased_tier_wal_invalidates_currency_at_actual_leaf(tmp_path: Path) -> None:
    root = tmp_path / "configured"
    root.mkdir()
    purchased = tmp_path / "purchased"
    purchased.mkdir()
    path = purchased / "embeddings.db"
    (root / "embeddings.db").symlink_to(path)

    def fingerprint() -> str:
        return embedding_readiness_currency(root, enabled=True, has_key=True, model="voyage-4", dimension=1024)

    with sqlite3.connect(path) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE neutral_measurements (value INTEGER)")
        writer.commit()
        initial = fingerprint()
        header = path.read_bytes()[:100]
        # A file beside the configured alias is not the purchased connection's
        # WAL. It must neither conceal nor invalidate that actual tier.
        (root / "embeddings.db-wal").write_bytes(b"neutral unrelated file")
        assert fingerprint() == initial
        writer.execute("INSERT INTO neutral_measurements VALUES (1)")
        writer.commit()
        assert path.read_bytes()[:100] == header
        assert fingerprint() != initial


def test_purchased_tier_retarget_changes_currency(tmp_path: Path) -> None:
    root = tmp_path / "configured"
    root.mkdir()
    first = tmp_path / "first.db"
    second = tmp_path / "second.db"
    first.write_bytes(b"neutral header")
    second.write_bytes(b"neutral header")
    alias = root / "embeddings.db"
    alias.symlink_to(first)
    first_currency = embedding_readiness_currency(root, enabled=True, has_key=True, model="voyage-4", dimension=1024)
    alias.unlink()
    alias.symlink_to(second)
    assert (
        embedding_readiness_currency(root, enabled=True, has_key=True, model="voyage-4", dimension=1024)
        != first_currency
    )


def test_unreadable_purchased_currency_preserves_domain_cause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        raise PermissionError("neutral unreadable tier")

    monkeypatch.setattr("polylogue.operations.embedding_readiness._file_currency", unreadable)
    with pytest.raises(EmbeddingReadinessUnavailableError) as raised:
        embedding_readiness_currency(tmp_path, enabled=True, has_key=True, model="voyage-4", dimension=1024)
    assert isinstance(raised.value.__cause__, PermissionError)


def test_nonregular_tier_is_unavailable_without_waiting_for_a_writer(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "embeddings.db")
    with pytest.raises(EmbeddingReadinessUnavailableError) as raised:
        embedding_readiness_currency(tmp_path, enabled=True, has_key=True, model="voyage-4", dimension=1024)
    assert isinstance(raised.value.__cause__, OSError)
