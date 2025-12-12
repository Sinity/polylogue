"""Tests for raw import storage functionality."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from polylogue.importers.raw_storage import (
    compute_hash,
    get_import_stats,
    mark_parse_failed,
    mark_parse_success,
    retrieve_raw_import,
    store_raw_import,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    try:
        yield db_path
    finally:
        if db_path.exists():
            db_path.unlink()


def test_compute_hash():
    """Test hash computation."""
    data1 = b"hello world"
    data2 = b"hello world"
    data3 = b"hello world!"

    hash1 = compute_hash(data1)
    hash2 = compute_hash(data2)
    hash3 = compute_hash(data3)

    # Same data produces same hash
    assert hash1 == hash2
    # Different data produces different hash
    assert hash1 != hash3
    # Hash is 64 characters (SHA-256 hex)
    assert len(hash1) == 64


def test_store_and_retrieve_raw_import(temp_db):
    """Test storing and retrieving raw imports."""
    data = b'{"test": "data", "messages": []}'

    # Store raw import
    data_hash = store_raw_import(
        data=data,
        provider="chatgpt",
        conversation_id="test-conv-1",
        source_path=Path("/tmp/test.json"),
        db_path=temp_db,
        compress=True,
    )

    # Hash should be consistent
    assert data_hash == compute_hash(data)

    # Retrieve should decompress and return original data
    retrieved = retrieve_raw_import(data_hash, db_path=temp_db)
    assert retrieved == data


def test_store_duplicate_hash(temp_db):
    """Test that storing same data twice doesn't fail (same conversation, same content)."""
    data = b'{"test": "data"}'

    hash1 = store_raw_import(data=data, provider="chatgpt", conversation_id="test-conv-2", db_path=temp_db)
    hash2 = store_raw_import(data=data, provider="chatgpt", conversation_id="test-conv-2", db_path=temp_db)

    # Should return same hash and skip duplicate
    assert hash1 == hash2


def test_mark_parse_status(temp_db):
    """Test marking parse success/failure."""
    data = b'{"test": "data"}'
    data_hash = store_raw_import(data=data, provider="chatgpt", conversation_id="test-conv-3", db_path=temp_db)

    # Mark as success
    mark_parse_success(data_hash, db_path=temp_db)

    # Mark as failed
    mark_parse_failed(data_hash, "Test error", db_path=temp_db)


def test_get_import_stats(temp_db):
    """Test getting import statistics."""
    # Store some imports with different statuses
    data1 = b'{"test": "data1"}'
    data2 = b'{"test": "data2"}'
    data3 = b'{"test": "data3"}'

    hash1 = store_raw_import(data=data1, provider="chatgpt", conversation_id="conv-1", db_path=temp_db)
    hash2 = store_raw_import(data=data2, provider="claude", conversation_id="conv-2", db_path=temp_db)
    hash3 = store_raw_import(data=data3, provider="chatgpt", conversation_id="conv-3", db_path=temp_db)

    mark_parse_success(hash1, db_path=temp_db)
    mark_parse_failed(hash2, "error", db_path=temp_db)
    # hash3 stays pending

    stats = get_import_stats(db_path=temp_db)

    assert stats["total"] == 3
    assert stats["by_status"]["success"] == 1
    assert stats["by_status"]["failed"] == 1
    assert stats["by_status"]["pending"] == 1
    assert stats["by_provider"]["chatgpt"] == 2
    assert stats["by_provider"]["claude"] == 1


def test_store_without_compression(temp_db):
    """Test storing data without compression."""
    data = b'{"test": "uncompressed"}'

    data_hash = store_raw_import(
        data=data,
        provider="chatgpt",
        conversation_id="test-conv-4",
        db_path=temp_db,
        compress=False,
    )

    retrieved = retrieve_raw_import(data_hash, db_path=temp_db)
    assert retrieved == data
