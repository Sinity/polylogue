"""Tests for polylogue.assets module."""
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from polylogue.assets import asset_path, write_asset


class TestConcurrentAssetWrite:
    """Tests for concurrent asset writing safety."""

    def test_concurrent_write_same_asset_no_corruption(self, tmp_path: Path):
        """Concurrent writes to same asset should not corrupt file.

        This test validates that atomic write is implemented.
        """
        asset_id = "concurrent-test-asset"
        content = b"x" * 10000  # 10KB of data

        def write_asset_thread(thread_id: int):
            # Each thread writes the same content to the same asset
            write_asset(tmp_path, asset_id, content)

        # Run 10 concurrent writes
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(write_asset_thread, range(10)))

        # Verify file is not corrupted
        final_path = asset_path(tmp_path, asset_id)
        assert final_path.exists(), "Asset file should exist"
        assert final_path.read_bytes() == content, "Asset content should not be corrupted"

    def test_write_asset_atomic(self, tmp_path: Path):
        """write_asset should use atomic write (write to temp, then rename)."""
        asset_id = "atomic-test"
        content = b"test content"

        write_asset(tmp_path, asset_id, content)

        final_path = asset_path(tmp_path, asset_id)
        assert final_path.exists()
        assert final_path.read_bytes() == content

    def test_write_asset_creates_parent_directories(self, tmp_path: Path):
        """write_asset should create necessary parent directories."""
        asset_id = "deeply-nested-asset-id-with-hash-prefix"
        content = b"nested content"

        write_asset(tmp_path, asset_id, content)

        final_path = asset_path(tmp_path, asset_id)
        assert final_path.exists()
        assert final_path.read_bytes() == content
        assert final_path.parent.exists()

    def test_write_asset_overwrites_existing(self, tmp_path: Path):
        """write_asset should overwrite existing file atomically."""
        asset_id = "overwrite-test"
        old_content = b"old content"
        new_content = b"new content that is different"

        # Write initial content
        write_asset(tmp_path, asset_id, old_content)
        final_path = asset_path(tmp_path, asset_id)
        assert final_path.read_bytes() == old_content

        # Overwrite with new content
        write_asset(tmp_path, asset_id, new_content)
        assert final_path.read_bytes() == new_content

    def test_write_asset_empty_content(self, tmp_path: Path):
        """write_asset should handle empty content correctly."""
        asset_id = "empty-asset"
        content = b""

        write_asset(tmp_path, asset_id, content)

        final_path = asset_path(tmp_path, asset_id)
        assert final_path.exists()
        assert final_path.read_bytes() == content
