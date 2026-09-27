"""A member refused for identity is a recorded gap, not a failed acquisition."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core import content_identity
from polylogue.sources.source_acquisition import iter_source_raw_data
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload


def test_a_refused_member_is_recorded_and_its_siblings_are_acquired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: let ``ContentIdentityRefusal`` escape the member loop and
    the ZIP yields nothing and records no named gap for the refused member."""
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 64)
    source_root = tmp_path / "inbox"
    source_root.mkdir()
    zip_path = source_root / "export.zip"
    kept = json.dumps([{"id": "kept", "mapping": {}}]).encode()
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("a.json", b'{"n": 1.' + b"2" * 200 + b"}")
        archive.writestr("b.json", kept)

    cursor_state: CursorStatePayload = {}
    records = list(
        iter_source_raw_data(
            Source(name="chatgpt", path=source_root),
            blob_store=BlobStore(tmp_path / "archive" / "blob"),
            cursor_state=cursor_state,
        )
    )

    acquired = {record.source_path for record in records}
    assert f"{zip_path}:b.json" in acquired
    assert f"{zip_path}:a.json" not in acquired
    failures = cursor_state.get("failed_files", [])
    assert any(failure["path"] == f"{zip_path}:a.json" and "number token" in failure["error"] for failure in failures)
