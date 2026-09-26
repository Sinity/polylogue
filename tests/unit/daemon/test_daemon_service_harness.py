from __future__ import annotations

from pathlib import Path

import pytest

from tests.infra.daemon_service_harness import record_private_lifecycle_probe


def test_lifecycle_probe_rejects_path_traversal_before_creating_receipt_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Receipt names cannot escape the private root or create its directory."""
    receipt_dir = tmp_path / "receipts"
    assert not receipt_dir.exists()
    monkeypatch.setenv("POLYLOGUE_LIFECYCLE_RECEIPT_DIR", str(receipt_dir))

    with pytest.raises(ValueError, match="simple filename stem"):
        record_private_lifecycle_probe(f"../../outside-{tmp_path.name}", {"synthetic": True})

    assert not receipt_dir.exists()
