from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import demo_shelf


def test_demo_shelf_writes_manifest_and_readable_bundle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "demos"
    (root / "01-demo").mkdir(parents=True)
    (root / "01-demo" / "README.md").write_text("# Demo\n\nReadable.\n")
    (root / "01-demo" / "payload.bin").write_bytes(b"\x00\x01")

    exit_code = demo_shelf.main(["--root", str(root)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "demo shelf refreshed" in output
    manifest = json.loads((root / "MANIFEST.readable.json").read_text())
    assert manifest["file_count"] == 2
    assert manifest["readable_count"] == 1
    paths = {item["path"]: item for item in manifest["files"]}
    assert paths["01-demo/README.md"]["readable"] is True
    assert paths["01-demo/payload.bin"]["readable"] is False
    assert "MANIFEST.readable.json" not in paths
    assert "CONCATENATED_READABLE.md" not in paths
    bundle = (root / "CONCATENATED_READABLE.md").read_text()
    assert "## 01-demo/README.md" in bundle
    assert "Readable." in bundle
    assert "payload.bin" not in bundle


def test_demo_shelf_check_reports_drift(tmp_path: Path) -> None:
    root = tmp_path / "demos"
    root.mkdir()
    (root / "README.md").write_text("# Demo\n")

    assert demo_shelf.main(["--root", str(root)]) == 0
    (root / "README.md").write_text("# Demo\n\nchanged\n")

    assert demo_shelf.main(["--root", str(root), "--check"]) == 1


def test_demo_shelf_check_json_reports_current_state(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "demos"
    root.mkdir()
    (root / "README.md").write_text("# Demo\n")
    assert demo_shelf.main(["--root", str(root)]) == 0
    capsys.readouterr()

    exit_code = demo_shelf.main(["--root", str(root), "--check", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["mode"] == "check"
    assert payload["file_count"] == 1
    assert payload["readable_count"] == 1
