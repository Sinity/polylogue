"""CLI adapter tests for the schema-inference gate receipt."""

from __future__ import annotations

import json
from pathlib import Path

from devtools import schema_inference_gate
from tests.unit.maintenance.test_schema_inference_gate import _seed_archive, _write_hash_receipt


def test_devtools_command_requires_caller_root_and_persists_receipt(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed_archive(root)
    blob_receipt = tmp_path / "blob.json"
    _write_hash_receipt(root, blob_receipt)
    output = tmp_path / "gate.json"

    exit_code = schema_inference_gate.main(
        [
            "--archive-root",
            str(root),
            "--blob-hash-receipt",
            str(blob_receipt),
            "--receipt",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["input_paths"]["archive_root"] == str(root.absolute())
