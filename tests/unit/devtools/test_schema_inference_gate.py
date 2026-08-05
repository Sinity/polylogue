"""CLI adapter tests for the schema-inference gate receipt."""

from __future__ import annotations

import json
from pathlib import Path

from devtools import schema_inference_gate
from polylogue.maintenance.schema_inference_gate import RECEIPT_FILENAME
from tests.infra.schema_inference import seed_schema_inference_archive as _seed_archive


def test_devtools_command_requires_caller_root_and_persists_receipt(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    output = tmp_path / RECEIPT_FILENAME

    exit_code = schema_inference_gate.main(
        [
            "--archive-root",
            str(root),
            "--ground-truth-root",
            f"codex-session={ground_truth}",
            "--receipt",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["input_paths"]["archive_root"] == str(root.absolute())
