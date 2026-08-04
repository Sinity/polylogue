from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.command_catalog import COMMANDS
from devtools.unknown_export_reclassification_apply import main
from polylogue.maintenance.unknown_export_reclassification_apply import UnknownExportReclassificationApplyReport


def test_unknown_export_apply_is_catalogued_and_exposes_receipt_contract() -> None:
    spec = COMMANDS["workspace unknown-export-reclassification-apply"]
    assert spec.module == "devtools.unknown_export_reclassification_apply"
    assert callable(spec.resolve_main())


def test_unknown_export_apply_json_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "verified-source-backup" / "manifest.json"
    expected = UnknownExportReclassificationApplyReport(
        scanned_count=41,
        reclassifiable_count=41,
        reclassifiable_bytes=123,
        chatgpt_reclassifiable_count=41,
        chatgpt_reclassifiable_bytes=123,
        non_chatgpt_reclassifiable_count=0,
        still_unknown_count=0,
        blob_missing_count=0,
        reclassified_count=41,
        reclassified_bytes=123,
        reclassified_raw_ids=("raw-1",),
        applied=True,
        source_path_like="%/browser-capture/chatgpt/%",
        backup_manifest=manifest,
    )
    monkeypatch.setattr(
        "devtools.unknown_export_reclassification_apply.apply_unknown_export_reclassification",
        lambda *args, **kwargs: expected,
    )

    assert main(["--archive-root", str(tmp_path), "--apply", "--backup-manifest", str(manifest), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is True
    assert payload["scanned_count"] == 41
    assert payload["chatgpt_reclassifiable_count"] == 41
    assert payload["non_chatgpt_reclassifiable_count"] == 0
    assert payload["reclassified_raw_ids"] == ["raw-1"]
    assert payload["index_reparse_required"] is True
    assert payload["index_rows_touched"] == 0
    assert payload["receipt_table"] == "raw_unknown_export_reclassification_receipts"
