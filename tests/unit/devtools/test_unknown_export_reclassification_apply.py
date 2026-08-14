from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.unknown_export_reclassification_apply import main
from polylogue.maintenance.unknown_export_reclassification_apply import UnknownExportReclassificationApplyReport


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


@pytest.mark.parametrize("source_path_like", ["", "%browser-capture%"])
def test_unknown_export_apply_refuses_to_widen_the_measured_scope(
    monkeypatch: pytest.MonkeyPatch,
    source_path_like: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _unexpected_apply(*args: object, **kwargs: object) -> None:
        raise AssertionError("the durable apply path must not run outside the measured spool")

    monkeypatch.setattr(
        "devtools.unknown_export_reclassification_apply.apply_unknown_export_reclassification", _unexpected_apply
    )

    assert main(["--apply", "--source-path-like", source_path_like]) == 1
    assert "refused: --apply is limited to the measured ChatGPT browser-capture spool" in capsys.readouterr().out
