from __future__ import annotations

from pathlib import Path

from polylogue.importers import ImportResult
from polylogue.render import AttachmentInfo, MarkdownDocument
from polylogue.results import OperationSummary, summarize_import_results


def _build_document(*, attachments: int = 0, attachment_bytes: int = 0) -> MarkdownDocument:
    attachment_list = [
        AttachmentInfo(
            name=f"file-{idx}.txt",
            link=f"attachments/file-{idx}.txt",
            local_path=Path(f"attachments/file-{idx}.txt"),
            size_bytes=attachment_bytes // attachments if attachments else None,
            remote=False,
        )
        for idx in range(attachments)
    ]
    metadata = {"attachmentBytes": attachment_bytes}
    stats = {
        "totalTokensApprox": 42,
        "totalWordsApprox": 10,
        "chunkCount": 3,
        "userTurns": 1,
        "modelTurns": 2,
    }
    return MarkdownDocument(body="content", metadata=metadata, attachments=attachment_list, stats=stats)


def test_operation_summary_lines(tmp_path):
    doc = _build_document(attachments=2, attachment_bytes=2048)
    res = ImportResult(
        markdown_path=tmp_path / "demo" / "conversation.md",
        html_path=None,
        attachments_dir=None,
        document=doc,
        slug="demo",
    )
    summary = summarize_import_results([res])
    lines = summary.lines()
    assert any("1 file(s)" in line for line in lines)
    assert any("Attachments: 2" in line for line in lines)
    assert any("Attachment size" in line for line in lines)
    assert any("Approx tokens" in line for line in lines)
    assert any("Total chunks" in line for line in lines)


def test_operation_summary_skipped(tmp_path):
    base_path = tmp_path / "demo"
    base_path.mkdir()
    skipped = ImportResult(
        markdown_path=base_path / "conversation.md",
        html_path=None,
        attachments_dir=None,
        document=None,
        slug="demo",
        skipped=True,
        skip_reason="up-to-date",
    )
    summary = summarize_import_results([skipped])
    assert summary.written == 0
    assert summary.skipped == 1
    assert "No files written" in summary.lines()[0]


def test_operation_summary_additional_counts():
    summary = OperationSummary(
        output_dir=Path("/tmp/output"),
        written=0,
        skipped=2,
        attachments=0,
        attachment_bytes=0,
        tokens=0,
        words=0,
        diffs=0,
        branches=0,
        extra_counts=[("User turns", 0), ("Model turns", 3)],
    )
    lines = summary.lines()
    assert "No files written" in lines[0]
    assert any("Model turns: 3" in line for line in lines)
