from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .importers import ImportResult


@dataclass
class OperationSummary:
    """Aggregated view of conversation processing output."""

    output_dir: Optional[Path]
    written: int
    skipped: int = 0
    attachments: int = 0
    attachment_bytes: int = 0
    tokens: int = 0
    words: int = 0
    diffs: int = 0
    branches: int = 0
    extra_counts: List[Tuple[str, int]] = field(default_factory=list)

    def lines(self) -> List[str]:
        lines: List[str] = []
        if self.output_dir:
            if self.written:
                lines.append(f"{self.written} file(s) → {self.output_dir}")
            else:
                lines.append(f"No files written (existing files up to date in {self.output_dir}).")

        if self.attachments:
            lines.append(f"Attachments: {self.attachments}")

        if self.attachment_bytes:
            mb = self.attachment_bytes / (1024 * 1024)
            lines.append(f"Attachment size: {mb:.2f} MiB")

        if self.diffs:
            lines.append(f"Diffs written: {self.diffs}")

        if self.tokens:
            if self.words:
                lines.append(f"Approx tokens: {self.tokens} (~{self.words} words)")
            else:
                lines.append(f"Approx tokens: {self.tokens}")

        if self.branches:
            lines.append(f"Branches rendered: {self.branches}")

        for label, value in self.extra_counts:
            if value:
                lines.append(f"{label}: {value}")

        if self.skipped and not self.written:
            lines.append(f"Skipped (up-to-date): {self.skipped}")

        return lines or ["No files written."]


def _sum_stat(results: Iterable[ImportResult], key: str) -> int:
    total = 0
    for res in results:
        if not res.document:
            continue
        value = res.document.stats.get(key)
        if isinstance(value, (int, float)):
            total += int(value)
    return total


def summarize_import_results(results: Sequence[ImportResult]) -> OperationSummary:
    if not results:
        return OperationSummary(output_dir=None, written=0, skipped=0)

    written = [res for res in results if not res.skipped]
    skipped = [res for res in results if res.skipped]

    if written:
        conversation_dir = written[0].markdown_path.parent
    else:
        conversation_dir = results[0].markdown_path.parent
    output_dir = conversation_dir.parent if conversation_dir.parent else conversation_dir

    attachments_total = sum(
        len(res.document.attachments) for res in written if res.document
    )
    attachment_bytes = sum(
        res.document.metadata.get("attachmentBytes", 0) or 0
        for res in written
        if res.document
    )
    tokens = sum(
        int(res.document.stats.get("totalTokensApprox", 0) or 0)
        for res in written
        if res.document
    )
    words = sum(
        int(res.document.stats.get("totalWordsApprox", 0) or 0)
        for res in written
        if res.document
    )
    diff_total = sum(1 for res in written if getattr(res, "diff_path", None))
    branch_total = sum(res.branch_count for res in written if hasattr(res, "branch_count"))

    extra_counts = [
        ("Total chunks", _sum_stat(written, "chunkCount")),
        ("User turns", _sum_stat(written, "userTurns")),
        ("Model turns", _sum_stat(written, "modelTurns")),
    ]

    return OperationSummary(
        output_dir=output_dir,
        written=len(written),
        skipped=len(skipped),
        attachments=attachments_total,
        attachment_bytes=attachment_bytes,
        tokens=tokens,
        words=words,
        diffs=diff_total,
        branches=branch_total,
        extra_counts=extra_counts,
    )
