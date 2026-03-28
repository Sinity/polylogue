from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional, Protocol

from rich.table import Table

from ..importers import ImportResult
from ..options import RenderResult
from ..results import summarize_import_results
from ..ui import ConsoleLike


class SummaryUI(Protocol):
    plain: bool
    console: ConsoleLike

    def summary(self, title: str, lines: Iterable[str]) -> None: ...


def summarize_import(
    ui: SummaryUI,
    title: str,
    results: List[ImportResult],
    *,
    extra_lines: Optional[List[str]] = None,
) -> None:
    summary = summarize_import_results(results)
    lines = summary.lines()

    written = [res for res in results if not res.skipped]
    if not written:
        skip_reasons = {res.skip_reason for res in results if res.skipped and res.skip_reason}
        if skip_reasons and skip_reasons != {"up-to-date"}:
            lines = [line for line in lines if not line.startswith("No files written (existing files up to date in ")]
            if not lines or not lines[0].startswith("No files written"):
                lines.insert(0, "No files written.")
    if written and not ui.plain:
        table = Table(title=title, show_lines=False)
        table.add_column("File")
        table.add_column("Attachments", justify="right")
        table.add_column("Attachment MiB", justify="right")
        table.add_column("Tokens (~words)", justify="right")
        for res in written:
            if res.document:
                att_count = len(res.document.attachments)
                att_bytes = res.document.metadata.get("attachmentBytes", 0) or 0
                tokens = int(res.document.stats.get("totalTokensApprox", 0) or 0)
                words = int(res.document.stats.get("totalWordsApprox", 0) or 0)
            else:
                att_count = 0
                att_bytes = 0
                tokens = 0
                words = 0
            table.add_row(
                res.slug,
                str(att_count),
                f"{att_bytes / (1024 * 1024):.2f}" if att_bytes else "0.00",
                f"{tokens} (~{words} words)" if tokens and words else str(tokens),
            )
        ui.console.print(table)

    skipped_reasons = Counter(res.skip_reason for res in results if res.skipped)
    for reason, count in skipped_reasons.items():
        label = f"Skipped ({reason})" if reason else "Skipped"
        lines.append(f"{label}: {count}")

    if extra_lines:
        lines.extend(extra_lines)
    ui.summary(title, lines)


def summarize_render(
    ui: SummaryUI,
    title: str,
    result: RenderResult,
    *,
    extra_lines: Optional[List[str]] = None,
) -> None:
    totals = result.total_stats or {}
    lines = [f"Rendered {result.count} file(s) → {result.output_dir}"]
    attachments_total = totals.get("attachments", 0)
    if attachments_total:
        lines.append(f"Attachments: {attachments_total}")
    skipped_total = totals.get("skipped", 0)
    if skipped_total:
        lines.append(f"Skipped: {skipped_total}")
    diffs_total = totals.get("diffs", 0)
    if diffs_total:
        lines.append(f"Diffs: {diffs_total}")
    if "totalTokensApprox" in totals:
        total_tokens = int(totals.get("totalTokensApprox") or 0)
        total_words = int(totals.get("totalWordsApprox") or 0)
        if total_words:
            lines.append(f"Approx tokens: {total_tokens} (~{total_words} words)")
        else:
            lines.append(f"Approx tokens: {total_tokens}")
    for key, label in (
        ("chunkCount", "Total chunks"),
        ("userTurns", "User turns"),
        ("modelTurns", "Model turns"),
    ):
        value = totals.get(key)
        if value:
            lines.append(f"{label}: {int(value)}")

    if result.files and not ui.plain:
        table = Table(title=title, show_lines=False)
        table.add_column("File")
        table.add_column("Attachments", justify="right")
        table.add_column("Tokens (~words)", justify="right")
        for file in result.files:
            tokens = int(file.stats.get("totalTokensApprox", 0) or 0)
            words = int(file.stats.get("totalWordsApprox", 0) or 0)
            token_label = f"{tokens} (~{words} words)" if words else str(tokens)
            table.add_row(file.slug, str(file.attachments), token_label)
        ui.console.print(table)

    if extra_lines:
        lines.extend(extra_lines)
    ui.summary(title, lines)
