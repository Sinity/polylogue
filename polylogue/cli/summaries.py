from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Protocol

from ..importers import ImportResult
from ..results import summarize_import_results


class SummaryUI(Protocol):
    plain: bool

    class Console(Protocol):
        def print(self, *args, **kwargs) -> None: ...

    console: Console

    def summary(self, title: str, lines: Iterable[str]) -> None: ...


def summarize_import(ui: SummaryUI, title: str, results: List[ImportResult]) -> None:
    summary = summarize_import_results(results)
    lines = summary.lines()

    written = [res for res in results if not res.skipped]
    if written and not ui.plain:
        try:
            from rich.table import Table

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
        except Exception:  # pragma: no cover - rich optional
            pass

    skipped_reasons = Counter(res.skip_reason for res in results if res.skipped)
    for reason, count in skipped_reasons.items():
        label = f"Skipped ({reason})" if reason else "Skipped"
        lines.append(f"{label}: {count}")

    ui.summary(title, lines)

