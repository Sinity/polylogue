#!/usr/bin/env python3
"""Generate README screenshots for the CLI."""
from __future__ import annotations

import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List


def _seed_inbox(inbox: Path) -> None:
    inbox.mkdir(parents=True, exist_ok=True)
    conversations = [
        {
            "id": "demo-1",
            "title": "Launch Plan",
            "messages": [
                {
                    "id": "m1",
                    "role": "user",
                    "content": "Draft the launch checklist for the pipeline review.",
                },
                {
                    "id": "m2",
                    "role": "assistant",
                    "content": "Checklist:\n- establish baseline latency\n- capture p95/p99\n- rollback plan",
                },
            ],
        },
        {
            "id": "demo-2",
            "title": "Search Tune-up",
            "messages": [
                {"id": "m1", "role": "user", "content": "How do we reduce ingest time?"},
                {
                    "id": "m2",
                    "role": "assistant",
                    "content": "Update the pipeline to use incremental indexing.",
                },
            ],
        },
    ]
    for idx, convo in enumerate(conversations, start=1):
        path = inbox / f"conversation-{idx}.json"
        path.write_text(json.dumps(convo, indent=2), encoding="utf-8")


def _dry_run_lines(counts: dict, sources: List[str], timestamp: int, cursors: dict) -> List[str]:
    lines = [
        f"Counts: {counts['conversations']} conv, {counts['messages']} msg",
    ]
    if sources:
        lines.insert(0, f"Sources: {', '.join(sources)}")
    if cursors:
        parts = []
        for name, cursor in cursors.items():
            file_count = cursor.get("file_count")
            latest = None
            latest_mtime = cursor.get("latest_mtime")
            if isinstance(latest_mtime, (int, float)):
                latest = datetime.fromtimestamp(int(latest_mtime)).isoformat(timespec="seconds")
            else:
                latest = cursor.get("latest_path") or cursor.get("latest_file_name")
            detail = []
            if isinstance(file_count, int):
                detail.append(f"{file_count} files")
            if isinstance(latest, str):
                detail.append(f"latest {Path(latest).name if '/' in latest else latest}")
            parts.append(f"{name} ({', '.join(detail)})")
        lines.append(f"Cursors: {'; '.join(parts)}")
    return lines


def _run_lines(run_result: object) -> List[str]:
    counts = run_result.counts
    return [
        f"Counts: {counts['conversations']} conv, {counts['messages']} msg",
        f"Index: {'ok' if run_result.indexed else 'up-to-date'}",
        f"Duration: {run_result.duration_ms}ms",
    ]


def _write_svg(path: Path, render) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    render(path)


def main() -> None:
    demo_root = Path(tempfile.mkdtemp(prefix="polylogue-readme-"))
    os.environ["XDG_STATE_HOME"] = str(demo_root / "state")
    os.environ["XDG_DATA_HOME"] = str(demo_root / "data")

    from rich.console import Console
    from rich.text import Text

    from polylogue.config import Config, Source
    from polylogue.run import plan_sources, run_sources
    from polylogue.search import search_messages
    from polylogue.ui.facade import ConsoleFacade

    inbox = demo_root / "inbox"
    archive_root = demo_root / "archive"
    _seed_inbox(inbox)

    config = Config(
        version=2,
        archive_root=archive_root,
        sources=[Source(name="inbox", path=inbox)],
        path=demo_root / "config.json",
    )

    plan_result = plan_sources(config)

    import polylogue.run as run_mod
    import uuid

    run_mod.uuid4 = lambda: uuid.UUID("12345678-1234-5678-1234-567812345678")
    perf_values = iter([1.0, 1.234])
    run_mod.time.perf_counter = lambda: next(perf_values)

    run_result = run_sources(
        config=config,
        stage="all",
        plan=None,
        ui=None,
        source_names=None,
    )

    search_result = search_messages("pipeline", archive_root=archive_root, limit=2)

    def new_console(facade: ConsoleFacade) -> Console:
        console = Console(
            record=True,
            width=96,
            force_terminal=True,
            color_system="truecolor",
            theme=facade.theme,
            file=io.StringIO(),
        )
        facade.console = console
        return console

    assets_dir = Path("docs/assets")

    def render_plan(path: Path) -> None:
        facade = ConsoleFacade(plain=False)
        console = new_console(facade)
        console.print(Text("$ polylogue run --preview", style="bold #94a3b8"))
        console.print()
        facade.summary(
            "Preview",
            _dry_run_lines(
                plan_result.counts,
                plan_result.sources,
                plan_result.timestamp,
                plan_result.cursors,
            ),
        )
        console.save_svg(str(path))

    def render_run(path: Path) -> None:
        facade = ConsoleFacade(plain=False)
        console = new_console(facade)
        console.print(Text("$ polylogue run", style="bold #94a3b8"))
        console.print()
        facade.summary("Run", _run_lines(run_result))
        latest = max((archive_root / "render").rglob("conversation.html"), key=lambda p: p.stat().st_mtime)
        console.print(f"Latest render: {latest}")
        console.save_svg(str(path))

    def render_search(path: Path) -> None:
        facade = ConsoleFacade(plain=False)
        console = new_console(facade)
        console.print(Text("$ polylogue search \"pipeline\" --limit 2 --list", style="bold #94a3b8"))
        console.print()
        facade.summary("Search", ["Results: 2", "Query: pipeline"])
        display_root = Path("~/.local/share/polylogue/archive")
        for idx, hit in enumerate(search_result.hits, start=1):
            title = hit.title or hit.conversation_id
            source_label = hit.provider_name
            if hit.source_name and hit.source_name != hit.provider_name:
                source_label = f"{hit.source_name}/{hit.provider_name}"
            elif hit.source_name:
                source_label = hit.source_name
            console.print(f"{idx}. {title} ({source_label})")
            console.print(f"   {hit.snippet}")
            display_path = (
                display_root / "render" / hit.provider_name / hit.conversation_id / "conversation.md"
            )
            console.print(f"   {display_path}")
        console.save_svg(str(path))

    _write_svg(assets_dir / "cli-plan.svg", render_plan)
    _write_svg(assets_dir / "cli-run.svg", render_run)
    _write_svg(assets_dir / "cli-search.svg", render_search)


if __name__ == "__main__":
    main()
