from __future__ import annotations

import time
from datetime import datetime, timezone

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, cast

from ..commands import CommandEnv, status_command
from ..config import CONFIG_ENV, CONFIG_PATH, DEFAULT_PATHS
from ..util import parse_input_time_to_epoch
from .context import DEFAULT_OUTPUT_ROOTS, DEFAULT_RENDER_OUT


def _dump_runs(ui, records: List[dict], destination: str) -> None:
    payload = json.dumps(records, indent=2)
    if destination == "-":
        print(payload)
        return
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    ui.console.print(f"[green]Wrote {len(records)} run(s) to {path}")


def _dump_summary(ui, payload: Dict[str, Any], destination: str) -> None:
    body = json.dumps(payload, indent=2)
    if destination == "-":
        print(body)
        return
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    ui.console.print(f"[green]Wrote status summary to {path}")


def _provider_filter(raw: Optional[str]) -> Optional[Set[str]]:
    if not raw:
        return None
    providers = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return providers or None


def run_status_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    json_lines = bool(getattr(args, "json_lines", False))
    if json_lines:
        setattr(args, "json", True)

    dump_only = getattr(args, "dump_only", False)
    summary_only = getattr(args, "summary_only", False)
    provider_filter = _provider_filter(getattr(args, "providers", None))

    if summary_only and not getattr(args, "summary", None):
        setattr(args, "summary", "-")

    def emit() -> None:
        runs_limit = max(1, getattr(args, "runs_limit", 200))
        dump_requested = getattr(args, "dump", None)
        dump_limit = max(1, getattr(args, "dump_limit", 1)) if dump_requested else None
        effective_limit = runs_limit
        if dump_limit is not None:
            effective_limit = max(effective_limit, dump_limit)
        result = status_command(env, runs_limit=effective_limit)
        run_summary = result.run_summary
        provider_summary = result.provider_summary
        if provider_filter:
            run_summary = {
                cmd: stats
                for cmd, stats in run_summary.items()
                if (stats.get("provider") or "").lower() in provider_filter
            }
            provider_summary = {
                name: stats
                for name, stats in provider_summary.items()
                if name.lower() in provider_filter
            }

        def _matches(record: dict) -> bool:
            if not provider_filter:
                return True
            provider_value = (record.get("provider") or "").lower()
            return provider_value in provider_filter or not provider_value

        filtered_recent_runs = [record for record in result.recent_runs if _matches(record)]
        filtered_runs = [record for record in result.runs if _matches(record)]
        console = ui.console

        if dump_only:
            destination = dump_requested or "-"
            limit = max(1, getattr(args, "dump_limit", 100))
            dump_records = filtered_runs[-limit:]
            _dump_runs(ui, dump_records, destination)
            return

        summary_requested = getattr(args, "summary", None)
        if summary_requested:
            summary_payload = {
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "runSummary": run_summary,
                "providerSummary": provider_summary,
            }
            _dump_summary(ui, summary_payload, summary_requested)
            if summary_only:
                return

        json_mode = getattr(args, "json", False)
        if json_mode:
            payload = {
                "credentials_present": result.credentials_present,
                "token_present": result.token_present,
                "state_path": str(result.state_path),
                "runs_path": str(result.runs_path),
                "recent_runs": filtered_recent_runs,
                "run_summary": run_summary,
                "provider_summary": provider_summary,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            if json_lines:
                print(json.dumps(payload, separators=(",", ":")), flush=True)
            else:
                print(json.dumps(payload, indent=2))
            return

        if ui.plain:
            console.print("Environment:")
            console.print(f"  credentials.json: {'present' if result.credentials_present else 'missing'}")
            console.print(f"  token.json: {'present' if result.token_present else 'missing'}")
            console.print(f"  state db: {result.state_path}")
            console.print(f"  runs log: {result.runs_path}")
            if run_summary:
                console.print("Run summary:")
                for cmd, stats in run_summary.items():
                    console.print(
                        f"  {cmd}: runs={stats['count']} attachments={stats['attachments']} (~{stats['attachmentBytes'] / (1024 * 1024):.2f} MiB) tokens={stats['tokens']} (~{stats.get('words', 0)} words) diffs={stats['diffs']} retries={stats.get('retries', 0)} failures={stats.get('failures', 0)}"
                    )
                    if stats.get("last"):
                        console.print(
                            f"    last={stats['last']} out={stats['last_out']} count={stats['last_count']} skipped={stats['skipped']} pruned={stats['pruned']}"
                        )
                    if stats.get("last_error"):
                        console.print(f"    last_error={stats['last_error']}")
            if provider_summary:
                console.print("Provider summary:")
                for provider, stats in provider_summary.items():
                    console.print(
                        f"  {provider}: runs={stats['count']} attachments={stats['attachments']} (~{stats['attachmentBytes'] / (1024 * 1024):.2f} MiB) tokens={stats['tokens']} (~{stats.get('words', 0)} words) diffs={stats['diffs']} retries={stats.get('retries', 0)} failures={stats.get('failures', 0)}"
                    )
                    if stats.get("last"):
                        console.print(
                            f"    last={stats['last']} out={stats['last_out']} commands={', '.join(stats['commands'])}"
                        )
                    if stats.get("last_error"):
                        console.print(f"    last_error={stats['last_error']}")
        else:
            from rich.table import Table

            table = Table(title="Environment", show_lines=False)
            table.add_column("Item")
            table.add_column("Value")
            table.add_row("credentials.json", "present" if result.credentials_present else "missing")
            table.add_row("token.json", "present" if result.token_present else "missing")
            table.add_row("state db", str(result.state_path))
            table.add_row("runs log", str(result.runs_path))
            console.print(table)
            if run_summary:
                summary_table = Table(title="Run Summary", show_lines=False)
                summary_table.add_column("Command")
                summary_table.add_column("Runs", justify="right")
                summary_table.add_column("Attachments", justify="right")
                summary_table.add_column("Attachment MiB", justify="right")
                summary_table.add_column("Tokens (~words)", justify="right")
                summary_table.add_column("Diffs", justify="right")
                summary_table.add_column("Retries", justify="right")
                summary_table.add_column("Failures", justify="right")
                summary_table.add_column("Last Run", justify="left")
                for cmd, stats in run_summary.items():
                    summary_table.add_row(
                        cmd,
                        str(stats["count"]),
                        str(stats["attachments"]),
                        f"{stats['attachmentBytes'] / (1024 * 1024):.2f}",
                        f"{stats['tokens']} (~{stats.get('words', 0)} words)" if stats.get("tokens") else "0",
                        str(stats["diffs"]),
                        str(stats.get("retries", 0)),
                        str(stats.get("failures", 0)),
                        (stats.get("last") or "-") + (f" → {stats.get('last_out')}" if stats.get("last_out") else ""),
                    )
                console.print(summary_table)
            if provider_summary:
                provider_table = Table(title="Provider Summary", show_lines=False)
                provider_table.add_column("Provider")
                provider_table.add_column("Runs", justify="right")
                provider_table.add_column("Attachments", justify="right")
                provider_table.add_column("Attachment MiB", justify="right")
                provider_table.add_column("Tokens (~words)", justify="right")
                provider_table.add_column("Diffs", justify="right")
                provider_table.add_column("Retries", justify="right")
                provider_table.add_column("Failures", justify="right")
                provider_table.add_column("Commands")
                provider_table.add_column("Last Run", justify="left")
                for provider, stats in provider_summary.items():
                    provider_table.add_row(
                        provider,
                        str(stats["count"]),
                        str(stats["attachments"]),
                        f"{stats['attachmentBytes'] / (1024 * 1024):.2f}",
                        f"{stats['tokens']} (~{stats.get('words', 0)} words)" if stats.get("tokens") else "0",
                        str(stats["diffs"]),
                        str(stats.get("retries", 0)),
                        str(stats.get("failures", 0)),
                        ", ".join(stats.get("commands", [])),
                        (stats.get("last") or "-") + (f" → {stats.get('last_out')}" if stats.get("last_out") else ""),
                    )
                console.print(provider_table)

        if dump_requested:
            destination = dump_requested or "-"
            limit = max(1, getattr(args, "dump_limit", 100))
            dump_records = filtered_runs[-limit:]
            _dump_runs(ui, dump_records, destination)

    if getattr(args, "watch", False):
        interval = getattr(args, "interval", 5.0)
        try:
            while True:
                emit()
                time.sleep(interval)
        except KeyboardInterrupt:  # pragma: no cover - user interrupt
            console = ui.console
            console.print("[cyan]Status watch stopped.")
        return

    emit()


def run_stats_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    from ..cli_common import sk_select

    ui = env.ui
    console = ui.console
    directory = Path(args.dir) if args.dir else DEFAULT_RENDER_OUT
    if not directory.exists():
        console.print(f"[red]Directory not found: {directory}")
        console.print(f"Hint: Create it with: mkdir -p {directory}")
        raise SystemExit(1)

    canonical_files = list(directory.rglob("conversation.md"))
    legacy_files = list(directory.glob("*.md"))
    md_files = sorted(set(legacy_files + canonical_files))
    if not md_files:
        ui.summary("Stats", ["No Markdown files found."])
        return

    try:
        frontmatter_mod = importlib.import_module("frontmatter")
    except Exception:  # pragma: no cover
        frontmatter_mod = None

    since_epoch = parse_input_time_to_epoch(getattr(args, "since", None))
    until_epoch = parse_input_time_to_epoch(getattr(args, "until", None))
    provider_filter = getattr(args, "provider", None)

    def _load_metadata(path: Path) -> Dict[str, Any]:
        if frontmatter_mod is not None:
            try:
                post = cast(Any, frontmatter_mod).load(path)
                return dict(post.metadata)
            except Exception:
                pass
        # Minimal front matter parser.
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return {}
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}
        meta: Dict[str, Any] = {}
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"')
        return meta

    totals: Dict[str, Any] = {
        "files": 0,
        "attachments": 0,
        "attachmentBytes": 0,
        "tokens": 0,
        "words": 0,
    }
    per_provider: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    filtered_out = 0

    for path in md_files:
        meta = _load_metadata(path)
        attachment_count = meta.get("attachmentCount") or meta.get("attachments")
        if isinstance(attachment_count, list):
            attachment_count = len(attachment_count)
        if attachment_count is None:
            attachment_count = 0
        attachment_bytes = meta.get("attachmentBytes") or 0
        tokens = meta.get("totalTokensApprox") or meta.get("tokensApprox") or 0
        words = meta.get("totalWordsApprox") or 0
        provider = meta.get("sourcePlatform") or "unknown"

        timestamp = None
        for key in (
            "sourceModifiedTime",
            "sourceCreatedTime",
            "sourceModified",
            "sourceCreated",
        ):
            value = meta.get(key)
            epoch = parse_input_time_to_epoch(value) if value else None
            if epoch is not None:
                timestamp = epoch
                break
        if timestamp is None:
            try:
                timestamp = path.stat().st_mtime
            except OSError:
                timestamp = None

        if since_epoch is not None and (timestamp is None or timestamp < since_epoch):
            filtered_out += 1
            continue
        if until_epoch is not None and (timestamp is None or timestamp > until_epoch):
            filtered_out += 1
            continue
        if provider_filter is not None and provider.lower() != provider_filter.lower():
            filtered_out += 1
            continue

        totals["files"] += 1
        totals["attachments"] += int(attachment_count) if attachment_count else 0
        totals["attachmentBytes"] += int(attachment_bytes) if attachment_bytes else 0
        totals["tokens"] += int(tokens) if tokens else 0
        totals["words"] += int(words) if words else 0

        prov = per_provider.setdefault(
            provider,
            {"files": 0, "attachments": 0, "attachmentBytes": 0, "tokens": 0, "words": 0},
        )
        prov["files"] += 1
        prov["attachments"] += int(attachment_count) if attachment_count else 0
        prov["attachmentBytes"] += int(attachment_bytes) if attachment_bytes else 0
        prov["tokens"] += int(tokens) if tokens else 0
        prov["words"] += int(words) if words else 0

        rows.append(
            {
                "file": path.name,
                "provider": provider,
                "attachments": int(attachment_count),
                "attachmentBytes": int(attachment_bytes),
                "tokens": int(tokens),
                "words": int(words),
            }
        )

    if getattr(args, "json", False):
        payload = {
            "cmd": "stats",
            "directory": str(directory),
            "totals": totals,
            "providers": per_provider,
            "files": rows,
            "filteredOut": filtered_out,
        }
        print(json.dumps(payload, indent=2))
        return

    lines = [
        f"Directory: {directory}",
        f"Files: {totals['files']} Attachments: {totals['attachments']} (~{totals['attachmentBytes'] / (1024 * 1024):.2f} MiB) Tokens≈ {totals['tokens']} (~{totals['words']} words)",
    ]
    if filtered_out:
        lines.append(f"Filtered out {filtered_out} file(s) outside date range.")
    if not ui.plain:
        try:
            from rich.table import Table
            table = Table(title="Provider Summary")
            table.add_column("Provider")
            table.add_column("Files", justify="right")
            table.add_column("Attachments", justify="right")
            table.add_column("Attachment MiB", justify="right")
            table.add_column("Tokens (~words)", justify="right")
            for provider, data in per_provider.items():
                table.add_row(
                    provider,
                    str(data["files"]),
                    str(data["attachments"]),
                    f"{data['attachmentBytes'] / (1024 * 1024):.2f}",
                    f"{data['tokens']} (~{data.get('words', 0)} words)" if data.get("tokens") else "0",
                )
            console.print(table)
        except Exception:
            pass
    ui.summary("Stats", lines)


__all__ = ["run_status_cli", "run_stats_cli"]
