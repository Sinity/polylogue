from __future__ import annotations

import time
from datetime import datetime, timezone

import csv
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, cast

from ..commands import CommandEnv, status_command
from ..version import POLYLOGUE_VERSION, SCHEMA_VERSION
from ..schema import stamp_payload
from ..util import parse_input_time_to_epoch
from .context import resolve_output_roots


def _dump_runs(ui, records: List[dict], destination: str, *, quiet: bool = False) -> None:
    payload = json.dumps(records, indent=2, sort_keys=True)
    if destination == "-":
        print(payload)
        return
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    if not quiet:
        ui.console.print(f"[green]Wrote {len(records)} run(s) to {path}")


def _dump_summary(ui, payload: Dict[str, Any], destination: str, *, quiet: bool = False) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True)
    if destination == "-":
        print(body)
        return
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    if not quiet:
        ui.console.print(f"[green]Wrote status summary to {path}")


def _provider_filter(raw: Optional[str]) -> Optional[Set[str]]:
    if not raw:
        return None
    providers = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return providers or None


def run_status_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    ui = env.ui
    json_lines = bool(getattr(args, "json_lines", False))
    json_mode = bool(getattr(args, "json", False) or json_lines)
    json_verbose = bool(getattr(args, "json_verbose", False))
    quiet_json = json_mode and not json_verbose
    show_inbox = bool(getattr(args, "inbox", False))
    if json_lines:
        setattr(args, "json", True)
        setattr(args, "quiet", True)

    dump_only = getattr(args, "dump_only", False)
    summary_only = getattr(args, "summary_only", False)
    provider_filter = _provider_filter(getattr(args, "providers", None))
    quiet = bool(getattr(args, "quiet", False))
    top_n = max(0, getattr(args, "top", 0))

    if summary_only and not getattr(args, "summary", None):
        setattr(args, "summary", "-")

    def emit() -> None:
        runs_limit = max(1, getattr(args, "runs_limit", 200))
        dump_requested = getattr(args, "dump", None)
        dump_limit = max(1, getattr(args, "dump_limit", 1)) if dump_requested else None
        effective_limit = runs_limit
        if dump_limit is not None:
            effective_limit = max(effective_limit, dump_limit)
        limit_arg = None if provider_filter else effective_limit
        result = status_command(env, runs_limit=limit_arg, provider_filter=provider_filter)
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
        top_runs: List[dict] = []
        if top_n:
            top_runs = sorted(filtered_runs, key=lambda r: (r.get("attachments", 0), r.get("tokens", 0)), reverse=True)[:top_n]

        inbox_summary = None
        if show_inbox:
            try:
                from ..local_sync import _discover_export_targets

                inbox_summary = {}
                for label, root in (("chatgpt", env.config.exports.chatgpt), ("claude", env.config.exports.claude)):
                    targets = _discover_export_targets(root, provider=label)
                    inbox_summary[label] = {"pending": len(targets), "root": str(root)}
            except Exception:
                inbox_summary = None
        console = ui.console

        if dump_only:
            destination = dump_requested or "-"
            limit = max(1, getattr(args, "dump_limit", 100))
            dump_records = filtered_runs[-limit:]
            _dump_runs(ui, dump_records, destination, quiet=quiet_json)
            return

        summary_requested = getattr(args, "summary", None)
        if summary_requested:
            summary_payload = stamp_payload(
                {
                    "generatedAt": datetime.now(timezone.utc).isoformat(),
                    "runSummary": run_summary,
                    "providerSummary": provider_summary,
                    "topRuns": top_runs,
                }
            )
            _dump_summary(ui, summary_payload, summary_requested, quiet=quiet_json)
            if summary_only:
                return
        if json_mode:
            payload = stamp_payload(
                {
                    "credentials_present": result.credentials_present,
                    "token_present": result.token_present,
                    "credential_path": str(result.credential_path),
                    "token_path": str(result.token_path),
                    "credential_env": result.credential_env,
                    "token_env": result.token_env,
                    "state_path": str(result.state_path),
                    "runs_path": str(result.runs_path),
                    "recent_runs": filtered_recent_runs,
                    "run_summary": run_summary,
                    "provider_summary": provider_summary,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "top_runs": top_runs,
                }
            )
            if json_lines:
                print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)
            else:
                print(json.dumps(payload, indent=2, sort_keys=True))
            return

        if ui.plain:
            console.print("Environment:")
            console.print(f"  credentials.json: {'present' if result.credentials_present else 'missing'} ({result.credential_path})")
            console.print(f"  token.json: {'present' if result.token_present else 'missing'} ({result.token_path})")
            if result.credential_env or result.token_env:
                console.print(f"  env overrides: cred={result.credential_env or '-'} token={result.token_env or '-'}")
            console.print(f"  state db: {result.state_path}")
            console.print(f"  runs log: {result.runs_path}")
            if run_summary and not quiet:
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
            if provider_summary and not quiet:
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
            table.add_row("credentials.json", f"{'present' if result.credentials_present else 'missing'} → {result.credential_path}")
            table.add_row("token.json", f"{'present' if result.token_present else 'missing'} → {result.token_path}")
            if result.credential_env or result.token_env:
                env_hint = f"cred={result.credential_env or '-'} token={result.token_env or '-'}"
                table.add_row("env overrides", env_hint)
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
            if inbox_summary:
                inbox_table = Table(title="Inbox Coverage", show_lines=False)
                inbox_table.add_column("Provider")
                inbox_table.add_column("Pending", justify="right")
                inbox_table.add_column("Root")
                for prov, stats in inbox_summary.items():
                    inbox_table.add_row(prov, str(stats.get("pending", 0)), str(stats.get("root", "-")))
                console.print(inbox_table)
            if top_runs:
                top_table = Table(title=f"Top Runs (attachments/tokens) [limit {len(top_runs)}]", show_lines=False)
                top_table.add_column("Timestamp")
                top_table.add_column("Provider")
                top_table.add_column("Attachments", justify="right")
                top_table.add_column("Tokens", justify="right")
                top_table.add_column("Command")
                for row in top_runs:
                    top_table.add_row(
                        str(row.get("timestamp", "-")),
                        str(row.get("provider", "-")),
                        str(row.get("attachments", 0)),
                        str(row.get("tokens", 0)),
                        str(row.get("cmd", "-")),
                    )
                console.print(top_table)

        if dump_requested:
            destination = dump_requested or "-"
            limit = max(1, getattr(args, "dump_limit", 100))
            dump_records = filtered_runs[-limit:]
            _dump_runs(ui, dump_records, destination, quiet=quiet_json)

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


def run_stats_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    from ..cli_common import sk_select
    from .path_policy import PathPolicy, resolve_path

    ui = env.ui
    console = ui.console
    json_lines = bool(getattr(args, "json_lines", False))
    csv_destination = getattr(args, "csv", None)
    json_mode = bool(getattr(args, "json", False) or json_lines)
    json_verbose = bool(getattr(args, "json_verbose", False))
    quiet_json = json_mode and not json_verbose
    sort_key = getattr(args, "sort", "tokens") or "tokens"
    limit = max(0, getattr(args, "limit", 0))
    empty_payload = {
        "directory": None,
        "directories": [],
        "totals": {"files": 0, "attachments": 0, "attachmentBytes": 0, "tokens": 0, "words": 0},
        "providers": {},
        "filteredOut": 0,
        "warnings": [],
    }
    warnings: List[str] = []

    def _fail(message: str, *, directory: Optional[Path] = None, directories: Optional[List[Path]] = None) -> None:
        payload = dict(empty_payload)
        payload["directory"] = str(directory) if directory else None
        if directories is not None:
            payload["directories"] = [str(item) for item in directories]
        if warnings:
            payload["warnings"] = warnings
        if json_mode:
            encoded = (
                json.dumps(payload, separators=(",", ":"), sort_keys=True)
                if json_lines
                else json.dumps(payload, indent=2, sort_keys=True)
            )
            print(encoded)
        else:
            ui.summary("Stats", [message])
        raise SystemExit(1)
    directory = None
    roots: List[Path] = []
    if args.dir:
        directory_input = Path(args.dir)
        if not directory_input.exists():
            return _fail("No Markdown files found (stats directory missing).", directory=directory_input, directories=[directory_input])
        directory = resolve_path(directory_input, PathPolicy.must_exist(), ui, json_mode=json_mode)
        if not directory:
            return _fail("No Markdown files found (stats directory missing).", directory=directory_input, directories=[directory_input])
        roots = [directory]
    else:
        roots = [path for path in resolve_output_roots(env.config) if path.exists()]
        if not roots:
            return _fail("No Markdown files found across configured output roots.")

    canonical_files: List[Path] = []
    legacy_files: List[Path] = []
    for root in roots:
        canonical_files.extend(root.rglob("conversation.md"))
        if not getattr(args, "ignore_legacy", False):
            legacy_files.extend(root.glob("*.md"))
    md_files = sorted(set(legacy_files + canonical_files))
    if not md_files:
        return _fail("No Markdown files found.", directories=roots, directory=directory)

    try:
        frontmatter_mod = importlib.import_module("frontmatter")
    except Exception:  # pragma: no cover
        frontmatter_mod = None

    def warn(message: str) -> None:
        warnings.append(message)
        if quiet_json:
            return
        console.print(f"[yellow]{message}")

    since_epoch = parse_input_time_to_epoch(getattr(args, "since", None))
    until_epoch = parse_input_time_to_epoch(getattr(args, "until", None))
    provider_filter = getattr(args, "provider", None)

    minimal_fallback = False

    def _load_metadata(path: Path) -> Dict[str, Any]:
        nonlocal minimal_fallback
        if frontmatter_mod is not None:
            try:
                post = cast(Any, frontmatter_mod).load(path)
                return dict(post.metadata)
            except Exception:
                warn(f"[yellow]frontmatter parse failed for {path}, falling back to minimal parser.")
        else:
            minimal_fallback = True
        # Minimal front matter parser using a permissive YAML load when available.
        try:
            import yaml  # type: ignore

            text = path.read_text(encoding="utf-8")
            if text.startswith("---"):
                # Extract front matter block
                lines = text.splitlines()
                fm_lines = []
                for line in lines[1:]:
                    if line.strip() == "---":
                        break
                    fm_lines.append(line)
                if fm_lines:
                    try:
                        parsed = yaml.safe_load("\n".join(fm_lines))
                        if isinstance(parsed, dict):
                            minimal_fallback = True
                            return parsed
                    except Exception:
                        pass
        except Exception:
            pass

        # Minimal key:value parser as last resort.
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
        minimal_fallback = True
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
                "path": str(path),
                "provider": provider,
                "attachments": int(attachment_count),
                "attachmentBytes": int(attachment_bytes),
                "tokens": int(tokens),
                "words": int(words),
                "timestamp": int(timestamp) if timestamp is not None else None,
            }
        )

    sorters = {
        "tokens": lambda row: row.get("tokens", 0),
        "attachments": lambda row: row.get("attachments", 0),
        "attachment-bytes": lambda row: row.get("attachmentBytes", 0),
        "words": lambda row: row.get("words", 0),
        "recent": lambda row: row.get("timestamp") or 0,
    }
    sorter = sorters.get(sort_key, sorters["tokens"])
    sorted_rows = sorted(rows, key=sorter, reverse=True)
    display_rows = sorted_rows[:limit] if limit else sorted_rows

    if minimal_fallback:
        warn("frontmatter metadata unavailable; used a minimal parser. Nested keys may be missing. Re-render to refresh metadata.")

    if json_lines:
        for row in display_rows:
            print(json.dumps(row, separators=(",", ":"), sort_keys=True))
        if warnings and quiet_json:
            print("\n".join(warnings), file=sys.stderr)
        return

    if getattr(args, "json", False):
        payload = {
            "cmd": "stats",
            "directory": str(directory) if directory else None,
            "directories": [str(root) for root in roots],
            "totals": totals,
            "providers": {k: per_provider[k] for k in sorted(per_provider)},
            "files": display_rows,
            "sort": sort_key,
            "limit": limit or None,
            "filteredOut": filtered_out,
            "filtered_out": filtered_out,
            "warnings": warnings,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if csv_destination:
        fieldnames = ["provider", "file", "path", "attachments", "attachmentBytes", "tokens", "words", "timestamp"]
        target = Path(csv_destination).expanduser()
        handle = sys.stdout if str(target) == "-" else None
        if handle is None:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(display_rows)
            console.print(f"[green]Wrote {len(display_rows)} row(s) to {target}")
        else:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(display_rows)

    lines = [
        "Directories:",
        *(f"  {root}" for root in roots),
        f"Files: {totals['files']} Attachments: {totals['attachments']} (~{totals['attachmentBytes'] / (1024 * 1024):.2f} MiB) Tokens≈ {totals['tokens']} (~{totals['words']} words)",
    ]
    if filtered_out:
        lines.append(f"Filtered out {filtered_out} file(s) outside date range.")
    if display_rows:
        pretty_sort = sort_key.replace("-", " ")
        lines.append(f"Top by {pretty_sort} (limit {len(display_rows)}):")
        for row in display_rows:
            lines.append(
                f"  {row['provider']}:{row['file']} attachments={row['attachments']} (~{row['attachmentBytes'] / (1024 * 1024):.2f} MiB) tokens={row['tokens']} words={row['words']}"
            )
    if warnings:
        lines.extend(warnings)
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
