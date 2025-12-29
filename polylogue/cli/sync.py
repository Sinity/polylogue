from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
import shutil

from ..cli_common import filter_chats, sk_select
from ..commands import CommandEnv, build_drive_client, list_command, sync_command
from ..drive import snapshot_drive_metrics
from ..local_sync import (
    LocalSyncResult,
    LOCAL_SYNC_PROVIDER_NAMES,
    get_local_provider,
)
from ..options import ListOptions, SyncOptions
from ..util import add_run, format_run_brief, latest_run, path_order_key, get_run_by_id, sanitize_filename
from ..schema import stamp_payload
from .context import (
    DEFAULT_COLLAPSE,
    default_sync_namespace,
    resolve_collapse_thresholds,
    resolve_html_enabled,
    resolve_output_path,
    merge_with_defaults,
    parse_meta_items,
)
from .summaries import summarize_import
from .failure_logging import record_failure


def _truthy(val: str) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_default_output(provider_name: str, env: CommandEnv) -> Path:
    outputs = env.config.defaults.output_dirs
    mapping = {
        "drive": outputs.sync_drive,
        "codex": outputs.sync_codex,
        "claude-code": outputs.sync_claude_code,
        "chatgpt": outputs.import_chatgpt,
        "claude": outputs.import_claude,
    }
    return mapping.get(provider_name, outputs.render)


def _apply_resume_from(args: SimpleNamespace, env: CommandEnv, *, run_id: int) -> None:
    run = get_run_by_id(int(run_id))
    if not run:
        raise SystemExit(f"Unknown run id: {run_id}")
    cmd = str(run.get("cmd") or "")
    provider = getattr(args, "provider", None)
    if cmd and not cmd.startswith("sync"):
        raise SystemExit(f"Run #{run_id} is not a sync run (cmd={cmd!r}).")

    if provider == "drive":
        failed = run.get("failedChats")
        failed_attachments = run.get("failedAttachments")
        ids: List[str] = []
        stage_counts: Dict[str, int] = {}
        attachment_counts: Dict[str, int] = {}
        if isinstance(failed, list):
            for item in failed:
                if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
                    ids.append(item["id"].strip())
                    stage = item.get("stage")
                    if isinstance(stage, str) and stage.strip():
                        stage_counts[stage.strip()] = stage_counts.get(stage.strip(), 0) + 1
        if isinstance(failed_attachments, list):
            for item in failed_attachments:
                if not isinstance(item, dict):
                    continue
                chat_id = item.get("id")
                if isinstance(chat_id, str) and chat_id.strip():
                    ids.append(chat_id.strip())
                attachments = item.get("attachments")
                if isinstance(chat_id, str) and chat_id.strip() and isinstance(attachments, list):
                    attachment_counts[chat_id.strip()] = len([x for x in attachments if x])
        if not ids:
            raise SystemExit(f"Run #{run_id} has no recorded failed Drive chats/attachments to resume.")
        if getattr(args, "links_only", False) and failed_attachments:
            raise SystemExit("Cannot resume attachment retries with --links-only (it disables attachment downloads).")
        args.chat_ids = list(dict.fromkeys(ids))
        args.resume_from = int(run_id)
        args.all = True
        args.prune = False
        extra = ""
        if stage_counts:
            stage_summary = ", ".join(f"{name}={count}" for name, count in sorted(stage_counts.items()))
            extra = f" ({stage_summary})"
        if attachment_counts:
            attachment_total = sum(attachment_counts.values())
            extra = f"{extra} (attachment failures={attachment_total})"
        env.ui.console.print(f"[dim]Resuming run #{run_id}: {len(args.chat_ids)} Drive chat(s){extra}[/dim]")
        return

    if provider in LOCAL_SYNC_PROVIDER_NAMES:
        failed = run.get("failedPaths")
        paths: List[Path] = []
        error_counts: Dict[str, int] = {}
        if isinstance(failed, list):
            for item in failed:
                if isinstance(item, str) and item.strip():
                    paths.append(Path(item.strip()))
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    paths.append(Path(item["path"]))
                    err = item.get("error")
                    if isinstance(err, str) and err.strip():
                        key = err.strip()
                        error_counts[key] = error_counts.get(key, 0) + 1
        if not paths:
            raise SystemExit(f"Run #{run_id} has no recorded failed local paths to resume.")
        args.sessions = [str(p) for p in paths]
        args.resume_from = int(run_id)
        args.all = True
        args.prune = False
        extra = ""
        if error_counts:
            top = sorted(error_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            extra = " (" + ", ".join(f"{count}×{err}" for err, count in top) + ")"
        env.ui.console.print(f"[dim]Resuming run #{run_id}: {len(args.sessions)} path(s){extra}[/dim]")
        return

    raise SystemExit(f"--resume-from is not supported for provider={provider!r}")


def _retry_drive_attachments_only(
    *,
    run: dict,
    env: CommandEnv,
    output_dir: Path,
    dry_run: bool,
    force: bool,
) -> dict:
    ui = env.ui
    drive = env.drive or build_drive_client(env)
    env.drive = drive

    failed_attachments = run.get("failedAttachments")
    if not isinstance(failed_attachments, list) or not failed_attachments:
        raise SystemExit("Run has no recorded failedAttachments to retry.")

    attempted = 0
    downloaded = 0
    skipped = 0
    failures: List[Dict[str, object]] = []

    for entry in failed_attachments:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug")
        name = entry.get("name")
        if not isinstance(slug, str) or not slug.strip():
            if isinstance(name, str) and name.strip():
                slug = sanitize_filename(name)
            else:
                continue
        slug = slug.strip()
        convo_dir = output_dir / slug
        attachments = entry.get("attachments")
        if not isinstance(attachments, list) or not attachments:
            continue

        for att in attachments:
            att_id: Optional[str] = None
            rel_path: Optional[str] = None
            if isinstance(att, dict):
                att_id = str(att.get("id") or "").strip() or None
                rel_path_raw = att.get("path")
                if isinstance(rel_path_raw, str) and rel_path_raw.strip():
                    rel_path = rel_path_raw.strip()
                else:
                    filename = att.get("filename")
                    if isinstance(filename, str) and filename.strip():
                        rel_path = str(Path("attachments") / filename.strip())
            elif isinstance(att, str) and att.strip():
                att_id = att.strip()

            if not att_id:
                continue

            if rel_path is None:
                meta = drive.attachment_meta(att_id)
                filename = None
                if isinstance(meta, dict):
                    raw_name = meta.get("name")
                    if isinstance(raw_name, str) and raw_name.strip():
                        filename = sanitize_filename(raw_name)
                if not filename:
                    filename = sanitize_filename(att_id)
                rel_path = str(Path("attachments") / filename)

            target_path = (convo_dir / rel_path).resolve()
            attempted += 1
            if target_path.exists() and not force:
                skipped += 1
                continue

            if dry_run:
                ui.console.print(f"[yellow][dry-run] Would download attachment {att_id} → {target_path}[/yellow]")
                downloaded += 1
                continue

            meta = drive.attachment_meta(att_id)
            ok = drive.download_attachment(att_id, target_path)
            if not ok:
                err = snapshot_drive_metrics(reset=False).get("lastError")
                failure = {"id": entry.get("id"), "slug": slug, "attachment": att_id, "path": str(target_path)}
                if isinstance(err, str) and err.strip():
                    failure["error"] = err.strip()
                failures.append(failure)
                continue
            downloaded += 1
            if isinstance(meta, dict):
                drive.touch_mtime(target_path, meta.get("modifiedTime"))

    payload = {
        "cmd": "sync drive",
        "provider": "drive",
        "attachmentsOnly": True,
        "count": 0,
        "out": str(output_dir),
        "attempted": attempted,
        "attachmentDownloads": downloaded,
        "skipped": skipped,
        "failures": len(failures),
    }
    if failures:
        payload["failedAttachmentDownloads"] = failures
    add_run(payload)
    return payload


def _apply_sync_prefs(args: SimpleNamespace, env: CommandEnv) -> SimpleNamespace:
    prefs = getattr(env, "prefs", {}) or {}
    sync_prefs = prefs.get("sync", {}) if isinstance(prefs, dict) else {}
    if not sync_prefs:
        return args

    def _apply_flag(flag: str, attr: str) -> None:
        if flag in sync_prefs and not getattr(args, attr, False) and _truthy(sync_prefs[flag]):
            setattr(args, attr, True)

    if "--html" in sync_prefs and getattr(args, "html_mode", "auto") == "auto":
        args.html_mode = "on" if _truthy(sync_prefs["--html"]) else "off"

    _apply_flag("--links-only", "links_only")
    _apply_flag("--diff", "diff")
    _apply_flag("--prune", "prune")
    _apply_flag("--watch", "watch")
    _apply_flag("--once", "once")
    _apply_flag("--offline", "offline")
    _apply_flag("--sanitize-html", "sanitize_html")
    if "--attachment-ocr" in sync_prefs and not getattr(args, "_attachment_ocr_explicit", False):
        args.attachment_ocr = _truthy(sync_prefs["--attachment-ocr"])
    if "--root" in sync_prefs and not getattr(args, "root", None):
        root_label = str(sync_prefs.get("--root") or "").strip()
        if root_label:
            args.root = root_label
    return args


def _log_local_sync(
    ui,
    title: str,
    result: LocalSyncResult,
    *,
    provider: str,
    footer: Optional[List[str]] = None,
    redacted: bool = False,
    meta: Optional[Dict[str, str]] = None,
) -> None:
    console = ui.console
    if result.written:
        summarize_import(ui, title, result.written, extra_lines=footer)
    else:
        console.print(f"[cyan]{title}: no new Markdown files.")
        if footer:
            for line in footer:
                console.print(line)
    skip_reasons = getattr(result, "skip_reasons", None) or {}
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items()):
            console.print(f"[cyan]{title}: skipped ({reason}) {count} item(s).")
    elif result.skipped:
        console.print(f"[cyan]{title}: skipped {result.skipped} up-to-date item(s).")
    if result.pruned:
        console.print(f"[cyan]{title}: pruned {result.pruned} path(s).")
    if getattr(result, "ignored", 0):
        console.print(f"[yellow]{title}: skipped {result.ignored} path(s) via .polylogueignore.")
    run_payload = {
        "cmd": f"sync {provider}",
        "provider": provider,
        "count": len(result.written),
        "out": str(result.output_dir),
        "attachments": result.attachments,
        "attachmentBytes": result.attachment_bytes,
        "tokens": result.tokens,
        "words": result.words,
        "diffs": result.diffs,
        "skipped": result.skipped,
        "pruned": result.pruned,
        "duration": getattr(result, "duration", 0.0),
    }
    if skip_reasons:
        run_payload["skipReasons"] = dict(skip_reasons)
    failed_items = getattr(result, "failed", None)
    if isinstance(failed_items, list) and failed_items:
        run_payload["failedPaths"] = failed_items
    failed_count = getattr(result, "failures", 0) or 0
    if failed_count:
        run_payload["failures"] = int(failed_count)
    if meta:
        run_payload["meta"] = dict(meta)
    if redacted:
        run_payload["redacted"] = True
    add_run(run_payload)


def run_list_cli(args: SimpleNamespace, env: CommandEnv, json_output: bool) -> None:
    options = ListOptions(
        folder_name=args.folder_name,
        folder_id=args.folder_id,
        since=args.since,
        until=args.until,
        name_filter=args.name_filter,
    )
    result = list_command(options, env)
    if json_output:
        payload = stamp_payload(
            {
                "cmd": "list",
                "folder_name": result.folder_name,
                "folder_id": result.folder_id,
                "count": len(result.files),
                "files": result.files,
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    console = env.ui.console
    console.print(f"{len(result.files)} chat(s) in {result.folder_name}:")
    for chat in result.files:
        console.print(f"- {chat.get('name')}  {chat.get('modifiedTime', '')}  {chat.get('id', '')}")


def run_sync_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    provider = getattr(args, "provider", None)
    settings = env.settings
    args = _apply_sync_prefs(args, env)
    if getattr(args, "attachment_ocr", None) is None:
        args.attachment_ocr = True
    provider = getattr(args, "provider", None)

    resume_from = getattr(args, "resume_from", None)
    if resume_from is not None:
        _apply_resume_from(args, env, run_id=resume_from)

    if getattr(args, "watch", False):
        if provider == "drive":
            raise SystemExit("Drive does not support --watch; use local providers like codex/claude-code/chatgpt.")
        from ..local_sync import get_local_provider
        provider_obj = get_local_provider(provider)
        if not provider_obj.supports_watch:
            raise SystemExit(
                f"{provider_obj.title} does not support watch mode "
                f"(use --watch with codex, claude-code, or chatgpt)"
            )
        from .watch import run_watch_cli

        run_watch_cli(args, env)
        return

    if getattr(args, "offline", False) and provider == "drive":
        env.ui.console.print("[red]Drive sync does not support --offline.")
        raise SystemExit(1)

    if getattr(args, "root", None):
        label = getattr(args, "root")
        defaults = env.config.defaults
        roots = getattr(defaults, "roots", {}) or {}
        paths = roots.get(label)
        if not paths:
            raise SystemExit(f"Unknown root label '{label}'. Define it in config or use a known label.")
        env.config.defaults.output_dirs = paths

    if provider == "drive":
        from ..drive_client import DEFAULT_CREDENTIALS

        cred_path = DEFAULT_CREDENTIALS
        if env.config.drive and env.config.drive.credentials_path:
            cred_path = env.config.drive.credentials_path
        if not cred_path.exists():
            raise SystemExit(
                f"Drive sync requires credentials.json at {cred_path} "
                f"(set drive.credentials_path in config)."
            )

    if provider in {"chatgpt", "claude"}:
        if getattr(args, "base_dir", None):
            exports_root = Path(getattr(args, "base_dir")).expanduser()
            hint = "create it or pass a valid --base-dir"
        else:
            exports_root = env.config.exports.chatgpt if provider == "chatgpt" else env.config.exports.claude
            hint = f"set exports.{provider} in config or pass --base-dir"
        if not exports_root.exists():
            raise SystemExit(f"{provider} exports directory not found: {exports_root} ({hint}).")
    if provider == "drive":
        merged = merge_with_defaults(default_sync_namespace("drive", settings), args)
        _run_sync_drive(merged, env)
    elif provider in LOCAL_SYNC_PROVIDER_NAMES:
        merged = merge_with_defaults(default_sync_namespace(provider, settings), args)
        if getattr(args, "offline", False):
            env.ui.console.print("[yellow]Offline mode: network-dependent steps will be skipped; results may be incomplete.")
        _run_local_sync(provider, merged, env)
    else:
        env.ui.console.print(f"[red]Unsupported provider for sync: {provider}")
        raise SystemExit(1)


def _run_sync_drive(args: SimpleNamespace, env: CommandEnv) -> None:
    ui = env.ui
    console = ui.console
    json_mode = getattr(args, "json", False)

    if getattr(args, "list_only", False):
        list_args = SimpleNamespace(
            folder_name=args.folder_name,
            folder_id=args.folder_id,
            since=args.since,
            until=args.until,
            name_filter=args.name_filter,
        )
        run_list_cli(list_args, env, json_output=json_mode)
        return

    if getattr(args, "attachments_only", False):
        run_id = getattr(args, "resume_from", None)
        if run_id is None:
            raise SystemExit("--attachments-only requires --resume-from <run-id> (to load failedAttachments).")
        if getattr(args, "links_only", False):
            raise SystemExit("--attachments-only cannot be combined with --links-only.")
        if getattr(args, "offline", False):
            raise SystemExit("--attachments-only does not support --offline for Drive.")

        run = get_run_by_id(int(run_id))
        if not run:
            raise SystemExit(f"Unknown run id: {run_id}")
        cmd = str(run.get("cmd") or "")
        if not cmd.startswith("sync drive"):
            raise SystemExit(f"Run #{run_id} is not a sync drive run (cmd={cmd!r}).")
        if run.get("failedChats"):
            raise SystemExit("Run has failedChats; rerun without --attachments-only to retry full chat processing.")

        out = args.out
        if out is None:
            stored = run.get("out")
            if isinstance(stored, str) and stored.strip():
                out = stored.strip()
        output_dir = resolve_output_path(out, _resolve_default_output("drive", env))
        output_dir.mkdir(parents=True, exist_ok=True)

        retries_override = getattr(args, "drive_retries", None)
        retry_base_override = getattr(args, "drive_retry_base", None)
        drive_cfg = getattr(env.config, "drive", None)
        drive_retries = retries_override if retries_override is not None else getattr(drive_cfg, "retries", None)
        drive_retry_base = retry_base_override if retry_base_override is not None else getattr(drive_cfg, "retry_base", None)

        env.drive = env.drive or build_drive_client(env, retries=drive_retries, retry_base=drive_retry_base)
        result_payload = _retry_drive_attachments_only(
            run=run,
            env=env,
            output_dir=output_dir,
            dry_run=bool(getattr(args, "dry_run", False)),
            force=bool(getattr(args, "force", False)),
        )
        if json_mode:
            print(json.dumps(stamp_payload(result_payload), indent=2, sort_keys=True))
            return
        console.print(
            f"Attachment retry complete: downloaded={result_payload.get('attachmentDownloads', 0)} "
            f"skipped={result_payload.get('skipped', 0)} failures={result_payload.get('failures', 0)}"
        )
        return

    download_attachments = not args.links_only
    if getattr(args, "offline", False):
        download_attachments = False

    previous_run_note = format_run_brief(latest_run(provider="drive", cmd="sync drive"))

    retries_override = getattr(args, "drive_retries", None)
    retry_base_override = getattr(args, "drive_retry_base", None)
    drive_cfg = getattr(env.config, "drive", None)
    drive_retries = retries_override if retries_override is not None else getattr(drive_cfg, "retries", None)
    drive_retry_base = retry_base_override if retry_base_override is not None else getattr(drive_cfg, "retry_base", None)

    drive = env.drive or build_drive_client(env, retries=drive_retries, retry_base=drive_retry_base)
    env.drive = drive
    folder_id = drive.resolve_folder_id(args.folder_name, args.folder_id)
    raw_chats = drive.list_chats(args.folder_name, folder_id)
    filtered = filter_chats(raw_chats, args.name_filter, args.since, args.until)

    cli_ids = [item.strip() for item in getattr(args, "chat_ids", []) if item and item.strip()]
    selected_ids: Optional[List[str]] = cli_ids or None
    if selected_ids is None and not ui.plain and not json_mode:
        if not filtered:
            console.print("No chats to sync")
            return
        if not getattr(args, "all", False):
            lines = [
                f"{c.get('name') or '(untitled)'}\t{c.get('modifiedTime') or ''}\t{c.get('id')}"
                for c in filtered
            ]
            selection = sk_select(lines, preview="printf '%s' {+}", plain=ui.plain)
            if selection is None:
                console.print("[yellow]Sync cancelled; no chats selected.")
                return
            if not selection:
                console.print("[yellow]No chats selected; nothing to sync.")
                return
            selected_ids = [line.split("\t")[-1] for line in selection]

    settings = env.settings
    collapse_thresholds = resolve_collapse_thresholds(args, settings)
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    meta = parse_meta_items(getattr(args, "meta", None)) or None
    prefetched = filtered
    if selected_ids:
        selected_set = set(selected_ids)
        prefetched = [chat for chat in filtered if chat.get("id") in selected_set]
    options = SyncOptions(
        folder_name=args.folder_name,
        folder_id=folder_id,
        output_dir=resolve_output_path(args.out, _resolve_default_output("drive", env)),
        collapse_threshold=collapse_thresholds["message"],
        collapse_thresholds=collapse_thresholds,
        download_attachments=download_attachments,
        dry_run=args.dry_run,
        force=args.force,
        prune=args.prune,
        since=args.since,
        until=args.until,
        name_filter=args.name_filter,
        selected_ids=selected_ids,
        html=html_enabled,
        html_theme=html_theme,
        diff=getattr(args, "diff", False),
        prefetched_chats=prefetched,
        attachment_ocr=getattr(args, "attachment_ocr", True),
        sanitize_html=getattr(args, "sanitize_html", False),
        meta=meta,
    )

    if getattr(args, "prune", False) and getattr(args, "prune_snapshot", False):
        from ..paths import STATE_HOME
        from ..util import preflight_disk_requirement
        output_dir = options.output_dir
        snapshot_root = STATE_HOME / "rollback"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        estimated = 0
        for path in output_dir.rglob("*"):
            try:
                if path.is_file():
                    estimated += path.stat().st_size
            except Exception:
                continue
        preflight_disk_requirement(projected_bytes=estimated, limit_gib=getattr(args, "max_disk", None), ui=ui)
        snapshot_path = snapshot_root / f"sync-drive-{int(time.time())}.zip"
        try:
            from zipfile import ZipFile

            with ZipFile(snapshot_path, "w") as zipf:
                for path in output_dir.rglob("*"):
                    try:
                        if path.is_file():
                            zipf.write(path, arcname=path.relative_to(output_dir))
                    except Exception:
                        continue
            ui.console.print(f"[dim]Prune snapshot saved to {snapshot_path}[/dim]")
        except Exception as exc:
            ui.console.print(f"[yellow]Snapshot failed: {exc}")

    try:
        result = sync_command(options, env)
    except Exception as exc:
        console.print(f"[red]Drive sync failed: {exc}")
        console.print("[cyan]Run `polylogue doctor` and `polylogue config show --json` to verify credentials, tokens, and output directories.")
        record_failure(args, exc, phase="sync")
        raise

    if json_mode:
        payload = {
            "cmd": "sync drive",
            "provider": "drive",
            "count": result.count,
            "out": str(result.output_dir),
            "folder_name": result.folder_name,
            "folder_id": result.folder_id,
            "files": [
                {
                    "id": item.id,
                    "name": item.name,
                    "output": str(item.output),
                    "slug": item.slug,
                    "attachments": item.attachments,
                    "stats": item.stats,
                    "html": str(item.html) if item.html else None,
                    "diff": str(item.diff) if getattr(item, "diff", None) else None,
                }
                for item in result.items
            ],
            "total_stats": result.total_stats,
            "retries": getattr(result, "retries", None),
            "retry_base": drive_retry_base,
        }
        if getattr(args, "resume_from", None) is not None:
            payload["resumeFrom"] = int(getattr(args, "resume_from") or 0)
        failed_chats = getattr(result, "failed_chats", None)
        if isinstance(failed_chats, list) and failed_chats:
            payload["failedChats"] = failed_chats
            payload["failures"] = len(failed_chats)
        failed_attachments = getattr(result, "failed_attachments", None)
        if isinstance(failed_attachments, list) and failed_attachments:
            payload["failedAttachments"] = failed_attachments
            payload["attachmentFailures"] = int(result.total_stats.get("attachmentFailures", 0) or 0)
        if meta:
            payload["meta"] = dict(meta)
        if getattr(args, "sanitize_html", False):
            payload["redacted"] = True
        if previous_run_note:
            payload["previousRun"] = previous_run_note
        print(json.dumps(stamp_payload(payload), indent=2, sort_keys=True))
        return

    lines = [f"Synced {result.count} chat(s) → {result.output_dir}"]
    failed_chats = getattr(result, "failed_chats", None)
    if isinstance(failed_chats, list) and failed_chats:
        lines.append(f"Failures: {len(failed_chats)}")
    attachment_failures_total = int(result.total_stats.get("attachmentFailures", 0) or 0)
    if attachment_failures_total:
        lines.append(f"Attachment failures: {attachment_failures_total}")
    if getattr(args, "print_paths", False):
        lines.append("Written paths:")
        for item in result.items:
            lines.append(f"  {item.output}")
    attachments_total = result.total_stats.get("attachments", 0)
    if attachments_total:
        lines.append(f"Attachments: {attachments_total}")
    skipped_total = result.total_stats.get("skipped", 0)
    if skipped_total:
        lines.append(f"Skipped: {skipped_total}")
    if "totalTokensApprox" in result.total_stats:
        total_tokens = int(result.total_stats["totalTokensApprox"])
        total_words = int(result.total_stats.get("totalWordsApprox", 0) or 0)
        if total_words:
            lines.append(f"Approx tokens: {total_tokens} (~{total_words} words)")
        else:
            lines.append(f"Approx tokens: {total_tokens}")
    for key, label in (
        ("chunkCount", "Total chunks"),
        ("userTurns", "User turns"),
        ("modelTurns", "Model turns"),
    ):
        value = result.total_stats.get(key)
        if value:
            lines.append(f"{label}: {int(value)}")
    for item in result.items:
        info = f"- {item.slug} (attachments: {item.attachments})"
        if item.html:
            info += " [+html]"
        if getattr(item, "diff", None):
            info += " [+diff]"
        lines.append(info)
    if drive_retries is not None or drive_retry_base is not None:
        lines.append(f"Drive retries: {drive_retries or 3} (base delay {drive_retry_base or 0.5}s)")
    if previous_run_note:
        lines.append(f"Previous run: {previous_run_note}")
    console.print("\n".join(lines))


def _collect_session_selection(ui, sessions: List[Path], header: str) -> Optional[List[Path]]:
    console = ui.console
    if not sessions:
        console.print("No sessions found.")
        return None
    name_width = min(max(len(path.stem) for path in sessions), 72)
    parent_width = min(max(len(path.parent.name) for path in sessions), 24)
    lines = [
        f"{path.stem[:name_width]:<{name_width}}\t{path.parent.name[:parent_width]:<{parent_width}}\t{path}"
        for path in sessions
    ]
    selection = sk_select(
        lines,
        header=f"{header} — tab to toggle, ctrl-a select all, enter accept",
        prompt="Sessions> ",
        plain=ui.plain,
    )
    if selection is None:
        console.print("[yellow]Sync cancelled; no sessions selected.")
        return None
    if not selection:
        console.print("[yellow]No sessions selected; nothing to do.")
        return []
    return [Path(line.split("\t")[-1]) for line in selection]


def _run_local_sync(provider_name: str, args: SimpleNamespace, env: CommandEnv) -> None:
    provider = get_local_provider(provider_name)
    ui = env.ui
    cmd_name = f"sync {provider.name}"
    previous = latest_run(provider=provider.name, cmd=cmd_name)
    if previous is None:
        legacy_cmd = provider.title.lower().replace(" ", "-")
        previous = latest_run(provider=provider.name, cmd=legacy_cmd)
    previous_run_note = format_run_brief(previous)
    if getattr(args, "diff", False) and not provider.supports_diff:
        ui.console.print(f"[red]{provider.title} does not support --diff output")
        raise SystemExit(1)
    if args.base_dir:
        base_dir = Path(args.base_dir).expanduser()
    elif provider.name == "chatgpt":
        base_dir = env.config.exports.chatgpt
    elif provider.name == "claude":
        base_dir = env.config.exports.claude
    else:
        base_dir = provider.default_base.expanduser()
    out_dir = resolve_output_path(args.out, _resolve_default_output(provider.name, env))
    settings = env.settings
    collapse_thresholds = resolve_collapse_thresholds(args, settings)
    collapse = collapse_thresholds["message"]
    html_enabled = resolve_html_enabled(args, settings)
    html_theme = settings.html_theme
    force = args.force
    prune = args.prune
    diff_enabled = getattr(args, "diff", False)
    meta = parse_meta_items(getattr(args, "meta", None)) or None
    jobs = max(1, int(getattr(args, "jobs", 1) or 1))

    if not base_dir.exists():
        if provider.create_base_dir and args.base_dir is None:
            base_dir.mkdir(parents=True, exist_ok=True)
        else:
            hint = f"pass --base-dir to override (current default: {base_dir})" if args.base_dir is None else "double-check the path or create it before running"
            ui.console.print(f"[red]Base directory does not exist: {base_dir}[/red]")
            ui.console.print(f"[yellow]{hint}[/yellow]")
            raise SystemExit(1)

    selected_paths: Optional[List[Path]] = None
    cli_sessions = getattr(args, "sessions", None)
    if cli_sessions:
        selected_paths = [Path(path).expanduser() for path in cli_sessions if path]
    elif not args.all and not ui.plain:
        sessions = provider.list_sessions(base_dir)
        selection = _collect_session_selection(ui, sessions, f"Select {provider.title} sessions")
        if selection is None:
            return
        if not selection:
            return
        selected_paths = selection

    disk_estimate = bool(getattr(args, "disk_estimate", False))
    max_disk = getattr(args, "max_disk", None)
    disk_estimate_bytes: Optional[int] = None
    disk_free_bytes: Optional[int] = None
    if disk_estimate or max_disk is not None:
        if selected_paths is not None:
            session_count = len(selected_paths)
        else:
            session_count = len(provider.list_sessions(base_dir))
        projected = 20 * 1024 * 1024 * max(1, session_count)
        disk_estimate_bytes = projected
        try:
            disk_free_bytes = int(shutil.disk_usage(Path.cwd()).free)
        except Exception:
            disk_free_bytes = None
        if not getattr(args, "json", False):
            extra = f", free={disk_free_bytes / (1024 ** 3):.2f} GiB" if disk_free_bytes is not None else ""
            limit = f", limit={max_disk:.2f} GiB" if max_disk is not None else ""
            ui.console.print(f"[dim]Disk estimate: projected={projected / (1024 ** 3):.2f} GiB{limit}{extra}[/dim]")
        if max_disk is not None:
            from ..util import preflight_disk_requirement

            preflight_disk_requirement(projected_bytes=projected, limit_gib=max_disk, ui=ui)

    if prune and getattr(args, "prune_snapshot", False):
        from ..paths import STATE_HOME
        from ..util import preflight_disk_requirement
        snapshot_root = STATE_HOME / "rollback"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        estimated = 0
        for path in out_dir.rglob("*"):
            try:
                if path.is_file():
                    estimated += path.stat().st_size
            except Exception:
                continue
        preflight_disk_requirement(projected_bytes=estimated, limit_gib=getattr(args, "max_disk", None), ui=ui)
        snapshot_path = snapshot_root / f"sync-{provider.name}-{int(time.time())}.zip"
        try:
            from zipfile import ZipFile

            with ZipFile(snapshot_path, "w") as zipf:
                for path in out_dir.rglob("*"):
                    try:
                        if path.is_file():
                            zipf.write(path, arcname=path.relative_to(out_dir))
                    except Exception:
                        continue
            ui.console.print(f"[dim]Prune snapshot saved to {snapshot_path}[/dim]")
        except Exception as exc:
            ui.console.print(f"[yellow]Snapshot failed: {exc}")

    try:
        sync_kwargs = dict(
            base_dir=base_dir,
            output_dir=out_dir,
            collapse_threshold=collapse,
            collapse_thresholds=collapse_thresholds,
            html=html_enabled,
            html_theme=html_theme,
            force=force,
            prune=prune,
            diff=diff_enabled,
            sessions=selected_paths,
            registrar=env.registrar,
            ui=env.ui,
            attachment_ocr=getattr(args, "attachment_ocr", True),
            sanitize_html=getattr(args, "sanitize_html", False),
            meta=meta,
        )
        if getattr(provider, "supports_jobs", False):
            sync_kwargs["jobs"] = jobs
        result = provider.sync_fn(**sync_kwargs)
    except Exception as exc:
        ui.console.print(f"[red]{provider.title} sync failed: {exc}")
        raise SystemExit(1) from exc

    attachments = result.attachments
    attachment_bytes = result.attachment_bytes
    tokens = result.tokens
    words = result.words

    if getattr(args, "json", False):
        files_payload = []
        for item in result.written:
            doc = item.document
            files_payload.append(
                {
                    "output": str(item.markdown_path),
                    "attachments": len(doc.attachments) if doc else 0,
                    "attachmentBytes": doc.metadata.get("attachmentBytes") if doc and doc.metadata else None,
                    "stats": doc.stats if doc and doc.stats else {},
                    "html": str(item.html_path) if item.html_path else None,
                    "diff": str(item.diff_path) if item.diff_path else None,
                }
            )
        skip_reasons = getattr(result, "skip_reasons", None)
        payload = {
            "cmd": f"sync {provider.name}",
            "provider": provider.name,
            "count": len(result.written),
            "out": str(result.output_dir),
            "skipped": result.skipped,
            "skipReasons": dict(skip_reasons or {}),
            "pruned": result.pruned,
            "attachments": attachments,
            "attachmentBytes": attachment_bytes,
            "tokensApprox": tokens,
            "wordsApprox": words,
            "diffs": result.diffs,
            "files": files_payload,
        }
        if disk_estimate_bytes is not None:
            payload["diskEstimateBytes"] = int(disk_estimate_bytes)
        if disk_free_bytes is not None:
            payload["diskFreeBytes"] = int(disk_free_bytes)
        if max_disk is not None:
            payload["maxDiskGiB"] = float(max_disk)
        if getattr(args, "resume_from", None) is not None:
            payload["resumeFrom"] = int(getattr(args, "resume_from") or 0)
        if getattr(result, "failures", 0):
            payload["failures"] = int(getattr(result, "failures") or 0)
        failed_items = getattr(result, "failed", None)
        if isinstance(failed_items, list) and failed_items:
            payload["failedPaths"] = failed_items
        if meta:
            payload["meta"] = dict(meta)
        if getattr(args, "sanitize_html", False):
            payload["redacted"] = True
        if previous_run_note:
            payload["previousRun"] = previous_run_note
        print(json.dumps(stamp_payload(payload), indent=2, sort_keys=True))
    else:
        footer_lines: List[str] = []
        if not result.written:
            footer_lines.append(f"Output dir: {result.output_dir}")
        if previous_run_note:
            footer_lines.append(f"Previous run: {previous_run_note}")
        summarize_import(ui, f"{provider.title} Sync", result.written, extra_lines=footer_lines or None)
        console = ui.console
        skip_reasons = getattr(result, "skip_reasons", None) or {}
        if skip_reasons:
            for reason, count in sorted(skip_reasons.items()):
                console.print(f"Skipped ({reason}): {count}")
        elif result.skipped:
            console.print(f"Skipped {result.skipped} item(s).")
        if result.pruned:
            console.print(f"Pruned {result.pruned} stale path(s).")
        if result.failures:
            prefix = "[red]Failures" if not ui.plain else "Failures"
            console.print(f"{prefix}: {result.failures}[/red]" if not ui.plain else f"{prefix}: {result.failures}")
            if isinstance(result.failed, list) and result.failed:
                for entry in result.failed[:5]:
                    path = entry.get("path") if isinstance(entry, dict) else None
                    error = entry.get("error") if isinstance(entry, dict) else None
                    if path and error:
                        console.print(f"- {path}: {error}")
                    elif path:
                        console.print(f"- {path}")
                if len(result.failed) > 5:
                    console.print(f"... {len(result.failed) - 5} more (rerun with --json for details)")

    run_payload = {
        "cmd": f"sync {provider.name}",
        "provider": provider.name,
        "count": len(result.written),
        "out": str(result.output_dir),
        "attachments": attachments,
        "attachmentBytes": attachment_bytes,
        "tokens": tokens,
        "words": words,
        "skipped": result.skipped,
        "pruned": result.pruned,
        "diffs": result.diffs,
    }
    if disk_estimate_bytes is not None:
        run_payload["diskEstimateBytes"] = int(disk_estimate_bytes)
    if disk_free_bytes is not None:
        run_payload["diskFreeBytes"] = int(disk_free_bytes)
    if max_disk is not None:
        run_payload["maxDiskGiB"] = float(max_disk)
    if getattr(args, "resume_from", None) is not None:
        run_payload["resumeFrom"] = int(getattr(args, "resume_from") or 0)
    if getattr(result, "failures", 0):
        run_payload["failures"] = int(getattr(result, "failures") or 0)
    failed_items = getattr(result, "failed", None)
    if isinstance(failed_items, list) and failed_items:
        run_payload["failedPaths"] = failed_items
    if meta:
        run_payload["meta"] = dict(meta)
    if getattr(args, "sanitize_html", False):
        run_payload["redacted"] = True
    add_run(run_payload)


__all__ = [
    "run_list_cli",
    "run_sync_cli",
    "_run_sync_drive",
    "_run_local_sync",
    "_collect_session_selection",
    "_log_local_sync",
]
