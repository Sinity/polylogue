from __future__ import annotations

import csv
import json
import shlex
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..cli_common import choose_single_entry
from ..commands import CommandEnv, search_command
from ..options import SearchHit, SearchOptions
from ..schema import stamp_payload
from .editor import get_editor, open_in_editor, open_in_browser

SCRIPT_MODULE = "polylogue.cli"


def run_search_cli(args: object, env: CommandEnv) -> None:
    ui = env.ui
    prefs = getattr(env, "prefs", {})
    search_prefs = prefs.get("search", {}) if isinstance(prefs, dict) else {}

    def _pref_bool(val: str) -> bool:
        return str(val).lower() in {"1", "true", "yes", "on"}

    # Apply saved defaults when caller did not override.
    if "--limit" in search_prefs and getattr(args, "limit", None) == 20:
        try:
            args.limit = int(search_prefs["--limit"])  # type: ignore[attr-defined]
        except Exception:
            pass
    if "--no-picker" in search_prefs and not getattr(args, "no_picker", False):
        if _pref_bool(search_prefs["--no-picker"]):
            args.no_picker = True  # type: ignore[attr-defined]
    if "--json" in search_prefs and not getattr(args, "json", False):
        if _pref_bool(search_prefs["--json"]):
            args.json = True  # type: ignore[attr-defined]
    if "--in-attachments" in search_prefs and not getattr(args, "in_attachments", False):
        if _pref_bool(search_prefs["--in-attachments"]):
            args.in_attachments = True  # type: ignore[attr-defined]

    query = getattr(args, "query", None)
    if getattr(args, "from_stdin", False) or query == "-":
        data = sys.stdin.read()
        if not data.strip():
            ui.console.print("[red]Search query is empty; provide a query via stdin or argument.")
            raise SystemExit(1)
        query = data.strip()

    if not query:
        ui.console.print("[red]Search requires a query argument (or use --from-stdin).")
        raise SystemExit(1)

    limit = int(getattr(args, "limit", 20) or 20)
    if limit <= 0:
        limit = 20

    has_attachments: Optional[bool]
    if getattr(args, "with_attachments", False):
        has_attachments = True
    elif getattr(args, "without_attachments", False):
        has_attachments = False
    else:
        has_attachments = None

    in_attachments = bool(getattr(args, "in_attachments", False))
    attachment_name = getattr(args, "attachment_name", None)
    options = SearchOptions(
        query=query,
        limit=limit,
        provider=getattr(args, "provider", None),
        slug=getattr(args, "slug", None),
        conversation_id=getattr(args, "conversation_id", None),
        branch_id=getattr(args, "branch", None),
        model=getattr(args, "model", None),
        since=getattr(args, "since", None),
        until=getattr(args, "until", None),
        has_attachments=has_attachments,
        in_attachments=in_attachments,
        attachment_name=attachment_name,
    )
    result = search_command(options, env)
    hits = result.hits

    export_fields = [field.strip() for field in (getattr(args, "fields", "") or "").split(",") if field.strip()]
    csv_target = getattr(args, "csv", None)
    json_lines = bool(getattr(args, "json_lines", False))
    json_mode = bool(getattr(args, "json", False) or json_lines)
    if json_lines:
        setattr(args, "json", True)

    def _row(hit: SearchHit) -> Dict[str, Any]:
        return {
            "provider": hit.provider,
            "conversationId": hit.conversation_id,
            "conversation_id": hit.conversation_id,
            "slug": hit.slug,
            "title": hit.title,
            "branchId": hit.branch_id,
            "branch_id": hit.branch_id,
            "messageId": hit.message_id,
            "message_id": hit.message_id,
            "position": hit.position,
            "timestamp": hit.timestamp,
            "attachments": hit.attachment_count,
            "kind": hit.kind,
            "attachmentName": hit.attachment_name,
            "attachment_name": hit.attachment_name,
            "attachmentPath": str(hit.attachment_path) if hit.attachment_path else None,
            "attachment_path": str(hit.attachment_path) if hit.attachment_path else None,
            "attachmentBytes": hit.attachment_bytes,
            "attachment_bytes": hit.attachment_bytes,
            "attachmentMime": hit.attachment_mime,
            "attachment_mime": hit.attachment_mime,
            "ocrUsed": hit.ocr_used,
            "ocr_used": hit.ocr_used,
            "score": hit.score,
            "snippet": hit.snippet,
            "body": hit.body,
            "model": hit.model,
            "conversationPath": str(hit.conversation_path) if hit.conversation_path else None,
            "conversation_path": str(hit.conversation_path) if hit.conversation_path else None,
            "branchPath": str(hit.branch_path) if hit.branch_path else None,
            "branch_path": str(hit.branch_path) if hit.branch_path else None,
        }

    if json_lines:
        fields = export_fields or [
            "provider",
            "conversationId",
            "slug",
            "branchId",
            "messageId",
            "position",
            "timestamp",
            "score",
            "model",
            "attachments",
            "kind",
            "attachmentName",
            "attachmentPath",
            "attachmentBytes",
            "snippet",
            "conversationPath",
            "branchPath",
        ]
        for hit in hits:
            row = _row(hit)
            print(json.dumps({k: row.get(k) for k in fields}, ensure_ascii=False))
        return

    if json_mode:
        fields = export_fields or [
            "provider",
            "conversationId",
            "slug",
            "title",
            "branchId",
            "messageId",
            "position",
            "timestamp",
            "score",
            "model",
            "attachments",
            "kind",
            "attachmentName",
            "attachmentPath",
            "attachmentBytes",
            "attachmentMime",
            "ocrUsed",
            "snippet",
            "conversationPath",
            "branchPath",
        ]
        payload_hits = []
        for hit in hits:
            row = _row(hit)
            payload_hits.append({k: row.get(k) for k in fields})
        payload = stamp_payload(
            {
                "cmd": "search",
                "query": query,
                "limit": limit,
                "count": len(hits),
                "hits": payload_hits,
            }
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return

    if csv_target:
        fieldnames = export_fields or [
            "provider",
            "conversationId",
            "slug",
            "branchId",
            "messageId",
            "position",
            "timestamp",
            "score",
            "model",
            "attachments",
            "kind",
            "attachmentName",
            "attachmentPath",
            "attachmentBytes",
            "snippet",
            "path",
        ]
        rows = [{k: _row(hit).get(k) for k in fieldnames} for hit in hits]
        destination = Path(csv_target)
        if str(destination) == "-":
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        else:
            destination = destination.expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            ui.console.print(f"[green]Wrote {len(rows)} search hit(s) to {destination}")
        return

    if not hits:
        ui.console.print("[yellow]No results found.")
        return

    summary_lines = [f"Hits: {len(hits)} (limit {limit})"]
    if in_attachments:
        summary_lines.append("Mode: attachment text")
    provider_counts = Counter(hit.provider for hit in hits)
    if provider_counts:
        provider_overview = ", ".join(f"{provider}×{count}" for provider, count in provider_counts.most_common(3))
        summary_lines.append(f"Providers: {provider_overview}")
    model_set = {hit.model for hit in hits if hit.model}
    if model_set:
        summary_lines.append("Models: " + ", ".join(sorted(model_set)))
    attachment_results = sum(1 for hit in hits if hit.kind == "attachment")
    if attachment_results:
        summary_lines.append(f"Attachment hits: {attachment_results}")
    attachment_hits = sum(1 for hit in hits if hit.attachment_count)
    if attachment_hits:
        summary_lines.append(f"With attachments: {attachment_hits}")
    ui.summary("Search", summary_lines)

    selected_hits: List[SearchHit]
    if not ui.plain and not getattr(args, "no_picker", False) and len(hits) > 1:
        picked, cancelled = _run_search_picker(ui, hits)
        if cancelled:
            ui.console.print("[yellow]Search cancelled.")
            return
        selected_hits = [picked] if picked is not None else hits
    else:
        selected_hits = hits

    for hit in selected_hits:
        _render_search_hit(hit, ui)

    if getattr(args, "open", False) and len(selected_hits) == 1:
        _open_single_search_hit(ui, selected_hits[0])
    elif getattr(args, "open", False) and len(selected_hits) > 1:
        ui.console.print("[yellow]Warning: --open requires a single search result. Use --limit 1 or select one result.")


def _open_single_search_hit(ui, hit: SearchHit) -> None:
    target_path = hit.attachment_path or hit.branch_path or hit.conversation_path
    line_hint = None
    anchor_label = None
    if target_path and hit.kind != "attachment" and hit.position is not None and hit.position >= 0:
        anchor_label = f"msg-{hit.position}"
        line_hint = find_anchor_line(Path(target_path), anchor_label)
    if not target_path:
        ui.console.print("[yellow]Warning: No file path available for selected result")
        return

    target_obj = Path(target_path)
    is_html = target_obj.suffix.lower() == ".html"
    if is_html:
        if open_in_browser(target_obj, anchor_label):
            suffix = f"#{anchor_label}" if anchor_label else ""
            ui.console.print(f"[dim]Opened {target_obj}{suffix} in browser[/dim]")
            return

    if open_in_editor(target_obj, line=line_hint):
        suffix = f" (line {line_hint})" if line_hint else ""
        label = f"{target_path}#{anchor_label}" if anchor_label else str(target_path)
        ui.console.print(f"[dim]Opened {label}{suffix} in editor[/dim]")
        return

    editor = get_editor()
    if not editor:
        ui.console.print("[yellow]Warning: $EDITOR not set. Cannot open file.")
        return
    label = f"{target_path}#{anchor_label}" if anchor_label else str(target_path)
    ui.console.print(f"[yellow]Warning: Could not open {label} in editor")


def _run_search_picker(ui, hits: List[SearchHit]) -> Tuple[Optional[SearchHit], bool]:
    if not hits:
        return None, False
    data_payload = [
        {
            "title": hit.title or hit.slug,
            "provider": hit.provider,
            "slug": hit.slug,
            "branch": hit.branch_id,
            "score": hit.score,
            "snippet": hit.snippet,
            "body": hit.body,
            "timestamp": hit.timestamp,
            "attachments": hit.attachment_count,
            "model": hit.model,
            "kind": hit.kind,
            "attachmentName": hit.attachment_name,
        }
        for hit in hits
    ]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".json") as handle:
            json.dump({"hits": data_payload}, handle)
            tmp_path = Path(handle.name)

        def _format(entry: SearchHit, _idx: int) -> str:
            snippet = entry.snippet or entry.body
            snippet = (snippet or "").replace("\n", " ")
            snippet = textwrap.shorten(snippet, width=72, placeholder="…")
            prefix = "ATT" if entry.kind == "attachment" else "MSG"
            branch_label = entry.branch_id or "-"
            return f"{prefix}:{entry.provider}:{entry.slug} [{branch_label}] score={entry.score:.3f} {snippet}"

        preview_cmd = _build_search_preview_command(tmp_path)
        selection, cancelled = choose_single_entry(
            ui,
            hits,
            format_line=_format,
            header="idx\tprovider:slug [branch]\tscore\tsnippet",
            prompt="search>",
            preview=preview_cmd,
        )
        if cancelled:
            return None, True
        if selection is None:
            return hits[0], False
        return selection, False
    finally:
        if tmp_path:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _build_search_preview_command(data_file: Path) -> str:
    python_cmd = f"{shlex.quote(sys.executable)} -m {SCRIPT_MODULE}"
    return (
        "bash -lc "
        f"\"{python_cmd} _search-preview --data-file {shlex.quote(str(data_file))} "
        "--index $(printf %s \\\"{}\\\" | awk '{print $1}')\""
    )


def find_anchor_line(path: Path, anchor: str) -> Optional[int]:
    """Return the 1-based line containing the given anchor id."""
    target = anchor.lstrip("#")
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    for idx, line in enumerate(lines, start=1):
        if target in line:
            return idx
    return None


def _render_search_hit(hit: SearchHit, ui) -> None:
    branch_label = hit.branch_id or "-"
    header = f"{hit.provider}/{hit.slug} [{branch_label}]"
    lines = [
        f"Kind: {hit.kind}",
        f"Score: {hit.score:.4f}",
    ]
    if hit.kind != "attachment":
        lines.append(f"Message: {hit.message_id} (position {hit.position})")
    if hit.kind == "attachment":
        if hit.attachment_name:
            lines.append(f"Attachment: {hit.attachment_name}")
        if hit.attachment_bytes is not None:
            lines.append(f"Attachment size: {hit.attachment_bytes} bytes")
        if hit.attachment_mime:
            lines.append(f"MIME: {hit.attachment_mime}")
        if hit.attachment_path:
            lines.append(f"Attachment path: {hit.attachment_path}")
        lines.append(f"OCR used: {'yes' if hit.ocr_used else 'no'}")
    if hit.timestamp:
        lines.append(f"Timestamp: {hit.timestamp}")
    if hit.model:
        lines.append(f"Model: {hit.model}")
    lines.append(f"Attachments: {hit.attachment_count}")
    if hit.conversation_path:
        lines.append(f"Conversation path: {hit.conversation_path}")
    if hit.branch_path:
        lines.append(f"Branch path: {hit.branch_path}")
    if hit.snippet:
        lines.append(f"Snippet: {hit.snippet}")
    ui.summary(hit.title or header, lines)

    body = (hit.body or "").strip()
    if not body:
        ui.console.print("[cyan](No text available for this result)")
        return

    if ui.plain:
        ui.console.print(body)
    else:
        ui.render_markdown(body)


def run_search_preview(args: object) -> None:
    try:
        data_file = getattr(args, "data_file")
        data = json.loads(Path(data_file).read_text(encoding="utf-8"))
    except Exception:
        return
    hits = data.get("hits")
    if not isinstance(hits, list):
        return
    index = getattr(args, "index", -1)
    try:
        index = int(index)
    except Exception:
        return
    if index < 0 or index >= len(hits):
        return

    hit = hits[index]
    if not isinstance(hit, dict):
        return
    title = hit.get("title") or "Search hit"
    provider = hit.get("provider") or ""
    slug = hit.get("slug") or ""
    branch = hit.get("branch") or ""
    score = hit.get("score")
    snippet = hit.get("snippet") or ""
    body = hit.get("body") or ""

    parts: List[str] = []
    header = f"{provider}/{slug} [{branch}]"
    if score is not None:
        header += f" score={score}"
    parts.append(header)
    parts.append(f"Title: {title}")
    if snippet:
        parts.append("")
        parts.append("Snippet:")
        parts.append(textwrap.fill(str(snippet), width=100))
    if body:
        parts.append("")
        parts.append("Body:")
        parts.append(str(body))
    print("\n".join(str(part) for part in parts))


__all__ = ["run_search_cli", "run_search_preview", "find_anchor_line"]
