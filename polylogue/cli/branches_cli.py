from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from ..branch_explorer import branch_diff, build_branch_html, format_branch_tree
from ..cli_common import choose_single_entry
from ..commands import CommandEnv, branches_command
from ..options import BranchExploreOptions
from .context import resolve_html_settings
from .editor import get_editor, open_in_editor


def run_branches_cli(args: object, env: CommandEnv) -> None:
    ui = env.ui
    settings = env.settings
    options = BranchExploreOptions(
        provider=getattr(args, "provider", None),
        slug=getattr(args, "slug", None),
        conversation_id=getattr(args, "conversation_id", None),
        min_branches=max(0, getattr(args, "min_branches", 1)),
    )
    result = branches_command(options)
    conversations = result.conversations
    if not conversations:
        target = []
        if options.slug:
            target.append(f"slug={options.slug}")
        if options.conversation_id:
            target.append(f"id={options.conversation_id}")
        if options.provider:
            target.append(f"provider={options.provider}")
        detail = f" ({', '.join(target)})" if target else ""
        ui.console.print(f"[yellow]No branchable conversations found{detail}.")
        return

    selected_conversations = conversations
    if not getattr(args, "no_picker", False) and len(conversations) > 1:

        def _format_conv(entry, _idx):
            branch_total = len(entry.nodes)
            title = entry.title or entry.slug
            return f"{entry.provider}:{entry.conversation_id}\t{entry.slug}\tbranches={branch_total}\t{title}"

        chosen, cancelled = choose_single_entry(
            ui,
            conversations,
            format_line=_format_conv,
            header="idx\tprovider:id\tslug\tbranches\ttitle",
            prompt="branch>",
        )
        if cancelled:
            ui.console.print("[yellow]Branch explorer cancelled.")
            return
        if chosen is None:
            ui.console.print("[yellow]No conversation selected.")
            return
        selected_conversations = [chosen]

    multi_html = len(selected_conversations) > 1
    for conv in selected_conversations:
        title = conv.title or conv.slug
        header_lines = [
            f"Provider: {conv.provider}",
            f"Slug: {conv.slug}",
            f"Conversation ID: {conv.conversation_id}",
            f"Branches: {len(conv.nodes)} (canonical: {conv.canonical_branch_id or 'unknown'})",
        ]
        if conv.last_updated:
            header_lines.append(f"Last updated: {conv.last_updated}")
        if conv.conversation_path:
            header_lines.append(f"Canonical file: {conv.conversation_path}")
        ui.summary(title, header_lines)

        tree = format_branch_tree(conv, use_color=not ui.plain)
        if not tree.strip():
            ui.console.print("[yellow]No branch data recorded.")
        else:
            if not ui.plain:
                result = subprocess.run(
                    ["gum", "format"],
                    input=f"```\n{tree}\n```",
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                output = result.stdout.strip()
                ui.console.print(output or tree)
            else:
                ui.console.print(tree)

        html_path = None
        html_enabled, html_explicit = resolve_html_settings(args, settings)
        html_out = getattr(args, "out", None)
        should_auto_html = html_enabled and not html_explicit and html_out is None and conv.branch_count > 1
        force_html = html_out is not None or (html_explicit and html_enabled)
        if force_html or should_auto_html:
            html_path = _generate_branch_html(
                conv,
                target=_resolve_html_output_path(conv, html_out, multi_html),
                theme=getattr(args, "theme", None) or settings.html_theme,
                ui=ui,
                auto=should_auto_html and not force_html,
            )

        branch_id = getattr(args, "branch", None)
        diff_requested = bool(getattr(args, "diff", False) or branch_id)
        if diff_requested:
            if not branch_id and not ui.plain:
                branch_id = _prompt_branch_choice(ui, conv)
            if not branch_id:
                non_canonical = [node.branch_id for node in conv.nodes.values() if not node.is_canonical]
                if getattr(args, "diff", False) and non_canonical:
                    branch_id = non_canonical[0]
        if branch_id:
            _display_branch_diff_for_id(conv, branch_id, ui)

        html_path = _prompt_branch_followups(ui, conv, args, html_path, settings)

        if getattr(args, "open", False):
            target_path = html_path if html_path else conv.conversation_path
            if target_path:
                if open_in_editor(Path(target_path)):
                    ui.console.print(f"[dim]Opened {target_path} in editor[/dim]")
                else:
                    editor = get_editor()
                    if not editor:
                        ui.console.print("[yellow]Warning: $EDITOR not set. Cannot open file.")
                    else:
                        ui.console.print(f"[yellow]Warning: Could not open {target_path} in editor")


def _generate_branch_html(conversation, target: Optional[Path], theme: str, ui, *, auto: bool) -> Optional[Path]:
    if target is None:
        return None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        html_text = build_branch_html(conversation, theme=theme)
        target.write_text(html_text, encoding="utf-8")
        if auto:
            ui.console.print(f"[green]Auto-generated branch explorer → {target}")
        else:
            ui.console.print(f"[green]Wrote branch explorer to {target}")
        return target
    except Exception as exc:
        action = "auto-generate" if auto else "write"
        ui.console.print(f"[red]Failed to {action} HTML explorer: {exc}")
        return None


def _resolve_html_output_path(conversation, requested: Optional[Path], multi: bool) -> Optional[Path]:
    target = requested
    if target is None:
        if conversation.conversation_dir:
            target = conversation.conversation_dir / "branches.html"
        else:
            target = Path(f"{conversation.slug}-branches.html")
    if multi:
        suffix = target.suffix or ".html"
        stem = target.stem or "branches"
        target = target.parent / f"{stem}-{conversation.slug}{suffix}"
    if not target.suffix:
        target = target.with_suffix(".html")
    return target


def _prompt_branch_choice(ui, conversation) -> Optional[str]:
    candidates = [node for node in conversation.nodes.values() if not node.is_canonical]
    if not candidates:
        return None

    def _format(node, _idx):
        delta = node.divergence_index + 1 if node.divergence_index else 0
        preview = node.divergence_snippet or ""
        role = node.divergence_role or ""
        return f"{node.branch_id}\tdelta#{delta}\t{role}: {preview}"

    selection, cancelled = choose_single_entry(
        ui,
        candidates,
        format_line=_format,
        header="idx\tbranch\tdelta\tpreview",
        prompt="branch>",
    )
    if cancelled:
        return None
    if selection is None:
        return candidates[0].branch_id
    return selection.branch_id


def _display_branch_diff_for_id(conversation, branch_id: str, ui) -> None:
    diff_text = branch_diff(conversation, branch_id)
    if diff_text is None:
        ui.console.print(f"[yellow]Unable to diff branch {branch_id}; ensure it exists and is not canonical.")
        return
    if not diff_text.strip():
        ui.console.print(f"[cyan]Branch {branch_id} matches the canonical transcript.")
        return
    _display_diff(diff_text, ui)


def _prompt_branch_followups(ui, conversation, args, html_path: Optional[Path], settings) -> Optional[Path]:
    if getattr(ui, "plain", False):
        return html_path

    current_html = html_path
    while True:
        options: List[str] = []
        if conversation.branch_count > 1:
            options.append("Diff a branch")
        if not current_html:
            options.append("Write HTML explorer")
        if current_html is not None:
            options.append("Show HTML path")
        options.append("Done")

        choice = ui.choose("Next action?", options)
        if not choice or choice == "Done":
            break
        if choice.startswith("Diff"):
            branch_choice = _prompt_branch_choice(ui, conversation)
            if branch_choice:
                _display_branch_diff_for_id(conversation, branch_choice, ui)
        elif choice.startswith("Write"):
            target = _resolve_html_output_path(conversation, getattr(args, "out", None), False)
            current_html = _generate_branch_html(
                conversation,
                target=target,
                theme=getattr(args, "theme", None) or settings.html_theme,
                ui=ui,
                auto=False,
            ) or current_html
        elif choice.startswith("Show") and current_html is not None:
            ui.console.print(f"[cyan]Branch explorer → {current_html}")
    return current_html


def _display_diff(diff_text: str, ui) -> None:
    if not diff_text.strip():
        ui.console.print("[cyan]No diff to display.")
        return
    if not ui.plain:
        subprocess.run(["gum", "pager"], input=diff_text, text=True, check=True)
        return
    ui.console.print(diff_text)


__all__ = ["run_branches_cli"]

