from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

from ..commands import CommandEnv
from ..config import CONFIG_PATH, IndexConfig, OutputDirs, is_config_declarative, persist_config
from ..settings import persist_settings


def _build_output_dirs(root: Path) -> OutputDirs:
    base = root.expanduser()
    return OutputDirs(
        render=base / "render",
        sync_drive=base / "gemini",
        sync_codex=base / "codex",
        sync_claude_code=base / "claude-code",
        import_chatgpt=base / "chatgpt",
        import_claude=base / "claude",
    )


def _persist_all(env: CommandEnv) -> None:
    settings = env.settings
    config_obj = env.config
    persist_settings(settings)
    output_root = config_obj.defaults.output_dirs.render.parent
    input_root = config_obj.exports.chatgpt
    persist_config(
        input_root=input_root,
        output_root=output_root,
        collapse_threshold=settings.collapse_threshold,
        html_previews=settings.html_previews,
        html_theme=settings.html_theme,
        index=config_obj.index,
        roots=config_obj.defaults.roots if getattr(config_obj.defaults, "roots", None) else None,
        path=CONFIG_PATH,
    )


def _summary_lines(env: CommandEnv) -> List[str]:
    settings = env.settings
    config_obj = env.config
    defaults = config_obj.defaults
    output_root = defaults.output_dirs.render.parent
    input_root = config_obj.exports.chatgpt
    backend = config_obj.index.backend if config_obj.index else "sqlite"
    lines = [
        f"Output root: {output_root}",
        f"Inbox root: {input_root}",
        f"HTML previews: {'on' if settings.html_previews else 'off'}",
        f"HTML theme: {settings.html_theme}",
        f"Collapse threshold: {settings.collapse_threshold}",
        f"Index backend: {backend}",
    ]
    if config_obj.index and backend == "qdrant":
        lines.extend(
            [
                f"Qdrant URL: {config_obj.index.qdrant_url or ''}",
                f"Qdrant collection: {config_obj.index.qdrant_collection or ''}",
                f"Vector size: {config_obj.index.qdrant_vector_size or ''}",
            ]
        )
    roots_map: Dict[str, OutputDirs] = getattr(defaults, "roots", {}) or {}
    if roots_map:
        lines.append("Labeled roots:")
        for label, paths in roots_map.items():
            lines.append(f"  {label}: {paths.render.parent}")
    return lines


def _edit_output_root(env: CommandEnv) -> None:
    ui = env.ui
    defaults = env.config.defaults
    current_root = defaults.output_dirs.render.parent
    raw = ui.input("New output root (base dir for all providers)", default=str(current_root))
    if not raw:
        return
    new_root = Path(raw).expanduser()
    defaults.output_dirs = _build_output_dirs(new_root)
    _persist_all(env)


def _edit_inbox_root(env: CommandEnv) -> None:
    ui = env.ui
    config_obj = env.config
    current_root = config_obj.exports.chatgpt
    raw = ui.input("New inbox root (for ChatGPT/Claude exports)", default=str(current_root))
    if not raw:
        return
    new_root = Path(raw).expanduser()
    config_obj.exports.chatgpt = new_root
    config_obj.exports.claude = new_root
    _persist_all(env)


def _edit_html_previews(env: CommandEnv) -> None:
    ui = env.ui
    settings = env.settings
    enabled = ui.confirm("Enable HTML previews by default?", default=bool(settings.html_previews))
    settings.html_previews = enabled
    _persist_all(env)


def _edit_html_theme(env: CommandEnv) -> None:
    ui = env.ui
    settings = env.settings
    current = settings.html_theme or "dark"
    chosen = ui.choose("Select default HTML theme", ["light", "dark"]) or current
    settings.html_theme = chosen
    _persist_all(env)


def _edit_collapse_threshold(env: CommandEnv) -> None:
    ui = env.ui
    settings = env.settings
    defaults = env.config.defaults
    current = settings.collapse_threshold if settings.collapse_threshold is not None else defaults.collapse_threshold
    raw = ui.input("Collapse threshold (0 disables collapsing)", default=str(current))
    if raw is None:
        return
    try:
        value = int(str(raw).strip())
    except Exception:
        return
    if value < 0:
        return
    settings.collapse_threshold = value
    _persist_all(env)


def _edit_index_backend(env: CommandEnv) -> None:
    ui = env.ui
    config_obj = env.config
    current = config_obj.index.backend if config_obj.index else "sqlite"
    backend = ui.choose("Select index backend", ["sqlite", "qdrant", "none"]) or current

    index_cfg = config_obj.index or IndexConfig()
    index_cfg.backend = backend
    if backend == "qdrant":
        url = ui.input("Qdrant URL", default=index_cfg.qdrant_url or "http://localhost:6333")
        api_key = ui.input("Qdrant API key (blank for none)", default=index_cfg.qdrant_api_key or "")
        collection = ui.input("Qdrant collection", default=index_cfg.qdrant_collection or "polylogue")
        vector_raw = ui.input("Vector size", default=str(index_cfg.qdrant_vector_size or 1536))
        try:
            vector_size = int(vector_raw) if vector_raw else 1536
        except Exception:
            vector_size = 1536
        index_cfg.qdrant_url = url or index_cfg.qdrant_url
        index_cfg.qdrant_api_key = api_key or None
        index_cfg.qdrant_collection = collection or index_cfg.qdrant_collection
        index_cfg.qdrant_vector_size = vector_size

    config_obj.index = index_cfg
    _persist_all(env)


def _edit_labeled_roots(env: CommandEnv) -> None:
    ui = env.ui
    defaults = env.config.defaults
    roots: Dict[str, OutputDirs] = getattr(defaults, "roots", {}) or {}
    while True:
        action = ui.choose(
            "Labeled roots",
            ["Add root", "Edit root", "Remove root", "Back"],
        )
        if not action or action == "Back":
            return
        if action == "Add root":
            label = ui.input("Root label", default=f"root{len(roots) + 1}")
            if not label:
                continue
            root_path = ui.input("Root path", default=str(defaults.output_dirs.render.parent))
            if not root_path:
                continue
            roots[label] = _build_output_dirs(Path(root_path))
            defaults.roots = roots
            _persist_all(env)
        elif action == "Edit root":
            if not roots:
                ui.console.print("[yellow]No labeled roots to edit.")
                continue
            label = ui.choose("Select root to edit", sorted(roots.keys()))
            if not label:
                continue
            current_root = roots[label].render.parent
            root_path = ui.input("New root path", default=str(current_root))
            if not root_path:
                continue
            roots[label] = _build_output_dirs(Path(root_path))
            defaults.roots = roots
            _persist_all(env)
        elif action == "Remove root":
            if not roots:
                ui.console.print("[yellow]No labeled roots to remove.")
                continue
            label = ui.choose("Select root to remove", sorted(roots.keys()))
            if not label:
                continue
            if not ui.confirm(f"Remove labeled root '{label}'?", default=False):
                continue
            roots.pop(label, None)
            defaults.roots = roots
            _persist_all(env)


def run_config_edit_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    ui = env.ui
    if ui.plain:
        ui.console.print("[red]config edit is interactive-only; rerun in a TTY or pass --interactive.")
        raise SystemExit(1)

    immutable, reason, cfg_path = is_config_declarative()
    if immutable:
        ui.console.print(
            f"[red]Configuration is managed declaratively ({cfg_path}): {reason}. "
            "Edit your Nix/flake module instead."
        )
        raise SystemExit(1)

    actions = [
        "Output root",
        "Inbox root",
        "HTML previews",
        "HTML theme",
        "Collapse threshold",
        "Index backend/Qdrant",
        "Labeled roots",
        "Quit",
    ]

    while True:
        ui.summary("Current Configuration", _summary_lines(env))
        choice = ui.choose("Edit which setting?", actions)
        if not choice or choice == "Quit":
            break
        if choice == "Output root":
            _edit_output_root(env)
        elif choice == "Inbox root":
            _edit_inbox_root(env)
        elif choice == "HTML previews":
            _edit_html_previews(env)
        elif choice == "HTML theme":
            _edit_html_theme(env)
        elif choice == "Collapse threshold":
            _edit_collapse_threshold(env)
        elif choice == "Index backend/Qdrant":
            _edit_index_backend(env)
        elif choice == "Labeled roots":
            _edit_labeled_roots(env)


__all__ = ["run_config_edit_cli"]
