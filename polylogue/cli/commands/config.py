"""Config command - manage polylogue configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..commands import CommandEnv


def setup_parser(subparsers: argparse._SubParsersAction, _add_command_parser, add_helpers) -> None:
    """Add config command parser.

    Args:
        subparsers: The main subparsers action
        _add_command_parser: Helper function to add command with examples
        add_helpers: Dict of helper functions for common argument patterns
    """
    p = _add_command_parser(
        subparsers,
        "config",
        help="Configuration (init/set/show)",
        description="Configure Polylogue settings",
        epilog=add_helpers["examples_epilog"]("config"),
    )
    config_sub = p.add_subparsers(dest="config_cmd", required=True)

    # config init
    p_config_init = config_sub.add_parser(
        "init",
        help="Interactive configuration setup",
        description="Interactive configuration setup wizard"
    )
    p_config_init.add_argument("--force", action="store_true", help="Overwrite existing configuration")

    # config set
    p_config_set = config_sub.add_parser(
        "set",
        help="Update settings",
        description="Show or update Polylogue defaults"
    )
    p_config_set.add_argument("--html", choices=["on", "off"], default=None, help="Enable or disable default HTML previews")
    p_config_set.add_argument("--theme", choices=["light", "dark"], default=None, help="Set the default HTML theme")
    p_config_set.add_argument("--collapse-threshold", type=int, default=None, help="Set the default collapse threshold for long outputs")
    p_config_set.add_argument("--output-root", type=Path, default=None, help="Set the output root for archives (overrides config.json)")
    p_config_set.add_argument("--input-root", type=Path, default=None, help="Set the inbox/input root for provider exports (overrides config.json)")
    p_config_set.add_argument("--reset", action="store_true", help="Reset to config defaults")
    p_config_set.add_argument("--json", action="store_true", help="Emit settings as JSON")

    # config show
    p_config_show = config_sub.add_parser(
        "show",
        help="Show configuration",
        description="Show resolved configuration and output paths"
    )
    p_config_show.add_argument("--json", action="store_true", help="Emit environment info as JSON")

    # config edit
    config_sub.add_parser(
        "edit",
        help="Interactively edit configuration",
        description="Interactive config editor for paths, defaults, and index settings",
    )


def dispatch(args: argparse.Namespace, env: CommandEnv) -> None:
    """Execute config command.

    Args:
        args: Parsed command-line arguments
        env: Command environment with config, UI, etc.
    """
    config_cmd = getattr(args, "config_cmd", None)
    if not config_cmd:
        env.ui.console.print("[red]config requires a sub-command (init/set/show)")
        raise SystemExit(1)

    if config_cmd == "init":
        from ..init import run_init_cli
        run_init_cli(args, env)
    elif config_cmd == "set":
        from ..settings_cli import run_settings_cli
        run_settings_cli(args, env)
    elif config_cmd == "edit":
        from ..config_editor import run_config_edit_cli
        run_config_edit_cli(args, env)
    else:  # show
        _run_config_show(args, env)


def _run_config_show(args: argparse.Namespace, env: CommandEnv) -> None:
    """Show current configuration (combines env + settings)."""
    from ...config import CONFIG, CONFIG_PATH, DEFAULT_CREDENTIALS, DEFAULT_TOKEN
    from ...schema import stamp_payload
    from ...settings import SETTINGS_PATH

    ui = env.ui
    settings = env.settings
    defaults = CONFIG.defaults

    credential_env = os.environ.get("POLYLOGUE_CREDENTIAL_PATH")
    token_env = os.environ.get("POLYLOGUE_TOKEN_PATH")
    drive_cfg = getattr(env, "config", None).drive if hasattr(env, "config") else None
    credential_path = drive_cfg.credentials_path if drive_cfg else DEFAULT_CREDENTIALS
    token_path = drive_cfg.token_path if drive_cfg else DEFAULT_TOKEN

    if getattr(args, "json", False):
        roots_map = getattr(env.config.defaults, "roots", {}) or {}
        payload = stamp_payload(
            {
                "configPath": str(CONFIG_PATH) if CONFIG_PATH else None,
                "settingsPath": str(SETTINGS_PATH),
                "ui": {
                    "html_previews": settings.html_previews,
                    "html_theme": settings.html_theme,
                    "collapse_threshold": settings.collapse_threshold,
                },
                "auth": {
                    "credentialPath": str(credential_path) if credential_path else None,
                    "tokenPath": str(token_path) if token_path else None,
                    "env": {
                        "POLYLOGUE_CREDENTIAL_PATH": credential_env,
                        "POLYLOGUE_TOKEN_PATH": token_env,
                    },
                },
                "outputs": {
                    "render": str(defaults.output_dirs.render),
                    "gemini": str(defaults.output_dirs.sync_drive),
                    "codex": str(defaults.output_dirs.sync_codex),
                    "claude_code": str(defaults.output_dirs.sync_claude_code),
                    "chatgpt": str(defaults.output_dirs.import_chatgpt),
                    "claude": str(defaults.output_dirs.import_claude),
                    "roots": {label: vars(paths) for label, paths in roots_map.items()},
                },
                "inputs": {
                    "chatgpt": str(CONFIG.exports.chatgpt),
                    "claude": str(CONFIG.exports.claude),
                },
                "index": {
                    "backend": CONFIG.index.backend if CONFIG.index else "sqlite",
                    "qdrant": {
                        "url": CONFIG.index.qdrant_url if CONFIG.index else None,
                        "api_key": CONFIG.index.qdrant_api_key if CONFIG.index else None,
                        "collection": CONFIG.index.qdrant_collection if CONFIG.index else None,
                        "vector_size": CONFIG.index.qdrant_vector_size if CONFIG.index else None,
                    },
                },
                "statePath": str(env.conversations.state_path),
                "runsDb": str(env.database.resolve_path()),
            }
        )
        print(json.dumps(payload))
        return

    summary_lines = [
        f"Config: {CONFIG_PATH or '(default)'}",
        f"Settings: {SETTINGS_PATH}",
        "",
        f"HTML previews: {'on' if settings.html_previews else 'off'}",
        f"HTML theme: {settings.html_theme}",
    ]
    if settings.collapse_threshold is not None:
        summary_lines.append(f"Collapse threshold: {settings.collapse_threshold}")

    summary_lines.extend(
        [
            "",
            "Auth paths:",
            f"  credentials: {credential_env or credential_path}",
            f"  token: {token_env or token_path}",
        ]
    )

    summary_lines.extend([
        "",
        "Output directories:",
        f"  render: {defaults.output_dirs.render}",
        f"  gemini: {defaults.output_dirs.sync_drive}",
        f"  codex: {defaults.output_dirs.sync_codex}",
        f"  claude-code: {defaults.output_dirs.sync_claude_code}",
        f"  chatgpt: {defaults.output_dirs.import_chatgpt}",
        f"  claude: {defaults.output_dirs.import_claude}",
    ])
    roots_map = getattr(env.config.defaults, "roots", {}) or {}
    if roots_map:
        summary_lines.append("  labeled roots:")
        for label, paths in roots_map.items():
            summary_lines.append(f"    {label}: render={paths.render} codex={paths.sync_codex}")
    summary_lines.extend([
        "",
        f"State DB: {env.conversations.state_path}",
        f"Runs DB: {env.database.resolve_path()}",
    ])
    ui.summary("Configuration", summary_lines)
