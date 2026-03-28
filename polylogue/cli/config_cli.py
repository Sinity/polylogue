from __future__ import annotations

import json
import os
from pathlib import Path

from ..commands import CommandEnv


def run_config_show(args: object, env: CommandEnv) -> None:
    """Show current configuration (combines env + settings)."""
    from ..config import CONFIG_PATH, DEFAULT_CREDENTIALS, DEFAULT_TOKEN
    from ..config import is_config_declarative
    from ..schema import stamp_payload
    from ..settings import SETTINGS_PATH

    ui = env.ui
    settings = env.settings
    defaults = env.config.defaults
    exports = env.config.exports
    index_cfg = env.config.index
    declarative, decl_reason, decl_target = is_config_declarative(CONFIG_PATH)

    credential_env = os.environ.get("POLYLOGUE_CREDENTIAL_PATH")
    token_env = os.environ.get("POLYLOGUE_TOKEN_PATH")
    drive_cfg = getattr(env, "config", None).drive if hasattr(env, "config") else None
    credential_path = (
        Path(credential_env).expanduser()
        if credential_env
        else (drive_cfg.credentials_path if drive_cfg else DEFAULT_CREDENTIALS)
    )
    token_path = (
        Path(token_env).expanduser()
        if token_env
        else (drive_cfg.token_path if drive_cfg else DEFAULT_TOKEN)
    )

    if getattr(args, "json", False):
        roots_map = getattr(env.config.defaults, "roots", {}) or {}
        roots_payload = {
            label: {key: str(value) for key, value in vars(paths).items()}
            for label, paths in roots_map.items()
        }
        payload = stamp_payload(
            {
                "configPath": str(CONFIG_PATH) if CONFIG_PATH else None,
                "settingsPath": str(SETTINGS_PATH),
                "configDeclarative": declarative,
                "configDeclarativeReason": decl_reason or None,
                "configDeclarativeTarget": str(decl_target) if decl_target else None,
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
                    "roots": roots_payload,
                },
                "inputs": {
                    "chatgpt": str(exports.chatgpt),
                    "claude": str(exports.claude),
                },
                "index": {
                    "backend": index_cfg.backend if index_cfg else "sqlite",
                    "qdrant": {
                        "url": index_cfg.qdrant_url if index_cfg else None,
                        "api_key": index_cfg.qdrant_api_key if index_cfg else None,
                        "collection": index_cfg.qdrant_collection if index_cfg else None,
                        "vector_size": index_cfg.qdrant_vector_size if index_cfg else None,
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
        f"Config mutability: {'declarative' if declarative else 'mutable'}",
    ]
    if declarative and decl_reason:
        summary_lines.append(f"  Reason: {decl_reason}")
        summary_lines.append("  Hint: edit your Nix/flake module; config init/set/edit are disabled.")
    summary_lines.extend(
        [
            "",
            f"HTML previews: {'on' if settings.html_previews else 'off'}",
            f"HTML theme: {settings.html_theme}",
        ]
    )
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

    summary_lines.extend(
        [
            "",
            "Output directories:",
            f"  render: {defaults.output_dirs.render}",
            f"  gemini: {defaults.output_dirs.sync_drive}",
            f"  codex: {defaults.output_dirs.sync_codex}",
            f"  claude-code: {defaults.output_dirs.sync_claude_code}",
            f"  chatgpt: {defaults.output_dirs.import_chatgpt}",
            f"  claude: {defaults.output_dirs.import_claude}",
        ]
    )
    roots_map = getattr(env.config.defaults, "roots", {}) or {}
    if roots_map:
        summary_lines.append("  labeled roots:")
        for label, paths in roots_map.items():
            summary_lines.append(
                "    "
                f"{label}: render={paths.render} gemini={paths.sync_drive} codex={paths.sync_codex} "
                f"claude-code={paths.sync_claude_code} chatgpt={paths.import_chatgpt} claude={paths.import_claude}"
            )
    summary_lines.extend(
        [
            "",
            f"State DB: {env.conversations.state_path}",
            f"Runs DB: {env.database.resolve_path()}",
        ]
    )
    ui.summary("Configuration", summary_lines)


__all__ = ["run_config_show"]
