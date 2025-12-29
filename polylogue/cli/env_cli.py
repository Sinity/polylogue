from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Any

from ..config import CONFIG_PATH, DEFAULT_CREDENTIALS, DEFAULT_TOKEN, is_config_declarative
from .. import paths as paths_module
from ..version import POLYLOGUE_VERSION, SCHEMA_VERSION
from ..commands import CommandEnv


@dataclass
class EnvCheck:
    name: str
    ok: bool
    detail: str
    severity: str = "info"  # info|warn|error


def _path_status(path: Path, *, required: bool = True, label: str | None = None) -> EnvCheck:
    exists = path.exists()
    severity = "error" if required and not exists else "warn" if not exists else "info"
    detail = f"{label or path}: {'present' if exists else 'missing'}"
    return EnvCheck(name=label or path.name, ok=exists or not required, detail=detail, severity=severity)


def run_env_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    checks: List[EnvCheck] = []
    drift_warnings: List[str] = []

    credential_env = os.environ.get("POLYLOGUE_CREDENTIAL_PATH")
    token_env = os.environ.get("POLYLOGUE_TOKEN_PATH")

    drive_cfg = getattr(env.config, "drive", None)
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

    checks.append(_path_status(paths_module.CONFIG_HOME, required=True, label="Config home"))
    checks.append(_path_status(paths_module.DATA_HOME, required=True, label="Data home"))
    checks.append(_path_status(paths_module.STATE_HOME, required=True, label="State home"))

    checks.append(_path_status(credential_path, required=False, label="Credentials"))
    checks.append(_path_status(token_path, required=False, label="Token"))
    if credential_env:
        checks.append(EnvCheck(name="POLYLOGUE_CREDENTIAL_PATH", ok=True, detail=str(credential_env)))
    if token_env:
        checks.append(EnvCheck(name="POLYLOGUE_TOKEN_PATH", ok=True, detail=str(token_env)))

    cfg_path = CONFIG_PATH if CONFIG_PATH else paths_module.CONFIG_HOME / "config.json"
    declarative, decl_reason, decl_target = is_config_declarative(cfg_path)
    checks.append(
        EnvCheck(
            name="Config mutability",
            ok=not declarative,
            detail=decl_reason or "config is mutable",
            severity="warn" if declarative else "info",
        )
    )
    checks.append(_path_status(cfg_path, required=False, label="Config file"))
    if cfg_path.exists():
        try:
            cfg_raw: Dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
            allowed_top = {"paths", "ui", "defaults", "index", "exports", "drive"}
            extra_keys = sorted(set(cfg_raw.keys()) - allowed_top)
            if extra_keys:
                drift_warnings.append(f"config contains unknown keys: {', '.join(extra_keys)}")
            defaults = cfg_raw.get("defaults") or cfg_raw.get("ui") or {}
            paths = cfg_raw.get("paths") or {}
            roots = {}
            if isinstance(defaults, dict):
                roots = defaults.get("roots") or roots
            if isinstance(paths, dict):
                roots = paths.get("roots") or roots
            if roots and not isinstance(roots, dict):
                drift_warnings.append("roots should be a mapping of labels to output paths")
        except Exception as exc:
            drift_warnings.append(f"config parse failed: {exc}")

    backend = env.config.index.backend if env.config and env.config.index else "sqlite"
    index_ok = backend in {"sqlite", "qdrant", "none"}
    checks.append(
        EnvCheck(name="Index backend", ok=index_ok, detail=f"backend={backend}", severity="error" if not index_ok else "info")
    )
    if backend == "qdrant":
        if not env.config.index.qdrant_url:
            checks.append(EnvCheck(name="Qdrant URL", ok=False, detail="Missing POLYLOGUE_QDRANT_URL", severity="error"))
        if not env.config.index.qdrant_collection:
            checks.append(EnvCheck(name="Qdrant collection", ok=False, detail="Missing collection name", severity="error"))

    output_roots = env.config.defaults.output_dirs
    for label, path in (
        ("render", output_roots.render),
        ("sync_drive", output_roots.sync_drive),
        ("sync_codex", output_roots.sync_codex),
        ("sync_claude_code", output_roots.sync_claude_code),
        ("import_chatgpt", output_roots.import_chatgpt),
        ("import_claude", output_roots.import_claude),
    ):
        checks.append(_path_status(path, required=False, label=f"Output:{label}"))

    # Detect mixed roots drift (different parents across providers)
    parents = {
        output_roots.render.parent,
        output_roots.sync_drive.parent,
        output_roots.sync_codex.parent,
        output_roots.sync_claude_code.parent,
        output_roots.import_chatgpt.parent,
        output_roots.import_claude.parent,
    }
    if len(parents) > 1:
        checks.append(
            EnvCheck(
                name="Output roots drift",
                ok=False,
                detail="Output roots have mixed parents; align with config set --output-root",
                severity="warn",
            )
        )

    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "polylogueVersion": POLYLOGUE_VERSION,
        "checks": [check.__dict__ for check in checks],
        "configPath": str(cfg_path),
        "configDeclarative": declarative,
        "configDeclarativeReason": decl_reason,
        "driftWarnings": drift_warnings,
    }
    all_ok = all(c.ok for c in checks if c.severity != "warn")

    if getattr(args, "json", False):
        import json

        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        lines = [f"polylogue={POLYLOGUE_VERSION} schema={SCHEMA_VERSION}"]
        lines.append(f"Config: {cfg_path}")
        if declarative:
            lines.append(f"[yellow]Config is declarative/read-only: {decl_reason}")
        for warning in drift_warnings:
            lines.append(f"[yellow]{warning}")
        for check in checks:
            status = "ok" if check.ok else "fail"
            lines.append(f"- {check.name}: {status} ({check.detail})")
        env.ui.summary("Environment", lines)

    if not all_ok:
        raise SystemExit(1)


__all__ = ["run_env_cli"]
