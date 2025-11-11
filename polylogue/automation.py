from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_MODULE = "polylogue.cli"


@dataclass
class AutomationTarget:
    name: str
    description: str
    command: List[str]
    defaults: Dict[str, Any] = field(default_factory=dict)


def _load_targets() -> Dict[str, AutomationTarget]:
    data = json.loads(resources.files(__package__).joinpath("automation_targets.json").read_text())
    return {key: AutomationTarget(**value) for key, value in data.items()}


TARGETS = _load_targets()


def resolve_target(target: str) -> AutomationTarget:
    try:
        return TARGETS[target]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported automation target: {target}") from exc


def _python_module_command(args: Iterable[str]) -> str:
    parts = [sys.executable, "-m", SCRIPT_MODULE] + list(args)
    return shlex.join(parts)


def build_command(target: AutomationTarget, extra_args: Iterable[str]) -> str:
    return _python_module_command(target.command + list(extra_args))


def describe_targets(target: Optional[str] = None) -> Dict[str, Dict[str, object]]:
    if target is not None:
        tgt = resolve_target(target)
        return {target: {
            "name": tgt.name,
            "description": tgt.description,
            "command": tgt.command,
            "defaults": tgt.defaults,
        }}
    return {
        key: {
            "name": value.name,
            "description": value.description,
            "command": value.command,
            "defaults": value.defaults,
        }
        for key, value in TARGETS.items()
    }


def _normalize_html_mode(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "on" if value else None
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"on", "off"}:
            return lowered
        if lowered == "auto":
            return None
    return None


def _apply_html_mode(args: List[str], mode: Optional[str]) -> List[str]:
    cleaned: List[str] = []
    skip_next = False
    for idx, token in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if token == "--html":
            if idx + 1 < len(args) and not args[idx + 1].startswith("-"):
                skip_next = True
            continue
        cleaned.append(token)
    if mode is None:
        return cleaned
    cleaned.append("--html")
    if mode != "on":
        cleaned.append(mode)
    return cleaned


def _merge_args(
    target: AutomationTarget,
    user_extra: Iterable[str],
    collapse_threshold: Optional[int],
    html_mode: Optional[str],
) -> List[str]:
    defaults = target.defaults or {}
    user_list = list(user_extra)
    args = list(defaults.get("extraArgs", []))
    args.extend(user_list)

    if collapse_threshold is not None:
        if "--collapse-threshold" not in args:
            args.extend(["--collapse-threshold", str(collapse_threshold)])
    else:
        default_collapse = defaults.get("collapseThreshold")
        if default_collapse is not None and "--collapse-threshold" not in args:
            args.extend(["--collapse-threshold", str(default_collapse)])

    user_requested_html = "--html" in user_list
    if html_mode is not None:
        args = _apply_html_mode(args, html_mode)
    elif not user_requested_html:
        default_mode = _normalize_html_mode(defaults.get("html"))
        if default_mode is not None:
            args = _apply_html_mode(args, default_mode)

    return args


def _escape_systemd_path(path: Path) -> str:
    text = str(path)
    return text.replace(" ", "\\x20")


def prepare_automation_command(
    target_key: str,
    *,
    user_extra: Iterable[str],
    collapse_threshold: Optional[int],
    html: Optional[object],
) -> Tuple[AutomationTarget, List[str]]:
    target = resolve_target(target_key)
    html_mode = _normalize_html_mode(html)
    merged = _merge_args(target, user_extra, collapse_threshold, html_mode)
    return target, merged


def systemd_snippet(
    *,
    target_key: str,
    interval: str,
    working_dir: Path,
    extra_args: Iterable[str] = (),
    boot_delay: str = "2m",
    collapse_threshold: Optional[int] = None,
    html: Optional[object] = None,
    status_log: Optional[Path] = None,
    status_limit: int = 50,
) -> str:
    target, merged_args = prepare_automation_command(
        target_key,
        user_extra=extra_args,
        collapse_threshold=collapse_threshold,
        html=html,
    )
    service_name = target.name
    command = build_command(target, merged_args)
    working_directory = _escape_systemd_path(working_dir)
    status_cmd = None
    if status_log is not None:
        status_cmd = _python_module_command(
            [
                "status",
                "--dump",
                str(status_log),
                "--dump-limit",
                str(max(1, status_limit)),
                "--dump-only",
            ]
        )
    service = dedent(
        f"""# ~/.config/systemd/user/{service_name}.service
[Unit]
Description={target.description}

[Service]
Type=oneshot
WorkingDirectory={working_directory}
ExecStart={command}
{f"ExecStartPost={status_cmd}" if status_cmd else ""}
"""
    ).strip()

    timer = dedent(
        f"""# ~/.config/systemd/user/{service_name}.timer
[Unit]
Description={target.description} timer

[Timer]
OnBootSec={boot_delay}
OnUnitActiveSec={interval}
Persistent=true

[Install]
WantedBy=default.target
"""
    ).strip()

    return f"{service}\n\n{timer}\n"


def cron_snippet(
    *,
    target_key: str,
    schedule: str,
    working_dir: Path,
    log_path: str,
    extra_args: Iterable[str] = (),
    state_env: str = '$HOME/.local/state',
    collapse_threshold: Optional[int] = None,
    html: Optional[object] = None,
    status_log: Optional[Path] = None,
    status_limit: int = 50,
) -> str:
    target, merged_args = prepare_automation_command(
        target_key,
        user_extra=extra_args,
        collapse_threshold=collapse_threshold,
        html=html,
    )
    command = build_command(target, merged_args)
    combined = command
    if status_log is not None:
        status_cmd = _python_module_command(
            [
                "status",
                "--dump",
                str(status_log),
                "--dump-limit",
                str(max(1, status_limit)),
                "--dump-only",
            ]
        )
        combined = f"({command} && {status_cmd})"
    snippet = (
        f"{schedule} XDG_STATE_HOME={shlex.quote(state_env)} cd {shlex.quote(str(working_dir))} && {combined} >> {shlex.quote(log_path)} 2>&1"
    )
    return snippet + "\n"
