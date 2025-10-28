from __future__ import annotations

import sys
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "polylogue.py"


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


def build_command(target: AutomationTarget, extra_args: Iterable[str]) -> str:
    parts = [sys.executable, str(SCRIPT_PATH)] + target.command + list(extra_args)
    return " ".join(parts)


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


def _merge_args(
    target: AutomationTarget,
    user_extra: Iterable[str],
    collapse_threshold: Optional[int],
    html: Optional[bool],
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
    html_pref: Optional[bool]
    if html is None:
        if user_requested_html:
            html_pref = True
        else:
            html_pref = defaults.get("html")
    else:
        html_pref = html

    if html_pref:
        if "--html" not in args:
            args.append("--html")
    else:
        args = [flag for flag in args if flag != "--html"]

    return args


def prepare_automation_command(
    target_key: str,
    *,
    user_extra: Iterable[str],
    collapse_threshold: Optional[int],
    html: Optional[bool],
) -> Tuple[AutomationTarget, List[str]]:
    target = resolve_target(target_key)
    merged = _merge_args(target, user_extra, collapse_threshold, html)
    return target, merged


def systemd_snippet(
    *,
    target_key: str,
    interval: str,
    working_dir: Path,
    extra_args: Iterable[str] = (),
    boot_delay: str = "2m",
    collapse_threshold: Optional[int] = None,
    html: Optional[bool] = None,
) -> str:
    target, merged_args = prepare_automation_command(
        target_key,
        user_extra=extra_args,
        collapse_threshold=collapse_threshold,
        html=html,
    )
    service_name = target.name
    command = build_command(target, merged_args)
    service = dedent(
        f"""# ~/.config/systemd/user/{service_name}.service
[Unit]
Description={target.description}

[Service]
Type=oneshot
WorkingDirectory={working_dir}
ExecStart={command}
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
    html: Optional[bool] = None,
) -> str:
    target, merged_args = prepare_automation_command(
        target_key,
        user_extra=extra_args,
        collapse_threshold=collapse_threshold,
        html=html,
    )
    command = build_command(target, merged_args)
    snippet = (
        f"{schedule} XDG_STATE_HOME=\"{state_env}\" cd {working_dir} && {command} >> \"{log_path}\" 2>&1"
    )
    return snippet + "\n"
