from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ..automation import REPO_ROOT, TARGETS, cron_snippet, describe_targets, systemd_snippet
from ..commands import CommandEnv


def run_automation_cli(args: argparse.Namespace, env: CommandEnv) -> None:  # noqa: D401
    """Print scheduler snippets for automation targets."""

    target = TARGETS[args.target]
    defaults = target.defaults or {}

    if args.automation_format == "describe":
        data = describe_targets(getattr(args, "target", None))
        import json

        print(json.dumps(data, indent=2))
        return

    working_dir_value = getattr(args, "working_dir", None)
    if working_dir_value is None and defaults.get("workingDir"):
        working_dir_value = defaults["workingDir"]
    working_dir = Path(working_dir_value) if working_dir_value else REPO_ROOT
    working_dir = working_dir.resolve()

    extra_args: List[str] = []
    out_value = getattr(args, "out", None)
    if out_value is None and defaults.get("outputDir"):
        out_value = defaults["outputDir"]
    if out_value:
        extra_args.extend(["--out", str(Path(out_value).resolve())])
    extra_args.extend(getattr(args, "extra_arg", []) or [])

    collapse_value = getattr(args, "collapse_threshold", None)
    html_mode = getattr(args, "html_mode", None)
    html_override = html_mode if html_mode not in (None, "auto") else None

    if args.automation_format == "systemd":
        snippet = systemd_snippet(
            target_key=args.target,
            interval=args.interval,
            working_dir=working_dir,
            extra_args=extra_args,
            boot_delay=args.boot_delay,
            collapse_threshold=collapse_value,
            html=html_override,
        )
    else:
        snippet = cron_snippet(
            target_key=args.target,
            schedule=args.schedule,
            working_dir=working_dir,
            log_path=args.log,
            extra_args=extra_args,
            state_env=args.state_home,
            collapse_threshold=collapse_value,
            html=html_override,
        )
    print(snippet, end="")


__all__ = ["run_automation_cli"]
