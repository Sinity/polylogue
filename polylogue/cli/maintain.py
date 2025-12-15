"""Maintain command - consolidated maintenance interface."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

from ..commands import CommandEnv
from ..schema import stamp_payload


def run_maintain_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    """Dispatch to appropriate maintenance subcommand."""
    from .prune_cli import run_prune_cli
    from .doctor import run_doctor_cli
    from .index_cli import run_index_cli
    from ..util import preflight_disk_requirement

    maintain_cmd = args.maintain_cmd

    if maintain_cmd == "prune":
        run_prune_cli(args, env)
    elif maintain_cmd == "doctor":
        run_doctor_cli(args, env)
    elif maintain_cmd == "index":
        run_index_cli(args, env)
    elif maintain_cmd == "restore":
        src: Path = Path(getattr(args, "src"))
        dest: Path = Path(getattr(args, "dest"))
        force = bool(getattr(args, "force", False))
        json_mode = bool(getattr(args, "json", False))

        if not src.exists() or not src.is_dir():
            raise SystemExit(f"Snapshot directory not found: {src}")
        if dest.exists():
            if not force:
                raise SystemExit(f"Destination already exists: {dest} (use --force to overwrite)")
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Estimate disk need (roughly size of src)
        try:
            total_bytes = sum(p.stat().st_size for p in src.rglob("*"))
        except Exception:
            total_bytes = 0
        preflight_disk_requirement(projected_bytes=total_bytes, limit_gib=getattr(args, "max_disk", None), ui=env.ui)

        shutil.copytree(src, dest, dirs_exist_ok=True)
        if json_mode:
            payload = stamp_payload({"from": str(src), "to": str(dest), "bytes": total_bytes})
            import json

            print(json.dumps(payload, indent=2, sort_keys=True))
            return
        env.ui.summary(
            "Restore",
            [
                f"Source: {src}",
                f"Destination: {dest}",
                f"Bytes (approx): {total_bytes}",
            ],
        )
    else:
        raise SystemExit(f"Unknown maintain sub-command: {maintain_cmd}")


__all__ = ["run_maintain_cli"]
