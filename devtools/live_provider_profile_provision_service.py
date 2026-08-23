"""Provision the fixed copied Chrome profile for ``live_provider_proof``.

This private AgentCTL operation copies only the current Chrome root's ``Local
State`` and ``Default`` profile into the fixed proof profile. It neither starts
Chrome nor accepts a caller-selected source, destination, or launch argument.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from devtools.sinnixd_service_context import require_declared_operation_context

_SOURCE_ROOT = Path.home() / ".config" / "google-chrome"
_DESTINATION_ROOT = Path("/realm/state/polylogue/live-provider-proof-profile")
_MAX_FILES = 20_000
_MAX_BYTES = 1_000_000_000
_EXCLUDED_NAMES = frozenset({"SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile"})
_MAX_ERROR_MESSAGE = 512


def _copy_selected_profile(*, source_root: Path, destination_root: Path) -> dict[str, int]:
    selected = (source_root / "Local State", source_root / "Default")
    missing = [str(path.relative_to(source_root)) for path in selected if not path.exists()]
    if missing:
        raise ValueError(f"copied-profile source is missing required path(s): {', '.join(missing)}")
    if not (source_root / "Default").is_dir():
        raise ValueError("copied-profile source Default must be a directory")
    if destination_root == source_root or source_root in destination_root.parents:
        raise ValueError("copied-profile destination must be separate from the live source")

    staged = destination_root.with_name(f".{destination_root.name}.staging-{os.getpid()}")
    if staged.exists():
        shutil.rmtree(staged)
    file_count = 0
    byte_count = 0
    try:
        staged.mkdir(parents=True)
        for source in selected:
            relative = source.relative_to(source_root)
            target = staged / relative
            if source.is_symlink():
                raise ValueError(f"copied-profile source cannot contain symlinked root: {relative}")
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                file_count += 1
                byte_count += source.stat().st_size
                continue
            target.mkdir(parents=True, exist_ok=True)
            for item in source.rglob("*"):
                relative_item = item.relative_to(source_root)
                if item.name in _EXCLUDED_NAMES:
                    continue
                if item.is_symlink():
                    raise ValueError(f"copied-profile source cannot contain symlink: {relative_item}")
                target_item = staged / relative_item
                if item.is_dir():
                    target_item.mkdir(parents=True, exist_ok=True)
                    continue
                if not item.is_file():
                    raise ValueError(f"copied-profile source has unsupported entry: {relative_item}")
                file_count += 1
                byte_count += item.stat().st_size
                if file_count > _MAX_FILES or byte_count > _MAX_BYTES:
                    raise ValueError("copied-profile source exceeds the declared provisioning bound")
                target_item.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target_item)
        if destination_root.exists():
            shutil.rmtree(destination_root)
        destination_root.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(destination_root)
    except BaseException:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    return {"files": file_count, "bytes": byte_count}


def provision_profile() -> dict[str, object]:
    """Copy the fixed current Chrome profile as a finite provisioning job."""
    require_declared_operation_context("live_provider_profile_provision")
    copied = _copy_selected_profile(source_root=_SOURCE_ROOT, destination_root=_DESTINATION_ROOT)
    return {"ok": True, "profile": {"files": copied["files"], "bytes": copied["bytes"]}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit one bounded JSON result.")
    parser.parse_args(argv)
    try:
        payload: dict[str, Any] = provision_profile()
    except (OSError, ValueError, shutil.Error) as error:
        payload = {"ok": False, "error": {"type": type(error).__name__, "message": str(error)[:_MAX_ERROR_MESSAGE]}}
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
