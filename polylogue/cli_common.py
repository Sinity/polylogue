from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from .util import parse_input_time_to_epoch, parse_rfc3339_to_epoch


def filter_chats(
    chats: List[Dict[str, Any]],
    name_filter: Optional[str],
    since: Optional[str],
    until: Optional[str],
) -> List[Dict[str, Any]]:
    out = chats
    if name_filter:
        try:
            rx = re.compile(name_filter)
            out = [c for c in out if rx.search(c.get("name", "") or "")]
        except re.error:
            pass
    s_epoch = parse_input_time_to_epoch(since)
    u_epoch = parse_input_time_to_epoch(until)
    if s_epoch or u_epoch:
        tmp: List[Dict[str, Any]] = []
        for c in out:
            mt = parse_rfc3339_to_epoch(c.get("modifiedTime"))
            if mt is None:
                continue
            if s_epoch and mt < s_epoch:
                continue
            if u_epoch and mt > u_epoch:
                continue
            tmp.append(c)
        out = tmp
    return out


DEFAULT_SK_BINDINGS = (
    "tab:toggle+down",
    "btab:toggle+up",
    "ctrl-a:select-all",
    "ctrl-d:deselect-all",
    "ctrl-space:toggle",
    "alt-a:select-all+accept",
)


def sk_select(
    lines: Sequence[str],
    *,
    multi: bool = True,
    preview: Optional[str] = None,
    header: Optional[str] = None,
    bindings: Optional[Sequence[str]] = None,
    prompt: Optional[str] = None,
    cycle: bool = True,
) -> Optional[List[str]]:
    """Return the lines selected via skim, or None if the picker was aborted."""

    if not lines:
        return []

    cmd = ["sk", "--ansi"]
    if multi:
        cmd.append("--multi")
    if cycle:
        cmd.append("--cycle")
    if prompt:
        cmd.extend(["--prompt", prompt])
    if preview:
        cmd.extend(["--preview", preview])
    if header:
        cmd.extend(["--header", header])
    effective_bindings = list(bindings) if bindings else []
    if multi and not bindings:
        effective_bindings = list(DEFAULT_SK_BINDINGS)
    for binding in effective_bindings:
        cmd.extend(["--bind", binding])

    try:
        proc = subprocess.run(
            cmd,
            input="\n".join(lines).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    output = proc.stdout.decode("utf-8").strip()
    if not output:
        return []
    return [line for line in output.splitlines() if line.strip()]


def compute_prune_paths(out_dir: Path, wanted_basenames: Set[str]) -> List[Path]:
    to_delete: List[Path] = []
    for p in out_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".md":
            base = p.stem
            if base not in wanted_basenames:
                to_delete.append(p)
        elif p.is_dir() and p.name.endswith("_attachments"):
            base = p.name[:-12]
            if base not in wanted_basenames:
                to_delete.append(p)
    return to_delete


def should_download(local_path: Path, remote_modified_time: Optional[str], *, force: bool) -> bool:
    if force:
        return True
    if not local_path.exists():
        return True
    mt = parse_rfc3339_to_epoch(remote_modified_time)
    if mt is None:
        return False
    try:
        return int(local_path.stat().st_mtime) != int(mt)
    except Exception:
        return True
