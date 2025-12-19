from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, TypeVar

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
        except re.error as e:
            raise ValueError(f"Invalid name filter regex '{name_filter}': {e}") from e
    s_epoch = parse_input_time_to_epoch(since)
    u_epoch = parse_input_time_to_epoch(until)
    if s_epoch is not None or u_epoch is not None:
        tmp: List[Dict[str, Any]] = []
        dropped_missing_modified = 0
        for c in out:
            mt = parse_rfc3339_to_epoch(c.get("modifiedTime"))
            if mt is None:
                dropped_missing_modified += 1
                continue
            if s_epoch is not None and mt < s_epoch:
                continue
            if u_epoch is not None and mt > u_epoch:
                continue
            tmp.append(c)
        out = tmp
        if dropped_missing_modified:
            print(
                f"Skipped {dropped_missing_modified} chat(s) missing modifiedTime; unable to apply --since/--until.",
                file=sys.stderr,
            )
    out.sort(
        key=lambda c: (
            -(parse_rfc3339_to_epoch(c.get("modifiedTime")) or 0.0),
            (c.get("name") or "").lower(),
            str(c.get("id") or ""),
        )
    )
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
    plain: bool = False,
) -> Optional[List[str]]:
    """Return the lines selected via skim, or None if the picker was aborted."""

    if not lines:
        return []

    import sys

    if plain or not sys.stdin.isatty():
        return None

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
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Required command 'sk' (skim fuzzy finder) is not available in PATH. "
            "Install it with: cargo install skim, or use --json mode for non-interactive operation."
        ) from exc
    except subprocess.CalledProcessError:
        return None

    output = proc.stdout.decode("utf-8").strip()
    if not output:
        return []
    return [line for line in output.splitlines() if line.strip()]


T = TypeVar("T")


def choose_single_entry(
    ui,
    entries: Sequence[T],
    *,
    format_line: Callable[[T, int], str],
    header: Optional[str] = None,
    prompt: Optional[str] = None,
    preview: Optional[str] = None,
) -> Tuple[Optional[T], bool]:
    """Unified helper for skim-based single selection.

    Returns (selection, cancelled). When cancelled is True, the caller should treat the
    action as aborted by the user. When selection is None and cancelled is False, the caller
    may fall back to a default behaviour (e.g., first entry or no-op).
    """

    if not entries:
        return None, False

    mapping: Dict[str, T] = {}
    lines: List[str] = []
    for idx, entry in enumerate(entries):
        body = format_line(entry, idx)
        line = f"{idx}\t{body}"
        mapping[str(idx)] = entry
        lines.append(line)

    if getattr(ui, "plain", False):
        # Avoid blocking on stdin in non-interactive environments.
        import sys

        if not sys.stdin.isatty():
            if hasattr(ui, "console"):
                ui.console.print("[yellow]Plain mode cannot prompt; pass --all/IDs or rerun with --interactive.")
            return None, True
        if header:
            ui.console.print(header)
        for idx, line in enumerate(lines):
            ui.console.print(line)
        while True:
            try:
                response = input((prompt or "select") + " [index or blank to cancel]: ").strip()
            except EOFError:
                return None, True
            if not response:
                return None, True
            if response.isdigit():
                value = int(response)
                if 0 <= value < len(entries):
                    return entries[value], False
            ui.console.print("[yellow]Enter a valid numeric index or cancel with Enter.")

    selection = sk_select(
        lines,
        multi=False,
        preview=preview,
        header=header,
        prompt=prompt,
    )
    if selection is None:
        return None, True  # cancelled or skim unavailable
    if not selection:
        return None, False
    key = selection[0].split("\t", 1)[0]
    return mapping.get(key), False


def compute_prune_paths(out_dir: Path, wanted_basenames: Set[str]) -> List[Path]:
    to_delete: List[Path] = []
    for p in out_dir.iterdir():
        if p.is_dir():
            slug = p.name
            if slug not in wanted_basenames:
                to_delete.append(p)
            continue
        if p.is_file() and p.suffix.lower() == ".md":
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


def handle_selection_result(
    selection: Optional[List],
    *,
    console,
    action: str,
    item_type: str = "items"
) -> Optional[List]:
    """Handle picker selection with standardized messages.

    Args:
        selection: Result from picker (None if cancelled, [] if empty)
        console: Console object for printing messages
        action: Action being performed (e.g., "sync", "import")
        item_type: Type of items being selected (e.g., "conversations", "files")

    Returns:
        The selection if valid, None if cancelled/empty
    """
    if selection is None:
        console.print(f"[yellow]{action.title()} cancelled; no {item_type} selected.")
        return None
    if not selection:
        console.print(f"[yellow]No {item_type} selected; nothing to {action}.")
        return None
    return selection


def resolve_inputs(path: Path, plain: bool) -> Optional[List[Path]]:
    """Resolve input JSON files from a path, with interactive picker if needed.

    Args:
        path: File or directory path to resolve
        plain: If True, skip interactive picker

    Returns:
        List of selected files, None if picker was cancelled, [] if none selected

    Raises:
        SystemExit: If path doesn't exist
    """
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Input path not found: {path}")
    candidates = [
        p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() == ".json"
    ]
    if len(candidates) <= 1:
        return candidates
    if plain:
        if not sys.stdin.isatty():
            print("[yellow]Plain mode cannot prompt for input selection; specify explicit file paths or rerun with --interactive.")
            return None
        print("[yellow]Multiple inputs found; plain mode requires an explicit selection. Rerun with --interactive or point at a single file.")
        return None
    lines = [str(p) for p in candidates]
    selection = sk_select(
        lines,
        preview="bat --style=plain {}",
        bindings=["ctrl-g:execute(glow --style=dark {+})"],
    )
    if selection is None:
        return None
    if not selection:
        return []
    return [Path(s) for s in selection]
