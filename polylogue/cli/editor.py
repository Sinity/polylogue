"""Editor integration utilities."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def get_editor() -> Optional[str]:
    """Get user's preferred editor from environment.

    Checks $EDITOR then $VISUAL environment variables.

    Returns:
        Editor command name/path, or None if not set
    """
    return os.environ.get("EDITOR") or os.environ.get("VISUAL")


def open_in_editor(path: Path, line: Optional[int] = None) -> bool:
    """Open file in user's preferred editor.

    Supports line number jumps for editors that support it (vim, nvim, code, subl).

    Args:
        path: File path to open
        line: Optional line number to jump to

    Returns:
        True if editor was launched successfully, False otherwise
    """
    editor = get_editor()
    if not editor:
        return False

    if not path.exists():
        return False

    # Build editor command with line number if applicable
    cmd = [editor]

    if line:
        # Different editors have different line jump syntax
        editor_lower = editor.lower()
        if "vim" in editor_lower or "nvim" in editor_lower:
            cmd.append(f"+{line}")
        elif "code" in editor_lower or "subl" in editor_lower or "atom" in editor_lower:
            # VS Code, Sublime Text, Atom
            cmd.append(f"{path}:{line}")
            return _run_editor(cmd)
        elif "emacs" in editor_lower:
            cmd.append(f"+{line}")
        # For unknown editors, just open the file without line number

    cmd.append(str(path))
    return _run_editor(cmd)


def _run_editor(cmd: list[str]) -> bool:
    """Run editor command.

    Args:
        cmd: Command list to execute

    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(cmd, check=False)
        return True
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False
