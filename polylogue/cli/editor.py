"""Editor/browser integration utilities."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import webbrowser
from pathlib import Path

# Pattern for detecting shell metacharacters that could enable command injection.
# NOTE: stricter than pipeline/events._UNSAFE_PATTERN — blocks ALL $ (not just $()
# because editor commands have no legitimate use for env var references.
_UNSAFE_PATTERN = re.compile(r'[;&|`$(){}[\]<>!\\]')


def validate_command(command: str, context: str = "command") -> None:
    """Validate command string for shell injection risks.

    Rejects commands containing shell metacharacters that could be used for
    injection attacks when parsed by shells or passed unsafely.

    Args:
        command: Command string to validate
        context: Description of context (for error messages)

    Raises:
        ValueError: If command contains unsafe shell metacharacters
    """
    if not command or not command.strip():
        raise ValueError(f"{context} cannot be empty")

    # Check for shell metacharacters
    if _UNSAFE_PATTERN.search(command):
        raise ValueError(
            f"{context} contains unsafe shell metacharacters: {command!r}. "
            "Use a simple command like 'vim' or '/usr/bin/code --wait'"
        )


def get_editor() -> str | None:
    """Get user's preferred editor from environment.

    Checks $EDITOR then $VISUAL environment variables.

    Returns:
        Editor command name/path, or None if not set
    """
    return os.environ.get("EDITOR") or os.environ.get("VISUAL")


def open_in_editor(path: Path, line: int | None = None) -> bool:
    """Open file in user's preferred editor.

    Supports line number jumps for editors that support it (vim, nvim, code, subl).

    Args:
        path: File path to open
        line: Optional line number to jump to

    Returns:
        True if editor was launched successfully, False otherwise

    Raises:
        ValueError: If EDITOR contains unsafe shell metacharacters
    """
    editor = get_editor()
    if not editor:
        return False

    if not path.exists():
        return False

    # Validate editor command for injection risks before use
    try:
        validate_command(editor, context="$EDITOR")
    except ValueError:
        return False

    try:
        cmd = shlex.split(editor)
    except ValueError:
        cmd = [editor]
    if not cmd:
        return False

    if line:
        # Different editors have different line jump syntax
        editor_lower = " ".join(cmd).lower()
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


def open_in_browser(path: Path, anchor: str | None = None) -> bool:
    """Open a file in the system browser/HTML handler.

    Raises:
        ValueError: If POLYLOGUE_BROWSER contains unsafe shell metacharacters
    """

    try:
        resolved = path.resolve()
        target = resolved.as_uri()
    except (ValueError, OSError):
        # Handle invalid paths (null characters, invalid Unicode, etc.)
        return False

    if anchor:
        target = f"{target}#{anchor}"

    custom_browser = os.environ.get("POLYLOGUE_BROWSER")
    if custom_browser:
        # Validate browser command for injection risks before use
        try:
            validate_command(custom_browser, context="$POLYLOGUE_BROWSER")
        except ValueError:
            return False

        try:
            cmd = shlex.split(custom_browser)
        except ValueError:
            cmd = [custom_browser]
        cmd.append(target)
        try:
            subprocess.Popen(cmd)
            return True
        except (OSError, subprocess.SubprocessError):
            # Failed to launch custom browser
            return False

    try:
        return webbrowser.open(target)
    except (OSError, webbrowser.Error):
        # Failed to open in default browser
        return False
