"""Executable contract for the optional zsh terminal-note widget."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def _zsh_widget_snippet() -> str:
    document = Path("docs/getting-started.md").read_text(encoding="utf-8")
    section = document.split("## Optional terminal note widget (zsh)", maxsplit=1)[1]
    return section.split("```zsh\n", maxsplit=1)[1].split("\n```", maxsplit=1)[0]


def test_zsh_widget_preserves_the_previous_status_and_quotes_history(tmp_path: Path) -> None:
    """The real documented widget survives earlier hooks and hostile history text.

    This executes the exact documentation snippet. Removing its first-hook
    registration makes the expected status become zero; replacing the zsh
    quote expansion with the former hand-written double quotes executes the
    command substitution and creates ``marker``.
    """

    marker = tmp_path / "history-command-substitution-ran"
    dangerous_history = f'quote"; $(touch {marker}); `touch {marker}`\nnext'
    expected_note_text = f"{dangerous_history} [exit 1]"
    script = f"""\
precmd_functions=(later_hook)
later_hook() {{ return 0; }}
zle() {{ :; }}
bindkey() {{ :; }}
fc() {{ print -r -- {shlex.quote(dangerous_history)}; }}
{_zsh_widget_snippet()}
false
for hook in "${{precmd_functions[@]}}"; do
  "$hook"
done
_polylogue_note_prefill
[[ "$POLYLOGUE_NOTE_LAST_STATUS" == 1 ]] || exit 10
eval "set -- $BUFFER"
[[ "$1" == polylogue && "$2" == note && "$3" == --ref && "$4" == last ]] || exit 11
[[ "$5" == {shlex.quote(expected_note_text)} ]] || exit 12
[[ ! -e {shlex.quote(str(marker))} ]] || exit 13
"""
    completed = subprocess.run(
        ["zsh", "-f", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
