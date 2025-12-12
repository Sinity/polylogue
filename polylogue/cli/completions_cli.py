from __future__ import annotations

import textwrap
from typing import Dict, List, Optional

import click

from ..commands import CommandEnv
from ..ui import create_ui
from .click_introspect import click_command_entries
from .completion_engine import CompletionEngine


def _bash_dynamic_script() -> str:
    return textwrap.dedent(
        """
        _polylogue_complete() {
            local IFS=$'\\n'
            local completions
            completions=$(polylogue _complete --shell bash --cword $COMP_CWORD -- "${COMP_WORDS[@]}" 2>/dev/null)
            if [[ $? -ne 0 ]]; then
                return
            fi
            local first=$(echo "$completions" | head -1)
            if [[ $first == "__PATH__" ]]; then
                COMPREPLY=( $(compgen -f -- "${COMP_WORDS[COMP_CWORD]}") )
                return
            fi
            COMPREPLY=( $(compgen -W "$completions" -- "${COMP_WORDS[COMP_CWORD]}") )
        }
        complete -F _polylogue_complete polylogue
        """
    ).strip()


def _fish_dynamic_script() -> str:
    return textwrap.dedent(
        """
        function __polylogue_complete
            set -l cmd (commandline -opc)
            set -l cword (count $cmd)
            polylogue _complete --shell fish --cword $cword -- $cmd 2>/dev/null | while read -l line
                if string match -q "__PATH__*" -- $line
                    __fish_complete_path
                    continue
                end
                if string match -q "*:*" -- $line
                    set -l parts (string split -m 1 ":" -- $line)
                    echo $parts[1]\\t$parts[2]
                else
                    echo $line
                end
            end
        end
        complete -c polylogue -f -a "(__polylogue_complete)"
        """
    ).strip()


def _completion_script(shell: str, commands: List[str], descriptions: Optional[Dict[str, str]] = None) -> str:
    # Deprecated fallback for bash/fish - all shells now use dynamic completions.
    if shell == "bash":
        return _bash_dynamic_script()
    # fish
    desc_map = descriptions or {}
    static_lines: List[str] = []
    for name in commands:
        if name.startswith("_"):
            continue
        desc = desc_map.get(name, "")
        if desc:
            escaped_desc = desc.replace('"', '\"')
            static_lines.append(f"complete -c polylogue -n '__fish_use_subcommand' -a '{name}' -d \"{escaped_desc}\"")
        else:
            static_lines.append(f"complete -c polylogue -n '__fish_use_subcommand' -a '{name}'")
    static_block = "\n".join(static_lines)
    return _fish_dynamic_script() + ("\n" + static_block if static_block else "")


def _zsh_dynamic_script() -> str:
    return textwrap.dedent(
        """
        #compdef polylogue

        _polylogue_complete() {
            local -a completions
            local IFS=$'\\n'
            completions=($(polylogue _complete --shell zsh --cword $CURRENT -- "${words[@]}"))
            if [[ $? -ne 0 ]]; then
                return
            fi
            if [[ ${#completions[@]} -gt 0 ]]; then
                local first=${completions[1]}
                if [[ $first == "__PATH__" || $first == "__PATH__:"* ]]; then
                    _files
                    return
                fi
            fi
            _describe 'values' completions
        }

        compdef _polylogue_complete polylogue
        """
    ).strip()


def run_completions_cli(args: object, env: CommandEnv, root: click.Group) -> None:
    entries = click_command_entries(root)
    commands = [name for name, _ in entries]
    descriptions = {name: desc for name, desc in entries if desc}
    shell = getattr(args, "shell", None)
    if shell == "zsh":
        print(_zsh_dynamic_script())
        return
    script = _completion_script(str(shell), commands, descriptions)
    print(script)


def run_complete_cli(args: object, env: Optional[CommandEnv], root: click.Group) -> None:
    env = env or CommandEnv(ui=create_ui(True))
    engine = CompletionEngine(env, root)
    completions = engine.complete(getattr(args, "shell", ""), getattr(args, "cword", 0), getattr(args, "words", ()) or ())
    for entry in completions:
        if entry.description:
            print(f"{entry.value}:{entry.description}")
        else:
            print(entry.value)


__all__ = ["run_completions_cli", "run_complete_cli"]

