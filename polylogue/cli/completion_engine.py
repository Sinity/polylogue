from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from ..local_sync import LOCAL_SYNC_PROVIDER_NAMES, get_local_provider
from ..commands import CommandEnv


@dataclass
class Completion:
    value: str
    description: str = ""


def _top_level_commands() -> List[Tuple[str, str]]:
    from .app import COMMAND_REGISTRY  # avoid circular import at module load

    entries: List[Tuple[str, str]] = []
    for name, info in COMMAND_REGISTRY._commands.items():  # pylint: disable=protected-access
        if name.startswith("_"):
            continue
        entries.append((name, info.help_text or ""))
    return sorted(entries, key=lambda item: item[0])


def _list_provider_names() -> List[str]:
    return ["drive", *LOCAL_SYNC_PROVIDER_NAMES]


class CompletionEngine:
    def __init__(self, env: CommandEnv, parser: argparse.ArgumentParser) -> None:
        self.env = env
        self.parser = parser

    def complete(self, shell: str, cword: int, words: Sequence[str]) -> List[Completion]:
        args = list(words)
        if args and args[0] == "--":
            args = args[1:]
        if args and args[0] == "polylogue":
            args = args[1:]
            cword = max(cword - 1, 0)
        if not args:
            return self._complete_commands()
        current_index = cword - 1
        if current_index < 0:
            current_index = 0
        if current_index >= len(args):
            args = args + [""]
            current_index = len(args) - 1
        current_word = args[current_index]
        command = args[0]
        if current_index == 0 and (current_word == "" or not current_word.startswith("-")):
            return self._complete_commands()
        if command == "render":
            return self._complete_render(current_index)
        if command == "sync":
            return self._complete_sync(args, current_index, current_word)
        if command == "import":
            return self._complete_import(args, current_index)
        if command == "inspect":
            return self._complete_inspect(args, current_index, current_word)
        if command == "watch":
            return self._complete_watch(args, current_index)
        if command == "status":
            return self._complete_status(current_word)
        if command == "settings":
            return self._complete_settings(current_word)
        if command == "help":
            return self._complete_commands()
        if command == "env":
            return self._complete_env()
        if command == "completions":
            return self._complete_completions()
        return []

    def _complete_commands(self) -> List[Completion]:
        return [Completion(value=name, description=desc) for name, desc in _top_level_commands()]

    def _complete_render(self, current_index: int) -> List[Completion]:
        if current_index == 1:
            return [Completion("__PATH__")]
        return []

    def _complete_sync(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
        if current_index == 1 and (not current_word or not current_word.startswith("-")):
            return [Completion(name) for name in _list_provider_names()]
        prev = args[current_index - 1] if current_index > 0 else ""
        provider = args[1] if len(args) > 1 else None
        if prev == "--session" and provider:
            return self._complete_local_sessions(provider, args)
        if prev == "--base-dir":
            return [Completion("__PATH__")]
        if prev == "--out":
            return [Completion("__PATH__")]
        if prev in {"--chat-id", "--conversation-id"} and provider == "drive":
            return self._drive_chat_ids()
        if current_word.startswith("--"):
            return self._option_completions("sync")
        return []

    def _complete_import(self, args: Sequence[str], current_index: int) -> List[Completion]:
        if current_index == 1:
            return [Completion(name) for name in ("chatgpt", "claude", "codex", "claude-code")]
        prev = args[current_index - 1] if current_index > 0 else ""
        if prev in {"--out", "--base-dir"}:
            return [Completion("__PATH__")]
        if current_index > 1 and args[1] in {"chatgpt", "claude"} and prev == "--source":
            return [Completion("__PATH__")]
        if current_word.startswith("--"):
            return self._option_completions("import")
        return []

    def _complete_inspect(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
        if current_index == 1:
            return [Completion(value) for value in ("branches", "search", "stats")]
        subcmd = args[1] if len(args) > 1 else None
        prev = args[current_index - 1] if current_index > 0 else ""
        if subcmd == "branches":
            if prev == "--slug":
                return self._slug_completions()
            if prev == "--conversation-id":
                return self._conversation_id_completions()
            if current_word.startswith("--"):
                return self._option_completions("inspect branches")
        if subcmd == "search" and current_word.startswith("--"):
            return self._option_completions("inspect search")
        if subcmd == "stats" and prev == "--dir":
            return [Completion("__PATH__")]
        if subcmd == "stats" and current_word.startswith("--"):
            return self._option_completions("inspect stats")
        return []

    def _complete_watch(self, args: Sequence[str], current_index: int) -> List[Completion]:
        if current_index == 1:
            return [Completion(name) for name in LOCAL_SYNC_PROVIDER_NAMES]
        prev = args[current_index - 1] if current_index > 0 else ""
        if prev in {"--base-dir", "--out"}:
            return [Completion("__PATH__")]
        if current_word.startswith("--"):
            return self._option_completions("watch")
        return []

    def _complete_status(self, current_word: str) -> List[Completion]:
        if current_word.startswith("--providers"):
            providers = sorted({provider for provider, *_ in self.env.conversations.iter_state()})
            return [Completion(p) for p in providers if p]
        if current_word.startswith("--dump") or current_word.startswith("--summary"):
            return [Completion("__PATH__")]
        if current_word.startswith("--"):
            return self._option_completions("status")
        return []

    def _complete_settings(self, current_word: str) -> List[Completion]:
        if current_word.startswith("--theme"):
            return [Completion("light"), Completion("dark")]
        if current_word.startswith("--html"):
            return [Completion("on"), Completion("off")]
        if current_word.startswith("--"):
            return self._option_completions("settings")
        return []

    def _complete_env(self) -> List[Completion]:
        return [Completion("--json", "Emit JSON output")]

    def _complete_completions(self) -> List[Completion]:
        return [Completion("--shell", "Target shell (bash/zsh/fish)")]

    def _option_completions(self, command: str) -> List[Completion]:
        parser = self._resolve_parser_for_command(command)
        if parser is None:
            return []
        pairs: List[Completion] = []
        for action in parser._actions:
            if not getattr(action, "option_strings", None):
                continue
            help_text = (action.help or "").replace("\n", " ")
            for opt in action.option_strings:
                pairs.append(Completion(opt, help_text))
        return pairs

    def _resolve_parser_for_command(self, command: str) -> argparse.ArgumentParser | None:
        parser = self.parser
        parts = command.split()
        if not parts:
            return None
        # First part should be a top-level command
        first = parts[0]
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction) and first in action.choices:
                parser = action.choices[first]
                break
        else:
            return None
        for part in parts[1:]:
            matched = False
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction) and part in action.choices:
                    parser = action.choices[part]
                    matched = True
                    break
            if not matched:
                return parser
        return parser

    def _slug_completions(self) -> List[Completion]:
        seen = set()
        entries: List[Completion] = []
        for _, _, payload in self.env.conversations.iter_state():
            slug = payload.get("slug")
            if slug and slug not in seen:
                entries.append(Completion(slug))
                seen.add(slug)
        entries.sort(key=lambda c: c.value)
        return entries

    def _conversation_id_completions(self) -> List[Completion]:
        seen = set()
        entries: List[Completion] = []
        for provider, conversation_id, payload in self.env.conversations.iter_state():
            if conversation_id and conversation_id not in seen:
                desc = f"{provider}:{payload.get('slug','')}"
                entries.append(Completion(conversation_id, desc))
                seen.add(conversation_id)
        entries.sort(key=lambda c: c.value)
        return entries

    def _drive_chat_ids(self) -> List[Completion]:
        seen = set()
        entries: List[Completion] = []
        for provider, _, payload in self.env.conversations.iter_state():
            if provider != "drive":
                continue
            extra = payload.get("extra_state") or payload.get("extraState") or {}
            drive_id = extra.get("driveFileId")
            slug = payload.get("slug")
            if drive_id and drive_id not in seen:
                entries.append(Completion(drive_id, slug or "drive chat"))
                seen.add(drive_id)
        entries.sort(key=lambda c: c.value)
        return entries

    def _complete_local_sessions(self, provider: str, args: Sequence[str]) -> List[Completion]:
        if provider == "drive":
            return []
        try:
            provider_obj = get_local_provider(provider)
        except ValueError:
            return []
        base_dir = self._get_option_value(args, "--base-dir")
        base = Path(base_dir).expanduser() if base_dir else provider_obj.default_base
        sessions = provider_obj.list_sessions(base)
        return [Completion(str(path)) for path in sessions]

    @staticmethod
    def _get_option_value(args: Sequence[str], option: str) -> str | None:
        for idx, word in enumerate(args):
            if word == option and idx + 1 < len(args):
                return args[idx + 1]
            if word.startswith(option + "="):
                return word.split("=", 1)[1]
        return None
