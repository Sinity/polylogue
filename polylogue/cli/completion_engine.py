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
        current_index = max(cword, 0)
        if current_index >= len(args):
            args = args + [""]
            current_index = len(args) - 1
        current_word = args[current_index]
        command = args[0]
        if current_index == 0 and len(args) == 1 and (current_word == "" or not current_word.startswith("-")):
            return self._complete_commands()
        if command == "search":
            return self._complete_search(args, current_index, current_word)
        if command == "sync":
            return self._complete_sync(args, current_index, current_word)
        if command == "import":
            return self._complete_import(args, current_index, current_word)
        if command == "browse":
            return self._complete_browse(args, current_index, current_word)
        if command == "maintain":
            return self._complete_maintain(args, current_index, current_word)
        if command == "config":
            return self._complete_config(args, current_index, current_word)
        if command == "help":
            return self._complete_commands()
        if command == "completions":
            return self._complete_completions()
        return []

    def _complete_commands(self) -> List[Completion]:
        return [Completion(value=name, description=desc) for name, desc in _top_level_commands()]

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

    def _complete_search(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
        prev = args[current_index - 1] if current_index > 0 else ""
        if prev == "--provider":
            return self._provider_completions()
        if prev == "--slug":
            return self._slug_completions()
        if prev == "--conversation-id":
            return self._conversation_id_completions()
        if prev == "--branch":
            return self._branch_completions()
        if prev == "--model":
            return self._model_completions()
        if current_word.startswith("--"):
            return self._option_completions("search")
        return []

    def _complete_import(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
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

    def _complete_browse(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
        if current_index == 1:
            return [Completion(value) for value in ("branches", "stats", "status", "runs")]
        subcmd = args[1] if len(args) > 1 else None
        prev = args[current_index - 1] if current_index > 0 else ""
        if subcmd == "branches":
            if prev == "--slug":
                return self._slug_completions()
            if prev == "--conversation-id":
                return self._conversation_id_completions()
            if prev == "--provider":
                return self._provider_completions()
            if prev == "--branch":
                return self._branch_completions()
            if prev in {"--out", "--theme"}:
                return [Completion("__PATH__")] if prev == "--out" else [Completion("light"), Completion("dark")]
            if current_word.startswith("--"):
                return self._option_completions("browse branches")
        if subcmd == "stats":
            if prev == "--dir":
                return [Completion("__PATH__")]
            if current_word.startswith("--"):
                return self._option_completions("browse stats")
        if subcmd == "status":
            if prev == "--providers":
                return self._provider_completions()
            if prev in {"--dump", "--summary"}:
                return [Completion("__PATH__")]
            if current_word.startswith("--"):
                return self._option_completions("browse status")
        if subcmd == "runs":
            if prev == "--providers":
                return self._provider_completions()
            if current_word.startswith("--"):
                return self._option_completions("browse runs")
        return []

    def _complete_maintain(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
        if current_index == 1:
            return [Completion(value) for value in ("prune", "doctor", "index")]
        subcmd = args[1] if len(args) > 1 else None
        prev = args[current_index - 1] if current_index > 0 else ""
        if subcmd == "prune":
            if prev == "--dir":
                return [Completion("__PATH__")]
            if current_word.startswith("--"):
                return self._option_completions("maintain prune")
        if subcmd == "doctor" and current_word.startswith("--"):
            return self._option_completions("maintain doctor")
        if subcmd == "index":
            if prev == "--dir":
                return [Completion("__PATH__")]
            if current_word.startswith("--"):
                return self._option_completions("maintain index")
        return []

    def _complete_config(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
        if current_index == 1:
            return [Completion(value) for value in ("init", "set", "show")]
        subcmd = args[1] if len(args) > 1 else None
        prev = args[current_index - 1] if current_index > 0 else ""
        if subcmd == "init" and current_word.startswith("--"):
            return self._option_completions("config init")
        if subcmd == "set":
            if prev == "--theme":
                return [Completion("light"), Completion("dark")]
            if prev == "--html":
                return [Completion("on"), Completion("off")]
            if current_word.startswith("--"):
                return self._option_completions("config set")
        if subcmd == "show":
            if current_word.startswith("--"):
                return self._option_completions("config show")
        return []

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
            help_text = self._format_action_help(action)
            for opt in action.option_strings:
                pairs.append(Completion(opt, help_text))
        return pairs

    def _format_action_help(self, action: argparse.Action) -> str:
        help_text = (action.help or "").replace("\n", " ").strip()
        default = getattr(action, "default", None)
        show_default = default is not argparse.SUPPRESS
        if isinstance(default, str) and not default:
            show_default = False
        if default in {None, False} and not isinstance(default, bool):
            show_default = False
        if not show_default:
            return help_text
        if isinstance(default, bool):
            default_text = "on" if default else "off"
        elif isinstance(default, (list, tuple)):
            default_text = ", ".join(str(value) for value in default) or "[]"
        else:
            default_text = str(default)
        if help_text:
            return f"{help_text} (default: {default_text})"
        return f"(default: {default_text})"

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

    def _provider_completions(self) -> List[Completion]:
        providers = sorted({provider for provider, *_ in self.env.conversations.iter_state()})
        return [Completion(p) for p in providers if p]

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

    def _branch_completions(self) -> List[Completion]:
        seen = set()
        entries: List[Completion] = []
        for _, _, payload in self.env.conversations.iter_state():
            branches = payload.get("branches", {})
            for branch_id in branches.keys():
                if branch_id and branch_id not in seen:
                    entries.append(Completion(branch_id))
                    seen.add(branch_id)
        entries.sort(key=lambda c: c.value)
        return entries

    def _model_completions(self) -> List[Completion]:
        seen = set()
        entries: List[Completion] = []
        for _, _, payload in self.env.conversations.iter_state():
            model = payload.get("model") or payload.get("metadata", {}).get("model")
            if model and model not in seen:
                entries.append(Completion(model))
                seen.add(model)
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
