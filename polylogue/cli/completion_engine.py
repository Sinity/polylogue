from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Sequence, Tuple

import click

from ..commands import CommandEnv
from ..local_sync import LOCAL_SYNC_PROVIDER_NAMES, get_local_provider
from .. import paths as paths_module
from .click_introspect import click_command_entries


@dataclass
class Completion:
    value: str
    description: str = ""


def _list_provider_names() -> List[str]:
    return ["drive", *LOCAL_SYNC_PROVIDER_NAMES]


class _CompletionCache:
    """Small persistent cache for CLI completions.

    Shell completion hooks run `polylogue _complete` for each keystroke, so we
    keep a cheap cache of state-derived lists. Cache invalidation is based on
    the state DB mtime/size, and local session listings use a short TTL.
    """

    version = 1

    def __init__(self) -> None:
        self.path = paths_module.CACHE_HOME / "completions.json"

    def load(self) -> dict:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"version": self.version}
        except Exception:
            return {"version": self.version}
        try:
            data = json.loads(raw)
        except Exception:
            return {"version": self.version}
        if not isinstance(data, dict) or data.get("version") != self.version:
            return {"version": self.version}
        return data

    def save(self, payload: dict) -> None:
        paths_module.CACHE_HOME.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload["version"] = self.version
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with NamedTemporaryFile("w", encoding="utf-8", dir=str(paths_module.CACHE_HOME), delete=False) as tmp:
            tmp.write(encoded)
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.path)


class CompletionEngine:
    def __init__(self, env: CommandEnv, root: click.Group) -> None:
        self.env = env
        self.root = root
        self._cache = _CompletionCache()

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
        if command == "render":
            return self._complete_render(args, current_index, current_word)
        if command == "search":
            return self._complete_search(args, current_index, current_word)
        if command == "sync":
            return self._complete_sync(args, current_index, current_word)
        if command == "import":
            return self._complete_import(args, current_index, current_word)
        if command == "verify":
            return self._complete_verify(args, current_index, current_word)
        if command == "browse":
            return self._complete_browse(args, current_index, current_word)
        if command == "doctor":
            return self._complete_doctor(args, current_index, current_word)
        if command == "config":
            return self._complete_config(args, current_index, current_word)
        if command == "help":
            return self._complete_commands()
        if command == "completions":
            return self._complete_completions()
        return []

    def _complete_render(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
        prev = args[current_index - 1] if current_index > 0 else ""
        if current_index == 1 and (not current_word or not current_word.startswith("-")):
            return [Completion("__PATH__")]
        if prev in {"--out"}:
            return [Completion("__PATH__")]
        generic = self._complete_click_option_values("render", prev)
        if generic:
            return generic
        if current_word.startswith("--"):
            return self._option_completions("render")
        return []

    def _complete_commands(self) -> List[Completion]:
        return [Completion(value=name, description=desc) for name, desc in click_command_entries(self.root)]

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
        generic = self._complete_click_option_values("sync", prev)
        if generic:
            return generic
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
        generic = self._complete_click_option_values("search", prev)
        if generic:
            return generic
        if current_word.startswith("--"):
            return self._option_completions("search")
        return []

    def _complete_import(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
        if current_index == 1 and (not current_word or not current_word.startswith("-")):
            return self._complete_group_subcommands("import")

        subcmd = args[1] if len(args) > 1 else None
        prev = args[current_index - 1] if current_index > 0 else ""

        if subcmd == "run":
            if current_index == 2 and (not current_word or not current_word.startswith("-")):
                return [Completion(name) for name in ("chatgpt", "claude", "codex", "claude-code")]
            if prev in {"--out", "--base-dir"}:
                return [Completion("__PATH__")]
            generic = self._complete_click_option_values("import run", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("import run")
            if current_index >= 3 and not current_word.startswith("-"):
                return [Completion("__PATH__")]

        if subcmd == "reprocess":
            generic = self._complete_click_option_values("import reprocess", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("import reprocess")
            return []

        if subcmd:
            generic = self._complete_click_option_values(f"import {subcmd}", prev)
            if generic:
                return generic
        return []

    def _complete_verify(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
        if current_index == 1 and (not current_word or not current_word.startswith("-")):
            return self._complete_group_subcommands("verify")
        subcmd = args[1] if len(args) > 1 else None
        prev = args[current_index - 1] if current_index > 0 else ""
        if subcmd == "check":
            if prev == "--provider":
                return self._provider_completions()
            if prev == "--slug":
                return self._slug_completions()
            if prev == "--conversation-id":
                return self._conversation_id_completions()
            generic = self._complete_click_option_values("verify check", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("verify check")
        if subcmd == "compare":
            if prev in {"--provider-a", "--provider-b"}:
                return self._provider_completions()
            generic = self._complete_click_option_values("verify compare", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("verify compare")
        if subcmd:
            generic = self._complete_click_option_values(f"verify {subcmd}", prev)
            if generic:
                return generic
        return []

    def _complete_browse(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
        if current_index == 1 and (not current_word or not current_word.startswith("-")):
            return self._complete_group_subcommands("browse")
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
            generic = self._complete_click_option_values("browse branches", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("browse branches")
        if subcmd == "stats":
            if prev == "--dir":
                return [Completion("__PATH__")]
            generic = self._complete_click_option_values("browse stats", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("browse stats")
        if subcmd == "runs":
            if prev == "--providers":
                return self._provider_completions()
            generic = self._complete_click_option_values("browse runs", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("browse runs")
        if subcmd == "metrics":
            if prev == "--providers":
                return self._provider_completions()
            generic = self._complete_click_option_values("browse metrics", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("browse metrics")
        if subcmd == "timeline":
            if prev == "--providers":
                return self._provider_completions()
            if prev == "--out":
                return [Completion("__PATH__")]
            if prev == "--theme":
                return [Completion("light"), Completion("dark")]
            generic = self._complete_click_option_values("browse timeline", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("browse timeline")
        if subcmd == "analytics":
            if prev == "--providers":
                return self._provider_completions()
            if prev == "--out":
                return [Completion("__PATH__")]
            if prev == "--theme":
                return [Completion("light"), Completion("dark")]
            generic = self._complete_click_option_values("browse analytics", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("browse analytics")
        if subcmd == "inbox":
            if prev == "--providers":
                return self._provider_completions()
            if prev in {"--dir", "--quarantine-dir"}:
                return [Completion("__PATH__")]
            generic = self._complete_click_option_values("browse inbox", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("browse inbox")
        if subcmd:
            generic = self._complete_click_option_values(f"browse {subcmd}", prev)
            if generic:
                return generic
        return []

    def _complete_doctor(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
        if current_index == 1 and (not current_word or not current_word.startswith("-")):
            return self._complete_group_subcommands("doctor")

        subcmd = args[1] if len(args) > 1 else None
        prev = args[current_index - 1] if current_index > 0 else ""

        if subcmd == "check":
            if prev in {"--codex-dir", "--claude-code-dir"}:
                return [Completion("__PATH__")]
            generic = self._complete_click_option_values("doctor check", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("doctor check")

        if subcmd == "env":
            generic = self._complete_click_option_values("doctor env", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("doctor env")

        if subcmd == "status":
            if prev == "--providers":
                return self._provider_completions()
            if prev in {"--dump", "--summary"}:
                return [Completion("__PATH__")]
            generic = self._complete_click_option_values("doctor status", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("doctor status")

        if subcmd == "prune":
            if prev == "--dir":
                return [Completion("__PATH__")]
            generic = self._complete_click_option_values("doctor prune", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("doctor prune")

        if subcmd == "index":
            if current_index == 2 and (not current_word or not current_word.startswith("-")):
                return self._complete_group_subcommands("doctor index")
            generic = self._complete_click_option_values("doctor index", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("doctor index")

        if subcmd == "restore":
            if prev in {"--from", "--to"}:
                return [Completion("__PATH__")]
            generic = self._complete_click_option_values("doctor restore", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("doctor restore")

        if subcmd == "attachments":
            if current_index == 2 and (not current_word or not current_word.startswith("-")):
                return self._complete_group_subcommands("doctor attachments")
            attachments_cmd = args[2] if len(args) > 2 else None
            if attachments_cmd == "stats":
                if prev == "--provider":
                    return self._provider_completions()
                if prev in {"--dir", "--csv"}:
                    return [Completion("__PATH__")]
                if prev == "--sort":
                    return [Completion("size"), Completion("name")]
                generic = self._complete_click_option_values("doctor attachments stats", prev)
                if generic:
                    return generic
                if current_word.startswith("--"):
                    return self._option_completions("doctor attachments stats")
            if attachments_cmd == "extract":
                if prev in {"--dir", "--out"}:
                    return [Completion("__PATH__")]
                generic = self._complete_click_option_values("doctor attachments extract", prev)
                if generic:
                    return generic
                if current_word.startswith("--"):
                    return self._option_completions("doctor attachments extract")

        if subcmd:
            generic = self._complete_click_option_values(f"doctor {subcmd}", prev)
            if generic:
                return generic
        return []

    def _complete_config(self, args: Sequence[str], current_index: int, current_word: str) -> List[Completion]:
        if current_index == 1 and (not current_word or not current_word.startswith("-")):
            return self._complete_group_subcommands("config")
        subcmd = args[1] if len(args) > 1 else None
        prev = args[current_index - 1] if current_index > 0 else ""
        if subcmd == "init" and current_word.startswith("--"):
            return self._option_completions("config init")
        if subcmd == "set":
            if prev == "--theme":
                return [Completion("light"), Completion("dark")]
            if prev == "--html":
                return [Completion("on"), Completion("off")]
            generic = self._complete_click_option_values("config set", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("config set")
        if subcmd == "show":
            generic = self._complete_click_option_values("config show", prev)
            if generic:
                return generic
            if current_word.startswith("--"):
                return self._option_completions("config show")
        if subcmd == "edit" and current_word.startswith("--"):
            return self._option_completions("config edit")
        if subcmd == "prefs":
            if current_index == 2 and (not current_word or not current_word.startswith("-")):
                return self._complete_group_subcommands("config prefs")
            prefs_cmd = args[2] if len(args) > 2 else None
            if prefs_cmd == "set" and prev == "--command":
                return [Completion(value) for value in ("search", "sync", "import", "render", "browse", "verify", "doctor", "config")]
            if prefs_cmd:
                generic = self._complete_click_option_values(f"config prefs {prefs_cmd}", prev)
                if generic:
                    return generic
                if current_word.startswith("--"):
                    return self._option_completions(f"config prefs {prefs_cmd}")
        if subcmd:
            generic = self._complete_click_option_values(f"config {subcmd}", prev)
            if generic:
                return generic
        return []

    def _complete_completions(self) -> List[Completion]:
        return [Completion("--shell", "Target shell (bash/zsh/fish)")]

    def _complete_group_subcommands(self, command: str) -> List[Completion]:
        click_command = self._resolve_command_for_path(command)
        if not isinstance(click_command, click.Group):
            return []
        return [Completion(value=name, description=desc) for name, desc in click_command_entries(click_command)]

    def _complete_click_option_values(self, command: str, option_name: str) -> List[Completion]:
        if not option_name.startswith("--"):
            return []
        click_command = self._resolve_command_for_path(command)
        if click_command is None:
            return []
        option = None
        for param in click_command.params:
            if not isinstance(param, click.Option):
                continue
            if option_name in param.opts or option_name in list(getattr(param, "secondary_opts", [])):
                option = param
                break
        if option is None:
            return []
        if getattr(option, "is_flag", False):
            return []
        if isinstance(option.type, click.Choice):
            return [Completion(str(value)) for value in option.type.choices]
        if isinstance(option.type, click.Path):
            return [Completion("__PATH__")]
        return []

    def _option_completions(self, command: str) -> List[Completion]:
        click_command = self._resolve_command_for_path(command)
        if click_command is None:
            return []
        pairs: List[Completion] = []
        for param in click_command.params:
            if not isinstance(param, click.Option):
                continue
            help_text = self._format_option_help(param)
            for opt in list(param.opts) + list(getattr(param, "secondary_opts", [])):
                pairs.append(Completion(opt, help_text))
        return pairs

    def _format_option_help(self, option: click.Option) -> str:
        help_text = (option.help or "").replace("\n", " ").strip()
        default = option.default
        show_default = option.show_default
        if show_default is None:
            show_default = default not in {None, False, ""}
        if not show_default:
            return help_text
        if isinstance(default, bool):
            default_text = "on" if default else "off"
        elif isinstance(default, (list, tuple, set)):
            default_text = ", ".join(str(value) for value in default) or "[]"
        else:
            default_text = str(default)
        if help_text:
            return f"{help_text} (default: {default_text})"
        return f"(default: {default_text})"

    def _resolve_command_for_path(self, command: str) -> click.Command | None:
        parts = command.split()
        if not parts:
            return None
        ctx = click.Context(self.root)
        cmd: click.Command = self.root
        for part in parts:
            if not isinstance(cmd, click.MultiCommand):
                return cmd
            next_cmd = cmd.get_command(ctx, part)
            if next_cmd is None:
                return None
            cmd = next_cmd
            ctx = click.Context(cmd, info_name=part, parent=ctx)
        return cmd

    def _provider_completions(self) -> List[Completion]:
        providers = self._cached_state_list(
            "providers",
            lambda: sorted({provider for provider, *_ in self.env.conversations.iter_state()}),
        )
        return [Completion(p) for p in providers if p]

    def _slug_completions(self) -> List[Completion]:
        slugs = self._cached_state_list(
            "slugs",
            lambda: sorted(
                {payload.get("slug") for _, _, payload in self.env.conversations.iter_state() if payload.get("slug")}
            ),
        )
        return [Completion(slug) for slug in slugs if isinstance(slug, str) and slug]

    def _branch_completions(self) -> List[Completion]:
        branch_ids = self._cached_state_list(
            "branches",
            lambda: sorted(
                {
                    branch_id
                    for _, _, payload in self.env.conversations.iter_state()
                    for branch_id in (payload.get("branches", {}) or {}).keys()
                    if branch_id
                }
            ),
        )
        return [Completion(branch_id) for branch_id in branch_ids if isinstance(branch_id, str) and branch_id]

    def _model_completions(self) -> List[Completion]:
        models = self._cached_state_list(
            "models",
            lambda: sorted(
                {
                    (payload.get("model") or payload.get("metadata", {}).get("model"))
                    for _, _, payload in self.env.conversations.iter_state()
                    if (payload.get("model") or payload.get("metadata", {}).get("model"))
                }
            ),
        )
        return [Completion(model) for model in models if isinstance(model, str) and model]

    def _conversation_id_completions(self) -> List[Completion]:
        items = self._cached_state_list(
            "conversation_ids",
            lambda: sorted(
                [
                    {"value": conversation_id, "description": f"{provider}:{payload.get('slug','')}"}
                    for provider, conversation_id, payload in self.env.conversations.iter_state()
                    if conversation_id
                ],
                key=lambda item: (item.get("value") or ""),
            ),
        )
        entries: List[Completion] = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            if not isinstance(value, str) or not value or value in seen:
                continue
            desc = item.get("description") if isinstance(item.get("description"), str) else ""
            entries.append(Completion(value, desc))
            seen.add(value)
        return entries

    def _drive_chat_ids(self) -> List[Completion]:
        items = self._cached_state_list(
            "drive_chat_ids",
            lambda: sorted(
                [
                    {
                        "value": (payload.get("extra_state") or payload.get("extraState") or {}).get("driveFileId"),
                        "description": payload.get("slug") or "drive chat",
                    }
                    for provider, _, payload in self.env.conversations.iter_state()
                    if provider == "drive"
                ],
                key=lambda item: (item.get("value") or ""),
            ),
        )
        entries: List[Completion] = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            if not isinstance(value, str) or not value or value in seen:
                continue
            desc = item.get("description") if isinstance(item.get("description"), str) else ""
            entries.append(Completion(value, desc))
            seen.add(value)
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
        sessions = self._cached_sessions_list(provider, base, provider_obj.list_sessions)
        return [Completion(str(path)) for path in sessions]

    def _cached_sessions_list(self, provider: str, base_dir: Path, lister) -> List[Path]:
        ttl_seconds = 15.0
        cache_key = f"{provider}|{str(base_dir)}"
        cache = self._cache.load()
        session_cache = cache.get("sessions") if isinstance(cache.get("sessions"), dict) else {}
        if isinstance(session_cache, dict):
            entry = session_cache.get(cache_key)
            if isinstance(entry, dict):
                ts = entry.get("ts")
                paths = entry.get("paths")
                if isinstance(ts, (int, float)) and isinstance(paths, list) and (time.time() - float(ts)) <= ttl_seconds:
                    return [Path(p) for p in paths if isinstance(p, str)]
        sessions = lister(base_dir)
        if not isinstance(session_cache, dict):
            session_cache = {}
        session_cache[cache_key] = {"ts": time.time(), "paths": [str(p) for p in sessions]}
        cache["sessions"] = session_cache
        self._cache.save(cache)
        return sessions

    def _cached_state_list(self, key: str, producer) -> list:
        cache = self._cache.load()
        signature = self._state_db_signature()
        cached_sig = cache.get("state_db")
        if not isinstance(cached_sig, dict) or cached_sig != signature:
            cache = {"version": cache.get("version", _CompletionCache.version), "state_db": signature}
        state_cache = cache.get("state") if isinstance(cache.get("state"), dict) else {}
        if isinstance(state_cache, dict) and key in state_cache:
            values = state_cache.get(key)
            if isinstance(values, list):
                return values
        values = producer()
        if not isinstance(values, list):
            values = list(values) if values is not None else []
        if not isinstance(state_cache, dict):
            state_cache = {}
        state_cache[key] = values
        cache["state"] = state_cache
        self._cache.save(cache)
        return values

    def _state_db_signature(self) -> dict:
        try:
            db_path = self.env.state_repo.database.resolve_path()
            stat = db_path.stat()
        except Exception:
            return {"path": "", "mtime_ns": 0, "size": 0}
        return {"path": str(db_path), "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))), "size": int(stat.st_size)}

    @staticmethod
    def _get_option_value(args: Sequence[str], option: str) -> str | None:
        for idx, word in enumerate(args):
            if word == option and idx + 1 < len(args):
                return args[idx + 1]
            if word.startswith(option + "="):
                return word.split("=", 1)[1]
        return None
