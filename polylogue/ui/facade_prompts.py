"""Prompt stub loading and plain-input helper functions."""

from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from polylogue.ui.facade import UIError

_PROMPT_STUB_SCALARS = (bool, int, float, str)


@dataclass(frozen=True)
class PromptStubEntry:
    """Normalized prompt-stub entry loaded from a JSONL file."""

    kind: str | None = None
    use_default: bool = False
    value: bool | int | float | str | None = None
    has_value: bool = False
    index: int | None = None

    @classmethod
    def from_json(cls, raw: object) -> PromptStubEntry | None:
        if not isinstance(raw, dict):
            return None
        kind = raw.get("type")
        entry_kind = kind if isinstance(kind, str) else None
        use_default = bool(raw.get("use_default", False))
        has_value = "value" in raw
        raw_value = raw.get("value")
        value: bool | int | float | str | None = (
            raw_value if raw_value is None or isinstance(raw_value, _PROMPT_STUB_SCALARS) else str(raw_value)
        )
        return cls(
            kind=entry_kind,
            use_default=use_default,
            value=value,
            has_value=has_value,
            index=_coerce_index(raw.get("index")),
        )


class _NoStubResponse:
    __slots__ = ()


_NO_STUB_RESPONSE: Final = _NoStubResponse()


def _coerce_index(raw: object) -> int | None:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def load_prompt_responses(
    ui_error_cls: type[UIError],
    *,
    prompt_stub_path: Path | None = None,
) -> deque[PromptStubEntry]:
    if prompt_stub_path is None:
        return deque()
    entries: deque[PromptStubEntry] = deque()
    for line in prompt_stub_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            entry = PromptStubEntry.from_json(data)
            if entry is not None:
                entries.append(entry)
        except json.JSONDecodeError as exc:
            raise ui_error_cls(f"Invalid prompt stub entry: {line}") from exc
    return entries


def pop_prompt_response(
    prompt_responses: deque[PromptStubEntry],
    kind: str,
    ui_error_cls: type[UIError],
) -> PromptStubEntry | None:
    if not prompt_responses:
        return None
    entry = prompt_responses.popleft()
    expected = entry.kind
    if expected and expected != kind:
        raise ui_error_cls(f"Prompt stub expected '{expected}' but got '{kind}'")
    return entry


def require_plain_prompt_tty(prompt_topic: str, ui_error_cls: type[UIError]) -> None:
    if not sys.stdin.isatty():
        raise ui_error_cls(
            f"Plain mode cannot prompt for {prompt_topic}",
            prompt_topic=prompt_topic,
        )


def consume_confirm_stub(
    prompt_responses: deque[PromptStubEntry],
    *,
    default: bool,
    ui_error_cls: type[UIError],
) -> bool | _NoStubResponse:
    response = pop_prompt_response(prompt_responses, "confirm", ui_error_cls)
    if response is None:
        return _NO_STUB_RESPONSE
    if response.use_default:
        return default
    value = response.value
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"y", "yes", "true", "1"}:
            return True
        if lowered in {"n", "no", "false", "0"}:
            return False
    return _NO_STUB_RESPONSE


def consume_choose_stub(
    prompt_responses: deque[PromptStubEntry],
    options: list[str],
    ui_error_cls: type[UIError],
) -> str | None | _NoStubResponse:
    response = pop_prompt_response(prompt_responses, "choose", ui_error_cls)
    if response is None:
        return _NO_STUB_RESPONSE
    if response.use_default:
        return options[0] if options else None
    value = response.value
    if isinstance(value, str) and value in options:
        return value
    if response.index is not None and 0 <= response.index < len(options):
        return options[response.index]
    return _NO_STUB_RESPONSE


def consume_input_stub(
    prompt_responses: deque[PromptStubEntry],
    *,
    default: str | None,
    ui_error_cls: type[UIError],
) -> str | None | _NoStubResponse:
    response = pop_prompt_response(prompt_responses, "input", ui_error_cls)
    if response is None:
        return _NO_STUB_RESPONSE
    if response.use_default:
        return default
    if response.has_value:
        value = response.value
        return None if value is None else str(value)
    return _NO_STUB_RESPONSE


__all__ = [
    "PromptStubEntry",
    "_NO_STUB_RESPONSE",
    "consume_choose_stub",
    "consume_confirm_stub",
    "consume_input_stub",
    "load_prompt_responses",
    "pop_prompt_response",
    "require_plain_prompt_tty",
]
