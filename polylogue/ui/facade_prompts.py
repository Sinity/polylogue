"""Prompt stub loading and plain-input helper functions."""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

_NO_STUB_RESPONSE = object()


def load_prompt_responses(
    ui_error_cls: type[Exception],
    *,
    prompt_stub_path: Path | None = None,
) -> deque[dict[str, object]]:
    if prompt_stub_path is None:
        return deque()
    entries: deque[dict[str, object]] = deque()
    for line in prompt_stub_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                entries.append(data)
        except json.JSONDecodeError as exc:
            raise ui_error_cls(f"Invalid prompt stub entry: {line}") from exc
    return entries


def pop_prompt_response(
    prompt_responses: deque[dict[str, object]],
    kind: str,
    ui_error_cls: type[Exception],
) -> dict[str, object] | None:
    if not prompt_responses:
        return None
    entry = prompt_responses.popleft()
    expected = entry.get("type")
    if expected and expected != kind:
        raise ui_error_cls(f"Prompt stub expected '{expected}' but got '{kind}'")
    return entry


def require_plain_prompt_tty(prompt_topic: str, ui_error_cls: type[Exception]) -> None:
    if not sys.stdin.isatty():
        raise ui_error_cls(
            f"Plain mode cannot prompt for {prompt_topic}",
            prompt_topic=prompt_topic,
        )


def consume_confirm_stub(
    prompt_responses: deque[dict[str, object]],
    *,
    default: bool,
    ui_error_cls: type[Exception],
) -> bool | object:
    response = pop_prompt_response(prompt_responses, "confirm", ui_error_cls)
    if response is None:
        return _NO_STUB_RESPONSE
    if response.get("use_default"):
        return default
    value = response.get("value")
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
    prompt_responses: deque[dict[str, object]],
    options: list[str],
    ui_error_cls: type[Exception],
) -> str | None | object:
    response = pop_prompt_response(prompt_responses, "choose", ui_error_cls)
    if response is None:
        return _NO_STUB_RESPONSE
    if response.get("use_default"):
        return options[0] if options else None
    if "value" in response:
        value = response["value"]
        if isinstance(value, str) and value in options:
            return value
    if "index" in response:
        try:
            index_val = response["index"]
            idx: int | None = None
            if isinstance(index_val, int):
                idx = index_val
            elif isinstance(index_val, str):
                idx = int(index_val)
            if idx is not None and 0 <= idx < len(options):
                return options[idx]
        except (KeyError, ValueError, TypeError):
            pass
    return _NO_STUB_RESPONSE


def consume_input_stub(
    prompt_responses: deque[dict[str, object]],
    *,
    default: str | None,
    ui_error_cls: type[Exception],
) -> str | None | object:
    response = pop_prompt_response(prompt_responses, "input", ui_error_cls)
    if response is None:
        return _NO_STUB_RESPONSE
    if response.get("use_default"):
        return default
    if "value" in response:
        value = response["value"]
        return None if value is None else str(value)
    return _NO_STUB_RESPONSE
