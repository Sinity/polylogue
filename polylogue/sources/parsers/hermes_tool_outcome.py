"""The Hermes tool-result outcome mapping, shared by both Hermes sources.

Hermes reaches Polylogue twice -- as a live state database (``hermes_state``)
and as a JSON export (``local_agent.parse_hermes``) -- carrying the same tool
content envelope. One mapping serves both so the two sources cannot drift into
different verdicts for the same record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from polylogue.sources.tool_result_reasons import unknown_reason

__all__ = ["JSON_ENVELOPE_PREFIX", "tool_result_outcome"]

#: Hermes prefixes a JSON-encoded content payload with this sentinel.
JSON_ENVELOPE_PREFIX = "\x00json:"


def tool_result_outcome(raw_content: object) -> tuple[bool | None, int | None, str | None]:
    """Extract (is_error, exit_code, unknown reason) from Hermes tool content.

    Hermes stores tool results as a JSON envelope (``{"output": ...}``) with
    one of ``exit_code`` (shell/command-style tools), ``success``
    (boolean-style tools, paired with an ``error`` message when false), or a
    bare ``error`` message (status-only tools) layered on top -- never all
    three. Absence of every signal means the source tool reported no outcome.

    A payload that announces the JSON envelope and does not decode is an
    outcome carrier the source declared but did not retain intact.
    """
    if _declares_undecoded_envelope(raw_content):
        return None, None, unknown_reason(is_error=None, source_intact=False)
    payload = _envelope_mapping(raw_content)
    raw_exit_code = payload.get("exit_code")
    exit_code = raw_exit_code if isinstance(raw_exit_code, int) and not isinstance(raw_exit_code, bool) else None
    if payload.get("error") is not None:
        return True, exit_code, None
    if "success" in payload:
        return not bool(payload["success"]), exit_code, None
    if exit_code is not None:
        return exit_code != 0, exit_code, None
    # ``exit_code``/``success`` present but off-type is a verdict this mapping
    # does not read, distinct from an envelope that states none.
    unread = "exit_code" in payload and exit_code is None
    return None, None, unknown_reason(is_error=None, outcome_field_present=unread)


def _declares_undecoded_envelope(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(JSON_ENVELOPE_PREFIX):
        return False
    try:
        json.loads(value[len(JSON_ENVELOPE_PREFIX) :])
    except json.JSONDecodeError:
        return True
    return False


def _envelope_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    if value.startswith(JSON_ENVELOPE_PREFIX):
        value = value[len(JSON_ENVELOPE_PREFIX) :]
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}
