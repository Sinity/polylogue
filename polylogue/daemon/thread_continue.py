"""Thread-continue deep-link generator for the reader (#1203).

The reader exposes a "continue this thread in <agent>" affordance per
message. Each affordance is a URL constructed from a per-agent template
plus an encoded prompt seed (the message text and, optionally, recent
ancestor context).

This module owns:

- the built-in template registry shipped with polylogue;
- the URL constructor that fills templates with safe, percent-encoded
  substitutions;
- the JSON envelope shipped by
  ``GET /api/thread-continue-templates`` so the reader can render the
  affordance without baking templates into the HTML shell.

The templates are deliberately stateless: an operator who wants to add
or override an agent (a local terminal wrapper, a custom URL scheme)
configures it via the ``POLYLOGUE_READER_AGENT_TEMPLATES`` environment
variable. The default registry covers the two coding agents polylogue
already ingests from — Claude Code and Codex — plus a generic
``copy-as-prompt`` clipboard fallback that does not require any
external integration.

Variables available inside a template:

- ``{prompt}`` — URL-encoded message text (plus optional context).
- ``{prompt_plain}`` — raw message text (no encoding); use only when the
  template's URL scheme handles encoding itself (rare).
- ``{session_id}`` — the source session id.
- ``{message_id}`` — the source message id, or empty when not provided.

The endpoint never returns the prompt contents — it only returns the
templates. The reader substitutes locally so the daemon never sees a
mirror of every message a user "continues" in another agent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Final
from urllib.parse import quote

#: Environment override.  Value must be a JSON array of objects with the
#: fields ``agent_id``, ``label``, ``url_template``.  An ``enabled`` key
#: may be set to ``false`` to drop a built-in entry.
TEMPLATE_ENV_VAR: Final[str] = "POLYLOGUE_READER_AGENT_TEMPLATES"


@dataclass(frozen=True)
class AgentTemplate:
    """One agent-specific deep-link template."""

    agent_id: str
    label: str
    url_template: str

    def to_dict(self) -> dict[str, str]:
        return {
            "agent_id": self.agent_id,
            "label": self.label,
            "url_template": self.url_template,
        }


#: Built-in templates. Each one is a coding-agent integration the
#: polylogue ecosystem already knows about. ``copy-as-prompt`` is a
#: clipboard fallback the reader applies when no URL handler is
#: installed; the JS side recognises the ``polylogue:copy-prompt``
#: scheme and routes to ``navigator.clipboard`` instead of opening a
#: link.
_BUILTIN_TEMPLATES: tuple[AgentTemplate, ...] = (
    AgentTemplate(
        agent_id="claude-code",
        label="Open in Claude Code",
        url_template="polylogue://thread?agent=claude-code&conv={session_id}&msg={message_id}&prompt={prompt}",
    ),
    AgentTemplate(
        agent_id="codex",
        label="Open in Codex",
        url_template="polylogue://thread?agent=codex&conv={session_id}&msg={message_id}&prompt={prompt}",
    ),
    AgentTemplate(
        agent_id="copy-prompt",
        label="Copy as prompt",
        url_template="polylogue:copy-prompt?prompt={prompt}",
    ),
)


def _parse_override(raw: str) -> tuple[AgentTemplate, ...]:
    """Parse ``POLYLOGUE_READER_AGENT_TEMPLATES`` payload.

    Returns the merged template list (built-ins first, overrides applied
    by ``agent_id``). Invalid payloads silently fall back to the
    built-ins — the reader must never crash on a malformed env var.
    """

    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return _BUILTIN_TEMPLATES
    if not isinstance(data, list):
        return _BUILTIN_TEMPLATES
    overrides: dict[str, AgentTemplate | None] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        agent_id = entry.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            continue
        enabled = entry.get("enabled", True)
        if enabled is False:
            overrides[agent_id] = None
            continue
        label = entry.get("label")
        template = entry.get("url_template")
        if not isinstance(label, str) or not isinstance(template, str):
            continue
        overrides[agent_id] = AgentTemplate(agent_id=agent_id, label=label, url_template=template)
    merged: list[AgentTemplate] = []
    seen: set[str] = set()
    for builtin in _BUILTIN_TEMPLATES:
        if builtin.agent_id in overrides:
            replacement = overrides[builtin.agent_id]
            if replacement is None:
                continue
            merged.append(replacement)
        else:
            merged.append(builtin)
        seen.add(builtin.agent_id)
    for agent_id, template in overrides.items():
        if agent_id in seen or template is None:
            continue
        merged.append(template)
    return tuple(merged)


def list_templates(env: dict[str, str] | None = None) -> tuple[AgentTemplate, ...]:
    """Return the active template registry.

    ``env`` is an optional mapping (defaults to :data:`os.environ`) so
    tests can pin the registry deterministically without mutating the
    process environment.
    """

    if env is None:
        env = dict(os.environ)
    raw = env.get(TEMPLATE_ENV_VAR)
    if not raw:
        return _BUILTIN_TEMPLATES
    return _parse_override(raw)


def build_url(
    template: AgentTemplate,
    *,
    prompt: str,
    session_id: str,
    message_id: str | None = None,
) -> str:
    """Substitute the template variables into ``template.url_template``.

    ``prompt`` is percent-encoded for the ``{prompt}`` placeholder and
    left raw for ``{prompt_plain}`` so an operator who configured a
    scheme handler that wants the raw text can opt in deliberately.
    ``session_id`` and ``message_id`` are URL-quoted.
    """

    encoded_prompt = quote(prompt, safe="")
    return template.url_template.format(
        prompt=encoded_prompt,
        prompt_plain=prompt,
        session_id=quote(session_id, safe=""),
        message_id=quote(message_id or "", safe=""),
    )


def build_templates_envelope(env: dict[str, str] | None = None) -> dict[str, object]:
    """Project the active registry into the public JSON envelope."""

    templates = list_templates(env)
    return {
        "templates": [t.to_dict() for t in templates],
        "count": len(templates),
        "env_var": TEMPLATE_ENV_VAR,
    }


__all__ = [
    "AgentTemplate",
    "TEMPLATE_ENV_VAR",
    "build_templates_envelope",
    "build_url",
    "list_templates",
]
