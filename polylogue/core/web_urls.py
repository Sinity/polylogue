"""Canonical web-URL projection for web-originated sessions.

A session's public web URL is *derivable structure*, not a stored fact: the
provider conversation id is preserved as the session's ``native_id`` (the part
of ``session_id`` after the ``origin:`` prefix), and each web provider has a
stable URL template. Reconstructing the URL on read avoids storing a
redundant column and keeps the URL correct even for historical rows.

For ChatGPT, a conversation may belong to a *project* (the ``g-p-<id>``
segment, surfaced in the backend payload as ``gizmo_id`` /
``conversation_template_id``). The project segment is optional in the URL —
``chatgpt.com/c/<id>`` redirects to the project-scoped URL — so the bare form
is always valid and the project form is produced only when the ref is known.

Origins that are local agent runtimes (Claude Code, Codex, Gemini CLI, …) have
no public web URL and return ``None``.
"""

from __future__ import annotations

from polylogue.core.enums import Origin

__all__ = ["canonical_session_url", "native_id_from_session_id"]


def native_id_from_session_id(session_id: str) -> str | None:
    """Extract the provider-native id from an ``origin:native_id`` session id."""
    head, sep, tail = session_id.partition(":")
    if not sep or not tail:
        return None
    return tail


def canonical_session_url(
    origin: Origin | str,
    native_id: str | None,
    project_ref: str | None = None,
) -> str | None:
    """Reconstruct the public web URL for a web-originated session.

    Returns ``None`` for local/agent origins that have no web surface, or when
    ``native_id`` is missing.

    ``project_ref`` is the ChatGPT project/workspace token (``g-p-<id>``); when
    supplied for a ChatGPT session the project-scoped URL is produced. It is
    ignored for other origins.
    """
    if not native_id:
        return None
    origin_value = origin.value if isinstance(origin, Origin) else str(origin)

    if origin_value == Origin.CHATGPT_EXPORT.value:
        if project_ref and project_ref.startswith("g-p-"):
            return f"https://chatgpt.com/g/{project_ref}/c/{native_id}"
        return f"https://chatgpt.com/c/{native_id}"
    if origin_value == Origin.CLAUDE_AI_EXPORT.value:
        return f"https://claude.ai/chat/{native_id}"
    # Grok web URL shape is not confirmed; do not fabricate one.
    return None
