"""Route archived local sessions back to their owning interactive harness."""

from __future__ import annotations

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.resume_routing import route_resume
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.types import SessionId


def _session(origin: Origin, native_id: str = "native-session") -> Session:
    return Session(
        id=SessionId(f"{origin}:{native_id}"),
        origin=origin,
        title="resume fixture",
        messages=MessageCollection(messages=[]),
        working_directories=("/workspace/polylogue",),
    )


def test_routes_claude_code_to_interactive_resume_command() -> None:
    route = route_resume(_session(Origin.CLAUDE_CODE_SESSION, "claude-native"))

    assert route.status == "supported"
    assert route.argv == ("claude", "--resume", "claude-native")
    assert route.command == "cd /workspace/polylogue && claude --resume claude-native"


def test_routes_codex_to_interactive_resume_command_not_headless_exec() -> None:
    route = route_resume(_session(Origin.CODEX_SESSION, "codex-native"))

    assert route.status == "supported"
    assert route.argv == ("codex", "resume", "codex-native")
    assert "exec" not in route.argv


def test_refuses_to_guess_for_unsupported_origin() -> None:
    route = route_resume(_session(Origin.CHATGPT_EXPORT))

    assert route.status == "unsupported"
    assert route.command is None
    assert route.detail is not None
