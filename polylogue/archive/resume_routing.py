"""Safe, explicit harness-resume commands for archived local sessions.

Only origins with a documented native resume contract receive a command.  The
route is deliberately a value object: callers can print it for a human or
execute it behind an explicit opt-in without guessing for unsupported exports.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Literal, Protocol

from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.core.web_urls import native_id_from_session_id

ResumeRouteStatus = Literal["supported", "unsupported"]
OpenSessionState = Literal["open", "closed", "unknown"]


class SessionOpenDetector(Protocol):
    """Optional integration for detecting an already-open harness session."""

    def is_open(self, session: Session) -> bool:
        """Return whether the harness currently has ``session`` open."""


@dataclass(frozen=True, slots=True)
class ResumeRoute:
    """One resolved resume route, safe to render or execute verbatim."""

    status: ResumeRouteStatus
    origin: str
    native_session_id: str
    argv: tuple[str, ...] = ()
    cwd: str | None = None
    detail: str | None = None
    open_state: OpenSessionState = "unknown"

    @property
    def command(self) -> str | None:
        """Shell-safe interactive command, prefixed with its known cwd."""

        if not self.argv:
            return None
        command = shlex.join(self.argv)
        return f"cd {shlex.quote(self.cwd)} && {command}" if self.cwd else command


def route_resume(session: Session, *, open_detector: SessionOpenDetector | None = None) -> ResumeRoute:
    """Map a local harness session to its interactive resume command.

    ``codex exec resume`` is intentionally not selected here: ``continue`` is
    human-facing and must reopen the interactive TUI.  It remains the correct
    headless alternative for automation.
    """

    origin = str(session.origin)
    native_id = native_id_from_session_id(session.id)
    cwd = next((path for path in session.working_directories if path), None)
    open_state: OpenSessionState = "unknown"
    if open_detector is not None:
        try:
            open_state = "open" if open_detector.is_open(session) else "closed"
        except Exception:
            # Desktop control planes are optional; a failed probe must not
            # prevent printing or executing a verified resume command.
            open_state = "unknown"
    if native_id is None:
        return ResumeRoute(
            status="unsupported",
            origin=origin,
            native_session_id="",
            cwd=cwd,
            detail="The archive session id has no native harness session id.",
            open_state=open_state,
        )
    commands: dict[Origin, tuple[str, ...]] = {
        Origin.CLAUDE_CODE_SESSION: ("claude", "--resume", native_id),
        Origin.CODEX_SESSION: ("codex", "resume", native_id),
    }
    argv = commands.get(session.origin)
    if argv is None:
        return ResumeRoute(
            status="unsupported",
            origin=origin,
            native_session_id=native_id,
            cwd=cwd,
            detail=f"No verified interactive resume command for origin {origin!r}.",
            open_state=open_state,
        )
    return ResumeRoute(
        status="supported",
        origin=origin,
        native_session_id=native_id,
        argv=argv,
        cwd=cwd,
        open_state=open_state,
    )


__all__ = ["OpenSessionState", "ResumeRoute", "ResumeRouteStatus", "SessionOpenDetector", "route_resume"]
