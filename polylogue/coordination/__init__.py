"""Agent coordination envelope over existing local evidence."""

from polylogue.coordination.envelope import build_coordination_envelope, project_coordination_envelope
from polylogue.coordination.payloads import AgentCoordinationPayload, CoordinationView

__all__ = [
    "AgentCoordinationPayload",
    "CoordinationView",
    "build_coordination_envelope",
    "project_coordination_envelope",
]
