"""Common semantic types that all providers map to.

This module defines the canonical types that represent the "common subset"
across all providers.

Types:
    - Role: Canonical message roles (user, assistant, system, tool, unknown)
    - CommonMessage: Basic message structure with role, text, timestamp
    - CommonToolCall: Tool invocation with name, input, output

For extraction logic, see polylogue.schemas.extractors (glom-based declarative)
or polylogue.schemas.unified (HarmonizedMessage with viewports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# Re-export Role from the canonical location
from polylogue.core.roles import Role


@dataclass
class CommonMessage:
    """The common subset all providers can provide."""

    role: Role
    text: str
    timestamp: datetime | None = None

    # Optional enrichments (not all providers have these)
    id: str | None = None
    model: str | None = None
    tokens: int | None = None
    cost_usd: float | None = None
    is_thinking: bool = False

    # Preserve original for debugging
    provider: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class CommonToolCall:
    """Tool invocation common subset."""

    name: str
    input: dict
    output: str | None = None
    success: bool | None = None

    provider: str = ""
    raw: dict = field(default_factory=dict)
