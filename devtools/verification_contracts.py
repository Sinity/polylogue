"""Typed scope fields shared by verification and merge receipts."""

from __future__ import annotations

from enum import StrEnum


class VerificationScope(StrEnum):
    AFFECTED = "affected"
    RELEASE_BASELINE = "release-baseline"
    NARROW_TERMINAL = "narrow-terminal"
    NON_TEST = "non-test"


class TerminalAuthorization(StrEnum):
    NARROW_TERMINAL = "narrow-terminal"


__all__ = ["TerminalAuthorization", "VerificationScope"]
