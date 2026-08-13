"""Typed scope fields shared by verification and merge receipts."""

from __future__ import annotations

from enum import StrEnum


class VerificationScope(StrEnum):
    AFFECTED = "affected"
    RELEASE_BASELINE = "release-baseline"
    NON_TEST = "non-test"


__all__ = ["VerificationScope"]
