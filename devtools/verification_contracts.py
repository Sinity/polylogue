"""Typed scope fields shared by verification and merge receipts."""

from __future__ import annotations

from enum import StrEnum


class VerificationScope(StrEnum):
    AFFECTED = "affected"
    NON_TEST = "non-test"


__all__ = ["VerificationScope"]
