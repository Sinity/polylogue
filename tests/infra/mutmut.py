"""Helpers for running tests under mutmut instrumentation."""

from __future__ import annotations

import os

_MUTMUT_ENV_KEYS = ("MUTANT_UNDER_TEST", "PY_IGNORE_IMPORTMISMATCH")


def preserved_mutmut_env() -> dict[str, str]:
    """Return the mutmut runtime markers that must survive env clearing in tests."""
    return {
        key: value
        for key in _MUTMUT_ENV_KEYS
        if (value := os.environ.get(key)) is not None
    }
