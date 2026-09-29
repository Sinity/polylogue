"""Enforce managed pytest admission before, and independently of, conftest loading."""

from __future__ import annotations

import os

import pytest

from devtools.agent_env import refuse_bare_pytest


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests() -> None:
    refusal = refuse_bare_pytest(os.environ)
    if refusal is not None:
        raise pytest.UsageError(refusal)
