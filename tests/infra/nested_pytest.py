"""Environment for a test that must spawn a real pytest.

Most harness behaviour can be tested by calling the function under test, and
should be: a nested pytest costs a subprocess and inherits state. Use this only
where a real process is genuinely the subject -- cgroup cleanup after SIGKILL, or
pytest's own lazy TempPathFactory clearing.

Where it IS genuine, the inherited environment is a trap. `devtools verify` sets
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 for its own child, which is right: the managed
command names its plugins explicitly. A nested pytest inherits that variable but
NOT the explicit plugin list, so it never loads pytest-benchmark while still
reading pyproject's addopts, and dies on arguments it cannot parse:

    error: unrecognized arguments: --benchmark-disable --benchmark-storage=...

The result is a family of tests that pass standalone and fail only inside
`devtools verify` -- the most expensive shape to diagnose, because the context
that breaks them is the one nobody runs while debugging.
"""

from __future__ import annotations

import os

__all__ = ["MANAGED_RUN_VARIABLES", "nested_pytest_env"]

#: Variables a managed run sets for ITS child, which a nested pytest must not
#: inherit. Each would otherwise make the child behave as half of the managed
#: run: adopting its plugin policy, its identity, or its scratch claim.
MANAGED_RUN_VARIABLES: tuple[str, ...] = (
    "PYTEST_ADDOPTS",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    "PYTEST_PLUGINS",
    "POLYLOGUE_VERIFY_RUN_ID",
    "POLYLOGUE_VERIFICATION_INVOCATION_ID",
    "POLYLOGUE_VERIFICATION_RECEIPT_PATH",
)


def nested_pytest_env(**overrides: str) -> dict[str, str]:
    """A copy of the environment with the managed run's own variables removed.

    Also drops every ``POLYLOGUE_PYTEST_*`` variable: those carry the outer run's
    identity and basetemp claim, and a nested run must establish its own.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in MANAGED_RUN_VARIABLES and not key.startswith("POLYLOGUE_PYTEST_")
    }
    env.update(overrides)
    return env
