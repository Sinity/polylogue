"""Unified command surface for repository-maintenance tools.

Delegates to Click dispatch from devtools.click_dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

# Before any third-party import: an inherited PYTHONPATH naming another
# checkout is already in ``sys.path``, and click would load from it.
from devtools.checkout_guard import normalize_checkout_environment  # noqa: E402

_CORRECTED = normalize_checkout_environment(Path(_REPO_ROOT))
if _CORRECTED:
    sys.stderr.write(
        f"devtools: rebound the environment to this checkout ({_REPO_ROOT}); "
        f"it named another: {'; '.join(_CORRECTED)}\n"
    )

from devtools.click_dispatch import main  # noqa: E402

__all__ = ["main"]


def _entrypoint() -> int:
    """Read sys.argv and delegate to the Click-based dispatch."""
    return main(argv=sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
