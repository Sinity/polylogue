"""Unified command surface for repository-maintenance tools.

Delegates to Click dispatch from devtools.click_dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

from devtools.click_dispatch import main  # noqa: E402

__all__ = ["main"]


def _entrypoint() -> int:
    """Read sys.argv and delegate to the Click-based dispatch."""
    return main(argv=sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
