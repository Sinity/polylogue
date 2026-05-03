"""Unified command surface for repository-maintenance tools.

Delegates to Click dispatch from devtools.click_dispatch.
"""

from __future__ import annotations

import sys

from devtools.click_dispatch import main


def _entrypoint() -> int:
    """Read sys.argv and delegate to the Click-based dispatch."""
    return main(argv=sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
