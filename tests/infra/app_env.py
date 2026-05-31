"""Test factory for a CLI ``AppEnv`` with a capturable console.

Introduced for the ``polylogue config`` secret-redaction tests (#1748), whose
``_run`` helper reads ``env.ui.console.file.getvalue()`` to assert on rendered
output. The module the test imported was never committed alongside it, which
broke collection of the whole unit suite (pytest aborts on a collection error);
this restores it.
"""

from __future__ import annotations

import io
from typing import cast

from rich.console import Console

from polylogue.cli.shared.types import AppEnv
from polylogue.ui import UI
from polylogue.ui.facade import ConsoleLike


def make_app_env() -> AppEnv:
    """Build an ``AppEnv`` whose console writes to an in-memory buffer.

    The console is plain (no ANSI) and wide (no wrapping) so tests can assert
    on exact rendered text via ``env.ui.console.file.getvalue()``. Services use
    the ``AppEnv`` default factory, so ``env.config`` resolves from the ambient
    ``POLYLOGUE_CONFIG`` the caller sets.
    """
    ui = UI(plain=True)
    ui.console = cast(ConsoleLike, Console(file=io.StringIO(), force_terminal=False, width=200))
    return AppEnv(ui=ui)
