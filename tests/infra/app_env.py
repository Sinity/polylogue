"""Test factory for a CLI ``AppEnv`` with a capturable console.

Introduced for the ``polylogue config`` secret-redaction tests (#1748), whose
``_run`` helper reads ``env.ui.console.file.getvalue()`` to assert on rendered
output. The module the test imported was never committed alongside it, which
broke collection of the whole unit suite (pytest aborts on a collection error);
this restores it.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import cast

from rich.console import Console

from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.services import RuntimeServices
from polylogue.ui import UI
from polylogue.ui.facade import ConsoleLike


def make_app_env(archive_root: Path | None = None) -> AppEnv:
    """Build an ``AppEnv`` whose console writes to an in-memory buffer.

    The console is plain (no ANSI) and wide (no wrapping) so tests can assert
    on exact rendered text via ``env.ui.console.file.getvalue()``.

    ``env.services`` is given an explicit, non-ambient ``Config`` (the
    "explicit library caller" compatibility path ``RuntimeServices`` itself
    documents) rather than left unresolved: #3079 ("close the 5-layer
    resolution gap with ResolvedRuntimeConfig") removed the old ambient
    fallback this factory's docstring used to describe, so a bare
    ``AppEnv(ui=ui)`` now raises ``ConfigError("RuntimeServices has no config
    projection")`` the first time any code path touches ``env.config`` /
    ``env.polylogue`` (polylogue-c66i triage: bisected 3 stale CLI test
    failures on this branch to this exact gap). Callers that need a specific
    archive root (or a real backend) should pass ``archive_root=tmp_path``;
    other callers get a disposable, never-written-to scratch directory.
    """
    ui = UI(plain=True)
    ui.console = cast(ConsoleLike, Console(file=io.StringIO(), force_terminal=False, width=200))
    root = archive_root if archive_root is not None else Path(tempfile.mkdtemp(prefix="polylogue-app-env-"))
    config = Config(archive_root=root, render_root=root, sources=[])
    return AppEnv(ui=ui, services=RuntimeServices(config=config))
