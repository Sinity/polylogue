"""The browser/HTTP surface must never open a database of its own.

polylogue-bp12n.2 AC2: public browser data reads follow the daemon-owned
operation policy, and a browser may display its static shell or a named
absence but must not reopen the archive through a second executor. AC3 adds
the operational half -- daemon loss must not enable direct mutation.

The half of that contract which holds today is the data half: no module in
the surface family opens a connection by any route. Nothing pinned it.
``gate controlled-read`` is the nearest existing check and it censuses
``ArchiveStore.open_existing`` only, by AST -- a plain
``sqlite3.connect(...)`` or ``aiosqlite.connect(...)`` inside
``polylogue/daemon/http.py`` passes it. So this ratchet is not a second copy
of that gate; it covers the openers that gate cannot see, over the one
module family where a second executor would defeat the isolation programme.

It says nothing about the process move itself. The surface is still 11,754
lines inside the daemon package with no separate host, and this test does not
pretend otherwise: it keeps the property that makes the move safe to do later
from silently decaying before anyone starts.

Anti-vacuity: add any connection-opening call to a family module and
``test_no_surface_module_opens_a_database`` goes red naming the module and the
call. ``test_the_detector_sees_a_planted_opener`` runs the same detector over
synthetic sources that DO open connections, so a detector that silently
matched nothing could not pass, and
``test_the_surface_family_is_the_measured_one`` fails if a family module is
renamed or removed without updating the list.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Every HTTP/web-shell/browser module still inside ``polylogue/daemon`` at
#: 6490c82f3. This is the family polylogue-bp12n.2 would relocate.
SURFACE_FAMILY = (
    "polylogue/daemon/api_auth.py",
    "polylogue/daemon/events_http.py",
    "polylogue/daemon/healthz.py",
    "polylogue/daemon/http.py",
    "polylogue/daemon/route_contracts.py",
    "polylogue/daemon/route_types.py",
    "polylogue/daemon/route_families/maintenance.py",
    "polylogue/daemon/route_families/operational.py",
    "polylogue/daemon/route_families/read_detail.py",
    "polylogue/daemon/route_families/read_query.py",
    "polylogue/daemon/route_families/user_overlay.py",
    "polylogue/daemon/route_families/workspace.py",
    "polylogue/daemon/topology_http.py",
    "polylogue/daemon/web_auth.py",
    "polylogue/daemon/webui.py",
    "polylogue/daemon/webui_data.py",
    "polylogue/daemon/workspace_routes.py",
)

#: Attribute calls that open a database handle. ``open_existing`` is included
#: even though ``gate controlled-read`` also censuses it, because a reader of
#: this test should see the whole boundary rather than two-thirds of it.
_OPENER_ATTRIBUTES = frozenset({"connect", "connect_async", "open_existing", "open_or_create"})

#: Bare names that open one after ``from sqlite3 import connect`` or
#: ``from ... import ArchiveStore``.
_OPENER_NAMES = frozenset({"connect", "ArchiveStore"})


def database_openers(source: str) -> list[str]:
    """Names of every connection-opening call in ``source``."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _OPENER_ATTRIBUTES:
            base = func.value
            prefix = base.id if isinstance(base, ast.Name) else ast.unparse(base)
            found.append(f"{prefix}.{func.attr}")
        elif isinstance(func, ast.Name) and func.id in _OPENER_NAMES:
            found.append(func.id)
    return found


@pytest.mark.parametrize("relative_path", SURFACE_FAMILY)
def test_no_surface_module_opens_a_database(relative_path: str) -> None:
    """A surface module reads through the daemon, never through its own handle."""
    openers = database_openers((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    assert not openers, (
        f"{relative_path} opens a database handle directly: {sorted(set(openers))}. "
        "The browser surface reads and writes through the daemon-owned operation "
        "route (polylogue-bp12n.2 AC2); a second executor here means daemon loss "
        "stops being a named absence and starts being an unsupervised writer. Route "
        "the read through an operation, or serve a named unavailable outcome."
    )


def test_the_detector_sees_a_planted_opener() -> None:
    """The detector must match real openers, not pass by matching nothing."""
    assert database_openers("import sqlite3\nc = sqlite3.connect('x.db')\n") == ["sqlite3.connect"]
    assert database_openers("from sqlite3 import connect\nc = connect('x.db')\n") == ["connect"]
    assert database_openers("a = ArchiveStore.open_existing(root)\n") == ["ArchiveStore.open_existing"]
    assert database_openers("import aiosqlite\nc = await aiosqlite.connect(p)\n") == ["aiosqlite.connect"]
    assert database_openers("store = ArchiveStore(root)\n") == ["ArchiveStore"]
    # And it must not fire on an ordinary call.
    assert database_openers("respond(payload)\n") == []


def test_the_surface_family_is_the_measured_one() -> None:
    """Every listed module must exist, or the parametrization silently shrinks."""
    missing = [path for path in SURFACE_FAMILY if not (REPO_ROOT / path).exists()]
    assert not missing, (
        f"these surface-family modules no longer exist: {missing}. If they moved to a "
        "separate surface host, that is the polylogue-bp12n.2 relocation actually "
        "happening -- point this ratchet at the new package rather than deleting it."
    )
    assert len(SURFACE_FAMILY) >= 11
