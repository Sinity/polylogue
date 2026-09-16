"""Every read view's declared execution route, proven by executing it.

polylogue-dutav: ``cli/read_view_registry.py`` declared ten views as
``session-read-projection`` over ``session.read``.  Only ``hooks`` is one --
``daemon_reads._SESSION_EVIDENCE_READERS`` serves exactly that kind and raises
for any other, and the transcript kind is the only other branch.  The rest read
the archive in this process through the Python API facade.  The old validator
(``_validate_read_view_classification``) checked the table only against itself,
so the claim could never be wrong.

These tests check it against the kernel: each view is dispatched with the daemon
absent, and the operations that actually reach
``cli.operation_kernel.dispatch`` are compared with the operations the row
declares.  A row that claims an operation it does not execute goes red, and so
does one that executes an operation it does not declare.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from polylogue.cli.read_view_registry import IN_PROCESS_READ_VIEWS, READ_VIEW_HANDLER_METADATA
from polylogue.cli.root_request import RootModeRequest
from polylogue.config import Config

SESSION_NATIVE_ID = "route-probe"


@pytest.fixture
def probe_env(tmp_path: Path) -> tuple[Any, Config, str]:
    """A real one-session archive plus the ``AppEnv`` the handlers expect."""

    from polylogue.api import Polylogue
    from tests.infra.storage_records import SessionBuilder

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    stamp = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc).isoformat()
    (
        SessionBuilder(archive_root / "index.db", SESSION_NATIVE_ID)
        .provider("codex")
        .title("Route probe")
        .created_at(stamp)
        .updated_at(stamp)
        .add_message("p1", role="user", text="which route does this view take?")
        .add_message("p2", role="assistant", text="the one the kernel records")
        .save()
    )
    config = Config(
        archive_root=archive_root,
        db_path=archive_root / "index.db",
        render_root=tmp_path / "render",
        sources=[],
    )
    # ``AppEnv`` builds ``config``/``polylogue`` lazily off its services and
    # exposes them as read-only properties, so the route probe injects a stand-in
    # rather than a half-built real one; a ``MagicMock`` base keeps any
    # incidental attribute a handler reaches for from becoming the failure under
    # test (the route, not the render, is what these assert).
    env = MagicMock()
    env.config = config
    env.polylogue = Polylogue(config=config)
    env.debug_timing = False

    # Read the id back from the archive rather than composing it: the builder
    # owns the provider -> origin mapping, and a guessed prefix would silently
    # turn every route probe into a "session not found" that reaches no
    # operation for the wrong reason.
    import sqlite3

    with sqlite3.connect(archive_root / "index.db") as conn:
        rows = conn.execute("SELECT session_id FROM sessions").fetchall()
    assert len(rows) == 1, rows
    return env, config, str(rows[0][0])


def _operations_reached(
    monkeypatch: pytest.MonkeyPatch,
    env: Any,
    config: Config,
    session_id: str,
    view: str,
) -> tuple[str, ...]:
    """Run one read view with the daemon absent, recording kernel dispatches.

    ``daemon_only=True`` is forced on every call, so a view that genuinely
    lowers to an operation refuses (the kernel raises
    ``OperationUnavailableError``) instead of falling back to the direct reader
    and looking, from the outside, exactly like a view that never used an
    operation at all.  What the assertion reads is the recorded operation
    names, so a handler failing afterwards is expected and irrelevant.
    """

    from polylogue.cli import operation_kernel
    from polylogue.cli.read_view_handlers import run_read_view
    from polylogue.cli.read_views.base import ReadViewInvocation

    reached: list[str] = []
    real_dispatch = operation_kernel.dispatch

    def _recording_dispatch(cfg: object, request: Any, **kwargs: object) -> object:
        reached.append(request.operation)
        kwargs["daemon_only"] = True
        kwargs["daemon_disabled"] = False
        return real_dispatch(cfg, request, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(operation_kernel, "dispatch", _recording_dispatch)

    metadata = READ_VIEW_HANDLER_METADATA[view]
    invocation = ReadViewInvocation(
        view=view,
        session_id=None if metadata.session_policy == "none" else session_id,
        output_format=None,
        destination="stdout",
        out_path=None,
    )
    request = RootModeRequest.from_params({"_config": config, "id": session_id})
    try:
        run_read_view(env, request, invocation)
    except BaseException:
        pass
    return tuple(dict.fromkeys(reached))


@pytest.mark.parametrize("view", sorted(READ_VIEW_HANDLER_METADATA))
def test_each_view_executes_the_route_it_declares(
    view: str, probe_env: tuple[Any, Config, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The declared operations are the operations that reach the kernel.

    Anti-vacuity: declare ``events`` (or any other in-process view) as
    ``session-read-projection`` over ``session.read`` and this goes red,
    because no ``session.read`` dispatch is recorded for it.  Equally, drop
    ``cli.query`` from ``summary`` and the recorded dispatch becomes an
    undeclared operation.
    """

    env, config, session_id = probe_env
    metadata = READ_VIEW_HANDLER_METADATA[view]
    reached = _operations_reached(monkeypatch, env, config, session_id, view)

    if metadata.execution_kind in {"in-process", "distinct-operation"}:
        assert reached == (), f"{view} declares no operation but dispatched {reached}"
        return

    assert set(metadata.operations), f"{view} declares {metadata.execution_kind} without an operation"
    assert set(reached) <= set(metadata.operations), (
        f"{view} dispatched {sorted(set(reached) - set(metadata.operations))}, which it does not declare"
    )
    assert reached, f"{view} declares {metadata.operations} but reached no operation"


def test_session_read_projections_name_a_kind_the_operation_actually_serves() -> None:
    """The second mirror: what ``session.read`` serves, not what a row claims.

    ``_session_read_payload`` answers ``transcript`` itself and delegates every
    other kind to ``_SESSION_EVIDENCE_READERS``, raising for anything absent
    from it.  A view declared as a ``session.read`` projection must therefore
    name a kind in that union.

    Anti-vacuity: re-declare ``file-edits`` as a ``session.read`` projection and
    this goes red, because ``session.read`` raises
    "does not serve kind 'file-edits'".
    """

    from polylogue.operations.daemon_reads import _SESSION_EVIDENCE_READERS

    served = {"transcript", *_SESSION_EVIDENCE_READERS}
    declared = {
        view_id
        for view_id, metadata in READ_VIEW_HANDLER_METADATA.items()
        if metadata.execution_kind == "session-read-projection"
    }

    assert declared <= served, f"declared session.read projections the operation does not serve: {declared - served}"


def test_the_public_preset_catalog_covers_every_declared_view() -> None:
    """The third mirror: ``surfaces/read_contract.READ_PRESETS``.

    It used to be a hand-written list and had silently lost ``lineage`` and
    ``effective_context``; nothing compared it with the view declaration.

    Anti-vacuity: restate ``READ_PRESETS`` as a literal tuple missing any view
    and this goes red.
    """

    from polylogue.archive.viewport import read_view_choices
    from polylogue.surfaces.read_contract import READ_PRESETS

    assert {preset.name for preset in READ_PRESETS} == set(read_view_choices())
    assert set(READ_VIEW_HANDLER_METADATA) == set(read_view_choices())


def test_the_in_process_baseline_is_exactly_what_is_declared() -> None:
    """The ratchet may shrink, never grow.

    Anti-vacuity: declare one more view ``in-process`` without removing it from
    the baseline and ``validate_read_view_metadata_registry`` raises at import;
    remove a still-in-process view from the baseline and this goes red.
    """

    declared = {
        view_id for view_id, metadata in READ_VIEW_HANDLER_METADATA.items() if metadata.execution_kind == "in-process"
    }

    assert declared == IN_PROCESS_READ_VIEWS


def test_operation_names_are_declared_operations() -> None:
    """A row may not name an operation the protocol does not declare.

    Anti-vacuity: name ``session.summary`` on any row and this goes red.
    """

    from polylogue.operations.daemon_protocol import daemon_operation_spec

    for view_id, metadata in READ_VIEW_HANDLER_METADATA.items():
        for operation in metadata.operations:
            assert daemon_operation_spec(operation) is not None, f"{view_id} names undeclared operation {operation}"


def test_probe_env_reads_a_real_session(probe_env: tuple[Any, Config, str]) -> None:
    """Guard the fixture: a route probe over an empty archive proves nothing.

    Anti-vacuity: seed no messages and this goes red, which is what would
    otherwise let every view "reach no operation" for the wrong reason.
    """

    from polylogue.api.sync.bridge import run_coroutine_sync

    env, config, session_id = probe_env
    assert config.db_path.exists()
    session = run_coroutine_sync(env.polylogue.get_session(session_id))
    assert session is not None
    assert len(session.messages) == 2
