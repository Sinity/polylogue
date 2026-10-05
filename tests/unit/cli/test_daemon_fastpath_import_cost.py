"""The daemon fast path must not eagerly import the heavy local-execution stack.

polylogue-g3jk: a daemon-served ``find`` paid the full ``polylogue.api`` import
cost (~1.7-2.8s of pydantic/provider-parser import time) even though the daemon
served the request over UDS and the CLI never touched ``ArchiveStore``, the
``Polylogue`` facade, or any local-execution renderer. Fixed by:

- ``polylogue/cli/query.py`` importing ``polylogue.core.async_bridge`` (a
  dependency-free coroutine driver) instead of ``polylogue.api.sync.bridge``,
  which is a submodule of the heavy ``polylogue.api`` package.
- ``polylogue/cli/archive_query.py`` and
  ``polylogue/cli/operation_kernel.py`` deferring every local-execution-only
  import (``ArchiveStore``, ``polylogue.surfaces.payloads``,
  ``polylogue.storage.search_providers``, ``polylogue.archive.stats``, the
  declared read handlers and the direct-read operation context) to the
  specific functions that call them, instead of the module's own top level.
- ``polylogue/cli/shared/types.py``'s ``AppEnv.config``/``.runtime``
  properties returning the already-resolved ``ResolvedRuntimeConfig``
  directly instead of forcing ``AppEnv.services`` (which imports the full
  ``polylogue.services`` -> ``storage.repository`` -> ``storage.sqlite``
  stack merely to hand back a ``Config`` projection that was already
  computed).

This is a subprocess-based behavioral contract (following the pattern
``tests/unit/cli/test_schema_drift_status.py::test_drift_marker_import_path_stays_light``
established for #3507): it drives ``execute_query_request`` through a real UDS
daemon request and inspects ``sys.modules`` in a fresh interpreter, rather than
grepping import statements. The daemon protocol validator legitimately loads
``surfaces.payloads`` for its response schema, so that module is not treated as
a local-execution import. Reverting any of the
three fixes above makes this fail: e.g. restoring
``from polylogue.api.sync.bridge import run_coroutine_sync`` at the top of
``query.py`` makes ``polylogue.api`` appear in ``sys.modules`` even though the
daemon served the request.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_FORBIDDEN_ON_DAEMON_HIT = (
    "polylogue.api",
    "polylogue.services",
    "polylogue.storage.repository",
    "polylogue.storage.sqlite.archive_tiers.archive",
    "polylogue.storage.sqlite.archive_tiers.write",
)

_PROBE = r"""
import sys

import io
import json
from contextlib import redirect_stdout

from polylogue.cli.query import execute_query_request
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import resolve_runtime_config

runtime = resolve_runtime_config()
env = AppEnv(runtime=runtime, plain=True)
request = RootModeRequest(params={"output_format": "json", "limit": 5}, query_terms=("demo", "session"))

buf = io.StringIO()
with redirect_stdout(buf):
    execute_query_request(env, request)

rendered = buf.getvalue()
document = json.loads(rendered)
assert document["source"] == "daemon", f"request was not served by the daemon: {document!r}"
assert __EXPECTED_SESSION_ID__ in rendered, f"seeded session did not reach output: {document!r}"

forbidden = __FORBIDDEN__
loaded = [m for m in forbidden if m in sys.modules]
print(",".join(loaded) if loaded else "CLEAN")
"""


@pytest.mark.uses_real_clock(
    "subprocess wall-clock is incidental; this test asserts the module import graph, not timing"
)
def test_daemon_served_query_does_not_import_heavy_local_execution_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daemon-served ``find`` must not import the local API/storage execution stack."""
    from polylogue.daemon.socket_path import daemon_socket_path
    from tests.infra.daemon_operations import running_daemon_operations
    from tests.infra.storage_records import SessionBuilder

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    builder = (
        SessionBuilder(archive_root / "index.db", "fastpath-probe")
        .provider("claude-code")
        .title("Demo session")
        .add_message("m1", role="user", text="demo session")
    )
    expected_session_id = builder.native_session_id()

    def seed(owned_root: Path) -> None:
        assert owned_root == archive_root
        builder.save()

    monkeypatch.delenv("POLYLOGUE_ARCHIVE_ROOT", raising=False)
    code = _PROBE.replace("__FORBIDDEN__", repr(_FORBIDDEN_ON_DAEMON_HIT)).replace(
        "__EXPECTED_SESSION_ID__", repr(expected_session_id)
    )
    client_home = tmp_path / "client-home"
    xdg_values = {
        "HOME": str(client_home),
        "XDG_CONFIG_HOME": str(client_home / ".config"),
        "XDG_DATA_HOME": str(client_home / ".local/share"),
        "XDG_CACHE_HOME": str(client_home / ".cache"),
        "XDG_STATE_HOME": str(client_home / ".local/state"),
    }
    client_home.mkdir()
    with running_daemon_operations(
        archive_root,
        seed_archive=seed,
        socket_path=daemon_socket_path(archive_root),
    ):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env={
                **os.environ,
                **xdg_values,
                "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
                "POLYLOGUE_FORCE_PLAIN": "1",
            },
        )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CLEAN", (
        f"heavy modules leaked into the daemon fast path: {result.stdout.strip()}\nstderr: {result.stderr}"
    )


@pytest.mark.parametrize(
    "probe,forbidden",
    [
        (
            "from polylogue.surfaces.machine_envelope import success; assert success({'value': 1}).to_dict()['result'] == {'value': 1}",
            ("polylogue.surfaces.authority",),
        ),
        (
            "from polylogue.core.bounded import run_bounded; assert run_bounded([sys.executable, '-c', 'pass'], 10).returncode == 0",
            ("asyncio",),
        ),
        (
            "from polylogue.cli.shared.machine_errors import error_runtime; error = error_runtime('synthetic failure').to_dict(); assert error['code'] == 'runtime_error'; assert 'outcome' not in error",
            ("polylogue.surfaces.outcome",),
        ),
        (
            "import click; from polylogue.cli.click_option_groups import _validate_origin_tokens; assert _validate_origin_tokens(click.Context(click.Command('probe')), click.Option(['--origin']), None) is None; assert _validate_origin_tokens(click.Context(click.Command('probe')), click.Option(['--exclude-origin']), '') is None",
            ("polylogue.sources.origin_specs",),
        ),
        (
            "from polylogue.archive.query.search_hits import bound_display_title; assert bound_display_title('synthetic title') == 'synthetic title'",
            ("polylogue.storage.archive_identity", "polylogue.storage.sqlite.archive_tiers.write"),
        ),
        (
            "import polylogue.coordination.envelope",
            (
                "polylogue.storage.archive_identity",
                "polylogue.storage.sqlite.connection_profile",
                "polylogue.storage.derived.topology",
                "polylogue.storage.derived.raw",
                "polylogue.storage.sqlite.run_projection_relations",
            ),
        ),
    ],
)
@pytest.mark.uses_real_clock("fresh subprocess observes actual imports and physical exit, not latency")
def test_unused_cold_branches_do_not_load_their_execution_owners(
    tmp_path: Path, probe: str, forbidden: tuple[str, ...]
) -> None:
    """Restoring an eager import loads an owner the selected branch never uses."""
    code = f"import sys\n{probe}\nassert not set({forbidden!r}) & sys.modules.keys()\n"
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
