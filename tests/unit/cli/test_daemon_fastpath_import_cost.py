"""The daemon fast path must not eagerly import the heavy local-execution stack.

polylogue-g3jk: a daemon-served ``find`` paid the full ``polylogue.api`` import
cost (~1.7-2.8s of pydantic/provider-parser import time) even though the daemon
served the request over UDS and the CLI never touched ``ArchiveStore``, the
``Polylogue`` facade, or any local-execution renderer. Fixed by:

- ``polylogue/cli/query.py`` importing ``polylogue.core.async_bridge`` (a
  dependency-free coroutine driver) instead of ``polylogue.api.sync.bridge``,
  which is a submodule of the heavy ``polylogue.api`` package.
- ``polylogue/cli/archive_query.py`` deferring its local-execution-only
  imports (``ArchiveStore``, ``polylogue.surfaces.payloads``,
  ``polylogue.storage.search_providers``, ``polylogue.archive.stats``, the
  attached-units/unit-results helpers) to the specific functions that call
  them, instead of the module's own top level.
- ``polylogue/cli/shared/types.py``'s ``AppEnv.config``/``.runtime``
  properties returning the already-resolved ``ResolvedRuntimeConfig``
  directly instead of forcing ``AppEnv.services`` (which imports the full
  ``polylogue.services`` -> ``storage.repository`` -> ``storage.sqlite``
  stack merely to hand back a ``Config`` projection that was already
  computed).

This is a subprocess-based behavioral contract (following the pattern
``tests/unit/cli/test_schema_drift_status.py::test_drift_marker_import_path_stays_light``
established for #3507): it actually drives ``execute_query_request`` through a
mocked daemon-success response and inspects ``sys.modules`` in a fresh
interpreter, rather than grepping import statements. Reverting any of the
three fixes above makes this fail: e.g. restoring
``from polylogue.api.sync.bridge import run_coroutine_sync`` at the top of
``query.py`` makes ``polylogue.api`` appear in ``sys.modules`` even though the
daemon served the request.
"""

from __future__ import annotations

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
    "polylogue.surfaces.payloads",
)

_PROBE = r"""
import sys

import polylogue.cli.archive_query as aq

_PAYLOAD = {
    "items": [
        {
            "id": "demo:1",
            "origin": "claude-code-session",
            "title": "demo session",
            "created_at": None,
            "updated_at": None,
            "message_count": 1,
        }
    ],
    "total": 1,
    "_daemon_elapsed_ms": 1,
}


def _fake_fetch(config, params, disabled=False):
    return dict(_PAYLOAD)


aq._fetch_daemon_sessions_payload = _fake_fetch

import io
from contextlib import redirect_stdout

from polylogue.cli.query import execute_query_request
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import resolve_runtime_config

runtime = resolve_runtime_config()
env = AppEnv(runtime=runtime, plain=True)
request = RootModeRequest(params={"output_format": "json", "limit": 5}, query_terms=("demo",))

buf = io.StringIO()
with redirect_stdout(buf):
    execute_query_request(env, request)

rendered = buf.getvalue()
assert '"demo:1"' in rendered, f"daemon-mock payload did not reach output: {rendered!r}"

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
    """A daemon-served ``find`` must never import ``polylogue.api``/ArchiveStore/payloads."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    monkeypatch.delenv("POLYLOGUE_ARCHIVE_ROOT", raising=False)
    env = dict(**{"POLYLOGUE_ARCHIVE_ROOT": str(archive_root), "POLYLOGUE_FORCE_PLAIN": "1"})
    code = _PROBE.replace("__FORBIDDEN__", repr(_FORBIDDEN_ON_DAEMON_HIT))
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env={**__import__("os").environ, **env},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CLEAN", (
        f"heavy modules leaked into the daemon fast path: {result.stdout.strip()}\nstderr: {result.stderr}"
    )
