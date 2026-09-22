"""Import-time archive-tier initialization ratchet (polylogue-62j1f AC2).

``polylogue/storage/sqlite/archive_tiers/__init__.py`` materializes the audit
column disposition at module scope, and that materialization runs the canonical
DDL: importing the package costs a complete six-tier ``ddl_fresh`` pass plus a
complete six-tier ``prototype_hit`` pass -- 12 initializations -- before a
caller has asked for an archive. Measured 2026-09-22 at 9ef655cb8: 30
``sqlite3.connect`` calls and 888.8 ms of ``self`` import time
(``python -X importtime``), against 1.46 s / 47.8 MiB for the whole package
import tree and 0.08 s / 0.0 MiB for ``import polylogue`` alone.

Every pytest worker, every ``polylogue`` command and every ``--collect-only``
pays it once, which is why a pure-parser selection of 167 tests still reports
exactly these 12 initializations and no more: the floor is the import, not the
tests. The ratchet keeps that floor from growing while the eager
materialization is decided (it is an import-time invariant assertion, so making
it lazy moves when the invariant fires and is not this test's call).

Anti-vacuity, both directions:

* Raise the floor -- add another module-scope archive bootstrap under
  ``polylogue/storage`` -- and ``test_import_floor`` goes red naming the count.
* The counter itself must be live, or a floor of "at most 12" would pass on a
  recorder that counts nothing. ``test_counter_is_live`` bootstraps one real
  archive in the same subprocess and requires the count to rise, so a broken or
  stubbed ``archive_tier_init_counts`` fails rather than certifying silence.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Initializations charged to importing the package, measured at 9ef655cb8.
#: A ratchet: this may shrink, never grow.
IMPORT_TIER_INIT_BASELINE = 12

_PROGRAM = """
import json, sys, tempfile
from pathlib import Path

sys.path.insert(0, {repo!r})
from polylogue.storage.sqlite.archive_tiers import bootstrap

at_import = bootstrap.archive_tier_init_counts()
with tempfile.TemporaryDirectory() as scratch:
    bootstrap.initialize_active_archive_root(Path(scratch) / "archive")
    after = bootstrap.archive_tier_init_counts()
print(json.dumps({{"at_import": at_import, "after_one_archive": after}}))
"""


@pytest.fixture(scope="module")
def import_probe() -> dict[str, dict[str, int]]:
    """Tier-init counts in a process that imported nothing else first.

    A subprocess is the only honest measurement: this test process has already
    imported the package through some other module, so its own counters carry
    that history.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _PROGRAM.format(repo=str(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, f"probe failed: {completed.stderr[-2000:]}"
    probe: dict[str, dict[str, int]] = json.loads(completed.stdout.strip().splitlines()[-1])
    return probe


def test_import_floor(import_probe: dict[str, dict[str, int]]) -> None:
    """Importing the tier package costs no more initializations than measured."""
    at_import = import_probe["at_import"]
    total = sum(at_import.values())
    assert total <= IMPORT_TIER_INIT_BASELINE, (
        f"importing polylogue.storage.sqlite.archive_tiers now costs {total} archive tier "
        f"initializations, up from the measured {IMPORT_TIER_INIT_BASELINE}: {at_import}. "
        "This is paid by every pytest worker, CLI invocation and collection."
    )
    assert "ops.ddl_reapply" not in at_import, f"a whole-tier DDL re-execution now runs at import: {at_import}"


def test_counter_is_live(import_probe: dict[str, dict[str, int]]) -> None:
    """Bootstrapping one archive raises the count the floor is measured with."""
    at_import = sum(import_probe["at_import"].values())
    after = sum(import_probe["after_one_archive"].values())
    assert after > at_import, (
        "archive_tier_init_counts did not move when a real archive was bootstrapped "
        f"({at_import} -> {after}); the floor above would pass on a dead counter."
    )
