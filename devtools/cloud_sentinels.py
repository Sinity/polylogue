"""The cloud-sandbox environment values, and where they may be honoured.

`.claude/settings.json` carries an `env` block so a Claude Code Web / Codex Cloud
sandbox has a writable archive, a small worker count and scratch paths that
exist. Claude Code applies that block in EVERY session, including on this
workstation, so each of those values leaks into local agent runs.

They were fixed one at a time, and the shape of the fix is identical every time:
recognise the exact cloud value and decline it where the workstation scratch
mount proves this is not a sandbox. What differed was where each literal lived
and how each site decided it was on a workstation -- three constants across three
modules, and five separate `is_dir()` calls expressing one predicate. That is how
the fifth leak stays open while four are closed: nothing enumerates the set, so
nothing can say which are handled.

This module is that enumeration. It holds the sentinels and the single predicate,
so a new cloud value is declared in one place and honoured consistently, and so
"which leaks are handled" is answerable by reading one file.

The rule, stated once: a cloud sentinel is honoured only in a cloud sandbox. Any
OTHER value of the same variable is a deliberate operator override and always
wins -- the sentinel is recognised by its exact cloud value, never by its name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = [
    "CLOUD_SENTINELS",
    "INDISTINGUISHABLE_SENTINELS",
    "cloud_sentinel_declined",
    "running_in_cloud_sandbox",
]

#: The workstation's canonical scratch mount. Its absence is what makes a
#: sandbox a sandbox: a cloud runner has no /realm, and no amount of small-/tmp
#: pressure on a workstation should be mistaken for one.
_WORKSTATION_SCRATCH_MOUNT: Final = Path("/realm/tmp")

#: Variable -> the exact value `.claude/settings.json` sets for cloud.
CLOUD_SENTINELS: Final[dict[str, str]] = {
    "POLYLOGUE_ARCHIVE_ROOT": "/tmp/polylogue-archive",
    "POLYLOGUE_FORCE_PLAIN": "1",
    "HYPOTHESIS_PROFILE": "ci",
    "POLYLOGUE_PYTEST_WORKERS": "2",
    "POLYLOGUE_PYTEST_BASETEMP_ROOT": "/tmp/polylogue-pytest",
}


#: Sentinels that CANNOT be declined, because their cloud value is
#: indistinguishable from a deliberate one.
#:
#: `POLYLOGUE_FORCE_PLAIN=1` is the whole set. Every other sentinel has a
#: distinguishing shape -- a specific path, "2", "ci" without POLYLOGUE_CI --
#: but "1" is exactly what anyone wanting plain output would write, and this
#: repository's own demo, proof and lab-scenario paths set precisely that on
#: purpose. Declining it by value would override them.
#:
#: Listed rather than omitted so the enumeration stays complete against
#: settings.json: this leak is open, its impact is cosmetic (interactive Rich
#: output), and closing it needs lane separation -- cloud-only settings not
#: living in a file both lanes read -- which is an operator decision, not a
#: guard.
INDISTINGUISHABLE_SENTINELS: Final[frozenset[str]] = frozenset({"POLYLOGUE_FORCE_PLAIN"})


def running_in_cloud_sandbox() -> bool:
    """Whether this is a sandbox rather than the workstation.

    One predicate, expressed once. Five call sites had spelled it as their own
    `DEFAULT_PYTEST_BASETEMP_ROOT.parent.is_dir()`, which is correct but reads
    as a statement about pytest temporary directories rather than about which
    machine this is.
    """
    return not _WORKSTATION_SCRATCH_MOUNT.is_dir()


def cloud_sentinel_declined(name: str, value: str | None) -> bool:
    """Whether ``value`` is the cloud sentinel for ``name`` and must be ignored.

    False for any other value, including a deliberate operator override that
    happens to resemble it, and False in an actual sandbox where the sentinel is
    exactly right.
    """
    if value is None:
        return False
    if name in INDISTINGUISHABLE_SENTINELS:
        return False
    sentinel = CLOUD_SENTINELS.get(name)
    if sentinel is None:
        return False
    return value.strip() == sentinel and not running_in_cloud_sandbox()
