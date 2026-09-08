"""Cloud-sandbox values shared with workstation sessions.

Ordinary two-worker requests must be honored on the workstation.
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


#: Sentinels whose values are also ordinary workstation requests.
INDISTINGUISHABLE_SENTINELS: Final[frozenset[str]] = frozenset({"POLYLOGUE_FORCE_PLAIN", "POLYLOGUE_PYTEST_WORKERS"})


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
