"""One producer for the quick-check fact published by every status surface.

Three surfaces published this fact and none of them measured it the same way:
``daemon/status.py`` hardcoded ``quick_check_result: "unknown"``,
``daemon/status_snapshot.py`` hardcoded ``quick_check_result: None``, and
``daemon/http.py`` ran a real probe for ``quick_check`` but still published a
literal ``quick_check_age_s: None`` beside it. ``"unknown"`` in particular
renders like an answer rather than like the absence of a measurement
(polylogue-20d.17.2).

The probe is deliberately the bounded index-openability read that
``/api/health`` has always run -- one ``sqlite_master`` lookup under the
interactive read profile -- not SQLite's ``PRAGMA quick_check``, which is a
full-file integrity scan and is never on a request path (``docs/internals.md``).
The field name is the long-standing public contract key; this module is where
its meaning is written down once.

Two key names survive because two public envelopes declare them:
``/api/status`` publishes ``quick_check_result`` and ``/api/health`` publishes
``quick_check``. Both are rendered from the same observation, so the two
surfaces cannot report different facts for one archive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from polylogue.core.evidence import Measured, Unavailable
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.tier_access import capture_sqlite_read

QuickCheckResult = Literal["pass", "error", "unavailable"]

STATUS_RESULT_KEY = "quick_check_result"
HEALTH_RESULT_KEY = "quick_check"
AGE_KEY = "quick_check_age_s"


class QuickCheckObservation(BaseModel):
    """What a status surface may honestly say about the index quick check.

    ``unavailable`` is the explicit third state: the probe did not run on this
    route, or its collection did not complete. It is not ``pass`` (which would
    certify an archive nothing opened) and not ``error`` (which would report a
    failure nothing observed).
    """

    result: QuickCheckResult = "unavailable"
    #: Seconds since the observation was captured. ``None`` whenever nothing
    #: was measured -- an unmeasured fact has no age.
    age_s: float | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the probe ran and succeeded. ``unavailable`` is never ok."""
        return self.result == "pass"

    def payload(self, *, result_key: str = STATUS_RESULT_KEY) -> dict[str, object]:
        """Render the two declared envelope fields from this one observation."""
        return {result_key: self.result, AGE_KEY: self.age_s}

    def with_age(self, age_s: float | None) -> QuickCheckObservation:
        """Attach the age of the snapshot this observation was served from."""
        if self.result == "unavailable":
            return self
        return self.model_copy(update={"age_s": None if age_s is None else round(age_s, 3)})


def unmeasured_quick_check(reason: str) -> QuickCheckObservation:
    """Return the explicit not-measured observation for a route that skipped it."""
    return QuickCheckObservation(result="unavailable", age_s=None, reason=reason)


def _probe(index_db: Path) -> None:
    from polylogue.operations.diagnostic_reads import one_shot_diagnostic_read

    with one_shot_diagnostic_read(index_db, tier=ArchiveTier.INDEX) as conn:
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()


def observe_quick_check(index_db: Path) -> QuickCheckObservation:
    """Run the bounded index-openability probe and report what it found.

    A missing index file is ``error``, not ``unavailable``: the probe ran and
    established that the archive cannot be opened. ``unavailable`` is reserved
    for routes that never ran the probe at all.

    The SQLite failure is classified once at the storage seam
    (``capture_sqlite_read`` -> ``polylogue.core.evidence``) rather than by an
    improvised handler here, so "errored" and "genuinely open" cannot become
    indistinguishable at this call site.
    """
    if not index_db.exists():
        return QuickCheckObservation(result="error", age_s=0.0, reason=f"index database is absent: {index_db}")
    try:
        evidence = capture_sqlite_read(lambda: _probe(index_db))
    except OSError as exc:
        return QuickCheckObservation(result="error", age_s=0.0, reason=f"{type(exc).__name__}: {exc}")
    if isinstance(evidence, Measured):
        return QuickCheckObservation(result="pass", age_s=0.0, reason=None)
    if isinstance(evidence, Unavailable):
        return QuickCheckObservation(result="error", age_s=0.0, reason=evidence.detail or evidence.reason)
    return QuickCheckObservation(result="error", age_s=0.0, reason="index probe produced unsupported evidence")


__all__ = [
    "AGE_KEY",
    "HEALTH_RESULT_KEY",
    "STATUS_RESULT_KEY",
    "QuickCheckObservation",
    "QuickCheckResult",
    "observe_quick_check",
    "unmeasured_quick_check",
]
