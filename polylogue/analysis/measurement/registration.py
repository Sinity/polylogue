"""Pre-registration with graph-provable ordering (rxdo.9.3).

A REGISTERED analysis is a finding-candidate whose ``expected`` is fixed
before the confirming query runs. Ordering is provable purely from
timestamps/epochs on the two events -- no self-report, no trust:

    registered_at < run_at             (declared BEFORE executed)
    AND run_epoch > registered_epoch   (tested on data that arrived after
                                         the hypothesis was written down)
    AND the run's metric/query refs match what was registered (no silent
        definition swap after exposure)

A claim renders ``"confirmed (pre-registered)"`` only under that ordering;
everything else renders as one of the explicit exploratory variants below.
Agents don't get moral credit for calling something pre-registered -- they
get a provable badge or they don't (prior art: polylogue-e5b5).

This module is the pure, storage-agnostic ordering proof. The durable
registration/finding storage this eventually reads from (rxdo.3's frame
contract, rxdo.4's finding.v1, or an ``ExperimentDefinition`` from ``stc``)
has not landed in this tree yet; wiring a concrete registration store is
deferred to whichever lane lands that storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

RegistrationStatus = Literal[
    "registered",
    "exploratory",
    "exploratory-definition-drift",
    "exploratory-post-hoc",
]


@dataclass(frozen=True, slots=True)
class PreRegistration:
    """A hypothesis fixed before its confirming run."""

    registration_id: str
    hypothesis: str
    expected: str
    metric_ref: str
    query_ref: str
    registered_at: datetime
    registered_epoch: int
    frame: str = ""
    exclusions: tuple[str, ...] = ()
    stopping_rule: str = ""
    analysis_plan: str = ""


@dataclass(frozen=True, slots=True)
class RegistrationEvaluation:
    """A later run that may or may not confirm a :class:`PreRegistration`."""

    registration_id: str
    metric_ref: str
    query_ref: str
    run_at: datetime
    run_epoch: int
    actual: str


def registration_status(registration: PreRegistration, evaluation: RegistrationEvaluation) -> RegistrationStatus:
    """Prove (or refuse) the pre-registration ordering from the graph alone.

    Raises:
        ValueError: if ``evaluation`` does not reference ``registration``.
    """

    if evaluation.registration_id != registration.registration_id:
        raise ValueError(
            f"evaluation registration_id {evaluation.registration_id!r} "
            f"does not reference registration {registration.registration_id!r}"
        )
    if evaluation.metric_ref != registration.metric_ref or evaluation.query_ref != registration.query_ref:
        return "exploratory-definition-drift"
    if evaluation.run_at <= registration.registered_at:
        return "exploratory"
    if evaluation.run_epoch <= registration.registered_epoch:
        return "exploratory-post-hoc"
    return "registered"


def render_badge(status: RegistrationStatus) -> str:
    """Render the human-facing badge for a proven registration status."""

    if status == "registered":
        return "confirmed (pre-registered)"
    if status == "exploratory-definition-drift":
        return "exploratory (metric/query definition changed after registration)"
    if status == "exploratory-post-hoc":
        return "exploratory (data was already observed at registration time)"
    return "exploratory"


__all__ = [
    "PreRegistration",
    "RegistrationEvaluation",
    "RegistrationStatus",
    "registration_status",
    "render_badge",
]
