"""Paired negative controls on findings (rxdo.9.7, mechanism G).

A finding MAY carry ``control_refs``: declared comparison definitions bound to
frame, matching variables, and expected-null behavior BEFORE execution. The
renderer shows claim-vs-control side by side; a registered claim with a
passing control outranks one without. Convention + validation only -- a
finding stays a ``finding.v1`` assertion with an additive ``controls`` field
in ``value_json``; no new storage tier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

ControlKind: TypeAlias = Literal[
    "matched_task_different_treatment",
    "shifted_window",
    "permuted_label",
    "unrelated_cohort",
]

_VALID_CONTROL_KINDS: frozenset[str] = frozenset(
    {"matched_task_different_treatment", "shifted_window", "permuted_label", "unrelated_cohort"}
)

#: Control kinds that must isolate the challenged mechanism by holding
#: declared matching variables constant. ``unrelated_cohort`` is deliberately
#: excluded -- it is validated by the confound-declaration path instead,
#: because "unrelated" and "matched" are opposite claims.
_MATCHED_CONTROL_KINDS: frozenset[str] = frozenset(
    {"matched_task_different_treatment", "shifted_window", "permuted_label"}
)


@dataclass(frozen=True, slots=True)
class NegativeControl:
    control_kind: ControlKind
    query_ref: str
    result_ref: str
    matching_variables: tuple[str, ...]
    expected_null: str
    confounds_checked: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.control_kind not in _VALID_CONTROL_KINDS:
            raise ValueError(f"unsupported control kind: {self.control_kind!r}")
        if not self.query_ref or not self.result_ref:
            raise ValueError("negative control requires query_ref and result_ref")
        if not self.expected_null.strip():
            raise ValueError("negative control requires a declared expected-null behavior")


@dataclass(frozen=True, slots=True)
class ControlValidation:
    control: NegativeControl
    accepted: bool
    reason: str


def validate_control(control: NegativeControl, *, claim_frame_variables: Sequence[str]) -> ControlValidation:
    """Reject a control that does not isolate the challenged mechanism.

    Matched-shape controls (matched-task/shifted-window/permuted-label) must
    declare matching variables that are a non-empty subset of the claim's own
    frame variables -- they hold something constant to isolate one changed
    mechanism. ``unrelated_cohort`` controls are rejected outright unless
    every claim frame variable is explicitly named as a checked confound;
    otherwise a deliberately divergent baseline would silently pass as a
    control.
    """

    frame_variables = set(claim_frame_variables)

    if control.control_kind == "unrelated_cohort":
        missing = frame_variables - set(control.confounds_checked)
        if missing:
            return ControlValidation(
                control=control,
                accepted=False,
                reason=f"unrelated_cohort control leaves confounds unchecked: {sorted(missing)}",
            )
        return ControlValidation(control=control, accepted=True, reason="all frame confounds declared checked")

    assert control.control_kind in _MATCHED_CONTROL_KINDS  # exhaustive over ControlKind, mypy-enforced
    if not control.matching_variables:
        return ControlValidation(
            control=control,
            accepted=False,
            reason=f"{control.control_kind} control declares no matching variables",
        )
    if not set(control.matching_variables) <= frame_variables:
        return ControlValidation(
            control=control,
            accepted=False,
            reason="matching variables are not a subset of the claim's frame variables",
        )
    return ControlValidation(control=control, accepted=True, reason="matched on declared frame variables")


@dataclass(frozen=True, slots=True)
class ControlOutcome:
    control: NegativeControl
    observed_null_held: bool


@dataclass(frozen=True, slots=True)
class ClaimWithControls:
    """A finding rendered claim-vs-control, side by side."""

    claim_ref: str
    controls: tuple[ControlOutcome, ...]

    @property
    def downgraded(self) -> bool:
        """A claim downgrades if ANY bound control's expected-null failed."""

        return any(not outcome.observed_null_held for outcome in self.controls)

    @property
    def rank_tier(self) -> Literal["controlled_pass", "controlled_fail", "uncontrolled"]:
        if not self.controls:
            return "uncontrolled"
        return "controlled_fail" if self.downgraded else "controlled_pass"


__all__ = [
    "ClaimWithControls",
    "ControlKind",
    "ControlOutcome",
    "ControlValidation",
    "NegativeControl",
    "validate_control",
]
