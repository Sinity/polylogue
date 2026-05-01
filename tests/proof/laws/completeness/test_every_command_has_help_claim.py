"""Every CLI command subject must be selected by every cli.command claim.

Failure mode protected against: a new command lands without one of the
help / no-traceback / plain-mode claims selecting it. The catalog still
renders cleanly because the existing claims select the *other* commands —
nothing forces full coverage. This law makes the missing-coverage case
loud at the test level.
"""

from __future__ import annotations

import pytest

from polylogue.proof.catalog import build_verification_catalog

pytestmark = pytest.mark.proof_law

_REQUIRED_COMMAND_CLAIMS = (
    "cli.command.help",
    "cli.command.no_traceback",
    "cli.command.plain_mode",
)


@pytest.mark.parametrize("claim_id", _REQUIRED_COMMAND_CLAIMS)
def test_every_command_subject_is_selected_by_required_command_claim(claim_id: str) -> None:
    catalog = build_verification_catalog()
    command_subjects = {subject.id for subject in catalog.subjects if subject.kind == "cli.command"}
    selected = {obligation.subject.id for obligation in catalog.obligations if obligation.claim.id == claim_id}
    missing = command_subjects - selected
    assert not missing, f"commands missing {claim_id}: {sorted(missing)}"
