"""Every cli.json_command subject must be selected by cli.command.json_envelope.

Failure mode: a new --json command lands but isn't covered by the
machine-envelope contract. The catalog stays renderable; the regression
shows up at the next CI run that exercises the JSON path.
"""

from __future__ import annotations

import pytest

from polylogue.proof.catalog import build_verification_catalog

pytestmark = pytest.mark.proof_law


def test_every_json_command_is_selected_by_envelope_claim() -> None:
    catalog = build_verification_catalog()
    json_command_subjects = {subject.id for subject in catalog.subjects if subject.kind == "cli.json_command"}
    selected = {
        obligation.subject.id
        for obligation in catalog.obligations
        if obligation.claim.id == "cli.command.json_envelope"
    }
    missing = json_command_subjects - selected
    assert not missing, f"--json commands missing cli.command.json_envelope: {sorted(missing)}"
