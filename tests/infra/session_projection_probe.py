"""Isolated startup probe for a synthetic shared projection declaration."""

from __future__ import annotations

import subprocess
import sys

_PROJECTION_PROBE = """
import sys
from polylogue.archive.session_projections import SESSION_LIST_PROJECTIONS, SessionListProjection
from polylogue.archive.viewport import get_read_view_profile
from polylogue.cli.read_view_handlers import READ_VIEW_HANDLERS
from polylogue.cli.read_view_registry import READ_VIEW_HANDLER_METADATA
from polylogue.surfaces.projection_spec import READ_VIEW_PROJECTION_FAMILIES

original_profile = get_read_view_profile("events")
original_handler = READ_VIEW_HANDLERS["events"]
original_metadata = READ_VIEW_HANDLER_METADATA["events"]
original_family = READ_VIEW_PROJECTION_FAMILIES["events"]

# Imports above inspect the unchanged contract in this disposable process.
# Reimport consumers after the new declaration, with no shared pytest state.
for name in tuple(sys.modules):
    if name.startswith(("polylogue.archive.viewport", "polylogue.cli.read_view_", "polylogue.surfaces.projection_spec")):
        del sys.modules[name]
SESSION_LIST_PROJECTIONS["projection-fixture"] = SessionListProjection(
    "projection-fixture", "get_session_events", "events", "events"
)
retire_original = sys.argv[1] == "retire"
if retire_original:
    del SESSION_LIST_PROJECTIONS["events"]

import polylogue.archive.viewport as viewport
import polylogue.cli.read_view_handlers as handlers
import polylogue.cli.read_view_registry as registry
import polylogue.surfaces.projection_spec as projections
registry.validate_read_view_metadata_registry()
handlers.validate_read_view_handler_registry()
assert "projection-fixture" in viewport.read_view_choices()
profile = viewport.get_read_view_profile("projection-fixture")
assert profile.included_kinds == original_profile.included_kinds
assert profile.evidence_policy == original_profile.evidence_policy
assert registry.READ_VIEW_HANDLER_METADATA["projection-fixture"].operations == original_metadata.operations
assert handlers.READ_VIEW_HANDLERS["projection-fixture"].handler is original_handler.handler
assert handlers.READ_VIEW_HANDLERS["projection-fixture"].accepted_options == original_handler.accepted_options
assert projections.projection_from_view("projection-fixture").projection.families == original_family
assert profile.to_payload()["view_id"] == "projection-fixture"
for vocabulary in (viewport.READ_VIEW_PROFILE_BY_ID, registry.READ_VIEW_HANDLER_METADATA,
                   handlers.READ_VIEW_HANDLERS, projections.READ_VIEW_PROJECTION_FAMILIES):
    assert ("events" in vocabulary) is not retire_original
print("projection_contract_bound")
"""


def run_projection_contract_probe(*, retire_original: bool) -> subprocess.CompletedProcess[str]:
    """Run actual metadata imports and validation without replacing parent types."""
    return subprocess.run(
        [sys.executable, "-c", _PROJECTION_PROBE, "retire" if retire_original else "keep"],
        capture_output=True,
        text=True,
        check=False,
    )
