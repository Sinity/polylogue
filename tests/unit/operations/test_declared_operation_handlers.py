"""Every module-dispatched operation handler resolves on its declared module.

polylogue-ms4uy: ``maintenance.insights.rebuild`` named a handler that lives on
``daemon_insights`` while dispatch resolved handlers on ``daemon_mutations``,
so the declaration was only safe while an unrelated special case kept it from
being dispatched.

Anti-vacuity: pointing any spec's ``handler``/``handler_module`` at a function
that does not exist on that module (e.g. reverting the rebuild spec to the
default module) makes ``validate_declared_handlers`` raise and this test red.
"""

from __future__ import annotations

import pytest

from polylogue.operations.daemon_protocol import (
    daemon_operation_spec,
    module_dispatched_specs,
    resolve_operation_handler,
    validate_declared_handlers,
)


def test_every_module_dispatched_handler_resolves() -> None:
    validate_declared_handlers()
    assert {spec.name for spec in module_dispatched_specs()} >= {
        "ingest",
        "maintenance.insights.rebuild",
        "mutation.session.mark",
    }


def test_insights_rebuild_resolves_to_its_owning_module() -> None:
    spec = daemon_operation_spec("maintenance.insights.rebuild")
    assert spec is not None
    handler = resolve_operation_handler(spec)
    assert handler.__module__ == "polylogue.operations.daemon_insights"
    assert handler.__name__ == "execute_insights_rebuild_operation"


def test_unresolvable_handler_is_refused() -> None:
    spec = daemon_operation_spec("maintenance.insights.rebuild")
    assert spec is not None
    broken = type(spec)(
        spec.name,
        spec.authority,
        spec.fallback,
        capability=spec.capability,
        request_model=spec.request_model,
        result_model=spec.result_model,
        handler="execute_insights_rebuild_operation",
    )
    with pytest.raises(RuntimeError, match="does not exist on"):
        resolve_operation_handler(broken)
