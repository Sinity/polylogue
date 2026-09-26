"""Executable declaration coverage for workspace and maintenance routes."""

from __future__ import annotations

from polylogue.daemon.route_contracts import DAEMON_ROUTE_DECLARATIONS
from polylogue.daemon.route_families.maintenance import ROUTES as MAINTENANCE_ROUTES
from polylogue.daemon.route_families.workspace import ROUTES as WORKSPACE_ROUTES
from polylogue.declarations.validation import validate_declaration


def test_workspace_route_family_binds_the_existing_http_handlers() -> None:
    routes = {route.path: route for route in WORKSPACE_ROUTES}

    assert set(routes) == {"/api/stack", "/api/compare"}
    assert set(WORKSPACE_ROUTES).issubset(DAEMON_ROUTE_DECLARATIONS)
    assert tuple(route.method for route in WORKSPACE_ROUTES) == ("GET", "GET")
    assert {route.path: route.kernel.handlers[0].symbol for route in WORKSPACE_ROUTES} == {
        "/api/stack": "_handle_stack",
        "/api/compare": "_handle_compare",
    }
    assert all(route.kernel.handlers[0].binding_key == f"GET {route.path}" for route in routes.values())
    assert all(route.domain_operation is None for route in routes.values())
    assert all(route.kind == "workspace" and route.stability == "shell_supported" for route in routes.values())
    assert all(route.auth_policy == "credential_if_configured" for route in routes.values())


def test_maintenance_route_family_names_only_the_operation_it_executes() -> None:
    routes = {route.path: route for route in MAINTENANCE_ROUTES}

    assert set(routes) == {"/api/reset", "/api/ingest", "/api/demo/augment"}
    assert set(MAINTENANCE_ROUTES).issubset(DAEMON_ROUTE_DECLARATIONS)
    assert {route.path: route.kernel.handlers[0].symbol for route in MAINTENANCE_ROUTES} == {
        "/api/reset": "_handle_reset",
        "/api/ingest": "_handle_ingest",
        "/api/demo/augment": "_handle_demo_augment",
    }
    assert all(route.method == "POST" and not route.passes_params for route in routes.values())
    assert all(route.kernel.handlers[0].binding_key == f"POST {route.path}" for route in routes.values())
    assert {path: route.domain_operation for path, route in routes.items()} == {
        "/api/reset": None,
        "/api/ingest": "ingest",
        "/api/demo/augment": None,
    }
    assert {path: (route.kind, route.stability) for path, route in routes.items()} == {
        "/api/reset": ("maintenance", "stable"),
        "/api/ingest": ("maintenance", "stable"),
        "/api/demo/augment": ("maintenance", "operational"),
    }
    assert all(route.auth_policy == "bearer_if_configured_and_same_origin" for route in routes.values())
    assert routes["/api/reset"].write_gate
    assert not routes["/api/ingest"].write_gate and not routes["/api/demo/augment"].write_gate
    assert all(route.migration_reason for route in (routes["/api/reset"], routes["/api/demo/augment"]))


def test_workspace_and_maintenance_declarations_have_complete_consumer_edges() -> None:
    for route in (*WORKSPACE_ROUTES, *MAINTENANCE_ROUTES):
        assert validate_declaration(route.kernel) == ()
