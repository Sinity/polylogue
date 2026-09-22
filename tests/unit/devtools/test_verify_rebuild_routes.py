"""The rebuild-route census must refuse a second derived-tier build path.

polylogue-b5l AC5 required that a derived tier is rebuilt only by ordinary
daemon convergence and cited ``devtools gate layering`` as its check. Measured
at 2b208175d that citation was false: an orchestration-only rebuild route added
under ``polylogue/maintenance/`` and awaiting
``pipeline.services.indexing.rebuild_index`` outside ``DaemonConverger`` left
the layering gate green (count 0, exit 0). Its writer census only fires when a
*new uncensused file executes its own DML*, so both realistic shapes of a
second build path -- orchestrating existing tier writers, and adding DML inside
an already-censused module -- passed it.

Anti-vacuity, stated as the mutations that must turn these tests red:

* Reduce the gate to "a new file bearing raw DML", the exact hole measured
  above, and ``test_an_orchestration_only_route_is_reported`` goes green
  wrongly: its probe module executes no SQL at all.
* Delete the upward-closure walk and keep only direct callers, and
  ``test_a_route_three_hops_from_the_entrypoint_is_reported`` goes green: that
  probe never names a rebuild entrypoint.
* Narrow the census to one module or one directory, and
  ``test_a_namespace_package_module_is_inspected`` goes green: its probe lives
  in a directory with no ``__init__.py``, the 85-module class that
  ``devtools/verify_layering.py`` documents grimp as silently dropping.
* Stop resolving declared entrypoints against real definitions, and
  ``test_a_renamed_entrypoint_refuses_instead_of_emptying_the_census`` goes
  green over an empty population -- a rename would otherwise disarm the gate
  in silence.
* Take ``route: daemon_convergence`` on trust, and
  ``test_an_unproven_daemon_convergence_claim_is_reported`` goes green. That
  token is the whole invariant; asserting it unchecked is what produced this
  bead.
* Let the declaration keep entries that no longer reproduce, and
  ``test_a_declaration_that_no_longer_reproduces_is_reported`` goes green,
  which is how a ratchet stops ratcheting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from devtools.gate import GATES_BY_NAME
from devtools.verify_rebuild_routes import (
    DECLARATION_PATH,
    ROUTE_VOCABULARY,
    Observation,
    build_call_graph,
    collect_violations,
    load_declaration,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

_ENTRYPOINT_MODULE = '''
"""Synthetic rebuild primitive."""


def rebuild_index(conn) -> None:
    conn.execute("DELETE FROM search_index")
'''

_STAGE_MODULE = '''
"""Synthetic daemon convergence stage list."""

from pkg.stages.owner import converge_owned_profiles


def make_default_convergence_stages():
    return (converge_owned_profiles,)
'''

_OWNER_MODULE = '''
"""Synthetic owner reached from the stage list."""

from pkg.rebuild import rebuild_index


def converge_owned_profiles(conn) -> None:
    rebuild_index(conn)
'''

_ORPHAN_MODULE = '''
"""Synthetic exported rebuild route with no caller."""

from pkg.rebuild import rebuild_index


def orphan_rebuild(conn) -> None:
    rebuild_index(conn)
'''


def _declaration(routes: list[dict[str, object]], *, entrypoints: list[str] | None = None) -> str:
    import yaml

    return yaml.safe_dump(
        {
            "package": "pkg",
            "entrypoints": entrypoints if entrypoints is not None else ["pkg.rebuild.rebuild_index"],
            "convergence_stage_seeds": ["pkg.stages.make_default_convergence_stages"],
            "routes": routes,
        }
    )


def _tree(tmp_path: Path, modules: dict[str, str]) -> Path:
    """Write a synthetic package and return its repository root."""
    for relative, body in modules.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return tmp_path


def _base_modules() -> dict[str, str]:
    return {
        "pkg/__init__.py": "",
        "pkg/rebuild.py": _ENTRYPOINT_MODULE,
        "pkg/stages/__init__.py": _STAGE_MODULE,
        "pkg/stages/owner.py": _OWNER_MODULE,
        "pkg/orphan.py": _ORPHAN_MODULE,
    }


def _owner_route(route: str = "daemon_convergence") -> dict[str, object]:
    return {
        "function": "pkg.stages.owner.converge_owned_profiles",
        "kinds": ["direct_caller"],
        "entrypoints": ["pkg.rebuild.rebuild_index"],
        "route": route,
        "reason": "reached from the declared stage list",
    }


def _stage_route() -> dict[str, object]:
    """The stage factory is itself the entry root of the legitimate path."""
    return {
        "function": "pkg.stages.make_default_convergence_stages",
        "kinds": ["entry_root"],
        "entrypoints": ["pkg.rebuild.rebuild_index"],
        "route": "daemon_convergence",
        "reason": "the declared convergence stage list",
    }


def _orphan_route() -> dict[str, object]:
    """An exported rebuild route with no caller -- the real tree's common shape."""
    return {
        "function": "pkg.orphan.orphan_rebuild",
        "kinds": ["direct_caller", "entry_root"],
        "entrypoints": ["pkg.rebuild.rebuild_index"],
        "route": "unreached",
        "reason": "exported and called by nothing at this head",
    }


def _base_routes() -> list[dict[str, object]]:
    return [_stage_route(), _owner_route(), _orphan_route()]


def _run(
    tmp_path: Path,
    modules: dict[str, str],
    routes: list[dict[str, object]],
    *,
    entrypoints: list[str] | None = None,
) -> tuple[list[dict[str, object]], Observation]:
    root = _tree(tmp_path, modules)
    declaration = root / "census.yaml"
    declaration.write_text(_declaration(routes, entrypoints=entrypoints), encoding="utf-8")
    violations, observation, _declaration_obj, _graph = collect_violations(repo_root=root, declaration_path=declaration)
    return violations, observation


def test_gate_is_registered_against_its_module() -> None:
    gate = GATES_BY_NAME["rebuild-routes"]
    assert gate.args == ("devtools.verify_rebuild_routes", "--json")
    assert gate.kind == "module"
    assert gate.in_quick is True
    assert gate.blocking is True


def test_a_fully_declared_tree_reports_nothing(tmp_path: Path) -> None:
    violations, observation = _run(tmp_path, _base_modules(), _base_routes())

    assert violations == []
    # Not vacuous: the entrypoint resolved and the closure actually found the owner.
    assert observation.unresolved_entrypoints == ()
    assert "pkg.stages.owner.converge_owned_profiles" in observation.direct_callers


def test_an_orchestration_only_route_is_reported(tmp_path: Path) -> None:
    """The exact shape measured at 2b208175d: no DML of its own, awaits a rebuild.

    ``devtools gate layering`` stayed green on this input. The probe module
    below contains no SQL string and no ``execute`` call, so a gate built on
    DML detection cannot see it.
    """
    modules = _base_modules()
    modules["pkg/maintenance/second_rebuild_route.py"] = (
        "from pkg.rebuild import rebuild_index\n\n\ndef run_second_rebuild(conn):\n    rebuild_index(conn)\n"
    )
    modules["pkg/maintenance/__init__.py"] = ""
    assert "execute" not in modules["pkg/maintenance/second_rebuild_route.py"]

    violations, _observation = _run(tmp_path, modules, _base_routes())

    undeclared = [item for item in violations if item["rule"] == "rebuild_route_undeclared"]
    assert [item["function"] for item in undeclared] == ["pkg.maintenance.second_rebuild_route.run_second_rebuild"]
    # The failure names the route *and* the entrypoint it reaches.
    assert undeclared[0]["entrypoints"] == ["pkg.rebuild.rebuild_index"]
    assert undeclared[0]["file"] == "pkg/maintenance/second_rebuild_route.py"
    assert sorted(cast("list[str]", undeclared[0]["kinds"])) == ["direct_caller", "entry_root"]


def test_a_route_three_hops_from_the_entrypoint_is_reported(tmp_path: Path) -> None:
    """A second path that never names a rebuild entrypoint anywhere.

    A direct-caller census alone cannot see this. The upward closure reports
    the new entry root, and reports that the intermediate route it now calls
    stopped being a root.
    """
    modules = _base_modules()
    modules["pkg/maintenance/__init__.py"] = ""
    modules["pkg/maintenance/indirect.py"] = (
        "from pkg.orphan import orphan_rebuild\n\n\n"
        "def _hop_two(conn):\n    return orphan_rebuild(conn)\n\n\n"
        "def _hop_one(conn):\n    return _hop_two(conn)\n\n\n"
        "def run_second_rebuild(conn):\n    return _hop_one(conn)\n"
    )
    assert "rebuild_index" not in modules["pkg/maintenance/indirect.py"]

    violations, _observation = _run(tmp_path, modules, _base_routes())

    by_rule = {item["rule"]: item for item in violations}
    assert by_rule["rebuild_route_undeclared"]["function"] == "pkg.maintenance.indirect.run_second_rebuild"
    assert by_rule["rebuild_route_undeclared"]["entrypoints"] == ["pkg.rebuild.rebuild_index"]
    # The orphan it reaches through is no longer an entry root, and the
    # declaration says it is -- a second consumer of one route is visible too.
    assert by_rule["rebuild_route_kind_drift"]["function"] == "pkg.orphan.orphan_rebuild"


def test_a_namespace_package_module_is_inspected(tmp_path: Path) -> None:
    """A directory with no ``__init__.py`` is censused like any other.

    ``devtools/verify_layering.py`` documents that grimp silently drops the 85
    namespace-package modules in this checkout. A census built on that graph
    would report every parser as reaching nothing.
    """
    modules = _base_modules()
    modules["pkg/parsers/rogue.py"] = (
        "from pkg.rebuild import rebuild_index\n\n\ndef reparse_and_rebuild(conn):\n    rebuild_index(conn)\n"
    )
    assert not (tmp_path / "pkg" / "parsers" / "__init__.py").exists()

    violations, _observation = _run(tmp_path, modules, _base_routes())

    assert [item["function"] for item in violations if item["rule"] == "rebuild_route_undeclared"] == [
        "pkg.parsers.rogue.reparse_and_rebuild"
    ]


def test_a_fully_qualified_call_is_reported(tmp_path: Path) -> None:
    """``import pkg.rebuild`` then ``pkg.rebuild.rebuild_index(conn)``.

    This is ordinary Python, and it is an ``ast.Attribute`` whose ``value`` is
    another ``ast.Attribute``. Reading only the chains rooted directly at an
    ``ast.Name`` saw ``pkg.rebuild`` -- the intermediate module -- and never
    the function, so a second rebuild route written with a fully qualified
    import kept this blocking gate green.
    """
    modules = _base_modules()
    modules["pkg/maintenance/__init__.py"] = ""
    modules["pkg/maintenance/qualified.py"] = (
        "import pkg.rebuild\n\n\ndef run_qualified_rebuild(conn):\n    pkg.rebuild.rebuild_index(conn)\n"
    )
    # The route names no symbol this census could pick up from a ``from``
    # import: the only mention of the entrypoint is the attribute chain.
    assert "from pkg.rebuild import" not in modules["pkg/maintenance/qualified.py"]

    violations, observation = _run(tmp_path, modules, _base_routes())

    assert "pkg.maintenance.qualified.run_qualified_rebuild" in observation.direct_callers
    undeclared = [item for item in violations if item["rule"] == "rebuild_route_undeclared"]
    assert [item["function"] for item in undeclared] == ["pkg.maintenance.qualified.run_qualified_rebuild"]
    assert undeclared[0]["entrypoints"] == ["pkg.rebuild.rebuild_index"]


def test_a_local_import_does_not_erase_another_route(tmp_path: Path) -> None:
    """Two functions, one name, two different imports.

    ``harmless`` locally imports an unrelated ``rebuild_index``. When every
    import in a module was merged into one dictionary the later binding won
    module-wide, ``real_rebuild`` lost its edge to the entrypoint, and the
    census reported no new route for a caller that reaches it.
    """
    modules = _base_modules()
    modules["pkg/noop.py"] = (
        '"""Unrelated function that shares a name."""\n\n\ndef rebuild_index(conn) -> None:\n    return None\n'
    )
    modules["pkg/shadow.py"] = (
        "def real_rebuild(conn):\n"
        "    from pkg.rebuild import rebuild_index\n\n"
        "    return rebuild_index(conn)\n\n\n"
        "def harmless(conn):\n"
        "    from pkg.noop import rebuild_index\n\n"
        "    return rebuild_index(conn)\n"
    )
    # Source order matters: the shadowing import is the LAST one in the file,
    # which is what a module-wide merge would keep.
    body = modules["pkg/shadow.py"]
    assert body.index("pkg.noop") > body.index("pkg.rebuild")

    violations, observation = _run(tmp_path, modules, _base_routes())

    assert "pkg.shadow.real_rebuild" in observation.direct_callers
    # The opposite direction is pinned too: scoping must not invent an edge
    # for the function that imported the harmless name.
    assert "pkg.shadow.harmless" not in observation.direct_callers
    undeclared = [item for item in violations if item["rule"] == "rebuild_route_undeclared"]
    assert [item["function"] for item in undeclared] == ["pkg.shadow.real_rebuild"]


def test_a_route_reached_through_a_package_reexport_is_resolved(tmp_path: Path) -> None:
    """Importing through a package ``__init__`` must not hide the route."""
    modules = _base_modules()
    modules["pkg/facade/__init__.py"] = "from pkg.orphan import orphan_rebuild\n"
    modules["pkg/surface.py"] = (
        "from pkg.facade import orphan_rebuild\n\n\ndef surface_rebuild(conn):\n    return orphan_rebuild(conn)\n"
    )

    violations, _observation = _run(tmp_path, modules, _base_routes())

    assert "pkg.surface.surface_rebuild" in {
        item["function"] for item in violations if item["rule"] == "rebuild_route_undeclared"
    }


def test_a_renamed_entrypoint_refuses_instead_of_emptying_the_census(tmp_path: Path) -> None:
    """A declared entrypoint that resolves to nothing is a finding, not silence."""
    violations, observation = _run(
        tmp_path,
        _base_modules(),
        [],
        entrypoints=["pkg.rebuild.rebuild_index_renamed"],
    )

    assert observation.direct_callers == frozenset()
    assert [item["rule"] for item in violations if item["rule"] == "rebuild_entrypoint_unresolved"] == [
        "rebuild_entrypoint_unresolved"
    ]


def test_a_declaration_that_no_longer_reproduces_is_reported(tmp_path: Path) -> None:
    routes: list[dict[str, object]] = [
        *_base_routes(),
        {
            "function": "pkg.retired.gone",
            "kinds": ["entry_root"],
            "entrypoints": ["pkg.rebuild.rebuild_index"],
            "route": "unreached",
            "reason": "retired last week",
        },
    ]

    violations, _observation = _run(tmp_path, _base_modules(), routes)

    assert [item["function"] for item in violations if item["rule"] == "rebuild_route_census_stale"] == [
        "pkg.retired.gone"
    ]


def test_an_unproven_daemon_convergence_claim_is_reported(tmp_path: Path) -> None:
    """The token the invariant rests on is the one the gate will not take on trust."""
    modules = _base_modules()
    modules["pkg/maintenance/__init__.py"] = ""
    modules["pkg/maintenance/second_rebuild_route.py"] = (
        "from pkg.rebuild import rebuild_index\n\n\ndef run_second_rebuild(conn):\n    rebuild_index(conn)\n"
    )
    routes: list[dict[str, object]] = [
        *_base_routes(),
        {
            "function": "pkg.maintenance.second_rebuild_route.run_second_rebuild",
            "kinds": ["direct_caller", "entry_root"],
            "entrypoints": ["pkg.rebuild.rebuild_index"],
            "route": "daemon_convergence",
            "reason": "claims to be a convergence stage",
        },
    ]

    violations, observation = _run(tmp_path, modules, routes)

    assert [item["function"] for item in violations] == ["pkg.maintenance.second_rebuild_route.run_second_rebuild"]
    assert violations[0]["rule"] == "rebuild_route_daemon_claim_unproven"
    # The owner that really is on the stage list keeps its claim, so the rule
    # rejects the unprovable claim rather than every claim.
    assert "pkg.stages.owner.converge_owned_profiles" in observation.daemon_reachable


def test_an_unknown_route_or_missing_reason_is_reported(tmp_path: Path) -> None:
    routes = [_stage_route(), _orphan_route(), {**_owner_route(route="because_i_said_so"), "reason": ""}]

    violations, _observation = _run(tmp_path, _base_modules(), routes)

    assert {item["rule"] for item in violations} == {
        "rebuild_route_unknown_classification",
        "rebuild_route_reason_missing",
    }
    assert set(ROUTE_VOCABULARY) >= {"daemon_convergence", "daemon_operation", "fixture_seeding", "unreached"}


def test_a_function_local_import_is_followed(tmp_path: Path) -> None:
    """Rebuild routes are routinely imported inside the function that uses them."""
    modules = _base_modules()
    modules["pkg/maintenance/__init__.py"] = ""
    modules["pkg/maintenance/lazy.py"] = (
        "def run_second_rebuild(conn):\n    from pkg.rebuild import rebuild_index\n\n    rebuild_index(conn)\n"
    )

    violations, _observation = _run(tmp_path, modules, _base_routes())

    assert [item["function"] for item in violations if item["rule"] == "rebuild_route_undeclared"] == [
        "pkg.maintenance.lazy.run_second_rebuild"
    ]


def test_the_live_declaration_matches_this_checkout(capsys: pytest.CaptureFixture[str]) -> None:
    """The real census reproduces, and it is not an empty population."""
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["violations"] == []
    declaration = load_declaration(REPO_ROOT / DECLARATION_PATH)
    graph = build_call_graph(REPO_ROOT / "polylogue", repo_root=REPO_ROOT)
    assert declaration.entrypoints, "the declaration must name at least one entrypoint"
    assert all(name in graph.defined for name in declaration.entrypoints)
    assert payload["direct_callers"] and payload["entry_roots"]
    assert set(payload["direct_callers"]) | set(payload["entry_roots"]) == set(declaration.entries)


def test_no_live_route_is_reachable_from_the_daemon_stage_list(capsys: pytest.CaptureFixture[str]) -> None:
    """The substantive finding this gate was built to record.

    At this head no rebuild entrypoint is structurally reached from
    ``make_default_convergence_stages``: the daemon's session-profile work runs
    through ``SessionProfileDerivation`` and the shared owner, not through
    ``rebuild_session_insights_sync``. Every declared route is therefore a demo
    seed, a daemon-hosted demo operation, or unreached. This test fails the day
    that changes, which is the day the declaration needs re-reading.
    """
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["daemon_stage_reachable"] == []
    declaration = load_declaration(REPO_ROOT / DECLARATION_PATH)
    assert {entry.route for entry in declaration.entries.values()} == {
        "fixture_seeding",
        "daemon_operation",
        "unreached",
    }
