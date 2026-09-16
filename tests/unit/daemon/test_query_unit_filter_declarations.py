"""The declared terminal filter surface must equal the live one.

``TERMINAL_FILTER_PARAMETERS`` is the single declaration behind the generated
``/api/query-units`` OpenAPI parameter list. Nothing else enumerates the names
that ``query_unit_session_filters(**params)`` accepts, so these tests resolve
every declared row against the two live routes it claims:

* the handler call in ``DaemonAPIHandler._handle_query_units`` (which public
  name is read, and which ``query_unit_request`` keyword it is passed as), and
* the real filter lowering (which ``session_filters`` key it reaches).

Anti-vacuity: deleting a declared row, renaming a public parameter in the
handler, adding a handler parameter without declaring it, or pointing a row at
the wrong ``filter_key`` each turns one of these red -- and a row that reached
no filter would make the generated OpenAPI advertise a dead parameter.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from polylogue.archive.message.types import MessageType
from polylogue.archive.query.spec import QUERY_ACTION_TYPES
from polylogue.archive.query.unit_results import (
    TERMINAL_FILTER_PARAMETERS,
    query_unit_session_filters,
)

_PARAM_READERS = {"_get_param", "_get_bool", "_get_int", "_csv_values"}


def _handler_filter_arguments() -> dict[str, str]:
    """Map public query-parameter name -> ``query_unit_request`` keyword.

    Read from the live handler source rather than a second copied list, so the
    declaration is checked against the route that actually serves requests.
    """

    from polylogue.daemon import http as daemon_http

    source = Path(inspect.getsourcefile(daemon_http) or "").read_text(encoding="utf-8")
    module = ast.parse(source)
    handler: ast.FunctionDef | None = None
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == "_handle_query_units":
            handler = node
            break
    assert handler is not None, "daemon http no longer defines _handle_query_units"

    arguments: dict[str, str] = {}
    for call in (node for node in ast.walk(handler) if isinstance(node, ast.Call)):
        if not (isinstance(call.func, ast.Name) and call.func.id == "query_unit_request"):
            continue
        for keyword in call.keywords:
            if keyword.arg in {None, "expression", "limit", "offset", "session_filters"}:
                continue
            value = keyword.value
            if not isinstance(value, ast.Call):
                continue
            func = value.func
            reader = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if reader not in _PARAM_READERS:
                continue
            literals = [arg.value for arg in value.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            if not literals:
                continue
            assert keyword.arg is not None
            arguments[literals[-1]] = keyword.arg
    return arguments


def test_declared_filter_parameters_match_the_live_handler_surface() -> None:
    """Mutation: renaming, adding or dropping a handler query parameter makes this red."""
    live = _handler_filter_arguments()
    assert live, "no terminal filter parameters were recovered from the handler"

    declared = {parameter.name: parameter.request_kwarg for parameter in TERMINAL_FILTER_PARAMETERS}
    assert declared == live


def test_every_declared_filter_parameter_reaches_its_declared_filter_key() -> None:
    """Mutation: a declared row whose filter_key is wrong, or whose name no longer reaches
    the lowering, produces an unchanged default and makes this red."""
    probes: dict[str, object] = {"string": "declaration-probe", "integer": "7", "boolean": "true"}
    baseline = query_unit_session_filters()

    for parameter in TERMINAL_FILTER_PARAMETERS:
        probe = probes[parameter.kind]
        if parameter.name in {"since", "until"}:
            probe = "2026-01-02"
        elif parameter.name == "message_type":
            probe = MessageType.TOOL_USE.value
        elif "action" in parameter.filter_key:
            # These normalizers validate against the declared action vocabulary.
            probe = sorted(QUERY_ACTION_TYPES)[0]
        lowered = query_unit_session_filters(**{parameter.request_kwarg: probe})
        assert parameter.filter_key in lowered, f"{parameter.name}: undeclared session_filters key"
        assert lowered[parameter.filter_key] != baseline[parameter.filter_key], (
            f"{parameter.name}: does not reach session_filters[{parameter.filter_key!r}]"
        )
