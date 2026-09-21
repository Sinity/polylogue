"""Seam A: the only module that turns Click parameter names into operations.

Every root-query capability the CLI serves is a declared operation, so the
adapter's whole job on the request side is to name the operation and project
the parsed argv onto its payload.  Concentrating that here is what makes the
claim "the CLI has one query implementation" checkable: a Click parameter name
appearing anywhere below this seam would be a second, unowned lowering.

Import-light on purpose — ``click`` for the typed refusals, the operation
kernel for the request type, and the declared parameter vocabulary.  No
archive reader, no storage, no daemon server.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import click

from polylogue.cli.operation_kernel import OperationRequest

if TYPE_CHECKING:
    from polylogue.archive.query.expression import WithUnitWindow
    from polylogue.cli.root_request import RootModeRequest

#: The declared bound on one completion request (``CompletionRequest.limit``
#: is ``ge=1, le=200``).  Restated here rather than imported because this
#: module is on the CLI's coldest path and must not pull in the Pydantic
#: request models to lower a TAB press; the registry test proves the two
#: agree.
COMPLETION_LIMIT_BOUNDS: tuple[int, int] = (1, 200)

__all__ = [
    "AGGREGATE_MODE_PARAMS",
    "COMPLETION_LIMIT_BOUNDS",
    "aggregate_mode",
    "desugar_cli_retrieval_lane",
    "lower_cli_query",
    "lower_completion",
    "lower_query_aggregate",
    "lower_query_units",
    "lower_session_read",
    "lower_session_reference",
]


# ``stats_by`` is checked first so ``--by`` wins over a bare ``--stats``; the
# root callback allows both to be set and the grouped answer is the specific one.
AGGREGATE_MODE_PARAMS: tuple[tuple[str, str], ...] = (
    ("stats_by", "stats_by"),
    ("stats_only", "stats"),
    ("count_only", "count"),
)


def desugar_cli_retrieval_lane(params: dict[str, object], query_terms: Sequence[str]) -> tuple[str, ...]:
    """Fold the CLI-only ``--retrieval-lane semantic`` spelling into the declared vocabulary.

    ``semantic`` is a CLI spelling, not a declared lane: ``QUERY_RETRIEVAL_LANES``
    has no such member, and the handler's ``normalize_retrieval_lane`` refuses
    it.  It means "rank these terms by similarity", which the declared
    vocabulary expresses as the ``auto`` lane with the terms promoted into
    ``similar_text``.  Applying it here — the one place that knows CLI
    spellings — is what keeps ``--semantic`` working now that the request is
    compiled by the operation rather than by the adapter.

    Returns the query terms that survive: a promoted similarity prompt leaves
    none, or the phrase would also be searched literally.
    """

    if params.get("retrieval_lane") != "semantic":
        return tuple(query_terms)
    params["retrieval_lane"] = "auto"
    if not params.get("similar_text"):
        prompt = " ".join(term for term in query_terms if term).strip()
        if prompt:
            params["similar_text"] = prompt
    return ()


# Presentation-only keys never reach a selection payload.  They do not change
# which rows are selected, and forwarding them would vary the operation's
# result-cache key for two requests that must share one answer.
_SELECTION_EXCLUDED = frozenset({"exclude_text"})


def aggregate_mode(params: Mapping[str, object]) -> str | None:
    """Name the aggregate this request asks for, or ``None`` for a page."""

    for param, mode in AGGREGATE_MODE_PARAMS:
        if params.get(param):
            return mode
    return None


def _selection_params(request: RootModeRequest) -> dict[str, object]:
    """Project the root request onto the declared selection vocabulary.

    The handler compiles its spec with the same ``SessionQuerySpec.from_params``
    the CLI used to call locally, so forwarding the recognised parameter set
    verbatim — rather than a hand-maintained rename table — is what makes the
    two routes compile *one* spec.  A rename table previously dropped
    ``--has-paste``/``--has-tool-use``/``--has-thinking`` silently and
    re-tokenised quoted phrases by joining the query terms into one string.

    ``exclude_text`` is deliberately still withheld.  It is a content
    post-filter the two routes answered differently — the operation applies it
    to a list page and, on a ranked page, to the count but not the hits, while
    the local branch ignored it everywhere — so forwarding it would change
    ``find --exclude-text`` results before anyone has decided which of those
    three answers is right (polylogue-v1mnm).
    """

    # Imported under its private name deliberately: ``archive/query/spec.py``
    # is IN the derived-schema identity closure, and adding a public alias
    # there would move the identity (and invalidate a rebuild) for a rename.
    from polylogue.archive.query.spec import _RECOGNIZED_PARAMS

    return {
        key: value
        for key, value in request.params.items()
        if key in _RECOGNIZED_PARAMS and key not in _SELECTION_EXCLUDED and value is not None and value not in ((), [])
    }


def _with_projection_params(
    *,
    with_units: Sequence[str],
    with_unit_fields: Mapping[str, Sequence[str]],
    with_unit_windows: Mapping[str, WithUnitWindow],
) -> dict[str, object]:
    """Encode a ``with <units>`` projection as declared wire values.

    ``WithUnitWindow`` is a parser dataclass, not a transport type; the handler
    rebuilds it from exactly this shape, so the encoding is stated once here
    and decoded once there.
    """

    if not with_units:
        return {}
    encoded: dict[str, object] = {"with_units": list(with_units)}
    if with_unit_fields:
        encoded["with_unit_fields"] = {unit: list(fields) for unit, fields in with_unit_fields.items()}
    if with_unit_windows:
        encoded["with_unit_windows"] = {
            unit: {
                "predicates": dict(window.predicates),
                "window": ({"kind": window.window[0], "n": window.window[1]} if window.window else None),
            }
            for unit, window in with_unit_windows.items()
        }
    return encoded


def lower_cli_query(
    request: RootModeRequest,
    *,
    limit: int,
    offset: int,
    sample: int | None = None,
    with_units: Sequence[str] = (),
    with_unit_fields: Mapping[str, Sequence[str]] | None = None,
    with_unit_windows: Mapping[str, WithUnitWindow] | None = None,
) -> OperationRequest:
    """Lower one root session page onto ``cli.query``."""

    params = _selection_params(request)
    params["query"] = list(desugar_cli_retrieval_lane(params, request.query_terms))
    params["limit"] = limit
    params["offset"] = offset
    if sample is not None:
        # ``sample`` names its own page size; carrying the page coordinates too
        # would let the handler and the adapter disagree about which won.
        params["sample"] = sample
        params["offset"] = 0
    params.update(
        _with_projection_params(
            with_units=with_units,
            with_unit_fields=with_unit_fields or {},
            with_unit_windows=with_unit_windows or {},
        )
    )
    return OperationRequest("cli.query", {"params": params})


def lower_query_aggregate(request: RootModeRequest, *, mode: str) -> OperationRequest:
    """Lower ``analyze --count`` / ``--stats`` / ``--by`` onto ``query.aggregate``.

    The aggregate reads the same selection vocabulary as a page, so the two
    share one projection: an aggregate and the page it summarises can never
    disagree about which sessions were selected.
    """

    params = _selection_params(request)
    params["query"] = list(desugar_cli_retrieval_lane(params, request.query_terms))
    payload: dict[str, object] = {"mode": mode, "params": params}
    if mode == "stats_by":
        group_by = str(request.params.get("stats_by") or "").strip()
        if not group_by:
            raise click.UsageError("Root query --by requires a grouping field.")
        payload["group_by"] = group_by
    return OperationRequest("query.aggregate", payload)


def lower_session_read(
    ref: str,
    *,
    kind: str = "transcript",
    limit: int | None = None,
    offset: int = 0,
    projection: Mapping[str, object] | None = None,
    continuation: str | None = None,
) -> OperationRequest:
    """Lower one bounded read for an exact reference onto ``session.read``.

    A continuation supersedes the window coordinates it was minted from, so
    passing both is a caller error rather than a silently ignored argument.
    Evidence kinds are answered whole and take no window at all; passing one
    is a caller error for the same reason.

    The windowed kinds are named by the operation contract rather than spelled
    again here: a kind that graduates to a window would otherwise keep being
    refused a window by this adapter.
    """

    from polylogue.operations.read_contracts import WINDOWED_SESSION_READ_KINDS

    if continuation is not None and (limit is not None or offset):
        raise click.UsageError("A transcript continuation already carries its window coordinates.")
    if kind not in WINDOWED_SESSION_READ_KINDS and (limit is not None or offset or continuation is not None):
        raise click.UsageError(f"A {kind} read is answered whole and takes no window coordinates.")
    payload: dict[str, object] = {"ref": ref}
    if kind != "transcript":
        payload["kind"] = kind
    if limit is not None:
        payload["limit"] = limit
    if offset:
        payload["offset"] = offset
    if projection:
        payload["projection"] = dict(projection)
    if continuation is not None:
        payload["continuation"] = continuation
    return OperationRequest("session.read", payload)


def lower_completion(source: str, incomplete: str, *, limit: int) -> OperationRequest:
    """Lower one archive-backed value completion onto ``completion``.

    The bound is clamped into the declared request range instead of being
    forwarded verbatim.  A completer that asks for more candidates than the
    contract admits is refused by the request model, and a completer has no
    channel to report a refusal on — so an out-of-range ``limit`` reached the
    shell as an empty candidate list, which reads as "the archive has no
    matching values".  That is the same lie the module's daemon-absent message
    exists to avoid, so the one place that knows the CLI's bounds fixes it
    here rather than letting each completer guess.
    """

    low, high = COMPLETION_LIMIT_BOUNDS
    return OperationRequest(
        "completion",
        {"source": source, "incomplete": incomplete, "limit": min(high, max(low, int(limit)))},
    )


def lower_session_reference(expression: str, *, limit: int | None = None) -> OperationRequest:
    """Lower a bare ``from <ref>`` root onto ``session.reference``."""

    payload: dict[str, object] = {"expression": expression}
    if limit is not None and limit >= 0:
        payload["limit"] = limit
    return OperationRequest("session.reference", payload)


_UNIT_FILTER_PARAMS: tuple[str, ...] = (
    "contains",
    "origin",
    "exclude_origin",
    "tag",
    "exclude_tag",
    "repo",
    "has_type",
    "tool",
    "exclude_tool",
    "action",
    "exclude_action",
    "action_sequence",
    "action_text",
    "referenced_path",
    "cwd_prefix",
    "title",
    "min_messages",
    "max_messages",
    "min_words",
    "max_words",
    "since",
    "until",
)

_UNIT_FLAG_PARAMS: tuple[tuple[str, str], ...] = (
    ("has_paste", "has_paste_evidence"),
    ("has_tool_use", "has_tool_use"),
    ("has_thinking", "has_thinking"),
)


def lower_query_units(
    request: RootModeRequest,
    *,
    expression: str,
    limit: int,
    offset: int,
) -> OperationRequest:
    """Lower a ``<unit> where ...`` root onto ``query.units``.

    This list stays explicit rather than reusing the session vocabulary:
    ``query.units`` forwards its filters as keyword arguments to
    ``query_unit_request``, so an unrecognised key is a ``TypeError`` there
    rather than an ignored parameter.
    """

    params = request.params
    query_params: dict[str, object] = {"limit": limit, "offset": offset, "expression": expression}
    raw_query = " ".join(term for term in request.query_terms if term).strip()
    if raw_query:
        query_params["query"] = raw_query
    for key in _UNIT_FILTER_PARAMS:
        value = params.get(key)
        if value is not None and value not in ("", (), []):
            query_params[key] = value
    for source_key, dest_key in _UNIT_FLAG_PARAMS:
        if params.get(source_key):
            query_params[dest_key] = "1"
    return OperationRequest("query.units", {"params": query_params})
