"""Canonical archive reads for the daemon machine-operation protocol.

The daemon opens and pins the reader.  This module deliberately receives that
reader instead of a root/configuration so an operation cannot accidentally
re-open whichever archive generation happens to be current.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import monotonic
from typing import TYPE_CHECKING, Literal, cast

from polylogue.operations.authority import authority_for_reader

if TYPE_CHECKING:
    from polylogue.archive.query.search_contract import LaneFailure
    from polylogue.config import Config, PolylogueConfig
    from polylogue.core.protocols import VectorProvider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


@dataclass(frozen=True, slots=True)
class VectorReadBinding:
    """Pinned configuration needed to bind one supplied vector snapshot."""

    voyage_key: str
    model: str
    dimension: int

    def provider_for_snapshot(self, connection: sqlite3.Connection) -> VectorProvider:
        """Create a reader over ``connection`` without resolving any path."""

        from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider

        return SqliteVecProvider.from_vector_read_snapshot(
            voyage_key=self.voyage_key,
            connection=connection,
            model=self.model,
            dimension=self.dimension,
        )


def vector_binding_from_config(config: Config) -> VectorReadBinding | None:
    """Project only the already-resolved vector settings from ``config``.

    This deliberately does not call the legacy provider factory: that factory
    can read configuration and resolve archive paths as fallbacks, neither of
    which is valid after an operation reader has been pinned.
    """

    index_config = config.index_config
    voyage_key = index_config.voyage_api_key if index_config is not None else None
    if not voyage_key:
        return None
    return VectorReadBinding(
        voyage_key=voyage_key,
        model=config.embedding_model,
        dimension=config.embedding_dimension,
    )


@dataclass(frozen=True, slots=True)
class DaemonReadDependencies:
    """Explicit non-SQL dependencies resolved by the daemon operation context.

    ``vector_failure`` records a failed/unavailable provider construction so a
    hybrid query retains its lexical answer with a named gap.  Provider
    construction never belongs to this execution seam: it would otherwise
    consult defaults or open a second generation outside the pinned snapshot.
    """

    vector_binding: VectorReadBinding | None = None
    vector_connection: sqlite3.Connection | None = None
    vector_failure: LaneFailure | None = None
    runtime_status: Mapping[str, object] | None = None
    status_now_ms: int | None = None
    status_config: Config | PolylogueConfig | None = None

    def with_vector_snapshot(self, connection: sqlite3.Connection | None) -> DaemonReadDependencies:
        """Bind the config-only vector dependency to one pinned SQL handle."""

        return replace(self, vector_connection=connection)

    @property
    def vector_provider(self) -> VectorProvider | None:
        if self.vector_binding is None or self.vector_connection is None:
            return None
        return self.vector_binding.provider_for_snapshot(self.vector_connection)


def execute_read_operation(
    name: str,
    payload: dict[str, object],
    *,
    archive: ArchiveStore,
    serving_identity: str,
    dependencies: DaemonReadDependencies | None = None,
) -> dict[str, object]:
    """Execute one declared read against ``archive``'s already-pinned snapshot."""

    dependencies = dependencies or DaemonReadDependencies()
    if name == "cli.query":
        params = _params(payload)
        return _query_payload(params, archive=archive, serving_identity=serving_identity, dependencies=dependencies)
    if name == "query.units":
        params = _params(payload)
        return _query_units_payload(params, archive=archive, serving_identity=serving_identity)
    if name == "completion":
        return _completion_payload(payload)
    if name == "facets":
        return _facets_payload(_params(payload), archive=archive)
    if name == "status":
        if dependencies.status_now_ms is None:
            raise ValueError("status operation requires an operation-captured now_ms")
        from polylogue.operations.daemon_status import produce_operation_status

        return produce_operation_status(
            archive=archive,
            now_ms=dependencies.status_now_ms,
            config=dependencies.status_config,
            runtime_status=dependencies.runtime_status,
            include_archive_readiness=_truthy(payload.get("include_archive_readiness"))
            or _truthy(_params(payload).get("include_archive_readiness")),
        )
    raise ValueError(f"read operation is not declared: {name}")


def requires_vector_snapshot(name: str, payload: Mapping[str, object]) -> bool:
    """Return whether this declared read needs a coherent vector handle."""

    if name != "cli.query":
        return False
    spec = _cli_query_spec(_params(payload))
    return bool(spec.similar_text or spec.similar_session_id or spec.retrieval_lane == "hybrid")


def _params(payload: Mapping[str, object]) -> dict[str, object]:
    raw = payload.get("params", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("read params must be an object")
    return {str(key): value for key, value in raw.items()}


def _query_payload(
    params: Mapping[str, object],
    *,
    archive: ArchiveStore,
    serving_identity: str,
    dependencies: DaemonReadDependencies,
) -> dict[str, object]:
    from polylogue.api.archive import _archive_count_sessions_for_spec, _archive_list_summaries_for_spec
    from polylogue.archive.hydration import archive_summary_to_domain
    from polylogue.archive.query.expression import compile_expression_into
    from polylogue.archive.query.spec import (
        DEFAULT_SESSION_LIST_LIMIT,
        SessionQuerySpec,
        clamp_query_limit,
        session_count_unit_label,
    )
    from polylogue.surfaces.outcome import decide_outcome
    from polylogue.surfaces.payloads import session_list_envelope_from_summary

    normalized, expression = _lower_cli_query_params(params)
    limit = clamp_query_limit(normalized.get("limit"), default=DEFAULT_SESSION_LIST_LIMIT)
    offset = _non_negative_int(normalized.get("offset"), default=0)
    # CLI root payloads retain presentation-only keys.  The existing query
    # contract intentionally ignores those while compiling selection intent.
    base = SessionQuerySpec.from_params({**normalized, "limit": limit, "offset": offset})
    spec = compile_expression_into(expression, base) if expression else base

    if (
        spec.query_terms
        or spec.contains_terms
        or spec.similar_text
        or spec.similar_session_id
        or spec.retrieval_lane == "hybrid"
    ):
        return _search_payload(spec, archive=archive, serving_identity=serving_identity, dependencies=dependencies)

    summaries = _archive_list_summaries_for_spec(
        archive, spec, default_limit=DEFAULT_SESSION_LIST_LIMIT, limit=limit, offset=offset
    )
    total = _archive_count_sessions_for_spec(archive, spec)
    outcome = decide_outcome(matched=total)
    return {
        "outcome": outcome.to_dict(),
        "items": [
            session_list_envelope_from_summary(
                archive_summary_to_domain(summary), message_count=summary.message_count
            ).model_dump(mode="json")
            for summary in summaries
        ],
        "total": total,
        "total_unit": session_count_unit_label(spec.root),
        "limit": limit,
        "offset": offset,
    }


def _search_payload(
    spec: object,
    *,
    archive: ArchiveStore,
    serving_identity: str,
    dependencies: DaemonReadDependencies,
) -> dict[str, object]:
    """Run lexical/vector search without reopening the archive for hydration."""

    from dataclasses import fields

    from polylogue.archive.query.archive_execution import archive_search_hits
    from polylogue.archive.query.search_contract import LaneFailure
    from polylogue.archive.query.search_hits import project_search_hits
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.core.errors import EmbeddingRetrievalNotReadyError
    from polylogue.surfaces.cursor_identity import search_cursor_request_identity
    from polylogue.surfaces.payloads import (
        SessionSearchHitPayload,
        build_search_envelope,
        decode_search_cursor,
        search_cursor_lane_matches_request,
    )

    assert isinstance(spec, SessionQuerySpec)
    request_identity = search_cursor_request_identity(
        {
            field.name: getattr(spec, field.name)
            for field in fields(spec)
            if field.name not in {"cursor", "offset", "limit", "vector_provider", "predicates"}
        }
    )
    cursor = decode_search_cursor(spec.cursor) if spec.cursor else None
    if cursor is not None and not search_cursor_lane_matches_request(cursor.lane, spec.retrieval_lane):
        from polylogue.surfaces.payloads import InvalidSearchCursorError

        raise InvalidSearchCursorError(
            f"cursor was minted for retrieval_lane={cursor.lane!r} but this request is {spec.retrieval_lane!r}"
        )
    if cursor is not None and cursor.query_hash is not None and cursor.query_hash != request_identity:
        from polylogue.surfaces.payloads import InvalidSearchCursorError

        raise InvalidSearchCursorError("cursor belongs to a different ranked-search request")
    display_limit = spec.limit or 50
    fetch_spec = replace(spec, offset=cursor.r, limit=display_limit * 2) if cursor is not None else spec
    needs_vector = bool(
        fetch_spec.similar_text or fetch_spec.similar_session_id or fetch_spec.retrieval_lane == "hybrid"
    )
    vector_provider = dependencies.vector_provider
    vector_failure = dependencies.vector_failure
    if needs_vector and vector_provider is None and vector_failure is None:
        vector_failure = LaneFailure(
            "vector",
            "unavailable",
            "no configured/constructible vector backend",
            "vector retrieval is unavailable; configure Voyage/sqlite-vec and retry",
        )
    if needs_vector and vector_provider is None and fetch_spec.retrieval_lane != "hybrid":
        raise EmbeddingRetrievalNotReadyError(
            "semantic retrieval is unavailable: no configured/constructible vector backend; configure Voyage/sqlite-vec and retry",
            readiness_status="failed" if vector_failure.kind != "unavailable" else "disabled",
        )
    plan = fetch_spec.to_plan(vector_provider=vector_provider)
    pairs, resolved_lane = archive_search_hits(
        plan,
        archive_root=archive.archive_root,
        config=None,
        archive=archive,
    )
    query_text = (
        " ".join((*fetch_spec.query_terms, *fetch_spec.contains_terms)).strip() or fetch_spec.similar_text or ""
    )
    hits = project_search_hits(plan, pairs, resolved_lane, vector_failure=vector_failure)
    hit_payloads = tuple(
        SessionSearchHitPayload.from_search_hit(hit, message_count=hit.summary.message_count) for hit in hits
    )
    # Vector backends deliberately expose a bounded nearest-neighbour page, not
    # an archive-wide cardinality.  ``None`` is the canonical honest total.
    if needs_vector:
        total: int | None = None
    else:
        from polylogue.api.archive import _archive_count_sessions_for_spec

        total = _archive_count_sessions_for_spec(archive, fetch_spec)
    authority = authority_for_reader(
        archive,
        server_identity=cast("Literal['daemon', 'direct']", "daemon" if serving_identity == "daemon" else "direct"),
        started_at=monotonic(),
    ).model_copy(update={"matched": len(hit_payloads), "analyzed": total})
    return build_search_envelope(
        hit_payloads,
        total=total,
        limit=display_limit,
        offset=spec.offset,
        query=query_text,
        retrieval_lane=resolved_lane,
        sort=spec.sort,
        cursor=cursor,
        request_identity=request_identity,
        execution=hits.execution,
        authority=authority,
    ).model_dump(mode="json")


def _query_units_payload(
    params: Mapping[str, object], *, archive: ArchiveStore, serving_identity: str
) -> dict[str, object]:
    from polylogue.archive.query.transaction import query_units_transaction_request
    from polylogue.archive.query.unit_results import query_unit_envelope, query_unit_request

    expression = str(params.get("expression") or "")
    request = query_unit_request(
        expression=expression,
        limit=_non_negative_int(params.get("limit"), default=50) or 50,
        offset=_non_negative_int(params.get("offset"), default=0),
        **{key: value for key, value in params.items() if key not in {"expression", "limit", "offset"}},
    )
    transaction_request = query_units_transaction_request(
        expression=expression,
        session_filters=request.session_filters or {},
        page_size=request.limit,
        offset=request.offset,
    )
    return query_unit_envelope(
        archive,
        request,
        transaction_request=transaction_request,
        serving_identity=serving_identity,
    ).model_dump(mode="json")


def _completion_payload(payload: Mapping[str, object]) -> dict[str, object]:
    from polylogue.archive.query.completions import query_completion_payload

    kind = cast("Literal['field', 'operator', 'value', 'unit']", payload.get("kind") or "field")
    return {
        "query_completions": query_completion_payload(
            kind,
            incomplete=str(payload.get("incomplete") or ""),
            unit=cast("str | None", payload.get("unit")),
            field=cast("str | None", payload.get("field")),
        )
    }


def _facets_payload(params: Mapping[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    """Reuse the API's canonical aggregate implementation over this reader."""

    from polylogue.api.archive import _archive_facet_buckets
    from polylogue.archive.query.expression import compile_expression_into
    from polylogue.archive.query.facets import compute_idf
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.surfaces.outcome import decide_outcome

    query = str(params.get("query") or "").strip()
    base = SessionQuerySpec.from_params(
        {
            key: value
            for key, value in params.items()
            if key not in {"query", "include_deferred", "include_expensive", "no_idf"}
        }
    )
    spec = compile_expression_into(query, base) if query else base
    include_deferred = _truthy(params.get("include_deferred")) or _truthy(params.get("include_expensive"))
    global_buckets = _archive_facet_buckets(archive, None, include_deferred=include_deferred)
    scoped_to_query = spec.has_filters()
    scoped = (
        _archive_facet_buckets(archive, spec, include_deferred=include_deferred) if scoped_to_query else global_buckets
    )
    active = scoped if scoped_to_query else global_buckets

    def buckets(value: object) -> dict[str, object]:
        return {
            "origins": dict(value.origins),
            "tags": dict(value.tags),
            "repos": dict(value.repos),
            "role_counts": dict(value.role_counts),
            "material_origins": dict(value.material_origins),
            "message_types": dict(value.message_types),
            "action_types": dict(value.action_types),
            "has_flags": dict(value.has_flags),
            "omitted": dict(value.omitted),
            "total_sessions": value.total_sessions,
            "total_messages": value.total_messages,
        }

    deferred = (
        {}
        if include_deferred
        else dict.fromkeys(
            ("repos", "role_counts", "material_origins", "message_types", "action_types", "has_flags", "omitted"),
            "deferred_by_default",
        )
    )
    complete = (
        (
            "origins",
            "tags",
            "repos",
            "role_counts",
            "material_origins",
            "message_types",
            "action_types",
            "has_flags",
            "omitted",
        )
        if include_deferred
        else ("origins", "tags")
    )
    return {
        "outcome": decide_outcome(matched=active.total_sessions).to_dict(),
        "scoped_to_query": scoped_to_query,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "stale": False,
        "budget_exceeded": False,
        "cost_class": "cheap",
        "complete_families": list(complete),
        "deferred_families": deferred,
        "family_errors": {},
        "origins": dict(active.origins),
        "tags": dict(active.tags),
        "repos": dict(active.repos),
        "role_counts": dict(active.role_counts),
        "material_origins": dict(active.material_origins),
        "message_types": dict(active.message_types),
        "action_types": dict(active.action_types),
        "has_flags": dict(active.has_flags),
        "omitted_facet_counts": dict(active.omitted),
        "total_sessions": active.total_sessions,
        "total_messages": active.total_messages,
        "scoped": buckets(scoped),
        "global": buckets(global_buckets),
        "idf": {} if _truthy(params.get("no_idf")) else compute_idf(global_buckets),
    }


def _lower_cli_query_params(params: Mapping[str, object]) -> tuple[dict[str, object], str]:
    """Lower CLI conveniences without importing Click's ``RootModeRequest``."""

    normalized = dict(params)
    raw_terms = normalized.pop("query", ())
    if isinstance(raw_terms, str):
        terms = (raw_terms,)
    elif isinstance(raw_terms, (list, tuple)):
        terms = tuple(str(item) for item in raw_terms)
    else:
        raise ValueError("query must be a string or sequence")
    lexical = bool(normalized.pop("lexical", False))
    semantic = bool(normalized.pop("semantic", False))
    if lexical and (semantic or normalized.get("similar_text")):
        raise ValueError("semantic retrieval cannot be combined with lexical retrieval")
    if semantic:
        if not terms:
            raise ValueError("semantic retrieval requires query terms")
        normalized["similar_text"] = " ".join(terms)
        terms = ()
    if lexical:
        normalized["retrieval_lane"] = "dialogue"
    return normalized, _expression_from_query_terms(terms)


def _cli_query_spec(params: Mapping[str, object]) -> object:
    """Compile the same CLI selection contract used by canonical execution."""

    from polylogue.archive.query.expression import compile_expression_into
    from polylogue.archive.query.spec import SessionQuerySpec

    normalized, expression = _lower_cli_query_params(params)
    base = SessionQuerySpec.from_params(normalized)
    return compile_expression_into(expression, base) if expression else base


def _expression_from_query_terms(terms: tuple[str, ...]) -> str:
    from polylogue.archive.query.root_lowering import expression_from_query_terms

    return expression_from_query_terms(terms)


def _non_negative_int(value: object, *, default: int) -> int:
    try:
        parsed = int(cast("int | str", value))
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _truthy(value: object) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"})


__all__ = [
    "DaemonReadDependencies",
    "VectorReadBinding",
    "execute_read_operation",
    "requires_vector_snapshot",
    "vector_binding_from_config",
]
