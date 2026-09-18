"""Canonical archive reads for the daemon machine-operation protocol.

The daemon opens and pins the reader.  This module deliberately receives that
reader instead of a root/configuration so an operation cannot accidentally
re-open whichever archive generation happens to be current.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, replace
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal, cast

from polylogue.operations.authority import authority_for_reader
from polylogue.operations.query_lowering import cli_query_spec, lower_cli_query_params

if TYPE_CHECKING:
    from polylogue.archive.query.expression import WithUnitWindow
    from polylogue.archive.query.search_contract import LaneFailure
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.config import Config, PolylogueConfig
    from polylogue.core.protocols import VectorProvider
    from polylogue.storage.embeddings.identity import EmbeddingRecipe
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary, ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveSessionEnvelope


@dataclass(frozen=True, slots=True)
class VectorReadBinding:
    """Pinned configuration needed to bind one supplied vector snapshot."""

    voyage_key: str
    model: str
    dimension: int

    @property
    def recipe(self) -> EmbeddingRecipe:
        """The whole configured embedding recipe this binding addresses with.

        Vector addressing is a function of model *and* dimension (and request
        shape); passing the model alone forced a hardcoded 1024 dimensions and
        made selection disagree with the embed path (polylogue-crcst).
        """

        from polylogue.storage.embeddings.identity import EmbeddingRecipe

        return EmbeddingRecipe.current(model=self.model, dimensions=self.dimension)

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


_SESSION_READ_WINDOW = 200
_SESSION_READ_PROJECTION = "session-read-v1"


def page_next_offset(*, offset: int, returned: int, total: int | None, limit: int) -> int | None:
    """Decide the next page offset for one bounded read, in one place.

    Both the list page (``cli.query`` unranked) and the ranked search envelope
    answer the same client question -- "is there another page, and where does
    it start?" -- so they must answer it identically.  Deciding it per payload
    is how the list page came to omit the key entirely, which silently ended
    ``query_complete_session_ids``' walk after one page and let
    ``delete --all`` act on the first page only (#1873 regression, polylogue-w3s0q).

    ``total`` may be an honest ``None`` (a vector lane exposes a bounded
    nearest-neighbour page, not an archive-wide cardinality).  Without a total
    the only evidence that more rows exist is a page that filled its own
    bound, so a full page continues and a short page terminates.
    """

    if returned <= 0:
        return None
    nxt = offset + returned
    if total is None:
        return nxt if limit > 0 and returned >= limit else None
    return nxt if nxt < total else None


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
    cacheable = _cacheable_read(name, payload)
    cache_key_payload = _params(payload) if name in {"cli.query", "facets"} else payload
    # A grammar completion is a pure protocol read and deliberately accepts the
    # minimal archive-shaped object used by the public operation seam, so it
    # skips index validation.  A completion that names an archive-backed
    # ``source`` does read the archive and is validated like any other read.
    # Every other operation retains the existing index validation, including
    # operations that do not use the result cache.
    generation: str | None = None
    if name != "completion" or completion_reads_archive(payload):
        generation = str(archive.index_db_path.resolve())
    if cacheable:
        assert generation is not None
        from polylogue.storage.search.cache import get_cached_result

        cached = get_cached_result(
            name,
            cache_key_payload,
            archive_root=archive.archive_root,
            generation=generation,
        )
        if cached is not None:
            return cached

    if name == "cli.query":
        params = _params(payload)
        result = _query_payload(params, archive=archive, serving_identity=serving_identity, dependencies=dependencies)
    elif name == "query.aggregate":
        result = _aggregate_payload(payload, archive=archive)
    elif name == "session.read":
        result = _session_read_payload(payload, archive=archive)
    elif name == "session.reference":
        result = _session_reference_payload(payload, archive=archive)
    elif name == "query.units":
        params = _params(payload)
        result = _query_units_payload(params, archive=archive, serving_identity=serving_identity)
    elif name == "completion":
        result = _completion_payload(payload, archive=archive)
    elif name == "facets":
        result = _facets_payload(_params(payload), archive=archive)
    elif name == "status":
        if dependencies.status_now_ms is None:
            raise ValueError("status operation requires an operation-captured now_ms")
        from polylogue.operations.daemon_status import produce_operation_status

        result = produce_operation_status(
            archive=archive,
            now_ms=dependencies.status_now_ms,
            config=dependencies.status_config,
            runtime_status=dependencies.runtime_status,
            include_archive_readiness=_truthy(payload.get("include_archive_readiness"))
            or _truthy(_params(payload).get("include_archive_readiness")),
        )
    else:
        raise ValueError(f"read operation is not declared: {name}")

    if cacheable:
        assert generation is not None
        from polylogue.storage.search.cache import put_cached_result

        put_cached_result(
            name,
            cache_key_payload,
            result,
            archive_root=archive.archive_root,
            generation=generation,
        )
    return result


def _cacheable_read(name: str, payload: Mapping[str, object]) -> bool:
    """Return whether a read is safe to reuse until the index revision moves.

    Status carries request-time freshness and completion metadata, while vector
    queries depend on an embeddings snapshot that can advance independently of
    the index.  Keep both out of this small result cache; ordinary list/search
    and facets reads are invalidated by the indexing write path.
    """
    if name == "facets":
        return True
    if name == "cli.query":
        return not requires_vector_snapshot(name, payload)
    return False


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
    from polylogue.archive.query.expression import compile_expression_into
    from polylogue.archive.query.spec import (
        DEFAULT_SESSION_LIST_LIMIT,
        SessionQuerySpec,
        clamp_query_limit,
        resolve_default_root_filter,
        session_count_unit_label,
    )
    from polylogue.surfaces.outcome import decide_outcome

    normalized, expression = _lower_cli_query_params(params)
    limit = clamp_query_limit(normalized.get("limit"), default=DEFAULT_SESSION_LIST_LIMIT)
    offset = _non_negative_int(normalized.get("offset"), default=0)
    # CLI root payloads retain presentation-only keys.  The existing query
    # contract intentionally ignores those while compiling selection intent.
    base = SessionQuerySpec.from_params({**normalized, "limit": limit, "offset": offset})
    spec = compile_expression_into(expression, base) if expression else base
    spec = _resolved_scope_spec(spec, archive=archive)

    searching = bool(
        spec.query_terms
        or spec.contains_terms
        or spec.similar_text
        or spec.similar_session_id
        or spec.retrieval_lane == "hybrid"
    )
    if spec.sample is not None:
        # ``--sample`` randomizes the ordering of a listed page; ranked
        # retrieval already owns its order, so the two never combine.
        if spec.sample <= 0:
            raise ValueError("sample must be positive")
        if searching:
            raise ValueError("sample does not combine with search terms")
        if spec.cursor:
            raise ValueError("sample does not combine with a cursor")
        limit = spec.sample
        offset = 0
    if searching:
        return _search_payload(spec, archive=archive, serving_identity=serving_identity, dependencies=dependencies)
    if spec.latest:
        # ``latest`` is applied by ``query_spec_to_plan`` on the ranked path
        # only; the list path reaches ``ArchiveStore.list_summaries`` without
        # it, so honour it here rather than returning a full page for
        # ``--latest``.
        limit = 1
    summaries = _archive_list_summaries_for_spec(
        archive, spec, default_limit=DEFAULT_SESSION_LIST_LIMIT, limit=limit, offset=offset
    )
    total = _archive_count_sessions_for_spec(archive, spec)
    outcome = decide_outcome(matched=total)
    session_ids = [summary.session_id for summary in summaries]
    attached = _attached_units_payload(session_ids, spec=spec, params=params, archive=archive)
    lineage_edges = _lineage_edges_payload(session_ids, spec=spec, archive=archive)
    return {
        "outcome": outcome.to_dict(),
        **({"attached_units": attached} if attached is not None else {}),
        "items": [{**_session_list_row(summary), **lineage_edges.get(summary.session_id, {})} for summary in summaries],
        "total": total,
        # The unit names the filter that actually ran, which is the *resolved*
        # root filter (``_archive_query_kwargs`` resolves it the same way), not
        # the unset spec field.
        "total_unit": session_count_unit_label(
            resolve_default_root_filter(spec.root, boolean_predicate=spec.boolean_predicate)
        ),
        "limit": limit,
        "offset": offset,
        # Decided by the same helper the ranked envelope uses: a client that
        # must see every match (mutating-verb cardinality) walks this key, and
        # its absence read as "complete" after one page.
        "next_offset": page_next_offset(offset=offset, returned=len(summaries), total=total, limit=limit),
    }


def _resolved_scope_spec(spec: SessionQuerySpec, *, archive: ArchiveStore) -> SessionQuerySpec:
    """Resolve an explicit session scope to a full session id before filtering.

    ``--id`` accepts any reference spelling the archive can resolve — a native
    id, a prefix, a full ``origin:native`` id.  The SQL filters compare against
    the full ``session_id``, so an unresolved spelling silently scopes the page
    to nothing and reports an empty result instead of the session the operator
    named.  Resolution failure is stated, never rendered as "no rows".
    """

    from dataclasses import replace as dataclass_replace

    scope = spec.session_id
    if not scope:
        return spec
    try:
        resolved = archive.resolve_session_id(scope)
    except KeyError as exc:
        raise ValueError(f"session not found: {scope}") from exc
    return spec if resolved == scope else dataclass_replace(spec, session_id=resolved)


#: The row vocabulary ``cli.query`` reports, declared here because this is
#: where the row is built. It used to be declared a second time on the client
#: (``archive_query._DAEMON_LIST_ITEM_KEEP_KEYS`` plus
#: ``_normalize_daemon_list_item``), which made the operation row and the CLI
#: row two shapes bridged by a translation step. The operation row is now the
#: CLI row: the renderer prints what it is handed.
#:
#: ``SessionListEnvelope`` carries three further fields -- ``title_source``,
#: ``title_ref`` and ``cost_provenance`` -- that describe how the row's title
#: and cost were derived rather than what the session is. They belong to the
#: reader surfaces that render provenance affordances, not to a terminal row,
#: and no CLI format has ever printed them.
_SESSION_LIST_ROW_FIELDS = (
    "id",
    "origin",
    "title",
    "target_ref",
    "anchor",
    "actions",
    "created_at",
    "updated_at",
    "message_count",
    "tags",
    "summary",
    "words",
    "repo",
    "cwd_display",
    "terminal_state",
    "total_cost_usd",
    "relative_time",
    "flags",
    # Projection columns the operation materialises only when the query asked
    # for them: the recursive-graph edges of a ``lineage:id:``-seeded page.
    "parent_refs",
    "child_refs",
    "continuation",
)


def _session_list_row(summary: ArchiveSessionSummary) -> dict[str, object]:
    """Render one archive summary as the canonical CLI session-list row.

    Two properties are load-bearing, and the CLI's own direct branch
    (``archive_query._summary_payload``, which built the very same
    ``SessionListRowPayload``) established both:

    * ``exclude_none`` -- the compact document.  Emitting explicit nulls here
      made one query render two different documents by transport.
    * the archive's own ``...Z`` timestamp spelling.  ``ArchiveStore`` renders
      its millisecond columns with ``_iso_from_ms`` (``...Z``); the
      domain round-trip inside ``session_list_envelope_from_summary`` re-renders
      the same instant through ``datetime.isoformat()`` as ``...+00:00``.  The
      fix belongs in that shared helper, but ``surfaces/payloads.py`` is IN the
      derived-schema identity closure, so it is applied here rather than moving
      the identity for a timestamp spelling.
    """
    from polylogue.archive.hydration import archive_summary_to_domain
    from polylogue.surfaces.payloads import session_list_envelope_from_summary
    from polylogue.surfaces.query_rows import session_row

    domain = archive_summary_to_domain(summary)
    row: dict[str, object] = session_list_envelope_from_summary(
        domain,
        message_count=summary.message_count,
        # Without this the row carries no word count at all: the CLI's own
        # branch renders ``words`` from the summary, and ``exclude_none`` would
        # otherwise drop the field rather than report it as zero.
        word_count=summary.word_count,
    ).model_dump(mode="json", exclude_none=True)
    for key, stored in (("created_at", summary.created_at), ("updated_at", summary.updated_at)):
        if stored is not None:
            row[key] = stored
    # ``terminal_state`` is a closed vocabulary with an explicit ``unknown``
    # member; the raw summary field is nullable and the shared row projection
    # is what resolves the two. Reporting the null instead would make an
    # unknown outcome indistinguishable from an absent field.
    row["terminal_state"] = session_row(domain, message_count=summary.message_count).outcome
    return {key: row[key] for key in _SESSION_LIST_ROW_FIELDS if row.get(key) is not None}


def _search_payload(
    spec: SessionQuerySpec,
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
    from polylogue.core.errors import EmbeddingRetrievalNotReadyError
    from polylogue.surfaces.cursor_identity import search_cursor_request_identity
    from polylogue.surfaces.payloads import (
        SessionSearchHitPayload,
        build_search_envelope,
        decode_search_cursor,
        search_cursor_lane_matches_request,
    )

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
            readiness_status=(
                "failed" if vector_failure is not None and vector_failure.kind != "unavailable" else "disabled"
            ),
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
        server_identity="daemon" if serving_identity == "daemon" else "direct",
        started_at=monotonic(),
    ).model_copy(update={"matched": len(hit_payloads), "analyzed": total})
    # The ranked envelope must name what it counted.  Without this a
    # ``--no-root`` search reports subagent/branch rows under the "top-level
    # sessions" label, because the renderer has nothing to read but a default.
    # Resolve the unit exactly as the list path does, from the *resolved* root
    # filter rather than the unset spec field.
    from polylogue.archive.query.spec import resolve_default_root_filter, session_count_unit_label

    envelope = build_search_envelope(
        hit_payloads,
        total=total,
        total_unit=session_count_unit_label(
            resolve_default_root_filter(fetch_spec.root, boolean_predicate=fetch_spec.boolean_predicate)
        ),
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
    # Match the direct branch's hit shape exactly: ``archive_query._hit_payload``
    # dumps the same ``SessionSearchHitPayload`` with ``exclude_none=True``.
    # The envelope keeps its own explicit nulls -- a vector page's ``total`` is
    # an honest ``None`` and dropping the key would read as "not reported".
    envelope["hits"] = [hit.model_dump(mode="json", exclude_none=True) for hit in hit_payloads]
    # The ranked envelope's own continuation is ``next_cursor``; ``next_offset``
    # is the offset-shaped answer the list page also gives, decided by the one
    # helper so a client walking pages cannot see the two paths disagree.
    envelope["next_offset"] = page_next_offset(
        offset=spec.offset, returned=len(hit_payloads), total=total, limit=display_limit
    )
    return envelope


def _query_units_payload(
    params: Mapping[str, object], *, archive: ArchiveStore, serving_identity: str
) -> dict[str, object]:
    from polylogue.archive.query.transaction import query_units_transaction_request
    from polylogue.archive.query.unit_results import query_unit_envelope, query_unit_request

    expression = str(params.get("expression") or "")
    filter_params = {
        key: value for key, value in params.items() if key not in {"expression", "limit", "offset", "session_filters"}
    }
    raw_session_filters = params.get("session_filters")
    if raw_session_filters is not None and not isinstance(raw_session_filters, Mapping):
        raise ValueError("session_filters must be an object")
    request = query_unit_request(
        expression=expression,
        limit=_non_negative_int(params.get("limit"), default=50) or 50,
        offset=_non_negative_int(params.get("offset"), default=0),
        session_filters=raw_session_filters,
        **filter_params,
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


#: Archive-backed completion vocabularies and how each is counted. Session ids
#: are listed; the rest are ``stats_by`` group counts, which is the same
#: aggregate ``analyze by <dimension>`` reports, so a completion can never
#: offer a value the corresponding query would not match.
_COMPLETION_VALUE_UNITS: Mapping[str, str] = {
    "repo": "sessions",
    "tag": "sessions",
    "tool": "actions",
}

#: How many sessions a session-id completion scans before filtering. The CLI's
#: own reader used the same window; it bounds the work a TAB press costs on a
#: large archive rather than the number of rows that come back.
_COMPLETION_SESSION_SCAN = 100

#: Every declared completion source reads archive content. ``origin`` is
#: deliberately not among them: it is a declared vocabulary that must stay
#: completable on an archive that does not exist yet, so the CLI answers it
#: from ``sources.origin_specs`` without a read.
ARCHIVE_COMPLETION_SOURCES: frozenset[str] = frozenset({"session_id", *_COMPLETION_VALUE_UNITS})


def completion_reads_archive(payload: Mapping[str, object]) -> bool:
    """Whether this completion request needs a validated index to answer."""

    source = payload.get("source")
    return isinstance(source, str) and source in ARCHIVE_COMPLETION_SOURCES


def _completion_payload(payload: Mapping[str, object], *, archive: ArchiveStore | None = None) -> dict[str, object]:
    """Answer one completion question from the grammar, or from the archive.

    ``source`` names an archive-backed vocabulary and is the only path that
    touches ``archive``; every other kind is a pure protocol read over the
    declared query grammar, which is why this operation is still servable
    against the minimal archive-shaped object the public operation seam passes.
    """

    source = payload.get("source")
    if isinstance(source, str) and source:
        return {"value_completions": _completion_values(source, payload, archive=archive)}

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


def _completion_values(
    source: str, payload: Mapping[str, object], *, archive: ArchiveStore | None
) -> dict[str, object]:
    incomplete = str(payload.get("incomplete") or "")
    raw_limit = payload.get("limit")
    limit = int(raw_limit) if isinstance(raw_limit, int) and raw_limit > 0 else 32
    values: list[dict[str, object]]
    if source == "session_id":
        values = _session_id_completions(incomplete, archive=_require_archive(archive, source), limit=limit)
    elif source in _COMPLETION_VALUE_UNITS:
        values = _grouped_completions(source, incomplete, archive=_require_archive(archive, source), limit=limit)
    else:
        raise ValueError(f"completion source is not declared: {source}")
    return {"source": source, "incomplete": incomplete, "values": values}


def _require_archive(archive: ArchiveStore | None, source: str) -> ArchiveStore:
    if archive is None:
        raise ValueError(f"completion source requires an archive: {source}")
    return archive


def _session_id_completions(incomplete: str, *, archive: ArchiveStore, limit: int) -> list[dict[str, object]]:
    current = incomplete.strip()
    current_lower = current.lower()
    values: list[dict[str, object]] = []
    for summary in archive.list_summaries(limit=_COMPLETION_SESSION_SCAN):
        session_id = str(summary.session_id)
        title = summary.title or ""
        if current and not (
            session_id.startswith(current)
            or (":" in session_id and current in session_id)
            or (title and current_lower in title.lower())
        ):
            continue
        values.append({"value": session_id, "help": f"{summary.origin} · {title or session_id}"})
    values.sort(key=lambda item: cast("str", item["value"]))
    return values[:limit]


def _grouped_completions(source: str, incomplete: str, *, archive: ArchiveStore, limit: int) -> list[dict[str, object]]:
    unit = _COMPLETION_VALUE_UNITS[source]
    # Tags live in the durable ``user.db`` and have their own reader; repo and
    # tool are index aggregates, which is the same ``stats_by`` dimension
    # ``analyze by <dimension>`` reports -- so a completion can never offer a
    # value the corresponding query would not match.
    grouped = archive.list_user_tags() if source == "tag" else archive.stats_by(source)
    prefix = incomplete.strip().lower()
    ordered = sorted(grouped.items(), key=lambda pair: (-pair[1], pair[0]))
    values: list[dict[str, object]] = []
    for value, count in ordered:
        if prefix and not value.lower().startswith(prefix):
            continue
        values.append({"value": value, "help": f"{count} {unit}"})
        if len(values) >= limit:
            break
    return values


def _facets_payload(params: Mapping[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    """Answer facets from the one canonical envelope the API surface builds.

    This used to restate that assembly over the same buckets and had drifted
    away from it: no ``family_status``, ``availability``, ``deadline_s``,
    ``elapsed_s`` or ``stale_age_s``; hard-coded ``budget_exceeded`` and
    ``cost_class``; its own family lists (``omitted`` in both complete and
    deferred, no ``total_counts``); and no ``PostFilterScopeTooLargeError``
    handling, so a too-large scope raised instead of degrading. Building the
    shared model and dumping it keeps the two surfaces equal by construction
    rather than by matching key lists.
    """

    import time

    from polylogue.api.archive import PostFilterScopeTooLargeError, _archive_facet_buckets, build_facets_response
    from polylogue.archive.query.expression import compile_expression_into
    from polylogue.archive.query.spec import SessionQuerySpec

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
    scoped_to_query = spec.has_filters()

    started_at = time.perf_counter()
    global_buckets = _archive_facet_buckets(archive, None, include_deferred=include_deferred)
    post_filter_gap: str | None = None
    if scoped_to_query:
        try:
            scoped_buckets = _archive_facet_buckets(archive, spec, include_deferred=include_deferred)
        except PostFilterScopeTooLargeError as exc:
            from polylogue.archive.query.facets import FacetBuckets

            scoped_buckets, post_filter_gap = FacetBuckets(), exc.gap_reason
    else:
        scoped_buckets = global_buckets

    response = build_facets_response(
        global_buckets=global_buckets,
        scoped_buckets=scoped_buckets,
        scoped_to_query=scoped_to_query,
        include_deferred=include_deferred,
        elapsed_s=time.perf_counter() - started_at,
        include_idf=not _truthy(params.get("no_idf")),
        post_filter_gap=post_filter_gap,
    )
    return cast(dict[str, object], response.model_dump(by_alias=True, mode="json"))


def _lineage_seed_from_predicate(predicate: object) -> str | None:
    """Return the ``lineage:id:`` seed session id carried by ``predicate``, if any.

    ``lineage:id:<ref>`` compiles to :class:`QueryLineagePredicate` (possibly
    ANDed with other clauses), which the SQL layer already uses to filter
    session rows to one shared-root lineage family. This walks the same
    boolean-predicate tree to detect that shape so the list route can
    materialize the declared recursive-graph projection columns for it (#z9gh.3).

    Only descends into ``and`` nodes. A ``lineage:id:X or repo:foo`` result
    set is NOT purely lineage X's family -- rows matched only via the ``or``
    branch would get X's parent_refs/child_refs/continuation stamped on them,
    which is wrong, not just imprecise. An ``or`` node (or a ``not`` wrapping
    the predicate, which isn't a ``QueryBoolPredicate``/``QueryLineagePredicate``
    at all) correctly yields no seed here.
    """
    from polylogue.archive.query.predicate import QueryBoolPredicate, QueryLineagePredicate

    if predicate is None:
        return None
    if isinstance(predicate, QueryLineagePredicate):
        return predicate.seed_session_id
    if isinstance(predicate, QueryBoolPredicate) and predicate.op == "and":
        for child in predicate.children:
            seed = _lineage_seed_from_predicate(child)
            if seed is not None:
                return seed
    return None


def _lineage_edges_payload(
    session_ids: list[str],
    *,
    spec: SessionQuerySpec,
    archive: ArchiveStore,
) -> dict[str, dict[str, object]]:
    """Materialise the recursive-graph columns of a ``lineage:id:``-seeded page.

    The seeded page is already one shared-root family, so the direct edges of
    the rows on it are the whole projection: one bounded lookup over the page
    rather than a second unbounded graph walk (#z9gh.3).
    """

    seed = _lineage_seed_from_predicate(spec.boolean_predicate)
    if seed is None or not session_ids:
        return {}
    edges = archive.session_lineage_edges(session_ids)
    projected: dict[str, dict[str, object]] = {}
    for session_id in session_ids:
        edge = edges.get(session_id)
        if edge is None:
            continue
        parent_id, child_ids = edge
        projected[session_id] = {
            "parent_refs": [parent_id] if parent_id else [],
            "child_refs": list(child_ids),
        }
    return projected


def _attached_units_payload(
    session_ids: list[str],
    *,
    spec: SessionQuerySpec,
    params: Mapping[str, object],
    archive: ArchiveStore,
) -> dict[str, object] | None:
    """Project ``with <unit>`` rows for the sessions on this page.

    The rows come from the shared attached-unit executor; this seam only
    decodes the wire form of the projection and densifies the sparse result so
    every requested unit names every session on the page.
    """

    units = _string_tuple(params.get("with_units")) or spec.with_units
    if not units:
        return None
    fields = _unit_fields(params.get("with_unit_fields")) or spec.with_unit_fields
    windows = _unit_windows(params.get("with_unit_windows")) or spec.with_unit_windows
    if not session_ids:
        return {unit: {} for unit in units}

    from polylogue.archive.query.attached_units import fetch_attached_units

    attached = fetch_attached_units(
        archive,
        session_ids,
        units,
        unit_fields=dict(fields) or None,
        unit_windows=dict(windows) or None,
    )
    return {
        unit: {session_id: list(attached.get(unit, {}).get(session_id, ())) for session_id in session_ids}
        for unit in units
    }


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes)):
        return (str(value),) if isinstance(value, str) and value else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise ValueError("expected a list of strings")


def _unit_fields(value: object) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("with_unit_fields must be an object")
    return {str(unit): _string_tuple(fields) for unit, fields in value.items()}


def _unit_windows(value: object) -> dict[str, WithUnitWindow]:
    """Rebuild ``WithUnitWindow`` values from their declared wire payload."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("with_unit_windows must be an object")
    from polylogue.archive.query.expression import WithUnitWindow as _WithUnitWindow

    decoded: dict[str, WithUnitWindow] = {}
    for unit, raw in value.items():
        if not isinstance(raw, Mapping):
            raise ValueError("each with_unit_window must be an object")
        predicates = raw.get("predicates") or {}
        if not isinstance(predicates, Mapping):
            raise ValueError("with_unit_window predicates must be an object")
        window = raw.get("window")
        bracket: tuple[Literal["first", "last"], int] | None = None
        if window is not None:
            if not isinstance(window, Mapping) or window.get("kind") not in {"first", "last"}:
                raise ValueError("with_unit_window window must name first or last")
            bracket = (cast("Literal['first', 'last']", window["kind"]), int(cast("int", window["n"])))
        decoded[str(unit)] = _WithUnitWindow(
            predicates=tuple((str(key), str(item)) for key, item in predicates.items()),
            window=bracket,
        )
    return decoded


def _aggregate_payload(payload: Mapping[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    """Compute one aggregate over the same selection vocabulary as ``cli.query``.

    Every number comes from the archive's own aggregate executors; this seam
    only chooses which one the declared mode names.
    """

    from dataclasses import asdict

    from polylogue.archive.query.filter_kwargs import spec_session_filter_kwargs, stats_filter_kwargs
    from polylogue.surfaces.outcome import decide_outcome

    mode = str(payload.get("mode") or "")
    if mode not in {"count", "stats", "stats_by"}:
        raise ValueError(f"aggregate mode is not declared: {mode!r}")
    raw_params = payload.get("params", {})
    if not isinstance(raw_params, Mapping):
        raise ValueError("aggregate params must be an object")
    spec = _cli_query_spec({str(key): value for key, value in raw_params.items()})
    if spec.similar_text or spec.similar_session_id or spec.retrieval_lane == "hybrid":
        raise ValueError("aggregates are computed over lexical and structural selection only")
    spec = _resolved_scope_spec(spec, archive=archive)

    filter_kwargs = spec_session_filter_kwargs(spec)
    query = " ".join((*spec.query_terms, *spec.contains_terms)).strip()
    scope_id = spec.session_id

    if mode == "count":
        count = (
            archive.count_search_sessions(query, session_id=scope_id, **cast("Any", filter_kwargs))
            if query
            else archive.count_sessions(session_id=scope_id, **cast("Any", filter_kwargs))
        )
        return {"outcome": decide_outcome(matched=count).to_dict(), "mode": "count", "count": count}

    session_ids = _matched_session_ids(
        archive, query=query, session_id=scope_id, limit=spec.limit, filters=filter_kwargs
    )
    empty_selection = bool(query) and not session_ids
    aggregate_kwargs = cast("Any", stats_filter_kwargs(filter_kwargs))

    if mode == "stats_by":
        group_by = str(payload.get("group_by") or "")
        if not group_by:
            raise ValueError("stats_by requires a group_by field")
        grouped: dict[str, int] = (
            {} if empty_selection else dict(archive.stats_by(group_by, **aggregate_kwargs, session_ids=session_ids))
        )
        return {
            "outcome": decide_outcome(matched=sum(grouped.values())).to_dict(),
            "mode": "stats_by",
            "group_by": group_by,
            "groups": grouped,
        }

    from polylogue.archive.stats import ArchiveStats

    stats = (
        ArchiveStats(total_sessions=0, total_messages=0)
        if empty_selection
        else archive.stats(**aggregate_kwargs, session_ids=session_ids)
    )
    return {
        "outcome": decide_outcome(matched=stats.total_sessions).to_dict(),
        "mode": "stats",
        "stats": asdict(stats),
    }


def _matched_session_ids(
    archive: ArchiveStore,
    *,
    query: str,
    session_id: str | None,
    limit: int | None,
    filters: Mapping[str, object],
) -> tuple[str, ...]:
    """Scope an aggregate to the sessions a text selection actually matched."""

    if session_id is not None:
        try:
            return (archive.resolve_session_id(session_id),)
        except KeyError:
            return ()
    if not query:
        return ()
    return tuple(archive.search_session_ids(query, limit=limit, **cast("Any", filters)))


def _session_identity_projection(
    envelope: ArchiveSessionEnvelope,
    *,
    excluded_blocks: frozenset[str],
) -> dict[str, object]:
    """Project one transcript window in the archive's own identity vocabulary.

    Deliberately *not* the browser reader's session-detail envelope.  That
    envelope is a display projection: it renames ``words``, adds reader anchors
    and actions, and replaces the raw ``origin`` token with its display label.
    An operation result is data for whichever surface asked, so a consumer that
    needs the archive's identities — message ids, block ids, raw origin, the
    canonical topology fields — must not have to reverse a label back into a
    token (which the Origin mapping does not permit anyway).

    ``excluded_blocks`` drops whole blocks by their declared ``block_type``,
    which is what a projection's ``exclude_block_kinds`` names; message rows
    are retained so the window's own coordinates stay honest.
    """

    from polylogue.archive.hydration import archive_message_to_domain
    from polylogue.surfaces.payloads import message_topology_from_domain

    return {
        "session_id": envelope.session_id,
        "native_id": envelope.native_id,
        "origin": envelope.origin,
        "title": envelope.title,
        "active_leaf_message_id": envelope.active_leaf_message_id,
        "created_at": envelope.created_at,
        "updated_at": envelope.updated_at,
        "messages": [
            {
                "message_id": message.message_id,
                "native_id": message.native_id,
                "role": message.role,
                "word_count": message.word_count,
                "has_tool_use": message.has_tool_use,
                **message_topology_from_domain(archive_message_to_domain(message)),
                "blocks": [
                    {
                        "block_id": block.block_id,
                        "message_id": block.message_id,
                        "block_type": block.block_type,
                        "text": block.text,
                        "tool_name": block.tool_name,
                        "tool_id": block.tool_id,
                        "semantic_type": block.semantic_type,
                    }
                    for block in message.blocks
                    if block.block_type not in excluded_blocks
                ],
            }
            for message in envelope.messages
        ],
    }


def _session_read_payload(payload: Mapping[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    """Read one bounded transcript window for an exact session reference.

    A whole transcript can exceed the declared 8 MiB result bound, so this
    operation is windowed by construction: the reader composes only
    ``[offset, offset + limit)`` at the storage layer and hands back a
    snapshot-bound continuation for the next window.
    """

    from polylogue.archive.query.transaction import (
        QueryContinuation,
        QueryContinuationInvalidError,
        QueryTransactionRequest,
        archive_snapshot_epoch,
        validate_continuation_epoch,
    )
    from polylogue.surfaces.outcome import decide_outcome
    from polylogue.surfaces.projection_spec import ProjectionSpec

    ref = str(payload.get("ref") or "").strip()
    if not ref:
        raise ValueError("session.read requires a session reference")
    kind = str(payload.get("kind") or "transcript")
    if kind != "transcript":
        return _session_evidence_payload(ref, kind=kind, archive=archive)
    raw_projection = payload.get("projection")
    if raw_projection is not None and not isinstance(raw_projection, Mapping):
        raise ValueError("projection must be an object")
    projection = ProjectionSpec.model_validate(dict(raw_projection)) if raw_projection else None

    limit = _non_negative_int(payload.get("limit"), default=_SESSION_READ_WINDOW) or _SESSION_READ_WINDOW
    offset = _non_negative_int(payload.get("offset"), default=0)
    if projection is not None:
        limit = projection.body_limit or limit
        offset = projection.body_offset if projection.body_offset is not None else offset

    arguments: dict[str, object] = {
        "ref": ref,
        "projection": dict(raw_projection) if raw_projection else {},
    }
    continuation_token = payload.get("continuation")
    if continuation_token:
        decoded = QueryContinuation.decode(str(continuation_token))
        transaction = decoded.request
        if (
            transaction.operation != "session.read"
            or transaction.projection != _SESSION_READ_PROJECTION
            or decoded.result_ref != transaction.result_ref
            or dict(transaction.arguments) != arguments
        ):
            raise QueryContinuationInvalidError("continuation belongs to another session read")
        limit, offset = transaction.page_size, transaction.offset
        framed = transaction.with_archive_epoch(validate_continuation_epoch(transaction, archive=archive))
    else:
        framed = QueryTransactionRequest(
            operation="session.read",
            arguments=arguments,
            page_size=limit,
            offset=offset,
            projection=_SESSION_READ_PROJECTION,
            stable_order="position",
        ).with_archive_epoch(archive_snapshot_epoch(archive))

    try:
        session_id = archive.resolve_session_id(ref.removeprefix("session:"))
    except KeyError as exc:
        raise ValueError(f"session not found: {ref}") from exc
    summary = archive.read_summary(session_id)
    envelope = archive.read_session_page(session_id, limit=limit, offset=offset)
    excluded_blocks = frozenset(projection.exclude_block_kinds) if projection is not None else frozenset()
    total = summary.message_count
    returned = len(envelope.messages)
    next_offset = offset + returned if offset + returned < total else None
    result: dict[str, object] = {
        "outcome": decide_outcome(matched=returned).to_dict(),
        "session": _session_identity_projection(envelope, excluded_blocks=excluded_blocks),
        "session_id": session_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
        "continuation": (
            QueryContinuation(framed.next(offset=next_offset), framed.result_ref).encode()
            if next_offset is not None
            else None
        ),
        "complete": next_offset is None,
    }
    _require_deliverable_window(result, limit=limit)
    return result


#: Bounded per-session evidence read models, keyed by the ``session.read``
#: kind that names them.  Each reads one relation the query grammar declares no
#: unit for (design D3), through the pinned reader rather than the API facade.
_SESSION_EVIDENCE_READERS: dict[str, str] = {
    "hooks": "hook_event_summary_for_session",
}


def _session_evidence_payload(ref: str, *, kind: str, archive: ArchiveStore) -> dict[str, object]:
    """Read one bounded per-session evidence relation for an exact reference.

    These relations are aggregates, not transcripts: they are answered whole,
    so the result reports ``complete`` with no continuation and the window
    coordinates describe the evidence rows rather than a message page.
    """

    from polylogue.surfaces.outcome import decide_outcome

    reader_name = _SESSION_EVIDENCE_READERS.get(kind)
    if reader_name is None:
        raise ValueError(f"session.read does not serve kind {kind!r}")
    try:
        session_id = archive.resolve_session_id(ref.removeprefix("session:"))
    except KeyError as exc:
        raise ValueError(f"session not found: {ref}") from exc

    evidence = getattr(archive, reader_name)(session_id)
    if evidence is None:
        raise ValueError(f"session not found: {ref}")
    evidence = dict(evidence)

    summary = archive.read_summary(session_id)
    total = _non_negative_int(evidence.get("total"), default=0)
    result: dict[str, object] = {
        "outcome": decide_outcome(matched=total).to_dict(),
        "session": {
            "session_id": summary.session_id,
            "native_id": summary.native_id,
            "origin": summary.origin,
            "title": summary.title,
            "created_at": summary.created_at,
            "updated_at": summary.updated_at,
            # An evidence read is not a message window; saying so with an empty
            # list keeps the identity projection's shape without implying that
            # a transcript page of zero messages was returned.
            "messages": [],
        },
        "session_id": session_id,
        "kind": kind,
        "evidence": evidence,
        "total": total,
        # The body is whole, so its window is itself.  ``limit`` is declared
        # ``ge=1``, and a zero-row aggregate is still one delivered answer.
        "limit": max(total, 1),
        "offset": 0,
        "next_offset": None,
        "continuation": None,
        "complete": True,
    }
    _require_deliverable_window(result, limit=total)
    return result


def _require_deliverable_window(result: Mapping[str, object], *, limit: int) -> None:
    """Refuse a window the transport cannot carry, naming the way out.

    Silently truncating would make ``complete``/``next_offset`` lie about what
    the caller received.
    """

    import json

    from polylogue.operations.daemon_protocol import MAX_OPERATION_RESULT_BYTES

    size = len(json.dumps(result, separators=(",", ":"), default=str).encode())
    if size > MAX_OPERATION_RESULT_BYTES:
        raise ValueError(
            f"session.read window of {limit} messages is {size} bytes, above the "
            f"{MAX_OPERATION_RESULT_BYTES}-byte operation result bound; retry with a smaller limit"
        )


def _session_reference_payload(payload: Mapping[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    """Resolve one bare ``from <ref>`` operand against durable reference state.

    Reference definitions live in the durable user tier and their evaluation
    plan in the pinned index, so both paths come from the pinned reader rather
    than from configuration.
    """

    from contextlib import closing

    from polylogue.api.archive import open_readonly_connection
    from polylogue.archive.query.evaluator import DurableRefResolver
    from polylogue.archive.query.expression import parse_reference_query_pipeline, resolve_ref_operand
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator
    from polylogue.surfaces.outcome import decide_outcome

    expression = str(payload.get("expression") or "").strip()
    if not expression:
        raise ValueError("session.reference requires a reference expression")
    pipeline = parse_reference_query_pipeline(expression)
    if pipeline is None:
        raise ValueError(f"expression is not a reference operand: {expression!r}")
    if pipeline.stages:
        raise ValueError("reference pipeline stages are not supported; only the bare `from <ref>` operand is supported")
    if not archive.user_db_path.exists():
        raise ValueError("archive is not initialized")

    evaluator = ArchiveCanonicalPlanEvaluator(archive.index_db_path)
    try:
        with closing(
            open_readonly_connection(archive.user_db_path, timeout_class="interactive-read", validate_schema=False)
        ) as connection:
            resolved = resolve_ref_operand(pipeline.operand, DurableRefResolver(connection, evaluator))
    except KeyError as exc:
        raise ValueError(f"reference not found: {pipeline.operand.reference.format()}") from exc

    raw_limit = payload.get("limit")
    limit = None if raw_limit is None else _non_negative_int(raw_limit, default=0)
    members = list(resolved.member_refs if limit is None else resolved.member_refs[:limit])
    return {
        "outcome": decide_outcome(matched=len(resolved.member_refs)).to_dict(),
        "source": pipeline.operand.reference.format(),
        "grain": str(getattr(resolved.grain, "value", resolved.grain)),
        "lineage": [ref.format() for ref in resolved.lineage],
        "member_count": len(resolved.member_refs),
        "members": members,
        "truncated": len(members) < len(resolved.member_refs),
    }


def _lower_cli_query_params(params: Mapping[str, object]) -> tuple[dict[str, object], str]:
    """Lower CLI conveniences without importing Click's ``RootModeRequest``."""

    return lower_cli_query_params(params)


def _cli_query_spec(params: Mapping[str, object]) -> SessionQuerySpec:
    """Compile the same CLI selection contract used by canonical execution."""

    return cli_query_spec(params)


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
