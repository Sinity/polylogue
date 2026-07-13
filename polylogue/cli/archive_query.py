"""Current archive query executor for the root CLI."""

from __future__ import annotations

import csv
import io
import json
import multiprocessing
import os
import re
import webbrowser
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import redirect_stdout
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, NoReturn, TypeVar, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import click
from typing_extensions import NotRequired, TypedDict

from polylogue.archive.message.types import validate_message_type_filter
from polylogue.archive.query.attached_units import fetch_attached_units
from polylogue.archive.query.expression import (
    QueryUnitSource,
    parse_unit_source_expression,
    split_with_projection_clause,
)
from polylogue.archive.query.metadata import query_unit_descriptor
from polylogue.archive.query.predicate import QueryPredicate
from polylogue.archive.query.spec import (
    QuerySpecError,
    SessionQuerySpec,
    normalize_action_sequence,
    normalize_action_terms,
    parse_query_date,
)
from polylogue.archive.query.unit_results import query_unit_rows, query_unit_session_filters
from polylogue.archive.stats import ArchiveStats
from polylogue.cli.query_contracts import QueryOutputSpec
from polylogue.cli.query_feedback import maybe_subcommand_typo_hint
from polylogue.cli.query_output import deliver_query_output
from polylogue.cli.query_output_contracts import QueryOutputDocument
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.helpers import load_effective_config
from polylogue.cli.shared.machine_errors import error_no_results
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.paths import archive_file_set_root_for_paths
from polylogue.storage.search_providers import create_vector_provider, reciprocal_rank_fusion
from polylogue.storage.sqlite.archive_tiers.archive import (
    ArchiveSessionSearchHit,
    ArchiveSessionSummary,
    ArchiveStore,
)
from polylogue.storage.sqlite.archive_tiers.write import ArchiveSessionEnvelope
from polylogue.surfaces.payloads import (
    SEARCH_CURSOR_VERSION,
    InvalidSearchCursorError,
    MutationOperation,
    MutationResultPayload,
    SearchCursor,
    SessionListRowPayload,
    SessionSearchHitPayload,
    SessionSearchMatchPayload,
    SessionSummaryPayload,
    TargetRefPayload,
    decode_search_cursor,
    model_json_document,
    reader_anchor,
    reader_message_actions,
)

_PageRow = TypeVar("_PageRow", ArchiveSessionSummary, ArchiveSessionSearchHit)

_UNSUPPORTED_PARAM_MESSAGES: dict[str, str] = {}
_QueryUnitTextLine = Callable[[dict[str, object]], str]
_DAEMON_FAST_PATH_TIMEOUT_S = 0.75
_NATIVE_REF_RE = re.compile(r"(?=.*\d)[A-Za-z0-9][A-Za-z0-9_.:-]{11,}")
_TIMING_ENV: ContextVar[AppEnv | None] = ContextVar("archive_query_timing_env", default=None)


def _object_int(value: object) -> int:
    if value is None:
        return 0
    return int(str(value))


class _ArchiveFilterKwargs(TypedDict):
    origin: str | None
    origins: tuple[str, ...]
    excluded_origins: tuple[str, ...]
    tags: tuple[str, ...]
    excluded_tags: tuple[str, ...]
    repo_names: tuple[str, ...]
    project_refs: tuple[str, ...]
    has_types: tuple[str, ...]
    has_tool_use: bool
    has_thinking: bool
    has_paste: bool
    tool_terms: tuple[str, ...]
    excluded_tool_terms: tuple[str, ...]
    action_terms: tuple[str, ...]
    excluded_action_terms: tuple[str, ...]
    action_sequence: tuple[str, ...]
    action_text_terms: tuple[str, ...]
    referenced_paths: tuple[str, ...]
    cwd_prefix: str | None
    typed_only: bool
    message_type: str | None
    title: str | None
    min_messages: int | None
    max_messages: int | None
    min_words: int | None
    max_words: int | None
    since_ms: int | None
    until_ms: int | None
    since_session_id: str | None
    boolean_predicate: NotRequired[QueryPredicate]


def execute_delete_by_session_ids(
    env: AppEnv,
    session_ids: list[str],
    *,
    force: bool,
    dry_run: bool = False,
) -> None:
    """Delete (or preview) a known set of session IDs, bypassing the query phase.

    Used by the delete verb after cardinality resolution — the IDs are already
    known so we skip the re-query (which would be capped at the default limit
    of 20, causing ``delete --yes --all`` to truncate silently). The dry-run
    preview routes through here too so the previewed set is the *same* full
    resolved set the real delete acts on (the guard, preview, and deleted sets
    must be identical, #1873).
    """
    config = load_effective_config(env)
    archive_root = archive_file_set_root_for_paths(archive_root_path=config.archive_root, db_anchor=config.db_path)
    params: dict[str, object] = {"force": force, "delete_matched": True, "dry_run": dry_run}
    with ArchiveStore.open_existing(archive_root) as archive:
        _emit_delete(env, archive, tuple(session_ids), params=params)


def execute_archive_query(env: AppEnv, request: RootModeRequest) -> None:
    """Execute the root query path."""
    timing_token = _TIMING_ENV.set(env)
    env.begin_timing("execute")
    try:
        output = QueryOutputSpec.from_params(request.params)
        if output.destination_labels() == ("stdout",) or request.params.get("stream"):
            _execute_archive_query_stdout(env, request)
            return
        rendered = io.StringIO()
        with redirect_stdout(rendered):
            _execute_archive_query_stdout(env, request)
        deliver_query_output(
            env,
            QueryOutputDocument(
                content=rendered.getvalue().rstrip("\n"),
                output_format=output.output_format,
                destinations=output.destinations,
            ),
        )
    finally:
        env.finish_timing("execute")
        _TIMING_ENV.reset(timing_token)


def _execute_archive_query_stdout(env: AppEnv, request: RootModeRequest) -> None:
    """Render root query output to stdout."""
    params = dict(request.params)
    _reject_unsupported_params(params)
    _validate_retrieval_params(params)
    config_started_at = perf_counter()
    config = load_effective_config(env)
    archive_root = archive_file_set_root_for_paths(archive_root_path=config.archive_root, db_anchor=config.db_path)
    index_db_path = archive_root / "index.db"
    env.record_timing("config", config_started_at)
    compile_started_at = perf_counter()
    typo_hint = maybe_subcommand_typo_hint(request.query_terms)
    raw_query = _query_text(request.query_terms, params)
    output_format = str(params.get("output_format") or "markdown")
    fields = _optional_str(params.get("fields"))
    # Split a trailing ``with <units>`` projection clause off the FTS text so it
    # is not searched literally; the units drive the attached-unit projection.
    unit_source_query = raw_query
    with_units: tuple[str, ...] = ()
    with_unit_fields: dict[str, tuple[str, ...]] = {}
    if unit_source_query and not (unit_source_query.startswith("{") or unit_source_query.startswith("[")):
        unit_source_query, with_units, with_unit_fields = split_with_projection_clause(unit_source_query)
    unit_source = (
        parse_unit_source_expression(unit_source_query)
        if unit_source_query and not _optional_str(params.get("similar_text"))
        else None
    )
    compiled_spec = (
        SessionQuerySpec.from_params(params)
        if unit_source is not None
        else _compiled_session_spec(request, params=params, raw_query=raw_query)
    )
    origins = compiled_spec.origins or _resolve_origins(params)
    origin = origins[0] if len(origins) == 1 else None
    query = _query_text(compiled_spec.query_terms, {"contains": compiled_spec.contains_terms})
    if compiled_spec.with_units:
        with_units = compiled_spec.with_units
        with_unit_fields = compiled_spec.with_unit_fields
    env.record_timing("compile", compile_started_at)

    tags_to_add = _tuple_tokens(params.get("add_tag"))
    metadata_to_set = _metadata_pairs(params.get("set_meta"))
    tags = compiled_spec.tags
    excluded_tags = compiled_spec.excluded_tags
    repo_names = compiled_spec.repo_names
    project_refs = compiled_spec.project_refs
    has_types = compiled_spec.has_types
    has_tool_use = compiled_spec.filter_has_tool_use
    has_thinking = compiled_spec.filter_has_thinking
    has_paste = compiled_spec.filter_has_paste
    tool_terms = compiled_spec.tool_terms
    excluded_tool_terms = compiled_spec.excluded_tool_terms
    action_terms = compiled_spec.action_terms
    excluded_action_terms = compiled_spec.excluded_action_terms
    action_sequence = compiled_spec.action_sequence
    action_text_terms = compiled_spec.action_text_terms
    referenced_paths = compiled_spec.referenced_path
    cwd_prefix = compiled_spec.cwd_prefix
    typed_only = compiled_spec.typed_only
    message_type = compiled_spec.message_type
    title_filter = compiled_spec.title
    min_messages = compiled_spec.min_messages
    max_messages = compiled_spec.max_messages
    min_words = compiled_spec.min_words
    max_words = compiled_spec.max_words
    since_ms = _spec_date_ms("since", compiled_spec)
    until_ms = _spec_date_ms("until", compiled_spec)
    since_session_id = compiled_spec.since_session_id
    limit = compiled_spec.limit if compiled_spec.limit is not None and compiled_spec.limit > 0 else _limit(params)
    offset = compiled_spec.offset if compiled_spec.offset > 0 else _offset(params)
    cursor = _decode_cursor(_optional_str(params.get("cursor")))
    page_offset = cursor.r if cursor is not None else offset
    sample_count = _optional_int(params.get("sample"))
    if sample_count is not None:
        if cursor is not None:
            raise click.UsageError("Root query does not combine --sample with --cursor.")
        if sample_count <= 0:
            raise click.UsageError("Root query --sample must be positive.")
        limit = sample_count
        page_offset = 0
    stream = bool(params.get("stream"))
    stream_output_format = QueryOutputSpec.from_params(params).stream_format()
    sort = compiled_spec.sort
    reverse = compiled_spec.reverse
    similar_text = compiled_spec.similar_text
    retrieval_lane = _optional_str(params.get("retrieval_lane")) or compiled_spec.retrieval_lane
    delete_matched = bool(params.get("delete_matched"))
    excluded_origins = compiled_spec.excluded_origins or _resolve_excluded_origins(params)
    filter_kwargs: _ArchiveFilterKwargs = {
        "origin": origin,
        "origins": origins,
        "excluded_origins": excluded_origins,
        "tags": tags,
        "excluded_tags": excluded_tags,
        "repo_names": repo_names,
        "project_refs": project_refs,
        "has_types": has_types,
        "has_tool_use": has_tool_use,
        "has_thinking": has_thinking,
        "has_paste": has_paste,
        "tool_terms": tool_terms,
        "excluded_tool_terms": excluded_tool_terms,
        "action_terms": action_terms,
        "excluded_action_terms": excluded_action_terms,
        "action_sequence": action_sequence,
        "action_text_terms": action_text_terms,
        "referenced_paths": referenced_paths,
        "cwd_prefix": cwd_prefix,
        "typed_only": typed_only,
        "message_type": message_type,
        "title": title_filter,
        "min_messages": min_messages,
        "max_messages": max_messages,
        "min_words": min_words,
        "max_words": max_words,
        "since_ms": since_ms,
        "until_ms": until_ms,
        "since_session_id": since_session_id,
    }
    if compiled_spec.boolean_predicate is not None:
        filter_kwargs["boolean_predicate"] = compiled_spec.boolean_predicate
    if _try_emit_daemon_session_page(
        env,
        config=config,
        request=request,
        params=params,
        compiled_spec=compiled_spec,
        unit_source=unit_source,
        with_units=with_units,
        with_unit_fields=with_unit_fields,
        query=query,
        limit=limit,
        offset=page_offset,
        output_format=output_format,
        origin=origin,
        fields=fields,
        typo_hint=typo_hint,
        tags_to_add=tags_to_add,
        metadata_to_set=metadata_to_set,
        delete_matched=delete_matched,
        stream=stream,
        sample_count=sample_count,
        cursor=cursor,
        sort=sort,
        reverse=reverse,
        similar_text=similar_text,
        retrieval_lane=retrieval_lane,
    ):
        return
    if _try_emit_daemon_unit_page(
        config=config,
        request=request,
        params=params,
        source=unit_source,
        expression=unit_source_query,
        limit=limit,
        offset=page_offset,
        output_format=output_format,
        fields=fields,
        stream=stream,
        tags_to_add=tags_to_add,
        metadata_to_set=metadata_to_set,
        delete_matched=delete_matched,
    ):
        return
    if not index_db_path.exists():
        if _emit_missing_archive_empty_read(
            params,
            output_format=output_format,
            origin=origin,
            query=query,
            fields=fields,
        ):
            return
        message = f"archive index database not found at {index_db_path}"
        if typo_hint is not None:
            message = f"{message}\n{typo_hint}"
        _fail(message)

    if cursor is not None and any(
        params.get(key) for key in ("stats_only", "stats_by", "count_only", "conv_id", "latest")
    ):
        raise click.UsageError("Root query --cursor is only supported for list and search pages.")
    if retrieval_lane == "hybrid" and not query:
        raise click.UsageError("Hybrid retrieval requires lexical query terms.")
    if tags_to_add and any(params.get(key) for key in ("stats_only", "stats_by", "count_only")):
        raise click.UsageError("--add-tag is only supported for matched sessions.")
    if delete_matched and any(params.get(key) for key in ("stats_only", "stats_by", "count_only")):
        raise click.UsageError("Delete is only supported for matched sessions.")
    if delete_matched and tags_to_add:
        raise click.UsageError("Root query cannot combine delete with --add-tag.")
    if delete_matched and metadata_to_set:
        raise click.UsageError("Root query cannot combine delete with --set.")

    db_open_started_at = perf_counter()
    with ArchiveStore.open_existing(archive_root) as archive:
        env.record_timing("db-open", db_open_started_at)
        if unit_source is not None:
            if any(
                (
                    params.get("stats_only"),
                    params.get("stats_by"),
                    params.get("count_only"),
                    stream,
                    tags_to_add,
                    metadata_to_set,
                    delete_matched,
                    params.get("open_result"),
                    params.get("conv_id"),
                    sample_count is not None,
                    since_session_id is not None,
                    cursor is not None,
                    sort is not None,
                    reverse,
                )
            ):
                unit_label = _unit_source_display_name(unit_source)
                raise click.UsageError(
                    f"{unit_label} where queries return {unit_source.unit} rows and do not combine "
                    "with session-only actions, aggregate modes, sort, reverse, or cursor."
                )
            _emit_unit_source_rows(
                archive,
                source=unit_source,
                query=unit_source_query,
                limit=limit,
                offset=page_offset,
                session_filters=_unit_source_session_filters(filter_kwargs),
                output_format=output_format,
                fields=fields,
            )
            return
        if params.get("stats_only") or params.get("stats_by"):
            if sample_count is not None:
                raise click.UsageError("Root query does not combine --sample with stats.")
            aggregate_limit = _optional_int(params.get("limit"))
            session_ids = _matched_session_ids_for_stats(
                archive,
                query=query,
                limit=aggregate_limit,
                filter_kwargs=filter_kwargs,
            )
            if params.get("stats_by"):
                group_by = str(params["stats_by"])
                if query and not session_ids:
                    _emit_stats_by(
                        {},
                        group_by=group_by,
                        output_format=output_format,
                        origin=origin,
                        query=query,
                        fields=fields,
                    )
                    return
                try:
                    grouped = archive.stats_by(
                        group_by,
                        **cast(Any, _stats_filter_kwargs(filter_kwargs)),
                        session_ids=session_ids,
                    )
                except ValueError as exc:
                    raise click.UsageError(str(exc)) from exc
                _emit_stats_by(
                    grouped,
                    group_by=group_by,
                    output_format=output_format,
                    origin=origin,
                    query=query,
                    fields=fields,
                )
                return
            if query and not session_ids:
                _emit_stats(
                    ArchiveStats(total_sessions=0, total_messages=0),
                    output_format=output_format,
                    origin=origin,
                    query=query,
                    fields=fields,
                )
                return
            stats = archive.stats(
                **cast(Any, _stats_filter_kwargs(filter_kwargs)),
                session_ids=session_ids,
            )
            _emit_stats(stats, output_format=output_format, origin=origin, query=query, fields=fields)
            return
        if params.get("count_only"):
            if query:
                _emit_count(
                    archive.count_search_sessions(
                        query,
                        origin=origin,
                        origins=origins,
                        excluded_origins=excluded_origins,
                        tags=tags,
                        excluded_tags=excluded_tags,
                        repo_names=repo_names,
                        project_refs=project_refs,
                        has_types=has_types,
                        has_tool_use=has_tool_use,
                        has_thinking=has_thinking,
                        has_paste=has_paste,
                        tool_terms=tool_terms,
                        excluded_tool_terms=excluded_tool_terms,
                        action_terms=action_terms,
                        excluded_action_terms=excluded_action_terms,
                        action_sequence=action_sequence,
                        action_text_terms=action_text_terms,
                        referenced_paths=referenced_paths,
                        cwd_prefix=cwd_prefix,
                        typed_only=typed_only,
                        message_type=message_type,
                        title=title_filter,
                        min_messages=min_messages,
                        max_messages=max_messages,
                        min_words=min_words,
                        max_words=max_words,
                        since_ms=since_ms,
                        until_ms=until_ms,
                        since_session_id=since_session_id,
                        boolean_predicate=filter_kwargs.get("boolean_predicate"),
                    ),
                    output_format=output_format,
                    origin=origin,
                )
                return
            _emit_count(
                archive.count_sessions(
                    origin=origin,
                    origins=origins,
                    excluded_origins=excluded_origins,
                    tags=tags,
                    excluded_tags=excluded_tags,
                    repo_names=repo_names,
                    project_refs=project_refs,
                    has_types=has_types,
                    has_tool_use=has_tool_use,
                    has_thinking=has_thinking,
                    has_paste=has_paste,
                    tool_terms=tool_terms,
                    excluded_tool_terms=excluded_tool_terms,
                    action_terms=action_terms,
                    excluded_action_terms=excluded_action_terms,
                    action_sequence=action_sequence,
                    action_text_terms=action_text_terms,
                    referenced_paths=referenced_paths,
                    cwd_prefix=cwd_prefix,
                    typed_only=typed_only,
                    message_type=message_type,
                    title=title_filter,
                    min_messages=min_messages,
                    max_messages=max_messages,
                    min_words=min_words,
                    max_words=max_words,
                    since_ms=since_ms,
                    until_ms=until_ms,
                    since_session_id=since_session_id,
                    boolean_predicate=filter_kwargs.get("boolean_predicate"),
                ),
                output_format=output_format,
                origin=origin,
            )
            return
        conv_id = compiled_spec.session_id or params.get("conv_id")
        if conv_id:
            try:
                session_id = archive.resolve_session_id(str(conv_id))
                if query or similar_text:
                    if sample_count is not None:
                        raise click.UsageError("Root query does not combine --sample with search terms.")
                    hits, resolved_lane = _query_hits(
                        archive,
                        config=config,
                        query=query,
                        similar_text=similar_text,
                        retrieval_lane=retrieval_lane,
                        limit=limit + 1,
                        offset=page_offset,
                        sort=sort,
                        reverse=reverse,
                        session_id=session_id,
                        filter_kwargs=filter_kwargs,
                    )
                    page_hits, next_cursor = _paginate_rows(
                        hits,
                        limit=limit,
                        offset=page_offset,
                        retrieval_lane=resolved_lane,
                    )
                    if stream:
                        if not page_hits:
                            _fail("Stream found no matching session.")
                        envelope = archive.read_session(session_id)
                        _emit_stream(
                            envelope,
                            output_format=stream_output_format,
                            message_limit=_stream_message_limit(params),
                        )
                        return
                    matched_session_ids = (session_id,) if page_hits else ()
                    if tags_to_add or metadata_to_set:
                        _emit_user_mutations(
                            archive,
                            matched_session_ids,
                            tags_to_add=tags_to_add,
                            metadata_to_set=metadata_to_set,
                        )
                        return
                    if delete_matched:
                        _emit_delete(env, archive, matched_session_ids, params=params)
                        return
                    if params.get("open_result"):
                        if not page_hits:
                            _emit_open_no_results(output_format=output_format, origin=origin)
                        _open_session(
                            env,
                            session_id,
                            output_format=output_format,
                            print_url=bool(params.get("print_url")),
                        )
                        return
                    _emit_search(
                        page_hits,
                        archive=archive,
                        query=similar_text or query,
                        limit=limit,
                        offset=page_offset,
                        next_cursor=next_cursor,
                        retrieval_lane=resolved_lane,
                        output_format=output_format,
                        origin=origin,
                        fields=fields,
                        typo_hint=typo_hint,
                        with_units=with_units,
                        with_unit_fields=with_unit_fields,
                    )
                    return
                if stream:
                    envelope = archive.read_session(session_id)
                    _emit_stream(
                        envelope,
                        output_format=stream_output_format,
                        message_limit=_stream_message_limit(params),
                    )
                    return
                if tags_to_add or metadata_to_set:
                    _emit_user_mutations(
                        archive, (session_id,), tags_to_add=tags_to_add, metadata_to_set=metadata_to_set
                    )
                    return
                if delete_matched:
                    _emit_delete(env, archive, (session_id,), params=params)
                    return
                if params.get("open_result"):
                    _open_session(env, session_id, output_format=output_format, print_url=bool(params.get("print_url")))
                    return
                envelope = archive.read_session(session_id)
            except KeyError:
                _fail(f"Session not found: {conv_id}")
            except ValueError as exc:
                raise click.UsageError(str(exc)) from exc
            _emit_session(
                envelope,
                output_format=output_format,
                fields=fields,
            )
            return
        if query and not similar_text:
            try:
                exact_session_id = _resolve_single_query_ref(archive, query)
            except ValueError as exc:
                raise click.UsageError(str(exc)) from exc
            if exact_session_id is not None:
                envelope = archive.read_session(exact_session_id)
                _emit_session(
                    envelope,
                    output_format=output_format,
                    fields=fields,
                )
                return
        if query or similar_text:
            if sample_count is not None:
                raise click.UsageError("Root query does not combine --sample with search terms.")
            fetch_limit = limit + 1
            hits, resolved_lane = _query_hits(
                archive,
                config=config,
                query=query,
                similar_text=similar_text,
                retrieval_lane=retrieval_lane,
                limit=fetch_limit,
                offset=page_offset,
                sort=sort,
                reverse=reverse,
                session_id=None,
                filter_kwargs=filter_kwargs,
            )
            page_hits, next_cursor = _paginate_rows(
                hits,
                limit=limit,
                offset=page_offset,
                retrieval_lane=resolved_lane,
            )
            if stream:
                if not page_hits:
                    _fail("Stream found no matching session.")
                envelope = archive.read_session(page_hits[0].session_id)
                _emit_stream(
                    envelope,
                    output_format=stream_output_format,
                    message_limit=_stream_message_limit(params),
                )
                return
            if tags_to_add or metadata_to_set:
                session_ids = tuple(hit.session_id for hit in page_hits)
                _emit_user_mutations(archive, session_ids, tags_to_add=tags_to_add, metadata_to_set=metadata_to_set)
                return
            if delete_matched:
                session_ids = tuple(hit.session_id for hit in page_hits)
                _emit_delete(env, archive, session_ids, params=params)
                return
            if params.get("open_result"):
                if not page_hits:
                    _fail("Open found no matching session.")
                _open_session(
                    env,
                    page_hits[0].session_id,
                    output_format=output_format,
                    print_url=bool(params.get("print_url")),
                )
                return
            _emit_search(
                page_hits,
                archive=archive,
                query=similar_text or query,
                limit=limit,
                offset=page_offset,
                next_cursor=next_cursor,
                retrieval_lane=resolved_lane,
                output_format=output_format,
                origin=origin,
                fields=fields,
                typo_hint=typo_hint,
                with_units=with_units,
                with_unit_fields=with_unit_fields,
            )
            return
        if params.get("latest"):
            limit = 1
        fetch_limit = limit if sample_count is not None else limit + 1
        summaries = archive.list_summaries(
            limit=fetch_limit,
            offset=page_offset,
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            project_refs=project_refs,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title_filter,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            since_session_id=since_session_id,
            sample=sample_count is not None,
            sort=sort,
            reverse=reverse,
        )
        page_summaries, next_cursor = _paginate_rows(summaries, limit=limit, offset=page_offset)
        if stream:
            if not page_summaries:
                _fail("Stream found no matching session.")
            envelope = archive.read_session(page_summaries[0].session_id)
            _emit_stream(
                envelope,
                output_format=stream_output_format,
                message_limit=_stream_message_limit(params),
            )
            return
        if tags_to_add or metadata_to_set:
            session_ids = tuple(summary.session_id for summary in page_summaries)
            _emit_user_mutations(archive, session_ids, tags_to_add=tags_to_add, metadata_to_set=metadata_to_set)
            return
        if delete_matched:
            session_ids = tuple(summary.session_id for summary in page_summaries)
            _emit_delete(env, archive, session_ids, params=params)
            return
        if params.get("open_result"):
            if not page_summaries:
                _emit_open_no_results(output_format=output_format, origin=origin)
            _open_session(
                env,
                page_summaries[0].session_id,
                output_format=output_format,
                print_url=bool(params.get("print_url")),
            )
            return
        _emit_list(
            page_summaries,
            limit=limit,
            offset=page_offset,
            next_cursor=next_cursor,
            output_format=output_format,
            origin=origin,
            fields=fields,
            archive=archive,
            with_units=with_units,
            with_unit_fields=with_unit_fields,
        )


def _reject_unsupported_params(params: dict[str, object]) -> None:
    for key, message in _UNSUPPORTED_PARAM_MESSAGES.items():
        if _has_value(params.get(key)):
            raise click.UsageError(message)


def _validate_retrieval_params(params: dict[str, object]) -> None:
    lane = _optional_str(params.get("retrieval_lane"))
    if lane not in {None, "auto", "dialogue", "semantic", "hybrid"}:
        raise click.UsageError("Root query retrieval lane must be auto, dialogue, semantic, or hybrid.")


def _query_hits(
    archive: ArchiveStore,
    *,
    config: Config,
    query: str,
    similar_text: str | None,
    retrieval_lane: str,
    limit: int,
    offset: int,
    sort: str | None,
    reverse: bool,
    session_id: str | None,
    filter_kwargs: _ArchiveFilterKwargs,
) -> tuple[list[ArchiveSessionSearchHit], str]:
    archive_root = archive_file_set_root_for_paths(archive_root_path=config.archive_root, db_anchor=config.db_path)
    embeddings_db = archive_root / "embeddings.db"
    # ``auto`` is intentionally lexical. A default ``find`` must not open or scan
    # embeddings.db before returning FTS results; large active archives can make
    # that probe block in I/O wait. Vector retrieval remains explicit through
    # ``--semantic`` / ``--similar`` / ``--retrieval-lane hybrid``.
    if similar_text is None and retrieval_lane in {"auto", "dialogue"}:
        return (
            archive.search_summaries(
                query,
                limit=limit,
                offset=offset,
                sort=sort,
                reverse=reverse,
                session_id=session_id,
                **filter_kwargs,
            ),
            "dialogue",
        )
    if retrieval_lane == "hybrid" and not query:
        raise click.UsageError("Hybrid retrieval requires lexical query terms.")
    vector_provider = create_vector_provider(config, db_path=embeddings_db)
    if vector_provider is None:
        raise click.UsageError("Vector retrieval requires configured sqlite-vec and Voyage embeddings.")
    semantic_query = similar_text or query
    semantic_scored = vector_provider.query(semantic_query, limit=max(limit + offset, limit) * 3)
    semantic_hits = archive.semantic_summaries(
        semantic_scored,
        limit=max(limit + offset, limit) * 3,
        offset=0,
        session_id=session_id,
        **filter_kwargs,
    )
    if retrieval_lane != "hybrid":
        return semantic_hits[offset : offset + limit], "semantic"

    lexical_hits = archive.search_summaries(
        query,
        limit=max(limit + offset, limit) * 3,
        offset=0,
        sort=sort,
        reverse=reverse,
        session_id=session_id,
        **filter_kwargs,
    )
    hit_by_session: dict[str, ArchiveSessionSearchHit] = {}
    for hit in [*lexical_hits, *semantic_hits]:
        hit_by_session.setdefault(hit.session_id, hit)
    fused = reciprocal_rank_fusion(
        [(hit.session_id, 0.0) for hit in lexical_hits],
        [(hit.session_id, 0.0) for hit in semantic_hits],
    )
    page = fused[offset : offset + limit]
    return [
        replace(hit_by_session[session_id], rank=offset + index)
        for index, (session_id, _score) in enumerate(page, start=1)
        if session_id in hit_by_session
    ], "hybrid"


def _try_emit_daemon_session_page(
    env: AppEnv,
    *,
    config: Config,
    request: RootModeRequest,
    params: dict[str, object],
    compiled_spec: SessionQuerySpec,
    unit_source: QueryUnitSource | None,
    with_units: tuple[str, ...],
    with_unit_fields: dict[str, tuple[str, ...]],
    query: str,
    limit: int,
    offset: int,
    output_format: str,
    origin: str | None,
    fields: str | None,
    typo_hint: str | None,
    tags_to_add: tuple[str, ...],
    metadata_to_set: tuple[tuple[str, str], ...],
    delete_matched: bool,
    stream: bool,
    sample_count: int | None,
    cursor: SearchCursor | None,
    sort: str | None,
    reverse: bool,
    similar_text: str | None,
    retrieval_lane: str,
) -> bool:
    """Use the running daemon for ordinary session pages when it is safe.

    The daemon already owns the web-reader `/api/sessions` contract.  This
    adapter lets the CLI reuse that route for the common list/search case while
    retaining local ArchiveStore execution for mutations, streaming, unit rows,
    stats, vector search, and features the HTTP route does not yet represent.
    """
    if compiled_spec.session_id is not None or _single_query_token_looks_like_ref(query):
        return False
    if not _daemon_session_page_supported(
        params,
        compiled_spec=compiled_spec,
        unit_source=unit_source,
        with_units=with_units,
        with_unit_fields=with_unit_fields,
        tags_to_add=tags_to_add,
        metadata_to_set=metadata_to_set,
        delete_matched=delete_matched,
        stream=stream,
        sample_count=sample_count,
        cursor=cursor,
        sort=sort,
        reverse=reverse,
        similar_text=similar_text,
        retrieval_lane=retrieval_lane,
    ):
        return False
    if bool(params.get("no_daemon")):
        return False
    daemon_params = _daemon_session_query_params(request, params, limit=limit, offset=offset)
    payload = _fetch_daemon_sessions_payload(config, daemon_params)
    if payload is None:
        return False
    if isinstance(payload.get("hits"), list):
        _emit_daemon_search_payload(
            payload,
            query=query or str(daemon_params.get("query") or ""),
            limit=limit,
            offset=offset,
            output_format=output_format,
            origin=origin,
            fields=fields,
            typo_hint=typo_hint,
        )
        return True
    if isinstance(payload.get("items"), list):
        _emit_daemon_list_payload(
            payload,
            limit=limit,
            offset=offset,
            output_format=output_format,
            origin=origin,
            fields=fields,
        )
        return True
    return False


def _try_emit_daemon_unit_page(
    *,
    config: Config,
    request: RootModeRequest,
    params: dict[str, object],
    source: QueryUnitSource | None,
    expression: str,
    limit: int,
    offset: int,
    output_format: str,
    fields: str | None,
    stream: bool,
    tags_to_add: tuple[str, ...],
    metadata_to_set: tuple[tuple[str, str], ...],
    delete_matched: bool,
) -> bool:
    """Render daemon query-unit envelopes with the existing CLI renderer."""

    if source is None or stream or tags_to_add or metadata_to_set or delete_matched:
        return False
    daemon_params = _daemon_session_query_params(request, params, limit=limit, offset=offset)
    daemon_params["expression"] = expression
    payload = _fetch_daemon_payload(
        config,
        "/api/query-units?" + urlencode(tuple(_daemon_query_pairs(daemon_params)), doseq=True),
        disabled=bool(params.get("no_daemon")),
    )
    if payload is None:
        return False
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        return False
    items = [item for item in raw_items if isinstance(item, dict)]
    if not items:
        _emit_unit_no_results(payload, unit=source.unit, output_format=output_format)
    text_line = (
        _aggregate_query_line if payload.get("mode") == "query-unit-aggregate" else _query_unit_text_line(source.unit)
    )
    _emit_rows(payload, items, output_format=output_format, text_line=text_line, fields=fields)
    return True


def _resolve_single_query_ref(archive: ArchiveStore, query: str) -> str | None:
    """Resolve a singleton query token as a session ref before FTS fallback."""
    if not _single_query_token_looks_like_ref(query):
        return None
    try:
        return archive.resolve_session_id(query)
    except KeyError:
        return None
    except ValueError as exc:
        # ``repo:polylogue`` and other structured field clauses can look
        # ref-shaped to the cheap syntactic probe. They are not identity queries;
        # let normal DSL/list execution handle them. Ambiguous suffix/prefix
        # matches are real identity failures and should not broaden to FTS.
        if "ambiguous" in str(exc):
            raise
        return None


def _single_query_token_looks_like_ref(query: str) -> bool:
    token = query.strip()
    return bool(token and " " not in token and (":" in token or _NATIVE_REF_RE.fullmatch(token)))


def _daemon_session_page_supported(
    params: dict[str, object],
    *,
    compiled_spec: SessionQuerySpec,
    unit_source: QueryUnitSource | None,
    with_units: tuple[str, ...],
    with_unit_fields: dict[str, tuple[str, ...]],
    tags_to_add: tuple[str, ...],
    metadata_to_set: tuple[tuple[str, str], ...],
    delete_matched: bool,
    stream: bool,
    sample_count: int | None,
    cursor: SearchCursor | None,
    sort: str | None,
    reverse: bool,
    similar_text: str | None,
    retrieval_lane: str,
) -> bool:
    if unit_source is not None or with_units or with_unit_fields:
        return False
    if any(params.get(key) for key in ("stats_only", "stats_by", "count_only", "conv_id", "latest", "open_result")):
        return False
    if stream or tags_to_add or metadata_to_set or delete_matched:
        return False
    if sample_count is not None or cursor is not None or sort is not None or reverse:
        return False
    if similar_text is not None or retrieval_lane not in {"auto", "dialogue"}:
        return False
    if compiled_spec.boolean_predicate is not None or compiled_spec.since_session_id is not None:
        return False
    return not (compiled_spec.project_refs or compiled_spec.typed_only or compiled_spec.message_type is not None)


def _daemon_session_query_params(
    request: RootModeRequest,
    params: dict[str, object],
    *,
    limit: int,
    offset: int,
) -> dict[str, object]:
    query_params: dict[str, object] = {"limit": limit, "offset": offset}
    raw_query = " ".join(term for term in request.query_terms if term).strip()
    if raw_query:
        query_params["query"] = raw_query
    for source_key, dest_key in (
        ("contains", "contains"),
        ("origin", "origin"),
        ("exclude_origin", "exclude_origin"),
        ("tag", "tag"),
        ("exclude_tag", "exclude_tag"),
        ("repo", "repo"),
        ("has_type", "has_type"),
        ("tool", "tool"),
        ("exclude_tool", "exclude_tool"),
        ("action", "action"),
        ("exclude_action", "exclude_action"),
        ("action_sequence", "action_sequence"),
        ("action_text", "action_text"),
        ("referenced_path", "referenced_path"),
        ("cwd_prefix", "cwd_prefix"),
        ("title", "title"),
        ("min_messages", "min_messages"),
        ("max_messages", "max_messages"),
        ("min_words", "min_words"),
        ("max_words", "max_words"),
        ("since", "since"),
        ("until", "until"),
    ):
        value = params.get(source_key)
        if _has_value(value):
            query_params[dest_key] = value
    for source_key, dest_key in (
        ("has_paste", "has_paste_evidence"),
        ("has_tool_use", "has_tool_use"),
        ("has_thinking", "has_thinking"),
    ):
        if bool(params.get(source_key)):
            query_params[dest_key] = "1"
    return query_params


def _daemon_disabled(*, flag: bool = False) -> bool:
    if flag:
        return True
    if os.environ.get("POLYLOGUE_NO_DAEMON", "").lower() in {"1", "true", "yes", "on"}:
        return True
    return os.environ.get("POLYLOGUE_DAEMON", "").lower() == "off"


def _fetch_daemon_sessions_payload(
    config: Config,
    query_params: Mapping[str, object],
    *,
    disabled: bool = False,
) -> dict[str, object] | None:
    return _fetch_daemon_payload(config, "/api/cli/query", body={"params": dict(query_params)}, disabled=disabled)


def _fetch_daemon_payload(
    config: Config,
    path: str,
    *,
    body: dict[str, object] | None = None,
    disabled: bool = False,
) -> dict[str, object] | None:
    if _daemon_disabled(flag=disabled):
        return None
    from polylogue.cli.daemon_client import DaemonClient
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
    from polylogue.version import POLYLOGUE_VERSION

    socket_path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "polylogue" / "daemon.sock"
    client = DaemonClient(socket_path)
    if (
        client.probe(
            archive_root=str(config.archive_root),
            index_schema_version=INDEX_SCHEMA_VERSION,
            daemon_version=POLYLOGUE_VERSION,
        )
        is None
    ):
        return None
    payload = client.request_json("POST" if body is not None else "GET", path, body)
    if payload is not None:
        return payload
    return None


def _fetch_daemon_sessions_payload_with_deadline(
    daemon_url: str,
    auth_token: str | None,
    query_params: Mapping[str, object],
    *,
    expected_archive_root: Path | None = None,
) -> dict[str, object] | None:
    ctx = multiprocessing.get_context("fork")
    queue: multiprocessing.Queue[dict[str, object] | None] = ctx.Queue(maxsize=1)
    worker = ctx.Process(
        target=_fetch_daemon_sessions_payload_worker,
        args=(daemon_url, auth_token, dict(query_params), expected_archive_root, queue),
        daemon=True,
    )
    worker.start()
    worker.join(_DAEMON_FAST_PATH_TIMEOUT_S)
    if worker.is_alive():
        worker.terminate()
        worker.join(0.2)
        return None
    if worker.exitcode != 0:
        return None
    try:
        payload = queue.get_nowait()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _fetch_daemon_sessions_payload_worker(
    daemon_url: str,
    auth_token: str | None,
    query_params: Mapping[str, object],
    expected_archive_root: Path | None,
    queue: multiprocessing.Queue[dict[str, object] | None],
) -> None:
    queue.put(
        _fetch_daemon_sessions_payload_once(
            daemon_url,
            auth_token,
            query_params,
            expected_archive_root=expected_archive_root,
        )
    )


def _fetch_daemon_sessions_payload_once(
    daemon_url: str,
    auth_token: str | None,
    query_params: Mapping[str, object],
    *,
    expected_archive_root: Path | None = None,
) -> dict[str, object] | None:
    if expected_archive_root is not None and not _daemon_matches_archive_root(
        daemon_url,
        auth_token,
        expected_archive_root,
    ):
        return None
    query_string = urlencode(tuple(_daemon_query_pairs(query_params)), doseq=True)
    url = f"{daemon_url}/api/sessions"
    if query_string:
        url = f"{url}?{query_string}"
    headers = {"Accept": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    try:
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=_DAEMON_FAST_PATH_TIMEOUT_S) as resp:
            if int(resp.status) != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _daemon_matches_archive_root(daemon_url: str, auth_token: str | None, expected_archive_root: Path) -> bool:
    status = _fetch_daemon_status_once(daemon_url, auth_token)
    if not isinstance(status, Mapping):
        return False
    active_root = status.get("active_archive_root")
    if not isinstance(active_root, str) or not active_root:
        return False
    try:
        return Path(active_root).expanduser().resolve() == expected_archive_root.expanduser().resolve()
    except OSError:
        return False


def _fetch_daemon_status_once(daemon_url: str, auth_token: str | None) -> dict[str, object] | None:
    headers = {"Accept": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    try:
        req = Request(f"{daemon_url}/api/status", headers=headers, method="GET")
        with urlopen(req, timeout=_DAEMON_FAST_PATH_TIMEOUT_S) as resp:
            if int(resp.status) != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _daemon_query_pairs(query_params: Mapping[str, object]) -> Iterable[tuple[str, str]]:
    for key, value in query_params.items():
        if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
            for item in value:
                if _has_value(item):
                    yield key, str(item)
        elif _has_value(value):
            yield key, str(value)


def _emit_daemon_list_payload(
    payload: Mapping[str, object],
    *,
    limit: int,
    offset: int,
    output_format: str,
    origin: str | None,
    fields: str | None,
) -> None:
    items = [dict(item) for item in cast(list[object], payload.get("items") or []) if isinstance(item, Mapping)]
    total = _object_int(payload.get("total") or len(items))
    next_offset = offset + limit if total > offset + limit else None
    envelope: dict[str, object] = {
        "mode": "list",
        "origin": origin,
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
        "next_cursor": None,
        "source": "daemon",
    }
    _emit_rows(envelope, items, output_format=output_format, text_line=_summary_line, fields=fields)


def _emit_daemon_search_payload(
    payload: Mapping[str, object],
    *,
    query: str,
    limit: int,
    offset: int,
    output_format: str,
    origin: str | None,
    fields: str | None,
    typo_hint: str | None,
) -> None:
    _emit_degraded_daemon_search_payload(payload, query=query, output_format=output_format, fields=fields)
    hits = [dict(item) for item in cast(list[object], payload.get("hits") or []) if isinstance(item, Mapping)]
    total = _object_int(payload.get("total") or len(hits))
    envelope: dict[str, object] = {
        "mode": "search",
        "origin": origin,
        "query": query,
        "retrieval_lane": str(payload.get("retrieval_lane") or "dialogue"),
        "items": hits,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": None,
        "next_cursor": None,
        "source": "daemon",
    }
    if not hits:
        _emit_no_results(envelope, output_format=output_format, typo_hint=typo_hint)
    _emit_rows(envelope, hits, output_format=output_format, text_line=_hit_line, fields=fields)


def _emit_degraded_daemon_search_payload(
    payload: Mapping[str, object],
    *,
    query: str,
    output_format: str,
    fields: str | None,
) -> None:
    route_state = payload.get("route_state")
    if not isinstance(route_state, Mapping) or route_state.get("state") != "degraded":
        return
    reason = str(route_state.get("reason") or "Search index unavailable.")
    envelope: dict[str, object] = {
        "mode": "search",
        "query": query,
        "retrieval_lane": str(payload.get("retrieval_lane") or "dialogue"),
        "items": [],
        "total": None,
        "source": "daemon",
        "route_state": dict(route_state),
    }
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        envelope["diagnostics"] = dict(diagnostics)
    if output_format in {"json", "yaml"}:
        _emit_rows(envelope, [], output_format=output_format, text_line=_hit_line, fields=fields)
    elif output_format in {"ndjson", "csv"}:
        pass
    else:
        click.echo(reason, err=True)
    raise SystemExit(1)


def _decode_cursor(token: str | None) -> SearchCursor | None:
    if token is None:
        return None
    try:
        cursor = decode_search_cursor(token)
    except InvalidSearchCursorError as exc:
        raise click.UsageError(f"invalid --cursor: {exc}") from exc
    return cursor


def _paginate_rows(
    rows: Sequence[_PageRow],
    *,
    limit: int,
    offset: int,
    retrieval_lane: str = "dialogue",
) -> tuple[list[_PageRow], str | None]:
    page = list(rows[:limit])
    if len(rows) <= limit or not page:
        return page, None
    return page, _build_cursor(page[-1], rank=offset + len(page), retrieval_lane=retrieval_lane)


def _build_cursor(row: ArchiveSessionSummary | ArchiveSessionSearchHit, *, rank: int, retrieval_lane: str) -> str:
    import base64

    cursor = SearchCursor(
        v=SEARCH_CURSOR_VERSION,
        r=rank,
        s=None,
        c=row.session_id,
        lane=retrieval_lane,
    )
    payload = cursor.model_dump_json(by_alias=True)
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def _open_session(env: AppEnv, session_id: str, *, output_format: str, print_url: bool) -> None:
    daemon_url = str(getattr(env, "daemon_url", None) or "http://127.0.0.1:8766").rstrip("/")
    web_url = f"{daemon_url}/?session={quote(session_id, safe='')}"
    if print_url:
        if output_format == "json":
            click.echo(json.dumps({"url": web_url}, indent=2))
        else:
            click.echo(web_url)
        return

    webbrowser.open(web_url)
    env.ui.console.print(f"Opened: {web_url}")


def _stream_message_limit(params: dict[str, object]) -> int | None:
    value = params.get("limit")
    if isinstance(value, int) and value > 0:
        return value
    return None


def _emit_stream(
    envelope: ArchiveSessionEnvelope,
    *,
    output_format: str,
    message_limit: int | None,
) -> None:
    messages = envelope.messages[:message_limit] if message_limit is not None else envelope.messages
    payload = _session_payload(
        ArchiveSessionEnvelope(
            session_id=envelope.session_id,
            native_id=envelope.native_id,
            origin=envelope.origin,
            title=envelope.title,
            active_leaf_message_id=envelope.active_leaf_message_id,
            messages=tuple(messages),
        )
    )
    raw_messages = payload["messages"]
    if not isinstance(raw_messages, list):
        raise TypeError("session payload messages must be a list")
    if output_format in {"json", "json-lines", "ndjson"}:
        for message in raw_messages:
            click.echo(json.dumps(message, sort_keys=True))
        return
    if output_format not in {"markdown", "plaintext"}:
        raise click.UsageError(f"Stream does not support --format {output_format}.")
    lines: list[str] = []
    for message in messages:
        lines.append(f"## {message.role}")
        text = "\n".join(block.text or "" for block in message.blocks if block.text)
        lines.append(text)
        lines.append("")
    click.echo("\n".join(lines).rstrip())


def _has_value(value: object) -> bool:
    if value is None or value is False:
        return False
    return not (value == "" or value == () or value == [])


def _resolve_origins(params: dict[str, object]) -> tuple[str, ...]:
    origin = params.get("origin")
    if not origin:
        return ()
    return tuple(dict.fromkeys(token.strip() for token in str(origin).split(",") if token.strip()))


def _resolve_excluded_origins(params: dict[str, object]) -> tuple[str, ...]:
    explicit_excluded = params.get("exclude_origin")
    if not explicit_excluded:
        return ()
    return tuple(token.strip() for token in str(explicit_excluded).split(",") if token.strip())


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tags(value: object) -> tuple[str, ...]:
    return _csv_tokens(value)


def _csv_tokens(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    # Click ``multiple=True`` options arrive as a (possibly empty) tuple/list;
    # each element may itself be a comma-separated string. Tokenize per element so
    # an empty tuple yields no tokens — str(()) would otherwise inject "()" and
    # turn every empty multi-option into a filter that matches nothing.
    elements: tuple[object, ...] = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    return tuple(token.strip() for element in elements for token in str(element).split(",") if token.strip())


def _tuple_tokens(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Iterable):
        return tuple(str(token).strip() for token in value if str(token).strip())
    return (str(value).strip(),) if str(value).strip() else ()


def _metadata_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    pairs: list[tuple[str, str]] = []
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        for item in value:
            if isinstance(item, Sequence) and not isinstance(item, str | bytes) and len(item) >= 2:
                pairs.append((str(item[0]), str(item[1])))
            else:
                raise click.UsageError("--set expects key/value pairs.")
    else:
        raise click.UsageError("--set expects key/value pairs.")
    return tuple(pairs)


def _tool_tokens(value: object) -> tuple[str, ...]:
    return tuple(token.lower() for token in _csv_tokens(value))


def _action_tokens(field: str, value: object) -> tuple[str, ...]:
    try:
        return normalize_action_terms(field, value)
    except QuerySpecError as exc:
        raise click.UsageError(f"invalid {exc.field}: {exc.value}") from exc


def _action_sequence_tokens(value: object) -> tuple[str, ...]:
    try:
        return normalize_action_sequence("action_sequence", value)
    except QuerySpecError as exc:
        raise click.UsageError(f"invalid {exc.field}: {exc.value}") from exc


def _message_type(value: object) -> str | None:
    if not value:
        return None
    try:
        return validate_message_type_filter(value).value
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _sort(value: object) -> str | None:
    if not value:
        return None
    sort = str(value)
    if sort not in {"date", "messages", "words", "longest", "tokens", "random"}:
        raise click.UsageError("Root query sort must be one of date, messages, words, longest, tokens, random.")
    return sort


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _optional_date_ms(field: str, value: object) -> int | None:
    if not value:
        return None
    try:
        parsed = parse_query_date(field, str(value))
    except QuerySpecError as exc:
        raise click.ClickException(f"Cannot parse date: {exc.value!r}") from exc
    if parsed is None:
        return None
    return int(parsed.timestamp() * 1000)


def _spec_date_ms(field: str, spec: SessionQuerySpec) -> int | None:
    value = spec.since if field == "since" else spec.until
    if value is None:
        return None
    try:
        parsed = parse_query_date(field, value)
    except QuerySpecError as exc:
        raise click.ClickException(f"Cannot parse date: {exc.value!r}") from exc
    if parsed is None:
        return None
    return int(parsed.timestamp() * 1000)


def _limit(params: dict[str, object]) -> int:
    value = params.get("limit")
    if isinstance(value, int) and value > 0:
        return value
    return 20


def _offset(params: dict[str, object]) -> int:
    value = params.get("offset")
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _query_text(query_terms: tuple[str, ...], params: dict[str, object]) -> str:
    terms = [term for term in query_terms if term]
    contains = params.get("contains")
    if isinstance(contains, Iterable) and not isinstance(contains, str | bytes):
        terms.extend(str(term) for term in contains if term)
    return " ".join(terms).strip()


def _compiled_session_spec(request: RootModeRequest, *, params: dict[str, object], raw_query: str) -> SessionQuerySpec:
    """Compile CLI selection terms while preserving CLI-only semantic lane spelling."""
    if params.get("retrieval_lane") != "semantic":
        return request.query_spec()
    spec_params = dict(params)
    spec_params["retrieval_lane"] = "auto"
    if raw_query and not spec_params.get("similar_text"):
        spec_params["similar_text"] = raw_query
    return RootModeRequest(params=spec_params, query_terms=()).query_spec()


def _emit_missing_archive_empty_read(
    params: dict[str, object],
    *,
    output_format: str,
    origin: str | None,
    query: str,
    fields: str | None,
) -> bool:
    if params.get("count_only"):
        _emit_count(0, output_format=output_format, origin=origin)
        return True
    if params.get("stats_by"):
        _emit_stats_by(
            {},
            group_by=str(params["stats_by"]),
            output_format=output_format,
            origin=origin,
            query=query,
            fields=fields,
        )
        return True
    if params.get("stats_only"):
        _emit_stats(
            ArchiveStats(total_sessions=0, total_messages=0),
            output_format=output_format,
            origin=origin,
            query=query,
            fields=fields,
        )
        return True
    if params.get("list_mode") and not query:
        _emit_list(
            [],
            limit=_limit(params),
            offset=_offset(params),
            next_cursor=None,
            output_format=output_format,
            origin=origin,
            fields=fields,
        )
        return True
    return False


def _emit_count(count: int, *, output_format: str, origin: str | None) -> None:
    if output_format == "json":
        click.echo(json.dumps({"mode": "count", "origin": origin, "count": count}, indent=2))
        return
    click.echo(count)


def _matched_session_ids_for_stats(
    archive: ArchiveStore,
    *,
    query: str,
    limit: int | None,
    filter_kwargs: _ArchiveFilterKwargs,
) -> tuple[str, ...]:
    if not query:
        return ()
    return archive.search_session_ids(query, limit=limit, **filter_kwargs)


def _stats_filter_kwargs(filter_kwargs: _ArchiveFilterKwargs) -> dict[str, object]:
    stats_kwargs = dict(filter_kwargs)
    stats_kwargs.pop("boolean_predicate", None)
    return stats_kwargs


def _emit_stats(
    stats: ArchiveStats,
    *,
    output_format: str,
    origin: str | None,
    query: str,
    fields: str | None,
) -> None:
    from polylogue.cli.convergence_feedback import convergence_warning_line

    convergence_warning = convergence_warning_line()
    payload = {
        "mode": "stats",
        "origin": origin,
        "query": query or None,
        **stats.to_dict(),
    }
    if convergence_warning is not None:
        payload["archive_converging"] = True
        payload["convergence_warning"] = convergence_warning
    if output_format == "json":
        click.echo(json.dumps(_project_payload(payload, fields), indent=2, sort_keys=True))
        return
    if output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(_project_payload(payload, fields), sort_keys=False, allow_unicode=True), nl=False)
        return
    if output_format not in {"markdown", "plaintext"}:
        raise click.UsageError(f"Stats do not support --format {output_format}.")
    lines = [
        f"Sessions: {stats.total_sessions}",
        f"Messages: {stats.total_messages}",
        f"Attachments: {stats.total_attachments}",
        f"Origins: {stats.origin_count}",
        f"Average messages: {stats.avg_messages_per_session:.1f}",
    ]
    if convergence_warning is not None:
        lines.insert(0, convergence_warning)
    click.echo("\n".join(lines))


def _emit_stats_by(
    grouped: dict[str, int],
    *,
    group_by: str,
    output_format: str,
    origin: str | None,
    query: str,
    fields: str | None,
) -> None:
    items = [{"group": key, "count": count} for key, count in grouped.items()]
    envelope: dict[str, object] = {
        "mode": "stats_by",
        "group_by": group_by,
        "origin": origin,
        "query": query or None,
        "items": items,
        "total": sum(grouped.values()),
    }
    _emit_rows(envelope, items, output_format=output_format, text_line=_stats_by_line, fields=fields)


def _emit_mutation(changed: int, *, operation: MutationOperation) -> None:
    click.echo(
        MutationResultPayload(status="ok", operation=operation, affected_count=changed).to_json(exclude_none=True)
    )


def _emit_user_mutations(
    archive: ArchiveStore,
    session_ids: tuple[str, ...],
    *,
    tags_to_add: tuple[str, ...],
    metadata_to_set: tuple[tuple[str, str], ...],
) -> None:
    changes: dict[str, int] = {}
    if metadata_to_set:
        changes["metadata"] = archive.set_user_metadata(session_ids, metadata_to_set)
    if tags_to_add:
        changes["tags"] = archive.add_user_tags(session_ids, tags_to_add)
    if set(changes) == {"tags"}:
        _emit_mutation(changes["tags"], operation="add_tag")
        return
    if set(changes) == {"metadata"}:
        _emit_mutation(changes["metadata"], operation="set_meta")
        return
    # Combined tag+metadata mutation: ``tag_count`` carries the number of
    # sessions that had a tag added, ``applied_count`` the number that had
    # metadata set (the two halves of the former ``changed`` dict).
    click.echo(
        MutationResultPayload(
            status="ok",
            operation="mutate",
            tag_count=changes.get("tags"),
            applied_count=changes.get("metadata"),
        ).to_json(exclude_none=True)
    )


def _emit_delete(
    env: AppEnv, archive: ArchiveStore, session_ids: tuple[str, ...], *, params: dict[str, object]
) -> None:
    dry_run = bool(params.get("dry_run"))
    force = bool(params.get("force"))
    count = len(session_ids)
    if dry_run:
        # ``session_count`` = matched, ``affected_count`` = deleted (0 in a
        # preview); ``session_ids`` enumerates the sessions that would be deleted.
        click.echo(
            MutationResultPayload(
                status="preview",
                operation="delete",
                session_count=count,
                affected_count=0,
                session_ids=tuple(session_ids),
            ).to_json(exclude_none=True)
        )
        return
    if count == 0:
        click.echo(
            MutationResultPayload(status="ok", operation="delete", session_count=0, affected_count=0).to_json(
                exclude_none=True
            )
        )
        return
    if not force:
        # Machine/non-interactive surfaces must never block on an interactive
        # confirmation prompt (#1818 P6). The delete verb always emits a JSON
        # MutationResultPayload, so in plain mode (``--format`` machine output,
        # a non-TTY pipe, or ``POLYLOGUE_FORCE_PLAIN``) we refuse without
        # prompting and emit a parseable ``aborted`` envelope that names the
        # required flag, mirroring the reset command's plain-mode guard rather
        # than relying on the generic plain-prompt SystemExit.
        if env.ui.plain:
            click.echo(
                MutationResultPayload(
                    status="aborted",
                    operation="delete",
                    session_count=count,
                    affected_count=0,
                    detail="confirmation_required",
                ).to_json(exclude_none=True)
            )
            return
        click.echo(f"About to delete {count} session(s):", err=True)
        for session_id in session_ids[:5]:
            click.echo(f"  - {session_id}", err=True)
        if count > 5:
            click.echo(f"  ... and {count - 5} more", err=True)
        if not env.ui.confirm("Proceed?", default=False):
            click.echo(
                MutationResultPayload(
                    status="aborted", operation="delete", session_count=count, affected_count=0
                ).to_json(exclude_none=True)
            )
            return
    deleted = archive.delete_sessions(session_ids)
    # ``session_count`` = matched, ``affected_count`` = sessions actually deleted.
    click.echo(
        MutationResultPayload(
            status="deleted" if deleted else "ok",
            operation="delete",
            session_count=count,
            affected_count=deleted,
        ).to_json(exclude_none=True)
    )


def _inject_attached_units(
    items: list[dict[str, object]],
    session_ids: Sequence[str],
    *,
    archive: ArchiveStore | None,
    with_units: tuple[str, ...],
    with_unit_fields: dict[str, tuple[str, ...]] | None = None,
) -> None:
    """Attach ``with <units>`` projection rows to each rendered row payload.

    Each item gains an ``attached_units`` key mapping every requested unit to a
    list of JSON-ready row payloads for that session (empty list when the
    session has no rows for the unit), keeping the output shape predictable.
    """

    if not with_units or archive is None or not items:
        return
    attached = fetch_attached_units(archive, session_ids, with_units, unit_fields=with_unit_fields)
    for item, session_id in zip(items, session_ids, strict=False):
        item["attached_units"] = {unit: list(by_session.get(session_id, ())) for unit, by_session in attached.items()}


def _emit_list(
    summaries: list[ArchiveSessionSummary],
    *,
    limit: int,
    offset: int,
    next_cursor: str | None,
    output_format: str,
    origin: str | None,
    fields: str | None,
    archive: ArchiveStore | None = None,
    with_units: tuple[str, ...] = (),
    with_unit_fields: dict[str, tuple[str, ...]] | None = None,
) -> None:
    items = [_summary_payload(summary) for summary in summaries]
    _inject_attached_units(
        items,
        [summary.session_id for summary in summaries],
        archive=archive,
        with_units=with_units,
        with_unit_fields=with_unit_fields,
    )
    envelope: dict[str, object] = {
        "mode": "list",
        "origin": origin,
        "items": items,
        "total": len(items),
        "limit": limit,
        "offset": offset,
        "next_offset": offset + limit if next_cursor is not None else None,
        "next_cursor": next_cursor,
    }
    # Browse/list mode emits an empty envelope (exit 0) when the archive has no
    # matching rows: "show me everything, there is nothing" is a valid success.
    # Only search mode (a lexical/semantic query that matched nothing) exits 2.
    _emit_rows(envelope, items, output_format=output_format, text_line=_summary_line, fields=fields)


def _emit_search(
    hits: list[ArchiveSessionSearchHit],
    *,
    archive: ArchiveStore,
    query: str,
    limit: int,
    offset: int,
    next_cursor: str | None,
    retrieval_lane: str,
    output_format: str,
    origin: str | None,
    fields: str | None,
    typo_hint: str | None = None,
    with_units: tuple[str, ...] = (),
    with_unit_fields: dict[str, tuple[str, ...]] | None = None,
) -> None:
    items = [
        _hit_payload(
            hit,
            summary=archive.read_summary(hit.session_id),
            retrieval_lane=retrieval_lane,
        )
        for hit in hits
    ]
    _inject_attached_units(
        items,
        [hit.session_id for hit in hits],
        archive=archive,
        with_units=with_units,
        with_unit_fields=with_unit_fields,
    )
    envelope: dict[str, object] = {
        "mode": "search",
        "origin": origin,
        "query": query,
        "retrieval_lane": retrieval_lane,
        "items": items,
        "total": len(items),
        "limit": limit,
        "offset": offset,
        "next_offset": offset + limit if next_cursor is not None else None,
        "next_cursor": next_cursor,
    }
    if not items:
        _emit_no_results(envelope, output_format=output_format, typo_hint=typo_hint)
    _emit_rows(envelope, items, output_format=output_format, text_line=_hit_line, fields=fields)


def _emit_session(
    envelope: ArchiveSessionEnvelope,
    *,
    output_format: str,
    fields: str | None,
) -> None:
    payload = _session_payload(envelope)
    if output_format == "json":
        click.echo(json.dumps(_project_payload(payload, fields), indent=2, sort_keys=True))
        return
    if output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(_project_payload(payload, fields), sort_keys=False, allow_unicode=True), nl=False)
        return
    if output_format == "ndjson":
        messages = payload["messages"]
        if not isinstance(messages, list):
            raise TypeError("session payload messages must be a list")
        for message in messages:
            click.echo(json.dumps(message, sort_keys=True))
        return
    if output_format not in {"markdown", "plaintext"}:
        raise click.UsageError(f"Full-session reads do not support --format {output_format}.")
    click.echo(_session_text(envelope))


def _emit_no_results(envelope: dict[str, object], *, output_format: str, typo_hint: str | None = None) -> NoReturn:
    """Emit the canonical no-results response and exit with status 2.

    Status 2 distinguishes "the query ran and matched nothing" from a
    successful read with results (0) and from an error (1), so callers can
    branch on an empty result set. Machine formats still receive a parseable
    empty envelope; text surfaces get the human-readable message.
    """
    from polylogue.cli.convergence_feedback import convergence_warning_line

    convergence_warning = convergence_warning_line()
    empty = {**envelope, "items": [], "total": 0}
    if convergence_warning is not None:
        empty["archive_converging"] = True
        empty["convergence_warning"] = convergence_warning
    if output_format == "json":
        click.echo(json.dumps(empty, indent=2, sort_keys=True))
    elif output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(empty, sort_keys=False, allow_unicode=True), nl=False)
    elif output_format in {"ndjson", "csv"}:
        pass  # no rows to emit
    else:
        if convergence_warning is not None:
            click.echo(convergence_warning)
        click.echo("No sessions matched.")
        if typo_hint is not None:
            click.echo(typo_hint)
    raise SystemExit(2)


def _emit_open_no_results(*, output_format: str, origin: str | None) -> NoReturn:
    if output_format == "json":
        error_no_results("No sessions matched.").emit(exit_code=2)
    _emit_no_results(
        {
            "mode": "open",
            "origin": origin,
            "items": [],
            "total": 0,
        },
        output_format=output_format,
    )


def _emit_rows(
    envelope: dict[str, object],
    items: list[dict[str, object]],
    *,
    output_format: str,
    text_line: Callable[[dict[str, object]], str],
    fields: str | None,
) -> None:
    env = _TIMING_ENV.get()
    if env is not None:
        env.finish_timing("execute")
        env.begin_timing("render")
    try:
        projected_items = [_project_payload(item, fields) for item in items]
        if output_format == "json":
            projected_envelope = {**envelope, "items": projected_items}
            click.echo(json.dumps(projected_envelope, indent=2, sort_keys=True))
            return
        if output_format == "ndjson":
            for item in projected_items:
                click.echo(json.dumps(item, sort_keys=True))
            return
        if output_format == "csv":
            click.echo(_csv(projected_items), nl=False)
            return
        if output_format == "yaml":
            import yaml

            projected_envelope = {**envelope, "items": projected_items}
            click.echo(yaml.safe_dump(projected_envelope, sort_keys=False, allow_unicode=True), nl=False)
            return
        if output_format not in {"markdown", "plaintext"}:
            raise click.UsageError(f"Root query does not support --format {output_format}.")
        click.echo("\n".join(text_line(item) for item in items))
    finally:
        if env is not None:
            env.finish_timing("render")


def _emit_unit_source_rows(
    archive: ArchiveStore,
    *,
    source: QueryUnitSource,
    query: str,
    limit: int,
    offset: int,
    session_filters: Mapping[str, object] | None = None,
    output_format: str,
    fields: str | None,
) -> None:
    envelope_model = query_unit_rows(
        archive,
        source,
        query=query,
        limit=limit,
        offset=offset,
        session_filters=session_filters,
    )
    envelope = envelope_model.model_dump(mode="json")
    items = [item.model_dump(mode="json") for item in envelope_model.items]
    text_line = (
        _aggregate_query_line if envelope_model.mode == "query-unit-aggregate" else _query_unit_text_line(source.unit)
    )

    if not items:
        _emit_unit_no_results(envelope, unit=source.unit, output_format=output_format)
    _emit_rows(envelope, items, output_format=output_format, text_line=text_line, fields=fields)


def _unit_source_display_name(source: QueryUnitSource) -> str:
    descriptor = query_unit_descriptor(source.unit)
    if descriptor is None:
        return source.unit
    return descriptor.plural_source


def _unit_source_session_filters(filter_kwargs: _ArchiveFilterKwargs) -> dict[str, object]:
    filters = dict(filter_kwargs)
    filters.pop("since_session_id", None)
    return query_unit_session_filters(**filters)


def _emit_unit_no_results(envelope: dict[str, object], *, unit: str, output_format: str) -> NoReturn:
    empty = {**envelope, "items": [], "total": 0}
    if output_format == "json":
        click.echo(json.dumps(empty, indent=2, sort_keys=True))
    elif output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(empty, sort_keys=False, allow_unicode=True), nl=False)
    elif output_format in {"ndjson", "csv"}:
        pass
    else:
        click.echo(f"No {unit}s matched.")
    raise SystemExit(2)


def _snippet(value: object, *, max_chars: int = 96) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _message_query_line(item: dict[str, object]) -> str:
    return f"{item['message_id']} [{item['role']}] {_snippet(item.get('text'))}"


def _action_query_line(item: dict[str, object]) -> str:
    action = item.get("semantic_type") or item.get("tool_name") or "action"
    detail = item.get("tool_path") or item.get("tool_command") or item.get("output_text") or ""
    return f"{item['tool_use_block_id']} [{action}] {_snippet(detail)}"


def _block_query_line(item: dict[str, object]) -> str:
    detail = item.get("text") or item.get("tool_path") or item.get("tool_command") or ""
    return f"{item['block_id']} [{item['block_type']}] {_snippet(detail)}"


def _file_query_line(item: dict[str, object]) -> str:
    detail = f"actions={item.get('action_count', 0)}"
    first_ref = item.get("first_tool_use_block_id") or item.get("first_message_id") or item.get("session_id")
    if first_ref:
        detail = f"{detail} first={first_ref}"
    return f"{item['path']} [{item['origin']}] {_snippet(detail)}"


def _assertion_query_line(item: dict[str, object]) -> str:
    detail = item.get("body_text") or item.get("key") or item.get("value") or item.get("target_ref") or ""
    return f"{item['assertion_id']} [{item['kind']}/{item['status']}] {_snippet(detail)}"


def _aggregate_query_line(item: dict[str, object]) -> str:
    group_by = item.get("group_by") or "all"
    group_key = item.get("group_key") or "all"
    return f"{group_by}={group_key} count={item['count']}"


def _run_query_line(item: dict[str, object]) -> str:
    detail_parts = [str(part) for part in (item.get("agent_ref"), item.get("title")) if part]
    detail = " ".join(detail_parts) or item.get("run_ref") or ""
    return f"{item['run_ref']} [{item['role']}/{item['status']}] {_snippet(detail)}"


def _observed_event_query_line(item: dict[str, object]) -> str:
    detail = item.get("summary") or item.get("subject_ref") or item.get("event_ref") or ""
    return f"{item['event_ref']} [{item['kind']}/{item['delivery_state']}] {_snippet(detail)}"


def _context_snapshot_query_line(item: dict[str, object]) -> str:
    detail = item.get("metadata") or item.get("segment_refs") or item.get("evidence_refs") or ""
    return f"{item['snapshot_ref']} [{item['boundary']}/{item['inheritance_mode']}] {_snippet(detail)}"


def _delegation_query_line(item: dict[str, object]) -> str:
    detail = item.get("instruction_preview") or item.get("artifact_preview") or item.get("child_session_id") or ""
    return f"{item['delegation_ref']} [{item['mapping_state']}/{item['result_status']}] {_snippet(detail)}"


_QUERY_UNIT_TEXT_LINES: dict[str, _QueryUnitTextLine] = {
    "message": _message_query_line,
    "action": _action_query_line,
    "block": _block_query_line,
    "file": _file_query_line,
    "assertion": _assertion_query_line,
    "run": _run_query_line,
    "observed-event": _observed_event_query_line,
    "context-snapshot": _context_snapshot_query_line,
    "delegation": _delegation_query_line,
}


def _query_unit_text_line(unit: str) -> _QueryUnitTextLine:
    descriptor = query_unit_descriptor(unit)
    renderer = descriptor.cli_plain_renderer if descriptor else None
    if renderer is None:
        raise click.UsageError(f"Unsupported query unit: {unit}")
    try:
        return _QUERY_UNIT_TEXT_LINES[renderer]
    except KeyError as exc:
        raise click.UsageError(f"Unsupported query unit renderer: {renderer}") from exc


def _project_payload(payload: dict[str, object], fields: str | None) -> dict[str, object]:
    selected = _selected_fields(fields)
    if selected is None:
        return dict(payload)
    return {key: value for key, value in payload.items() if key in selected}


def _selected_fields(fields: str | None) -> frozenset[str] | None:
    if not fields:
        return None
    selected = frozenset(field.strip() for field in fields.split(",") if field.strip())
    return selected or None


def _csv(items: list[dict[str, object]]) -> str:
    if not items:
        return ""
    fields = list(items[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(items)
    return buf.getvalue()


def _summary_payload(summary: ArchiveSessionSummary) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        model_json_document(
            SessionListRowPayload(
                id=summary.session_id,
                origin=summary.origin,
                title=_snippet(summary.title or summary.session_id, max_chars=96),
                target_ref=TargetRefPayload.session(summary.session_id),
                anchor=reader_anchor("session", summary.session_id),
                created_at=summary.created_at,
                updated_at=summary.updated_at,
                message_count=summary.message_count,
                tags=summary.tags,
                words=summary.word_count,
                repo=summary.git_repository_url,
                cwd_display=summary.working_directories[0] if summary.working_directories else None,
            ),
            exclude_none=True,
        ),
    )


def _hit_payload(
    hit: ArchiveSessionSearchHit,
    *,
    summary: ArchiveSessionSummary,
    retrieval_lane: str,
) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        model_json_document(
            SessionSearchHitPayload(
                session=SessionSummaryPayload(
                    id=summary.session_id,
                    origin=summary.origin,
                    title=_snippet(summary.title or summary.session_id, max_chars=96),
                    message_count=summary.message_count,
                    target_ref=TargetRefPayload.session(summary.session_id),
                    anchor=reader_anchor("session", summary.session_id),
                ),
                match=SessionSearchMatchPayload(
                    rank=hit.rank,
                    retrieval_lane=retrieval_lane,
                    match_surface="message",
                    target_ref=TargetRefPayload.message(session_id=hit.session_id, message_id=hit.message_id),
                    anchor=reader_anchor("message", hit.message_id),
                    actions=reader_message_actions(),
                    message_id=hit.message_id,
                    snippet=_snippet(hit.snippet, max_chars=320),
                    score=None,
                    score_kind=None,
                ),
            ),
            exclude_none=True,
        ),
    )


def _session_payload(envelope: ArchiveSessionEnvelope) -> dict[str, object]:
    return {
        "mode": "session",
        "session_id": envelope.session_id,
        "native_id": envelope.native_id,
        "origin": envelope.origin,
        "source": envelope.origin,
        "title": envelope.title,
        "active_leaf_message_id": envelope.active_leaf_message_id,
        "messages": [
            {
                "message_id": message.message_id,
                "native_id": message.native_id,
                "role": message.role,
                "position": message.position,
                "variant_index": message.variant_index,
                "is_active_path": message.is_active_path,
                "is_active_leaf": message.is_active_leaf,
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
                ],
            }
            for message in envelope.messages
        ],
    }


def _session_text(envelope: ArchiveSessionEnvelope) -> str:
    lines = [f"# {envelope.title or envelope.session_id}", "", f"`{envelope.session_id}`", ""]
    for message in envelope.messages:
        lines.append(f"## {message.role}")
        text = "\n".join(block.text or "" for block in message.blocks if block.text)
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip()


def _ellipsize(value: str, max_width: int) -> str:
    if max_width <= 3:
        return value[:max_width]
    return (value[: max_width - 3] + "...") if len(value) > max_width else value


def _summary_line(item: dict[str, object]) -> str:
    session_id = str(item["id"])
    title = _snippet(item.get("title") or session_id, max_chars=50)
    date = str(item.get("updated_at") or item.get("created_at") or "unknown")[:10]
    origin = str(item["origin"])
    message_count = item.get("message_count") or 0
    line = f"{session_id[:24]:24s}  {date:10s}  {origin:24s}  {title} ({message_count} msgs)"
    return line + _attached_units_suffix(item)


def _attached_units_suffix(item: dict[str, object]) -> str:
    """Render a compact ``[+unit:N]`` summary of attached projection units."""

    attached = item.get("attached_units")
    if not isinstance(attached, dict) or not attached:
        return ""
    parts = [f"{unit}:{len(rows)}" for unit, rows in attached.items() if isinstance(rows, list)]
    return f"  [+{' '.join(parts)}]" if parts else ""


def _hit_line(item: dict[str, object]) -> str:
    session = item.get("session")
    match = item.get("match")
    if not isinstance(session, dict) or not isinstance(match, dict):
        return str(item)
    title = _snippet(session.get("title") or session.get("id"), max_chars=96)
    snippet = _snippet(match.get("snippet"), max_chars=320)
    line = f"{match['rank']}. {session['origin']}  {title}  {snippet}"
    return line + _attached_units_suffix(item)


def _stats_by_line(item: dict[str, object]) -> str:
    return f"{item['group']}: {item['count']}"


def _fail(message: str) -> NoReturn:
    click.echo(f"Error: {message}", err=True)
    raise SystemExit(1)


__all__ = ["execute_archive_query", "execute_delete_by_session_ids"]
