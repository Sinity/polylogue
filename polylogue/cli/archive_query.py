"""Current archive query executor for the root CLI."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import webbrowser
from collections.abc import Callable, Iterable, Sequence
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import NoReturn, TypeVar
from urllib.parse import quote

import click
from typing_extensions import TypedDict

from polylogue.archive.message.roles import MessageRoleFilter, Role, normalize_message_roles
from polylogue.archive.message.types import validate_message_type_filter
from polylogue.archive.query.spec import (
    QuerySpecError,
    normalize_action_sequence,
    normalize_action_terms,
    parse_query_date,
)
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
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
from polylogue.core.enums import Provider
from polylogue.paths import archive_file_set_root_for_paths
from polylogue.storage.search_providers import create_vector_provider, reciprocal_rank_fusion
from polylogue.storage.sqlite.archive_tiers.archive import (
    ArchiveSessionSearchHit,
    ArchiveSessionSummary,
    ArchiveStore,
)
from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow, ArchiveMessageRow, ArchiveSessionEnvelope
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.surfaces.payloads import (
    SEARCH_CURSOR_VERSION,
    InvalidSearchCursorError,
    SearchCursor,
    decode_search_cursor,
)

_PageRow = TypeVar("_PageRow", ArchiveSessionSummary, ArchiveSessionSearchHit)

_PROVIDER_TO_ARCHIVE_ORIGIN: dict[str, str] = {
    Provider.CLAUDE_CODE.value: "claude-code-session",
    Provider.CODEX.value: "codex-session",
    Provider.GEMINI_CLI.value: "gemini-cli-session",
    Provider.HERMES.value: "hermes-session",
    Provider.ANTIGRAVITY.value: "antigravity-session",
    Provider.CHATGPT.value: "chatgpt-export",
    Provider.CLAUDE_AI.value: "claude-ai-export",
    Provider.GEMINI.value: "aistudio-drive",
    Provider.UNKNOWN.value: "unknown-export",
}

_UNSUPPORTED_PARAM_MESSAGES: dict[str, str] = {}


class _ArchiveFilterKwargs(TypedDict):
    origin: str | None
    origins: tuple[str, ...]
    excluded_origins: tuple[str, ...]
    tags: tuple[str, ...]
    excluded_tags: tuple[str, ...]
    repo_names: tuple[str, ...]
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
    since_ms: int | None
    until_ms: int | None
    since_session_id: str | None


def execute_archive_query(env: AppEnv, request: RootModeRequest) -> None:
    """Execute the root query path."""
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


def _execute_archive_query_stdout(env: AppEnv, request: RootModeRequest) -> None:
    """Render root query output to stdout."""
    params = dict(request.params)
    _reject_unsupported_params(params)
    _validate_retrieval_params(params)
    config = load_effective_config(env)
    archive_root = archive_file_set_root_for_paths(archive_root_path=config.archive_root, db_anchor=config.db_path)
    index_db_path = archive_root / "index.db"
    typo_hint = maybe_subcommand_typo_hint(request.query_terms)
    origins = _resolve_origins(params)
    origin = origins[0] if len(origins) == 1 else None
    output_format = str(params.get("output_format") or "markdown")
    fields = _optional_str(params.get("fields"))
    query = _query_text(request.query_terms, params)
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

    tags_to_add = _tuple_tokens(params.get("add_tag"))
    metadata_to_set = _metadata_pairs(params.get("set_meta"))
    tags = _tags(params.get("tag"))
    excluded_tags = _tags(params.get("exclude_tag"))
    repo_names = _csv_tokens(params.get("repo"))
    has_types = _csv_tokens(params.get("has_type"))
    has_tool_use = bool(params.get("filter_has_tool_use"))
    has_thinking = bool(params.get("filter_has_thinking"))
    has_paste = bool(params.get("filter_has_paste"))
    tool_terms = _tool_tokens(params.get("tool"))
    excluded_tool_terms = _tool_tokens(params.get("exclude_tool"))
    action_terms = _action_tokens("action", params.get("action"))
    excluded_action_terms = _action_tokens("exclude_action", params.get("exclude_action"))
    action_sequence = _action_sequence_tokens(params.get("action_sequence"))
    action_text_terms = _csv_tokens(params.get("action_text"))
    referenced_paths = _tuple_tokens(params.get("referenced_path"))
    cwd_prefix = _optional_str(params.get("cwd_prefix"))
    typed_only = bool(params.get("typed_only"))
    message_type = _message_type(params.get("message_type"))
    title_filter = _optional_str(params.get("title"))
    min_messages = _optional_int(params.get("min_messages"))
    max_messages = _optional_int(params.get("max_messages"))
    min_words = _optional_int(params.get("min_words"))
    since_ms = _optional_date_ms("since", params.get("since"))
    until_ms = _optional_date_ms("until", params.get("until"))
    since_session_id = _optional_str(params.get("since_session_id"))
    limit = _limit(params)
    offset = _offset(params)
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
    message_roles = _message_roles(params)
    content_projection = ContentProjectionSpec.from_params(params)
    transform = _transform(params.get("transform"))
    stream = bool(params.get("stream"))
    stream_output_format = QueryOutputSpec.from_params(params).stream_format()
    sort = _sort(params.get("sort"))
    reverse = bool(params.get("reverse"))
    similar_text = _optional_str(params.get("similar_text"))
    retrieval_lane = _optional_str(params.get("retrieval_lane")) or "auto"
    delete_matched = bool(params.get("delete_matched"))
    excluded_origins = _resolve_excluded_origins(params)
    filter_kwargs: _ArchiveFilterKwargs = {
        "origin": origin,
        "origins": origins,
        "excluded_origins": excluded_origins,
        "tags": tags,
        "excluded_tags": excluded_tags,
        "repo_names": repo_names,
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
        "since_ms": since_ms,
        "until_ms": until_ms,
        "since_session_id": since_session_id,
    }

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

    with ArchiveStore.open_existing(archive_root) as archive:
        if params.get("stats_only") or params.get("stats_by"):
            if sample_count is not None:
                raise click.UsageError("Root query does not combine --sample with stats.")
            session_ids = _matched_session_ids_for_stats(archive, query=query, limit=limit, filter_kwargs=filter_kwargs)
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
                    grouped = archive.stats_by(group_by, **filter_kwargs, session_ids=session_ids)
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
            stats = archive.stats(**filter_kwargs, session_ids=session_ids)
            _emit_stats(stats, output_format=output_format, origin=origin, query=query, fields=fields)
            return
        if params.get("count_only"):
            _emit_count(
                archive.count_sessions(
                    origin=origin,
                    origins=origins,
                    excluded_origins=excluded_origins,
                    tags=tags,
                    excluded_tags=excluded_tags,
                    repo_names=repo_names,
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
                    since_ms=since_ms,
                    until_ms=until_ms,
                    since_session_id=since_session_id,
                ),
                output_format=output_format,
                origin=origin,
            )
            return
        conv_id = params.get("conv_id")
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
                            message_roles=message_roles,
                            content_projection=content_projection,
                            transform=transform,
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
                    if transform is not None:
                        envelopes = [archive.read_session(hit.session_id) for hit in page_hits]
                        _emit_sessions(
                            envelopes,
                            output_format=output_format,
                            fields=fields,
                            message_roles=message_roles,
                            content_projection=content_projection,
                            transform=transform,
                        )
                        return
                    _emit_search(
                        page_hits,
                        query=similar_text or query,
                        limit=limit,
                        offset=page_offset,
                        next_cursor=next_cursor,
                        retrieval_lane=resolved_lane,
                        output_format=output_format,
                        origin=origin,
                        fields=fields,
                        typo_hint=typo_hint,
                    )
                    return
                if stream:
                    envelope = archive.read_session(session_id)
                    _emit_stream(
                        envelope,
                        output_format=stream_output_format,
                        message_roles=message_roles,
                        content_projection=content_projection,
                        transform=transform,
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
                message_roles=message_roles,
                content_projection=content_projection,
                transform=transform,
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
                    message_roles=message_roles,
                    content_projection=content_projection,
                    transform=transform,
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
            if transform is not None:
                envelopes = [archive.read_session(hit.session_id) for hit in page_hits]
                _emit_sessions(
                    envelopes,
                    output_format=output_format,
                    fields=fields,
                    message_roles=message_roles,
                    content_projection=content_projection,
                    transform=transform,
                )
                return
            _emit_search(
                page_hits,
                query=similar_text or query,
                limit=limit,
                offset=page_offset,
                next_cursor=next_cursor,
                retrieval_lane=resolved_lane,
                output_format=output_format,
                origin=origin,
                fields=fields,
                typo_hint=typo_hint,
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
                message_roles=message_roles,
                content_projection=content_projection,
                transform=transform,
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
        if transform is not None:
            envelopes = [archive.read_session(summary.session_id) for summary in page_summaries]
            _emit_sessions(
                envelopes,
                output_format=output_format,
                fields=fields,
                message_roles=message_roles,
                content_projection=content_projection,
                transform=transform,
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
        )


def _reject_unsupported_params(params: dict[str, object]) -> None:
    for key, message in _UNSUPPORTED_PARAM_MESSAGES.items():
        if _has_value(params.get(key)):
            raise click.UsageError(message)


def _validate_retrieval_params(params: dict[str, object]) -> None:
    lane = _optional_str(params.get("retrieval_lane"))
    if lane not in {None, "auto", "dialogue", "semantic", "hybrid"}:
        raise click.UsageError("Root query retrieval lane must be auto, dialogue, semantic, or hybrid.")


def _archive_embeddings_retrieval_ready(embeddings_db: Path) -> bool:
    """Report whether the archive holds retrieval-ready (non-stale) embeddings.

    Reads the ``embeddings.db`` provenance ledger (``message_embeddings_meta``, a
    regular table that mirrors the ``message_embeddings`` vec0 rows). The archive
    is retrieval-ready when at least one message is embedded and no embedded
    message is flagged ``needs_reindex`` — i.e. live vectors outnumber stale ones,
    matching the legacy ``embedded_messages > stale_messages`` gate (#1217/#1780).
    A missing or unreadable ``embeddings.db`` is treated as "not ready" so ``auto``
    stays on the lexical lane.
    """
    if not embeddings_db.exists():
        return False
    try:
        conn = open_readonly_connection(embeddings_db)
    except sqlite3.OperationalError:
        return False
    try:
        meta_present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'message_embeddings_meta' LIMIT 1"
        ).fetchone()
        if meta_present is None:
            return False
        embedded = int(conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] or 0)
        stale = int(
            conn.execute("SELECT COUNT(*) FROM message_embeddings_meta WHERE needs_reindex = 1").fetchone()[0] or 0
        )
        return embedded > 0 and embedded > stale
    finally:
        conn.close()


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
    # #1780: the implicit ``auto`` lane elevates to hybrid (FTS5 + vector RRF)
    # only when (a) a vector provider is configured, (b) the active archive holds
    # retrieval-ready (non-stale) embeddings, and (c) the query carries lexical
    # terms. Otherwise ``auto`` stays on the fast lexical (dialogue) lane.
    # ``--lexical`` (dialogue), ``--semantic`` (similar_text), and an explicit
    # ``--hybrid`` bypass this branch and are honored as written.
    elevated_vector_provider = None
    if (
        retrieval_lane == "auto"
        and similar_text is None
        and query
        and _archive_embeddings_retrieval_ready(embeddings_db)
    ):
        elevated_vector_provider = create_vector_provider(config, db_path=embeddings_db)
        if elevated_vector_provider is not None:
            retrieval_lane = "hybrid"
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
    vector_provider = elevated_vector_provider or create_vector_provider(config, db_path=embeddings_db)
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
    message_roles: MessageRoleFilter,
    content_projection: ContentProjectionSpec,
    transform: str | None,
    message_limit: int | None,
) -> None:
    if transform is not None:
        click.echo("Warning: --transform is ignored in --stream mode (messages are streamed individually).", err=True)
    projected = _project_session_envelope(
        envelope,
        message_roles=message_roles,
        content_projection=content_projection,
        transform=None,
    )
    messages = projected.messages[:message_limit] if message_limit is not None else projected.messages
    payload = _session_payload(
        ArchiveSessionEnvelope(
            session_id=projected.session_id,
            native_id=projected.native_id,
            origin=projected.origin,
            title=projected.title,
            active_leaf_message_id=projected.active_leaf_message_id,
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
    if origin:
        return tuple(dict.fromkeys(token.strip() for token in str(origin).split(",") if token.strip()))
    provider = params.get("provider")
    if not provider:
        return ()
    providers = [token.strip() for token in str(provider).split(",") if token.strip()]
    origins: list[str] = []
    for provider_token in providers:
        origin = _PROVIDER_TO_ARCHIVE_ORIGIN.get(provider_token)
        if origin is None:
            raise click.UsageError(f"Root query cannot map provider {provider_token!r} to an origin.")
        origins.append(origin)
    return tuple(dict.fromkeys(origins))


def _resolve_excluded_origins(params: dict[str, object]) -> tuple[str, ...]:
    explicit_excluded = params.get("exclude_origin")
    if explicit_excluded:
        return tuple(token.strip() for token in str(explicit_excluded).split(",") if token.strip())
    excluded = params.get("exclude_provider")
    if not excluded:
        return ()
    origins: list[str] = []
    for provider in (token.strip() for token in str(excluded).split(",") if token.strip()):
        origin = _PROVIDER_TO_ARCHIVE_ORIGIN.get(provider)
        if origin is None:
            raise click.UsageError(f"Root query cannot map excluded provider {provider!r} to an origin.")
        origins.append(origin)
    return tuple(origins)


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


def _message_roles(params: dict[str, object]) -> MessageRoleFilter:
    roles = params.get("message_role") or params.get("message_roles")
    if roles:
        try:
            return normalize_message_roles(roles)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
    if params.get("dialogue_only"):
        return (Role.USER, Role.ASSISTANT)
    return ()


def _sort(value: object) -> str | None:
    if not value:
        return None
    sort = str(value)
    if sort not in {"date", "messages", "words", "longest", "tokens", "random"}:
        raise click.UsageError("Root query sort must be one of date, messages, words, longest, tokens, random.")
    return sort


def _transform(value: object) -> str | None:
    if not value:
        return None
    transform = str(value)
    if transform not in {"strip-tools", "strip-thinking", "strip-all"}:
        raise click.UsageError("Root query transform must be one of strip-tools, strip-thinking, strip-all.")
    return transform


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
    limit: int,
    filter_kwargs: _ArchiveFilterKwargs,
) -> tuple[str, ...]:
    if not query:
        return ()
    hits = archive.search_summaries(query, limit=limit, **filter_kwargs)
    return tuple(dict.fromkeys(hit.session_id for hit in hits))


def _emit_stats(
    stats: ArchiveStats,
    *,
    output_format: str,
    origin: str | None,
    query: str,
    fields: str | None,
) -> None:
    payload = {
        "mode": "stats",
        "origin": origin,
        "query": query or None,
        **stats.to_dict(),
    }
    if output_format == "json":
        click.echo(json.dumps(_project_payload(payload, fields), indent=2, sort_keys=True))
        return
    if output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(_project_payload(payload, fields), sort_keys=False, allow_unicode=True), nl=False)
        return
    if output_format not in {"markdown", "plaintext"}:
        raise click.UsageError(f"Stats do not support --format {output_format}.")
    click.echo(
        "\n".join(
            [
                f"Sessions: {stats.total_sessions}",
                f"Messages: {stats.total_messages}",
                f"Attachments: {stats.total_attachments}",
                f"Origins: {stats.origin_count}",
                f"Average messages: {stats.avg_messages_per_session:.1f}",
            ]
        )
    )


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


def _emit_mutation(changed: int, *, operation: str) -> None:
    click.echo(json.dumps({"mode": "mutation", "operation": operation, "changed": changed}, indent=2))


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
    click.echo(
        json.dumps(
            {"mode": "mutation", "operation": "mutate", "changed": changes},
            indent=2,
            sort_keys=True,
        )
    )


def _emit_delete(
    env: AppEnv, archive: ArchiveStore, session_ids: tuple[str, ...], *, params: dict[str, object]
) -> None:
    dry_run = bool(params.get("dry_run"))
    force = bool(params.get("force"))
    count = len(session_ids)
    if count == 0:
        click.echo(json.dumps({"mode": "mutation", "operation": "delete", "matched": 0, "deleted": 0}, indent=2))
        return
    if dry_run:
        click.echo(
            json.dumps(
                {
                    "mode": "mutation",
                    "operation": "delete",
                    "dry_run": True,
                    "matched": count,
                    "deleted": 0,
                    "session_ids": list(session_ids),
                },
                indent=2,
            )
        )
        return
    if not force:
        click.echo(f"About to delete {count} session(s):", err=True)
        for session_id in session_ids[:5]:
            click.echo(f"  - {session_id}", err=True)
        if count > 5:
            click.echo(f"  ... and {count - 5} more", err=True)
        if not env.ui.confirm("Proceed?", default=False):
            click.echo(
                json.dumps(
                    {
                        "mode": "mutation",
                        "operation": "delete",
                        "matched": count,
                        "deleted": 0,
                        "aborted": True,
                    },
                    indent=2,
                )
            )
            return
    deleted = archive.delete_sessions(session_ids)
    click.echo(
        json.dumps(
            {"mode": "mutation", "operation": "delete", "matched": count, "deleted": deleted},
            indent=2,
        )
    )


def _emit_list(
    summaries: list[ArchiveSessionSummary],
    *,
    limit: int,
    offset: int,
    next_cursor: str | None,
    output_format: str,
    origin: str | None,
    fields: str | None,
) -> None:
    items = [_summary_payload(summary) for summary in summaries]
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
    query: str,
    limit: int,
    offset: int,
    next_cursor: str | None,
    retrieval_lane: str,
    output_format: str,
    origin: str | None,
    fields: str | None,
    typo_hint: str | None = None,
) -> None:
    items = [_hit_payload(hit) for hit in hits]
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
    message_roles: MessageRoleFilter,
    content_projection: ContentProjectionSpec,
    transform: str | None,
) -> None:
    projected_envelope = _project_session_envelope(
        envelope,
        message_roles=message_roles,
        content_projection=content_projection,
        transform=transform,
    )
    payload = _session_payload(projected_envelope)
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
    click.echo(_session_text(projected_envelope))


def _emit_sessions(
    envelopes: Sequence[ArchiveSessionEnvelope],
    *,
    output_format: str,
    fields: str | None,
    message_roles: MessageRoleFilter,
    content_projection: ContentProjectionSpec,
    transform: str | None,
) -> None:
    if not envelopes:
        _fail("Transform found no matching session.")
    projected = [
        _project_session_envelope(
            envelope,
            message_roles=message_roles,
            content_projection=content_projection,
            transform=transform,
        )
        for envelope in envelopes
    ]
    if len(projected) == 1:
        _emit_session(
            projected[0],
            output_format=output_format,
            fields=fields,
            message_roles=(),
            content_projection=ContentProjectionSpec(),
            transform=None,
        )
        return

    items = [_project_payload(_session_payload(envelope), fields) for envelope in projected]
    payload: dict[str, object] = {
        "mode": "sessions",
        "items": items,
        "total": len(projected),
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), nl=False)
        return
    if output_format == "ndjson":
        for item in items:
            click.echo(json.dumps(item, sort_keys=True))
        return
    if output_format not in {"markdown", "plaintext"}:
        raise click.UsageError(f"Transformed session reads do not support --format {output_format}.")
    click.echo("\n\n---\n\n".join(_session_text(envelope) for envelope in projected))


def _emit_no_results(envelope: dict[str, object], *, output_format: str, typo_hint: str | None = None) -> NoReturn:
    """Emit the canonical no-results response and exit with status 2.

    Status 2 distinguishes "the query ran and matched nothing" from a
    successful read with results (0) and from an error (1), so callers can
    branch on an empty result set. Machine formats still receive a parseable
    empty envelope; text surfaces get the human-readable message.
    """
    empty = {**envelope, "items": [], "total": 0}
    if output_format == "json":
        click.echo(json.dumps(empty, indent=2, sort_keys=True))
    elif output_format == "yaml":
        import yaml

        click.echo(yaml.safe_dump(empty, sort_keys=False, allow_unicode=True), nl=False)
    elif output_format in {"ndjson", "csv"}:
        pass  # no rows to emit
    else:
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
    return {
        "id": summary.session_id,
        "session_id": summary.session_id,
        "native_id": summary.native_id,
        "origin": summary.origin,
        "source": summary.origin,
        "title": summary.title,
        "created_at": summary.created_at,
        "updated_at": summary.updated_at,
        "message_count": summary.message_count,
        "word_count": summary.word_count,
        "tags": list(summary.tags),
    }


def _hit_payload(hit: ArchiveSessionSearchHit) -> dict[str, object]:
    return {
        "rank": hit.rank,
        "id": hit.session_id,
        "session_id": hit.session_id,
        "block_id": hit.block_id,
        "message_id": hit.message_id,
        "origin": hit.origin,
        "source": hit.origin,
        "title": hit.title,
        "snippet": hit.snippet,
    }


def _project_session_envelope(
    envelope: ArchiveSessionEnvelope,
    *,
    message_roles: MessageRoleFilter,
    content_projection: ContentProjectionSpec,
    transform: str | None,
) -> ArchiveSessionEnvelope:
    if not message_roles and content_projection.is_default() and transform is None:
        return envelope
    role_values = {role.value for role in message_roles}
    messages: list[ArchiveMessageRow] = []
    for message in envelope.messages:
        if role_values and message.role not in role_values:
            continue
        if _message_removed_by_transform(message, transform):
            continue
        blocks = tuple(_project_archive_blocks(message.blocks, content_projection))
        if content_projection.filters_content() and not blocks:
            continue
        messages.append(
            ArchiveMessageRow(
                message_id=message.message_id,
                native_id=message.native_id,
                role=message.role,
                position=message.position,
                variant_index=message.variant_index,
                is_active_path=message.is_active_path,
                is_active_leaf=message.is_active_leaf,
                blocks=blocks,
            )
        )
    return ArchiveSessionEnvelope(
        session_id=envelope.session_id,
        native_id=envelope.native_id,
        origin=envelope.origin,
        title=envelope.title,
        active_leaf_message_id=envelope.active_leaf_message_id,
        messages=tuple(messages),
    )


def _message_removed_by_transform(message: ArchiveMessageRow, transform: str | None) -> bool:
    if transform is None:
        return False
    has_tool = any(block.block_type in {"tool_use", "tool_result"} for block in message.blocks)
    has_thinking = any(block.block_type == "thinking" for block in message.blocks)
    if transform == "strip-tools":
        return has_tool
    if transform == "strip-thinking":
        return has_thinking
    if transform == "strip-all":
        return has_tool or has_thinking
    return False


def _project_archive_blocks(
    blocks: tuple[ArchiveBlockRow, ...],
    content_projection: ContentProjectionSpec,
) -> list[ArchiveBlockRow]:
    if content_projection.is_default():
        return list(blocks)
    tool_semantics = {
        block.tool_id: block.semantic_type
        for block in blocks
        if block.block_type == "tool_use" and block.tool_id and block.semantic_type
    }
    return [block for block in blocks if _keep_archive_block(block, content_projection, tool_semantics)]


def _keep_archive_block(
    block: ArchiveBlockRow,
    content_projection: ContentProjectionSpec,
    tool_semantics: dict[str, str],
) -> bool:
    if block.block_type == "thinking":
        return content_projection.include_reasoning
    if block.block_type == "code":
        return content_projection.include_code
    if block.block_type == "tool_use":
        return content_projection.include_tool_calls
    if block.block_type == "tool_result":
        semantic_type = tool_semantics.get(block.tool_id or "", block.semantic_type or "")
        if semantic_type == "file_read":
            return content_projection.include_file_reads and content_projection.include_tool_outputs
        return content_projection.include_tool_outputs
    if block.block_type in {"image", "document", "file"}:
        return content_projection.include_attachments
    return content_projection.include_prose


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
    session_id = str(item["session_id"])
    title = _ellipsize(str(item.get("title") or session_id), 50)
    date = str(item.get("updated_at") or item.get("created_at") or "unknown")[:10]
    origin = str(item["origin"])
    message_count = item.get("message_count") or 0
    return f"{session_id[:24]:24s}  {date:10s}  {origin:24s}  {title} ({message_count} msgs)"


def _hit_line(item: dict[str, object]) -> str:
    title = item.get("title") or item["session_id"]
    return f"{item['rank']}. {item['origin']}  {title}  {item['snippet']}"


def _stats_by_line(item: dict[str, object]) -> str:
    return f"{item['group']}: {item['count']}"


def _fail(message: str) -> NoReturn:
    click.echo(f"Error: {message}", err=True)
    raise SystemExit(1)


__all__ = ["execute_archive_query"]
