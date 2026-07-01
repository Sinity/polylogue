"""Bounded chronological read-view handler."""

from __future__ import annotations

import json

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.cli.query import _create_query_vector_provider
from polylogue.cli.read_view_registry import CHRONICLE_READ_VIEW_OPTION_NAMES
from polylogue.cli.read_views.base import (
    ReadViewChronicleOptions,
    ReadViewInvocation,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.core.enums import MaterialOrigin
from polylogue.paths import archive_file_set_root_for_paths
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.surfaces.chronicle import (
    ChronicleProjectionPayload,
    ChronicleSessionPayload,
    build_chronicle_projection_payload,
    build_chronicle_session_payload,
    chronicle_json_document,
    render_chronicle_markdown,
)

DEFAULT_CHRONICLE_EDGE_LIMIT = 8


def build_chronicle_options(values: ReadViewOptionValues) -> ReadViewChronicleOptions:
    """Build chronicle options from read command values."""

    raw_limit = values.get("limit")
    edge_limit = int(raw_limit) if isinstance(raw_limit, int | str) else DEFAULT_CHRONICLE_EDGE_LIMIT
    return ReadViewChronicleOptions(edge_limit=max(edge_limit, 1))


async def _selected_summaries(config: Config, request: RootModeRequest) -> list[SessionSummary]:
    from dataclasses import replace

    spec = request.query_spec()
    if spec.limit is None:
        spec = replace(spec, limit=5)
    archive_root = archive_file_set_root_for_paths(
        archive_root_path=config.archive_root,
        db_anchor=config.db_path,
    )
    vector_provider = _create_query_vector_provider(config, db_path=archive_root / "embeddings.db")
    return list(await spec.list_summaries(config, vector_provider=vector_provider))


def build_read_chronicle_payload(
    config: Config,
    request: RootModeRequest,
    *,
    edge_limit: int = DEFAULT_CHRONICLE_EDGE_LIMIT,
) -> ChronicleProjectionPayload:
    """Build a bounded first/last prose projection for selected sessions."""

    async def _run() -> ChronicleProjectionPayload:
        summaries = await _selected_summaries(config, request)
        archive_root = archive_file_set_root_for_paths(
            archive_root_path=config.archive_root,
            db_anchor=config.db_path,
        )
        backend = SQLiteBackend(db_path=archive_root / "index.db")
        session_payloads: list[ChronicleSessionPayload] = []
        try:
            for summary in summaries:
                first_messages, last_messages, total = await backend.get_message_edge_windows(
                    str(summary.id),
                    message_role=(Role.USER, Role.ASSISTANT),
                    message_type="message",
                    material_origin=(MaterialOrigin.HUMAN_AUTHORED, MaterialOrigin.ASSISTANT_AUTHORED),
                    edge_limit=edge_limit * 5,
                )
                session_payloads.append(
                    build_chronicle_session_payload(
                        summary,
                        first_messages=first_messages,
                        last_messages=last_messages,
                        total_matching_messages=total,
                        edge_limit=edge_limit,
                    )
                )
        finally:
            await backend.close()
        return build_chronicle_projection_payload(session_payloads, edge_limit=edge_limit)

    return run_coroutine_sync(_run())


def run_read_chronicle(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Run the chronicle read view and deliver markdown/json output."""

    options = invocation.options if isinstance(invocation.options, ReadViewChronicleOptions) else None
    projection = invocation.projection_spec.projection if invocation.projection_spec is not None else None
    if projection is not None and projection.edge_limit is not None:
        edge_limit = projection.edge_limit
    elif options is not None:
        edge_limit = options.edge_limit
    else:
        edge_limit = DEFAULT_CHRONICLE_EDGE_LIMIT
    payload = build_read_chronicle_payload(env.config, request, edge_limit=edge_limit)
    fmt = invocation.output_format or "markdown"
    if fmt == "json":
        content = json.dumps(chronicle_json_document(payload), indent=2) + "\n"
    else:
        content = render_chronicle_markdown(payload)
    deliver_content(env, content, destination=invocation.destination, out_path=invocation.out_path)


__all__ = [
    "CHRONICLE_READ_VIEW_OPTION_NAMES",
    "DEFAULT_CHRONICLE_EDGE_LIMIT",
    "build_chronicle_options",
    "build_read_chronicle_payload",
    "run_read_chronicle",
]
