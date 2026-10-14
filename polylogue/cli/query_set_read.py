"""Query-set reading for every session matching the parent filter chain."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from tempfile import TemporaryFile

import click

from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.archive.session.domain_models import Session
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.session_rows import SessionSelection, query_session_selection
from polylogue.cli.shared.types import AppEnv
from polylogue.cli.transcript import MetadataWalk, TranscriptRecords, transcript_records
from polylogue.rendering.formatting import write_session
from polylogue.surfaces.outcome import OutcomeEnvelope

_PER_LINE_FORMATS = frozenset({"jsonl"})
_SEPARATED_FORMATS = frozenset({"markdown", "obsidian", "org", "plaintext", "html", "yaml"})
_ARRAY_FORMATS = frozenset({"json"})


def _read_selected_session(
    env: AppEnv, request: RootModeRequest, session_id: str, epoch: str, lifetime: ExitStack
) -> tuple[Session, TranscriptRecords, MetadataWalk, OutcomeEnvelope]:
    """Hydrate bounded resident pages of the exact selected view."""
    from polylogue.archive.message.messages import MessageCollection
    from polylogue.archive.message.models import Message
    from polylogue.cli.operation_kernel import OperationEnvelopeError, OperationRequest
    from polylogue.cli.read_dispatch import daemon_route_disabled, dispatch_read
    from polylogue.surfaces.outcome import OutcomeEnvelope, combine_outcomes

    offset = 0
    page_limit = 100
    continuation = None
    first = None
    records = lifetime.enter_context(transcript_records())
    metadata = MetadataWalk(lifetime.enter_context(transcript_records()))
    messages_complete = False
    total = None
    outcome = None
    while True:
        result, _authority = dispatch_read(
            env.config,
            OperationRequest(
                "session.read",
                {
                    "ref": session_id,
                    "kind": "transcript",
                    "session_projection": "domain",
                    "selection_epoch": epoch,
                    "limit": 0 if messages_complete else page_limit,
                    "metadata_limit": metadata.page_limit,
                    "metadata_continuation": metadata.continuation,
                    "include_user_tags": True,
                    "offset": offset,
                    "continuation": None if messages_complete else continuation,
                },
            ),
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        )
        if result.get("selection_epoch") != epoch or result.get("session_id") != session_id:
            raise OperationEnvelopeError("session.read changed the selected view")
        body = result.get("session")
        if not isinstance(body, Mapping):
            raise OperationEnvelopeError("session.read omitted its domain page")
        page = Session.model_validate(body)
        if page.id != session_id:
            raise OperationEnvelopeError("session.read changed the selected session")
        raw_total = result.get("total")
        if isinstance(raw_total, bool) or not isinstance(raw_total, int) or raw_total < 0:
            raise OperationEnvelopeError("session.read omitted its composed message count")
        if total is not None and raw_total != total:
            raise OperationEnvelopeError("session.read changed its message count")
        total = raw_total
        if result.get("offset") != offset:
            raise OperationEnvelopeError("session.read changed its requested offset")
        try:
            declared = OutcomeEnvelope.model_validate(result.get("outcome"))
        except ValueError as exc:
            raise OperationEnvelopeError("session.read returned no valid outcome") from exc
        if outcome is None:
            outcome = declared
        elif not declared.rows_are_authoritative:
            composed = combine_outcomes((outcome, declared))
            outcome = composed.model_copy(update={"detail": {**outcome.detail, **declared.detail, **composed.detail}})
        if first is None:
            first = page.model_copy(
                update={"messages": MessageCollection.empty(), "working_directories": (), "tags_m2m": ()}
            )
        if not metadata.complete:
            metadata.accept(result.get("metadata_page"), page.working_directories, page.tags_m2m)
            if metadata.complete:
                metadata.continuation = None
        for message in page.messages:
            records.append(Message.model_validate(message).model_dump(mode="json"))
        next_offset = result.get("next_offset")
        if next_offset is None:
            if result.get("complete") is not True or records.count != total:
                raise OperationEnvelopeError("session.read did not complete its message relation")
            messages_complete = True
            if metadata.complete:
                break
            offset = total
            continue
        if (
            isinstance(next_offset, bool)
            or not isinstance(next_offset, int)
            or next_offset != offset + len(page.messages)
            or next_offset <= offset
            or next_offset > total
        ):
            raise OperationEnvelopeError("session.read returned a non-progressing page")
        token = result.get("continuation")
        if not isinstance(token, str) or not token or result.get("complete") is not False:
            raise OperationEnvelopeError("session.read omitted its continuation")
        continuation, offset = token, next_offset
    assert first is not None
    assert outcome is not None
    first = first.model_copy(
        update={"working_directories": (metadata.first_directory,) if metadata.first_directory else ()}
    )
    return first, records, metadata, outcome


def run_query_set_read(
    env: AppEnv,
    request: RootModeRequest,
    *,
    output_format: str,
    fields: str | None,
    content_projection: ContentProjectionSpec | None = None,
) -> SessionSelection:
    """Stage resident pages before delivering a complete query-set document."""
    from dataclasses import replace

    from polylogue.surfaces.outcome import OutcomeEnvelope, combine_outcomes

    spec = request.query_spec()
    selection = query_session_selection(
        env.config,
        request,
        limit=spec.limit,
        offset=spec.offset,
        daemon_disabled=bool(request.params.get("no_daemon")),
    )
    selection.require_bound()
    outcome = selection.outcome
    render_format = "json" if output_format == "jsonl" else output_format
    with TemporaryFile(mode="w+t", encoding="utf-8", newline="") as staged:
        if output_format in _ARRAY_FORMATS:
            staged.write("[\n")
        csv_header: list[str] | None = None
        for index, row in enumerate(selection.rows):
            with ExitStack() as read_lifetime:
                session, records, metadata, read_outcome = _read_selected_session(
                    env, request, row.session_id, selection.snapshot_epoch or "", read_lifetime
                )
                assert isinstance(read_outcome, OutcomeEnvelope)
                if outcome is None:
                    outcome = read_outcome
                elif not read_outcome.rows_are_authoritative:
                    composed = combine_outcomes((outcome, read_outcome))
                    outcome = composed.model_copy(
                        update={"detail": {**outcome.detail, **read_outcome.detail, **composed.detail}}
                    )
                from polylogue.archive.message.models import Message

                def messages(records: TranscriptRecords = records) -> Iterable[Message]:
                    return (Message.model_validate(record) for record in records.records())

                def tags(metadata: MetadataWalk = metadata, session: Session = session) -> Iterable[str]:
                    return metadata.values(session.tags)

                if index and output_format in _SEPARATED_FORMATS:
                    staged.write("\n---\n\n")
                if output_format == "csv":
                    with TemporaryFile(mode="w+t", encoding="utf-8", newline="") as csv_stage:
                        write_session(
                            session,
                            messages,
                            csv_stage,
                            render_format,
                            fields,
                            content_projection=content_projection,
                            tags=tags,
                        )
                        csv_stage.seek(0)
                        csv_rows = csv.reader(csv_stage)
                        header = next(csv_rows, None)
                        if header is None or (csv_header is not None and header != csv_header):
                            raise ValueError("session CSV changed its columns")
                        writer = csv.writer(staged)
                        if csv_header is None:
                            writer.writerow(header)
                            csv_header = header
                        writer.writerows(csv_rows)
                    continue
                write_session(
                    session,
                    messages,
                    staged,
                    render_format,
                    fields,
                    content_projection=content_projection,
                    compact_json=output_format in _PER_LINE_FORMATS,
                    tags=tags,
                )
                if output_format in _ARRAY_FORMATS and index < len(selection.rows) - 1:
                    staged.write(",")
                staged.write("\n")
        if output_format in _ARRAY_FORMATS:
            staged.write("]\n")
        staged.seek(0)
        while chunk := staged.read(65536):
            click.echo(chunk, nl=False)
    return replace(selection, outcome=outcome)


__all__ = ["run_query_set_read"]
