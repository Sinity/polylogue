"""CLI execution for messages and raw verbs."""

from __future__ import annotations

from typing import cast

import click

from polylogue.api.archive import SessionNotFoundError
from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.surfaces.payloads import (
    SessionMessageRowPayload,
    SessionMessagesResponsePayload,
    model_json_document,
)


def run_messages(
    env: AppEnv,
    request: RootModeRequest,
    *,
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    output_format: str | None = None,
) -> None:
    """Execute the messages verb."""
    from polylogue.api import Polylogue

    async def _run() -> None:
        async with Polylogue.open(config=cast(Config, request.params.get("_config"))) as api:
            try:
                messages, total = await api.get_messages_paginated(
                    session_id,
                    limit=limit,
                    offset=offset,
                )
            except SessionNotFoundError:
                env.ui.error(f"Session not found: {session_id}")
                return

            fmt = output_format or "markdown"

            def _message_document(m: object) -> dict[str, object]:
                return cast(
                    "dict[str, object]",
                    model_json_document(
                        SessionMessageRowPayload.from_message(m, session_id=session_id),  # type: ignore[arg-type]
                        exclude_none=True,
                    ),
                )

            if fmt == "json":
                import json as _json

                # Finite machine-output contract (#1818): one JSON value.
                payload = model_json_document(
                    SessionMessagesResponsePayload(
                        session_id=session_id,
                        messages=tuple(
                            SessionMessageRowPayload.from_message(m, session_id=session_id) for m in messages
                        ),
                        total=total,
                        limit=limit,
                        offset=offset,
                    ),
                    exclude_none=True,
                )
                # Machine output goes through click.echo (raw stdout), NOT
                # env.ui.print: the Rich console defaults markup=True and would
                # interpret/strip bracket sequences like "[bold]" inside message
                # text, corrupting the exact bytes json.dumps produced (#1818).
                click.echo(_json.dumps(payload, indent=2))
            elif fmt == "ndjson":
                import json as _json

                # Streaming machine-output contract (#1818): one JSON document
                # per line. Each line is self-contained, carrying session_id so
                # downstream consumers do not need an out-of-band envelope.
                # Raw click.echo (not env.ui.print) so Rich markup never mangles
                # message text inside the JSON document.
                for m in messages:
                    line = {"session_id": session_id, **_message_document(m)}
                    click.echo(_json.dumps(line))
            else:
                for msg in messages:
                    role = str(msg.role)
                    message_type_label = str(msg.message_type.value)
                    text = msg.text or ""
                    if text:
                        env.ui.print(f"[{role} {message_type_label}] {text[:500]}{'...' if len(text) > 500 else ''}")
                        env.ui.print("---")

    run_coroutine_sync(_run())


def run_raw(
    env: AppEnv,
    request: RootModeRequest,
    *,
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    output_format: str = "json",
) -> None:
    """Execute the raw verb."""
    from polylogue.api import Polylogue

    async def _run() -> None:
        async with Polylogue.open(config=cast(Config, request.params.get("_config"))) as api:
            artifacts, total = await api.get_raw_artifacts_for_session(
                session_id,
                limit=limit,
                offset=offset,
            )

            if not artifacts:
                env.ui.error(f"No raw artifacts found for session: {session_id}")
                return

            if output_format == "json":
                import json as _json

                payload = {
                    "session_id": session_id,
                    "artifacts": [
                        {
                            "raw_id": r.get("raw_id", ""),
                            "source_name": r.get("source_name", ""),
                            "source_path": r.get("source_path", ""),
                            "blob_size": r.get("blob_size", 0),
                        }
                        for r in artifacts
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
                # Machine output uses raw stdout so Rich markup never rewrites
                # JSON bytes and read-view delivery can capture file/clipboard
                # targets consistently.
                click.echo(_json.dumps(payload, indent=2))
            else:
                import yaml

                click.echo(yaml.dump(artifacts))

    run_coroutine_sync(_run())


__all__ = ["run_messages", "run_raw"]
