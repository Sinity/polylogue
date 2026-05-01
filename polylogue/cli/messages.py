"""CLI execution for messages and raw verbs."""

from __future__ import annotations

from typing import cast

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.message.roles import MessageRoleFilter, normalize_message_roles
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.storage.backends.queries.message_query_reads import MessageTypeName


def run_messages(
    env: AppEnv,
    request: RootModeRequest,
    *,
    conversation_id: str,
    message_role: tuple[str, ...] = (),
    message_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    no_code_blocks: bool = False,
    no_tool_calls: bool = False,
    no_tool_outputs: bool = False,
    no_file_reads: bool = False,
    prose_only: bool = False,
    output_format: str | None = None,
) -> None:
    """Execute the messages verb."""
    from polylogue.api import Polylogue

    async def _run() -> None:
        async with Polylogue.open(config=cast(Config, request.params.get("_config"))) as api:
            roles: MessageRoleFilter = normalize_message_roles(message_role) if message_role else ()
            projection = ContentProjectionSpec.from_params(
                {
                    "no_code_blocks": no_code_blocks,
                    "no_tool_calls": no_tool_calls,
                    "no_tool_outputs": no_tool_outputs,
                    "no_file_reads": no_file_reads,
                    "prose_only": prose_only,
                }
            )

            result = await api.get_messages_paginated(
                conversation_id,
                message_role=roles,
                message_type=cast(MessageTypeName, message_type),
                limit=limit,
                offset=offset,
                content_projection=projection,
            )

            if result is None:
                env.ui.error(f"Conversation not found: {conversation_id}")
                return

            messages, total = result
            fmt = output_format or "markdown"

            if fmt == "json":
                import json as _json

                payload = {
                    "conversation_id": conversation_id,
                    "messages": [
                        {
                            "id": m.get("id", ""),
                            "role": m.get("role", ""),
                            "message_type": m.get("message_type", "message"),
                            "text": m.get("text", ""),
                        }
                        for m in messages
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
                env.ui.print(_json.dumps(payload, indent=2))
            else:
                for msg in messages:
                    role = str(msg.get("role", "unknown"))
                    message_type_label = str(msg.get("message_type", "message"))
                    text = str(msg.get("text", ""))
                    if text:
                        env.ui.print(f"[{role} {message_type_label}] {text[:500]}{'...' if len(text) > 500 else ''}")
                        env.ui.print("---")

    run_coroutine_sync(_run())


def run_raw(
    env: AppEnv,
    request: RootModeRequest,
    *,
    conversation_id: str,
    limit: int = 50,
    offset: int = 0,
    output_format: str = "json",
) -> None:
    """Execute the raw verb."""
    from polylogue.api import Polylogue

    async def _run() -> None:
        async with Polylogue.open(config=cast(Config, request.params.get("_config"))) as api:
            records, total = await api.get_raw_records_for_conversation(
                conversation_id,
                limit=limit,
                offset=offset,
            )

            if not records:
                env.ui.error(f"No raw records found for conversation: {conversation_id}")
                return

            if output_format == "json":
                import json as _json

                payload = {
                    "conversation_id": conversation_id,
                    "records": [
                        {
                            "raw_id": r.get("raw_id", ""),
                            "provider_name": r.get("provider_name", ""),
                            "source_path": r.get("source_path", ""),
                            "blob_size": r.get("blob_size", 0),
                        }
                        for r in records
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
                env.ui.print(_json.dumps(payload, indent=2))
            else:
                import yaml

                env.ui.print(yaml.dump(records))

    run_coroutine_sync(_run())


__all__ = ["run_messages", "run_raw"]
