"""CLI execution for messages and raw verbs."""

from __future__ import annotations

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.lib.message.roles import MessageRoleFilter, normalize_message_roles
from polylogue.lib.semantic.content_projection import ContentProjectionSpec


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
        async with Polylogue.open(config=request.params.get("_config")) as api:
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
                message_type=message_type,
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
                        {"id": m.get("id", ""), "role": m.get("role", ""), "text": m.get("text", "")} for m in messages
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
                env.ui.print(_json.dumps(payload, indent=2))
            else:
                for msg in messages:
                    role = msg.get("role", "unknown")
                    text = msg.get("text", "")
                    if text:
                        env.ui.print(f"[{role}] {text[:500]}{'...' if len(text) > 500 else ''}")
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
        async with Polylogue.open(config=request.params.get("_config")) as api:
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
