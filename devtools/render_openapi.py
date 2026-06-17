"""Render the OpenAPI schema for stable daemon query HTTP surfaces.

The surface payload models in :mod:`polylogue.surfaces.payloads` are the
typed result contracts shared across CLI JSON, MCP, the Python API, and
daemon HTTP. To make the daemon HTTP surface machine-discoverable — and
to let external tooling auto-generate clients without touching private
Python types — this command writes an OpenAPI 3.1 document at
``docs/openapi/search.yaml`` that exposes both request parameters and
response schemas for stable daemon query routes such as
``GET /api/sessions`` and ``GET /api/query-units``.

The schema is derived from Pydantic models. Do not edit the output by
hand; regenerate it with ``devtools render-openapi``. The ``--check``
mode is for CI sync verification.

This is intentionally route-backed: only endpoints with typed payload
contracts and daemon handlers should be added here.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from devtools.command_catalog import control_plane_command
from devtools.render_support import write_if_changed
from polylogue.surfaces.payloads import (
    RANKING_POLICY_MIXED,
    RANKING_POLICY_VERSION,
    QueryUnitEnvelope,
    SearchEnvelope,
    SessionListRowPayload,
    SessionSearchHitPayload,
)

DEFAULT_OUTPUT_PATH = Path("docs/openapi/search.yaml")
OPENAPI_VERSION = "3.1.0"
POLYLOGUE_API_VERSION = "1"

# Pydantic models whose JSON Schema is published in ``components.schemas``.
_PUBLISHED_MODELS: tuple[type[BaseModel], ...] = (
    SearchEnvelope,
    QueryUnitEnvelope,
    SessionSearchHitPayload,
    SessionListRowPayload,
)


def _collect_component_schemas() -> dict[str, Any]:
    """Build the OpenAPI ``components.schemas`` map from Pydantic models.

    ``pydantic.BaseModel.model_json_schema(mode="serialization")`` emits a
    JSON Schema document for the model's serialised shape, with nested
    submodel schemas inlined under ``$defs``. For OpenAPI we hoist those
    nested definitions into a single top-level ``schemas`` map, and
    rewrite ``$ref`` paths from ``#/$defs/X`` to ``#/components/schemas/X``.
    """
    schemas: dict[str, Any] = {}
    for model in _PUBLISHED_MODELS:
        schema = model.model_json_schema(
            mode="serialization",
            ref_template="#/components/schemas/{model}",
        )
        # Pydantic stores nested model definitions under "$defs"; move
        # them to the top-level OpenAPI ``schemas`` map.
        nested = schema.pop("$defs", {})
        for name, sub_schema in nested.items():
            schemas.setdefault(name, sub_schema)
        title = schema.get("title", model.__name__)
        schemas[title] = schema
    return schemas


def _build_openapi_document() -> dict[str, Any]:
    schemas = _collect_component_schemas()
    return {
        "openapi": OPENAPI_VERSION,
        "info": {
            "title": "Polylogue Search API",
            "version": POLYLOGUE_API_VERSION,
            "description": (
                "Local daemon HTTP API exposing typed query envelopes shared "
                "across CLI JSON, MCP, and the Python API. This document is "
                "generated from Pydantic models in `polylogue.surfaces.payloads`; "
                "do not edit by hand. "
                f"Run `{control_plane_command('render-openapi')}` to regenerate."
            ),
        },
        "servers": [
            {
                "url": "http://127.0.0.1:{port}",
                "description": "Local Polylogue daemon (loopback only).",
                "variables": {
                    "port": {
                        "default": "8765",
                        "description": "Daemon HTTP port; see `polylogued status`.",
                    }
                },
            }
        ],
        "paths": {
            "/api/sessions": {
                "get": {
                    "summary": "Ranked session search and list",
                    "description": (
                        "When ``query`` is supplied, returns a ranked search "
                        "envelope (``SearchEnvelope``) with per-hit match "
                        "evidence and pagination handles. When ``query`` is "
                        "omitted, returns the session list envelope. "
                        "Both envelopes carry ``total``, ``limit``, "
                        "``offset``, and ``next_cursor`` for cursor-based "
                        "pagination."
                    ),
                    "operationId": "searchSessions",
                    "parameters": [
                        {
                            "name": "query",
                            "in": "query",
                            "description": "FTS5 search query string.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "Page size (default 50).",
                            "required": False,
                            "schema": {"type": "integer", "minimum": 1, "default": 50},
                        },
                        {
                            "name": "offset",
                            "in": "query",
                            "description": (
                                "Row offset. Best-effort for ranked search; use ``next_cursor`` for stable pagination."
                            ),
                            "required": False,
                            "schema": {"type": "integer", "minimum": 0, "default": 0},
                        },
                        {
                            "name": "provider",
                            "in": "query",
                            "description": "Provider name filter.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "since",
                            "in": "query",
                            "description": "ISO 8601 lower bound on session date.",
                            "required": False,
                            "schema": {"type": "string", "format": "date-time"},
                        },
                        {
                            "name": "retrieval_lane",
                            "in": "query",
                            "description": (
                                "Retrieval lane: ``auto``, ``dialogue``, ``actions``, ``hybrid``, ``semantic``."
                            ),
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["auto", "dialogue", "actions", "hybrid", "semantic"],
                                "default": "auto",
                            },
                        },
                        {
                            "name": "cursor",
                            "in": "query",
                            "description": (
                                "Opaque keyset cursor previously returned as ``next_cursor`` on a search "
                                "response. Stable across archive growth and daemon restart (#1268). "
                                "MUST be passed back unchanged. Rejected with 400 when malformed or when "
                                "the retrieval lane no longer matches."
                            ),
                            "required": False,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": (
                                "Ranked search envelope (when ``query`` is supplied) or session list envelope."
                            ),
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/SearchEnvelope"}}
                            },
                        }
                    },
                }
            },
            "/api/query-units": {
                "get": {
                    "summary": "Terminal message/action/block query rows",
                    "description": (
                        "Returns terminal row results for explicit "
                        "``messages/actions/blocks where ...`` expressions. "
                        "This endpoint shares the ``QueryUnitEnvelope`` contract "
                        "with CLI JSON output, MCP ``query_units``, and "
                        "``Polylogue.query_units()``."
                    ),
                    "operationId": "queryUnits",
                    "parameters": [
                        {
                            "name": "expression",
                            "in": "query",
                            "description": (
                                "Explicit unit-source expression such as "
                                "``messages where role:assistant AND text:timeout``."
                            ),
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "Page size (default 50).",
                            "required": False,
                            "schema": {"type": "integer", "minimum": 1, "default": 50},
                        },
                        {
                            "name": "offset",
                            "in": "query",
                            "description": "Row offset for terminal query-unit pagination.",
                            "required": False,
                            "schema": {"type": "integer", "minimum": 0, "default": 0},
                        },
                        {
                            "name": "origin",
                            "in": "query",
                            "description": "Optional session-origin scope for terminal row results.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "tag",
                            "in": "query",
                            "description": "Optional session tag scope for terminal row results.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "exclude_tag",
                            "in": "query",
                            "description": "Optional comma-separated session tags to exclude from terminal row results.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "repo",
                            "in": "query",
                            "description": "Optional comma-separated repo-name scope for terminal row results.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "has_type",
                            "in": "query",
                            "description": "Optional comma-separated block types required on the containing session.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "referenced_path",
                            "in": "query",
                            "description": "Optional comma-separated referenced paths required on the containing session.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "cwd_prefix",
                            "in": "query",
                            "description": "Optional working-directory prefix required on the containing session.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "tool",
                            "in": "query",
                            "description": "Optional comma-separated tool names required on the containing session.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "exclude_tool",
                            "in": "query",
                            "description": "Optional comma-separated tool names excluded from the containing session.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "action",
                            "in": "query",
                            "description": "Optional comma-separated action kinds required on the containing session.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "exclude_action",
                            "in": "query",
                            "description": "Optional comma-separated action kinds excluded from the containing session.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "action_sequence",
                            "in": "query",
                            "description": "Optional action sequence required on the containing session.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "action_text",
                            "in": "query",
                            "description": "Optional action text required on the containing session.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "title",
                            "in": "query",
                            "description": "Optional session-title substring scope for terminal row results.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "since",
                            "in": "query",
                            "description": "Optional session lower time bound, using the shared query date parser.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "until",
                            "in": "query",
                            "description": "Optional session upper time bound, using the shared query date parser.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "has_tool_use",
                            "in": "query",
                            "description": "Restrict terminal rows to sessions with tool-use evidence.",
                            "required": False,
                            "schema": {"type": "boolean", "default": False},
                        },
                        {
                            "name": "has_paste",
                            "in": "query",
                            "description": "Restrict terminal rows to sessions with paste evidence.",
                            "required": False,
                            "schema": {"type": "boolean", "default": False},
                        },
                        {
                            "name": "has_thinking",
                            "in": "query",
                            "description": "Restrict terminal rows to sessions with thinking blocks.",
                            "required": False,
                            "schema": {"type": "boolean", "default": False},
                        },
                        {
                            "name": "typed_only",
                            "in": "query",
                            "description": "Restrict terminal rows to typed sessions without paste evidence.",
                            "required": False,
                            "schema": {"type": "boolean", "default": False},
                        },
                        {
                            "name": "min_messages",
                            "in": "query",
                            "description": "Restrict terminal rows to sessions with at least this many messages.",
                            "required": False,
                            "schema": {"type": "integer", "minimum": 0},
                        },
                        {
                            "name": "max_messages",
                            "in": "query",
                            "description": "Restrict terminal rows to sessions with at most this many messages.",
                            "required": False,
                            "schema": {"type": "integer", "minimum": 0},
                        },
                        {
                            "name": "min_words",
                            "in": "query",
                            "description": "Restrict terminal rows to sessions with at least this many words.",
                            "required": False,
                            "schema": {"type": "integer", "minimum": 0},
                        },
                        {
                            "name": "max_words",
                            "in": "query",
                            "description": "Restrict terminal rows to sessions with at most this many words.",
                            "required": False,
                            "schema": {"type": "integer", "minimum": 0},
                        },
                        {
                            "name": "message_type",
                            "in": "query",
                            "description": "Restrict terminal rows by session message-type evidence.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Terminal query-unit result envelope.",
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/QueryUnitEnvelope"}}
                            },
                        },
                        "400": {
                            "description": "Malformed query or non-terminal session expression.",
                        },
                    },
                }
            },
        },
        "components": {
            "schemas": schemas,
            "x-polylogue-ranking-policy": {
                "default": RANKING_POLICY_MIXED,
                "version": RANKING_POLICY_VERSION,
                "description": (
                    "The ``ranking_policy`` and ``ranking_policy_version`` "
                    "fields on ``SearchEnvelope`` declare the ordering "
                    "semantics in use. Consumers should pin a known "
                    "version and treat any change as a contract event."
                ),
            },
        },
    }


def _build_yaml_body() -> str:
    document = _build_openapi_document()
    header = (
        "# Generated by `" + control_plane_command("render-openapi") + "`. Do not edit by hand.\n"
        "# Source models: polylogue.surfaces.payloads query envelopes.\n"
    )
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=120)
    return header + body


def render(output_path: Path, *, check: bool) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = _build_yaml_body()
    if check:
        try:
            current = output_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            current = ""
        if current != body:
            print(f"render-openapi: out of sync: {output_path}", file=sys.stderr)
            print(
                "render-openapi: run: " + control_plane_command("render-openapi"),
                file=sys.stderr,
            )
            return 1
        print(f"render-openapi: sync OK: {output_path}")
        return 0
    write_if_changed(output_path, body)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the daemon query OpenAPI schema from Pydantic payload models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the rendered schema is out of sync.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return render(args.output, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "render"]
