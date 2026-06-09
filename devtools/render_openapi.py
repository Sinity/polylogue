"""Render the OpenAPI schema for the daemon search HTTP surface.

The :class:`~polylogue.surfaces.payloads.SearchEnvelope` is the typed
ranked-result contract shared across CLI JSON, MCP, the Python API, and
daemon HTTP (#1266). To make the daemon HTTP surface machine-discoverable
— and to let external tooling auto-generate clients without touching
private Python types — this command writes an OpenAPI 3.1 document at
``docs/openapi/search.yaml`` that exposes both the request shape (query
parameters) and the response schema (the typed ``SearchEnvelope`` and
its referenced models) for ``GET /api/sessions``.

The schema is derived from Pydantic models. Do not edit the output by
hand; regenerate it with ``devtools render-openapi``. The ``--check``
mode is for CI sync verification.

This is intentionally narrow: it covers the single ranked-search
endpoint that motivated #1266. Future slices (#1218, #1224) can extend
the same emitter to describe other daemon HTTP endpoints under one
OpenAPI tree.
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
                "Local daemon HTTP API exposing the typed ranked-result envelope "
                "(`SearchEnvelope`) shared across CLI JSON, MCP, and the Python "
                "API (#1266). This document is generated from Pydantic models "
                "in `polylogue.surfaces.payloads`; do not edit by hand. "
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
            }
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
        "# Source models: polylogue.surfaces.payloads.SearchEnvelope (#1266).\n"
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
        description="Render the daemon search OpenAPI schema from Pydantic models (#1266).",
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
