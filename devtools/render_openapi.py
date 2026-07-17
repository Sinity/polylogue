"""Render the OpenAPI schema for stable daemon query HTTP surfaces.

The payload models in :mod:`polylogue.surfaces.payloads` and the browser
credential models in :mod:`polylogue.daemon.web_auth` are typed contracts
shared by daemon HTTP clients and other Polylogue surfaces. To make the daemon
HTTP surface machine-discoverable — and
to let external tooling auto-generate clients without touching private
Python types — this command writes an OpenAPI 3.1 document at
``docs/openapi/search.yaml`` that exposes both request parameters and
response schemas for stable daemon query routes such as
``GET /api/sessions`` and ``GET /api/query-units``.

The schema is derived from Pydantic models. Do not edit the output by
hand; regenerate it with ``devtools render openapi``. The ``--check``
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
from polylogue.archive.query.metadata import terminal_query_examples, terminal_query_source_list
from polylogue.archive.viewport import (
    read_view_http_choices,
    read_view_http_format_choices,
    read_view_http_query_params,
)
from polylogue.daemon.route_contracts import ROUTE_CONTRACTS, RouteContract
from polylogue.daemon.web_auth import (
    WebCredentialBootstrapPayload,
    WebCredentialFailurePayload,
    WebCredentialRevocationPayload,
)
from polylogue.surfaces.payloads import (
    RANKING_POLICY_MIXED,
    RANKING_POLICY_VERSION,
    AssertionClaimListPayload,
    QueryErrorPayload,
    QueryUnitAggregateEnvelope,
    QueryUnitEnvelope,
    SearchEnvelope,
    SessionListRowPayload,
    SessionReadViewEnvelope,
    SessionSearchHitPayload,
)

DEFAULT_OUTPUT_PATH = Path("docs/openapi/search.yaml")
OPENAPI_VERSION = "3.1.0"
POLYLOGUE_API_VERSION = "1"

# Pydantic models whose JSON Schema is published in ``components.schemas``.
_PUBLISHED_MODELS: tuple[type[BaseModel], ...] = (
    SearchEnvelope,
    QueryUnitEnvelope,
    QueryUnitAggregateEnvelope,
    SessionReadViewEnvelope,
    AssertionClaimListPayload,
    SessionSearchHitPayload,
    SessionListRowPayload,
    WebCredentialBootstrapPayload,
    WebCredentialRevocationPayload,
    WebCredentialFailurePayload,
    QueryErrorPayload,
)

_WEB_CREDENTIAL_FAILURE_STATES = [
    "web_credential_missing",
    "web_credential_invalid",
    "web_credential_expired",
    "web_credential_revoked",
    "web_credential_wrong_origin",
    "web_credential_insufficient_scope",
]


def _protected_read_security() -> list[dict[str, list[str]]]:
    return [{"machineBearer": []}, {"webCredentialCookie": []}]


def _route_contract_payload(contract: RouteContract) -> dict[str, str]:
    """Return the OpenAPI vendor-extension shape for one daemon route contract."""
    payload = {
        "method": contract.method,
        "pattern": contract.pattern,
        "kind": contract.kind,
        "stability": contract.stability,
        "auth_policy": contract.auth_policy,
        "response_contract": contract.response_contract,
    }
    if contract.notes:
        payload["notes"] = contract.notes
    return payload


def _read_view_http_description() -> str:
    """Return route text derived from the read-view HTTP capability contract."""

    views = ", ".join(f"``{view}``" for view in read_view_http_choices())
    return (
        "Executes one supported single-session read profile and returns the "
        "stable ``SessionReadViewEnvelope``. Supported profiles currently "
        f"include {views}. The envelope shape is stable; individual ``payload`` "
        "content is the shared DTO for the selected read profile."
    )


def _read_view_parameter(name: str) -> dict[str, Any]:
    """Return an OpenAPI query parameter for a contract-owned read-view option."""

    details: dict[str, dict[str, Any]] = {
        "confidence_threshold": {
            "description": "Minimum confidence for correlation evidence.",
            "schema": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.3},
        },
        "include_messages": {
            "description": "Include message excerpts in context-image payloads.",
            "schema": {"type": "boolean", "default": True},
        },
        "limit": {
            "description": "Messages/neighbor page size where applicable.",
            "schema": {"type": "integer", "minimum": 1},
        },
        "max_tokens": {
            "description": "Token budget for the compiled context image; over-budget segments are omitted.",
            "schema": {"type": "integer", "minimum": 1},
        },
        "message_role": {
            "description": "Restrict the messages view to one or more message roles.",
            "schema": {"type": "string"},
        },
        "message_type": {
            "description": "Restrict the messages view by message type.",
            "schema": {"type": "string"},
        },
        "no_code_blocks": {
            "description": "Exclude code blocks from the messages projection.",
            "schema": {"type": "boolean", "default": False},
        },
        "no_file_reads": {
            "description": "Exclude file-read material from the messages projection.",
            "schema": {"type": "boolean", "default": False},
        },
        "no_redact": {
            "description": "Return unredacted context-image path/provenance details where available.",
            "schema": {"type": "boolean", "default": False},
        },
        "no_tool_calls": {
            "description": "Exclude tool-call material from the messages projection.",
            "schema": {"type": "boolean", "default": False},
        },
        "offset": {
            "description": "Message offset for the messages view.",
            "schema": {"type": "integer", "minimum": 0},
        },
        "related_limit": {
            "description": "Number of related sessions to include in the context view.",
            "schema": {"type": "integer", "minimum": 1, "default": 5},
        },
        "repo_path": {
            "description": "Repository path for correlation evidence.",
            "schema": {"type": "string"},
        },
        "since_hours": {
            "description": "Correlation scan window on each side of the session.",
            "schema": {"type": "integer", "minimum": 1, "default": 2},
        },
        "window_hours": {
            "description": "Neighbor discovery window in hours.",
            "schema": {"type": "integer", "minimum": 1, "default": 24},
        },
    }
    default = {
        "description": f"Read-view option advertised by /api/read-view-profiles: {name}.",
        "schema": {"type": "string"},
    }
    detail = details.get(name, default)
    return {"name": name, "in": "query", "required": False, **detail}


def _read_view_http_parameters() -> list[dict[str, Any]]:
    """Return OpenAPI parameters for the read-view route from shared capabilities."""

    parameters: list[dict[str, Any]] = [
        {
            "name": "session_id",
            "in": "path",
            "description": "Resolved Polylogue session id.",
            "required": True,
            "schema": {"type": "string"},
        },
        {
            "name": "view",
            "in": "query",
            "description": "Read-view id from /api/read-view-profiles.",
            "required": False,
            "schema": {"type": "string", "enum": list(read_view_http_choices()), "default": "messages"},
        },
        {
            "name": "format",
            "in": "query",
            "description": "Output format accepted by the selected read view.",
            "required": False,
            "schema": {"type": "string", "enum": list(read_view_http_format_choices()), "default": "json"},
        },
    ]
    parameters.extend(_read_view_parameter(name) for name in read_view_http_query_params())
    return parameters


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


def _web_credential_error_response(description: str) -> dict[str, Any]:
    """Return the generated-client contract for one credential failure."""

    return {
        "description": description,
        "headers": {
            "X-Polylogue-Web-Credential-State": {
                "schema": {
                    "type": "string",
                    "enum": _WEB_CREDENTIAL_FAILURE_STATES,
                }
            }
        },
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/WebCredentialFailurePayload"}}},
    }


def _query_error_response(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/QueryErrorPayload"}}},
    }


def _protected_auth_error_response(description: str) -> dict[str, Any]:
    """Describe either machine-bearer admission errors or typed web states."""

    return {
        "description": description,
        "headers": {
            "X-Polylogue-Web-Credential-State": {
                "description": "Present when the first-party web credential was evaluated.",
                "schema": {"type": "string", "enum": _WEB_CREDENTIAL_FAILURE_STATES},
            }
        },
        "content": {
            "application/json": {
                "schema": {
                    "anyOf": [
                        {"$ref": "#/components/schemas/QueryErrorPayload"},
                        {"$ref": "#/components/schemas/WebCredentialFailurePayload"},
                    ]
                }
            }
        },
    }


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
                "generated from Pydantic models in `polylogue.surfaces.payloads` "
                "and `polylogue.daemon.web_auth`; "
                "do not edit by hand. "
                f"Run `{control_plane_command('render openapi')}` to regenerate."
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
            "/api/web-auth/session": {
                "post": {
                    "summary": "Bootstrap or rotate a first-party web credential",
                    "description": (
                        "Loopback, exact-origin browser bootstrap. The opaque credential is returned only in an "
                        "HttpOnly SameSite=Strict cookie; the JSON body contains lifecycle metadata, scopes, and "
                        "expiry but never credential bytes."
                    ),
                    "operationId": "bootstrapWebCredential",
                    "security": [],
                    "responses": {
                        "201": {
                            "description": "Credential issued and protected cookie set.",
                            "headers": {
                                "Set-Cookie": {
                                    "description": "Protected HttpOnly first-party credential transport.",
                                    "schema": {"type": "string", "writeOnly": True},
                                },
                                "X-Polylogue-Web-Credential-State": {"schema": {"type": "string", "const": "ready"}},
                            },
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/WebCredentialBootstrapPayload"}
                                }
                            },
                        },
                        "400": _query_error_response("Credential-shaped query parameters are forbidden."),
                        "403": _web_credential_error_response("Bootstrap origin was not the daemon's authority."),
                    },
                    "x-polylogue-recoverable-states": _WEB_CREDENTIAL_FAILURE_STATES,
                },
                "delete": {
                    "summary": "Revoke the current first-party web credential",
                    "operationId": "revokeWebCredential",
                    "security": [{"webCredentialCookie": []}],
                    "responses": {
                        "200": {
                            "description": "Credential revoked and browser cookie expired.",
                            "headers": {
                                "Set-Cookie": {
                                    "description": "Expires the protected first-party credential cookie.",
                                    "schema": {"type": "string", "writeOnly": True},
                                },
                                "X-Polylogue-Web-Credential-State": {
                                    "schema": {"type": "string", "const": "web_credential_revoked"}
                                },
                            },
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/WebCredentialRevocationPayload"}
                                }
                            },
                        },
                        "400": _query_error_response("Credential-shaped query parameters are forbidden."),
                        "401": _web_credential_error_response("Credential is missing, invalid, expired, or revoked."),
                        "403": _web_credential_error_response("Credential origin or scope is not admitted."),
                    },
                    "x-polylogue-recoverable-states": _WEB_CREDENTIAL_FAILURE_STATES,
                },
            },
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
                    "security": _protected_read_security(),
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
                        },
                        "400": _query_error_response("Malformed query or credential-shaped query parameter."),
                        "401": _protected_auth_error_response("Machine bearer or web credential was not accepted."),
                        "403": _protected_auth_error_response(
                            "Host, origin, or web credential scope was not admitted."
                        ),
                    },
                }
            },
            "/api/query-units": {
                "get": {
                    "summary": "Terminal query-unit rows",
                    "description": (
                        "Returns terminal row results for explicit "
                        f"``{terminal_query_source_list()} where ...`` expressions. "
                        "This endpoint shares the query-unit row and aggregate envelope contracts "
                        "with CLI JSON output, MCP ``query_units``, and "
                        "``Polylogue.query_units()``."
                    ),
                    "operationId": "queryUnits",
                    "security": _protected_read_security(),
                    "parameters": [
                        {
                            "name": "expression",
                            "in": "query",
                            "description": (
                                "Explicit unit-source expression. Examples: "
                                + "; ".join(f"``{example}``" for example in terminal_query_examples())
                                + "."
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
                            "name": "has_paste_evidence",
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
                            "description": "Terminal query-unit row or aggregate result envelope.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "oneOf": [
                                            {"$ref": "#/components/schemas/QueryUnitEnvelope"},
                                            {"$ref": "#/components/schemas/QueryUnitAggregateEnvelope"},
                                        ]
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Malformed query or non-terminal session expression.",
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/QueryErrorPayload"}}
                            },
                        },
                        "401": _protected_auth_error_response("Machine bearer or web credential was not accepted."),
                        "403": _protected_auth_error_response(
                            "Host, origin, or web credential scope was not admitted."
                        ),
                    },
                }
            },
            "/api/sessions/{session_id}/read": {
                "get": {
                    "summary": "Execute a session read-view",
                    "description": _read_view_http_description(),
                    "operationId": "readSessionView",
                    "security": _protected_read_security(),
                    "parameters": _read_view_http_parameters(),
                    "responses": {
                        "200": {
                            "description": "Executed read-view envelope.",
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/SessionReadViewEnvelope"}}
                            },
                        },
                        "400": _query_error_response("Unsupported view/format or credential-shaped query parameter."),
                        "404": {"description": "Session not found."},
                        "401": _protected_auth_error_response("Machine bearer or web credential was not accepted."),
                        "403": _protected_auth_error_response(
                            "Host, origin, or web credential scope was not admitted."
                        ),
                    },
                }
            },
            "/api/assertions": {
                "get": {
                    "summary": "List assertion-backed overlay claims",
                    "description": (
                        "Returns assertion claims through the stable "
                        "``AssertionClaimListPayload`` shared by the web workbench, "
                        "Python API, and MCP assertion-read surface. Defaults to active "
                        "and candidate claims; use ``status=all`` for an explicit full "
                        "lifecycle read."
                    ),
                    "operationId": "listAssertionClaims",
                    "security": _protected_read_security(),
                    "parameters": [
                        {
                            "name": "target_ref",
                            "in": "query",
                            "description": "Optional target object ref, such as ``session:<id>``.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "scope_ref",
                            "in": "query",
                            "description": "Optional assertion scope object ref, such as ``repo:polylogue``.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "kind",
                            "in": "query",
                            "description": "Assertion kind filter. May be repeated or comma-separated.",
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "status",
                            "in": "query",
                            "description": (
                                "Lifecycle status filter. Defaults to active,candidate; "
                                "``all`` includes every persisted status."
                            ),
                            "required": False,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "context_inject",
                            "in": "query",
                            "description": "Restrict by assertion context_policy.inject.",
                            "required": False,
                            "schema": {"type": "boolean"},
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "Maximum claims returned.",
                            "required": False,
                            "schema": {"type": "integer", "minimum": 1, "default": 50},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Assertion claim list envelope.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AssertionClaimListPayload"}
                                }
                            },
                        },
                        "400": _query_error_response("Malformed or credential-shaped query parameter."),
                        "401": _protected_auth_error_response("Machine bearer or web credential was not accepted."),
                        "403": _protected_auth_error_response(
                            "Host, origin, or web credential scope was not admitted."
                        ),
                    },
                }
            },
        },
        "components": {
            "schemas": schemas,
            "securitySchemes": {
                "machineBearer": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "opaque",
                    "description": "Machine-client token configured by --api-auth-token.",
                },
                "webCredentialCookie": {
                    "type": "apiKey",
                    "in": "cookie",
                    "name": "polylogue_web_credential",
                    "description": (
                        "Short-lived, origin-bound HttpOnly credential issued by bootstrapWebCredential. "
                        "Browser clients use credentials: same-origin and never read the cookie value."
                    ),
                },
            },
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
        "x-polylogue-route-contracts": [_route_contract_payload(contract) for contract in ROUTE_CONTRACTS],
    }


def _build_yaml_body() -> str:
    document = _build_openapi_document()
    header = (
        "# Generated by `" + control_plane_command("render openapi") + "`. Do not edit by hand.\n"
        "# Source models: polylogue.surfaces.payloads query envelopes and polylogue.daemon.web_auth.\n"
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
            print(f"render openapi: out of sync: {output_path}", file=sys.stderr)
            print(
                "render openapi: run: " + control_plane_command("render openapi"),
                file=sys.stderr,
            )
            return 1
        print(f"render openapi: sync OK: {output_path}")
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
