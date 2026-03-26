"""Shared support helpers for semantic-proof surface functions."""

from __future__ import annotations

import json
from typing import Any

from polylogue.rendering.semantic_proof_models import SemanticConversationProof
from polylogue.rendering.semantic_surface_registry import evaluate_semantic_contracts


def _build_surface_proof(
    *,
    surface: str,
    conversation_id: str,
    provider: str,
    input_facts: dict[str, Any],
    output_facts: dict[str, Any],
) -> SemanticConversationProof:
    return SemanticConversationProof(
        conversation_id=conversation_id,
        provider=provider,
        surface=surface,
        input_facts=input_facts,
        output_facts=output_facts,
        checks=evaluate_semantic_contracts(surface, input_facts, output_facts),
    )


def _parse_json_payload(rendered_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(rendered_text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_row(rendered_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(rendered_text)
    except Exception:
        return {}
    row = parsed[0] if isinstance(parsed, list) and parsed else {}
    return row if isinstance(row, dict) else {}


def _parse_yaml_payload(rendered_text: str) -> dict[str, Any]:
    try:
        import yaml

        parsed = yaml.safe_load(rendered_text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_yaml_row(rendered_text: str) -> dict[str, Any]:
    try:
        import yaml

        parsed = yaml.safe_load(rendered_text)
    except Exception:
        return {}
    row = parsed[0] if isinstance(parsed, list) and parsed else {}
    return row if isinstance(row, dict) else {}


__all__ = [
    "_build_surface_proof",
    "_parse_json_payload",
    "_parse_json_row",
    "_parse_yaml_payload",
    "_parse_yaml_row",
]
