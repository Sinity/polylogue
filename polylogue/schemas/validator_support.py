"""Shared validation sampling and drift-detection helpers."""

from __future__ import annotations

import re
from typing import Any

from polylogue.lib.raw_payload import extract_payload_samples
from polylogue.types import Provider

_RECORD_VALIDATION_PROVIDERS = {Provider.CLAUDE_CODE, Provider.CODEX}

_UUID_KEY_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def validation_samples(
    payload: Any,
    *,
    schema: dict[str, Any],
    provider: Provider | None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Extract representative objects from a payload for validation."""
    granularity = schema.get("x-polylogue-sample-granularity")
    if not isinstance(granularity, str):
        granularity = "record" if provider in _RECORD_VALIDATION_PROVIDERS else "document"
    return extract_payload_samples(
        payload,
        sample_granularity=granularity,
        max_samples=max_samples,
    )


def format_validation_error(error: Any) -> str:
    path = ".".join(str(part) for part in error.absolute_path) or "root"
    return f"{path}: {error.message}"


def detect_drift(
    data: dict[str, Any],
    schema: dict[str, Any],
    path: str,
) -> list[str]:
    """Detect fields in data not present in schema (drift)."""
    warnings: list[str] = []
    schema_props = set(schema.get("properties", {}).keys())
    has_additional = schema.get("additionalProperties", True)

    for key, value in data.items():
        current_path = f"{path}.{key}" if path else key

        if key not in schema_props:
            if has_additional is False:
                warnings.append(f"Unexpected field: {current_path}")
            elif has_additional is True:
                pass
            elif isinstance(has_additional, dict):
                dynamic_container = bool(schema.get("x-polylogue-dynamic-keys"))
                if not dynamic_container and not looks_dynamic_key(key):
                    warnings.append(f"Unexpected field: {current_path}")
                if isinstance(value, dict):
                    warnings.extend(detect_drift(value, has_additional, current_path))
        else:
            prop_schema = schema["properties"][key]
            if isinstance(value, dict) and "properties" in prop_schema:
                warnings.extend(detect_drift(value, prop_schema, current_path))
            elif isinstance(value, list) and "items" in prop_schema:
                items_schema = prop_schema["items"]
                if isinstance(items_schema, dict) and "properties" in items_schema:
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            warnings.extend(detect_drift(item, items_schema, f"{current_path}[{i}]"))

    return warnings


def looks_dynamic_key(key: str) -> bool:
    """Detect dynamic identifier keys (UUIDs, hashes, generated IDs)."""
    if _UUID_KEY_RE.match(key):
        return True
    if re.match(r"^[0-9a-f]{24,}$", key, re.IGNORECASE):
        return True
    return bool(re.match(r"^(msg|node|conv|item|att)-[0-9a-f-]+$", key, re.IGNORECASE))
