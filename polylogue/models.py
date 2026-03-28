from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


def _sanitise_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    data = deepcopy(chunk)

    # Normalise aliases produced by various importers.
    if "token_count" in data and "tokenCount" not in data:
        data["tokenCount"] = data.pop("token_count")
    if "finish_reason" in data and "finishReason" not in data:
        data["finishReason"] = data.pop("finish_reason")

    role = data.get("role")
    if not isinstance(role, str):
        data["role"] = "model"

    if "content" in data and isinstance(data["content"], list):
        sanitised_content = [dict(part) for part in data["content"] if isinstance(part, dict)]
        if sanitised_content:
            data["content"] = sanitised_content
        else:
            data.pop("content", None)

    text = data.get("text")
    if text is not None and not isinstance(text, str):
        data.pop("text", None)

    token_count = data.get("tokenCount")
    if token_count is not None:
        try:
            data["tokenCount"] = int(token_count)
        except (TypeError, ValueError):
            data.pop("tokenCount", None)

    is_thought = data.get("isThought")
    if is_thought is not None and not isinstance(is_thought, bool):
        data.pop("isThought", None)

    # Drop explicit None values to match the previous behaviour of model_dump.
    return {key: value for key, value in data.items() if value is not None}


def validate_chunks(raw_chunks: List[Any]) -> List[dict]:
    """Validate and normalise chunk payloads without requiring Pydantic."""

    validated: List[dict] = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        validated.append(_sanitise_chunk(chunk))
    return validated
