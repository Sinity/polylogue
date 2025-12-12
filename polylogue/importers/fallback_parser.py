"""Heuristic fallback parser for when strict schema validation fails.

When a provider changes their export format and the strict Pydantic schemas fail,
this parser attempts to extract readable text from the raw JSON structure using
heuristics. It's better to show the user *something* than to show nothing.
"""
from __future__ import annotations

from typing import Any, Dict, List


def extract_text_recursively(obj: Any, min_length: int = 50, max_depth: int = 10) -> List[str]:
    """Recursively extract strings from a JSON structure.

    Args:
        obj: The object to search (dict, list, str, etc.)
        min_length: Minimum string length to consider "meaningful text"
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        List of extracted text strings
    """
    if max_depth <= 0:
        return []

    texts: List[str] = []

    if isinstance(obj, str):
        # Found a string - check if it's meaningful
        if len(obj) >= min_length and not obj.startswith("http"):
            # Avoid URLs and other non-content strings
            texts.append(obj)
    elif isinstance(obj, dict):
        # Recursively search dictionary values
        for key, value in obj.items():
            # Prioritize keys that likely contain content
            if key in ["text", "content", "message", "body", "parts"]:
                texts.extend(extract_text_recursively(value, min_length, max_depth - 1))
            else:
                texts.extend(extract_text_recursively(value, min_length, max_depth - 1))
    elif isinstance(obj, list):
        # Recursively search list items
        for item in obj:
            texts.extend(extract_text_recursively(item, min_length, max_depth - 1))

    return texts


def extract_timestamps(obj: Any, max_depth: int = 5) -> List[float]:
    """Extract timestamp-like numbers from JSON structure.

    Args:
        obj: The object to search
        max_depth: Maximum recursion depth

    Returns:
        List of potential timestamps
    """
    if max_depth <= 0:
        return []

    timestamps: List[float] = []

    if isinstance(obj, (int, float)):
        # Check if this looks like a Unix timestamp (reasonable year range)
        # Timestamps between 2020 and 2030
        if 1577836800 <= obj <= 1893456000:
            timestamps.append(float(obj))
    elif isinstance(obj, dict):
        # Recurse through all values; timestamp candidates are filtered by range.
        for _key, value in obj.items():
            timestamps.extend(extract_timestamps(value, max_depth - 1))
    elif isinstance(obj, list):
        for item in obj:
            timestamps.extend(extract_timestamps(item, max_depth - 1))

    return timestamps


def extract_messages_heuristic(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Attempt to extract message-like structures from unknown JSON.

    This is a best-effort parser that looks for common patterns in
    conversation exports:
    - Arrays of objects (likely messages)
    - Objects with "text" or "content" fields
    - Timestamp-like numbers
    - Role indicators ("user", "assistant", "human", "ai")

    Args:
        data: Raw JSON data from a provider export

    Returns:
        List of message-like dictionaries with 'text' and 'timestamp' fields
    """
    messages: List[Dict[str, Any]] = []

    # Strategy 1: Look for top-level arrays
    for key in ["messages", "chat_messages", "mapping", "history", "conversation"]:
        if key in data:
            value = data[key]
            if isinstance(value, list):
                # Found a list - extract from each item
                for item in value:
                    if isinstance(item, dict):
                        texts = extract_text_recursively(item, min_length=20)
                        timestamps = extract_timestamps(item)
                        if texts:
                            messages.append({
                                "text": "\n\n".join(texts),
                                "timestamp": timestamps[0] if timestamps else None,
                                "role": _guess_role(item),
                                "source": "heuristic_array",
                            })
            elif isinstance(value, dict):
                # Mapping structure (like ChatGPT)
                for item_id, item in value.items():
                    if isinstance(item, dict):
                        texts = extract_text_recursively(item, min_length=20)
                        timestamps = extract_timestamps(item)
                        if texts:
                            messages.append({
                                "text": "\n\n".join(texts),
                                "timestamp": timestamps[0] if timestamps else None,
                                "role": _guess_role(item),
                                "source": "heuristic_mapping",
                            })

    # Strategy 2: If no messages found, do a deep search
    if not messages:
        texts = extract_text_recursively(data, min_length=30)
        if texts:
            # Group consecutive texts as messages
            for i, text in enumerate(texts):
                messages.append({
                    "text": text,
                    "timestamp": None,
                    "role": "unknown",
                    "index": i,
                    "source": "heuristic_deep_search",
                })

    return messages


def _guess_role(obj: Dict[str, Any]) -> str:
    """Guess the message role from common field patterns."""
    role_keys = ["role", "sender", "author", "from"]
    for key in role_keys:
        if key in obj:
            value = obj[key]
            if isinstance(value, str):
                return value.lower()
            if isinstance(value, dict) and "role" in value:
                return value["role"].lower()

    # Check for common role indicators in any string value
    text = str(obj).lower()
    if any(word in text for word in ["assistant", "ai", "model", "gpt", "claude"]):
        return "assistant"
    if any(word in text for word in ["user", "human"]):
        return "user"

    return "unknown"


def create_degraded_markdown(messages: List[Dict[str, Any]], title: str = "Recovered Conversation") -> str:
    """Create a markdown document from heuristically extracted messages.

    Args:
        messages: Messages extracted via heuristic parsing
        title: Title for the document

    Returns:
        Markdown string
    """
    lines = [
        "---",
        f"title: {title}",
        "status: DEGRADED_MODE",
        "parser: heuristic_fallback",
        "warning: This conversation was recovered using fallback parsing. Formatting may be incomplete.",
        "---",
        "",
        f"# {title}",
        "",
        "⚠️ **DEGRADED MODE**: The original parser failed. This is a best-effort text extraction.",
        "",
    ]

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        text = msg.get("text", "")
        timestamp = msg.get("timestamp")

        lines.append(f"## Message {i + 1} ({role})")
        if timestamp:
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp)
            lines.append(f"*{dt.isoformat()}*")
        lines.append("")
        lines.append(text)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
